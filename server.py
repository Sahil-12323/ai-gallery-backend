"""
AI Gallery v2.0 — Your Life, Queryable
FastAPI + Groq + Supabase (PostgreSQL + Storage)
"""
import os, json, uuid, base64, logging
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Dict, Any, Tuple
from collections import Counter
from fastapi.responses import RedirectResponse
import urllib.parse
from fastapi import FastAPI, APIRouter, HTTPException
from fastapi.responses import HTMLResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import AsyncGroq
from supabase import create_client, Client as SupabaseClient
import httpx

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / ".env")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ai_gallery")
GOOGLE_PHOTOS_API = "https://photoslibrary.googleapis.com/v1/mediaItems"
GOOGLE_ACCESS_TOKEN = None

SUPABASE_URL   = os.environ["SUPABASE_URL"]
SUPABASE_KEY   = os.environ["SUPABASE_SERVICE_KEY"]
GROQ_KEY       = os.environ["GROQ_API_KEY"]
STORAGE_BUCKET = os.environ.get("STORAGE_BUCKET", "photos")
GOOGLE_CLIENT_ID     = os.environ.get("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET", "")

supabase: SupabaseClient = create_client(SUPABASE_URL, SUPABASE_KEY)
groq_client = AsyncGroq(api_key=GROQ_KEY, timeout=60.0, max_retries=2)

VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
CHAT_MODEL   = "llama-3.3-70b-versatile"
DEFAULT_USER = "default_user"

app        = FastAPI(title="AI Gallery", version="2.0.0")
api_router = APIRouter(prefix="/api")

# ── Models ────────────────────────────────────────────────────────────────────
class PhotoCreate(BaseModel):
    image_base64: str
    mime_type:    str = "image/jpeg"
    taken_at:     Optional[str] = None
    location:     Optional[str] = None

class PhotoAnalysis(BaseModel):
    description:      str       = ""
    people:           List[str] = []
    people_count:     int       = 0
    emotion:          str       = "neutral"
    emotion_score:    float     = 0.5
    objects:          List[str] = []
    clothing:         List[str] = []
    colors:           List[str] = []
    scene:            str       = ""
    ocr_text:         str       = ""
    tags:             List[str] = []
    event_type:       str       = ""
    face_descriptors: List[str] = []

class ChatRequest(BaseModel):
    message:         str
    conversation_id: Optional[str] = None

class ChatResponse(BaseModel):
    conversation_id: str
    reply:           str
    photo_ids:       List[str] = []

class SearchRequest(BaseModel):
    query: str

class NarrateRequest(BaseModel):
    story_id:  str
    photo_ids: List[str]

class GoogleTokenRequest(BaseModel):
    code:         str
    redirect_uri: str

class ImportUrlRequest(BaseModel):
    url:       str
    filename:  str = ""
    mime_type: str = "image/jpeg"
    taken_at:  Optional[str] = None
    location:  Optional[str] = None

# ── Helpers ───────────────────────────────────────────────────────────────────
def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()

EMOTION_SCORES = {
    "happy":0.95,"excited":0.90,"romantic":0.85,"energetic":0.75,"calm":0.65,
    "neutral":0.50,"nostalgic":0.40,"reflective":0.38,"melancholic":0.20,"sad":0.10,
}

def _photo_url(storage_path: str) -> str:
    return f"{SUPABASE_URL}/storage/v1/object/public/{STORAGE_BUCKET}/{storage_path}"

def _row_to_photo(row: Dict[str, Any]) -> Dict[str, Any]:
    pid = str(row["id"])
    return {
        "id":               pid,
        "url":              _photo_url(row.get("storage_path", "")),
        "thumbnail_base64": "",
        "mime_type":        row.get("mime_type", "image/jpeg"),
        "taken_at":         str(row.get("taken_at", "")),
        "location":         row.get("location"),
        "created_at":       str(row.get("created_at", "")),
        "is_favorite":      row.get("is_favorite", False),
        "analysis": {
            "description":      row.get("description", ""),
            "people":           row.get("people", []),
            "people_count":     row.get("people_count", 0),
            "emotion":          row.get("emotion", "neutral"),
            "emotion_score":    float(row.get("emotion_score", 0.5)),
            "objects":          row.get("objects", []),
            "clothing":         row.get("clothing", []),
            "colors":           row.get("colors", []),
            "scene":            row.get("scene", ""),
            "ocr_text":         row.get("ocr_text", ""),
            "tags":             row.get("tags", []),
            "event_type":       row.get("event_type", ""),
            "face_descriptors": row.get("face_descriptors", []),
        },
    }

async def _analyze(b64: str, mime: str) -> PhotoAnalysis:
    prompt = (
        'Analyze this photo. Return ONLY valid JSON:\n'
        '{"description":"2-3 sentence vivid description","people":["descriptions"],"people_count":0,'
        '"emotion":"happy|calm|excited|nostalgic|melancholic|energetic|romantic|reflective|neutral",'
        '"emotion_score":0.5,"objects":["objects"],"clothing":["item+color"],"colors":["dominant colors"],'
        '"scene":"indoor|outdoor|beach|mountain|city|restaurant|home|gym|office|park|etc",'
        '"ocr_text":"exact visible text or empty","tags":["8-12 tags"],'
        '"event_type":"travel|birthday|wedding|workout|food|pet|nature|concert|party|selfie|landscape|daily|etc",'
        '"face_descriptors":["adult-woman-dark-hair per face"]}'
    )
    try:
        r = await groq_client.chat.completions.create(
            model=VISION_MODEL,
            messages=[{"role":"user","content":[
                {"type":"text","text":prompt},
                {"type":"image_url","image_url":{"url":f"data:{mime};base64,{b64}"}},
            ]}],
            temperature=0.2, max_completion_tokens=900,
            response_format={"type":"json_object"},
        )
        p = json.loads(r.choices[0].message.content or "{}")
        p.setdefault("emotion_score", EMOTION_SCORES.get(p.get("emotion","neutral"), 0.5))
        return PhotoAnalysis(**{k: p.get(k, v) for k, v in PhotoAnalysis().dict().items()})
    except Exception as e:
        logger.error(f"Vision failed: {e}")
        return PhotoAnalysis(description="Analysis unavailable", emotion="neutral")

def _ctx_line(row: Dict[str, Any]) -> str:
    return (
        f"[id={row['id']} date={str(row.get('taken_at',''))[:10]} "
        f"loc={row.get('location') or 'N/A'} fav={'yes' if row.get('is_favorite') else 'no'}] "
        f"{row.get('description','')} | emotion={row.get('emotion','')} | "
        f"event={row.get('event_type','')} | scene={row.get('scene','')} | "
        f"tags={','.join((row.get('tags') or [])[:8])} | ocr={str(row.get('ocr_text',''))[:60]}"
    )

def _get_photos(limit: int = 80) -> List[Dict[str, Any]]:
    res = (supabase.table("photos").select("*")
           .eq("user_id", DEFAULT_USER).order("taken_at", desc=True).limit(limit).execute())
    return res.data or []

def _upload_storage(b64: str, mime: str, pid: str) -> str:
    raw  = base64.b64decode(b64)
    ext  = mime.split("/")[-1].replace("jpeg", "jpg")
    path = f"{DEFAULT_USER}/{pid}.{ext}"
    supabase.storage.from_(STORAGE_BUCKET).upload(
        path, raw, file_options={"content-type": mime, "upsert": "true"}
    )
    return path

def _del_storage(path: str):
    try:
        supabase.storage.from_(STORAGE_BUCKET).remove([path])
    except Exception as e:
        logger.warning(f"Storage delete failed: {e}")

# ── Root ──────────────────────────────────────────────────────────────────────
@api_router.get("/")
async def root():
    return {"message": "AI Gallery v2.0", "status": "ok"}

# ── Photos ────────────────────────────────────────────────────────────────────
@api_router.post("/photos")
async def upload_photo(payload: PhotoCreate):
    if not payload.image_base64:
        raise HTTPException(400, "Empty image")
    if len(payload.image_base64) * 0.75 > 6_000_000:
        raise HTTPException(413, "Image exceeds 6MB")
    analysis = await _analyze(payload.image_base64, payload.mime_type)
    pid = str(uuid.uuid4())
    try:
        storage_path = _upload_storage(payload.image_base64, payload.mime_type, pid)
    except Exception as e:
        raise HTTPException(502, f"Storage upload failed: {e}")
    row = {
        "id": pid, "user_id": DEFAULT_USER,
        "storage_path": storage_path, "mime_type": payload.mime_type,
        "taken_at": payload.taken_at or _iso_now(), "location": payload.location,
        "created_at": _iso_now(), "is_favorite": False,
        "description": analysis.description, "people": analysis.people,
        "people_count": analysis.people_count, "emotion": analysis.emotion,
        "emotion_score": analysis.emotion_score, "objects": analysis.objects,
        "clothing": analysis.clothing, "colors": analysis.colors,
        "scene": analysis.scene, "ocr_text": analysis.ocr_text,
        "tags": analysis.tags, "event_type": analysis.event_type,
        "face_descriptors": analysis.face_descriptors,
    }
    result = supabase.table("photos").insert(row).execute()
    if not result.data:
        raise HTTPException(500, "DB insert failed")
    supabase.table("insights_cache").delete().eq("user_id", DEFAULT_USER).execute()
    return _row_to_photo(result.data[0])

@api_router.get("/photos")
async def list_photos(limit: int = 300):
    res = (supabase.table("photos").select("*")
           .eq("user_id", DEFAULT_USER).order("taken_at", desc=True).limit(limit).execute())
    return [_row_to_photo(r) for r in (res.data or [])]

@api_router.get("/photos/{pid}")
async def get_photo(pid: str):
    res = (supabase.table("photos").select("*")
           .eq("id", pid).eq("user_id", DEFAULT_USER).limit(1).execute())
    if not res.data:
        raise HTTPException(404, "Photo not found")
    return _row_to_photo(res.data[0])


@api_router.get("/google/photos")
async def get_google_photos(token: str):
    async with httpx.AsyncClient(timeout=15.0) as http:
        r = await http.get(
            GOOGLE_PHOTOS_API,
            headers={
                "Authorization": f"Bearer {token}"
            },
            params={"pageSize": 50}
        )

    if r.status_code != 200:
        raise HTTPException(400, f"Google API error: {r.text}")

    data = r.json()

    photos = [
        {
            "id": item["id"],
            "baseUrl": item["baseUrl"]
        }
        for item in data.get("mediaItems", [])
    ]

    return {"photos": photos}


@api_router.patch("/photos/{pid}/favorite")
async def toggle_fav(pid: str):
    res = (supabase.table("photos").select("is_favorite")
           .eq("id", pid).eq("user_id", DEFAULT_USER).limit(1).execute())
    if not res.data:
        raise HTTPException(404, "Not found")
    new_val = not res.data[0]["is_favorite"]
    supabase.table("photos").update({"is_favorite": new_val}).eq("id", pid).execute()
    return {"is_favorite": new_val}

@api_router.delete("/photos/{pid}")
async def delete_photo(pid: str):
    logger.info(f"DELETE photo id={pid}")
    res = (supabase.table("photos").select("storage_path")
           .eq("id", pid).eq("user_id", DEFAULT_USER).limit(1).execute())
    if not res.data:
        logger.warning(f"Photo {pid} not found")
        raise HTTPException(404, "Photo not found")
    storage_path = res.data[0].get("storage_path", "")
    logger.info(f"Deleting storage: {storage_path}")
    if storage_path:
        _del_storage(storage_path)
    del_res = supabase.table("photos").delete().eq("id", pid).execute()
    logger.info(f"DB delete: {del_res.data}")
    supabase.table("insights_cache").delete().eq("user_id", DEFAULT_USER).execute()
    return {"deleted": pid}

# ── Chat ──────────────────────────────────────────────────────────────────────
@api_router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(400, "Empty message")
    photos = _get_photos(100)
    if not photos:
        return ChatResponse(
            conversation_id=req.conversation_id or str(uuid.uuid4()),
            reply="Your gallery is empty. Upload photos or load demo memories!",
            photo_ids=[],
        )
    cid = req.conversation_id or str(uuid.uuid4())
    conv_res = (supabase.table("conversations").select("messages")
                .eq("id", cid).eq("user_id", DEFAULT_USER).execute())
    history = conv_res.data[0]["messages"] if conv_res.data else []
    ctx = "\n".join(_ctx_line(p) for p in photos)
    system = (
        "You are the user's warm, insightful personal memory assistant. "
        "Respond conversationally (2-4 sentences), reference real dates/places. "
        "ALWAYS end with:\nPHOTO_IDS: [id1, id2, ...]\n"
        "Include up to 8 relevant ids. If none match, use PHOTO_IDS: []\n\n"
        f"LIBRARY ({len(photos)} photos):\n{ctx}"
    )
    msgs = [{"role":"system","content":system}]
    for m in history[-12:]:
        msgs.append({"role": m["role"], "content": m["content"]})
    msgs.append({"role":"user","content":req.message})
    try:
        resp = await groq_client.chat.completions.create(
            model=CHAT_MODEL, messages=msgs, temperature=0.75, max_completion_tokens=700
        )
        full = resp.choices[0].message.content or ""
    except Exception as e:
        raise HTTPException(502, "AI unavailable")
    pid_list: List[str] = []
    text = full
    valid = {str(p["id"]) for p in photos}
    if "PHOTO_IDS:" in full:
        parts = full.rsplit("PHOTO_IDS:", 1)
        text = parts[0].strip()
        for tok in parts[1].strip().strip("[]").replace('"','').replace("'",'').split(","):
            tok = tok.strip()
            if tok and tok in valid:
                pid_list.append(tok)
        pid_list = pid_list[:8]
    new_msgs = history + [
        {"role":"user","content":req.message,"timestamp":_iso_now(),"photo_ids":[]},
        {"role":"assistant","content":text,"timestamp":_iso_now(),"photo_ids":pid_list},
    ]
    supabase.table("conversations").upsert({
        "id": cid, "user_id": DEFAULT_USER, "messages": new_msgs, "updated_at": _iso_now(),
    }).execute()
    return ChatResponse(conversation_id=cid, reply=text, photo_ids=pid_list)

@api_router.get("/chat/{cid}")
async def get_conv(cid: str):
    res = (supabase.table("conversations").select("id, messages")
           .eq("id", cid).eq("user_id", DEFAULT_USER).execute())
    if not res.data:
        return {"id": cid, "messages": []}
    return {"id": res.data[0]["id"], "messages": res.data[0]["messages"]}

@api_router.delete("/chat/{cid}")
async def del_conv(cid: str):
    supabase.table("conversations").delete().eq("id", cid).eq("user_id", DEFAULT_USER).execute()
    return {"deleted": cid}

# ── Insights ──────────────────────────────────────────────────────────────────
@api_router.get("/insights")
async def get_insights(refresh: bool = False):
    if not refresh:
        cached = (supabase.table("insights_cache").select("*")
                  .eq("user_id", DEFAULT_USER).execute())
        if cached.data and cached.data[0].get("insights"):
            row = cached.data[0]
            return {"user_id":DEFAULT_USER,"insights":row["insights"],
                    "generated_at":row["generated_at"],"photo_count":row.get("photo_count",0)}
    photos = _get_photos(150)
    if len(photos) < 2:
        return {"user_id":DEFAULT_USER,"insights":[],"generated_at":_iso_now()}
    ctx = "\n".join(_ctx_line(p) for p in photos)
    emotions  = [p.get("emotion","")    for p in photos if p.get("emotion")]
    events    = [p.get("event_type","") for p in photos if p.get("event_type")]
    locations = [p.get("location","")   for p in photos if p.get("location")]
    stats = (f"{len(photos)} photos | emotions={dict(Counter(emotions).most_common(3))} | "
             f"events={dict(Counter(events).most_common(3))} | locs={dict(Counter(locations).most_common(3))}")
    prompt = (
        "Analyze this user's photo library and generate 5-7 personal, emotionally-intelligent life insights. "
        "Be specific — reference actual dates, places, people. "
        'Return ONLY JSON: {"insights":[{"title":"max 5 words","body":"1-2 sentence insight",'
        '"icon":"heart|trend-up|trend-down|sparkles|calendar|users|map|camera|flame|moon",'
        '"emotion":"positive|neutral|reflective"}]}\n\n'
        f"STATS: {stats}\n\nDATA:\n{ctx}"
    )
    try:
        r = await groq_client.chat.completions.create(
            model=CHAT_MODEL, messages=[{"role":"user","content":prompt}],
            temperature=0.7, max_completion_tokens=1000,
            response_format={"type":"json_object"},
        )
        ins = json.loads(r.choices[0].message.content or "{}").get("insights", [])
    except Exception as e:
        logger.error(f"Insights error: {e}")
        ins = []
    now = _iso_now()
    supabase.table("insights_cache").upsert({
        "user_id":DEFAULT_USER,"insights":ins,"photo_count":len(photos),"generated_at":now,
    }).execute()
    return {"user_id":DEFAULT_USER,"insights":ins,"generated_at":now,"photo_count":len(photos)}

# ── Emotion Timeline ───────────────────────────────────────────────────────────
@api_router.get("/emotions/timeline")
async def emotion_timeline(days: int = 90):
    since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    res = (supabase.table("photos").select("taken_at, emotion, emotion_score")
           .eq("user_id", DEFAULT_USER).gte("taken_at", since).order("taken_at").execute())
    buckets: Dict[str, List[float]] = {}
    for row in (res.data or []):
        try:
            t = datetime.fromisoformat(str(row["taken_at"]).replace("Z","+00:00"))
            wk = t.strftime("%Y-W%W")
        except: continue
        score = float(row.get("emotion_score") or EMOTION_SCORES.get(row.get("emotion","neutral"), 0.5))
        buckets.setdefault(wk, []).append(score)
    timeline = [{"week":wk,"avg_score":round(sum(v)/len(v),3),"count":len(v)} for wk,v in sorted(buckets.items())]
    all_res = (supabase.table("photos").select("emotion").eq("user_id", DEFAULT_USER).execute())
    dist: Dict[str,int] = {}
    for row in (all_res.data or []):
        em = row.get("emotion","")
        if em: dist[em] = dist.get(em,0) + 1
    return {"timeline": timeline, "distribution": dist}

# ── People ────────────────────────────────────────────────────────────────────
@api_router.get("/people")
async def get_people():
    res = (supabase.table("photos")
           .select("id, storage_path, mime_type, taken_at, face_descriptors, emotion")
           .eq("user_id", DEFAULT_USER).order("taken_at", desc=True).limit(500).execute())
    groups: Dict[str, List[str]] = {}
    pm: Dict[str, Dict] = {}
    for row in (res.data or []):
        pid = str(row["id"])
        pm[pid] = row
        for desc in (row.get("face_descriptors") or []):
            key = str(desc).lower().strip()
            groups.setdefault(key, [])
            if pid not in groups[key]: groups[key].append(pid)
    result = []
    for desc, ids in sorted(groups.items(), key=lambda x: len(x[1]), reverse=True):
        if len(ids) >= 2:
            cover = pm.get(ids[0], {})
            result.append({"descriptor":desc,"photo_count":len(ids),"photo_ids":ids[:20],
                           "cover_url":_photo_url(cover.get("storage_path","")) if cover.get("storage_path") else "",
                           "cover_mime":cover.get("mime_type","image/jpeg")})
    return {"groups": result[:15]}

# ── Memory Rewind ─────────────────────────────────────────────────────────────
@api_router.get("/memories/on-this-day")
async def on_this_day():
    today = datetime.now(timezone.utc)
    mm = f"{today.month:02d}"; dd = f"{today.day:02d}"
    all_rows = (supabase.table("photos").select("*").eq("user_id", DEFAULT_USER).execute()).data or []
    matches = [_row_to_photo(r) for r in all_rows
               if len(str(r.get("taken_at",""))) >= 10
               and str(r["taken_at"])[5:7] == mm and str(r["taken_at"])[8:10] == dd]
    matches.sort(key=lambda p: p["taken_at"], reverse=True)
    month_rows = [r for r in all_rows if len(str(r.get("taken_at",""))) >= 7 and str(r["taken_at"])[5:7] == mm]
    prediction = None
    if month_rows:
        events: Dict[str,int] = {}; locs: Dict[str,int] = {}
        for r in month_rows:
            ev = r.get("event_type",""); loc = r.get("location","")
            if ev: events[ev] = events.get(ev,0)+1
            if loc: locs[loc] = locs.get(loc,0)+1
        parts = []
        if events: parts.append(f"'{max(events.items(),key=lambda x:x[1])[0]}' moments")
        if locs: parts.append(f"often near {max(locs.items(),key=lambda x:x[1])[0]}")
        if parts: prediction = f"In {today.strftime('%B')}, you usually have {' and '.join(parts)}."
    return {"date":today.strftime("%B %d"),"photos":matches,"prediction":prediction,"month_count":len(month_rows)}

# ── Stories ───────────────────────────────────────────────────────────────────
@api_router.get("/stories")
async def get_stories():
    res = (supabase.table("photos").select("*").eq("user_id", DEFAULT_USER).order("taken_at").execute())
    rows = res.data or []
    if not rows: return {"stories": []}
    clusters: List[List[Dict]] = []
    for row in rows:
        ev = row.get("event_type","")
        try: t = datetime.fromisoformat(str(row.get("taken_at","")).replace("Z","+00:00"))
        except: t = datetime.now(timezone.utc)
        placed = False
        for cl in clusters:
            lev = cl[-1].get("event_type","")
            try: lt = datetime.fromisoformat(str(cl[-1].get("taken_at","")).replace("Z","+00:00"))
            except: lt = t
            if lev == ev and abs((t - lt).days) <= 5:
                cl.append(row); placed = True; break
        if not placed: clusters.append([row])
    clusters = [c for c in clusters if len(c) >= 2]
    clusters.sort(key=lambda c: str(c[0].get("taken_at","")), reverse=True)
    clusters = clusters[:12]
    stories = []
    for cl in clusters:
        fd = str(cl[0].get("taken_at",""))[:10]; ld = str(cl[-1].get("taken_at",""))[:10]
        descs = " | ".join(r.get("description","") for r in cl[:6])
        ems = [r.get("emotion","") for r in cl]
        dom_em = Counter(ems).most_common(1)[0][0] if ems else "neutral"
        avg_sc = round(sum(EMOTION_SCORES.get(e,0.5) for e in ems) / max(len(ems),1), 2)
        try:
            r2 = await groq_client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[{"role":"user","content":(f"Story title (3-5 words) and 1-sentence subtitle. Emotion: {dom_em}. Photos: {descs}. JSON: {{\"title\":\"...\",\"subtitle\":\"...\"}}")}],
                temperature=0.85, max_completion_tokens=120, response_format={"type":"json_object"},
            )
            p2 = json.loads(r2.choices[0].message.content or "{}")
            title = p2.get("title","Untitled"); subtitle = p2.get("subtitle","")
        except: title = "Untitled Memory"; subtitle = ""
        stories.append({
            "id": str(uuid.uuid4()), "title":title, "subtitle":subtitle,
            "cover_url": _photo_url(cl[0].get("storage_path","")),
            "cover_base64": "", "mime_type": cl[0].get("mime_type","image/jpeg"),
            "date_range": f"{fd} → {ld}" if fd != ld else fd,
            "photo_count": len(cl), "photo_ids": [str(r["id"]) for r in cl],
            "dominant_emotion": dom_em, "mood_score": avg_sc,
            "photos": [_row_to_photo(r) for r in cl],
        })
    return {"stories": stories}

@api_router.post("/stories/narrate")
async def narrate_story(req: NarrateRequest):
    pdata = []
    for pid in req.photo_ids[:10]:
        res = (supabase.table("photos").select("description, taken_at")
               .eq("id", pid).eq("user_id", DEFAULT_USER).execute())
        if res.data: pdata.append(res.data[0])
    if not pdata: return {"narration":"A collection of memories."}
    descs = "\n".join(f"- {r.get('description','')} ({str(r.get('taken_at',''))[:10]})" for r in pdata)
    try:
        r = await groq_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role":"user","content":f"Write a warm, poetic 3-sentence narration for this photo story.\n\nPhotos:\n{descs}"}],
            temperature=0.9, max_completion_tokens=200,
        )
        return {"narration": r.choices[0].message.content or "A beautiful collection."}
    except: return {"narration": "A beautiful collection of memories."}

# ── Search ────────────────────────────────────────────────────────────────────
@api_router.post("/search")
async def search(req: SearchRequest):
    if not req.query.strip(): return {"results": []}
    all_photos = _get_photos(300)
    q = req.query.lower()
    scored: List[Tuple[float, Dict]] = []
    for p in all_photos:
        hay = " ".join([p.get("description",""),p.get("ocr_text",""),p.get("scene",""),
                        p.get("event_type",""),p.get("emotion",""),
                        " ".join(p.get("people") or [])," ".join(p.get("objects") or []),
                        " ".join(p.get("clothing") or [])," ".join(p.get("tags") or []),
                        p.get("location") or ""]).lower()
        score = 0.0
        for tok in q.split():
            if tok in hay:
                score += 1.0
                if tok in p.get("ocr_text","").lower(): score += 1.5
        if score > 0: scored.append((score, p))
    if len(scored) < 3:
        ctx = "\n".join(_ctx_line(p) for p in all_photos)
        try:
            r = await groq_client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[{"role":"user","content":f'Search: "{req.query}"\nSelect up to 12 relevant ids. JSON: {{"ids":["id1",...]}}\nLIBRARY:\n{ctx}'}],
                temperature=0.2, max_completion_tokens=400, response_format={"type":"json_object"},
            )
            ids = json.loads(r.choices[0].message.content or "{}").get("ids",[])
            vm = {str(p["id"]): p for p in all_photos}
            seen = {str(p["id"]) for _,p in scored}
            for i in ids:
                if i in vm and i not in seen:
                    scored.append((0.5, vm[i])); seen.add(i)
        except: pass
    scored.sort(key=lambda x: x[0], reverse=True)
    return {"results": [_row_to_photo(p) for _, p in scored[:24]]}

# ── Favorites & Stats ─────────────────────────────────────────────────────────
@api_router.get("/favorites")
async def favorites():
    res = (supabase.table("photos").select("*")
           .eq("user_id", DEFAULT_USER).eq("is_favorite", True)
           .order("taken_at", desc=True).limit(200).execute())
    return [_row_to_photo(r) for r in (res.data or [])]

@api_router.get("/stats")
async def stats():
    total_res = supabase.table("photos").select("id", count="exact").eq("user_id", DEFAULT_USER).execute()
    fav_res   = supabase.table("photos").select("id", count="exact").eq("user_id", DEFAULT_USER).eq("is_favorite", True).execute()
    oldest_res = supabase.table("photos").select("taken_at").eq("user_id", DEFAULT_USER).order("taken_at").limit(1).execute()
    newest_res = supabase.table("photos").select("taken_at").eq("user_id", DEFAULT_USER).order("taken_at", desc=True).limit(1).execute()
    all_res = supabase.table("photos").select("event_type, emotion").eq("user_id", DEFAULT_USER).execute()
    ev_dist: Dict[str,int] = {}; em_dist: Dict[str,int] = {}
    for row in (all_res.data or []):
        ev = row.get("event_type",""); em = row.get("emotion","")
        if ev: ev_dist[ev] = ev_dist.get(ev,0)+1
        if em: em_dist[em] = em_dist.get(em,0)+1
    return {
        "total": total_res.count or 0, "favorites": fav_res.count or 0,
        "oldest": str(oldest_res.data[0]["taken_at"])[:10] if oldest_res.data else None,
        "newest": str(newest_res.data[0]["taken_at"])[:10] if newest_res.data else None,
        "event_distribution": dict(Counter(ev_dist).most_common(8)),
        "emotion_distribution": dict(Counter(em_dist).most_common(8)),
    }

# ── Settings ──────────────────────────────────────────────────────────────────
@api_router.get("/settings")
async def get_settings():
    res = supabase.table("user_settings").select("*").eq("user_id", DEFAULT_USER).execute()
    return res.data[0] if res.data else {"user_id":DEFAULT_USER,"user_name":"You","notifications_enabled":True,"auto_insights":True}

@api_router.put("/settings")
async def update_settings(payload: Dict[str, Any]):
    payload["user_id"] = DEFAULT_USER; payload["updated_at"] = _iso_now(); payload.pop("_id",None)
    supabase.table("user_settings").upsert(payload).execute()
    return payload

# ── Google OAuth ──────────────────────────────────────────────────────────────
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"

@api_router.post("/auth/google/token")
async def google_token_exchange(req: GoogleTokenRequest):
    if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
        raise HTTPException(501, "Set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET in backend/.env")
    import httpx
    async with httpx.AsyncClient(timeout=15.0) as http:
        r = await http.post(GOOGLE_TOKEN_URL, data={
            "code": req.code, "client_id": GOOGLE_CLIENT_ID,
            "client_secret": GOOGLE_CLIENT_SECRET,
            "redirect_uri": req.redirect_uri, "grant_type": "authorization_code",
        }, headers={"Content-Type": "application/x-www-form-urlencoded"})
        if r.status_code != 200:
            logger.error(f"Google token failed: {r.status_code} {r.text}")
            raise HTTPException(400, f"Token exchange failed: {r.text}")
        data = r.json()
        if "error" in data:
            raise HTTPException(400, f"Google error: {data['error']}")
        return {"access_token": data.get("access_token"), "expires_in": data.get("expires_in", 3600)}

@api_router.get("/auth/google/login")
async def google_login():
    if not GOOGLE_CLIENT_ID:
        raise HTTPException(500, "Missing GOOGLE_CLIENT_ID")

    redirect_uri = os.getenv("GOOGLE_REDIRECT_URI")
    if not redirect_uri:
        raise HTTPException(500, "Missing GOOGLE_REDIRECT_URI")

    params = {
        "client_id": GOOGLE_CLIENT_ID,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": "https://www.googleapis.com/auth/photoslibrary.readonly",
        "access_type": "offline",
        "prompt": "consent"
    }

    url = "https://accounts.google.com/o/oauth2/v2/auth?" + urllib.parse.urlencode(params)

    return RedirectResponse(url)

@api_router.get("/auth/google/callback")
async def google_callback(code: str = ""):
    if not code:
        return HTMLResponse("<h2>No code received</h2>")

    async with httpx.AsyncClient() as http:
        r = await http.post(
            GOOGLE_TOKEN_URL,
            data={
                "code": code,
                "client_id": GOOGLE_CLIENT_ID,
                "client_secret": GOOGLE_CLIENT_SECRET,
                "redirect_uri": os.getenv("GOOGLE_REDIRECT_URI"),
                "grant_type": "authorization_code",
            },
        )

    data = r.json()
    access_token = data.get("access_token")

    # 🔥 IMPORTANT: redirect back to app with token
    redirect_url = f"ai-gallery://oauth?token={access_token}"

    return RedirectResponse(redirect_url)
# ── Import from URL ───────────────────────────────────────────────────────────
@api_router.post("/photos/import-url")
async def import_photo_from_url(payload: ImportUrlRequest):
    import httpx
    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as http:
        r = await http.get(payload.url, headers={"User-Agent": "AI-Gallery/2.0"})
        if r.status_code != 200:
            raise HTTPException(400, f"Download failed: HTTP {r.status_code}")
        if len(r.content) > 6_000_000:
            raise HTTPException(413, "Image too large")
        b64  = base64.b64encode(r.content).decode()
        mime = payload.mime_type or r.headers.get("content-type","image/jpeg").split(";")[0]
    analysis = await _analyze(b64, mime)
    pid = str(uuid.uuid4())
    storage_path = _upload_storage(b64, mime, pid)
    row = {
        "id": pid, "user_id": DEFAULT_USER,
        "storage_path": storage_path, "mime_type": mime,
        "taken_at": payload.taken_at or _iso_now(), "location": payload.location,
        "created_at": _iso_now(), "is_favorite": False,
        "description": analysis.description, "people": analysis.people,
        "people_count": analysis.people_count, "emotion": analysis.emotion,
        "emotion_score": analysis.emotion_score, "objects": analysis.objects,
        "clothing": analysis.clothing, "colors": analysis.colors,
        "scene": analysis.scene, "ocr_text": analysis.ocr_text,
        "tags": analysis.tags, "event_type": analysis.event_type,
        "face_descriptors": analysis.face_descriptors,
    }
    result = supabase.table("photos").insert(row).execute()
    supabase.table("insights_cache").delete().eq("user_id", DEFAULT_USER).execute()
    return _row_to_photo(result.data[0])

# ── Seed & Reset ──────────────────────────────────────────────────────────────
DEMO_IMAGES = [
    ("https://images.unsplash.com/photo-1658326227968-f9f1d9087680?w=800&q=80", -30, "Goa Beach"),
    ("https://images.unsplash.com/photo-1763357617062-65a9737e13e5?w=800&q=80", -28, "Goa Beach"),
    ("https://images.pexels.com/photos/36120641/pexels-photo-36120641.png?auto=compress&w=800", -60, "Gold's Gym"),
    ("https://images.pexels.com/photos/31849599/pexels-photo-31849599.jpeg?auto=compress&w=800", -45, "Gold's Gym"),
    ("https://images.pexels.com/photos/11364283/pexels-photo-11364283.jpeg?auto=compress&w=800", -10, "Restaurant"),
    ("https://images.pexels.com/photos/28408919/pexels-photo-28408919.jpeg?auto=compress&w=800", -15, None),
    ("https://images.unsplash.com/photo-1504674900247-0877df9cc836?w=800&q=80", -7, "Home Kitchen"),
    ("https://images.unsplash.com/photo-1441986300917-64674bd600d8?w=800&q=80", -20, "Downtown"),
    ("https://images.unsplash.com/photo-1529158062015-cad636e205a0?w=800&q=80", -365, "Old Mountains"),
]

@api_router.post("/seed")
async def seed_demo():
    import httpx
    existing = supabase.table("photos").select("storage_path").eq("user_id", DEFAULT_USER).execute()
    for row in (existing.data or []):
        _del_storage(row.get("storage_path",""))
    supabase.table("photos").delete().eq("user_id", DEFAULT_USER).execute()
    supabase.table("conversations").delete().eq("user_id", DEFAULT_USER).execute()
    supabase.table("insights_cache").delete().eq("user_id", DEFAULT_USER).execute()
    inserted = 0
    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as http:
        for url, days_ago, loc in DEMO_IMAGES:
            try:
                r = await http.get(url)
                if r.status_code != 200 or len(r.content) > 4_000_000: continue
                b64  = base64.b64encode(r.content).decode()
                mime = r.headers.get("content-type","image/jpeg").split(";")[0]
                an   = await _analyze(b64, mime)
                pid  = str(uuid.uuid4())
                taken = (datetime.now(timezone.utc) + timedelta(days=days_ago)).isoformat()
                sp = _upload_storage(b64, mime, pid)
                supabase.table("photos").insert({
                    "id":pid,"user_id":DEFAULT_USER,"storage_path":sp,"mime_type":mime,
                    "taken_at":taken,"location":loc,"created_at":_iso_now(),"is_favorite":False,
                    "description":an.description,"people":an.people,"people_count":an.people_count,
                    "emotion":an.emotion,"emotion_score":an.emotion_score,"objects":an.objects,
                    "clothing":an.clothing,"colors":an.colors,"scene":an.scene,"ocr_text":an.ocr_text,
                    "tags":an.tags,"event_type":an.event_type,"face_descriptors":an.face_descriptors,
                }).execute()
                inserted += 1
            except Exception as e:
                logger.warning(f"Seed skip {url}: {e}")
    return {"inserted": inserted}

@api_router.delete("/reset")
async def reset_all():
    existing = supabase.table("photos").select("storage_path").eq("user_id", DEFAULT_USER).execute()
    for row in (existing.data or []): _del_storage(row.get("storage_path",""))
    supabase.table("photos").delete().eq("user_id", DEFAULT_USER).execute()
    supabase.table("conversations").delete().eq("user_id", DEFAULT_USER).execute()
    supabase.table("insights_cache").delete().eq("user_id", DEFAULT_USER).execute()
    return {"ok": True}

# ── App (include_router MUST be last) ─────────────────────────────────────────
app.include_router(api_router)
app.add_middleware(CORSMiddleware, allow_credentials=True, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

@app.on_event("startup")
async def startup():
    logger.info(f"AI Gallery v2.0 started — Supabase: {SUPABASE_URL}")
