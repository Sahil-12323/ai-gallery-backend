"""Backend tests for AI Gallery API.

Run: pytest /app/backend/tests/test_ai_gallery_api.py -v
"""
import os
import pytest
import requests

BASE_URL = os.environ.get("EXPO_PUBLIC_BACKEND_URL", "https://moment-lens-1.preview.emergentagent.com").rstrip("/")
API = f"{BASE_URL}/api"


@pytest.fixture(scope="module")
def api_client():
    s = requests.Session()
    s.headers.update({"Content-Type": "application/json"})
    return s


# --- Health ---
class TestHealth:
    def test_root_ok(self, api_client):
        r = api_client.get(f"{API}/", timeout=30)
        assert r.status_code == 200
        data = r.json()
        assert data.get("status") == "ok"
        assert "message" in data


# --- Photos ---
class TestPhotos:
    def test_list_photos_has_items(self, api_client):
        r = api_client.get(f"{API}/photos", timeout=60)
        assert r.status_code == 200
        photos = r.json()
        assert isinstance(photos, list)
        assert len(photos) >= 1, "Seed should have inserted photos"
        # pick first
        pytest.FIRST_PHOTO = photos[0]

    def test_photo_fields(self, api_client):
        r = api_client.get(f"{API}/photos", timeout=30)
        photos = r.json()
        assert len(photos) > 0
        p = photos[0]
        for k in ("id", "thumbnail_base64", "mime_type", "taken_at", "created_at", "analysis"):
            assert k in p, f"Missing key {k}"
        a = p["analysis"]
        # Description / emotion / tags / event_type should exist (may be empty strings if analysis failed)
        for k in ("description", "emotion", "tags", "event_type"):
            assert k in a, f"Missing analysis key {k}"

    def test_get_photo_by_id(self, api_client):
        r = api_client.get(f"{API}/photos", timeout=30)
        photos = r.json()
        pid = photos[0]["id"]
        r2 = api_client.get(f"{API}/photos/{pid}", timeout=30)
        assert r2.status_code == 200
        assert r2.json()["id"] == pid

    def test_get_nonexistent_photo_404(self, api_client):
        r = api_client.get(f"{API}/photos/does-not-exist-xyz", timeout=30)
        assert r.status_code == 404


# --- Chat ---
class TestChat:
    def test_chat_happiest_moments(self, api_client):
        r = api_client.post(
            f"{API}/chat",
            json={"message": "Show me my happiest moments"},
            timeout=90,
        )
        assert r.status_code == 200, r.text
        data = r.json()
        assert "conversation_id" in data
        assert "reply" in data
        assert isinstance(data["reply"], str) and len(data["reply"]) > 0
        assert "photo_ids" in data
        assert isinstance(data["photo_ids"], list)

        # Validate photo_ids are real
        photos = api_client.get(f"{API}/photos", timeout=30).json()
        valid = {p["id"] for p in photos}
        for pid in data["photo_ids"]:
            assert pid in valid, f"Invalid photo_id returned: {pid}"

    def test_chat_empty_message_400(self, api_client):
        r = api_client.post(f"{API}/chat", json={"message": "   "}, timeout=30)
        assert r.status_code == 400


# --- Insights ---
class TestInsights:
    def test_insights_returns_list(self, api_client):
        r = api_client.get(f"{API}/insights", timeout=90)
        assert r.status_code == 200
        data = r.json()
        assert "insights" in data
        assert isinstance(data["insights"], list)
        # The requirement says at least 1 item
        assert len(data["insights"]) >= 1, "Expected at least 1 insight"
        item = data["insights"][0]
        for k in ("title", "body", "icon", "emotion"):
            assert k in item, f"Missing {k} in insight"


# --- Stories ---
class TestStories:
    def test_stories_shape(self, api_client):
        r = api_client.get(f"{API}/stories", timeout=120)
        assert r.status_code == 200
        data = r.json()
        assert "stories" in data
        assert isinstance(data["stories"], list)


# --- Search ---
class TestSearch:
    def test_search_beach(self, api_client):
        r = api_client.post(f"{API}/search", json={"query": "beach"}, timeout=60)
        assert r.status_code == 200
        data = r.json()
        assert "results" in data
        assert isinstance(data["results"], list)
        # With seed including Goa Beach photos, should return some results
        assert len(data["results"]) >= 1, "Beach search should return at least 1 result"


# --- Memories ---
class TestMemories:
    def test_on_this_day_shape(self, api_client):
        r = api_client.get(f"{API}/memories/on-this-day", timeout=30)
        assert r.status_code == 200
        data = r.json()
        assert "date" in data and isinstance(data["date"], str)
        assert "photos" in data and isinstance(data["photos"], list)
        assert "prediction" in data  # may be None


# --- Photo Delete (run last) ---
class TestPhotoDelete:
    @pytest.mark.order("last")
    def test_delete_and_verify(self, api_client):
        # Use a non-critical one; do not delete if only 1 photo left
        photos = api_client.get(f"{API}/photos", timeout=30).json()
        if len(photos) < 2:
            pytest.skip("Not enough photos to safely delete")
        # Pick the oldest to delete (likely the 1-year-old mountain photo)
        target = photos[-1]
        pid = target["id"]
        r = api_client.delete(f"{API}/photos/{pid}", timeout=30)
        assert r.status_code == 200
        assert r.json().get("deleted") == pid
        # Verify gone
        r2 = api_client.get(f"{API}/photos/{pid}", timeout=30)
        assert r2.status_code == 404
