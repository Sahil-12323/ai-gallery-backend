-- ============================================================
-- AI Gallery v2.0 — Supabase PostgreSQL Schema
-- Run this in Supabase → SQL Editor → New Query
-- ============================================================

-- Enable UUID extension (already enabled in Supabase by default)
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ── Photos ────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS photos (
    id            UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id       TEXT NOT NULL DEFAULT 'default_user',
    storage_path  TEXT NOT NULL,          -- path in Supabase Storage bucket
    mime_type     TEXT NOT NULL DEFAULT 'image/jpeg',
    taken_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    location      TEXT,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    is_favorite   BOOLEAN NOT NULL DEFAULT FALSE,
    -- AI analysis stored as structured JSONB
    description       TEXT DEFAULT '',
    people            TEXT[] DEFAULT '{}',
    people_count      INTEGER DEFAULT 0,
    emotion           TEXT DEFAULT 'neutral',
    emotion_score     FLOAT DEFAULT 0.5,
    objects           TEXT[] DEFAULT '{}',
    clothing          TEXT[] DEFAULT '{}',
    colors            TEXT[] DEFAULT '{}',
    scene             TEXT DEFAULT '',
    ocr_text          TEXT DEFAULT '',
    tags              TEXT[] DEFAULT '{}',
    event_type        TEXT DEFAULT '',
    face_descriptors  TEXT[] DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS photos_user_taken ON photos(user_id, taken_at DESC);
CREATE INDEX IF NOT EXISTS photos_user_fav   ON photos(user_id, is_favorite);
CREATE INDEX IF NOT EXISTS photos_emotion    ON photos(user_id, emotion);
CREATE INDEX IF NOT EXISTS photos_event      ON photos(user_id, event_type);
-- Full-text search index over all text fields
CREATE INDEX IF NOT EXISTS photos_fts ON photos USING GIN (
    to_tsvector('english',
        coalesce(description,'') || ' ' ||
        coalesce(ocr_text,'') || ' ' ||
        coalesce(scene,'') || ' ' ||
        coalesce(event_type,'') || ' ' ||
        coalesce(emotion,'') || ' ' ||
        coalesce(location,'') || ' ' ||
        array_to_string(tags,'  ') || ' ' ||
        array_to_string(people,' ') || ' ' ||
        array_to_string(objects,' ') || ' ' ||
        array_to_string(clothing,' ')
    )
);

-- ── Conversations ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS conversations (
    id         UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id    TEXT NOT NULL DEFAULT 'default_user',
    messages   JSONB NOT NULL DEFAULT '[]',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS conv_user ON conversations(user_id, updated_at DESC);

-- ── Insights cache ────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS insights_cache (
    user_id      TEXT PRIMARY KEY DEFAULT 'default_user',
    insights     JSONB NOT NULL DEFAULT '[]',
    photo_count  INTEGER DEFAULT 0,
    generated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ── User settings ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS user_settings (
    user_id                TEXT PRIMARY KEY DEFAULT 'default_user',
    user_name              TEXT DEFAULT 'You',
    notifications_enabled  BOOLEAN DEFAULT TRUE,
    auto_insights          BOOLEAN DEFAULT TRUE,
    updated_at             TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ── Storage bucket ────────────────────────────────────────────
-- Create this in Supabase Dashboard → Storage → New Bucket
-- Bucket name: "photos"
-- Public: true (or use signed URLs)
-- Run this to create via SQL:
INSERT INTO storage.buckets (id, name, public)
VALUES ('photos', 'photos', true)
ON CONFLICT (id) DO NOTHING;

-- Allow public read on the photos bucket
CREATE POLICY IF NOT EXISTS "Photos are publicly readable"
ON storage.objects FOR SELECT
USING (bucket_id = 'photos');

CREATE POLICY IF NOT EXISTS "Anyone can upload photos"
ON storage.objects FOR INSERT
WITH CHECK (bucket_id = 'photos');

CREATE POLICY IF NOT EXISTS "Anyone can delete photos"
ON storage.objects FOR DELETE
USING (bucket_id = 'photos');

-- ============================================================
-- Done! Your schema is ready.
-- ============================================================
