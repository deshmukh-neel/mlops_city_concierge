-- Postgres initialisation script
-- Runs automatically on first container start via docker-entrypoint-initdb.d/
--
-- BASELINE SCHEMA ONLY — do not edit in place. New schema changes must ship as
-- Alembic migrations (`make migration MSG="..."`) so existing databases pick
-- them up. Editing this file only affects fresh container starts; Cloud SQL
-- prod and any teammate's existing local DB will silently miss the change.

-- Enable the pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Enable the uuid extension for primary keys
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Cities table: one row per city document chunk
CREATE TABLE IF NOT EXISTS city_chunks (
    id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    city        TEXT NOT NULL,
    content     TEXT NOT NULL,
    embedding   vector(1536),          -- matches text-embedding-3-small output dim
    metadata    JSONB DEFAULT '{}',
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

-- HNSW index for fast approximate nearest-neighbour search
CREATE INDEX IF NOT EXISTS city_chunks_embedding_idx
    ON city_chunks
    USING hnsw (embedding vector_cosine_ops);

-- Normalized Google Places storage for pull-once and local querying
CREATE TABLE IF NOT EXISTS places_raw (
    place_id              TEXT PRIMARY KEY,
    name                  TEXT,
    primary_type          TEXT,
    formatted_address     TEXT,
    latitude              DOUBLE PRECISION,
    longitude             DOUBLE PRECISION,
    rating                DOUBLE PRECISION,
    user_rating_count     INTEGER,
    price_level           TEXT,
    website_uri           TEXT,
    business_status       TEXT,
    maps_uri              TEXT,
    types                 TEXT[] DEFAULT '{}',
    editorial_summary     TEXT,
    regular_opening_hours JSONB DEFAULT '{}',
    source_query          TEXT,
    source_city           TEXT DEFAULT 'San Francisco',
    source_json           JSONB NOT NULL,
    source_updated_at     TIMESTAMPTZ DEFAULT NOW(),
    created_at            TIMESTAMPTZ DEFAULT NOW(),
    updated_at            TIMESTAMPTZ DEFAULT NOW()
);

-- Keep existing databases compatible when this file is re-run manually.
ALTER TABLE IF EXISTS places_raw ADD COLUMN IF NOT EXISTS primary_type TEXT;
ALTER TABLE IF EXISTS places_raw ADD COLUMN IF NOT EXISTS editorial_summary TEXT;
ALTER TABLE IF EXISTS places_raw ADD COLUMN IF NOT EXISTS regular_opening_hours JSONB DEFAULT '{}';

CREATE INDEX IF NOT EXISTS idx_places_raw_name ON places_raw(name);
CREATE INDEX IF NOT EXISTS idx_places_raw_types ON places_raw USING gin(types);
CREATE INDEX IF NOT EXISTS idx_places_raw_source_updated_at ON places_raw(source_updated_at);

-- Query hit evidence table: preserves every query/place match without churning places_raw.
CREATE TABLE IF NOT EXISTS place_query_hits (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    place_id        TEXT NOT NULL REFERENCES places_raw(place_id) ON DELETE CASCADE,
    query_text      TEXT NOT NULL,
    field_mode      TEXT NOT NULL,
    page_number     INTEGER NOT NULL,
    rank_in_page    INTEGER NOT NULL,
    seen_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (place_id, query_text, field_mode, page_number, rank_in_page)
);

CREATE INDEX IF NOT EXISTS idx_place_query_hits_place_id ON place_query_hits(place_id);
CREATE INDEX IF NOT EXISTS idx_place_query_hits_query_text ON place_query_hits(query_text);

-- Query-level checkpoint table for resumable ingestion runs
CREATE TABLE IF NOT EXISTS places_ingest_query_checkpoints (
    query_text              TEXT PRIMARY KEY,
    status                  TEXT NOT NULL,
    pages_processed         INTEGER NOT NULL DEFAULT 0,
    api_calls               INTEGER NOT NULL DEFAULT 0,
    rows_seen               INTEGER NOT NULL DEFAULT 0,
    rows_changed            INTEGER NOT NULL DEFAULT 0,
    last_error              TEXT,
    last_run_at             TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at            TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_places_ingest_query_status
    ON places_ingest_query_checkpoints(status);

-- Embeddings table (1 row per place) for semantic retrieval
CREATE TABLE IF NOT EXISTS place_embeddings (
    place_id              TEXT PRIMARY KEY REFERENCES places_raw(place_id) ON DELETE CASCADE,
    embedding             vector(1536) NOT NULL,
    embedding_model       TEXT NOT NULL,
    embedding_text        TEXT NOT NULL,
    embedded_at           TIMESTAMPTZ DEFAULT NOW(),
    source_updated_at     TIMESTAMPTZ NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_place_embeddings_vector
    ON place_embeddings
    USING hnsw (embedding vector_cosine_ops);
