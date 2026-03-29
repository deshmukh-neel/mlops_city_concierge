-- Postgres initialisation script
-- Runs automatically on first container start via docker-entrypoint-initdb.d/

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
