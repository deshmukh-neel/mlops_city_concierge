-- Parallel embeddings table. Same schema as place_embeddings; lives next to it
-- so we can A/B-compare retrieval quality before flipping the app over.
-- Once W0a + W1 + W6 confirm v2 wins, v1 may be dropped in a follow-up PR.

CREATE TABLE IF NOT EXISTS place_embeddings_v2 (
    place_id              TEXT PRIMARY KEY REFERENCES places_raw(place_id) ON DELETE CASCADE,
    embedding             vector(1536) NOT NULL,
    embedding_model       TEXT NOT NULL,
    embedding_text        TEXT NOT NULL,
    embedded_at           TIMESTAMPTZ DEFAULT NOW(),
    source_updated_at     TIMESTAMPTZ NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_place_embeddings_v2_vector
    ON place_embeddings_v2
    USING hnsw (embedding vector_cosine_ops);

COMMENT ON TABLE place_embeddings_v2 IS
  'Cleaned embeddings (no URLs, no structured facts, with neighborhood + landmark names). Drives retrieval when EMBEDDING_TABLE=place_embeddings_v2.';
