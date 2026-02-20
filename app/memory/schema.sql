CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS decision_traces (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    hypothesis TEXT NOT NULL,
    trace JSONB NOT NULL,
    summary TEXT NOT NULL,
    embedding vector(768) NOT NULL,
    failure_reason TEXT
);

CREATE INDEX IF NOT EXISTS decision_traces_embedding_idx
    ON decision_traces USING ivfflat (embedding vector_cosine_ops);
