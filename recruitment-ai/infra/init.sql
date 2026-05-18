CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS cv_documents (
  id bigserial PRIMARY KEY,
  candidate_id text UNIQUE NOT NULL,
  raw_text text NOT NULL,
  embedding vector(384),
  metadata jsonb DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS cv_documents_embedding_hnsw
ON cv_documents USING hnsw (embedding vector_cosine_ops);
