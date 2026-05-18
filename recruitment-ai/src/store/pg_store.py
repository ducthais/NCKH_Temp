# src/store/pg_store.py
from __future__ import annotations
import json
import psycopg

def as_pgvector(values):
    return "[" + ",".join(f"{float(v):.6f}" for v in values) + "]"

def upsert_embedding(conn_str: str, candidate_id: str, raw_text: str, embedding, metadata: dict):
    with psycopg.connect(conn_str) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO cv_documents(candidate_id, raw_text, embedding, metadata)
                VALUES (%s, %s, %s::vector, %s::jsonb)
                ON CONFLICT (candidate_id) DO UPDATE
                SET raw_text = EXCLUDED.raw_text,
                    embedding = EXCLUDED.embedding,
                    metadata = EXCLUDED.metadata
                """,
                (candidate_id, raw_text, as_pgvector(embedding), json.dumps(metadata, ensure_ascii=False)),
            )
        conn.commit()

def query_similar(conn_str: str, query_embedding, top_k: int = 10):
    with psycopg.connect(conn_str) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT candidate_id, 1 - (embedding <=> %s::vector) AS cosine_sim
                FROM cv_documents
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (as_pgvector(query_embedding), as_pgvector(query_embedding), top_k),
            )
            return cur.fetchall()
