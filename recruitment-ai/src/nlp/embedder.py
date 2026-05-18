from __future__ import annotations
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
E5_MODEL = "intfloat/multilingual-e5-large"

class Embedder:
    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def encode_query(self, texts: list[str]):
        if "e5" in self.model_name.lower():
            texts = [f"query: {t}" for t in texts]
        return self.model.encode(texts, normalize_embeddings=True)

    def encode_passage(self, texts: list[str]):
        if "e5" in self.model_name.lower():
            texts = [f"passage: {t}" for t in texts]
        return self.model.encode(texts, normalize_embeddings=True)
