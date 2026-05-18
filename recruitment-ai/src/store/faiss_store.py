# src/store/faiss_store.py
from __future__ import annotations
from pathlib import Path
import faiss
import json
import numpy as np

class FaissStore:
    def __init__(self, dim: int):
        self.index = faiss.IndexFlatIP(dim)
        self.ids = []

    def add(self, ids: list[str], vectors: np.ndarray):
        vectors = vectors.astype("float32")
        faiss.normalize_L2(vectors)
        self.index.add(vectors)
        self.ids.extend(ids)

    def search(self, query_vec: np.ndarray, top_k: int = 10):
        query_vec = query_vec.astype("float32").reshape(1, -1)
        faiss.normalize_L2(query_vec)
        scores, indices = self.index.search(query_vec, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append({"candidate_id": self.ids[idx], "score": float(score)})
        return results

    def save(self, index_path="artifacts/faiss/cv.index", meta_path="artifacts/faiss/ids.json"):
        Path(index_path).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, index_path)
        Path(meta_path).write_text(json.dumps(self.ids, ensure_ascii=False), encoding="utf-8")
