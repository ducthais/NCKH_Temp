# src/api/main.py
from __future__ import annotations
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Recruitment AI MVP")

class RankRequest(BaseModel):
    jd_text: str
    candidates: list[dict]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/rank")
def rank(req: RankRequest):
    # Ở bản MVP gọi scorer vào đây
    return {"count": len(req.candidates), "jd_preview": req.jd_text[:120]}
