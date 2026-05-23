from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
import numpy as np

CURRENT_YEAR = datetime.now().year

@dataclass
class CandidateRecord:
    candidate_id: str
    raw_text: str
    skills_normalized: list[str]
    years_experience_est: float
    education_text: str

def estimate_years(date_ranges: list[dict]) -> float:
    years = 0
    for item in date_ranges:
        try:
            start = int(item["start"])
            end_raw = item["end"].lower()
            end = CURRENT_YEAR if end_raw in {"present", "now", "hiện tại", "nay"} else int(end_raw)
            if end >= start:
                years += end - start
        except Exception:
            pass
    return max(0, min(years, 30))

def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / max(len(a | b), 1)

def score_candidate(jd_text: str, jd_skills: list[str], cv: CandidateRecord, embedder, bm25_score: float = 0.0) -> dict:
    q_emb = embedder.encode_query([jd_text])[0]
    d_emb = embedder.encode_passage([cv.raw_text])[0]
    semantic = float(np.dot(q_emb, d_emb))

    skill_overlap = jaccard(set(jd_skills), set(cv.skills_normalized))
    years_score = min(cv.years_experience_est / 5.0, 1.0)

    # Simplified Hybrid heuristic: normalize bm25 roughly by dividing by 30 (arbitrary max for short docs)
    norm_bm25 = min(bm25_score / 30.0, 1.0)

    total = (
        0.30 * norm_bm25 +
        0.40 * semantic +
        0.20 * skill_overlap +
        0.10 * years_score
    )

    return {
        "candidate_id": cv.candidate_id,
        "bm25_raw": round(bm25_score, 4),
        "semantic": round(semantic, 4),
        "skill_overlap": round(skill_overlap, 4),
        "years_score": round(years_score, 4),
        "total_score": round(total, 4),
        "matched_skills": sorted(set(jd_skills) & set(cv.skills_normalized)),
        "missing_skills": sorted(set(jd_skills) - set(cv.skills_normalized)),
    }
