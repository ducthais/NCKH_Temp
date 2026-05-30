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

def parse_month(m_str: str | None) -> int:
    if not m_str:
        return 1
    import re
    m_str = m_str.lower().strip()
    if m_str.isdigit():
        return max(1, min(int(m_str), 12))
    match = re.search(r"tháng\s*(\d{1,2})", m_str)
    if match:
        return max(1, min(int(match.group(1)), 12))
    months_map = {
        "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
        "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12
    }
    for k, v in months_map.items():
        if m_str.startswith(k):
            return v
    return 1

def estimate_years(date_ranges: list[dict]) -> float:
    intervals = []
    for item in date_ranges:
        try:
            start_year = int(item["start_year"])
            start_month = parse_month(item.get("start_month"))
            
            end_present = item.get("end_present", "")
            if end_present and end_present.lower() in {"present", "now", "hiện tại", "nay"}:
                end_year = CURRENT_YEAR
                end_month = datetime.now().month
            else:
                end_year = int(item["end_year"])
                end_month = parse_month(item.get("end_month")) if item.get("end_month") else 12
            
            start_total = start_year * 12 + start_month
            end_total = end_year * 12 + end_month
            
            if end_total >= start_total:
                intervals.append([start_total, end_total])
        except Exception:
            pass
            
    if not intervals:
        return 0.0
        
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]
    for current in intervals[1:]:
        last = merged[-1]
        if current[0] <= last[1] + 1:
            last[1] = max(last[1], current[1])
        else:
            merged.append(current)
            
    total_months = sum(e - s + 1 for s, e in merged)
    years = total_months / 12.0
    return max(0.0, min(float(years), 30.0))

def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / max(len(a | b), 1)

def score_candidate(jd_text: str, jd_skills: list[str], cv: CandidateRecord, embedder, bm25_raw: float = 0.0, bm25_norm: float = 0.0) -> dict:
    q_emb = embedder.encode_query([jd_text])[0]
    d_emb = embedder.encode_passage([cv.raw_text])[0]
    semantic = float(np.dot(q_emb, d_emb))

    skill_overlap = jaccard(set(jd_skills), set(cv.skills_normalized))
    years_score = min(cv.years_experience_est / 5.0, 1.0)

    total = (
        0.30 * bm25_norm +
        0.40 * semantic +
        0.20 * skill_overlap +
        0.10 * years_score
    )

    return {
        "candidate_id": cv.candidate_id,
        "bm25_raw": round(bm25_raw, 4),
        "bm25_norm": round(bm25_norm, 4),
        "semantic": round(semantic, 4),
        "skill_overlap": round(skill_overlap, 4),
        "years_score": round(years_score, 4),
        "total_score": round(total, 4),
        "matched_skills": sorted(set(jd_skills) & set(cv.skills_normalized)),
        "missing_skills": sorted(set(jd_skills) - set(cv.skills_normalized)),
    }
