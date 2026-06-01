from __future__ import annotations
import math
import re
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import numpy as np

CURRENT_YEAR = datetime.now().year

@dataclass
class CandidateRecord:
    candidate_id: str
    raw_text: str
    skills_normalized: list[str]
    years_experience_est: float
    education_text: str
    semantic_text: str = ""  # Focused text (SKILLS + EXPERIENCE) cho semantic encoding

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
            # Dùng regex để match các biến thể OCR typo: presant, presenl, ongoing, v.v.
            if end_present and re.search(r"pres[ae]n[tl]|now|current|hiện tại|nay|ongoing|till", end_present, re.I):
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


# ---------------------------------------------------------------------------
# Tải trọng số kỹ năng từ skills.csv (dùng cột "weight")
# ---------------------------------------------------------------------------
def _load_skill_weights(skills_csv=None) -> dict[str, float]:
    """Load skill weights from skills.csv.  Trả về dict {canonical_name: weight}."""
    if skills_csv is None:
        skills_csv = Path(__file__).parent.parent.parent / "data/dictionaries/skills.csv"
    weights = {}
    try:
        with open(skills_csv, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                canonical = row["canonical"].strip().lower()
                w = float(row.get("weight", 1.0))
                weights[canonical] = w
    except Exception:
        pass
    return weights

SKILL_WEIGHTS = _load_skill_weights()

# Trọng số mặc định cho kỹ năng không có trong từ điển
_DEFAULT_SKILL_WEIGHT = 0.5


def weighted_jaccard(cv_skills: set, jd_skills: set, skill_weights: dict[str, float] | None = None) -> float:
    """Tính Jaccard có trọng số, **hướng về phía JD** (JD-centric).

    Công thức:
        score = Σ weight(s) for s in (cv ∩ jd) / Σ weight(s) for s in jd

    Ý nghĩa: Đo tỷ lệ "mức độ đáp ứng yêu cầu JD" chứ không phải tổng
    hợp cả kỹ năng thừa của ứng viên.  Kỹ năng quan trọng (weight cao)
    đóng góp nhiều hơn kỹ năng phụ (weight thấp).
    """
    if not jd_skills:
        return 0.0
    if skill_weights is None:
        skill_weights = SKILL_WEIGHTS

    matched = cv_skills & jd_skills
    # Tử số: tổng trọng số các kỹ năng ứng viên đáp ứng
    numerator = sum(skill_weights.get(s, _DEFAULT_SKILL_WEIGHT) for s in matched)
    # Mẫu số: tổng trọng số tất cả kỹ năng JD yêu cầu
    denominator = sum(skill_weights.get(s, _DEFAULT_SKILL_WEIGHT) for s in jd_skills)
    return numerator / denominator if denominator > 0 else 0.0


def jaccard(a: set, b: set) -> float:
    """Jaccard đơn giản — giữ lại cho tương thích ngược nếu cần."""
    if not a and not b:
        return 0.0
    return len(a & b) / max(len(a | b), 1)


def score_candidate(jd_text: str, jd_skills: list[str], cv: CandidateRecord, embedder, bm25_raw: float = 0.0, bm25_norm: float = 0.0, jd_semantic_text: str = "") -> dict:
    # Sử dụng focused text (SKILLS + EXPERIENCE) cho semantic encoding
    # để giảm noise từ layout CV phức tạp, OCR artifacts, quotes, v.v.
    q_text = jd_semantic_text if jd_semantic_text.strip() else jd_text
    cv_text = cv.semantic_text if cv.semantic_text.strip() else cv.raw_text
    q_emb = embedder.encode_query([q_text])[0]
    d_emb = embedder.encode_passage([cv_text])[0]
    semantic = float(np.dot(q_emb, d_emb))

    cv_set = set(cv.skills_normalized)
    jd_set = set(jd_skills)

    # ---------------------------------------------------------------
    # (1) Weighted Jaccard: kỹ năng quan trọng đáp ứng → điểm cao hơn
    # ---------------------------------------------------------------
    skill_overlap = weighted_jaccard(cv_set, jd_set, SKILL_WEIGHTS)

    # ---------------------------------------------------------------
    # (2) Logarithmic experience: benchmark 10 năm, diminishing returns
    # ---------------------------------------------------------------
    years_score = min(math.log(cv.years_experience_est + 1) / math.log(10 + 1), 1.0)

    # ---------------------------------------------------------------
    # (3) Relevance gate: phạt nặng nếu không match kỹ năng nào từ JD
    #     - 0 kỹ năng match      → factor = 0.40  (giảm 60% tổng điểm)
    #     - 1-2 kỹ năng match    → factor tuyến tính 0.40 → 0.85
    #     - >= 3 kỹ năng match   → factor = 1.0   (không phạt)
    # ---------------------------------------------------------------
    n_matched = len(cv_set & jd_set)
    if n_matched == 0:
        relevance_factor = 0.40
    elif n_matched < 3:
        # Tuyến tính từ 0.40 (0 match) đến 1.0 (3 match)
        relevance_factor = 0.40 + (n_matched / 3.0) * 0.60
    else:
        relevance_factor = 1.0

    # ---------------------------------------------------------------
    # (4) Công thức tổng: ưu tiên kỹ năng > ngữ nghĩa > kinh nghiệm
    #     - skill_overlap:  35%  (tăng từ 20%)
    #     - semantic:       30%  (giảm từ 40%)
    #     - bm25_norm:      20%  (giảm từ 30%)
    #     - years_score:    15%  (tăng nhẹ từ 10%, nhưng bị gate bởi relevance)
    # ---------------------------------------------------------------
    raw_total = (
        0.20 * bm25_norm +
        0.30 * semantic +
        0.35 * skill_overlap +
        0.15 * years_score
    )

    total = raw_total * relevance_factor

    return {
        "candidate_id": cv.candidate_id,
        "bm25_raw": round(bm25_raw, 4),
        "bm25_norm": round(bm25_norm, 4),
        "semantic": round(semantic, 4),
        "skill_overlap": round(skill_overlap, 4),
        "years_score": round(years_score, 4),
        "years_experience_est": round(cv.years_experience_est, 1),
        "relevance_factor": round(relevance_factor, 2),
        "raw_total": round(raw_total, 4),
        "total_score": round(total, 4),
        "matched_skills": sorted(cv_set & jd_set),
        "missing_skills": sorted(jd_set - cv_set),
    }
