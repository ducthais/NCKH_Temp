from __future__ import annotations
import re
from collections import defaultdict

SECTION_PATTERNS = {
    "CONTACT": r"^(thông tin liên hệ|contact|liên hệ)$",
    "SUMMARY": r"^(mục tiêu nghề nghiệp|tóm tắt|summary|profile|objective)$",
    "EXPERIENCE": r"^(kinh nghiệm|work experience|experience|employment)$",
    "EDUCATION": r"^(học vấn|education|academic)$",
    "SKILLS": r"^(kỹ năng|skills|technical skills)$",
    "PROJECTS": r"^(dự án|projects?)$",
    "CERTS": r"^(chứng chỉ|certificates?|certifications?)$",
    "LANGUAGES": r"^(ngoại ngữ|languages?)$",
}

def normalize_line(line: str) -> str:
    return re.sub(r"\s+", " ", line.strip().lower())

def split_sections(text: str) -> dict[str, str]:
    current = "HEADER"
    buffers = defaultdict(list)

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        norm = normalize_line(line)
        matched = None
        for section_name, pattern in SECTION_PATTERNS.items():
            if re.match(pattern, norm, flags=re.I):
                matched = section_name
                break

        if matched:
            current = matched
        else:
            buffers[current].append(line)

    return {k: "\n".join(v).strip() for k, v in buffers.items() if v}
