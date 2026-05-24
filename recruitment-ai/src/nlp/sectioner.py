from __future__ import annotations
import re
from collections import defaultdict

SECTION_PATTERNS = {
    "CONTACT": r"^(thông tin liên hệ|contact|liên hệ|liên lạc)$",
    "SUMMARY": r"^(mục tiêu nghề nghiệp|tóm tắt|summary|profile|objective|giới thiệu|giới thiệu bản thân|tóm tắt năng lực|năng lực|about me)$",
    "EXPERIENCE": r"^(kinh nghiệm|work experience|experience|employment|kinh nghiệm làm việc|kinh nghiệm thực tế|programming experience|work history|lịch sử làm việc)$",
    "EDUCATION": r"^(học vấn|education|academic|trình độ học vấn|quá trình học tập|học tập|bằng cấp|bằng cấp & chứng chỉ|trình độ)$",
    "SKILLS": r"^(kỹ năng|skills|technical skills|kỹ năng chuyên môn|kỹ năng kỹ thuật|kỹ năng chuyên ngành|kĩ năng|kĩ năng chuyên ngành|chuyên môn|năng lực chuyên môn|programming skills|core skills|areas of expertise)$",
    "SOFT_SKILLS": r"^(kỹ năng mềm|soft skills|kĩ năng mềm|kỹ năng mềm & cá nhân)$",
    "PROJECTS": r"^(dự án|projects?|applied projects?|personal projects?|side projects?|dự án cá nhân|dự án tiêu biểu|academic projects?|hoạt động dự án|dự án tiêu biểu)$",
    "RESEARCH": r"^(nghiên cứu|research|research experience|công trình nghiên cứu|bài báo|publications?)$",
    "CERTS": r"^(chứng chỉ|certificates?|certifications?|chứng chỉ khác|bằng cấp & chứng chỉ|licenses? & certifications?)$",
    "LANGUAGES": r"^(ngoại ngữ|languages?|foreign languages?|language skills?)$",
    "AWARDS": r"^(giải thưởng|awards?|honors?|achievements?|thành tích|thành tựu|thành tựu nổi bật|giải thưởng & thành tích)$",
    "ACTIVITIES": r"^(hoạt động|activities|hoạt động ngoại khóa|extracurricular|volunteer|tình nguyện|câu lạc bộ|organizations?)$",
    "REFERENCES": r"^(người tham chiếu|references?|người giới thiệu)$",
    "INTERESTS": r"^(sở thích|interests?|hobbies?)$",
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
