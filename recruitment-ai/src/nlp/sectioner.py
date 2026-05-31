from __future__ import annotations
import re
from collections import defaultdict

SECTION_PATTERNS = {
    "CONTACT": r"^(thông tin liên hệ|contact\s*(info(rmation)?)?|liên hệ|liên lạc|personal\s*info(rmation)?)\s*[:\-\|]?\s*$",
    "SUMMARY": r"^(mục tiêu nghề nghiệp|tóm tắt|summary|profile\s*(summary)?|career\s*objective|objective|giới thiệu|giới thiệu bản thân|tóm tắt năng lực|năng lực|about\s*me|professional\s*summary|career\s*summary|executive\s*summary|personal\s*statement)\s*[:\-\|]?\s*$",
    "EXPERIENCE": r"^(kinh nghiệm|work\s*experience|experience|employment(\s*history)?|kinh nghiệm làm việc|kinh nghiệm thực tế|programming\s*experience|work\s*history|lịch sử làm việc|professional\s*experience|career\s*history|relevant\s*experience|employment\s*experience|job\s*experience)\s*[:\-\|]?\s*$",
    "EDUCATION": r"^(học vấn|education(al\s*background)?|academic(\s*background)?|trình độ học vấn|quá trình học tập|học tập|bằng cấp|bằng cấp \& chứng chỉ|trình độ|academic\s*qualifications?|education\s*&?\s*training)\s*[:\-\|]?\s*$",
    "SKILLS": r"^(kỹ năng|skills|technical\s*skills|kỹ năng chuyên môn|kỹ năng kỹ thuật|kỹ năng chuyên ngành|kĩ năng|kĩ năng chuyên ngành|chuyên môn|năng lực chuyên môn|programming\s*skills|core\s*skills|areas?\s*of\s*expertise|key\s*skills|professional\s*skills|competenc(y|ies)|core\s*competenc(y|ies)|technical\s*competenc(y|ies)|it\s*skills|computer\s*skills)\s*[:\-\|]?\s*$",
    "SOFT_SKILLS": r"^(kỹ năng mềm|soft\s*skills|kĩ năng mềm|kỹ năng mềm \& cá nhân|interpersonal\s*skills|personal\s*skills)\s*[:\-\|]?\s*$",
    "PROJECTS": r"^(dự án|projects?|applied\s*projects?|personal\s*projects?|side\s*projects?|dự án cá nhân|dự án tiêu biểu|academic\s*projects?|hoạt động dự án|key\s*projects?|notable\s*projects?|selected\s*projects?)\s*[:\-\|]?\s*$",
    "RESEARCH": r"^(nghiên cứu|research|research\s*experience|công trình nghiên cứu|bài báo|publications?)\s*[:\-\|]?\s*$",
    "CERTS": r"^(chứng chỉ|certificates?|certifications?|chứng chỉ khác|bằng cấp \& chứng chỉ|licenses?\s*&?\s*certifications?|professional\s*certifications?|credentials?)\s*[:\-\|]?\s*$",
    "LANGUAGES": r"^(ngoại ngữ|languages?|foreign\s*languages?|language\s*skills?|language\s*proficiency)\s*[:\-\|]?\s*$",
    "AWARDS": r"^(giải thưởng|awards?(\s*&?\s*honors?)?|honors?|achievements?|thành tích|thành tựu|thành tựu nổi bật|giải thưởng \& thành tích|accomplishments?)\s*[:\-\|]?\s*$",
    "ACTIVITIES": r"^(hoạt động|activities|hoạt động ngoại khóa|extracurricular(\s*activities)?|volunteer(ing)?(\s*experience)?|tình nguyện|câu lạc bộ|organizations?|community\s*(involvement|service))\s*[:\-\|]?\s*$",
    "REFERENCES": r"^(người tham chiếu|references?|người giới thiệu)\s*[:\-\|]?\s*$",
    "INTERESTS": r"^(sở thích|interests?|hobbies?)\s*[:\-\|]?\s*$",
}

# Fallback keywords để nhận diện section header khi OCR bị lỗi
# Mỗi keyword map đến section name. Dùng khi dòng ngắn (< 40 ký tự)
# và chứa keyword quan trọng, giúp handle text bị garble nhẹ từ OCR.
_FUZZY_KEYWORDS = {
    "EXPERIENCE": ["experience", "employment", "career history", "kinh nghiệm", "work history"],
    "EDUCATION": ["education", "academic", "học vấn", "trình độ"],
    "SKILLS": ["skills", "competenc", "kỹ năng", "kĩ năng", "expertise"],
    "CERTS": ["certification", "certificate", "chứng chỉ", "credential", "licenses"],
    "PROJECTS": ["project", "dự án"],
    "SUMMARY": ["summary", "objective", "profile", "mục tiêu", "giới thiệu", "about me"],
    "AWARDS": ["award", "honor", "achievement", "giải thưởng", "thành tích"],
    "LANGUAGES": ["language", "ngoại ngữ"],
    "ACTIVITIES": ["activit", "volunteer", "hoạt động", "extracurricular"],
}


def normalize_line(line: str) -> str:
    """Chuẩn hóa dòng text: lowercase, bỏ ký tự OCR noise, gộp khoảng trắng."""
    line = line.strip()
    # Bỏ ký tự OCR noise phổ biến ở đầu/cuối dòng
    line = re.sub(r'^[\s_\-|:/\u2022\u2023\u2043\u25aa\u25b8\u25b6\u25c6\u25a0\u25a1\u00ab\u00bb\u201e\u201c\u201d\u2018\u2019\u2013\u2014\u00b7\u2219\u2027\u25cf\u25cb]+', '', line)
    line = re.sub(r'[\s_\-|:/\u2022\u2023\u2043\u25aa\u25b8\u25b6\u25c6\u25a0\u25a1\u00ab\u00bb\u201e\u201c\u201d\u2018\u2019\u2013\u2014\u00b7\u2219\u2027\u25cf\u25cb]+$', '', line)
    # Gộp khoảng trắng
    return re.sub(r"\s+", " ", line.lower()).strip()


def _fuzzy_match_section(norm: str) -> str | None:
    """Fallback: nhận diện section header bằng keyword matching.
    Chỉ áp dụng cho dòng ngắn (< 50 ký tự) để tránh false positive."""
    if len(norm) > 50:
        return None
    for section_name, keywords in _FUZZY_KEYWORDS.items():
        for kw in keywords:
            if kw in norm:
                return section_name
    return None


def split_sections(text: str) -> dict[str, str]:
    current = "HEADER"
    buffers = defaultdict(list)

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        norm = normalize_line(line)
        if not norm:
            continue

        matched = None
        # Ưu tiên exact regex match
        for section_name, pattern in SECTION_PATTERNS.items():
            if re.match(pattern, norm, flags=re.I):
                matched = section_name
                break

        # Fallback: fuzzy keyword match cho dòng ngắn
        if not matched:
            matched = _fuzzy_match_section(norm)

        if matched:
            current = matched
        else:
            buffers[current].append(line)

    return {k: "\n".join(v).strip() for k, v in buffers.items() if v}
