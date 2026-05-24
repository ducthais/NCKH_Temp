# tests/unit/test_sectioner.py
from src.nlp.sectioner import split_sections

# --- English headers ---
def test_section_skills_english():
    text = "Skills\nPython, Docker"
    result = split_sections(text)
    assert "SKILLS" in result
    assert "Python" in result["SKILLS"]

def test_section_experience_english():
    text = "Work Experience\n2020-2023 Software Engineer at VNG"
    result = split_sections(text)
    assert "EXPERIENCE" in result

def test_section_education_english():
    text = "Education\nBachelor of Computer Science, 2022"
    result = split_sections(text)
    assert "EDUCATION" in result

def test_section_projects_english():
    text = "Projects\nBuild a chatbot"
    result = split_sections(text)
    assert "PROJECTS" in result

def test_section_applied_projects():
    text = "Applied Projects\nRAG system using LangChain"
    result = split_sections(text)
    assert "PROJECTS" in result

# --- Vietnamese headers ---
def test_section_kynang_vietnamese():
    text = "Kỹ năng\nPython, Java"
    result = split_sections(text)
    assert "SKILLS" in result

def test_section_kinhnghiem_vietnamese():
    text = "Kinh nghiệm làm việc\n2021-2023 Lập trình viên tại FPT"
    result = split_sections(text)
    assert "EXPERIENCE" in result

def test_section_hocvan_vietnamese():
    text = "Học vấn\nĐại học Bách Khoa, 2022"
    result = split_sections(text)
    assert "EDUCATION" in result

def test_section_thanh_tuu_vietnamese():
    text = "Thành tựu nổi bật\nGiải nhất Hackathon 2025"
    result = split_sections(text)
    assert "AWARDS" in result

def test_section_soft_skills():
    text = "Kỹ năng mềm\nGiao tiếp tốt, Làm việc nhóm"
    result = split_sections(text)
    assert "SOFT_SKILLS" in result

def test_section_certificates():
    text = "Chứng chỉ\nIELTS 6.5, AWS Certified"
    result = split_sections(text)
    assert "CERTS" in result

def test_section_mixed_cv():
    text = """Summary
Software developer with 3 years experience.
Education
SGU, 2022
Skills
Python, React
Work Experience
2021-2023 Backend Developer"""
    result = split_sections(text)
    assert "SUMMARY" in result
    assert "EDUCATION" in result
    assert "SKILLS" in result
    assert "EXPERIENCE" in result

def test_empty_text():
    result = split_sections("")
    assert result == {} or isinstance(result, dict)
