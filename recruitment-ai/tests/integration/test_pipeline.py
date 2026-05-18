# tests/integration/test_pipeline.py
from src.nlp.sectioner import split_sections
from src.nlp.extractor import extract_entities

def test_pipeline_smoke():
    text = """
    KỸ NĂNG
    Python, SQL
    KINH NGHIỆM
    Backend Developer tại ABC 2022-2024
    Email: abc@example.com
    """
    sections = split_sections(text)
    entities = extract_entities(text, sections)
    assert "abc@example.com" in entities["emails"]
    assert len(entities["skills_raw"]) >= 1
