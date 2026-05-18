# tests/unit/test_normalizer.py
from src.nlp.normalizer import normalize_skill

def test_normalize_skill_exact():
    assert normalize_skill("python") == "python"

def test_normalize_skill_fuzzy():
    assert normalize_skill("PostgreSQL") == "sql"