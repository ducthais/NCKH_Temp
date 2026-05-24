# tests/unit/test_normalizer.py
from src.nlp.normalizer import normalize_skill, normalize_skill_list

# --- Exact match ---
def test_normalize_skill_exact_python():
    assert normalize_skill("python") == "python"

def test_normalize_skill_exact_docker():
    assert normalize_skill("docker") == "docker"

# --- Synonym / alias lookup ---
def test_normalize_skill_synonym_reactjs():
    result = normalize_skill("reactjs")
    assert result == "react"

def test_normalize_skill_synonym_nodejs():
    result = normalize_skill("node.js")
    assert result == "nodejs"

def test_normalize_skill_synonym_ml():
    result = normalize_skill("machine learning")
    assert result == "machine learning"

# --- Case insensitivity ---
def test_normalize_skill_case_insensitive():
    assert normalize_skill("Python") == "python"
    assert normalize_skill("DOCKER") == "docker"

# --- Unknown skill returns None ---
def test_normalize_skill_unknown():
    result = normalize_skill("xyzzyunknowntech9999")
    assert result is None

# --- Empty string ---
def test_normalize_skill_empty():
    assert normalize_skill("") is None
    assert normalize_skill("   ") is None

# --- List normalization ---
def test_normalize_skill_list():
    raw = ["Python", "reactjs", "xyzzyunknown", "Docker"]
    result = normalize_skill_list(raw)
    assert "python" in result
    assert "react" in result
    assert "docker" in result
    assert "xyzzyunknown" not in result

def test_normalize_skill_list_dedup():
    raw = ["python", "Python", "py"]
    result = normalize_skill_list(raw)
    assert result.count("python") == 1