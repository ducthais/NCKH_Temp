# tests/unit/test_scorer.py
from src.matching.scorer import estimate_years, jaccard, CandidateRecord

# --- estimate_years ---
def test_estimate_years_basic():
    ranges = [{"start": "2020", "end": "2023"}]
    assert estimate_years(ranges) == 3.0

def test_estimate_years_present():
    import datetime
    current = datetime.datetime.now().year
    ranges = [{"start": "2020", "end": "present"}]
    result = estimate_years(ranges)
    assert result == float(current - 2020)

def test_estimate_years_hientai():
    import datetime
    current = datetime.datetime.now().year
    ranges = [{"start": "2022", "end": "hiện tại"}]
    assert estimate_years(ranges) == float(current - 2022)

def test_estimate_years_multiple_ranges():
    ranges = [
        {"start": "2018", "end": "2020"},
        {"start": "2021", "end": "2023"},
    ]
    assert estimate_years(ranges) == 4.0

def test_estimate_years_invalid():
    ranges = [{"start": "invalid", "end": "2023"}]
    assert estimate_years(ranges) == 0.0

def test_estimate_years_cap_at_30():
    ranges = [{"start": "1980", "end": "2026"}]
    assert estimate_years(ranges) == 30.0

# --- jaccard ---
def test_jaccard_perfect():
    assert jaccard({"a", "b"}, {"a", "b"}) == 1.0

def test_jaccard_no_overlap():
    assert jaccard({"a"}, {"b"}) == 0.0

def test_jaccard_partial():
    result = jaccard({"a", "b", "c"}, {"a", "b", "d"})
    # intersection=2, union=4 → 0.5
    assert abs(result - 0.5) < 1e-6

def test_jaccard_empty_both():
    assert jaccard(set(), set()) == 0.0

def test_jaccard_one_empty():
    assert jaccard({"a"}, set()) == 0.0
