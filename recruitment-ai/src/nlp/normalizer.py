from __future__ import annotations
import csv
from rapidfuzz import process, fuzz, utils
from pathlib import Path

def load_skill_catalog(skills_csv=None):
    if skills_csv is None:
        skills_csv = Path(__file__).parent.parent.parent / "data/dictionaries/skills.csv"
    catalog = {}
    aliases = {}
    with open(skills_csv, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            canonical = row["canonical"].strip().lower()
            catalog[canonical] = float(row.get("weight", 1.0))
            aliases[canonical] = canonical
            for s in row["synonyms"].split("|"):
                s = s.strip().lower()
                if s:
                    aliases[s] = canonical
    return catalog, aliases

CATALOG, ALIASES = load_skill_catalog()

def normalize_skill(term: str, threshold: int = 92) -> str | None:
    term = term.strip().lower()
    if not term or len(term) < 2:
        return None

    if term in ALIASES:
        return ALIASES[term]

    # Skip short terms (<=3 chars) for fuzzy matching to avoid false positives
    # from NER fragments like 'git', 'sql' → these still match via exact ALIASES lookup above
    if len(term) <= 3:
        return None

    candidate = process.extractOne(
        term,
        list(CATALOG.keys()),
        scorer=fuzz.WRatio,
        processor=utils.default_process,
    )
    if candidate and candidate[1] >= threshold:
        return candidate[0]
    return None

def normalize_skill_list(skills_raw: list[str]) -> list[str]:
    normalized = []
    for skill in skills_raw:
        x = normalize_skill(skill)
        if x:
            normalized.append(x)
    return sorted(set(normalized))
