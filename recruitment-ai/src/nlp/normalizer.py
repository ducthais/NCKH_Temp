from __future__ import annotations
import csv
from rapidfuzz import process, fuzz, utils

def load_skill_catalog(skills_csv="data/dictionaries/skills.csv"):
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

def normalize_skill(term: str, threshold: int = 88) -> str | None:
    term = term.strip().lower()
    if not term:
        return None

    if term in ALIASES:
        return ALIASES[term]

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
