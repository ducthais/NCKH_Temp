from __future__ import annotations
import csv
import re
from pathlib import Path
import spacy

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"(\+?\d[\d\-\.\s]{8,}\d)")
DATE_RANGE_RE = re.compile(
    r"((?:19|20)\d{2})\s*[-–/]\s*((?:19|20)\d{2}|present|now|hiện tại|nay)",
    re.I
)

DEGREE_KEYWORDS = [
    "cử nhân", "kỹ sư", "thạc sĩ", "tiến sĩ",
    "bachelor", "engineer", "master", "phd"
]

JOB_TITLE_HINTS = [
    "developer", "engineer", "analyst", "scientist",
    "backend", "frontend", "fullstack", "intern",
    "chuyên viên", "kỹ sư", "thực tập sinh"
]

def build_skill_ruler(skills_csv="data/dictionaries/skills.csv"):
    nlp = spacy.blank("xx")
    ruler = nlp.add_pipe("entity_ruler", config={"overwrite_ents": True})

    patterns = []
    with open(skills_csv, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            canonical = row["canonical"].strip().lower()
            synonyms = [s.strip().lower() for s in row["synonyms"].split("|") if s.strip()]
            for synonym in [canonical] + synonyms:
                if synonym:
                    token_pattern = [{"LOWER": token} for token in synonym.split()]
                    patterns.append({"label": "SKILL", "pattern": token_pattern, "id": canonical})

    ruler.add_patterns(patterns)
    return nlp

NLP = build_skill_ruler()

def extract_entities(text: str, sections: dict[str, str]) -> dict:
    doc = NLP(text)

    skills = sorted({ent.ent_id_ or ent.text.lower() for ent in doc.ents if ent.label_ == "SKILL"})
    emails = sorted(set(EMAIL_RE.findall(text)))
    phones = sorted(set(x[0] if isinstance(x, tuple) else x for x in PHONE_RE.findall(text)))
    date_ranges = DATE_RANGE_RE.findall(text)

    degrees = []
    for line in text.splitlines():
        low = line.lower()
        if any(k in low for k in DEGREE_KEYWORDS):
            degrees.append(line.strip())

    job_titles = []
    for line in sections.get("EXPERIENCE", "").splitlines():
        low = line.lower()
        if any(k in low for k in JOB_TITLE_HINTS):
            job_titles.append(line.strip())

    return {
        "emails": emails,
        "phones": phones,
        "skills_raw": skills,
        "degrees": degrees[:5],
        "job_titles": job_titles[:10],
        "date_ranges": [{"start": s, "end": e} for s, e in date_ranges],
        "experience_text": sections.get("EXPERIENCE", ""),
        "education_text": sections.get("EDUCATION", ""),
        "skills_text": sections.get("SKILLS", ""),
    }
