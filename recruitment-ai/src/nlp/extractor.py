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

def build_skill_ruler(skills_csv=None):
    if skills_csv is None:
        skills_csv = Path(__file__).parent.parent.parent / "data/dictionaries/skills.csv"
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

import os
import warnings
from transformers import pipeline

NER_PIPELINE = None
NER_LOADED = False
MODEL_PATH = Path(__file__).parent.parent.parent / "experiments" / "phobert-ner-final"

if MODEL_PATH.exists():
    try:
        NER_PIPELINE = pipeline(
            "ner", 
            model=str(MODEL_PATH), 
            aggregation_strategy="simple"
        )
        NER_LOADED = True
        print("Da tai thanh cong mo hinh NER Deep Learning!")
    except Exception as e:
        print(f"Loi khi tai mo hinh NER: {e}")

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

    dl_skills = set()
    if NER_PIPELINE is not None:
        try:
            # Lấy 2000 ký tự đầu tiên để tránh vượt quá max_length của model
            results = NER_PIPELINE(text[:2000])
            for ent in results:
                word = ent["word"].strip(" _")
                if len(word) < 2: continue
                if ent["entity_group"] == "SKILL":
                    dl_skills.add(word)
                elif ent["entity_group"] == "JOB_TITLE":
                    job_titles.append(word)
                elif ent["entity_group"] == "DEGREE":
                    degrees.append(word)
        except Exception:
            pass

    skills.extend([s.lower() for s in dl_skills])
    skills = sorted(set(skills))

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
