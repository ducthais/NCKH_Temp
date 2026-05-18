# src/app/streamlit_app.py
from __future__ import annotations
import json
import streamlit as st
from pathlib import Path
from src.parsers.pdf_parser import parse_pdf_native
from src.parsers.ocr_tesseract import ocr_pdf
from src.nlp.sectioner import split_sections
from src.nlp.extractor import extract_entities
from src.nlp.normalizer import normalize_skill_list
from src.nlp.embedder import Embedder
from src.matching.scorer import CandidateRecord, estimate_years, score_candidate

st.set_page_config(page_title="Recruitment AI MVP", layout="wide")
st.title("Recruitment AI MVP")

embedder = Embedder()

jd_text = st.text_area("Nhập JD", height=220)
uploaded_files = st.file_uploader("Upload CV PDF", type=["pdf"], accept_multiple_files=True)

if st.button("Chấm điểm") and jd_text and uploaded_files:
    results = []
    for f in uploaded_files:
        temp_path = Path("data/interim") / f.name
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path.write_bytes(f.read())

        parsed = parse_pdf_native(temp_path)
        if parsed["needs_ocr"]:
            parsed = ocr_pdf(temp_path)

        sections = split_sections(parsed["raw_text"])
        extracted = extract_entities(parsed["raw_text"], sections)
        skills = normalize_skill_list(extracted["skills_raw"])

        cv = CandidateRecord(
            candidate_id=parsed["candidate_id"],
            raw_text=parsed["raw_text"],
            skills_normalized=skills,
            years_experience_est=estimate_years(extracted["date_ranges"]),
            education_text=extracted["education_text"],
        )

        jd_skills = normalize_skill_list(extract_entities(jd_text, split_sections(jd_text))["skills_raw"])
        row = score_candidate(jd_text, jd_skills, cv, embedder)
        results.append(row)

    results = sorted(results, key=lambda x: x["total_score"], reverse=True)
    st.subheader("Xếp hạng")
    st.dataframe(results, use_container_width=True)

    if results:
        top = results[0]
        st.subheader(f"Chi tiết: {top['candidate_id']}")
        st.json(top)
