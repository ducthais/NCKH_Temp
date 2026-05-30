import streamlit as st
import json
import os
from collections import Counter

import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt

from src.store.database import get_db
from src.store.models import Campaign, Candidate
from src.parsers.pdf_parser import parse_pdf_native
from src.parsers.ocr_tesseract import ocr_pdf, ocr_image
from src.parsers.docx_parser import parse_docx_native
from src.nlp.sectioner import split_sections
from src.nlp.extractor import extract_entities, NER_LOADED
from src.nlp.normalizer import normalize_skill_list
from src.nlp.embedder import Embedder
from src.matching.scorer import CandidateRecord, estimate_years, score_candidate, jaccard
from src.matching.bm25_retriever import BM25Retriever

st.set_page_config(page_title="Phân Tích CV", page_icon="🚀", layout="wide")
st.title("Phân tích CV")

db = next(get_db())

@st.cache_resource
def load_embedder():
    return Embedder()

embedder = load_embedder()

# --- Lấy danh sách Campaign ---
campaigns = db.query(Campaign).order_by(Campaign.created_at.desc()).all()
if not campaigns:
    st.warning("Vui lòng vào trang 'Quản đợt tuyển dụng' để tạo chiến dịch tuyển dụng trước!")
    st.stop()

campaign_options = {c.id: f"{c.job_title} ({c.created_at.strftime('%Y-%m-%d')})" for c in campaigns}
selected_campaign_id = st.selectbox(
    "Chọn đợt tuyển dụng:",
    options=list(campaign_options.keys()),
    format_func=lambda x: campaign_options[x]
)

selected_campaign = db.query(Campaign).filter(Campaign.id == selected_campaign_id).first()

with st.expander("Xem Mô tả công việc (JD)"):
    st.text(selected_campaign.job_description)

if not NER_LOADED:
    st.warning("Mô hình AI nhận diện thực thể (NER) chưa được tải. Hệ thống đang trích xuất kỹ năng bằng từ điển cơ bản.")

st.markdown("### Tải lên CV")
uploaded_files = st.file_uploader(
    "Tải lên tập CV ứng viên (PDF, DOCX, JPG, PNG)",
    type=["pdf", "docx", "jpg", "jpeg", "png"],
    accept_multiple_files=True
)

process_btn = st.button("Bắt đầu Phân tích", type="primary", use_container_width=True)

if process_btn and uploaded_files:
    with st.spinner("..."):
        # Trích xuất kỹ năng từ JD
        jd_sections = split_sections(selected_campaign.job_description)
        jd_entities = extract_entities(selected_campaign.job_description, jd_sections)
        jd_skills = normalize_skill_list(jd_entities.get("skills_raw", []))

        if not jd_skills:
            st.warning("Không tìm thấy kỹ năng nào trong JD! Kết quả chấm điểm có thể không chính xác.")

        cv_records = []       # list[CandidateRecord]
        cv_raw_data = []      # raw text + sections for display later
        bm25_docs = []        # list[dict] cho BM25Retriever.fit()

        progress_bar = st.progress(0)
        total_files = len(uploaded_files)

        for i, file in enumerate(uploaded_files):
            # Lưu file tạm
            temp_path = f"temp_{file.name}"
            with open(temp_path, "wb") as f:
                f.write(file.read())

            ext = file.name.lower().split('.')[-1]

            # Parse theo định dạng
            if ext == "pdf":
                parsed = parse_pdf_native(temp_path)
                if parsed.get("status") == "error" or not parsed.get("raw_text", "").strip() or parsed.get("needs_ocr"):
                    parsed = ocr_pdf(temp_path)
            elif ext == "docx":
                parsed = parse_docx_native(temp_path)
            else:
                parsed = ocr_image(temp_path)

            os.remove(temp_path)

            raw_text = parsed.get("raw_text", "")
            sections = split_sections(raw_text)
            entities = extract_entities(raw_text, sections)
            # BUG FIX #1: extractor trả về key "skills_raw", không phải "skills"
            cv_skills = normalize_skill_list(entities.get("skills_raw", []))
            years_exp = estimate_years(entities.get("date_ranges", []))

            candidate_id = file.name.rsplit('.', 1)[0]

            # BUG FIX #2: CandidateRecord dùng đúng field names trong scorer.py
            cv_rec = CandidateRecord(
                candidate_id=candidate_id,
                raw_text=raw_text,
                skills_normalized=cv_skills,
                years_experience_est=years_exp,
                education_text=sections.get("EDUCATION", "")
            )
            cv_records.append(cv_rec)
            cv_raw_data.append({
                "raw_text": raw_text,
                "projects_text": sections.get("PROJECTS", "")
            })

            # BUG FIX #3: BM25Retriever.fit() yêu cầu list[dict] với key "id" và "text"
            bm25_docs.append({"id": candidate_id, "text": raw_text})

            progress_bar.progress((i + 1) / total_files)

        # --- BM25 fit và retrieve ---
        bm25 = BM25Retriever()
        bm25.fit(bm25_docs)

        # BUG FIX #4: method là retrieve(), không phải search()
        bm25_results_list = bm25.retrieve(selected_campaign.job_description, top_k=len(cv_records))
        # Chuyển kết quả thành dict {candidate_id: score} để tra nhanh
        bm25_score_map = {r["doc_id"]: r["score"] for r in bm25_results_list}
        
        if bm25_score_map:
            max_bm25 = max(bm25_score_map.values())
            min_bm25 = min(bm25_score_map.values())
        else:
            max_bm25, min_bm25 = 1.0, 0.0

        results = []
        all_skills_in_cvs = []

        for idx, cv in enumerate(cv_records):
            bm25_raw = bm25_score_map.get(cv.candidate_id, 0.0)
            bm25_norm = (bm25_raw - min_bm25) / (max_bm25 - min_bm25) if max_bm25 > min_bm25 else 0.0

            # BUG FIX #5: score_candidate(jd_text, jd_skills, cv, embedder, bm25_score)
            row = score_candidate(
                jd_text=selected_campaign.job_description,
                jd_skills=jd_skills,
                cv=cv,
                embedder=embedder,
                bm25_raw=bm25_raw,
                bm25_norm=bm25_norm
            )

            # Thêm thông tin display (không lưu trong scorer)
            row["raw_text"] = cv_raw_data[idx]["raw_text"]
            row["projects_text"] = cv_raw_data[idx]["projects_text"]
            row["cv_skills"] = cv.skills_normalized

            results.append(row)
            all_skills_in_cvs.extend(cv.skills_normalized)

            # Lưu vào Database
            # BUG FIX #6: Dùng đúng column names trong models.py
            db_candidate = Candidate(
                campaign_id=selected_campaign.id,
                file_name=file.name if idx < len(uploaded_files) else cv.candidate_id,
                candidate_name=cv.candidate_id,
                skills=",".join(cv.skills_normalized),
                skill_overlap=row["skill_overlap"],
                semantic_score=row["semantic"],
                bm25_score=row.get("bm25_norm", 0.0),
                experience_score=row.get("years_score", 0.0),
                total_score=row["total_score"],
                analysis_json=json.dumps(row)
            )
            db.add(db_candidate)

        db.commit()

        results = sorted(results, key=lambda x: x["total_score"], reverse=True)
        st.session_state["results"] = results
        st.session_state["all_skills"] = all_skills_in_cvs
        st.session_state["jd_skills"] = jd_skills
        st.success(f"Phân tích thành công {len(cv_records)} CV")

# --- HIỂN THỊ KẾT QUẢ ---
if "results" in st.session_state:
    results = st.session_state["results"]
    all_skills = st.session_state["all_skills"]
    jd_skills = st.session_state["jd_skills"]

    tab1, tab2, tab3 = st.tabs(["Tổng quan đợt tuyển dụng", "Danh sách Ứng viên", "So sánh Chuyên sâu"])

    # ==========================================
    # TAB 1: TỔNG QUAN
    # ==========================================
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Phễu Ứng Viên")
            thresholds = {
                "Tổng CV": len(results),
                "Pass (>50%)": len([r for r in results if r["total_score"] > 0.5]),
                "Top Talent (>75%)": len([r for r in results if r["total_score"] > 0.75])
            }
            fig_funnel = go.Figure(go.Funnel(
                y=list(thresholds.keys()),
                x=list(thresholds.values()),
                textinfo="value+percent initial",
                marker={"color": ["#636EFA", "#00CC96", "#EF553B"]}
            ))
            st.plotly_chart(fig_funnel, use_container_width=True)

        with col2:
            st.subheader("Top 10 Kỹ năng xuất hiện nhiều nhất")
            if all_skills:
                skill_counts = Counter(all_skills).most_common(10)
                skills, counts = zip(*skill_counts)
                
                fig_bar = px.bar(
                    x=counts, 
                    y=skills, 
                    orientation='h',
                    labels={'x': 'Số lượng CV', 'y': 'Kỹ năng'},
                    color=counts,
                    color_continuous_scale='Viridis'
                )
                fig_bar.update_layout(
                    yaxis={'categoryorder': 'total ascending'}, 
                    xaxis=dict(range=[1, max(counts) + 0.5], dtick=1),
                    showlegend=False
                )
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.info("Không tìm thấy kỹ năng nào trong các CV.")

    # ==========================================
    # TAB 2: DANH SÁCH ỨNG VIÊN (XAI)
    # ==========================================
    with tab2:
        st.subheader("Bảng xếp hạng")
        for idx, row in enumerate(results):
            with st.expander(f"Hạng {idx+1}: {row['candidate_id']} — Tổng điểm: {row['total_score']*100:.1f}%"):
                col_a, col_b = st.columns([1, 2])

                with col_a:
                    st.metric("Kỹ năng đáp ứng", f"{row['skill_overlap']*100:.1f}%")
                    st.metric("Tỷ lệ tương đồng với JD", f"{row['semantic']*100:.1f}%")
                    st.metric("Kinh nghiệm (ước tính)", f"{row.get('years_score', 0)*5:.1f} năm")

                    # Radar Chart
                    norm_bm25 = min(row.get("bm25_raw", 0) / 30.0, 1.0)
                    categories = ['Skill Match', 'Semantic', 'BM25']
                    values = [row['skill_overlap'], row['semantic'], norm_bm25]
                    fig_radar = go.Figure()
                    fig_radar.add_trace(go.Scatterpolar(
                        r=values + [values[0]],
                        theta=categories + [categories[0]],
                        fill='toself',
                        name='Ứng viên'
                    ))
                    fig_radar.update_layout(
                        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                        showlegend=False,
                        height=250,
                        margin=dict(l=20, r=20, t=20, b=20)
                    )
                    st.plotly_chart(fig_radar, use_container_width=True)

                with col_b:
                    st.markdown("**🔍 Phân tích Kỹ năng ()**")
                    cv_skills_set = set(row.get("cv_skills", []))
                    jd_skills_set = set(jd_skills)

                    matched = cv_skills_set.intersection(jd_skills_set)
                    missing = jd_skills_set - cv_skills_set
                    bonus = cv_skills_set - jd_skills_set

                    if matched:
                        st.markdown("✅ **Đáp ứng:** " + " ".join([f"`{s}`" for s in sorted(matched)]))
                    if missing:
                        st.markdown("❌ **Thiếu:** " + " ".join([f"`{s}`" for s in sorted(missing)]))
                    if bonus:
                        st.markdown("✨ **Kỹ năng bổ sung:** " + " ".join([f"`{s}`" for s in sorted(bonus)]))

                st.markdown("---")
                st.markdown("**📄 Trích xuất nội dung CV**")
                detail_tab1, detail_tab2 = st.tabs(["Dự án", "Toàn văn bản"])
                with detail_tab1:
                    st.text_area("Nội dung Dự án", row.get("projects_text") or "Không tìm thấy nội dung dự án", height=200, key=f"proj_{idx}", disabled=True)
                with detail_tab2:
                    st.text_area("Toàn văn CV", row.get("raw_text") or "Không có nội dung", height=200, key=f"raw_{idx}", disabled=True)

    # ==========================================
    # TAB 3: SO SÁNH CHUYÊN SÂU
    # ==========================================
    with tab3:
        if len(results) > 1:
            st.subheader("So sánh tương quan: Kỹ năng vs Ngữ nghĩa")
            compare_data = results[:15]
            scatter_data = []
            for r in compare_data:
                scatter_data.append({
                    "Ứng viên": r["candidate_id"],
                    "Kỹ năng (%)": r["skill_overlap"] * 100,
                    "Tương đồng JD (%)": r["semantic"] * 100,
                    "Tổng điểm": r["total_score"] * 100
                })

            fig_scatter = px.scatter(
                scatter_data, x="Kỹ năng (%)", y="Tương đồng JD (%)",
                size="Tổng điểm", color="Ứng viên",
                hover_name="Ứng viên", size_max=40, height=450
            )
            fig_scatter.add_shape(
                type="rect", x0=50, y0=50, x1=100, y1=100,
                fillcolor="LightGreen", opacity=0.15, layer="below", line_width=0
            )
            fig_scatter.add_annotation(
                x=75, y=95, text="Vùng ứng viên tốt nhất",
                showarrow=False, font=dict(color="green", size=11)
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

            # Bảng so sánh nhanh
            st.subheader("Bảng Tổng Hợp")
            table_data = []
            for i, r in enumerate(results):
                table_data.append({
                    "Hạng": i + 1,
                    "Ứng viên": r["candidate_id"],
                    "Kỹ năng (%)": f"{r['skill_overlap']*100:.1f}",
                    "Tương đồng JD (%)": f"{r['semantic']*100:.1f}",
                    "BM25 (%)": f"{r.get('bm25_norm', 0.0)*100:.1f}",
                    "Điểm Kinh nghiệm (%)": f"{r.get('years_score', 0.0)*100:.1f}",
                    "Kinh nghiệm (ước)": f"{r.get('years_score',0)*5:.1f} năm",
                    "Tổng điểm (%)": f"{r['total_score']*100:.1f}"
                })
            import pandas as pd
            st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)
        else:
            st.info("Cần ít nhất 2 ứng viên để so sánh.")
