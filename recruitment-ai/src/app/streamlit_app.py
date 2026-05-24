# src/app/streamlit_app.py
from __future__ import annotations
import json
import sys
from collections import Counter

# Patch stderr for tqdm in streamlit
if not hasattr(sys.stderr, "flush"):
    sys.stderr.flush = lambda: None

import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
import transformers
transformers.utils.logging.disable_progress_bar()

import streamlit as st
# pyrefly: ignore [missing-import]
import plotly.express as px
# pyrefly: ignore [missing-import]
import plotly.graph_objects as go
# pyrefly: ignore [missing-import]
from wordcloud import WordCloud
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
from pathlib import Path

from src.parsers.pdf_parser import parse_pdf_native
from src.parsers.ocr_tesseract import ocr_pdf
from src.nlp.sectioner import split_sections
from src.nlp.extractor import extract_entities
from src.nlp.normalizer import normalize_skill_list
from src.nlp.embedder import Embedder
from src.matching.scorer import CandidateRecord, estimate_years, score_candidate
from src.matching.bm25_retriever import BM25Retriever

st.set_page_config(page_title="AI Recruitment Dashboard", layout="wide", initial_sidebar_state="expanded")

# --- Xử lý Caching cho Model để tránh load lại nhiều lần ---
@st.cache_resource
def load_embedder():
    return Embedder()

embedder = load_embedder()

# --- GIAO DIỆN CHÍNH ---
st.title("AI Recruitment Dashboard - NCKH")
st.markdown("Hệ thống chấm điểm và phân tích CV thông minh ứng dụng PhoBERT và BM25.")

# --- SIDEBAR: CẤU HÌNH NHẬP LIỆU ---
with st.sidebar:
    st.header("1. Nhập liệu đợt tuyển dụng")
    jd_text = st.text_area("Nhập Job Description (JD)", height=200, help="Dán mô tả công việc vào đây.")
    uploaded_files = st.file_uploader("Tải lên tập CV ứng viên (PDF, JPG, PNG)", type=["pdf", "jpg", "jpeg", "png"], accept_multiple_files=True)
    process_btn = st.button("Bắt đầu Phân tích", type="primary", use_container_width=True)

if process_btn and jd_text and uploaded_files:
    with st.spinner("Đang phân tích và chấm điểm hàng loạt CV..."):
        # 1. Parse all CVs
        cvs = []
        cv_data = []  # Lưu (parsed, sections) riêng cho từng CV
        all_skills_in_cvs = []
        for f in uploaded_files:
            temp_path = Path(__file__).parent.parent.parent / "data/interim" / f.name
            temp_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path.write_bytes(f.read())

            ext = temp_path.suffix.lower()
            if ext == ".pdf":
                parsed = parse_pdf_native(temp_path)
                if parsed["needs_ocr"]:
                    parsed = ocr_pdf(temp_path)
            else:
                # File ảnh (jpg, jpeg, png) -> OCR trực tiếp
                from src.parsers.ocr_tesseract import ocr_image
                parsed = ocr_image(temp_path)

            sections = split_sections(parsed["raw_text"])
            extracted = extract_entities(parsed["raw_text"], sections)
            skills = normalize_skill_list(extracted["skills_raw"])
            all_skills_in_cvs.extend(skills)

            cv = CandidateRecord(
                candidate_id=parsed["candidate_id"],
                raw_text=parsed["raw_text"],
                skills_normalized=skills,
                years_experience_est=estimate_years(extracted["date_ranges"]),
                education_text=extracted["education_text"],
            )
            cvs.append(cv)
            cv_data.append({"parsed": parsed, "sections": sections})  # Lưu riêng từng CV

        # 2. Fit BM25
        bm25 = BM25Retriever()
        bm25.fit([{"id": cv.candidate_id, "text": cv.raw_text} for cv in cvs])
        bm25_results = {res["doc_id"]: res["score"] for res in bm25.retrieve(jd_text, top_k=len(cvs))}

        # 3. Score CVs — dùng đúng parsed/sections cho từng CV
        results = []
        jd_skills = normalize_skill_list(extract_entities(jd_text, split_sections(jd_text))["skills_raw"])
        
        for i, cv in enumerate(cvs):
            bm25_score = bm25_results.get(cv.candidate_id, 0.0)
            row = score_candidate(jd_text, jd_skills, cv, embedder, bm25_score=bm25_score)
            # Fix bug: dùng đúng parsed/sections của từng CV theo index
            this_parsed = cv_data[i]["parsed"]
            this_sections = cv_data[i]["sections"]
            row["raw_text"] = this_parsed["raw_text"]
            row["projects_text"] = this_sections.get("PROJECTS", "Không tìm thấy mục dự án trong CV này.")
            row["experience_text"] = this_sections.get("EXPERIENCE", "Không tìm thấy mục kinh nghiệm làm việc.")
            results.append(row)

        results = sorted(results, key=lambda x: x["total_score"], reverse=True)
        
        # Lưu vào session_state để không bị mất khi chuyển tab
        st.session_state["results"] = results
        st.session_state["all_skills"] = all_skills_in_cvs
        st.session_state["jd_skills"] = jd_skills
        st.success(f"✅ Phân tích thành công {len(cvs)} CV!")

# --- HIỂN THỊ KẾT QUẢ THEO TAB ---
if "results" in st.session_state:
    results = st.session_state["results"]
    all_skills = st.session_state["all_skills"]
    jd_skills = st.session_state["jd_skills"]
    
    tab1, tab2, tab3 = st.tabs(["Tổng quan đợt tuyển dụng", "Danh sách Ứng viên (XAI)", "So sánh Chuyên sâu"])
    
    # ==========================================
    # TAB 1: TỔNG QUAN
    # ==========================================
    with tab1:
        st.header("Khu vực 1: Phân tích Tổng quan")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("1.1 Phễu Sàng Lọc Ứng Viên")
            total_cvs = len(results)
            passed_hard_req = sum(1 for r in results if r["total_score"] >= 0.3)
            high_match = sum(1 for r in results if r["total_score"] >= 0.6)
            top_potentials = sum(1 for r in results if r["total_score"] >= 0.8)
            
            funnel_data = dict(
                number=[total_cvs, passed_hard_req, high_match, top_potentials],
                stage=["Tổng CV tiếp nhận", "Đạt yêu cầu tối thiểu", "Phù hợp cao (>60%)", "Tiềm năng nhất (>80%)"]
            )
            fig_funnel = px.funnel(funnel_data, x='number', y='stage')
            st.plotly_chart(fig_funnel, use_container_width=True)
            
        with col2:
            st.subheader("1.2 Phân phối Điểm số")
            scores = [r["total_score"] * 100 for r in results]
            fig_hist = px.histogram(x=scores, nbins=10, labels={'x': 'Điểm Match (%)', 'y': 'Số lượng CV'},
                                    color_discrete_sequence=['indianred'])
            st.plotly_chart(fig_hist, use_container_width=True)
            
        st.subheader("1.3 Lưới Từ khóa Kỹ năng Nổi bật (Word Cloud)")
        if all_skills:
            skill_counts = Counter(all_skills)
            wc = WordCloud(width=800, height=300, background_color='white', colormap='viridis').generate_from_frequencies(skill_counts)
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.imshow(wc, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
        else:
            st.info("Không tìm thấy đủ kỹ năng trong tập CV để tạo biểu đồ.")

    # ==========================================
    # TAB 2: DANH SÁCH ỨNG VIÊN
    # ==========================================
    with tab2:
        st.header("Khu vực 2: Danh sách Xếp hạng & X-Quang Ứng viên")
        
        # Thống kê nhanh
        highly_suitable = sum(1 for r in results if r["total_score"] >= 0.7)
        potential = sum(1 for r in results if 0.5 <= r["total_score"] < 0.7)
        not_suitable = sum(1 for r in results if r["total_score"] < 0.5)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("🟢 Rất phù hợp (>70%)", highly_suitable)
        c2.metric("🟡 Tiềm năng (50-70%)", potential)
        c3.metric("🔴 Không phù hợp (<50%)", not_suitable)
        st.divider()

        for idx, row in enumerate(results):
            score_pct = int(row["total_score"] * 100)
            badge = "🟢 RẤT PHÙ HỢP" if score_pct >= 70 else ("🟡 TIỀM NĂNG" if score_pct >= 50 else "🔴 KHÔNG PHÙ HỢP")
            
            with st.expander(f"Hạng {idx+1}: [{score_pct}%] {row['candidate_id']} - {badge}", expanded=(idx==0)):
                c_left, c_right = st.columns([1, 1])
                
                with c_left:
                    st.markdown("**1. Điểm mạnh / Kỹ năng đáp ứng:**")
                    if row["matched_skills"]:
                        st.success(", ".join(row["matched_skills"]))
                    else:
                        st.warning("Không tìm thấy kỹ năng khớp chính xác.")
                        
                    st.markdown("**2. Lỗ hổng / Kỹ năng còn thiếu:**")
                    if row["missing_skills"]:
                        st.error(", ".join(row["missing_skills"]))
                    else:
                        st.success("Đáp ứng toàn bộ kỹ năng yêu cầu!")
                        
                    st.markdown("**3. Phân tích thành phần (Progress Bars):**")
                    sem_val = max(0.0, min(row["semantic"], 1.0))
                    lex_val = max(0.0, min(row["bm25_raw"] / 30.0, 1.0))
                    exp_val = max(0.0, min(row["years_score"], 1.0))
                    
                    st.progress(sem_val, text=f"Ngữ nghĩa (Semantic): {sem_val*100:.0f}%")
                    st.progress(lex_val, text=f"Từ khóa (Lexical): {lex_val*100:.0f}%")
                    st.progress(exp_val, text=f"Kinh nghiệm: {exp_val*100:.0f}%")
                
                with c_right:
                    st.markdown("**4. Biểu đồ Radar 'Chân dung năng lực'**")
                    categories = ['Kỹ năng lõi', 'Từ khóa', 'Ngữ cảnh', 'Kinh nghiệm']
                    # Normalize scores for radar
                    skill_score = max(0.0, min(row["skill_overlap"], 1.0))
                    lex_score = lex_val
                    sem_score = sem_val
                    exp_score = exp_val
                    
                    fig_radar = go.Figure()
                    # Mức tối thiểu yêu cầu (Fake threshold for UI demonstration)
                    fig_radar.add_trace(go.Scatterpolar(
                        r=[0.5, 0.5, 0.5, 0.5],
                        theta=categories,
                        fill='toself',
                        name='Yêu cầu tối thiểu (JD)',
                        line_color='red',
                        opacity=0.3
                    ))
                    # Năng lực thực tế của ứng viên
                    fig_radar.add_trace(go.Scatterpolar(
                        r=[skill_score, lex_score, sem_score, exp_score],
                        theta=categories,
                        fill='toself',
                        name='Năng lực Ứng viên',
                        line_color='green'
                    ))
                    fig_radar.update_layout(
                        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                        showlegend=True,
                        margin=dict(l=20, r=20, t=20, b=20)
                    )
                    st.plotly_chart(fig_radar, use_container_width=True, key=f"radar_{idx}")

                st.markdown("---")
                st.markdown("**📄 Trích xuất nội dung CV**")
                detail_tab1, detail_tab2, detail_tab3 = st.tabs(["🚀 Dự án cá nhân (Projects)", "💼 Kinh nghiệm (Experience)", "📝 Toàn văn (Raw Text)"])
                
                with detail_tab1:
                    st.text_area("Nội dung Dự án", row.get("projects_text", "Không có"), height=200, key=f"proj_{idx}", disabled=True)
                with detail_tab2:
                    st.text_area("Nội dung Kinh nghiệm", row.get("experience_text", "Không có"), height=200, key=f"exp_{idx}", disabled=True)
                with detail_tab3:
                    st.text_area("Toàn văn CV", row.get("raw_text", "Không có"), height=200, key=f"raw_{idx}", disabled=True)

    # ==========================================
    # TAB 3: SO SÁNH CHUYÊN SÂU
    # ==========================================
    with tab3:
        st.header("Khu vực 3: Bàn Cân Ứng Viên")
        
        candidate_names = [r["candidate_id"] for r in results]
        selected_candidates = st.multiselect("Chọn 2-3 ứng viên để so sánh đối đầu:", candidate_names, default=candidate_names[:min(3, len(candidate_names))])
        
        if selected_candidates:
            compare_data = [r for r in results if r["candidate_id"] in selected_candidates]
            
            col_bar, col_scatter = st.columns(2)
            
            with col_bar:
                st.subheader("So sánh Tiêu chí (Grouped Bar)")
                # Chuẩn bị dữ liệu cho Grouped Bar Chart
                bar_data = []
                for r in compare_data:
                    bar_data.append({"Ứng viên": r["candidate_id"], "Tiêu chí": "Kỹ năng", "Điểm": r["skill_overlap"] * 100})
                    bar_data.append({"Ứng viên": r["candidate_id"], "Tiêu chí": "Ngữ nghĩa", "Điểm": r["semantic"] * 100})
                    bar_data.append({"Ứng viên": r["candidate_id"], "Tiêu chí": "Kinh nghiệm", "Điểm": r["years_score"] * 100})
                
                fig_bar = px.bar(bar_data, x="Tiêu chí", y="Điểm", color="Ứng viên", barmode="group", height=400)
                st.plotly_chart(fig_bar, use_container_width=True)
                
            with col_scatter:
                st.subheader("Ma trận Quyết định (Decision Matrix)")
                # Phân tán: X=Kỹ năng (Skill), Y=Ngữ nghĩa (Semantic), Size=Total
                scatter_data = []
                for r in compare_data:
                    scatter_data.append({
                        "Ứng viên": r["candidate_id"],
                        "Kỹ năng (%)": r["skill_overlap"] * 100,
                        "Ngữ nghĩa (%)": r["semantic"] * 100,
                        "Tổng điểm": r["total_score"] * 100
                    })
                
                fig_scatter = px.scatter(scatter_data, x="Kỹ năng (%)", y="Ngữ nghĩa (%)", 
                                         size="Tổng điểm", color="Ứng viên",
                                         hover_name="Ứng viên", size_max=40, height=400)
                # Vẽ vùng "Lý tưởng" góc trên phải (x>50, y>50)
                fig_scatter.add_shape(type="rect", x0=50, y0=50, x1=100, y1=100, 
                                      fillcolor="LightGreen", opacity=0.2, layer="below", line_width=0)
                st.plotly_chart(fig_scatter, use_container_width=True)
