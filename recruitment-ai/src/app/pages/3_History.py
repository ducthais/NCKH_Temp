import streamlit as st
import json
import pandas as pd
from collections import Counter

import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt

from src.store.database import get_db
from src.store.models import Campaign, Candidate

st.set_page_config(page_title="Lịch Sử Chiến Dịch", layout="wide")
st.title("Lịch Sử")

db = next(get_db())

campaigns = db.query(Campaign).order_by(Campaign.created_at.desc()).all()
if not campaigns:
    st.info("Chưa có đợt tuyển dụng nào.")
    st.stop()

campaign_options = {c.id: f"{c.job_title} ({c.created_at.strftime('%Y-%m-%d')})" for c in campaigns}
selected_campaign_id = st.selectbox("Chọn đợt tuyển dụng để xem lại:", options=list(campaign_options.keys()), format_func=lambda x: campaign_options[x])

selected_campaign = db.query(Campaign).filter(Campaign.id == selected_campaign_id).first()
candidates = db.query(Candidate).filter(Candidate.campaign_id == selected_campaign_id).order_by(Candidate.total_score.desc()).all()

if not candidates:
    st.warning("Đợt tuyển dụng này chưa có ứng viên nào được phân tích.")
    st.stop()

st.success(f"Đã tải {len(candidates)} ứng viên cho chiến dịch: {selected_campaign.job_title}")

# Restore results array and skills from DB
results = []
all_skills = []

for c in candidates:
    try:
        row = json.loads(c.analysis_json)
        results.append(row)
        if c.skills:
            all_skills.extend(c.skills.split(","))
    except Exception as e:
        st.error(f"Lỗi khi đọc dữ liệu ứng viên {c.file_name}: {e}")

# ==========================================
# RE-RENDER UI
# ==========================================
tab1, tab2, tab3 = st.tabs(["Tổng quan", "Danh sách Ứng viên", "So sánh"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Phễu Ứng Viên")
        thresholds = {"Tổng CV": len(results), "Pass (>50%)": len([r for r in results if r["total_score"] > 0.5]), "Top Talent (>75%)": len([r for r in results if r["total_score"] > 0.75])}
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

with tab2:
    st.subheader("Bảng xếp hạng")
    for idx, row in enumerate(results):
        with st.expander(f"Hạng {idx+1}: {row['candidate_id']} - Điểm: {row['total_score']*100:.1f}%"):
            col_a, col_b = st.columns([1, 2])
            with col_a:
                st.metric("Kỹ năng đáp ứng", f"{row['skill_overlap']*100:.1f}%")
                st.metric("Tỷ lệ tương đồng với JD", f"{row['semantic']*100:.1f}%")
                st.metric("Kinh nghiệm (ước tính)", f"{row.get('years_score', 0)*5:.1f} năm")
                
            with col_b:
                st.markdown("**Kỹ năng trích xuất được**")
                cv_skills_set = set(row.get("cv_skills", []))
                st.markdown(", ".join([f"`{s}`" for s in cv_skills_set]))
                
            st.markdown("---")
            st.markdown("**Trích xuất nội dung CV**")
            detail_tab1, detail_tab2 = st.tabs(["Dự án (Projects)", "Toàn văn (Raw Text)"])
            with detail_tab1:
                st.text_area("Nội dung Dự án", row.get("projects_text", "Không tìm thấy"), height=200, key=f"proj_hist_{idx}", disabled=True)
            with detail_tab2:
                st.text_area("Toàn văn CV", row.get("raw_text", "Không có"), height=200, key=f"raw_hist_{idx}", disabled=True)

with tab3:
    if len(results) > 1:
        st.subheader("So sánh tương quan: Kỹ năng và độ tương đồng")
        compare_data = results[:10]
        scatter_data = []
        for r in compare_data:
            scatter_data.append({
                "Ứng viên": r["candidate_id"],
                "Kỹ năng (%)": r["skill_overlap"] * 100,
                "Độ tương đồng JD (%)": r["semantic"] * 100,
                "Tổng điểm": r["total_score"] * 100
            })
        
        fig_scatter = px.scatter(scatter_data, x="Kỹ năng (%)", y="Độ tương đồng JD (%)", 
                                 size="Tổng điểm", color="Ứng viên",
                                 hover_name="Ứng viên", size_max=40, height=400)
        st.plotly_chart(fig_scatter, use_container_width=True)
