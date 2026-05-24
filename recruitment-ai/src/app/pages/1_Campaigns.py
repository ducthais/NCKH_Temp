import streamlit as st
import pandas as pd
from src.store.database import get_db
from src.store.models import Campaign

st.set_page_config(page_title="Quản Lý Chiến Dịch", page_icon="💼", layout="wide")

st.title("💼 Quản Lý Chiến Dịch Tuyển Dụng")

db = next(get_db())

# Create a new campaign
with st.expander("➕ Tạo Chiến Dịch Mới", expanded=False):
    with st.form("new_campaign_form"):
        job_title = st.text_input("Tên vị trí tuyển dụng (VD: Data Engineer, Backend Developer)")
        job_desc = st.text_area("Mô tả công việc (JD)", height=200)
        submitted = st.form_submit_button("Tạo Chiến Dịch", type="primary")

        if submitted:
            if not job_title or not job_desc:
                st.error("Vui lòng điền đầy đủ Tên vị trí và Mô tả công việc!")
            else:
                new_campaign = Campaign(job_title=job_title, job_description=job_desc)
                db.add(new_campaign)
                db.commit()
                st.success(f"Đã tạo chiến dịch: {job_title}")
                st.rerun()

st.markdown("---")
st.subheader("📋 Danh sách Chiến dịch hiện có")

campaigns = db.query(Campaign).order_by(Campaign.created_at.desc()).all()

if not campaigns:
    st.info("Chưa có chiến dịch nào được tạo.")
else:
    data = []
    for c in campaigns:
        data.append({
            "ID": c.id,
            "Vị trí": c.job_title,
            "Ngày tạo": c.created_at.strftime("%Y-%m-%d %H:%M"),
            "Số ứng viên đã nộp": len(c.candidates)
        })
    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True, hide_index=True)
