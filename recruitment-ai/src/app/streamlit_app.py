# src/app/streamlit_app.py
from __future__ import annotations
import sys
import os

# Patch stderr for tqdm in streamlit
if not hasattr(sys.stderr, "flush"):
    sys.stderr.flush = lambda: None

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
import transformers
transformers.utils.logging.disable_progress_bar()

import streamlit as st

st.set_page_config(page_title="AI Recruitment System", page_icon="🎯", layout="wide")

# Define pages explicitly using st.navigation
home_page = st.Page("pages/0_Home.py", title="Trang chủ", default=True)
campaign_page = st.Page("pages/1_Campaigns.py", title="Đợt tuyển dụng")
analysis_page = st.Page("pages/2_Analysis.py", title="Phân tích CV")
history_page = st.Page("pages/3_History.py", title="Lịch sử")

pg = st.navigation([home_page, campaign_page, analysis_page, history_page])
pg.run()

