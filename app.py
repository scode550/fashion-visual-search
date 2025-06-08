# app.py

import os
import warnings
import streamlit as st
from PIL import Image
from search_engine import (
    get_similar_dresses,
    get_personalized_recommendations,
    get_trending_recommendations,
)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings("ignore", category=UserWarning)

st.set_page_config(page_title="Fashion Visual Search", layout="wide")
st.title(" Fashion Visual Similarity Search")

if "search_history" not in st.session_state:
    st.session_state.search_history = []

uploaded_file = st.file_uploader("Upload a dress image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=300)

    with st.spinner("Finding similar dresses..."):
        results, matched_pid = get_similar_dresses(image, return_top_match=True)
        st.session_state.search_history.append(matched_pid)

    st.markdown("### Similar Dresses")
    cols = st.columns(5)
    for col, item in zip(cols, results):
        col.image(item["image_path"], use_container_width=True)
        col.caption(item["product_name"])

    st.markdown("### Based on Your Search History")
    personalized = get_personalized_recommendations(st.session_state.search_history)
    cols = st.columns(5)
    for col, item in zip(cols, personalized):
        col.image(item["image_path"], use_container_width=True)
        col.caption(item["product_name"])

    st.markdown("### 🔥 Trending  Dresses")
    trending = get_trending_recommendations(matched_pid)
    cols = st.columns(5)
    for col, item in zip(cols, trending):
        col.image(item["image_path"], use_container_width=True)
        col.caption(item["product_name"])
