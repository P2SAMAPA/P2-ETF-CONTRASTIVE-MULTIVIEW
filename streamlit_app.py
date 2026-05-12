"""
Streamlit dashboard: Multi‑View Contrastive Engine
"""
import streamlit as st
import pandas as pd
import json
from huggingface_hub import HfFileSystem
import config
from us_calendar import next_trading_day

st.set_page_config(page_title="Contrastive Multi‑View Engine", layout="wide")
st.title("🎯 Multi‑View Contrastive + Fusion Engine")
st.caption("Learns regime‑invariant embeddings from returns, macro, graph, and shape. Ranks ETFs by similarity to historical high‑return embeddings.")

OUTPUT_REPO = config.OUTPUT_REPO
HF_TOKEN = config.HF_TOKEN

@st.cache_data(ttl=3600)
def list_repo_files():
    fs = HfFileSystem(token=HF_TOKEN)
    try:
        files = [f['name'] for f in fs.ls(f"datasets/{OUTPUT_REPO}", detail=True, recursive=True) if f['type'] == 'file']
        return files
    except Exception as e:
        return [f"Error: {e}"]

def find_latest_json(files):
    json_files = [f for f in files if f.endswith('.json') and 'contrastive_' in f]
    if not json_files:
        return None
    json_files.sort(reverse=True)
    return json_files[0]

@st.cache_data(ttl=3600)
def load_json(path):
    fs = HfFileSystem(token=HF_TOKEN)
    try:
        with fs.open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        return {"error": str(e)}

files = list_repo_files()
latest = find_latest_json(files)
if not latest:
    st.error("No results found. Run trainer first.")
    st.stop()

data = load_json(latest)
if "error" in data:
    st.error(f"Error: {data['error']}")
    st.stop()

st.sidebar.header("ℹ️ Info")
st.sidebar.write(f"**Run date:** {data['run_date']}")
st.sidebar.write(f"**Next trading day:** {next_trading_day()}")
st.sidebar.write("**Method:** Barlow Twins (returns+macro, graph, shape)")

universes = data["universes"]
for universe_name, uni_data in universes.items():
    st.subheader(f"🌍 {universe_name}")
    top_etfs = uni_data.get("top_etfs", [])
    if not top_etfs:
        st.info("No predictions")
        continue
    cols = st.columns(3)
    for i, etf in enumerate(top_etfs):
        with cols[i]:
            st.metric(f"#{i+1} {etf['ticker']}", f"similarity = {etf['similarity']:.3f}")
    st.divider()

st.caption("Higher similarity to the target embedding (average of top‑performing ETFs) indicates stronger buy signal.")
