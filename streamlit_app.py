"""
Streamlit dashboard – Multi‑View Contrastive + Fusion Engine
Professional layout with cards, similarity explanation, and top ETF rankings.
"""
import streamlit as st
import pandas as pd
import json
import plotly.express as px
from huggingface_hub import HfFileSystem
import config
from us_calendar import next_trading_day

# Page configuration
st.set_page_config(
    page_title="Contrastive Multi‑View Engine",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        margin-bottom: 2rem;
    }
    .universe-title {
        font-size: 1.5rem;
        font-weight: 600;
        margin-top: 1rem;
        margin-bottom: 1rem;
        padding-left: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .etf-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 15px;
        padding: 1rem;
        margin: 0.5rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
        transition: transform 0.2s;
    }
    .etf-card:hover {
        transform: translateY(-5px);
    }
    .etf-ticker {
        font-size: 1.3rem;
        font-weight: bold;
    }
    .etf-score {
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }
    .positive {
        color: #00cc96;
    }
    .negative {
        color: #ef553b;
    }
    .info-box {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header">🎯 Multi‑View Contrastive + Fusion Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Learns regime‑invariant embeddings from returns, macro, graph, and shape. Ranks ETFs by similarity to historical high‑return patterns.</div>', unsafe_allow_html=True)

# Sidebar – clean and compact
st.sidebar.markdown("## 🧠 Contrastive Engine")
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Run Date:** `{st.session_state.get('run_date', 'Not loaded')}`")
st.sidebar.markdown(f"**Next Trading Day:** `{next_trading_day()}`")
st.sidebar.markdown("**Method:** Barlow Twins (returns+macro, graph, shape)")
st.sidebar.markdown("**View fusion:** Concatenation of three embeddings")
st.sidebar.markdown("---")
st.sidebar.caption("Data: [P2SAMAPA/fi-etf-macro-signal-master-data](https://huggingface.co/datasets/P2SAMAPA/fi-etf-macro-signal-master-data)")

# Explanation of similarity score
with st.expander("📖 What does the similarity score mean?"):
    st.markdown("""
    <div class="info-box">
    The <strong>similarity score</strong> measures how closely an ETF's current multi‑view embedding 
    (derived from recent returns, macro conditions, graph connections, and statistical shapes) 
    resembles the <strong>target embedding</strong>.  
    The target embedding is the average embedding of ETFs that performed best in the recent past 
    (top 20% by forward return).  
    <br><br>
    <strong>Interpretation:</strong><br>
    - <span style="color:#00cc96">High similarity (>0.5)</span> → The ETF is currently in a state similar to historically high‑return regimes → <strong>Strong buy signal</strong>.<br>
    - <span style="color:#ef553b">Low similarity (<0.3)</span> → The ETF does not resemble high‑return patterns → <strong>Weak or sell signal</strong>.<br>
    - Similarity ranges from -1 to 1, but for ETF embeddings it typically falls between 0 and 1.<br>
    <br>
    The engine is retrained daily on the last 252 days, adapting to changing market regimes.
    </div>
    """, unsafe_allow_html=True)

# Load data
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
    st.error("No contrastive results found. Run `trainer.py` first.")
    st.stop()

data = load_json(latest)
if "error" in data:
    st.error(f"Error loading JSON: {data['error']}")
    st.stop()

# Store run date in session state for sidebar
st.session_state['run_date'] = data['run_date']

universes = data["universes"]
if not universes:
    st.warning("No universe data available.")
    st.stop()

# Main content
st.header("📈 Top ETFs to Trade Tomorrow")
st.markdown("*Ranked by cosine similarity to the target (high‑return) embedding.*")

# Display each universe as a separate card deck
for universe_name, uni_data in universes.items():
    st.markdown(f'<div class="universe-title">{universe_name.replace("_", " ").title()}</div>', unsafe_allow_html=True)
    
    top_etfs = uni_data.get("top_etfs", [])
    if not top_etfs:
        st.info(f"No predictions for {universe_name}")
        continue
    
    # Create 3 columns for top 3 ETFs
    cols = st.columns(3)
    for idx, etf in enumerate(top_etfs):
        with cols[idx]:
            ticker = etf["ticker"]
            similarity = etf["similarity"]
            # Color based on similarity threshold (green >0.5, orange 0.3-0.5, red <0.3)
            if similarity >= 0.5:
                delta_color = "normal"
                delta_text = "Strong"
            elif similarity >= 0.3:
                delta_color = "off"
                delta_text = "Moderate"
            else:
                delta_color = "inverse"
                delta_text = "Weak"
            st.markdown(f"""
            <div class="etf-card">
                <div class="etf-ticker">{ticker}</div>
                <div class="etf-score">similarity = {similarity:.3f}</div>
            </div>
            """, unsafe_allow_html=True)
            st.caption(f"Signal strength: {delta_text}")
    st.divider()

# Optional historical chart (if multiple JSON files)
st.header("📉 Similarity Trend (Last 30 days)")
with st.spinner("Loading historical data..."):
    json_files = [f for f in files if f.endswith('.json') and 'contrastive_' in f]
    json_files.sort(reverse=True)
    history_data = []
    for fname in json_files[:30]:
        try:
            fs = HfFileSystem(token=HF_TOKEN)
            with fs.open(f"datasets/{OUTPUT_REPO}/{fname}", "r") as f:
                hist = json.load(f)
                run_date = hist['run_date']
                for uni, val in hist['universes'].items():
                    if 'top_etfs' in val and val['top_etfs']:
                        # Take the top ETF's similarity as a proxy
                        top_sim = val['top_etfs'][0]['similarity']
                        history_data.append({
                            "date": run_date,
                            "universe": uni,
                            "top_similarity": top_sim
                        })
        except:
            pass
    if history_data:
        df_hist = pd.DataFrame(history_data)
        fig = px.line(df_hist, x="date", y="top_similarity", color="universe",
                      title="Top ETF Similarity Over Time",
                      labels={"top_similarity": "Top ETF Similarity", "date": "Run Date"})
        fig.update_layout(height=400, legend_title="Universe")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough historical data to plot trend.")

st.caption("Higher similarity → stronger alignment with past high‑return regimes. The engine is retrained daily on the last 252 days.")
