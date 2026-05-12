"""
Configuration for Multi‑View Contrastive + Fusion Engine (full per‑ETF version).
"""
import os

HF_TOKEN = os.environ.get("HF_TOKEN", "")
DATA_REPO = "P2SAMAPA/fi-etf-macro-signal-master-data"
OUTPUT_REPO = "P2SAMAPA/p2-etf-contrastive-results"

UNIVERSES = {
    "FI_COMMODITIES": ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"],
    "EQUITY_SECTORS": [
        "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY",
        "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "XBI", "IWM", "IWD", "IWO"
    ],
    "COMBINED": [
        "TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV",
        "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY",
        "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "XBI", "IWM", "IWD", "IWO"
    ]
}

# Real macro column names in the dataset
MACRO_COLUMNS = ["VIX", "DXY", "T10Y2Y", "TBILL_3M", "IG_SPREAD", "HY_SPREAD"]

# Rolling window for training (days)
ROLLING_WINDOW = 252
# Number of recent days to use for each ETF's feature vector
LOOKBACK_DAYS = 20

# Barlow Twins hyperparameters
BT_LAMBDA = 0.005
BT_EPOCHS = 100
BT_BATCH_SIZE = 64
BT_LEARNING_RATE = 1e-3
EMBEDDING_DIM = 64

# Target embedding: average of top 20% performing ETFs (by next‑day return)
TOP_TARGET_PERCENTILE = 80

# Ranking
TOP_N = 3
