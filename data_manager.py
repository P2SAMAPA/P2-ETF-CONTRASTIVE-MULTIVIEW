"""
Data loading and feature engineering for per‑ETF contrastive learning.
"""
import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download
import config

def load_master_data():
    path = hf_hub_download(repo_id=config.DATA_REPO, filename="master_data.parquet", repo_type="dataset", token=config.HF_TOKEN)
    df = pd.read_parquet(path)
    if df.index.name != 'date':
        df.index.name = 'date'
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
    return df

def prepare_etf_macro_matrix(df, universe_tickers):
    returns = pd.DataFrame(index=df.index)
    for ticker in universe_tickers:
        if ticker in df.columns:
            price = df[ticker]
            if not price.isna().all():
                returns[ticker] = np.log(price / price.shift(1))
    macro = df[config.MACRO_COLUMNS].copy() if all(c in df.columns for c in config.MACRO_COLUMNS) else pd.DataFrame()
    combined = pd.concat([returns, macro], axis=1).dropna()
    return combined

def compute_shape_features(returns_df, window=20):
    """Rolling statistical shapes for each ETF."""
    shape_list = []
    for col in returns_df.columns:
        if col in config.MACRO_COLUMNS:
            continue
        roll = returns_df[col].rolling(window)
        mean = roll.mean()
        std = roll.std()
        skew = roll.skew()
        kurt = roll.kurt()
        q25 = roll.quantile(0.25)
        q75 = roll.quantile(0.75)
        shape = pd.DataFrame({
            f"{col}_mean": mean,
            f"{col}_std": std,
            f"{col}_skew": skew,
            f"{col}_kurt": kurt,
            f"{col}_q25": q25,
            f"{col}_q75": q75
        })
        shape_list.append(shape)
    shapes = pd.concat(shape_list, axis=1).dropna()
    return shapes

def compute_graph_features(returns_df, lookback=20, top_k=5):
    """
    For each day and each ETF, compute its average correlation with top k correlated ETFs.
    Returns DataFrame with columns: ticker_corr1, ticker_corr2, ..., ticker_corrk.
    """
    etfs = [c for c in returns_df.columns if c not in config.MACRO_COLUMNS]
    n = len(returns_df)
    graph_features = []
    for i in range(lookback, n):
        window = returns_df.iloc[i-lookback:i]
        corr = window.corr().values
        # For each ETF, get top k correlations (excluding self)
        row = []
        for j, etf in enumerate(etfs):
            corr_row = corr[j]
            # sort descending, skip self (index j)
            top_indices = np.argsort(corr_row)[::-1][1:top_k+1] if top_k<len(etfs)-1 else np.argsort(corr_row)[::-1][1:]
            top_vals = corr_row[top_indices]
            row.extend(top_vals)
        graph_features.append(row)
    return pd.DataFrame(graph_features, index=returns_df.index[lookback:], 
                        columns=[f"{etf}_corr_{k+1}" for etf in etfs for k in range(top_k)])
