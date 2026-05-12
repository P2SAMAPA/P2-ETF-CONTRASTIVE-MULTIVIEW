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
    """
    Returns DataFrame with ETF log returns + macro levels.
    Index is datetime. Macro columns are those in config.MACRO_COLUMNS that exist.
    """
    # ETF log returns
    returns = pd.DataFrame(index=df.index)
    for ticker in universe_tickers:
        if ticker in df.columns:
            price = df[ticker]
            if not price.isna().all():
                returns[ticker] = np.log(price / price.shift(1))
    # Macro columns that actually exist
    avail_macro = [c for c in config.MACRO_COLUMNS if c in df.columns]
    macro = df[avail_macro].copy() if avail_macro else pd.DataFrame(index=df.index)
    combined = pd.concat([returns, macro], axis=1).dropna()
    return combined

def compute_shape_features(returns_df, window=20):
    """
    Compute rolling statistical shapes for each ETF.
    Returns DataFrame with same index as returns_df (initial rows will be NaN).
    Columns: ETF_mean, ETF_std, ETF_skew, ETF_kurt, ETF_q25, ETF_q75.
    """
    etfs = [c for c in returns_df.columns if c not in config.MACRO_COLUMNS]
    shape_list = []
    for col in etfs:
        roll = returns_df[col].rolling(window, min_periods=window)
        mean = roll.mean()
        std = roll.std()
        skew = roll.skew()
        kurt = roll.kurt()
        q25 = roll.quantile(0.25)
        q75 = roll.quantile(0.75)
        # Give names to avoid conflicts across ETFs
        shape = pd.DataFrame({
            f"{col}_mean": mean,
            f"{col}_std": std,
            f"{col}_skew": skew,
            f"{col}_kurt": kurt,
            f"{col}_q25": q25,
            f"{col}_q75": q75
        })
        shape_list.append(shape)
    shapes = pd.concat(shape_list, axis=1)
    return shapes

def compute_graph_features(returns_df, lookback=20, top_k=5):
    """
    For each day and each ETF, compute its average correlation with top k correlated ETFs.
    Returns DataFrame with same index as returns_df (initial rows NaN).
    Columns: ETF_corr1, ETF_corr2, ..., ETF_corrk.
    """
    etfs = [c for c in returns_df.columns if c not in config.MACRO_COLUMNS]
    n = len(returns_df)
    features = []
    dates = returns_df.index
    for i in range(n):
        if i < lookback:
            # Not enough data, fill with NaN
            row = [np.nan] * (len(etfs) * top_k)
            features.append(row)
            continue
        window = returns_df.iloc[i-lookback:i]
        corr = window[etfs].corr().values
        row = []
        for j, etf in enumerate(etfs):
            corr_row = corr[j]
            # get indices of top k correlations (excluding self)
            sorted_indices = np.argsort(corr_row)[::-1]
            top_indices = [idx for idx in sorted_indices if idx != j][:top_k]
            top_vals = corr_row[top_indices]
            if len(top_vals) < top_k:
                top_vals = np.pad(top_vals, (0, top_k - len(top_vals)), constant_values=np.nan)
            row.extend(top_vals)
        features.append(row)
    cols = [f"{etf}_corr_{k+1}" for etf in etfs for k in range(top_k)]
    graph_df = pd.DataFrame(features, index=dates, columns=cols)
    return graph_df
