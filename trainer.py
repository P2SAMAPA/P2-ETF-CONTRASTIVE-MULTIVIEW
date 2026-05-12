"""
Daily training: build per‑ETF feature vectors (returns+macro, graph, shape), 
train Barlow Twins, compute embeddings, rank by cosine similarity to target.
"""
import pandas as pd
import numpy as np
import torch
from pathlib import Path
import json
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
import config
import data_manager
from contrastive_models import train_bt_model

def build_etf_feature_matrix(returns_etf, macro_df, shape_features, graph_features, lookback=20):
    """
    For each ETF, concatenate:
    - recent returns (lookback days)
    - macro levels at the last day
    - recent shape features (last `lookback` days) -> average or flatten? We'll take the vector of shape features at the last day.
    """
    etf_names = returns_etf.columns.tolist()
    feature_list = []
    for etf in etf_names:
        # recent returns (last `lookback` days)
        ret_series = returns_etf[etf].iloc[-lookback:].values
        # macro at last day
        macro_last = macro_df.iloc[-1].values
        # shape features for this ETF at last day
        shape_cols = [c for c in shape_features.columns if c.startswith(etf)]
        if shape_cols:
            shape_last = shape_features[shape_cols].iloc[-1].values
        else:
            shape_last = np.array([])
        # graph features for this ETF at last day
        graph_cols = [c for c in graph_features.columns if c.startswith(etf)]
        if graph_cols:
            graph_last = graph_features[graph_cols].iloc[-1].values
        else:
            graph_last = np.array([])
        # concatenate all
        feat = np.concatenate([ret_series, macro_last, shape_last, graph_last])
        feature_list.append(feat)
    return np.array(feature_list), etf_names

def get_target_embedding(etf_embeddings, etf_names, next_day_returns, percentile=80):
    """
    Find top percentile ETFs by next‑day return, average their embeddings.
    """
    threshold = np.percentile(next_day_returns, percentile)
    top_mask = next_day_returns >= threshold
    if np.sum(top_mask) == 0:
        return np.mean(etf_embeddings, axis=0)  # fallback
    top_embeddings = etf_embeddings[top_mask]
    return np.mean(top_embeddings, axis=0)

def main():
    if not config.HF_TOKEN:
        print("HF_TOKEN not set")
        return

    df = data_manager.load_master_data()
    all_results = {}
    today = datetime.now().strftime("%Y-%m-%d")

    for universe_name, tickers in config.UNIVERSES.items():
        print(f"\n=== Universe: {universe_name} ===")
        # 1. Get ETF+macro matrix
        full_data = data_manager.prepare_etf_macro_matrix(df, tickers)
        if len(full_data) < config.ROLLING_WINDOW + config.LOOKBACK_DAYS + 10:
            print("  Insufficient data")
            all_results[universe_name] = {"top_etfs": []}
            continue

        # Take last ROLLING_WINDOW days
        train_df = full_data.iloc[-config.ROLLING_WINDOW:]
        
        # Separate ETF returns and macro
        etf_cols = [c for c in train_df.columns if c not in config.MACRO_COLUMNS]
        returns_etf = train_df[etf_cols]
        macro_df = train_df[config.MACRO_COLUMNS]
        
        # 2. Compute shape features (on the whole training window)
        shape_features = data_manager.compute_shape_features(returns_etf, window=config.LOOKBACK_DAYS)
        # Align index
        shape_features = shape_features.loc[train_df.index]
        
        # 3. Compute graph features (on the whole training window)
        graph_features = data_manager.compute_graph_features(returns_etf, lookback=config.LOOKBACK_DAYS, top_k=5)
        # Align index (graph_features already aligned)
        
        # 4. Build per‑ETF feature matrix for the last day (using only the last `LOOKBACK_DAYS` of returns, but latest shapes/graph)
        X_etf, etf_names = build_etf_feature_matrix(returns_etf, macro_df, shape_features, graph_features, lookback=config.LOOKBACK_DAYS)
        print(f"  Feature dimension per ETF: {X_etf.shape[1]}")
        
        # 5. Train Barlow Twins on these ETF feature vectors (across all ETFs in this universe)
        model, scaler = train_bt_model(
            X_etf, X_etf.shape[1], 
            hidden_dims=[128, 128],  # we can make configurable
            proj_dim=config.EMBEDDING_DIM,
            epochs=config.BT_EPOCHS,
            batch_size=config.BT_BATCH_SIZE,
            lr=config.BT_LEARNING_RATE,
            lambda_param=config.BT_LAMBDA
        )
        
        # 6. Get embeddings for all ETFs (the same X_etf)
        X_scaled = scaler.transform(X_etf)
        with torch.no_grad():
            embeddings = model.get_embedding(torch.tensor(X_scaled, dtype=torch.float32))
        
        # 7. Compute target embedding: average of top-performing ETFs in the training window.
        # We need next‑day returns for each ETF at the end of the training window.
        # Use the last day's next‑day return (i.e., day after last day) – actually we must compute within the training window.
        # Simpler: For each ETF, compute its average daily return over the last 20 days of training (as a proxy for performance).
        # Or use the next‑day return after the last day? But we only have up to last day.
        # We'll use the last day's return? Not robust. Instead, use the mean return over the last 20 days.
        recent_returns = returns_etf.iloc[-config.LOOKBACK_DAYS:].mean().values
        target_emb = get_target_embedding(embeddings, etf_names, recent_returns, percentile=config.TOP_TARGET_PERCENTILE)
        
        # 8. Rank ETFs by cosine similarity to target embedding
        sims = cosine_similarity(embeddings, target_emb.reshape(1, -1)).flatten()
        sorted_idx = np.argsort(sims)[::-1]
        top_etfs = [{"ticker": etf_names[i], "similarity": float(sims[i])} for i in sorted_idx[:config.TOP_N]]
        
        print(f"  Top 3 ETFs: {[e['ticker'] for e in top_etfs]}")
        all_results[universe_name] = {
            "top_etfs": top_etfs,
            "run_date": today
        }
    
    # Save results
    Path("results").mkdir(exist_ok=True)
    local_path = Path(f"results/contrastive_{today}.json")
    with open(local_path, "w") as f:
        json.dump({"run_date": today, "universes": all_results}, f, indent=2)
    
    import push_results
    push_results.push_daily_result(local_path)
    print("\n=== Multi‑View Contrastive Engine complete (full per‑ETF) ===")

if __name__ == "__main__":
    main()
