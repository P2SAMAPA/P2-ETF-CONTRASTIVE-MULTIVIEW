# Multi‑View Contrastive + Fusion Engine

**Barlow Twins** on three views (returns+macro, graph, shape) to learn ETF embeddings.  
Ranks ETFs by cosine similarity to target embedding (average of top‑performing ETFs).

- Daily refit on rolling 252‑day window
- Outputs top 3 ETFs per universe
- Uploads results to Hugging Face
