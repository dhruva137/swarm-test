"""Embed masked edit text on the local GPU (RTX 5050, 8GB)."""
import sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
col = sys.argv[1] if len(sys.argv) > 1 else "text_masked"
model_name = "BAAI/bge-base-en-v1.5"
df = pd.read_parquet(ROOT / "data" / "edits.parquet")
m = SentenceTransformer(model_name, device="cuda")
m.half()
m.max_seq_length = 512
emb = m.encode(df[col].str.slice(0, 3000).tolist(), batch_size=64, normalize_embeddings=True,
               show_progress_bar=False, convert_to_numpy=True)
np.save(ROOT / "data" / ("emb_bge.npy" if col == "text_masked" else f"emb_bge_{col}.npy"), emb.astype(np.float32))
print(emb.shape, torch.cuda.max_memory_allocated() / 1e9, "GB")
