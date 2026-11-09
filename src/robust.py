"""Robustness checks for the main result (content > metadata).

R1 strict masking: also hide domains, file names, long ids (re-embedded text).
R2 near-duplicate removal: drop positive pairs whose texts are near copies (emb_cos > 0.95).
R3 ground-truth variants: distinctive URL = used by 2-10 agents, or 2-200 agents.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
sys.path.insert(0, str(Path(__file__).parent))
from run_eval import cv_predict, scores
from pairs import build_pairs
from features import pair_features, FIELD_OF

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_parquet(ROOT / "data" / "edits.parquet")
emb = np.load(ROOT / "data" / "emb_bge.npy")
emb_s = np.load(ROOT / "data" / "emb_bge_text_strict.npy")
day = (df.set_index("eid").time.astype("int64") // 10**9 // 86400).values
P = pd.read_parquet(ROOT / "results" / "pairs.parquet")
feats = [c for c in P.columns if c in FIELD_OF]
meta = [f for f in feats if FIELD_OF[f] != "content"]
cont = [f for f in feats if FIELD_OF[f] == "content"]
out = {}


def run(Pf, name):
    y, nt, g = Pf.y.values, Pf.neg_type.values, day[Pf.a.values]
    r = {}
    for cname, cols in [("all fields", feats), ("content only", cont), ("metadata only", meta)]:
        r[cname] = scores(y, cv_predict(Pf[cols], y, g), nt, n_boot=100)
        print(f"{name:28s} {cname:14s} hard AUROC={r[cname]['hard']['auroc']:.3f} R@1%={r[cname]['hard']['recall_at_fpr1']:.3f}", flush=True)
    out[name] = r


run(P, "baseline")
# R1 strict mask
S = P.copy()
txt = df.text_strict.str.slice(0, 4000).values
X = TfidfVectorizer(min_df=2, max_features=50000, sublinear_tf=True).fit_transform(txt)
S["tfidf_cos"] = np.asarray(X[S.a.values].multiply(X[S.b.values]).sum(1)).ravel()
S["emb_cos"] = (emb_s[S.a.values] * emb_s[S.b.values]).sum(1)
run(S, "R1 strict mask")
# R2 near-duplicate removal (applied with strict features too)
keep = ~((S.y == 1) & (S.emb_cos > 0.95))
out["R2_removed_pos_share"] = float(1 - keep[S.y == 1].mean())
run(S[keep].reset_index(drop=True), "R2 strict + no near-dups")
# R3 ground-truth variants
for lo, hi in [(2, 10), (2, 200)]:
    pv, _ = build_pairs(df, min_ag=lo, max_ag=hi, seed=3)
    Fv = pair_features(df, pv, emb)
    run(pv.join(Fv), f"R3 GT {lo}-{hi} agents")
json.dump(out, open(ROOT / "results" / "robust.json", "w"), indent=1, default=float)
