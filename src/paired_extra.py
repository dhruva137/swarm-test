"""Extra paired comparisons on the v2 sample: probe vs supervised cheap; primed vs neutral verbal."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
ROOT = Path(__file__).resolve().parents[1]
S = pd.read_parquet(ROOT / "results" / "llm_v2_scores.parquet")
hard = S.neg_type.values != "easy"
cl = S.groupby("agent_a").apply(lambda g: g.index.values).values
def macro(c1, c2, ii):
    v = []
    for k in range(5):
        jj = ii[(S.fold.values[ii] == k) & hard[ii]]; yy = S.y.values[jj]
        if yy.min() == yy.max(): continue
        v.append(roc_auc_score(yy, S[c1].values[jj]) - roc_auc_score(yy, S[c2].values[jj]))
    return np.mean(v)
def paired(c1, c2, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    d0 = macro(c1, c2, np.arange(len(S)))
    bs = [macro(c1, c2, np.concatenate([cl[j] for j in rng.integers(0, len(cl), len(cl))])) for _ in range(n)]
    return dict(diff=float(d0), ci=[float(np.quantile(bs, .025)), float(np.quantile(bs, .975))])
out = {}
models = sorted({c.split("|")[1].split("_")[0] for c in S.columns if c.startswith("llm|")})
for m in models:
    for c in ["primed_meta", "primed_content", "neutral_meta"]:
        k = f"probe|{m}_{c}"
        if k in S: out[f"{k} - cheap_all"] = paired(k, "cheap_all")
    if f"llm|{m}_neutral_meta" in S and f"llm|{m}_primed_meta" in S:
        out[f"llm|{m}: neutral_meta - primed_meta"] = paired(f"llm|{m}_neutral_meta", f"llm|{m}_primed_meta")
        out[f"probe|{m}: neutral_meta - primed_meta"] = paired(f"probe|{m}_neutral_meta", f"probe|{m}_primed_meta")
for k, v in out.items():
    print(f"{k:55s} {v['diff']:+.3f} CI [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}]")
json.dump(out, open(ROOT / "results" / "paired_extra.json", "w"), indent=1)
