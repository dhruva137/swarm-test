"""Monitor scaling: Qwen3 8B vs 14B-AWQ vs 32B-AWQ on the same 1,500 pairs, vs the cheap monitor."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from run_eval import scores

ROOT = Path(__file__).resolve().parents[1]
S = pd.read_parquet(ROOT / "results" / "stack_pairs.parquet")
L8 = pd.read_json(ROOT / "results" / "kaggle_llm" / "llm_scores.jsonl", lines=True).assign(model="Qwen/Qwen3-8B")
LS = pd.read_json(ROOT / "results" / "kaggle_scale" / "llm_scale_scores.jsonl", lines=True)
ids = set(LS.pair_idx)
L = pd.concat([L8[L8.pair_idx.isin(ids)], LS])
out = {"n_pairs": len(ids)}
sub = S[S.pair_idx.isin(ids)]
for col in ["cheap", "emb_cos"]:
    out[col] = scores(sub.y.values, sub[col].values, sub.neg_type.values, n_boot=500)
    print(f"{col:30s} hard AUROC={out[col]['hard']['auroc']:.3f} CI={np.round(out[col]['hard']['ci'],3)} R@1%={out[col]['hard']['recall_at_fpr1']:.3f}")
for (m, meta), g in L.groupby(["model", "meta"]):
    k = f"{m.split('/')[-1]} | {'content+meta' if meta else 'content'}"
    s = scores(g.y.values, g.score.values, g.neg_type.values, n_boot=500)
    s["mean_score_neg"] = float(g[g.y == 0].score.mean()); s["mean_score_pos"] = float(g[g.y == 1].score.mean())
    out[k] = s
    print(f"{k:30s} hard AUROC={s['hard']['auroc']:.3f} CI={np.round(s['hard']['ci'],3)} R@1%={s['hard']['recall_at_fpr1']:.3f} "
          f"meanP(yes) pos={s['mean_score_pos']:.2f} neg={s['mean_score_neg']:.2f}")
json.dump(out, open(ROOT / "results" / "scale.json", "w"), indent=1, default=float)
