"""Score the Qwen3-8B linkage monitor (Kaggle 2xT4) and compare with the cheap model on its sample."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from run_eval import scores

ROOT = Path(__file__).resolve().parents[1]
L = pd.read_json(ROOT / "results" / "kaggle_llm" / "llm_scores.jsonl", lines=True)
out = {"n": int(L.groupby("meta").size().iloc[0]), "yes_no_mass_median": float(L.mass.median())}
for meta, g in L.groupby("meta"):
    name = "Qwen3-8B, content + metadata" if meta else "Qwen3-8B, content only"
    s = scores(g.y.values, g.score.values, g.neg_type.values, n_boot=500)
    out[name] = s
    print(f"{name:32s} all={s['all']['auroc']:.3f} hard={s['hard']['auroc']:.3f} CI={np.round(s['hard']['ci'],3)} R@1%={s['hard']['recall_at_fpr1']:.3f}")
json.dump(out, open(ROOT / "results" / "llm_eval.json", "w"), indent=1, default=float)
