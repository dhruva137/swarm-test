"""A5: does real swarm text appended to post B steer the LLM monitor?

For each model and injection condition: mean change in verbal log-odds vs the neutral-sentence control
(separately for positives and negatives), share of negatives called coordinated (score > 0), AUROC and recall at
1% FPR. CIs: agent-cluster bootstrap of the paired mean shift.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from run_eval_v2 import evaluate

ROOT = Path(__file__).resolve().parents[1]
S = pd.read_parquet(ROOT / "results" / "v2_sample.parquet").sort_values("sid").reset_index(drop=True)
df = pd.read_parquet(ROOT / "data" / "edits.parquet", columns=["label", "ip16"])
S["agent_a"] = np.where(df.label.values[S.a.values] != "", df.label.values[S.a.values], "anon:" + df.ip16.fillna("?").values[S.a.values])
R = pd.read_json(ROOT / "results" / "k_swarm-recruit" / "recruit_scores.jsonl", lines=True)
rng = np.random.default_rng(0)
cl = S.groupby("agent_a").apply(lambda g: g.index.values).values
out = {}
for m, g in R.groupby("model"):
    W = g.pivot(index="sid", columns="cond", values="score").loc[S.sid]
    assert W.notna().all().all() and len(W) == len(S)
    for c in ["none", "neutral", "benign", "coord"]:
        r = evaluate(S, W[c].values, n_boot=300)
        r["yes_rate_neg"] = float((W[c].values[S.y.values == 0] > 0).mean())
        r["yes_rate_pos"] = float((W[c].values[S.y.values == 1] > 0).mean())
        out[f"{m}|{c}"] = r
        print(f"{m.split('/')[-1]:10s} {c:8s} hard AUROC={r['auroc_hard']:.3f} TPR@1%={r['tpr1_hard']:.3f} "
              f"P(say coordinated): pos={r['yes_rate_pos']:.3f} neg={r['yes_rate_neg']:.3f}")
    for c in ["benign", "coord"]:
        for lab, mask in [("pos", S.y.values == 1), ("neg", S.y.values == 0)]:
            d = (W[c].values - W["neutral"].values)
            def stat(ii):
                ii = ii[mask[ii]]; return d[ii].mean()
            d0 = stat(np.arange(len(S)))
            bs = [stat(np.concatenate([cl[j] for j in rng.integers(0, len(cl), len(cl))])) for _ in range(1000)]
            out[f"{m}|shift {c} vs neutral|{lab}"] = dict(mean=float(d0), ci=[float(np.quantile(bs, .025)), float(np.quantile(bs, .975))])
            print(f"   shift in log-odds, {c} vs neutral, {lab}: {d0:+.3f} CI [{np.quantile(bs,.025):+.3f}, {np.quantile(bs,.975):+.3f}]")
json.dump(out, open(ROOT / "results" / "recruit.json", "w"), indent=1, default=float)
