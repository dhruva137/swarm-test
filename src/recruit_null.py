"""Specificity check for recruitment: is the effect of a swarm sentence different from the effect of ANY sentence?
Contrasts on paired pairs (same pair, different appended sentence), mean log-odds, agent-cluster bootstrap:
  any-sentence effect   = neutral - none
  specific effect       = benign - neutral   (exculpatory)   and   coord - neutral (relay)
  vs-nothing effect     = benign - none      and   coord - none
"""
import json, glob, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
S = pd.read_parquet(ROOT / "results" / "v2_sample.parquet").sort_values("sid").reset_index(drop=True)
df = pd.read_parquet(ROOT / "data" / "edits.parquet", columns=["label", "ip16"])
lab = df.label.values; ipv = df.ip16.fillna("?").values
S["agent_a"] = np.where(lab[S.a.values] != "", lab[S.a.values], "anon:" + ipv[S.a.values])
y = S.y.values; hard = S.neg_type.values != "easy"
cl = S.groupby("agent_a").indices; keys = list(cl)
rng = np.random.default_rng(0)
sources = {}
R = pd.read_json(ROOT / "results" / "k_swarm-recruit" / "recruit_scores.jsonl", lines=True)
for m, g in R.groupby("model"):
    W = g.pivot(index="sid", columns="cond", values="score").loc[S.sid]
    sources[f"{m.split('/')[-1]} verbal [security prompt]"] = {c: W[c].values for c in W.columns}
for f in glob.glob(str(ROOT / "results" / "k_swarm-recruit8n" / "recruit_scores.jsonl")):
    R2 = pd.read_json(f, lines=True)
    for m, g in R2.groupby("model"):
        W = g.pivot(index="sid", columns="cond", values="score").loc[S.sid]
        sources[f"{m.split('/')[-1]} verbal [neutral prompt]"] = {c: W[c].values for c in W.columns}
for style in ["neutral", "primed"]:
    fs = {c: ROOT / "results" / "local_inject" / f"Qwen3-1.7B_{style}_{c}.npz" for c in ["none", "neutral", "benign", "coord"]}
    if all(p.exists() for p in fs.values()):
        sources[f"Qwen3-1.7B verbal [{style} prompt]"] = {c: np.load(p)["score"] for c, p in fs.items()}
out = {}
def ci(d, mask):
    m0 = d[mask].mean()
    bs = []
    for _ in range(1000):
        ii = np.concatenate([cl[keys[j]] for j in rng.integers(0, len(keys), len(keys))]); ii = ii[mask[ii]]
        bs.append(d[ii].mean())
    return float(m0), [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]
for name, sc in sources.items():
    print(f"\n{name}")
    res = {}
    for lab_, a, b, mask in [("any sentence, linked", "neutral", "none", y == 1), ("any sentence, unlinked", "neutral", "none", (y == 0) & hard),
                             ("exculpatory vs neutral, linked", "benign", "neutral", y == 1), ("exculpatory vs nothing, linked", "benign", "none", y == 1),
                             ("relay vs neutral, unlinked", "coord", "neutral", (y == 0) & hard), ("relay vs nothing, unlinked", "coord", "none", (y == 0) & hard)]:
        m0, c = ci(sc[a] - sc[b], mask); res[lab_] = dict(mean=m0, ci=c)
        print(f"  {lab_:34s} {m0:+.3f} [{c[0]:+.3f}, {c[1]:+.3f}]")
    out[name] = res
json.dump(out, open(ROOT / "results" / "recruit_null.json", "w"), indent=1)
