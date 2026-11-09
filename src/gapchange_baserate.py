"""(1) Framing-specific know-say gap: (probe - verbal | primed) - (probe - verbal | neutral), paired cluster bootstrap.
   The probe is supervised in both framings, so supervision cancels out of this difference.
(2) Coordination base rate under different distinctive-URL caps, as share of link-citing edits and of all edits."""
import json, collections, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(Path(__file__).parent))
from pairs_v2 import is_ext
ROOT = Path(__file__).resolve().parents[1]
S = pd.read_parquet(ROOT / "results" / "llm_v2_scores.parquet")
hard = S.neg_type.values != "easy"; y = S.y.values; fold = S.fold.values
cl = S.groupby("agent_a").indices; keys = list(cl)
def gap(m, style, ii):
    v = []
    for k in range(5):
        jj = ii[(fold[ii] == k) & hard[ii]]; yy = y[jj]
        if yy.min() == yy.max(): continue
        v.append(roc_auc_score(yy, S[f"probe|{m}_{style}_meta"].values[jj]) - roc_auc_score(yy, S[f"llm|{m}_{style}_meta"].values[jj]))
    return np.mean(v)
out = {}
rng = np.random.default_rng(0)
for m in ["Qwen3-1.7B", "Qwen3-4B"]:
    f = lambda ii: gap(m, "primed", ii) - gap(m, "neutral", ii)
    d0 = f(np.arange(len(S)))
    bs = [f(np.concatenate([cl[keys[j]] for j in rng.integers(0, len(keys), len(keys))])) for _ in range(1000)]
    out[f"{m} gap_primed - gap_neutral"] = dict(diff=float(d0), ci=[float(np.quantile(bs, .025)), float(np.quantile(bs, .975))],
                                               gap_neutral=float(gap(m, "neutral", np.arange(len(S)))), gap_primed=float(gap(m, "primed", np.arange(len(S)))))
    r = out[f"{m} gap_primed - gap_neutral"]
    print(f"{m}: know-say gap neutral {r['gap_neutral']:+.3f}, primed {r['gap_primed']:+.3f}; framing-specific widening {d0:+.3f} CI [{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}]")
df = pd.read_parquet(ROOT / "data" / "edits.parquet")
agent = np.where(df.label != "", df.label, "anon:" + df.ip16.fillna("?"))
ext = df.urls.map(lambda us: [u for u in us if is_ext(u)])
u2ag = collections.defaultdict(set)
for a, us in zip(agent, ext):
    for u in us: u2ag[u].add(a)
has = ext.str.len().values > 0
for lo, hi in [(2, 10), (2, 50), (2, 200)]:
    dist = {u for u, s in u2ag.items() if lo <= len(s) <= hi}
    coord = np.array([any(u in dist and len(u2ag[u] - {a}) > 0 for u in us) for a, us in zip(agent, ext)])
    for q in [0.5, 0.7]:
        cut = int(len(df) * q); fut = slice(cut, None)
        lc = coord[fut][has[fut]].mean(); al = coord[fut].mean()
        out[f"base_rate cap {lo}-{hi} cut{int(q*100)}"] = dict(of_link_citing=float(lc), of_all=float(al))
        print(f"cap {lo}-{hi}, after {q:.0%} cut: coordinated = {lc:.2f} of link-citing edits, {al:.2f} of all edits")
json.dump(out, open(ROOT / "results" / "gapchange_baserate.json", "w"), indent=1)
