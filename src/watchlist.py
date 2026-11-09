"""Budgeted, causal swarm oversight: who/what should a monitor watch?

Split the timeline at a cut time. Policies may use ONLY observation-period edits (and their ground-truth labels,
i.e. a past investigation) plus the incoming edit itself. They are scored on future edits.
  Ground truth (future): an edit is 'coordinated' if it cites a distinctive external URL (cited by 2-50 distinct
  agents overall) that a *different* agent also cites. URLs are never shown to any policy.
Policies (alert budget = share of future edits flagged):
  agent watchlist   - agent labels ranked by #coordinated edits in observation
  ip watchlist      - /16 IP prefixes ranked the same way
  page watchlist    - pages ranked the same way
  content watchlist - max cosine (URL-free embedding) of the incoming edit to observation coordinated edits,
                      minus max cosine to observation non-coordinated edits (kNN contrast)
  random            - flags a random subset
Reported: recall of future coordinated edits at fixed alert budgets, identity churn, and results on the
'novel' subset (future coordinated edits with no near-copy, cos < 0.95, among observation edits).
"""
import collections, json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from pairs_v2 import is_ext

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_parquet(ROOT / "data" / "edits.parquet")
emb = np.load(ROOT / "data" / "emb_bge_text_nourl.npy")
df["agent"] = np.where(df.label != "", df.label, "anon:" + df.ip16.fillna("?"))
ext = df.urls.map(lambda us: [u for u in us if is_ext(u)])
u2ag = collections.defaultdict(set)
for a, us in zip(df.agent, ext):
    for u in us:
        u2ag[u].add(a)
dist = {u for u, s in u2ag.items() if 2 <= len(s) <= 50}
# coordinated: cites a distinctive URL that some OTHER agent cites
df["coord"] = [any(u in dist and len(u2ag[u] - {a}) > 0 for u in us) for a, us in zip(df.agent, ext)]
BUDGETS = [0.01, 0.02, 0.05, 0.10, 0.20]


def recall_at_budget(score, y, budget, rng):
    # ties broken at random so that discrete watchlists are compared fairly at exactly the same alert volume
    order = np.lexsort((rng.random(len(score)), -score))
    k = int(round(budget * len(score)))
    flagged = np.zeros(len(score), bool); flagged[order[:k]] = True
    return float(flagged[y].sum() / max(1, y.sum()))


def run(cut_q):
    rng = np.random.default_rng(0)
    cut = int(len(df) * cut_q)              # eids are time-ordered
    obs, fut = df.iloc[:cut], df.iloc[cut:]
    y = fut.coord.values
    res = {"cut_time": str(df.time.iloc[cut]), "n_obs": len(obs), "n_future": len(fut),
           "future_coord_share": float(y.mean())}
    # identity churn
    for col in ["agent", "ip16", "page_key"]:
        seen = set(obs[col])
        res[f"churn_{col}"] = float(np.mean([v not in seen for v in fut[col].values[y]]))
    scores = {}
    for col, name in [("agent", "agent watchlist"), ("ip16", "ip watchlist"), ("page_key", "page watchlist")]:
        cnt = obs[obs.coord].groupby(col).size()
        scores[name] = fut[col].map(cnt).fillna(0).values.astype(float)
    Eo, Ef = emb[obs.eid.values], emb[fut.eid.values]
    pos, neg = Eo[obs.coord.values], Eo[~obs.coord.values]
    def maxcos(A, B):
        return np.concatenate([(A[i:i + 2048] @ B.T).max(1) for i in range(0, len(A), 2048)])
    mp, mn = maxcos(Ef, pos), maxcos(Ef, neg)
    scores["content watchlist"] = mp - mn
    scores["content (max cos to past coordinated)"] = mp
    scores["random"] = rng.random(len(fut))
    nearest_any = maxcos(Ef, Eo)
    novel = y & (nearest_any < 0.95)
    res["novel_share_of_coord"] = float(novel.sum() / max(1, y.sum()))
    for name, s in scores.items():
        res[name] = {f"recall@{b:.0%}": recall_at_budget(s, y, b, rng) for b in BUDGETS}
        # novel subset: coordinated edits that are not near-copies of anything seen before; non-coordinated kept
        keep = novel | ~y
        res[name].update({f"novel_recall@{b:.0%}": recall_at_budget(s[keep], novel[keep], b, rng) for b in BUDGETS})
    return res


out = {}
for q in [0.5, 0.7]:
    r = run(q); out[f"cut{int(q*100)}"] = r
    print(f"\n=== cut at {q:.0%} of edits ({r['cut_time']}); future coordinated share {r['future_coord_share']:.2f}; "
          f"novel share {r['novel_share_of_coord']:.2f} ===")
    print(f"churn: unseen agent {r['churn_agent']:.2f}  unseen ip16 {r['churn_ip16']:.2f}  unseen page {r['churn_page_key']:.2f}")
    for name in ["agent watchlist", "ip watchlist", "page watchlist", "content watchlist", "content (max cos to past coordinated)", "random"]:
        d = r[name]
        print(f"{name:40s} " + " ".join(f"{b:.0%}:{d[f'recall@{b:.0%}']:.2f}" for b in BUDGETS) +
              "  | novel " + " ".join(f"{d[f'novel_recall@{b:.0%}']:.2f}" for b in BUDGETS))
json.dump(out, open(ROOT / "results" / "watchlist.json", "w"), indent=1, default=float)
