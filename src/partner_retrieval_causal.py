"""Causal partner retrieval: 'who did this agent coordinate with?'

Query: an edit e with >= 1 true partner among EARLIER edits by a different agent (true partner = cites the same
distinctive external URL, 2-50 agents). Candidates: every earlier edit by a different agent (thousands).
Each method ranks candidates using only fields visible at query time (URLs never visible). Hit@k = a true partner
is among the top k. Methods:
  content-emb    cosine of URL-free embeddings              content-tfidf  cosine of URL-free TF-IDF
  recency        most recent first                           page           same page first, then recency
  ip             same /16 first, then recency                agent-history  edits by agents who previously shared a
                                                                            page with e's agent, then recency
  metadata       page > ip > recency (lexicographic)          content+meta   content-emb + small page/recency bonus
Also reported on 'novel' queries: no true earlier partner is a near-copy (embedding cos >= 0.95).
Ties broken at random.
"""
import collections, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
sys.path.insert(0, str(Path(__file__).parent))
from pairs_v2 import is_ext

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_parquet(ROOT / "data" / "edits.parquet")
emb = np.load(ROOT / "data" / "emb_bge_text_nourl.npy")
TXT = df.text_nourl.str.slice(0, 4000).values
CK = 1000
Xc = {}
for c0 in range(CK, len(df) + CK, CK):          # vectorizer fitted on edits [0, c0) only
    Xc[c0] = TfidfVectorizer(min_df=2, max_features=50000, sublinear_tf=True).fit(TXT[:c0]).transform(TXT)
def Xfor(e):
    return Xc[max(CK, (e // CK) * CK)] if e >= CK else Xc[CK]
agent = np.where(df.label != "", df.label, "anon:" + df.ip16.fillna("?"))
ext = df.urls.map(lambda us: [u for u in us if is_ext(u)]).values
u2ag = collections.defaultdict(set)
for a, us in zip(agent, ext):
    for u in us:
        u2ag[u].add(a)
dist = {u for u, s in u2ag.items() if 2 <= len(s) <= 50}
dset = [set(u for u in us if u in dist) for us in ext]
t = df.time.astype("int64").values // 10**9
page, ip = df.page_key.values, df.ip16.fillna("?").values
rng = np.random.default_rng(0)
# agent -> pages seen so far (built incrementally, causal)
K = [1, 5, 10, 50, 100]
methods = ["content-emb", "content-tfidf", "recency", "page", "ip", "agent-history", "metadata", "content+meta", "random"]
hits = {m: {k: [] for k in K} for m in methods}
novel_flag, n_cands, q_agent = [], [], []
aid = pd.factorize(agent)[0]
ag_pages = collections.defaultdict(set)     # agent id -> pages it edited so far (causal)
page_ags = collections.defaultdict(set)     # page -> agent ids that edited it so far (causal)
u2e = collections.defaultdict(list)         # distinctive url -> earlier edits
for e in range(len(df)):
    cand_true = set()
    for u in dset[e]:
        cand_true.update(x for x in u2e[u] if aid[x] != aid[e])
    if cand_true and e >= CK:
        C = np.arange(e)
        C = C[(aid[C] != aid[e]) & (t[C] <= t[e])]
        if len(C) > 0:
            y = np.isin(C, list(cand_true))
            if y.any():
                tie = rng.random(len(C)) * 1e-9
                rec = (t[C] - t[e]) / 1e7                                  # in (-inf, 0], more recent = higher
                s_page = (page[C] == page[e]).astype(float)
                s_ip = (ip[C] == ip[e]).astype(float)
                # agents that previously shared a page with e's agent (identity graph so far)
                collab = set().union(*[page_ags[p] for p in ag_pages[aid[e]]]) if ag_pages[aid[e]] else set()
                collab.discard(aid[e])
                s_hist = np.isin(aid[C], list(collab)).astype(float) if collab else np.zeros(len(C))
                ce = emb[C] @ emb[e]
                Xe = Xfor(e); ct = np.asarray(Xe[C].multiply(Xe[e]).sum(1)).ravel()
                S = {"content-emb": ce, "content-tfidf": ct, "recency": rec, "page": s_page * 10 + rec,
                     "ip": s_ip * 10 + rec, "agent-history": s_hist * 10 + rec, "metadata": s_page * 100 + s_ip * 10 + rec,
                     "content+meta": ce + 0.05 * s_page + 0.01 * np.exp(rec * 100), "random": rng.random(len(C))}
                for m, s in S.items():
                    order = np.argsort(-(s + tie))
                    first = np.argmax(y[order])                          # rank of first true partner
                    for k in K:
                        hits[m][k].append(first < k)
                novel_flag.append(bool((emb[list(cand_true)] @ emb[e]).max() < 0.95)); q_agent.append(aid[e])
                n_cands.append(len(C))
    ag_pages[aid[e]].add(page[e]); page_ags[page[e]].add(aid[e])
    for u in dset[e]:
        u2e[u].append(e)
nov = np.array(novel_flag); qa = np.array(q_agent)
rngb = np.random.default_rng(1); grp = pd.Series(np.arange(len(qa))).groupby(qa).indices; gk = list(grp)
def boot(v, mask=None, n=1000):
    v = np.asarray(v, float); mask = np.ones(len(v), bool) if mask is None else mask; bs = []
    for _ in range(n):
        ii = np.concatenate([grp[gk[j]] for j in rngb.integers(0, len(gk), len(gk))]); ii = ii[mask[ii]]; bs.append(v[ii].mean())
    return [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]
out = {"n_queries": len(nov), "n_novel": int(nov.sum()), "median_candidates": float(np.median(n_cands))}
print(out)
for m in methods:
    h = {k: np.array(v) for k, v in hits[m].items()}
    out[m] = {f"hit@{k}": float(h[k].mean()) for k in K}
    out[m].update({f"novel_hit@{k}": float(h[k][nov].mean()) for k in K})
    out[m]["hit@1_ci"] = boot(h[1]); out[m]["novel_hit@1_ci"] = boot(h[1], nov)
    print(f"{m:15s} " + " ".join(f"@{k}:{out[m][f'hit@{k}']:.3f}" for k in K) + "  | novel " +
          " ".join(f"@{k}:{out[m][f'novel_hit@{k}']:.3f}" for k in K))
json.dump(out, open(ROOT / "results" / "partner_retrieval_causal.json", "w"), indent=1, default=float)
