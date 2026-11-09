"""Ground-truth linkage pairs from masked shared artifacts.

Positive pair: two edits by different agents that cite the same *distinctive* external URL
(used by between MIN_AG and MAX_AG distinct agents). Very common URLs (the task's own data files)
are excluded because they reflect a common cause, not linkage.
Negatives: random different-agent pairs that share no URL. "hard" negatives are also from the
same wiki and within 24h of each other, so time proximity alone cannot separate them.
URLs are masked from every detector (see data.text_masked).
"""
import collections, itertools
import numpy as np
import pandas as pd

WIKI_HOSTS = ("wikiservice.at", "probierwiki", "example.com", "localhost")


def agent_id(df):
    return np.where(df["label"] != "", df["label"], "anon:" + df["ip16"].fillna("?"))


def build_pairs(df, min_ag=2, max_ag=50, n_pos=20000, n_neg=20000, seed=0):
    rng = np.random.default_rng(seed)
    df = df.copy()
    df["agent"] = agent_id(df)
    u2e = collections.defaultdict(list)
    for eid, urls in zip(df.eid, df.urls):
        for u in urls:
            if not any(h in u.split("/")[0] for h in WIKI_HOSTS):
                u2e[u].append(eid)
    agent = df.set_index("eid")["agent"]
    pos = set()
    url_used = []
    for u, es in u2e.items():
        ags = set(agent[e] for e in es)
        if not (min_ag <= len(ags) <= max_ag):
            continue
        url_used.append(u)
        es = sorted(set(es))
        cand = [(a, b) for a, b in itertools.combinations(es, 2) if agent[a] != agent[b]]
        if len(cand) > 200:  # cap per URL so a few big URLs do not dominate
            cand = [cand[i] for i in rng.choice(len(cand), 200, replace=False)]
        pos.update(cand)
    pos = list(pos)
    if len(pos) > n_pos:
        pos = [pos[i] for i in rng.choice(len(pos), n_pos, replace=False)]
    url_sets = df.set_index("eid")["urls"].map(set)
    t = df.set_index("eid")["time"].astype("int64") // 10**9
    wiki = df.set_index("eid")["wiki"]
    eids = df.eid.values

    def ok_neg(a, b):
        return a != b and agent[a] != agent[b] and not (url_sets[a] & url_sets[b])

    neg_easy, neg_hard = [], []
    while len(neg_easy) < n_neg // 2:
        a, b = rng.choice(eids, 2)
        if ok_neg(a, b):
            neg_easy.append((min(a, b), max(a, b)))
    order = np.argsort(t.values)
    ts = t.values[order]
    while len(neg_hard) < n_neg // 2:
        a = rng.choice(eids)
        lo, hi = np.searchsorted(ts, [t[a] - 86400, t[a] + 86400])
        b = eids[order[rng.integers(lo, hi)]]
        if ok_neg(a, b) and wiki[a] == wiki[b]:
            neg_hard.append((min(a, b), max(a, b)))
    P = pd.DataFrame(pos, columns=["a", "b"]).assign(y=1, neg_type="pos")
    E = pd.DataFrame(neg_easy, columns=["a", "b"]).assign(y=0, neg_type="easy")
    H = pd.DataFrame(neg_hard, columns=["a", "b"]).assign(y=0, neg_type="hard")
    out = pd.concat([P, E, H], ignore_index=True).drop_duplicates(["a", "b"])
    return out, len(url_used)
