"""v2 ground truth, fixing the audit findings.

- Every edit used in any pair (positive or negative) cites >= 1 external URL, so URL presence is not a label cue.
- Edits are split into K contiguous time blocks with equal numbers of URL-bearing edits. All pairs are built
  *within* a block, so no edit appears in both a training and a test fold.
- Per block, #negatives == #positives (half easy, half hard), so every fold has the same base rate.
- Positive: different agents, both cite the same distinctive URL (cited by MIN_AG..MAX_AG distinct agents overall).
- Negative: different agents, no URL in common at all. Hard = same wiki and within 24 h.
"""
import collections, itertools
import numpy as np
import pandas as pd

WIKI_HOSTS = ("wikiservice.at", "probierwiki", "example.com", "localhost")


def is_ext(u):
    return not any(h in u.split("/")[0] for h in WIKI_HOSTS)


def build_pairs_v2(df, K=5, min_ag=2, max_ag=50, cap_per_url=200, seed=0, nearcopy_emb=None):
    rng = np.random.default_rng(seed)
    df = df.copy()
    df["agent"] = np.where(df.label != "", df.label, "anon:" + df.ip16.fillna("?"))
    df["ext"] = df.urls.map(lambda us: [u for u in us if is_ext(u)])
    elig = df[df.ext.str.len() > 0].sort_values("eid")
    # contiguous time blocks with equal numbers of eligible edits
    block = pd.Series(-1, index=df.eid)
    for k, chunk in enumerate(np.array_split(elig.eid.values, K)):
        block[chunk] = k
    df["block"] = block.values
    agent = df.set_index("eid").agent; blk = df.set_index("eid").block
    urlset = df.set_index("eid").urls.map(set)
    t = df.set_index("eid").time.astype("int64") // 10**9
    wiki = df.set_index("eid").wiki
    u2e = collections.defaultdict(list)
    for e, us in zip(elig.eid, elig.ext):
        for u in us:
            u2e[u].append(e)
    n_ag = {u: len(set(agent[es])) for u, es in u2e.items()}
    distinctive = [u for u, n in n_ag.items() if min_ag <= n <= max_ag]
    rows = []
    for k in range(K):
        pos = set()
        for u in distinctive:
            es = sorted(e for e in set(u2e[u]) if blk[e] == k)
            cand = [(a, b) for a, b in itertools.combinations(es, 2) if agent[a] != agent[b]]
            if len(cand) > cap_per_url:
                cand = [cand[i] for i in rng.choice(len(cand), cap_per_url, replace=False)]
            pos.update(cand)
        pos = sorted(pos)
        ek =elig.eid.values[blk[elig.eid].values == k]
        tk = t[ek].values; order = np.argsort(tk); ek_sorted, tk_sorted = ek[order], tk[order]
        n_half = len(pos) // 2

        def ok(a, b):
            return a != b and agent[a] != agent[b] and not (urlset[a] & urlset[b])

        easy, hard, seen = [], [], set(pos)
        tries = 0
        while len(easy) < n_half and tries < 50 * n_half + 1000:
            tries += 1
            a, b = rng.choice(ek, 2)
            a, b = min(a, b), max(a, b)
            if ok(a, b) and (a, b) not in seen:
                easy.append((a, b)); seen.add((a, b))
        tries = 0
        while len(hard) < len(pos) - n_half and tries < 200 * n_half + 1000:
            tries += 1
            a = rng.choice(ek)
            lo, hi = np.searchsorted(tk_sorted, [t[a] - 86400, t[a] + 86400])
            b = ek_sorted[rng.integers(lo, hi)]
            a2, b2 = min(a, b), max(a, b)
            if ok(a2, b2) and wiki[a2] == wiki[b2] and (a2, b2) not in seen:
                hard.append((a2, b2)); seen.add((a2, b2))
        for lst, nt, y in [(pos, "pos", 1), (easy, "easy", 0), (hard, "hard", 0)]:
            rows += [(a, b, y, nt, k) for a, b in lst]
    P = pd.DataFrame(rows, columns=["a", "b", "y", "neg_type", "fold"])
    assert (blk[P.a].values == blk[P.b].values).all()
    assert not P.duplicated(["a", "b"]).any()
    return P.reset_index(drop=True), len(distinctive)
