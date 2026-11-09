"""Pairwise linkage features, grouped by the log field each one needs.

Every feature only uses masked text (no URLs), so ground truth never leaks.
Log fields: time, identity (agent label), network (ip16), location (page), content (text).
"""
import collections
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

FIELD_OF = {
    "dt_log": "time", "same_hour": "time", "same_day": "time",
    "agent_page_jaccard": "identity", "agent_coactive": "identity",
    "same_ip": "network",
    "same_page": "location", "page_coagent": "location",
    "tfidf_cos": "content", "emb_cos": "content", "len_ratio": "content",
}


def edit_tables(df):
    df = df.copy()
    df["agent"] = np.where(df.label != "", df.label, "anon:" + df.ip16.fillna("?"))
    return df.set_index("eid")


def pair_features(df, pairs, emb=None, time_res=None):
    """time_res: None = exact seconds, or bin size in seconds (coarsened clocks)."""
    E = edit_tables(df)
    a, b = pairs.a.values, pairs.b.values
    t = E.time.astype("int64").values // 10**9
    if time_res:
        t = (t // time_res) * time_res
    ta, tb = t[a], t[b]
    F = pd.DataFrame(index=pairs.index)
    dt = np.abs(ta - tb)
    F["dt_log"] = -np.log1p(dt)
    F["same_hour"] = (ta // 3600 == tb // 3600).astype(float)
    F["same_day"] = (ta // 86400 == tb // 86400).astype(float)
    # identity: do the two agents work on overlapping pages / at overlapping times overall?
    ag_pages = E.groupby("agent").page_key.agg(set)
    ag_days = E.assign(d=t // 86400).groupby("agent").d.agg(set)
    aa, ab = E.agent.values[a], E.agent.values[b]
    jac, coact = [], []
    for x, y, pk_a, pk_b in zip(aa, ab, E.page_key.values[a], E.page_key.values[b]):
        px, py = ag_pages[x] - {pk_a}, ag_pages[y] - {pk_b}  # exclude the pair's own pages
        jac.append(len(px & py) / max(1, len(px | py)))
        coact.append(len(ag_days[x] & ag_days[y]) / max(1, len(ag_days[x] | ag_days[y])))
    F["agent_page_jaccard"] = jac
    F["agent_coactive"] = coact
    ip = E.ip16.fillna("?").values
    F["same_ip"] = (ip[a] == ip[b]).astype(float)
    pk = E.page_key.values
    F["same_page"] = (pk[a] == pk[b]).astype(float)
    page_agents = E.groupby("page_key").agent.agg(set)
    F["page_coagent"] = [len(page_agents[x] & page_agents[y]) / max(1, len(page_agents[x] | page_agents[y]))
                         for x, y in zip(pk[a], pk[b])]
    txt = E.text_masked.str.slice(0, 4000).values
    X = TfidfVectorizer(min_df=2, max_features=50000, sublinear_tf=True).fit_transform(txt)
    F["tfidf_cos"] = np.asarray(X[a].multiply(X[b]).sum(1)).ravel()
    if emb is not None:
        F["emb_cos"] = (emb[a] * emb[b]).sum(1)
    la, lb = E.body_len.values[a], E.body_len.values[b]
    F["len_ratio"] = np.minimum(la, lb) / np.maximum(1, np.maximum(la, lb))
    return F
