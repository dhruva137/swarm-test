"""v2 pairwise features, grouped by the log field each one needs.

Changes from v1 (audit): text has URLs deleted (no placeholder); length uses the URL-free text; the agents'
co-activity feature is a TIME feature (it is computed from day bins) and is computed from the clock under test.
Identity/page aggregates use the whole unlabeled log (a forensic monitor has the full log; no labels are used).
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

FIELD_OF = {
    "dt_log": "time", "same_hour": "time", "same_day": "time", "agent_coactive": "time",
    "agent_page_jaccard": "identity",
    "same_ip": "network",
    "same_page": "location", "page_coagent": "location",
    "tfidf_cos": "content", "emb_cos": "content", "len_ratio": "content",
}


def pair_features_v2(df, pairs, emb, text_col="text_nourl", time_res=None, drop_time=False):
    E = df.copy()
    E["agent"] = np.where(E.label != "", E.label, "anon:" + E.ip16.fillna("?"))
    E = E.set_index("eid")
    a, b = pairs.a.values, pairs.b.values
    F = pd.DataFrame(index=pairs.index)
    if not drop_time:
        t = E.time.astype("int64").values // 10**9
        if time_res:
            t = (t // time_res) * time_res
        ta, tb = t[a], t[b]
        F["dt_log"] = -np.log1p(np.abs(ta - tb))
        F["same_hour"] = (ta // 3600 == tb // 3600).astype(float)
        F["same_day"] = (ta // 86400 == tb // 86400).astype(float)
        bin_s = max(86400, time_res or 0)
        ag_bins = E.assign(d=t // bin_s).groupby("agent").d.agg(set)
        F["agent_coactive"] = [len(ag_bins[x] & ag_bins[y]) / max(1, len(ag_bins[x] | ag_bins[y]))
                               for x, y in zip(E.agent.values[a], E.agent.values[b])]
    ag_pages = E.groupby("agent").page_key.agg(set)
    pk = E.page_key.values
    F["agent_page_jaccard"] = [len((ag_pages[x] - {p}) & (ag_pages[y] - {q})) / max(1, len((ag_pages[x] - {p}) | (ag_pages[y] - {q})))
                               for x, y, p, q in zip(E.agent.values[a], E.agent.values[b], pk[a], pk[b])]
    ip = E.ip16.fillna("?").values
    F["same_ip"] = (ip[a] == ip[b]).astype(float)
    F["same_page"] = (pk[a] == pk[b]).astype(float)
    page_agents = E.groupby("page_key").agent.agg(set)
    F["page_coagent"] = [len(page_agents[x] & page_agents[y]) / max(1, len(page_agents[x] | page_agents[y]))
                         for x, y in zip(pk[a], pk[b])]
    txt = E[text_col].str.slice(0, 4000).values
    X = TfidfVectorizer(min_df=2, max_features=50000, sublinear_tf=True).fit_transform(txt)
    F["tfidf_cos"] = np.asarray(X[a].multiply(X[b]).sum(1)).ravel()
    F["emb_cos"] = (emb[a] * emb[b]).sum(1)
    la, lb = E.len_nourl.values[a], E.len_nourl.values[b]
    F["len_ratio"] = np.minimum(la, lb) / np.maximum(1, np.maximum(la, lb))
    return F
