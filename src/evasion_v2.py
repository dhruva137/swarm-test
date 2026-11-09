"""v2 paraphrase-evasion test (audit-corrected).

Test pairs = v2 sample (1,500 pairs over 5 time blocks). The B-side post (URL-free, first 1,500 chars) was
paraphrased by Qwen3-8B. Every content feature, for training and test, original and paraphrased, is computed on
the same 1,500-char truncation. Supervised monitors are trained on v2 pairs of the other time blocks (edit-disjoint).
"""
import json, re, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(Path(__file__).parent))
from run_eval_v2 import evaluate
from features_v2 import FIELD_OF

ROOT = Path(__file__).resolve().parents[1]
N = 1500
df = pd.read_parquet(ROOT / "data" / "edits.parquet")
agent = np.where(df.label != "", df.label, "anon:" + df.ip16.fillna("?"))
T = df.text_nourl.str.slice(0, N).values
S = pd.read_parquet(ROOT / "results" / "v2_sample.parquet").sort_values("sid").reset_index(drop=True)
S["agent_a"] = agent[S.a.values]
para = {}
boiler = 0
for r in map(json.loads, open(ROOT / "results" / "k_swarm-para-v2" / "paraphrases.jsonl", encoding="utf-8")):
    t = r["para"].strip()
    if re.match(r"^(here is|here's|sure|below is)\b[^\n]*\n", t, re.I):
        t = t.split("\n", 1)[1].strip(); boiler += 1
    para[r["eid"]] = (t[:N], r["finish"])
assert set(S.b) <= set(para), "missing paraphrases"
PV = pd.read_parquet(ROOT / "results" / "pairs_v2.parquet")

vec = TfidfVectorizer(min_df=2, max_features=50000, sublinear_tf=True).fit(T)
X = vec.transform(T)
from sentence_transformers import SentenceTransformer
m = SentenceTransformer("BAAI/bge-base-en-v1.5", device="cuda"); m.half(); m.max_seq_length = 512
E = m.encode(T.tolist(), batch_size=64, normalize_embeddings=True, convert_to_numpy=True)
L = np.array([len(t) for t in T])


def content_feats(a, b, Xb, Eb, Lb):
    return pd.DataFrame({"tfidf_cos": np.asarray(X[a].multiply(Xb).sum(1)).ravel(),
                         "emb_cos": (E[a] * Eb).sum(1),
                         "len_ratio": np.minimum(L[a], Lb) / np.maximum(1, np.maximum(L[a], Lb))})


# training features (other blocks), same truncation
trF = content_feats(PV.a.values, PV.b.values, X[PV.b.values], E[PV.b.values], L[PV.b.values])
meta_cols = [c for c in PV.columns if c in FIELD_OF and FIELD_OF[c] != "content"]
trF = pd.concat([trF, PV[meta_cols].reset_index(drop=True)], axis=1)
key = PV.set_index(["a", "b"])
Smeta = key.loc[list(zip(S.a, S.b)), meta_cols].reset_index(drop=True)
ptxt = [para[b][0] for b in S.b]
Xp = vec.transform(ptxt)
Ep = m.encode(ptxt, batch_size=64, normalize_embeddings=True, convert_to_numpy=True)
Lp = np.array([len(t) for t in ptxt])
orig = pd.concat([content_feats(S.a.values, S.b.values, X[S.b.values], E[S.b.values], L[S.b.values]), Smeta], axis=1)
parf = pd.concat([content_feats(S.a.values, S.b.values, Xp, Ep, Lp), Smeta], axis=1)
out = {"n_pairs": len(S), "boilerplate_stripped": boiler,
       "para_truncated_share": float(np.mean([para[b][1] == "length" for b in set(S.b)])),
       "self_cos_median": float(np.median((E[S.b.values] * Ep).sum(1))),
       "self_tfidf_median": float(np.median(np.asarray(X[S.b.values].multiply(Xp).sum(1)).ravel()))}
content = ["tfidf_cos", "emb_cos", "len_ratio"]
for name, cols in [("zero-shot tfidf", None), ("zero-shot emb", None), ("supervised content only", content),
                   ("supervised all fields", content + meta_cols), ("supervised metadata only", meta_cols)]:
    for cond, F in [("original", orig), ("paraphrased", parf)]:
        if cols is None:
            p = F["tfidf_cos" if "tfidf" in name else "emb_cos"].values
        else:
            p = np.zeros(len(S))
            for k in range(5):
                tr = PV.fold.values != k; te = S.fold.values == k
                p[te] = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0).fit(
                    trF.loc[tr, cols], PV.y.values[tr]).predict_proba(F.loc[te, cols])[:, 1]
        S[f"{name}|{cond}"] = p
        r = evaluate(S, p, n_boot=500); out[f"{name} | {cond}"] = r
        print(f"{name:26s} {cond:12s} hard AUROC={r['auroc_hard']:.3f} [{r['auroc_hard_ci'][0]:.3f},{r['auroc_hard_ci'][1]:.3f}] TPR@1%={r['tpr1_hard']:.3f}", flush=True)
print({k: v for k, v in out.items() if not isinstance(v, dict)})
json.dump(out, open(ROOT / "results" / "evasion_v2.json", "w"), indent=1, default=float)
