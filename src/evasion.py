"""Evasion test: the swarm paraphrases its posts. Monitors are trained on normal traffic and tested on
held-out pairs whose B-side post was paraphrased by Qwen3-8B. Metadata is untouched by paraphrasing."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
sys.path.insert(0, str(Path(__file__).parent))
from run_eval import scores
from features import FIELD_OF

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_parquet(ROOT / "data" / "edits.parquet")
emb_s = np.load(ROOT / "data" / "emb_bge_text_strict.npy")
P = pd.read_parquet(ROOT / "results" / "pairs.parquet")
sub = pd.read_parquet(ROOT / "results" / "para_pairs.parquet")
para = {r["eid"]: r for r in map(json.loads, open(ROOT / "results" / "kaggle_para" / "paraphrases.jsonl", encoding="utf-8"))}
sub = sub[sub.b.isin(para)].reset_index(drop=True)

# strict-mask content features for every pair (the paraphrases were made from strict text)
txt = df.text_strict.str.slice(0, 4000).tolist()
vec = TfidfVectorizer(min_df=2, max_features=50000, sublinear_tf=True).fit(txt)
X = vec.transform(txt)
P["tfidf_cos"] = np.asarray(X[P.a.values].multiply(X[P.b.values]).sum(1)).ravel()
P["emb_cos"] = (emb_s[P.a.values] * emb_s[P.b.values]).sum(1)
feats = [c for c in P.columns if c in FIELD_OF]
groups = {"all fields": feats, "content only": [f for f in feats if FIELD_OF[f] == "content"],
          "metadata only": [f for f in feats if FIELD_OF[f] != "content"]}

test = P.loc[sub.row.values].copy().reset_index(drop=True)
train = P.drop(index=sub.row.values)
# paraphrased B side
m = SentenceTransformer("BAAI/bge-base-en-v1.5", device="cuda"); m.half(); m.max_seq_length = 512
ptxt = [para[b]["para"][:3000] for b in sub.b]
pe = m.encode(ptxt, batch_size=64, normalize_embeddings=True, convert_to_numpy=True)
tp = test.copy()
Xp = vec.transform([t[:4000] for t in ptxt])
tp["tfidf_cos"] = np.asarray(X[test.a.values].multiply(Xp).sum(1)).ravel()
tp["emb_cos"] = (emb_s[test.a.values] * pe).sum(1)
lb, lp = df.body_len.values[test.b.values], np.array([len(t) for t in ptxt])
la = df.body_len.values[test.a.values]
tp["len_ratio"] = np.minimum(la, lp) / np.maximum(1, np.maximum(la, lp))
# how much did the paraphrase change the post? (self-similarity original B vs paraphrased B)
self_sim = (emb_s[test.b.values] * pe).sum(1)

out = {"n_test": len(test), "para_self_cos_median": float(np.median(self_sim)),
       "para_truncated_share": float(np.mean([para[b]["finish"] == "length" for b in sub.b]))}
for g, cols in groups.items():
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0).fit(train[cols], train.y)
    for cond, T in [("original", test), ("paraphrased", tp)]:
        s = scores(T.y.values, clf.predict_proba(T[cols])[:, 1], T.neg_type.values, n_boot=200)
        out[f"{g} | {cond}"] = s
        print(f"{g:14s} {cond:12s} hard AUROC={s['hard']['auroc']:.3f} CI={s['hard']['ci']} R@1%={s['hard']['recall_at_fpr1']:.3f}")
print({k: v for k, v in out.items() if not isinstance(v, dict)})
json.dump(out, open(ROOT / "results" / "evasion.json", "w"), indent=1, default=float)
