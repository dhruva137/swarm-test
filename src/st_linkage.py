"""Replication on the Hugging Face attack swarm (SwarmTraces).

Linkage task: two artifacts from *different* reconstruction roots that reference the same Artifactory mailbox
folder (zz...) belong to one coordination channel. Mailbox names are masked (plus a strict variant that also
masks long all-caps/underscore identifiers). Content is the only field SwarmTraces has.
Negatives: artifacts that also use mailboxes, but never a shared one.
Outputs results/st_eval.json and kaggle/st_data/pairs.jsonl for the LLM monitor.
"""
import gzip, json, re, sys, collections, itertools
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
sys.path.insert(0, str(Path(__file__).parent))
from run_eval import scores

ROOT = Path(__file__).resolve().parents[1]
rows = [json.loads(l) for l in gzip.open(ROOT / "data" / "raw_st" / "r.jsonl.gz", "rt", encoding="utf-8")]
Z = re.compile(r"\bzz[A-Z0-9][A-Za-z0-9_]{3,80}")
STRICT = re.compile(r"\b[A-Z0-9_]*[A-Z][A-Z0-9_]*_[A-Z0-9_]+\b|\b[A-Z]{2,}[0-9]+[A-Z0-9]*\b")
par = {r["id"]: r["parent_id"] for r in rows}
root = lambda i: par.get(i) or i
R = [r for r in rows if Z.search(r["text"] or "")]
df = pd.DataFrame({"id": [r["id"] for r in R], "kind": [r["kind"] for r in R],
                   "root": [root(r["id"]) for r in R],
                   "zz": [set(Z.findall(r["text"])) for r in R],
                   "text": [Z.sub("[MAILBOX]", r["text"])[:3000] for r in R]})
df["text_strict"] = df.text.map(lambda t: STRICT.sub("[ID]", t))
df = df.reset_index(drop=True)
rng = np.random.default_rng(0)
z2i = collections.defaultdict(list)
for i, zs in enumerate(df.zz):
    for z in zs:
        z2i[z].append(i)
pos = set()
for z, ii in z2i.items():
    for a, b in itertools.combinations(sorted(set(ii)), 2):
        if df.root[a] != df.root[b]:
            pos.add((a, b))
pos = list(pos)
neg = set()
while len(neg) < len(pos):
    a, b = sorted(rng.choice(len(df), 2, replace=False))
    if df.root[a] != df.root[b] and not (df.zz[a] & df.zz[b]):
        neg.add((a, b))
P = pd.DataFrame(pos + list(neg), columns=["a", "b"])
P["y"] = [1] * len(pos) + [0] * len(neg)
P["neg_type"] = ["pos"] * len(pos) + ["hard"] * len(neg)
print("artifacts", len(df), "pairs", P.y.value_counts().to_dict())

out = {"n_artifacts": len(df), "n_pos": len(pos), "n_neg": len(neg)}
from sentence_transformers import SentenceTransformer
m = SentenceTransformer("BAAI/bge-base-en-v1.5", device="cuda"); m.half(); m.max_seq_length = 512
for col in ["text", "text_strict"]:
    E = m.encode(df[col].tolist(), batch_size=64, normalize_embeddings=True, convert_to_numpy=True)
    X = TfidfVectorizer(min_df=2, sublinear_tf=True, token_pattern=r"[A-Za-z_][A-Za-z0-9_]+").fit_transform(df[col])
    P[f"emb_{col}"] = (E[P.a] * E[P.b]).sum(1)
    P[f"tfidf_{col}"] = np.asarray(X[P.a.values].multiply(X[P.b.values]).sum(1)).ravel()
    for f in [f"emb_{col}", f"tfidf_{col}"]:
        out[f] = scores(P.y.values, P[f].values, P.neg_type.values, n_boot=300)["all"]
        print(f, round(out[f]["auroc"], 3), "R@1%", round(out[f]["recall_at_fpr1"], 3))
P.to_parquet(ROOT / "results" / "st_pairs.parquet")
json.dump(out, open(ROOT / "results" / "st_eval.json", "w"), indent=1, default=float)

# export a balanced sample for the LLM monitor (strict-masked text, truncated)
d = ROOT / "kaggle" / "st_data"; d.mkdir(parents=True, exist_ok=True)
S = pd.concat([P[P.y == 1].sample(min(750, len(pos)), random_state=1), P[P.y == 0].sample(min(750, len(neg)), random_state=1)])
with open(d / "pairs.jsonl", "w", encoding="utf-8") as f:
    for i, r in S.iterrows():
        f.write(json.dumps({"pair_idx": int(i), "y": int(r.y), "neg_type": r.neg_type,
                            "a": {"text": df.text_strict[r.a][:1500]}, "b": {"text": df.text_strict[r.b][:1500]}}) + "\n")
json.dump({"title": "swarm-st-pairs", "id": "dhruvapgowda/swarm-st-pairs", "licenses": [{"name": "other"}]},
          open(d / "dataset-metadata.json", "w"))
print("exported", len(S))
