"""Leakage control for the SwarmTraces replication.

The original mask removed 'zz...' mailbox tokens (case-sensitive) and caps/underscore identifiers. Fragments of a
mailbox name (e.g. 'RECON', 'MODAL') can survive elsewhere in the text. Two conservative checks:
  A) ultra-strict mask: case-insensitive removal of zz-tokens AND of every >=5-char fragment of ANY mailbox name,
     from all texts; then recompute TF-IDF and embedding similarity on all pairs.
  B) leak-free subset: drop every positive pair in which any >=5-char fragment of a shared mailbox name occurs in
     both (originally masked) texts; re-score similarity and the LLM on the remaining pairs.
"""
import gzip, json, re, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
sys.path.insert(0, str(Path(__file__).parent))
from run_eval import scores

ROOT = Path(__file__).resolve().parents[1]
rows = [json.loads(l) for l in gzip.open(ROOT / "data" / "raw_st" / "r.jsonl.gz", "rt", encoding="utf-8")]
Z = re.compile(r"\bzz[A-Z0-9][A-Za-z0-9_]{3,80}")          # identical to st_linkage.py (defines the artifact set)
ZI = re.compile(r"zz[A-Za-z0-9_]{4,80}", re.I)               # case-insensitive, no word boundary
STRICT = re.compile(r"\b[A-Z0-9_]*[A-Z][A-Z0-9_]*_[A-Z0-9_]+\b|\b[A-Z]{2,}[0-9]+[A-Z0-9]*\b")
R = [r for r in rows if Z.search(r["text"] or "")]            # same order as st_linkage.py
zz = [set(Z.findall(r["text"])) for r in R]
orig = [STRICT.sub("[ID]", Z.sub("[MAILBOX]", r["text"]))[:3000] for r in R]
frags = sorted({f for s in zz for z in s for f in re.split(r"_", z[2:].lower()) if len(f) >= 5}, key=len, reverse=True)
FR = re.compile("|".join(re.escape(f) for f in frags), re.I)
ultra = [FR.sub("[F]", ZI.sub("[MAILBOX]", STRICT.sub("[ID]", r["text"])))[:3000] for r in R]
print("fragments masked:", len(frags))

P = pd.read_parquet(ROOT / "results" / "st_pairs.parquet")
assert len(P) and P.a.max() < len(R), "pair indices out of range: artifact order changed"


def leaky(a, b):
    for z in zz[a] & zz[b]:
        for f in [f for f in re.split(r"_", z[2:].lower()) if len(f) >= 5]:
            if f in orig[a].lower() and f in orig[b].lower():
                return True
    return False


P["leak"] = [leaky(a, b) if y == 1 else False for a, b, y in zip(P.a, P.b, P.y)]
out = {"n_leaky_pos": int(P.leak.sum()), "n_pos": int(P.y.sum())}
# A) ultra-strict mask on all pairs
X = TfidfVectorizer(min_df=2, sublinear_tf=True, token_pattern=r"[A-Za-z_][A-Za-z0-9_]+").fit_transform(ultra)
P["tfidf_ultra"] = np.asarray(X[P.a.values].multiply(X[P.b.values]).sum(1)).ravel()
from sentence_transformers import SentenceTransformer
m = SentenceTransformer("BAAI/bge-base-en-v1.5", device="cuda"); m.half(); m.max_seq_length = 512
E = m.encode(ultra, batch_size=64, normalize_embeddings=True, convert_to_numpy=True)
P["emb_ultra"] = (E[P.a] * E[P.b]).sum(1)
nt = P.neg_type.values
for c in ["tfidf_text_strict", "emb_text_strict", "tfidf_ultra", "emb_ultra"]:
    out[f"all_pairs | {c}"] = scores(P.y.values, P[c].values, nt, n_boot=300)["all"]
# B) leak-free subset
Q = P[~P.leak]
for c in ["tfidf_text_strict", "emb_text_strict", "tfidf_ultra", "emb_ultra"]:
    out[f"leak_free | {c}"] = scores(Q.y.values, Q[c].values, Q.neg_type.values, n_boot=300)["all"]
L = pd.read_json(ROOT / "results" / "k_swarm-st-llm" / "st_llm_scores.jsonl", lines=True)
L = L[L.pair_idx.isin(Q.index)]
assert (L.y.values == P.loc[L.pair_idx, "y"].values).all(), "LLM pair labels do not match st_pairs rows"
for v, g in L.groupby("variant"):
    out[f"leak_free | qwen8b_{v}"] = scores(g.y.values, g.score.values, np.array(["hard"] * len(g)), n_boot=300)["all"]
    sub = Q.loc[g.pair_idx]
    out[f"leak_free_llm_sample | tfidf_ultra ({v} pairs)"] = scores(sub.y.values, sub.tfidf_ultra.values, np.array(["hard"] * len(sub)), n_boot=300)["all"]
for k, v in out.items():
    print(f"{k:50s}", v if not isinstance(v, dict) else f"AUROC={v['auroc']:.3f} CI={np.round(v['ci'],3)} R@1%={v['recall_at_fpr1']:.3f}")
json.dump(out, open(ROOT / "results" / "st_leakcheck.json", "w"), indent=1, default=float)
