"""v2 head-to-head on the 1,500-pair v2 sample (audit-corrected).

All monitors see the same URL-free text truncated to 1,500 chars per post.
  zero-shot:  TF-IDF cosine, embedding cosine, LLM verbal score (logsumexp YES - logsumexp NO, full vocab)
  supervised: cheap HGB (trained on v2 pairs of the other time blocks; edit-disjoint), linear probes on LLM
              activations (trained on sample pairs of the other blocks; layer chosen by nested CV on training blocks)
Metrics: macro over the 5 time blocks; agent-cluster bootstrap CIs; paired bootstrap for differences.
"""
import json, sys, glob
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(Path(__file__).parent))
from run_eval_v2 import evaluate
from features_v2 import FIELD_OF

ROOT = Path(__file__).resolve().parents[1]
KD = ROOT / "results" / "k_swarm-probe-acts"
df = pd.read_parquet(ROOT / "data" / "edits.parquet")
S = pd.read_parquet(ROOT / "results" / "v2_sample.parquet").sort_values("sid").reset_index(drop=True)
agent = np.where(df.label != "", df.label, "anon:" + df.ip16.fillna("?"))
S["agent_a"] = agent[S.a.values]
T = df.text_nourl.str.slice(0, 1500)
# zero-shot similarity on exactly the text the LLM saw
X = TfidfVectorizer(min_df=2, max_features=50000, sublinear_tf=True).fit_transform(T.values)
S["tfidf_1500"] = np.asarray(X[S.a.values].multiply(X[S.b.values]).sum(1)).ravel()
from sentence_transformers import SentenceTransformer
m = SentenceTransformer("BAAI/bge-base-en-v1.5", device="cuda"); m.half(); m.max_seq_length = 512
ids = np.unique(np.r_[S.a.values, S.b.values])
Eb = dict(zip(ids, m.encode(T.values[ids].tolist(), batch_size=64, normalize_embeddings=True, convert_to_numpy=True)))
S["emb_1500"] = [float(Eb[a] @ Eb[b]) for a, b in zip(S.a, S.b)]
# supervised cheap monitor: trained on v2 pairs of other blocks (full features), scored on the sample
PV = pd.read_parquet(ROOT / "results" / "pairs_v2.parquet")
feats = [c for c in PV.columns if c in FIELD_OF]
key = PV.set_index(["a", "b"])
Ssub = key.loc[list(zip(S.a, S.b)), feats].reset_index(drop=True)
assert len(Ssub) == len(S)
for name, cols in [("cheap_all", feats), ("cheap_content", [f for f in feats if FIELD_OF[f] == "content"])]:
    p = np.zeros(len(S))
    for k in range(5):
        tr = PV.fold.values != k; te = S.fold.values == k
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0).fit(PV.loc[tr, cols], PV.y.values[tr])
        p[te] = clf.predict_proba(Ssub.loc[te, cols])[:, 1]
    S[name] = p
out = {}


def ev(col):
    r = evaluate(S, S[col].values, n_boot=500); out[col] = r
    print(f"{col:40s} hard AUROC={r['auroc_hard']:.3f} [{r['auroc_hard_ci'][0]:.3f},{r['auroc_hard_ci'][1]:.3f}] "
          f"TPR@1%={r['tpr1_hard']:.3f} [{r['tpr1_hard_ci'][0]:.3f},{r['tpr1_hard_ci'][1]:.3f}] all={r['auroc_all']:.3f}", flush=True)


def paired_diff(c1, c2, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    hard = S.neg_type.values != "easy"
    def macro(ii):
        vals = []
        for k in range(5):
            jj = ii[(S.fold.values[ii] == k) & hard[ii]]
            yy = S.y.values[jj]
            if yy.min() == yy.max(): continue
            vals.append(roc_auc_score(yy, S[c1].values[jj]) - roc_auc_score(yy, S[c2].values[jj]))
        return np.mean(vals)
    cl = S.groupby("agent_a").apply(lambda g: g.index.values).values
    d0 = macro(np.arange(len(S)))
    bs = [macro(np.concatenate([cl[j] for j in rng.integers(0, len(cl), len(cl))])) for _ in range(n_boot)]
    r = dict(diff=float(d0), ci=[float(np.quantile(bs, .025)), float(np.quantile(bs, .975))])
    out[f"diff[{c1} - {c2}]"] = r
    print(f"  paired hard-AUROC diff {c1} - {c2}: {r['diff']:+.3f} CI {np.round(r['ci'], 3)}")


for c in ["tfidf_1500", "emb_1500", "cheap_all", "cheap_content"]:
    ev(c)
runs = sorted(glob.glob(str(KD / "*.npz")))
for f in runs:
    tag = Path(f).stem
    Z = np.load(f)
    assert (Z["sid"] == S.sid.values).all(), "activation rows not aligned with sample"
    S[f"llm|{tag}"] = Z["score"]
    out[f"pmass|{tag}"] = float(Z["p_yes_mass"].mean())
    out[f"yesrate_neg|{tag}"] = float((Z["score"][S.y.values == 0] > 0).mean())
    ev(f"llm|{tag}")
    print(f"  mean P(YES-variants)={out[f'pmass|{tag}']:.3f}  share of negatives answered YES={out[f'yesrate_neg|{tag}']:.3f}")
    # probes with nested layer selection
    layers = sorted(int(k[1:]) for k in Z.files if k.startswith("L") and k[1:].isdigit())
    p = np.zeros(len(S)); chosen = []
    mk = lambda: make_pipeline(StandardScaler(), PCA(128, random_state=0), LogisticRegression(C=0.05, max_iter=3000))
    for k in range(5):
        tr_folds = [j for j in range(5) if j != k]
        best_l, best_s = None, -1
        for l in layers[1:]:
            Xl = Z[f"L{l}"].astype(np.float32); inner = []
            for j in tr_folds:
                itr = np.isin(S.fold.values, [x for x in tr_folds if x != j]); ite = S.fold.values == j
                inner.append(roc_auc_score(S.y.values[ite], mk().fit(Xl[itr], S.y.values[itr]).predict_proba(Xl[ite])[:, 1]))
            if np.mean(inner) > best_s:
                best_l, best_s = l, np.mean(inner)
        chosen.append(best_l)
        Xl = Z[f"L{best_l}"].astype(np.float32); tr = np.isin(S.fold.values, tr_folds); te = S.fold.values == k
        p[te] = mk().fit(Xl[tr], S.y.values[tr]).predict_proba(Xl[te])[:, 1]
    S[f"probe|{tag}"] = p; out[f"probe_layers|{tag}"] = chosen
    ev(f"probe|{tag}"); print(f"  probe layers chosen per fold: {chosen}")
    paired_diff("tfidf_1500", f"llm|{tag}")
    paired_diff(f"probe|{tag}", f"llm|{tag}")
    paired_diff("tfidf_1500", f"probe|{tag}")
    # stacking on standardized log-odds: does the LLM add to the cheap monitor?
    st = np.zeros(len(S))
    lo = lambda q: np.log(np.clip(q, 1e-6, 1 - 1e-6) / (1 - np.clip(q, 1e-6, 1 - 1e-6)))
    F2 = np.c_[lo(S.cheap_all.values), S[f"llm|{tag}"].values]
    for k in range(5):
        tr = S.fold.values != k; te = ~tr
        st[te] = make_pipeline(StandardScaler(), LogisticRegression()).fit(F2[tr], S.y.values[tr]).predict_proba(F2[te])[:, 1]
    S[f"stack|{tag}"] = st
    ev(f"stack|{tag}"); paired_diff(f"stack|{tag}", "cheap_all")
S.to_parquet(ROOT / "results" / "llm_v2_scores.parquet")
json.dump(out, open(ROOT / "results" / "llm_v2.json", "w"), indent=1, default=float)
