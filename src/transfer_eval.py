"""Cross-swarm transfer: a linkage probe trained on the WIKI swarm, applied unchanged to the HUGGING FACE attack swarm.

Same model, same prompt (security-primed, content only; SwarmTraces has no metadata).
Layer AND weights are chosen using wiki data only (5-fold CV over wiki time blocks); no SwarmTraces label is used
for training or selection. Compared on the 1,500 SwarmTraces pairs with: the model's own verbal score (zero-shot),
TF-IDF similarity (zero-shot), and an in-swarm probe (5-fold CV on SwarmTraces, an upper bound).
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
sys.path.insert(0, str(Path(__file__).parent))
from run_eval import scores

ROOT = Path(__file__).resolve().parents[1]
mk = lambda: make_pipeline(StandardScaler(), PCA(128, random_state=0), LogisticRegression(C=0.05, max_iter=3000))
model = sys.argv[1] if len(sys.argv) > 1 else "Qwen3-1.7B"
W = np.load(ROOT / "results" / "k_swarm-probe-acts" / f"{model}_primed_content.npz")
Sw = pd.read_parquet(ROOT / "results" / "v2_sample.parquet").sort_values("sid").reset_index(drop=True)
assert (W["sid"] == Sw.sid.values).all()
yw, fw = Sw.y.values, Sw.fold.values
st_path = {"Qwen3-1.7B": ROOT / "results" / "local_st" / "ST_Qwen3-1.7B_primed_content.npz",
           "Qwen3-4B": ROOT / "results" / "k_swarm-st-acts" / "ST_Qwen3-4B_primed_content.npz"}[model]
T = np.load(st_path)
st = [json.loads(l) for l in open(ROOT / "kaggle" / "st_data" / "pairs_sid.jsonl", encoding="utf-8")]
assert (T["sid"] == np.array([r["sid"] for r in st])).all()
yt = np.array([r["y"] for r in st])
layers = sorted(int(k[1:]) for k in W.files if k.startswith("L") and k[1:].isdigit())
# 1) choose layer on wiki only
cv = {}
for l in layers[1:]:
    X = W[f"L{l}"].astype(np.float32)
    cv[l] = np.mean([roc_auc_score(yw[fw == k], mk().fit(X[fw != k], yw[fw != k]).predict_proba(X[fw == k])[:, 1]) for k in range(5)])
best = max(cv, key=cv.get)
clf = mk().fit(W[f"L{best}"].astype(np.float32), yw)            # all wiki pairs
p_transfer = clf.decision_function(T[f"L{best}"].astype(np.float32))
# 2) baselines on the same SwarmTraces pairs
texts_a = [r["a"]["text"] for r in st]; texts_b = [r["b"]["text"] for r in st]
vec = TfidfVectorizer(min_df=1, sublinear_tf=True, token_pattern=r"[A-Za-z_][A-Za-z0-9_]+").fit(texts_a + texts_b)
A, B = vec.transform(texts_a), vec.transform(texts_b)
tfidf = np.asarray(A.multiply(B).sum(1)).ravel()
# 3) in-swarm probe upper bound (stratified 5-fold on SwarmTraces, same layer)
p_in = np.zeros(len(yt)); Xt = T[f"L{best}"].astype(np.float32)
for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(Xt, yt):
    p_in[te] = mk().fit(Xt[tr], yt[tr]).predict_proba(Xt[te])[:, 1]
nt = np.array(["hard"] * len(yt))
out = {"model": model, "layer": best, "wiki_cv_auroc_at_layer": float(cv[best])}
for name, s in [("probe trained on wiki swarm (transfer)", p_transfer), ("verbal judge (zero-shot)", T["score"]),
                ("TF-IDF similarity (zero-shot)", tfidf), ("probe trained in-swarm (5-fold CV, upper bound)", p_in)]:
    r = scores(yt, s, nt, n_boot=1000)["all"]; out[name] = r
    print(f"{model} | {name:48s} AUROC={r['auroc']:.3f} [{r['ci'][0]:.3f},{r['ci'][1]:.3f}] TPR@1%={r['recall_at_fpr1']:.3f}")
print(f"layer {best} (wiki CV {cv[best]:.3f}); per-layer wiki CV:", {l: round(v, 3) for l, v in cv.items()})
# transfer at every layer (descriptive only; layer was NOT chosen on this)
out["transfer_by_layer"] = {}
for l in layers[1:]:
    c = mk().fit(W[f"L{l}"].astype(np.float32), yw)
    out["transfer_by_layer"][l] = float(roc_auc_score(yt, c.decision_function(T[f"L{l}"].astype(np.float32))))
print("transfer AUROC by layer (descriptive):", {l: round(v, 3) for l, v in out["transfer_by_layer"].items()})
json.dump(out, open(ROOT / "results" / f"transfer_{model}.json", "w"), indent=1, default=float)
