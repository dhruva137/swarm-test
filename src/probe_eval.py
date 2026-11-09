"""White-box: does Qwen3-8B 'know' linkage internally even though its YES/NO answer is poor?

Linear probes (logistic regression) on residual-stream activations at the answer token, per layer.
Folds are grouped so that no edit appears in both train and test (connected components of the pair graph).
Compared on identical pairs with: the model's own verbal logit (YES-NO), embedding cosine, and the cheap monitor.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from scipy.sparse.csgraph import connected_components
from scipy.sparse import coo_matrix
sys.path.insert(0, str(Path(__file__).parent))
from run_eval import scores

ROOT = Path(__file__).resolve().parents[1]
S = pd.read_parquet(ROOT / "results" / "stack_pairs.parquet").set_index("pair_idx")
out = {}
for tag in ["meta", "content"]:
    f = ROOT / "results" / "k_swarm-probe-acts" / f"acts_{tag}.npz"
    if not f.exists():
        continue
    Z = np.load(f)
    idx = Z["pair_idx"]; y = Z["y"]; nt = Z["neg_type"]
    sub = S.loc[idx]
    # group folds by connected component of shared edits (no edit leaks across folds)
    nodes = {e: i for i, e in enumerate(sorted(set(sub.a) | set(sub.b)))}
    m = coo_matrix((np.ones(len(sub)), ([nodes[a] for a in sub.a], [nodes[b] for b in sub.b])), shape=(len(nodes),) * 2)
    _, comp = connected_components(m, directed=False)
    groups = comp[[nodes[a] for a in sub.a]]
    res = {"verbal_logit": scores(y, Z["logit_diff"], nt, n_boot=300)}
    layers = sorted(int(k[1:]) for k in Z.files if k.startswith("L"))
    for l in layers:
        X = Z[f"L{l}"].astype(np.float32)
        p = np.zeros(len(y))
        for tr, te in GroupKFold(5).split(X, y, groups):
            clf = make_pipeline(StandardScaler(), PCA(256, random_state=0), LogisticRegression(C=0.05, max_iter=2000))
            p[te] = clf.fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        res[f"probe_L{l}"] = scores(y, p, nt, n_boot=300)
        print(tag, f"L{l:2d} probe hard AUROC={res[f'probe_L{l}']['hard']['auroc']:.3f} R@1%={res[f'probe_L{l}']['hard']['recall_at_fpr1']:.3f}", flush=True)
    for c in ["cheap", "emb_cos"]:
        res[c] = scores(y, sub[c].values, nt, n_boot=300)
    res["n_groups"] = int(len(set(groups)))
    v = res["verbal_logit"]["hard"]; best = max((k for k in res if k.startswith("probe")), key=lambda k: res[k]["hard"]["auroc"])
    print(tag, "verbal", round(v["auroc"], 3), "| best", best, round(res[best]["hard"]["auroc"], 3),
          "| cheap", round(res["cheap"]["hard"]["auroc"], 3), "| emb", round(res["emb_cos"]["hard"]["auroc"], 3), "| groups", res["n_groups"])
    out[tag] = res
json.dump(out, open(ROOT / "results" / "probe.json", "w"), indent=1, default=float)
