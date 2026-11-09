"""How is swarm linkage encoded inside an LLM monitor? (real activations, Qwen3-1.7B and 4B, 1,500 v2 pairs)

M1 Linearity: per layer, out-of-fold (5 time blocks) hard-negative AUROC of
     linear  : logistic regression on PCA-128 of standardized activations
     MLP     : 1 hidden layer (256, ReLU, early stopping) on the same PCA-128
     quad    : logistic regression on PCA-24 + all pairwise products (degree-2 polynomial)
   PCA and scalers are fit on training blocks only.
M2 Subspace rank (INLP, Ravfogel et al. 2020 style): inside each training fold, repeatedly fit a linear direction
   (PCA-256 space), project it out of train and test, and re-measure held-out AUROC with a fresh linear probe AND an
   MLP. Control: remove the same number of random directions. Rank = number of removals until linear AUROC < 0.55.
M3 Decision entropy of the verbal judge: mean binary entropy of P(YES|YES or NO) per pair, and its spread.
"""
import json, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
KD = ROOT / "results" / "k_swarm-probe-acts"
S = pd.read_parquet(ROOT / "results" / "v2_sample.parquet").sort_values("sid").reset_index(drop=True)
y, fold = S.y.values, S.fold.values
hard = S.neg_type.values != "easy"
rng = np.random.default_rng(0)


def auc_hard_fold(p, te):
    m = te & hard
    return roc_auc_score(y[m], p[m])


def oof(make, X):
    p = np.zeros(len(y))
    for k in range(5):
        tr, te = fold != k, fold == k
        p[te] = make().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return float(np.mean([auc_hard_fold(p, fold == k) for k in range(5)]))


lin = lambda: make_pipeline(StandardScaler(), PCA(128, random_state=0), LogisticRegression(C=0.05, max_iter=3000))
mlp = lambda: make_pipeline(StandardScaler(), PCA(128, random_state=0),
                            MLPClassifier(hidden_layer_sizes=(256,), alpha=1e-2, early_stopping=True, max_iter=400, random_state=0))
quad = lambda: make_pipeline(StandardScaler(), PCA(24, random_state=0), PolynomialFeatures(2, include_bias=False),
                             StandardScaler(), LogisticRegression(C=0.01, max_iter=4000))


def inlp_curve(X, K=16, random=False):
    lin_auc = np.zeros((5, K + 1)); mlp_auc = np.zeros((5, K + 1))
    for k in range(5):
        tr, te = fold != k, fold == k
        sc = StandardScaler().fit(X[tr]); pca = PCA(256, random_state=0).fit(sc.transform(X[tr]))
        Ztr, Zte = pca.transform(sc.transform(X[tr])), pca.transform(sc.transform(X[te]))
        Ztr = Ztr / Ztr[:, :1].std() if False else Ztr
        for r in range(K + 1):
            for arr, model in [(lin_auc, LogisticRegression(C=0.05, max_iter=3000)),
                               (mlp_auc, MLPClassifier(hidden_layer_sizes=(256,), alpha=1e-2, early_stopping=True, max_iter=400, random_state=0))]:
                s = StandardScaler().fit(Ztr)
                pr = model.fit(s.transform(Ztr), y[tr]).predict_proba(s.transform(Zte))[:, 1]
                yy, hh = y[te], hard[te]
                arr[k, r] = roc_auc_score(yy[hh], pr[hh])
            if r == K:
                break
            if random:
                d = rng.standard_normal(Ztr.shape[1])
            else:
                s = StandardScaler().fit(Ztr)
                d = LogisticRegression(C=0.05, max_iter=3000).fit(s.transform(Ztr), y[tr]).coef_[0] / s.scale_
            d = d / np.linalg.norm(d)
            Ztr = Ztr - np.outer(Ztr @ d, d); Zte = Zte - np.outer(Zte @ d, d)
    return lin_auc.mean(0).tolist(), mlp_auc.mean(0).tolist()


out = {}
for model in ["Qwen3-1.7B", "Qwen3-4B"]:
    for style in ["neutral_meta", "primed_meta"]:
        f = KD / f"{model}_{style}.npz"
        if not f.exists():
            continue
        Z = np.load(f); assert (Z["sid"] == S.sid.values).all()
        layers = sorted(int(k[1:]) for k in Z.files if k.startswith("L") and k[1:].isdigit())[1:]
        res = {"layers": layers, "linear": [], "mlp": [], "quad": []}
        for l in layers:
            X = Z[f"L{l}"].astype(np.float32)
            res["linear"].append(oof(lin, X)); res["mlp"].append(oof(mlp, X)); res["quad"].append(oof(quad, X))
            print(f"{model} {style} L{l:2d}: linear {res['linear'][-1]:.3f}  MLP {res['mlp'][-1]:.3f}  quad {res['quad'][-1]:.3f}", flush=True)
        # best layer by linear OOF (same rule for all probes, reported as such)
        best = layers[int(np.argmax(res["linear"]))]
        X = Z[f"L{best}"].astype(np.float32)
        li, ml = inlp_curve(X, K=16); lr_, mr_ = inlp_curve(X, K=16, random=True)
        res.update(inlp_layer=best, inlp_linear=li, inlp_mlp=ml, rand_linear=lr_, rand_mlp=mr_,
                   inlp_rank_linear_below_055=next((i for i, v in enumerate(li) if v < 0.55), None))
        print(f"   INLP@L{best} linear: {np.round(li, 3)}")
        print(f"   INLP@L{best} MLP   : {np.round(ml, 3)}")
        print(f"   random   linear: {np.round(lr_, 3)}")
        p = 1 / (1 + np.exp(-Z["score"]))
        H = -(p * np.log2(p + 1e-12) + (1 - p) * np.log2(1 - p + 1e-12))
        res.update(entropy_mean_bits=float(H.mean()), entropy_pos=float(H[y == 1].mean()), entropy_neg=float(H[y == 0].mean()),
                   p_yes_mean_neg=float(p[y == 0].mean()))
        print(f"   decision entropy {H.mean():.3f} bits; P(YES) on unlinked {p[y == 0].mean():.3f}", flush=True)
        out[f"{model}|{style}"] = res
json.dump(out, open(ROOT / "results" / "mech.json", "w"), indent=1, default=float)
