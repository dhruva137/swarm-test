"""Main experiment: which log fields let a monitor link coordinated edits across agents?

Outputs results/eval.json and results/pairs.parquet.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).parent))
from pairs import build_pairs
from features import pair_features, FIELD_OF

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"; RES.mkdir(exist_ok=True)
FIELDS = ["time", "identity", "network", "location", "content"]


def cv_predict(X, y, groups, seed=0):
    p = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=seed)
        m.fit(X.iloc[tr], y[tr])
        p[te] = m.predict_proba(X.iloc[te])[:, 1]
    return p


def scores(y, p, neg_type, n_boot=200, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for name, mask in [("all", np.ones(len(y), bool)), ("hard", neg_type != "easy")]:
        yy, pp = y[mask], p[mask]
        boots = []
        for _ in range(n_boot):
            i = rng.integers(0, len(yy), len(yy))
            if yy[i].min() != yy[i].max():
                boots.append(roc_auc_score(yy[i], pp[i]))
        out[name] = dict(auroc=roc_auc_score(yy, pp), ci=[float(np.quantile(boots, .025)), float(np.quantile(boots, .975))],
                         ap=average_precision_score(yy, pp), recall_at_fpr1=tpr_at_fpr(yy, pp, 0.01))
    return out


def tpr_at_fpr(y, p, target=0.01):
    """Recall (TPR) at a fixed false-positive rate, read off the full ROC curve by linear interpolation.
    Linear interpolation inside a tied-score segment equals the expected TPR under random tie-breaking,
    so saturated scores (e.g. many LLM outputs at exactly 1.0) are handled correctly."""
    fpr, tpr, _ = roc_curve(y, p, drop_intermediate=False)
    return float(np.interp(target, fpr, tpr))


def main():
    df = pd.read_parquet(ROOT / "data" / "edits.parquet")
    emb_path = ROOT / "data" / "emb_bge.npy"
    emb = np.load(emb_path) if emb_path.exists() else None
    pairs, n_urls = build_pairs(df)
    print("pairs", pairs.neg_type.value_counts().to_dict(), "defining urls", n_urls)
    y = pairs.y.values; nt = pairs.neg_type.values
    day = (df.set_index("eid").time.astype("int64") // 10**9 // 86400)
    groups = day.values[pairs.a.values]  # split folds by day so linked pairs do not leak across folds
    results = {"n_pairs": pairs.neg_type.value_counts().to_dict(), "n_defining_urls": n_urls}

    F = pair_features(df, pairs, emb)
    pairs.join(F).to_parquet(RES / "pairs.parquet")
    feats = list(F.columns)
    results["single_feature"] = {f: scores(y, F[f].values, nt, n_boot=50) for f in feats}

    conds = {"all fields": feats}
    for fl in FIELDS:
        conds[f"drop {fl}"] = [f for f in feats if FIELD_OF[f] != fl]
        conds[f"only {fl}"] = [f for f in feats if FIELD_OF[f] == fl]
    conds["metadata only (gateway view)"] = [f for f in feats if FIELD_OF[f] != "content"]
    conds["content only (SwarmTraces-like)"] = [f for f in feats if FIELD_OF[f] == "content"]
    conds["content + location"] = [f for f in feats if FIELD_OF[f] in ("content", "location")]
    results["conditions"] = {}
    for name, cols in conds.items():
        p = cv_predict(F[cols], y, groups)
        results["conditions"][name] = scores(y, p, nt)
        r = results["conditions"][name]
        print(f"{name:34s} AUROC all={r['all']['auroc']:.3f} hard={r['hard']['auroc']:.3f} R@FPR1%={r['hard']['recall_at_fpr1']:.3f}")

    # clock resolution sweep (all other fields kept)
    results["time_resolution"] = {}
    for label, res in [("1 s", None), ("1 min", 60), ("1 h", 3600), ("6 h", 21600), ("1 day", 86400), ("none", "none")]:
        if res == "none":
            cols = conds["drop time"]; Fr = F
        else:
            Fr = pair_features(df, pairs, emb, time_res=res); cols = feats
        p = cv_predict(Fr[cols], y, groups)
        results["time_resolution"][label] = scores(y, p, nt)
        print("clock", label, round(results["time_resolution"][label]["hard"]["auroc"], 3))
    json.dump(results, open(RES / "eval.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
