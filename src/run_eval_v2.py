"""v2 main experiment (audit-corrected).

CV: leave-one-time-block-out over K=5 contiguous blocks; pairs never cross blocks, so train/test share no edit.
Metrics are computed per test fold and macro-averaged (no pooling of scores across folds).
CIs: cluster bootstrap over agents (agent of edit a) within each fold, 300 resamples.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(Path(__file__).parent))
from run_eval import tpr_at_fpr
from pairs_v2 import build_pairs_v2
from features_v2 import pair_features_v2, FIELD_OF

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIELDS = ["time", "identity", "network", "location", "content"]


def fold_metrics(y, p, nt):
    hard = nt != "easy"
    return dict(auroc_all=roc_auc_score(y, p), auroc_hard=roc_auc_score(y[hard], p[hard]),
                tpr1_hard=tpr_at_fpr(y[hard], p[hard], 0.01))


def evaluate(P, p, n_boot=300, seed=0):
    """Macro-average over folds + agent-cluster bootstrap CI."""
    rng = np.random.default_rng(seed)
    y, nt, fold, ag = P.y.values, P.neg_type.values, P.fold.values, P.agent_a.values
    folds = sorted(set(fold))
    idx = {k: np.where(fold == k)[0] for k in folds}
    clusters = {k: pd.Series(idx[k]).groupby(ag[idx[k]]).apply(np.array).values for k in folds}
    per = [fold_metrics(y[idx[k]], p[idx[k]], nt[idx[k]]) for k in folds]
    out = {m: float(np.mean([d[m] for d in per])) for m in per[0]}
    out["per_fold_auroc_hard"] = [d["auroc_hard"] for d in per]
    boots = {m: [] for m in per[0]}
    for _ in range(n_boot):
        ds = []
        for k in folds:
            cl = clusters[k]
            ii = np.concatenate([cl[j] for j in rng.integers(0, len(cl), len(cl))])
            yy = y[ii]
            if yy.min() == yy.max() or (yy[nt[ii] != "easy"].min() == yy[nt[ii] != "easy"].max()):
                continue
            ds.append(fold_metrics(yy, p[ii], nt[ii]))
        for m in boots:
            boots[m].append(np.mean([d[m] for d in ds]))
    for m in boots:
        out[f"{m}_ci"] = [float(np.quantile(boots[m], .025)), float(np.quantile(boots[m], .975))]
    return out


def cv_predict(P, cols, train_mask=None):
    p = np.full(len(P), np.nan)
    for k in sorted(P.fold.unique()):
        te = P.fold.values == k
        tr = ~te if train_mask is None else (~te & train_mask)
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0)
        m.fit(P.loc[tr, cols], P.y.values[tr])
        p[te] = m.predict_proba(P.loc[te, cols])[:, 1]
    return p


def conditions(feats):
    c = {"all fields": feats}
    for fl in FIELDS:
        c[f"drop {fl}"] = [f for f in feats if FIELD_OF[f] != fl]
        c[f"only {fl}"] = [f for f in feats if FIELD_OF[f] == fl]
    c["metadata only (gateway view)"] = [f for f in feats if FIELD_OF[f] != "content"]
    c["content only (SwarmTraces-like)"] = [f for f in feats if FIELD_OF[f] == "content"]
    return c


def show(name, r):
    print(f"{name:36s} hard AUROC={r['auroc_hard']:.3f} [{r['auroc_hard_ci'][0]:.3f},{r['auroc_hard_ci'][1]:.3f}] "
          f"TPR@1%={r['tpr1_hard']:.3f}  all={r['auroc_all']:.3f}  folds={np.round(r['per_fold_auroc_hard'],3)}", flush=True)


def main():
    df = pd.read_parquet(ROOT / "data" / "edits.parquet")
    emb = np.load(ROOT / "data" / "emb_bge_text_nourl.npy")
    emb_s = np.load(ROOT / "data" / "emb_bge_text_nourl_strict.npy")
    assert len(emb) == len(df) == len(emb_s)
    agent = np.where(df.label != "", df.label, "anon:" + df.ip16.fillna("?"))
    R = {}

    def run_variant(name, P, text_col="text_nourl", E=emb, sweep=False, extra_filter=None):
        P = P.copy(); P["agent_a"] = agent[P.a.values]; P["agent_b"] = agent[P.b.values]
        F = pair_features_v2(df, P, E, text_col=text_col)
        P = P.join(F)
        if extra_filter is not None:
            P = extra_filter(P).reset_index(drop=True)
        feats = list(F.columns)
        out = {"n": P.neg_type.value_counts().to_dict(), "per_fold_n": P.groupby("fold").size().to_dict()}
        print(f"\n=== {name}: {out['n']} ===", flush=True)
        out["single_feature"] = {f: evaluate(P, P[f].values, n_boot=100) for f in feats}
        out["conditions"] = {}
        for cname, cols in conditions(feats).items():
            out["conditions"][cname] = evaluate(P, cv_predict(P, cols)); show(cname, out["conditions"][cname])
        if sweep:
            out["time_resolution"] = {}
            for lab, res in [("1 s", None), ("1 min", 60), ("1 h", 3600), ("6 h", 21600), ("1 day", 86400), ("none", "none")]:
                Fr = pair_features_v2(df, P[["a", "b"]], E, text_col=text_col, time_res=None if res == "none" else res,
                                      drop_time=(res == "none"))
                Pr = P[["a", "b", "y", "neg_type", "fold", "agent_a", "agent_b"]].join(Fr)
                out["time_resolution"][lab] = evaluate(Pr, cv_predict(Pr, list(Fr.columns)))
                show(f"clock {lab}", out["time_resolution"][lab])
            # agent-disjoint: train only on pairs whose agents never appear in the test fold
            pa = np.zeros(len(P))
            for k in sorted(P.fold.unique()):
                te = P.fold.values == k
                test_agents = set(P.agent_a[te]) | set(P.agent_b[te])
                tr = ~te & ~P.agent_a.isin(test_agents).values & ~P.agent_b.isin(test_agents).values
                out.setdefault("agent_disjoint_train_sizes", []).append(int(tr.sum()))
            out["agent_disjoint"] = {}
            for cname in ["all fields", "content only (SwarmTraces-like)", "metadata only (gateway view)"]:
                cols = conditions(feats)[cname]
                p = np.full(len(P), np.nan)
                for k in sorted(P.fold.unique()):
                    te = P.fold.values == k
                    test_agents = set(P.agent_a[te]) | set(P.agent_b[te])
                    tr = ~te & ~P.agent_a.isin(test_agents).values & ~P.agent_b.isin(test_agents).values
                    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0).fit(P.loc[tr, cols], P.y.values[tr])
                    p[te] = m.predict_proba(P.loc[te, cols])[:, 1]
                out["agent_disjoint"][cname] = evaluate(P, p); show("agent-disjoint " + cname, out["agent_disjoint"][cname])
            P.to_parquet(RES / "pairs_v2.parquet")
        R[name] = out

    P0, n_urls = build_pairs_v2(df)
    R["n_defining_urls"] = n_urls
    run_variant("main", P0, sweep=True)
    run_variant("strict mask", P0, text_col="text_nourl_strict", E=emb_s)
    # near-copy removal applied to BOTH classes (all pairs with emb cosine > 0.95 dropped)
    run_variant("near-copies removed (both classes)", P0, extra_filter=lambda P: P[P.emb_cos <= 0.95])
    for lo, hi in [(2, 10), (2, 200)]:
        Pv, _ = build_pairs_v2(df, min_ag=lo, max_ag=hi, seed=3)
        run_variant(f"GT {lo}-{hi} agents", Pv)
    json.dump(R, open(RES / "eval_v2.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
