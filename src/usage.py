"""Scale-free 'usage' of a direction by the verdict (fixes the per-SD vs per-norm ambiguity of the steering ratio).

usage(d) = [coupling(d) per SD] * [linked-minus-unlinked gap of the projection on d, in SD] / [verdict's own
           linked-minus-unlinked gap in log-odds]
i.e. if a pair were moved all the way from the unlinked mean to the linked mean along d, what fraction of the
verdict's actual linked/unlinked difference would that produce?  ~1 for a direction the verdict relies on fully,
~0 for a direction it ignores. Gaps use hard negatives on the fitting pairs.
Also: per-unit-norm ratio verbal/linkage (coupling / SD).
For Qwen3-1.7B the directions are re-fitted exactly as in src/steer3.py (deterministic) and checked against the SDs
stored in results/steer3.json. For Qwen3-4B the gaps come from the Kaggle job (results/k_swarm-steer4b/steer4b.json).
"""
import json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
out = {}
# ---------- Qwen3-1.7B (re-fit, must reproduce steer3 SDs)
S = pd.read_parquet(ROOT / "results" / "v2_sample.parquet").sort_values("sid").reset_index(drop=True)
A = {st: np.load(ROOT / "results" / "k_swarm-probe-acts" / f"Qwen3-1.7B_{st}_meta.npz") for st in ["neutral", "primed"]}
y = S.y.values; L = 20
sel = pd.concat([S[S.y == 1].sample(60, random_state=11), S[S.neg_type == "hard"].sample(60, random_state=11)])
fit = ~S.index.isin(sel.index)
hardfit = fit & ((y == 1) | (S.neg_type.values == "hard"))
J3 = json.load(open(ROOT / "results" / "steer3.json"))
rows3 = pd.read_csv(ROOT / "results" / "steer3_rows.csv")
unit = lambda v: v / np.linalg.norm(v)
rng = np.random.default_rng(1)
res17 = {}
for st in ["neutral", "primed"]:
    X = A[st][f"L{L}"].astype(np.float32); sc_ = A[st]["score"]
    Xf = X[fit]; s = StandardScaler().fit(Xf)
    dl = unit(LogisticRegression(C=0.05, max_iter=3000).fit(s.transform(Xf), y[fit]).coef_[0] / s.scale_)
    dv = unit(Ridge(alpha=1000.0).fit(s.transform(Xf), sc_[fit]).coef_ / s.scale_)
    sd_l, sd_v = float((Xf @ dl).std()), float((Xf @ dv).std())
    chk = {"linkage": (sd_l, J3["sd"][f"{st}/linkage"]), "verbal": (sd_v, J3["sd"][f"{st}/verbal"])}
    for k, (a, b) in chk.items():
        assert abs(a - b) / b < 1e-3, f"refit SD mismatch {st}/{k}: {a} vs {b}"
    vgap = float(sc_[hardfit & (y == 1)].mean() - sc_[hardfit & (y == 0)].mean())
    gl = float(((X @ dl)[hardfit & (y == 1)].mean() - (X @ dl)[hardfit & (y == 0)].mean()) / sd_l)
    gv = float(((X @ dv)[hardfit & (y == 1)].mean() - (X @ dv)[hardfit & (y == 0)].mean()) / sd_v)
    r = {}
    for name, g, sdd in [("linkage", gl, sd_l), ("verbal", gv, sd_v)]:
        cpl = rows3[(rows3.prompt == st) & (rows3.direction == name) & (rows3.alpha == 2.0)].coupling.values
        bs = [rng.choice(cpl, len(cpl)).mean() * g / vgap for _ in range(2000)]
        r[name] = dict(gap_sd=g, coupling_per_sd=float(cpl.mean()), usage=float(cpl.mean() * g / vgap),
                       usage_ci=[float(np.quantile(bs, .025)), float(np.quantile(bs, .975))],
                       coupling_per_unit_norm=float(cpl.mean() / sdd), sd=sdd)
    r["verdict_gap_logodds"] = vgap
    r["per_norm_ratio_verbal_over_linkage"] = r["verbal"]["coupling_per_unit_norm"] / r["linkage"]["coupling_per_unit_norm"]
    res17[st] = r
    print(f"1.7B {st:8s} verdict gap {vgap:+.2f} log-odds | linkage: gap {gl:.2f} SD, usage {r['linkage']['usage']:+.4f} {np.round(r['linkage']['usage_ci'], 4)} | "
          f"verbal: gap {gv:.2f} SD, usage {r['verbal']['usage']:+.3f} {np.round(r['verbal']['usage_ci'], 3)} | per-norm ratio {r['per_norm_ratio_verbal_over_linkage']:.0f}x")
out["Qwen3-1.7B"] = res17
# ---------- Qwen3-4B (from Kaggle job, if present)
p4 = ROOT / "results" / "k_swarm-steer4b" / "steer4b.json"
if p4.exists():
    J = json.load(open(p4)); R4 = pd.read_csv(ROOT / "results" / "k_swarm-steer4b" / "steer4b_rows.csv")
    res4 = {}
    for st in ["neutral", "primed"]:
        vgap = J["verdict_gap_logodds"][st]; r = {}
        for name in ["linkage", "verbal"]:
            g = J["gap_sd"][f"{st}/{name}"]; sdd = J["sd"][f"{st}/{name}"]
            cpl = R4[(R4.prompt == st) & (R4.direction == name) & (R4.alpha == 2.0)].coupling.values
            bs = [rng.choice(cpl, len(cpl)).mean() * g / vgap for _ in range(2000)]
            r[name] = dict(gap_sd=g, coupling_per_sd=float(cpl.mean()), usage=float(cpl.mean() * g / vgap),
                           usage_ci=[float(np.quantile(bs, .025)), float(np.quantile(bs, .975))],
                           coupling_per_unit_norm=float(cpl.mean() / sdd), sd=sdd)
        r["verdict_gap_logodds"] = vgap
        r["per_norm_ratio_verbal_over_linkage"] = r["verbal"]["coupling_per_unit_norm"] / r["linkage"]["coupling_per_unit_norm"] if r["linkage"]["coupling_per_unit_norm"] else None
        nulls = [R4[(R4.prompt == st) & (R4.direction == f"perm{i}")].coupling.mean() for i in range(5)]
        r["null_coupling_range"] = [float(min(nulls)), float(max(nulls))]
        res4[st] = r
        print(f"4B   {st:8s} verdict gap {vgap:+.2f} log-odds | linkage: gap {r['linkage']['gap_sd']:.2f} SD, coupling {r['linkage']['coupling_per_sd']:+.4f}, usage {r['linkage']['usage']:+.4f} {np.round(r['linkage']['usage_ci'], 4)} | "
              f"verbal: coupling {r['verbal']['coupling_per_sd']:+.3f}, usage {r['verbal']['usage']:+.3f} | nulls {np.round(r['null_coupling_range'], 4)}")
    out["Qwen3-4B"] = res4
json.dump(out, open(ROOT / "results" / "usage.json", "w"), indent=1)
