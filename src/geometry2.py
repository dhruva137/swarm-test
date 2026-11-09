"""Audit-corrected geometry (replaces the cosine-based T1/T3/T4 claims of geometry.py).

G1 Per-layer out-of-fold hard-negative AUROC of four readouts (Qwen3-1.7B and 4B):
     N-axis on N, N-axis on P (transfer across framing), P-axis on P, and each prompt's verbal score.
G2 Framing shift at the final layer in interpretable units, with pair-bootstrap CIs:
     shift of the answer coordinate (h.a) in neutral within-class SD; shift of the out-of-fold linkage coordinate in
     neutral within-class SD and as a fraction of the neutral linked-unlinked gap.
G3 Stability of the linkage coordinate across prompts vs a MATCHED control: the same N-vs-P correlation for
     out-of-fold axes fitted on label-permuted data (captures shared-text variance unrelated to linkage).
"""
import json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from safetensors import safe_open
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoConfig

ROOT = Path(__file__).resolve().parents[1]
S = pd.read_parquet(ROOT / "results" / "v2_sample.parquet").sort_values("sid").reset_index(drop=True)
y, fold = S.y.values, S.fold.values
hard = S.neg_type.values != "easy"
rng = np.random.default_rng(0)


def fit_dir(X, yy):
    sc = StandardScaler().fit(X)
    w = LogisticRegression(C=0.05, max_iter=3000).fit(sc.transform(X), yy).coef_[0] / sc.scale_
    return w / np.linalg.norm(w)


def oof(Xfit, Xapp, yy):
    s = np.zeros(len(yy))
    for k in range(5):
        tr, te = fold != k, fold == k
        s[te] = Xapp[te] @ fit_dir(Xfit[tr], yy[tr])
    return s


def auc(s):
    return float(np.mean([roc_auc_score(y[(fold == k) & hard], s[(fold == k) & hard]) for k in range(5)]))


out = {}
for model in ["Qwen3-1.7B", "Qwen3-4B"]:
    N = np.load(ROOT / "results" / "k_swarm-probe-acts" / f"{model}_neutral_meta.npz")
    P = np.load(ROOT / "results" / "k_swarm-probe-acts" / f"{model}_primed_meta.npz")
    assert (N["sid"] == S.sid.values).all() and (P["sid"] == S.sid.values).all()
    layers = sorted(int(k[1:]) for k in N.files if k.startswith("L") and k[1:].isdigit())[1:]
    g1 = {"layers": layers, "N_on_N": [], "N_on_P": [], "P_on_P": [], "verbal_N": auc(N["score"]), "verbal_P": auc(P["score"])}
    for l in layers:
        XN, XP = N[f"L{l}"].astype(np.float32), P[f"L{l}"].astype(np.float32)
        g1["N_on_N"].append(auc(oof(XN, XN, y))); g1["N_on_P"].append(auc(oof(XN, XP, y))); g1["P_on_P"].append(auc(oof(XP, XP, y)))
    print(model, "verbal N/P", round(g1["verbal_N"], 3), round(g1["verbal_P"], 3))
    for i, l in enumerate(layers):
        print(f"  L{l:2d}: N-axis on N {g1['N_on_N'][i]:.3f} | N-axis on P {g1['N_on_P'][i]:.3f} | P-axis on P {g1['P_on_P'][i]:.3f}")
    res = {"G1": g1}
    # G2/G3 at the final layer (post-norm, where the answer axis is defined)
    Lf = AutoConfig.from_pretrained(f"Qwen/{model}").num_hidden_layers
    XN, XP = N[f"L{Lf}"].astype(np.float32), P[f"L{Lf}"].astype(np.float32)
    tok = AutoTokenizer.from_pretrained(f"Qwen/{model}")
    yes = [tok.encode(w, add_special_tokens=False)[0] for w in [" YES", "YES", " Yes", "Yes"]]
    no = [tok.encode(w, add_special_tokens=False)[0] for w in [" NO", "NO", " No", "No"]]
    idx = json.load(open(hf_hub_download(f"Qwen/{model}", "model.safetensors.index.json")))["weight_map"]
    key = "lm_head.weight" if "lm_head.weight" in idx else "model.embed_tokens.weight"
    with safe_open(hf_hub_download(f"Qwen/{model}", idx[key]), "pt") as f:
        W = f.get_tensor(key).float().numpy()
    a = W[yes].mean(0) - W[no].mean(0); a /= np.linalg.norm(a)
    del W
    xN, xP = oof(XN, XN, y), oof(XN, XP, y)          # out-of-fold linkage coordinate (N-fitted axis)
    aN, aP = XN @ a, XP @ a
    wsd = lambda v: np.sqrt(0.5 * (v[y == 1].var() + v[y == 0].var()))
    def boot(f, n=1000):
        vals = [f(rng.integers(0, len(y), len(y))) for _ in range(n)]
        return [float(np.quantile(vals, .025)), float(np.quantile(vals, .975))]
    sh_a = lambda ii: (aP[ii] - aN[ii]).mean() / wsd(aN)
    sh_x = lambda ii: (xP[ii] - xN[ii]).mean() / wsd(xN)
    gap = (xN[y == 1].mean() - xN[y == 0].mean())
    sh_x_gap = lambda ii: (xP[ii] - xN[ii]).mean() / gap
    all_i = np.arange(len(y))
    g2 = dict(answer_shift_withinSD=float(sh_a(all_i)), answer_shift_ci=boot(sh_a),
              linkage_shift_withinSD=float(sh_x(all_i)), linkage_shift_ci=boot(sh_x),
              linkage_shift_frac_of_gap=float(sh_x_gap(all_i)), linkage_shift_frac_ci=boot(sh_x_gap),
              neutral_gap_withinSD=float(gap / wsd(xN)))
    # G3: stability vs matched permuted-label control
    r_real = float(np.corrcoef(xN, xP)[0, 1]); r_ans = float(np.corrcoef(aN, aP)[0, 1])
    r_perm = []
    for _ in range(20):
        yp = rng.permutation(y)
        sN, sP = np.zeros(len(y)), np.zeros(len(y))
        for k in range(5):
            tr, te = fold != k, fold == k
            d = fit_dir(XN[tr], yp[tr]); sN[te] = XN[te] @ d; sP[te] = XP[te] @ d
        r_perm.append(np.corrcoef(sN, sP)[0, 1])
    g3 = dict(r_linkage_N_vs_P=r_real, r_answer_N_vs_P=r_ans, r_permuted_axis_mean=float(np.mean(r_perm)),
              r_permuted_axis_p95=float(np.quantile(r_perm, .95)))
    res.update(G2=g2, G3=g3)
    print(" G2", {k: (round(v, 3) if isinstance(v, float) else [round(x, 3) for x in v]) for k, v in g2.items()})
    print(" G3", {k: round(v, 3) for k, v in g3.items()})
    out[model] = res
json.dump(out, open(ROOT / "results" / "geometry2.json", "w"), indent=1, default=float)
