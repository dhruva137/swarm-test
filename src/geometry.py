"""Geometry of 'saying' vs 'knowing' inside an LLM swarm monitor (Qwen3-1.7B, real activations).

Objects (residual stream at the answer position, same 1,500 pairs under two prompts N=neutral, P=collusion-framed):
  answer axis a   = mean unembedding(YES variants) - mean unembedding(NO variants)  (final, post-norm layer)
  linkage axis w_l = logistic-regression direction separating linked/unlinked pairs at layer l (raw residual units)
CHECK 1: final-layer hidden . a must reproduce the model's recorded verbal score (corr ~ 1).
Tests:
  T1 framing shift: D = mean_pairs(h_P - h_N). Report cos(D, a), cos(D, w), and |cos| for random directions.
  T2 say-know plane: per pair x = h . w_N (linkage coordinate, w fitted on N training blocks), y = h . a.
     AUROC of x under P using the N-fitted w (does the linkage axis survive framing?), AUROC of y (= verbal).
  T3 depth: cosine between linkage axes of all layer pairs, N vs P; label-shuffled null.
  T4 swarms: cosine between wiki-swarm and attack-swarm linkage axes per layer (same prompt), vs shuffled null.
All AUROCs out-of-fold (5 time blocks). Nulls: 50 label permutations.
"""
import json, glob
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from safetensors import safe_open
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoConfig

ROOT = Path(__file__).resolve().parents[1]
MODEL = "Qwen/Qwen3-1.7B"
KD = ROOT / "results" / "k_swarm-probe-acts"
N = np.load(KD / "Qwen3-1.7B_neutral_meta.npz"); P = np.load(KD / "Qwen3-1.7B_primed_meta.npz")
PC = np.load(KD / "Qwen3-1.7B_primed_content.npz")
S = pd.read_parquet(ROOT / "results" / "v2_sample.parquet").sort_values("sid").reset_index(drop=True)
for Z in (N, P, PC):
    assert (Z["sid"] == S.sid.values).all()
y, fold = S.y.values, S.fold.values
hard = S.neg_type.values != "easy"
layers = sorted(int(k[1:]) for k in N.files if k.startswith("L") and k[1:].isdigit())
cfg = AutoConfig.from_pretrained(MODEL); Lfin = cfg.num_hidden_layers
assert Lfin in layers, "final layer not saved"

# ---- answer axis from the (tied) embedding / lm_head
tok = AutoTokenizer.from_pretrained(MODEL)
yes = [tok.encode(w, add_special_tokens=False)[0] for w in [" YES", "YES", " Yes", "Yes"]]
no = [tok.encode(w, add_special_tokens=False)[0] for w in [" NO", "NO", " No", "No"]]
idx = json.load(open(hf_hub_download(MODEL, "model.safetensors.index.json"))) if True else None
wm = idx["weight_map"]
key = "lm_head.weight" if "lm_head.weight" in wm else "model.embed_tokens.weight"
with safe_open(hf_hub_download(MODEL, wm[key]), "pt") as f:
    W = f.get_tensor(key).float().numpy()
a = W[yes].mean(0) - W[no].mean(0)
out = {"weight_used_for_answer_axis": key}

# CHECK 1: does final-layer state . a reproduce the recorded verbal score?
for name, Z in [("neutral", N), ("primed", P)]:
    h = Z[f"L{Lfin}"].astype(np.float32)
    lg = h @ W[yes + no].T
    lse = lambda v: np.log(np.exp(v - v.max(1, keepdims=True)).sum(1)) + v.max(1)
    rec = lse(lg[:, :4]) - lse(lg[:, 4:])
    out[f"check_{name}"] = dict(corr_exact_logsumexp=float(np.corrcoef(rec, Z["score"])[0, 1]),
                                max_abs_err=float(np.abs(rec - Z["score"]).max()),
                                corr_axis=float(np.corrcoef(h @ a, Z["score"])[0, 1]))
    print(f"CHECK {name}: recomputed score vs recorded r={out[f'check_{name}']['corr_exact_logsumexp']:.5f} "
          f"max|err|={out[f'check_{name}']['max_abs_err']:.3f}; (h.a vs score r={out[f'check_{name}']['corr_axis']:.4f})")


def fit_dir(X, yy):
    sc = StandardScaler().fit(X)
    m = LogisticRegression(C=0.05, max_iter=3000).fit(sc.transform(X), yy)
    w = m.coef_[0] / sc.scale_
    return w / np.linalg.norm(w)


def oof_proj(Xfit, Xapply, yy):
    """Out-of-fold linkage coordinate: direction fitted on 4 blocks of Xfit, projected for the 5th block of Xapply."""
    s = np.zeros(len(yy))
    for k in range(5):
        tr, te = fold != k, fold == k
        s[te] = Xapply[te] @ fit_dir(Xfit[tr], yy[tr])
    return s


def auc_hard(s):
    return float(np.mean([roc_auc_score(y[(fold == k) & hard], s[(fold == k) & hard]) for k in range(5)]))


rng = np.random.default_rng(0)
hN, hP = N[f"L{Lfin}"].astype(np.float32), P[f"L{Lfin}"].astype(np.float32)
# ---- T1 framing shift
D = (hP - hN).mean(0)
wN = fit_dir(hN, y)
ua = a / np.linalg.norm(a)
rand = rng.standard_normal((2000, len(D))); rand /= np.linalg.norm(rand, axis=1, keepdims=True)
cosD = lambda v: float(D @ v / np.linalg.norm(D) / np.linalg.norm(v))
out["T1"] = dict(cos_shift_answer=cosD(ua), cos_shift_linkage=cosD(wN), cos_answer_linkage=float(ua @ wN),
                 random_abs_cos_p99=float(np.quantile(np.abs(rand @ (D / np.linalg.norm(D))), .99)),
                 shift_along_answer=float(D @ ua), shift_along_linkage=float(D @ wN),
                 sd_answer_coord_N=float((hN @ ua).std()), sd_linkage_coord_N=float((hN @ wN).std()))
t = out["T1"]
print(f"T1 framing shift: cos(D,answer)={t['cos_shift_answer']:+.3f}  cos(D,linkage)={t['cos_shift_linkage']:+.3f}  "
      f"random |cos| p99={t['random_abs_cos_p99']:.3f}; shift along answer = {t['shift_along_answer']/t['sd_answer_coord_N']:+.2f} SD, "
      f"along linkage = {t['shift_along_linkage']/t['sd_linkage_coord_N']:+.2f} SD; cos(answer,linkage)={t['cos_answer_linkage']:+.3f}")

# ---- T2 say-know plane (final layer), all out-of-fold
xN = oof_proj(hN, hN, y); xP = oof_proj(hN, hP, y)            # linkage axis fitted on NEUTRAL, applied to both
yN, yP = hN @ ua, hP @ ua
out["T2"] = dict(auc_linkage_axis_neutral=auc_hard(xN), auc_linkage_axis_primed_using_neutral_axis=auc_hard(xP),
                 auc_answer_axis_neutral=auc_hard(yN), auc_answer_axis_primed=auc_hard(yP),
                 auc_verbal_neutral=auc_hard(N["score"]), auc_verbal_primed=auc_hard(P["score"]),
                 corr_linkage_coord_N_vs_P=float(np.corrcoef(xN, xP)[0, 1]),
                 corr_answer_coord_N_vs_P=float(np.corrcoef(yN, yP)[0, 1]))
print("T2", {k: round(v, 3) for k, v in out["T2"].items()})
np.savez(ROOT / "results" / "geometry_plane.npz", xN=xN, xP=xP, yN=yN, yP=yP, y=y, hard=hard)

# ---- T3 depth: linkage axes per layer, N and P; cross cosines + shuffled null
dirsN = {l: fit_dir(N[f"L{l}"].astype(np.float32), y) for l in layers[1:]}
dirsP = {l: fit_dir(P[f"L{l}"].astype(np.float32), y) for l in layers[1:]}
Ls = layers[1:]
M = np.array([[dirsN[i] @ dirsP[j] for j in Ls] for i in Ls])
MNN = np.array([[dirsN[i] @ dirsN[j] for j in Ls] for i in Ls])
null = []
for _ in range(50):
    yp = rng.permutation(y)
    l = Ls[len(Ls) // 2]
    null.append(abs(fit_dir(N[f"L{l}"].astype(np.float32), yp) @ fit_dir(P[f"L{l}"].astype(np.float32), rng.permutation(y))))
out["T3"] = dict(layers=Ls, cos_N_P=M.tolist(), cos_N_N=MNN.tolist(), diag_N_P=[float(M[i, i]) for i in range(len(Ls))],
                 null_abs_cos_p99=float(np.quantile(null, .99)),
                 auc_oof_by_layer_N=[auc_hard(oof_proj(N[f"L{l}"].astype(np.float32), N[f"L{l}"].astype(np.float32), y)) for l in Ls],
                 auc_oof_by_layer_P_from_N=[auc_hard(oof_proj(N[f"L{l}"].astype(np.float32), P[f"L{l}"].astype(np.float32), y)) for l in Ls])
print("T3 diag cos(N,P) by layer", dict(zip(Ls, np.round(out['T3']['diag_N_P'], 3))), "null p99", round(out["T3"]["null_abs_cos_p99"], 3))
print("T3 OOF AUROC N", dict(zip(Ls, np.round(out['T3']['auc_oof_by_layer_N'], 3))))
print("T3 OOF AUROC P using N axis", dict(zip(Ls, np.round(out['T3']['auc_oof_by_layer_P_from_N'], 3))))

# ---- T4 swarms: wiki (primed, content) vs attack swarm (primed, content), same prompt and model
st = ROOT / "results" / "local_st" / "ST_Qwen3-1.7B_primed_content.npz"
if st.exists():
    T = np.load(st)
    yt = np.array([json.loads(l)["y"] for l in open(ROOT / "kaggle" / "st_data" / "pairs_sid.jsonl", encoding="utf-8")])
    rows = []
    for l in Ls:
        dw = fit_dir(PC[f"L{l}"].astype(np.float32), y); ds = fit_dir(T[f"L{l}"].astype(np.float32), yt)
        nl = [abs(fit_dir(PC[f"L{l}"].astype(np.float32), rng.permutation(y)) @ fit_dir(T[f"L{l}"].astype(np.float32), rng.permutation(yt))) for _ in range(20)]
        rows.append(dict(layer=l, cos_wiki_attack=float(dw @ ds), null_p99=float(np.quantile(nl, .99)),
                         transfer_auc=float(roc_auc_score(yt, T[f"L{l}"].astype(np.float32) @ dw))))
    out["T4"] = rows
    for r in rows:
        print(f"T4 layer {r['layer']:2d}: cos(wiki axis, attack axis)={r['cos_wiki_attack']:+.3f} (null p99 {r['null_p99']:.3f}); wiki-axis AUROC on attack swarm {r['transfer_auc']:.3f}")
json.dump(out, open(ROOT / "results" / "geometry.json", "w"), indent=1, default=float)
