"""Can the swarm's own words move the monitor's REPRESENTATION, or only its VERDICT?

Data: activations + verbal scores for 1,500 pairs under four injection conditions on post B
  none (no sentence), neutral (real neutral swarm sentence), benign (real exculpatory sentence), coord (real relay
  instruction); two prompts (neutral, collusion-framed); Qwen3-1.7B (results/local_inject) and Qwen3-4B
  (results/k_swarm-inject).
Readouts (both log-odds-like):
  verbal = model's YES-vs-NO log-odds (zero-shot)
  probe  = logistic probe trained ONLY on clean ('none') activations of the 4 training blocks, layer chosen by nested
           CV on clean training blocks, applied unchanged to every condition of the held-out block.
Each readout is expressed in units of its own pooled within-class SD under 'none' (hard pairs), so shifts are comparable.
Effects (paired, same pairs, no selection), agent-cluster bootstrap CIs (1,000):
  relay on unlinked (hard) pairs: coord - none, and coord - neutral
  exculpatory on linked pairs:   benign - none, and benign - neutral
  difference |verbal shift| - |probe shift|
AUROC (hard) of each readout under: none, neutral-sentence control, and attack (linked: benign, unlinked: coord).
"""
import json, glob
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
S = pd.read_parquet(ROOT / "results" / "v2_sample.parquet").sort_values("sid").reset_index(drop=True)
df = pd.read_parquet(ROOT / "data" / "edits.parquet", columns=["label", "ip16"])
lab = df.label.values; ipv = df.ip16.fillna("?").values
S["agent_a"] = np.where(lab[S.a.values] != "", lab[S.a.values], "anon:" + ipv[S.a.values])
y, fold = S.y.values, S.fold.values
hard = S.neg_type.values != "easy"
mk = lambda: make_pipeline(StandardScaler(), PCA(128, random_state=0), LogisticRegression(C=0.05, max_iter=3000))
clusters = S.groupby("agent_a").indices; ckeys = list(clusters)
rng = np.random.default_rng(0)
srcs = {"Qwen3-1.7B": ROOT / "results" / "local_inject", "Qwen3-4B": ROOT / "results" / "k_swarm-inject"}
CONDS = ["none", "neutral", "benign", "coord"]


def auc_macro(sc_pos, sc_neg):
    v = []
    for k in range(5):
        m = (fold == k) & hard
        yy = y[m]; s = np.where(yy == 1, sc_pos[m], sc_neg[m])
        v.append(roc_auc_score(yy, s))
    return float(np.mean(v))


def boot_mean(d, mask, n=1000):
    vals = []
    for _ in range(n):
        ii = np.concatenate([clusters[ckeys[j]] for j in rng.integers(0, len(ckeys), len(ckeys))]); ii = ii[mask[ii]]
        vals.append(d[ii].mean())
    return [float(np.quantile(vals, .025)), float(np.quantile(vals, .975))]


out = {}
for model, d in srcs.items():
    for style in ["neutral", "primed"]:
        files = {c: d / f"{model}_{style}_{c}.npz" for c in CONDS}
        if not all(f.exists() for f in files.values()):
            print("missing", model, style); continue
        Z = {c: np.load(f) for c, f in files.items()}
        for c in Z:
            assert (Z[c]["sid"] == S.sid.values).all(), (model, style, c)
        layers = sorted(int(k[1:]) for k in Z["none"].files if k.startswith("L") and k[1:].isdigit())[1:]
        probe = {c: np.zeros(len(S)) for c in CONDS}; chosen = []
        for k in range(5):
            trf = [j for j in range(5) if j != k]; best, bs = None, -1
            for l in layers:
                X = Z["none"][f"L{l}"].astype(np.float32); inner = []
                for j in trf:
                    itr = np.isin(fold, [x for x in trf if x != j]); ite = fold == j
                    inner.append(roc_auc_score(y[ite], mk().fit(X[itr], y[itr]).predict_proba(X[ite])[:, 1]))
                if np.mean(inner) > bs: best, bs = l, np.mean(inner)
            chosen.append(best)
            tr, te = np.isin(fold, trf), fold == k
            clf = mk().fit(Z["none"][f"L{best}"].astype(np.float32)[tr], y[tr])
            for c in CONDS:
                probe[c][te] = clf.decision_function(Z[c][f"L{best}"].astype(np.float32)[te])
        verbal = {c: Z[c]["score"].astype(float) for c in CONDS}
        res = {"probe_layers": chosen}
        unl, lnk = (y == 0) & hard, (y == 1)
        for name, r in [("verbal", verbal), ("probe", probe)]:
            base = r["none"]
            wsd = np.sqrt(0.5 * (base[lnk].var() + base[unl].var()))
            z = {c: r[c] / wsd for c in CONDS}
            e = {}
            for lab_, a, b, m in [("relay_vs_none_unlinked", "coord", "none", unl), ("relay_vs_neutral_unlinked", "coord", "neutral", unl),
                                  ("anysent_vs_none_unlinked", "neutral", "none", unl),
                                  ("excul_vs_none_linked", "benign", "none", lnk), ("excul_vs_neutral_linked", "benign", "neutral", lnk)]:
                dd = z[a] - z[b]
                e[lab_] = dict(mean=float(dd[m].mean()), ci=boot_mean(dd, m))
            e["auroc_none"] = auc_macro(r["none"], r["none"])
            e["auroc_neutral_control"] = auc_macro(r["neutral"], r["neutral"])
            e["auroc_attack"] = auc_macro(r["benign"], r["coord"])
            e["within_class_sd_none"] = float(wsd)
            res[name] = e
        # paired difference in absolute relay shift (verbal minus probe), unlinked pairs, SD units
        dv = verbal["coord"] / np.sqrt(0.5 * (verbal["none"][lnk].var() + verbal["none"][unl].var())) - verbal["none"] / np.sqrt(0.5 * (verbal["none"][lnk].var() + verbal["none"][unl].var()))
        dp = probe["coord"] / np.sqrt(0.5 * (probe["none"][lnk].var() + probe["none"][unl].var())) - probe["none"] / np.sqrt(0.5 * (probe["none"][lnk].var() + probe["none"][unl].var()))
        dd = np.abs(dv) - np.abs(dp)
        res["abs_relay_shift_verbal_minus_probe"] = dict(mean=float(dd[unl].mean()), ci=boot_mean(dd, unl))
        out[f"{model}|{style}"] = res
        v, p = res["verbal"], res["probe"]
        print(f"\n{model} [{style}] probe layers {chosen}")
        for key in ["relay_vs_none_unlinked", "relay_vs_neutral_unlinked", "anysent_vs_none_unlinked", "excul_vs_none_linked"]:
            print(f"  {key:28s} verbal {v[key]['mean']:+.3f} [{v[key]['ci'][0]:+.3f},{v[key]['ci'][1]:+.3f}]   probe {p[key]['mean']:+.3f} [{p[key]['ci'][0]:+.3f},{p[key]['ci'][1]:+.3f}]  (within-class SD units)")
        print(f"  AUROC none/control/attack: verbal {v['auroc_none']:.3f}/{v['auroc_neutral_control']:.3f}/{v['auroc_attack']:.3f}   probe {p['auroc_none']:.3f}/{p['auroc_neutral_control']:.3f}/{p['auroc_attack']:.3f}")
        print(f"  |relay shift| verbal - probe: {res['abs_relay_shift_verbal_minus_probe']['mean']:+.3f} {np.round(res['abs_relay_shift_verbal_minus_probe']['ci'], 3)}", flush=True)
json.dump(out, open(ROOT / "results" / "robust_repr.json", "w"), indent=1, default=float)
