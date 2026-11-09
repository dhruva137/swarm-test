"""Stress tests for the coordination-talk result of src/topic_vs_relation.py.

Same pre-specified instrument: cosine between a post's frozen bge-base-en-v1.5 embedding (URL-free text, first 1,500
chars) and the centroid of the five real relay sentences used in the injection experiment. Nothing is tuned.
C1 framing effect: beta_coordtalk(framed) - beta_coordtalk(neutral) for the same model, paired agent-cluster bootstrap
   (the same resample for both prompts).
C2 hard pairs only: the OLS refit on positives + hard negatives (the 250 easy negatives dropped).
C3 lexical instrument: coordination-talk = TF-IDF cosine (English stop words removed) to the relay-sentence centroid,
   instead of bge.
C4 false-positive pressure: among unlinked hard pairs only, OLS z(score) ~ z(similarity) + z(coordtalk).
C5 synthetic swarms: per-post coordination-talk with the same bge instrument (digits and city names masked, as in
   src/monoculture.py), open vs warned, paired by seed; bootstrap over seeds and count of seeds with a drop.
"""
import json, glob, re
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]
RELAY = ["Please keep relaying R4/R5 exact wording and whether/when sequence terminates; your lead is invaluable.",
         "Please keep posting R5+ immediately.", "Fast agents please keep waiting; relay C3 immediately.",
         "Please keep posting confirmations.", "Please keep thread alive."]
S = pd.read_parquet(ROOT / "results" / "llm_v2_scores.parquet").sort_values("sid").reset_index(drop=True)
df = pd.read_parquet(ROOT / "data" / "edits.parquet", columns=["text_nourl"])
T = df.text_nourl.str.slice(0, 1500).values
from sentence_transformers import SentenceTransformer
m = SentenceTransformer("BAAI/bge-base-en-v1.5", device="cuda"); m.half(); m.max_seq_length = 512
ids = np.unique(np.r_[S.a.values, S.b.values])
E = dict(zip(ids, m.encode(T[ids].tolist(), batch_size=64, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)))
c = m.encode(RELAY, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32).mean(0); c /= np.linalg.norm(c)
S["coordtalk"] = [0.5 * (E[a] @ c + E[b] @ c) for a, b in zip(S.a, S.b)]
# C3 lexical instrument
tf = TfidfVectorizer(sublinear_tf=True, stop_words="english").fit(list(T[ids]) + RELAY)
Xp = tf.transform(T[ids]); cl_ = np.asarray(tf.transform(RELAY).mean(0)).ravel(); cl_ /= np.linalg.norm(cl_)
lex = dict(zip(ids, np.asarray(Xp @ cl_).ravel()))
S["coordtalk_lex"] = [0.5 * (lex[a] + lex[b]) for a, b in zip(S.a, S.b)]
y = S.y.values.astype(float)
hardall = S.neg_type.values != "easy"
unl = (S.y.values == 0) & hardall
z = lambda v: (v - v.mean()) / v.std()
cl = S.groupby("agent_a").indices; keys = list(cl); rng = np.random.default_rng(0)
B = [np.concatenate([cl[keys[j]] for j in rng.integers(0, len(keys), len(keys))]) for _ in range(1000)]


def beta(ii, col, ct="coordtalk", with_y=True):
    cols = ([z(y[ii])] if with_y else []) + [z(S.emb_1500.values[ii]), z(S[ct].values[ii])]
    X = np.c_[np.ones(len(ii)), np.column_stack(cols)]
    return np.linalg.lstsq(X, z(S[col].values[ii]), rcond=None)[0][-1]


def est(col, mask=None, ct="coordtalk", with_y=True):
    base = np.arange(len(S)) if mask is None else np.where(mask)[0]
    b0 = beta(base, col, ct, with_y)
    bs = []
    for ii in B:
        jj = ii if mask is None else ii[mask[ii]]
        bs.append(beta(jj, col, ct, with_y))
    return dict(beta=float(b0), ci=[float(np.quantile(bs, .025)), float(np.quantile(bs, .975))])


out = {"corr_bge_vs_lexical_instrument": float(np.corrcoef(S.coordtalk, S.coordtalk_lex)[0, 1])}
from sklearn.metrics import roc_auc_score
mlh = (y == 1) | hardall
for ct in ["coordtalk", "coordtalk_lex"]:
    out[f"{ct}_auroc_linked_vs_hard_unlinked"] = float(roc_auc_score(y[mlh], S[ct].values[mlh]))
    print(f"{ct} alone as a linkage score (linked vs hard unlinked, pooled): AUROC {out[f'{ct}_auroc_linked_vs_hard_unlinked']:.3f}")
print(f"instrument agreement: r(bge coordtalk, lexical coordtalk) = {out['corr_bge_vs_lexical_instrument']:.3f}")
for mdl in ["Qwen3-1.7B", "Qwen3-4B"]:
    for kind in ["llm", "probe"]:
        cn, cf = f"{kind}|{mdl}_neutral_meta", f"{kind}|{mdl}_primed_meta"
        d0 = beta(np.arange(len(S)), cf) - beta(np.arange(len(S)), cn)
        ds = [beta(ii, cf) - beta(ii, cn) for ii in B]
        r = {"C1_framed_minus_neutral": dict(diff=float(d0), ci=[float(np.quantile(ds, .025)), float(np.quantile(ds, .975))])}
        d0l = beta(np.arange(len(S)), cf, "coordtalk_lex") - beta(np.arange(len(S)), cn, "coordtalk_lex")
        dsl = [beta(ii, cf, "coordtalk_lex") - beta(ii, cn, "coordtalk_lex") for ii in B]
        r["C1_lexical_framed_minus_neutral"] = dict(diff=float(d0l), ci=[float(np.quantile(dsl, .025)), float(np.quantile(dsl, .975))])
        print(f"{kind}|{mdl}: LEXICAL framed - neutral loading {d0l:+.3f} [{np.quantile(dsl, .025):+.3f},{np.quantile(dsl, .975):+.3f}]")
        for st, col in [("neutral", cn), ("framed", cf)]:
            r[st] = {"C2_hard_only": est(col, mask=(y == 1) | hardall),
                     "C3_lexical": est(col, ct="coordtalk_lex"),
                     "C4_unlinked_hard": est(col, mask=unl, with_y=False)}
        out[f"{kind}|{mdl}"] = r
        f = lambda e: f"{e['beta']:+.3f} [{e['ci'][0]:+.3f},{e['ci'][1]:+.3f}]"
        print(f"\n{kind}|{mdl}: framed - neutral coord-talk loading {r['C1_framed_minus_neutral']['diff']:+.3f} "
              f"[{r['C1_framed_minus_neutral']['ci'][0]:+.3f},{r['C1_framed_minus_neutral']['ci'][1]:+.3f}]")
        for st in ["neutral", "framed"]:
            print(f"  {st:7s} hard-only {f(r[st]['C2_hard_only'])} | lexical {f(r[st]['C3_lexical'])} | unlinked-only (FP pressure) {f(r[st]['C4_unlinked_hard'])}")
# C5 synthetic swarms
CITIES = ["Oslo", "Lima", "Accra", "Hanoi", "Quito", "Riga", "Doha", "Perth", "Turin", "Dakar", "Cusco", "Malmo",
          "Izmir", "Kobe", "Nantes", "Porto", "Tartu", "Leon", "Graz", "Haifa"]
CRE = re.compile(r"\b(" + "|".join(CITIES) + r")\b", re.I)
rows = []
for model, d in [("Qwen3-8B", "k_swarm-sim"), ("Qwen3-1.7B", "sim_1p7b")]:
    for fpath in sorted(glob.glob(str(ROOT / "results" / d / "sim_*.json"))):
        D = json.load(open(fpath, encoding="utf-8"))
        for rec in D["log"]:
            if rec["post"].strip():
                rows.append(dict(model=model, cond=rec["cond"], seed=rec["seed"], text=CRE.sub(" CITY ", re.sub(r"\d+", " NUM ", rec["post"]))))
P = pd.DataFrame(rows)
P["coordtalk"] = m.encode(P.text.tolist(), batch_size=64, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32) @ c
P["coordtalk_lex"] = np.asarray(tf.transform(P.text.tolist()) @ cl_).ravel()
out["C5_wiki_post_mean_coordtalk"] = float(np.mean([E[i] @ c for i in ids]))
print(f"\nC5 reference: mean coord-talk of wiki posts in the pair sample = {out['C5_wiki_post_mean_coordtalk']:.3f}")
for model in ["Qwen3-8B", "Qwen3-1.7B"]:
    for ct in ["coordtalk", "coordtalk_lex"]:
        g = P[P.model == model].groupby(["seed", "cond"])[ct].mean().unstack()
        dd = (g["covert"] - g["open"]).values
        bs = [rng.choice(dd, len(dd)).mean() for _ in range(5000)]
        key = f"C5_{model}_{ct}"
        out[key] = dict(open_mean=float(g["open"].mean()), covert_mean=float(g["covert"].mean()), diff=float(dd.mean()),
                        ci=[float(np.quantile(bs, .025)), float(np.quantile(bs, .975))],
                        seeds_lower=int((dd < 0).sum()), n_seeds=int(len(dd)),
                        per_seed_open=[float(v) for v in g["open"].values], per_seed_covert=[float(v) for v in g["covert"].values])
        o = out[key]
        print(f"C5 {model} {ct:13s}: open {o['open_mean']:.3f} -> covert {o['covert_mean']:.3f}, diff {o['diff']:+.3f} "
              f"[{o['ci'][0]:+.3f},{o['ci'][1]:+.3f}], lower in {o['seeds_lower']}/{o['n_seeds']} seeds")
json.dump(out, open(ROOT / "results" / "coordtalk_checks.json", "w"), indent=1)
