"""Third coordination-talk instrument: a sentence encoder from a different model family (all-MiniLM-L6-v2).

Same design as src/topic_vs_relation.py and src/coordtalk_checks.py: coordination-talk of a pair = mean cosine of
its two posts (URL-free, first 1,500 chars) to the centroid of the same five relay sentences; OLS of z(readout) on
z(linked), z(pair similarity emb_1500), z(coordination-talk); agent-cluster bootstrap (1,000). Also the paired
framed-minus-neutral difference, and the open-vs-warned change on synthetic swarm posts (digits and city names masked).
"""
import json, glob, re
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RELAY = ["Please keep relaying R4/R5 exact wording and whether/when sequence terminates; your lead is invaluable.",
         "Please keep posting R5+ immediately.", "Fast agents please keep waiting; relay C3 immediately.",
         "Please keep posting confirmations.", "Please keep thread alive."]
S = pd.read_parquet(ROOT / "results" / "llm_v2_scores.parquet").sort_values("sid").reset_index(drop=True)
T = pd.read_parquet(ROOT / "data" / "edits.parquet", columns=["text_nourl"]).text_nourl.str.slice(0, 1500).values
from sentence_transformers import SentenceTransformer
m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda"); m.max_seq_length = 512
ids = np.unique(np.r_[S.a.values, S.b.values])
E = dict(zip(ids, m.encode(T[ids].tolist(), batch_size=64, normalize_embeddings=True, convert_to_numpy=True)))
c = m.encode(RELAY, normalize_embeddings=True, convert_to_numpy=True).mean(0); c /= np.linalg.norm(c)
S["ct"] = [0.5 * (E[a] @ c + E[b] @ c) for a, b in zip(S.a, S.b)]
y = S.y.values.astype(float)
z = lambda v: (v - v.mean()) / v.std()
cl = S.groupby("agent_a").indices; keys = list(cl); rng = np.random.default_rng(0)
B = [np.concatenate([cl[keys[j]] for j in rng.integers(0, len(keys), len(keys))]) for _ in range(1000)]


def beta(ii, col):
    X = np.c_[np.ones(len(ii)), z(y[ii]), z(S.emb_1500.values[ii]), z(S.ct.values[ii])]
    return np.linalg.lstsq(X, z(S[col].values[ii]), rcond=None)[0][-1]


q = lambda v: [float(np.quantile(v, .025)), float(np.quantile(v, .975))]
out = {"corr_with_bge_instrument": None}
for mdl in ["Qwen3-1.7B", "Qwen3-4B"]:
    for kind in ["llm", "probe"]:
        cn, cf = f"{kind}|{mdl}_neutral_meta", f"{kind}|{mdl}_primed_meta"
        allix = np.arange(len(S))
        r = {}
        for st, col in [("neutral", cn), ("framed", cf)]:
            r[st] = dict(beta=float(beta(allix, col)), ci=q([beta(ii, col) for ii in B]))
        r["framed_minus_neutral"] = dict(diff=float(beta(allix, cf) - beta(allix, cn)), ci=q([beta(ii, cf) - beta(ii, cn) for ii in B]))
        out[f"{kind}|{mdl}"] = r
        f = lambda e: f"{e['beta']:+.3f} [{e['ci'][0]:+.3f},{e['ci'][1]:+.3f}]"
        print(f"{kind}|{mdl}: neutral {f(r['neutral'])} | framed {f(r['framed'])} | framed-neutral {r['framed_minus_neutral']['diff']:+.3f} "
              f"[{r['framed_minus_neutral']['ci'][0]:+.3f},{r['framed_minus_neutral']['ci'][1]:+.3f}]", flush=True)
# agreement with the bge instrument
from sentence_transformers import SentenceTransformer as ST2
mb = ST2("BAAI/bge-base-en-v1.5", device="cuda"); mb.half(); mb.max_seq_length = 512
Eb = dict(zip(ids, mb.encode(T[ids].tolist(), batch_size=64, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)))
cb = mb.encode(RELAY, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32).mean(0); cb /= np.linalg.norm(cb)
ctb = np.array([0.5 * (Eb[a] @ cb + Eb[b] @ cb) for a, b in zip(S.a, S.b)])
out["corr_with_bge_instrument"] = float(np.corrcoef(S.ct, ctb)[0, 1])
print("r(MiniLM instrument, bge instrument) =", round(out["corr_with_bge_instrument"], 3))
# synthetic swarms
CITIES = ["Oslo", "Lima", "Accra", "Hanoi", "Quito", "Riga", "Doha", "Perth", "Turin", "Dakar", "Cusco", "Malmo",
          "Izmir", "Kobe", "Nantes", "Porto", "Tartu", "Leon", "Graz", "Haifa"]
CRE = re.compile(r"\b(" + "|".join(CITIES) + r")\b", re.I)
rows = []
for model, d in [("Qwen3-8B", "k_swarm-sim"), ("Qwen3-1.7B", "sim_1p7b")]:
    for fp in sorted(glob.glob(str(ROOT / "results" / d / "sim_*.json"))):
        for rec in json.load(open(fp, encoding="utf-8"))["log"]:
            if rec["post"].strip():
                rows.append(dict(model=model, cond=rec["cond"], seed=rec["seed"], text=CRE.sub(" CITY ", re.sub(r"\d+", " NUM ", rec["post"]))))
P = pd.DataFrame(rows)
P["ct"] = m.encode(P.text.tolist(), batch_size=64, normalize_embeddings=True, convert_to_numpy=True) @ c
for model in ["Qwen3-8B", "Qwen3-1.7B"]:
    g = P[P.model == model].groupby(["seed", "cond"]).ct.mean().unstack(); dd = (g["covert"] - g["open"]).values
    out[f"sim_{model}"] = dict(open=float(g["open"].mean()), covert=float(g["covert"].mean()), diff=float(dd.mean()),
                               ci=q([rng.choice(dd, len(dd)).mean() for _ in range(5000)]), seeds_lower=int((dd < 0).sum()), n_seeds=int(len(dd)))
    o = out[f"sim_{model}"]
    print(f"sim {model}: open {o['open']:.3f} -> warned {o['covert']:.3f}, diff {o['diff']:+.3f} {np.round(o['ci'], 3)}, lower in {o['seeds_lower']}/{o['n_seeds']}")
json.dump(out, open(ROOT / "results" / "coordtalk_minilm.json", "w"), indent=1)
