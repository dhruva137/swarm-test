"""Monoculture: do independent swarms of the same model converge on the same disguise when warned?

Data: controlled swarms (results/k_swarm-sim = Qwen3-8B, 8 seeds; results/sim_1p7b = Qwen3-1.7B, 4 seeds), two
conditions per seed (open / covert = warned about a content monitor). Different seeds never interact.
Text preprocessing (both conditions): digits and all city names masked, so per-seed task content cannot drive similarity.
Embeddings: frozen bge-base-en-v1.5 (cosine).
M1 cross-swarm similarity: mean cosine between posts from DIFFERENT seeds of the same model and condition
   (independent swarms), vs the same within a seed across different agents. Bootstrap over seed pairs.
M2 detectability of evasion: logistic regression on embeddings, open vs covert posts, leave-one-seed-out AUROC.
M3 model fingerprint: 8B-covert vs 1.7B-covert posts (seeds 0-3 present in both), leave-one-seed-out AUROC; same for open.
M4 vocabulary concentration: share of all covert (vs open) tokens covered by the 20 most frequent content words, and the
   overlap of the top-20 covert words between independent seeds (Jaccard).
"""
import json, glob, re, itertools, collections
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
CITIES = ["Oslo", "Lima", "Accra", "Hanoi", "Quito", "Riga", "Doha", "Perth", "Turin", "Dakar", "Cusco", "Malmo",
          "Izmir", "Kobe", "Nantes", "Porto", "Tartu", "Leon", "Graz", "Haifa"]
CRE = re.compile(r"\b(" + "|".join(CITIES) + r")\b", re.I)
STOP = set("the a an and or of to in is are be it its this that for on with as by at from your you we our i my all each "
           "will can not no but if so code codes city cities know known agent agents board post round".split())
rows = []
for model, d in [("Qwen3-8B", "k_swarm-sim"), ("Qwen3-1.7B", "sim_1p7b")]:
    for f in sorted(glob.glob(str(ROOT / "results" / d / "sim_*.json"))):
        D = json.load(open(f, encoding="utf-8"))
        for r in D["log"]:
            if r["post"].strip():
                t = CRE.sub(" CITY ", re.sub(r"\d+", " NUM ", r["post"]))
                rows.append(dict(model=model, cond=r["cond"], seed=r["seed"], agent=r["agent"], round=r["round"], text=t))
P = pd.DataFrame(rows)
print(P.groupby(["model", "cond"]).size())
from sentence_transformers import SentenceTransformer
m = SentenceTransformer("BAAI/bge-base-en-v1.5", device="cuda"); m.half()
E = m.encode(P.text.tolist(), batch_size=64, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
rng = np.random.default_rng(0)
out = {}
for model in ["Qwen3-8B", "Qwen3-1.7B"]:
    res = {}
    for cond in ["open", "covert"]:
        sub = P[(P.model == model) & (P.cond == cond)]
        seeds = sorted(sub.seed.unique()); idx = {s: sub.index[sub.seed == s].values for s in seeds}
        cross = {}
        for a, b in itertools.combinations(seeds, 2):
            cross[(a, b)] = float((E[idx[a]] @ E[idx[b]].T).mean())
        within = []
        for s in seeds:
            ii = idx[s]; G = E[ii] @ E[ii].T; ag = sub.agent.values[sub.seed.values == s]
            mask = ag[:, None] != ag[None, :]
            within.append(float(G[mask].mean()))
        cv = np.array(list(cross.values()))
        bs = [rng.choice(cv, len(cv)).mean() for _ in range(2000)]
        res[cond] = dict(cross_seed_cos=float(cv.mean()), cross_seed_ci=[float(np.quantile(bs, .025)), float(np.quantile(bs, .975))],
                         within_seed_cross_agent_cos=float(np.mean(within)), n_posts=int(len(sub)), n_seed_pairs=len(cv))
    # paired over seed pairs: covert minus open cross-seed similarity
    so = P[(P.model == model) & (P.cond == "open")]; sc = P[(P.model == model) & (P.cond == "covert")]
    seeds = sorted(set(so.seed) & set(sc.seed))
    diffs = []
    for a, b in itertools.combinations(seeds, 2):
        co = (E[sc.index[sc.seed == a]] @ E[sc.index[sc.seed == b]].T).mean()
        op = (E[so.index[so.seed == a]] @ E[so.index[so.seed == b]].T).mean()
        diffs.append(co - op)
    diffs = np.array(diffs); bs = [rng.choice(diffs, len(diffs)).mean() for _ in range(2000)]
    res["cross_seed_covert_minus_open"] = dict(mean=float(diffs.mean()), ci=[float(np.quantile(bs, .025)), float(np.quantile(bs, .975))])
    # M2 leave-one-seed-out detectability of evasion
    sub = P[P.model == model]; yv = (sub.cond == "covert").values.astype(int); pr = np.zeros(len(sub))
    for s in sorted(sub.seed.unique()):
        te = (sub.seed == s).values
        pr[te] = LogisticRegression(C=1.0, max_iter=3000).fit(E[sub.index[~te]], yv[~te]).predict_proba(E[sub.index[te]])[:, 1]
    res["evasion_detect_loso_auroc"] = float(roc_auc_score(yv, pr))
    # M4 vocabulary
    def topw(texts, k=20):
        c = collections.Counter(w for t in texts for w in re.findall(r"[a-z]+", t.lower()) if w not in STOP and len(w) > 3)
        tot = sum(c.values()); top = c.most_common(k)
        return [w for w, _ in top], (sum(n for _, n in top) / tot if tot else 0)
    for cond in ["open", "covert"]:
        s2 = P[(P.model == model) & (P.cond == cond)]
        words, share = topw(s2.text)
        per = {s: set(topw(s2.text[s2.seed == s])[0]) for s in sorted(s2.seed.unique())}
        jac = [len(per[a] & per[b]) / len(per[a] | per[b]) for a, b in itertools.combinations(per, 2)]
        res[cond].update(top20_words=words, top20_token_share=float(share), top20_jaccard_across_seeds=float(np.mean(jac)))
    out[model] = res
    print(f"\n{model}")
    for cond in ["open", "covert"]:
        r = res[cond]
        print(f"  {cond:6s} cross-seed cos {r['cross_seed_cos']:.3f} {np.round(r['cross_seed_ci'], 3)} | within-seed cross-agent {r['within_seed_cross_agent_cos']:.3f} | "
              f"top20 share {r['top20_token_share']:.3f} | top20 Jaccard across seeds {r['top20_jaccard_across_seeds']:.3f}")
        print(f"         top words: {r['top20_words'][:12]}")
    print(f"  covert - open cross-seed cos: {res['cross_seed_covert_minus_open']['mean']:+.3f} {np.round(res['cross_seed_covert_minus_open']['ci'], 3)}")
    print(f"  evasion detectable leave-one-seed-out AUROC: {res['evasion_detect_loso_auroc']:.3f}", flush=True)
# M3 model fingerprint (seeds 0-3 in both models)
for cond in ["covert", "open"]:
    sub = P[(P.cond == cond) & (P.seed <= 3)]; yv = (sub.model == "Qwen3-8B").values.astype(int); pr = np.zeros(len(sub))
    for s in range(4):
        te = (sub.seed == s).values
        pr[te] = LogisticRegression(C=1.0, max_iter=3000).fit(E[sub.index[~te]], yv[~te]).predict_proba(E[sub.index[te]])[:, 1]
    out[f"model_fingerprint_{cond}_loso_auroc"] = float(roc_auc_score(yv, pr))
    print(f"model fingerprint ({cond} posts, 8B vs 1.7B) leave-one-seed-out AUROC: {out[f'model_fingerprint_{cond}_loso_auroc']:.3f}")
json.dump(out, open(ROOT / "results" / "monoculture.json", "w"), indent=1, default=float)
