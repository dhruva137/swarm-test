"""Does the verdict answer 'does this SOUND like coordination?' while the representation tracks 'are these LINKED?'

Pre-specified 'coordination-talk' score of a pair: mean over its two posts (the exact URL-free 1,500-char text the
monitors saw) of the cosine between the post's frozen bge embedding and the centroid of the five real swarm relay
sentences used in the injection experiment (fixed before this analysis; nothing is tuned here).
For each readout r (verbal log-odds; out-of-fold linear probe score), on the 1,500-pair v2 sample:
  OLS of z(r) on z(linked), z(text similarity = emb cosine of the pair), z(coordination-talk)
  -> standardized coefficient on coordination-talk, agent-cluster bootstrap CI (1,000)
  Spearman correlation of r with coordination-talk among UNLINKED (hard) pairs only (false-positive pressure).
Models/prompts: Qwen3-1.7B and 4B, neutral and collusion-framed (scores from results/llm_v2_scores.parquet).
"""
import json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr

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
y = S.y.values.astype(float); unl = (S.y.values == 0) & (S.neg_type.values != "easy")
z = lambda v: (v - v.mean()) / v.std()
cl = S.groupby("agent_a").indices; keys = list(cl); rng = np.random.default_rng(0)


def ols_coef(ii, col):
    X = np.c_[np.ones(len(ii)), z(y[ii]), z(S.emb_1500.values[ii]), z(S.coordtalk.values[ii])]
    b = np.linalg.lstsq(X, z(S[col].values[ii]), rcond=None)[0]
    return b


out = {"coordtalk_mean_linked": float(S.coordtalk[S.y == 1].mean()), "coordtalk_mean_unlinked": float(S.coordtalk[S.y == 0].mean())}
cols = [c_ for c_ in S.columns if (c_.startswith("llm|") or c_.startswith("probe|")) and c_.endswith("_meta")]
for col in sorted(cols):
    allix = np.arange(len(S)); b = ols_coef(allix, col)
    bs = np.array([ols_coef(np.concatenate([cl[keys[j]] for j in rng.integers(0, len(keys), len(keys))]), col) for _ in range(1000)])
    rho, p = spearmanr(S[col].values[unl], S.coordtalk.values[unl])
    out[col] = dict(beta_linked=float(b[1]), beta_similarity=float(b[2]), beta_coordtalk=float(b[3]),
                    beta_coordtalk_ci=[float(np.quantile(bs[:, 3], .025)), float(np.quantile(bs[:, 3], .975))],
                    beta_linked_ci=[float(np.quantile(bs[:, 1], .025)), float(np.quantile(bs[:, 1], .975))],
                    spearman_unlinked_coordtalk=float(rho), spearman_p=float(p))
    o = out[col]
    print(f"{col:38s} beta(linked) {o['beta_linked']:+.3f}  beta(similarity) {o['beta_similarity']:+.3f}  "
          f"beta(coord-talk) {o['beta_coordtalk']:+.3f} [{o['beta_coordtalk_ci'][0]:+.3f},{o['beta_coordtalk_ci'][1]:+.3f}]  "
          f"| unlinked: rho(score, coord-talk) {o['spearman_unlinked_coordtalk']:+.3f}")
json.dump(out, open(ROOT / "results" / "topic_vs_relation.json", "w"), indent=1)
