"""Head-to-head on the LLM's exact 3,000 pairs: cheap monitor vs Qwen3-8B, and whether the LLM adds
complementary signal (stacked with the cheap model's score)."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
sys.path.insert(0, str(Path(__file__).parent))
from run_eval import scores
from features import pair_features, FIELD_OF

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_parquet(ROOT / "data" / "edits.parquet")
emb = np.load(ROOT / "data" / "emb_bge.npy")
key = {(r.page_key, r.time.isoformat(), r.label or "(no name)"): r.eid for r in df.itertuples()}
rows = []
for l in open(ROOT / "kaggle" / "llm_data" / "pairs.jsonl", encoding="utf-8"):
    r = json.loads(l)
    a = key.get((r["a"]["page"], r["a"]["time"], r["a"]["agent"]))
    b = key.get((r["b"]["page"], r["b"]["time"], r["b"]["agent"]))
    rows.append(dict(pair_idx=r["pair_idx"], a=a, b=b, y=r["y"], neg_type=r["neg_type"]))
S = pd.DataFrame(rows)
print("matched", S.a.notna().mean(), S.b.notna().mean())
S = S.dropna().astype({"a": int, "b": int})
L = pd.read_json(ROOT / "results" / "kaggle_llm" / "llm_scores.jsonl", lines=True)
for meta, g in L.groupby("meta"):
    S = S.merge(g[["pair_idx", "score"]].rename(columns={"score": f"llm_{'meta' if meta else 'content'}"}), on="pair_idx")
P = pd.read_parquet(ROOT / "results" / "pairs.parquet")
feats = [c for c in P.columns if c in FIELD_OF]
# train the cheap model on all pairs that share no edit with the evaluation sample
used = set(S.a) | set(S.b)
train = P[~P.a.isin(used) & ~P.b.isin(used)]
FS = pair_features(df, S[["a", "b"]].reset_index(drop=True), emb)
S = S.reset_index(drop=True).join(FS)
out = {"n_eval": len(S), "n_train": len(train)}
clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0).fit(train[feats], train.y)
S["cheap"] = clf.predict_proba(S[feats])[:, 1]
cont = [f for f in feats if FIELD_OF[f] == "content"]
S["cheap_content"] = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0).fit(train[cont], train.y).predict_proba(S[cont])[:, 1]
for col in ["cheap", "cheap_content", "emb_cos", "llm_content", "llm_meta"]:
    out[col] = scores(S.y.values, S[col].values, S.neg_type.values, n_boot=500)
    print(f"{col:14s} hard AUROC={out[col]['hard']['auroc']:.3f} CI={np.round(out[col]['hard']['ci'],3)} R@1%={out[col]['hard']['recall_at_fpr1']:.3f}")
# stacking: does adding the LLM score to the cheap score help? (5-fold within the sample)
for extra in ["llm_content", "llm_meta"]:
    p = np.zeros(len(S)); X = np.c_[np.log(S.cheap / (1 - S.cheap + 1e-9) + 1e-9), S[extra]]
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, S.y):
        p[te] = LogisticRegression().fit(X[tr], S.y.values[tr]).predict_proba(X[te])[:, 1]
    out[f"cheap+{extra}"] = scores(S.y.values, p, S.neg_type.values, n_boot=500)
    print(f"cheap+{extra:10s} hard AUROC={out[f'cheap+{extra}']['hard']['auroc']:.3f} R@1%={out[f'cheap+{extra}']['hard']['recall_at_fpr1']:.3f}")
# where does the LLM fail? near-copies vs not
pos = S[S.y == 1]
out["llm_meta_score_on_nearcopy_pos"] = float(pos[pos.emb_cos > 0.95].llm_meta.mean())
out["llm_meta_score_on_other_pos"] = float(pos[pos.emb_cos <= 0.95].llm_meta.mean())
out["llm_meta_score_on_neg"] = float(S[S.y == 0].llm_meta.mean())
print({k: round(v, 3) for k, v in out.items() if isinstance(v, float)})
S.to_parquet(ROOT / "results" / "stack_pairs.parquet")
json.dump(out, open(ROOT / "results" / "stack.json", "w"), indent=1, default=float)
