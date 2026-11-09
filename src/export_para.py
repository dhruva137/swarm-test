"""Export a stratified pair sample (with eids) whose B-side posts will be paraphrased on Kaggle (evasion test)."""
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_parquet(ROOT / "data" / "edits.parquet").set_index("eid")
P = pd.read_parquet(ROOT / "results" / "pairs.parquet").reset_index().rename(columns={"index": "row"})
sub = pd.concat([g.sample(1000, random_state=11) for _, g in P.groupby("neg_type")])
sub[["row", "a", "b", "y", "neg_type"]].to_parquet(ROOT / "results" / "para_pairs.parquet")
d = ROOT / "kaggle" / "para_data"; d.mkdir(parents=True, exist_ok=True)
eids = sorted(set(sub.b))
with open(d / "posts.jsonl", "w", encoding="utf-8") as f:
    for e in eids:
        f.write(json.dumps({"eid": int(e), "text": df.loc[e, "text_strict"][:2000]}) + "\n")
json.dump({"title": "swarm-para-posts", "id": "dhruvapgowda/swarm-para-posts", "licenses": [{"name": "other"}]},
          open(d / "dataset-metadata.json", "w"))
print(len(sub), "pairs,", len(eids), "posts to paraphrase")
