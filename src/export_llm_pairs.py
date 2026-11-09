"""Export a stratified pair sample (with masked text + metadata) for the LLM monitor on Kaggle."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from pairs import build_pairs

ROOT = Path(__file__).resolve().parents[1]
N = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
df = pd.read_parquet(ROOT / "data" / "edits.parquet").set_index("eid")
pairs, _ = build_pairs(df.reset_index())
pairs = pairs.reset_index().rename(columns={"index": "pair_idx"})
rng = np.random.default_rng(1)
sub = pd.concat([g.sample(min(N, len(g)), random_state=1) for _, g in pairs.groupby("neg_type")])
out_dir = ROOT / "kaggle" / "llm_data"; out_dir.mkdir(parents=True, exist_ok=True)


def side(e):
    r = df.loc[e]
    return dict(text=r.text_masked[:1500], agent=r.label or "(no name)", page=r.page_key, wiki=r.wiki,
                time=r.time.isoformat(), ip16=r.ip16)


with open(out_dir / "pairs.jsonl", "w", encoding="utf-8") as f:
    for r in sub.itertuples():
        f.write(json.dumps(dict(pair_idx=int(r.pair_idx), y=int(r.y), neg_type=r.neg_type, a=side(r.a), b=side(r.b))) + "\n")
json.dump({"title": "swarm-llm-pairs", "id": "dhruvapgowda/swarm-llm-pairs", "licenses": [{"name": "other"}]},
          open(out_dir / "dataset-metadata.json", "w"))
print(len(sub), "pairs ->", out_dir)
