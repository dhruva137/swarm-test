"""Export the v2 LLM evaluation sample: 1,500 pairs (per time block: 150 pos, 50 easy, 100 hard), URL-matched,
URLs deleted, every text truncated to the same 1,500 chars that all monitors will see. Carries eids explicitly."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from pairs_v2 import build_pairs_v2

ROOT = Path(__file__).resolve().parents[1]
N = 1500
df = pd.read_parquet(ROOT / "data" / "edits.parquet")
P, _ = build_pairs_v2(df)
quota = {"pos": 150, "easy": 50, "hard": 100}
S = pd.concat([g.sample(quota[t], random_state=5) for (k, t), g in P.groupby(["fold", "neg_type"])])
S = S.reset_index().rename(columns={"index": "pair_row"})
S["sid"] = range(len(S))
E = df.set_index("eid")


def side(e):
    r = E.loc[e]
    return dict(eid=int(e), text=r.text_nourl[:N], agent=r.label or "(no name)", page=r.page_key, wiki=r.wiki,
                time=r.time.isoformat(), ip16=r.ip16)


d = ROOT / "kaggle" / "v2_data"; d.mkdir(parents=True, exist_ok=True)
with open(d / "pairs.jsonl", "w", encoding="utf-8") as f:
    for r in S.itertuples():
        f.write(json.dumps(dict(sid=int(r.sid), pair_row=int(r.pair_row), fold=int(r.fold), y=int(r.y),
                                neg_type=r.neg_type, a=side(r.a), b=side(r.b))) + "\n")
S.to_parquet(ROOT / "results" / "v2_sample.parquet")
json.dump({"title": "swarm-v2-pairs", "id": "dhruvapgowda/swarm-v2-pairs", "licenses": [{"name": "other"}]},
          open(d / "dataset-metadata.json", "w"))
print(S.groupby(["fold", "neg_type"]).size().unstack(), len(S))
