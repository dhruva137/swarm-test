"""Ground-truth validation: blind annotator labels vs the masked-artifact ground truth on 200 pairs.
Sources: Claude Sonnet annotators (validation/labels_*.json) and, if present, human labels (validation/human_labels.json)."""
import json, glob
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
key = {int(k): v for k, v in json.load(open(ROOT / "results" / "labeler_key.json")).items()}
out = {}


def summarize(labels, name):
    df = pd.DataFrame([(int(k), v["label"] if isinstance(v, dict) else v) for k, v in labels.items()], columns=["pair_idx", "label"])
    df["type"] = df.pair_idx.map(key)
    tab = pd.crosstab(df.type, df.label)
    print(name); print(tab)
    r = {"n": len(df), "table": tab.to_dict()}
    for t in ["pos", "hard", "easy"]:
        g = df[df.type == t]
        dec = g[g.label != "unclear"]
        r[f"{t}_yes_rate_decided"] = float((dec.label == "yes").mean()) if len(dec) else None
        r[f"{t}_unclear_rate"] = float((g.label == "unclear").mean())
    # precision of a ground-truth positive and of a negative, among decided labels
    dec = df[df.label != "unclear"]
    r["gt_pos_precision"] = float((dec[dec.type == "pos"].label == "yes").mean())
    r["gt_neg_precision"] = float((dec[dec.type != "pos"].label == "no").mean())
    r["agreement_decided"] = float(((dec.type == "pos") == (dec.label == "yes")).mean())
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k != "table"})
    out[name] = r


son = {}
for f in sorted(glob.glob(str(ROOT / "validation" / "labels_*.json"))):
    son.update(json.load(open(f, encoding="utf-8")))
summarize(son, "claude_sonnet_blind")
hp = ROOT / "validation" / "human_labels.json"
if hp.exists():
    hum = json.load(open(hp))
    summarize(hum, "human_blind")
    both = {k: (son[k]["label"], hum[k]) for k in hum if k in son}
    agree = sum(a == b for a, b in both.values()) / max(1, len(both))
    out["human_vs_sonnet_agreement"] = agree
    print("human vs sonnet agreement", round(agree, 3))
json.dump(out, open(ROOT / "results" / "validation.json", "w"), indent=1, default=float)
