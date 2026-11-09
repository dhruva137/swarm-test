"""Express the null-controlled relay effects (recruit_null.json, log-odds) in units of each judge's own
pooled within-class SD of its clean ('none') verdict score (linked vs hard-unlinked pairs), and as a fraction of the
judge's clean linked-vs-unlinked gap. Same pairs and conditions as src/recruit_null.py."""
import json
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
S = pd.read_parquet(ROOT / "results" / "v2_sample.parquet").sort_values("sid").reset_index(drop=True)
y = S.y.values; hard = S.neg_type.values != "easy"
lnk, unl = y == 1, (y == 0) & hard
N = json.load(open(ROOT / "results" / "recruit_null.json"))
src = {"Qwen3-8B verbal [security prompt]": ("k_swarm-recruit", "Qwen/Qwen3-8B"), "Qwen3-4B verbal [security prompt]": ("k_swarm-recruit", "Qwen/Qwen3-4B"),
       "Qwen3-8B verbal [neutral prompt]": ("k_swarm-recruit8n", "Qwen/Qwen3-8B")}
out = {}
for key, (d, model) in src.items():
    R = pd.read_json(ROOT / "results" / d / "recruit_scores.jsonl", lines=True)
    W = R[R.model == model].pivot(index="sid", columns="cond", values="score").loc[S.sid]
    base = W["none"].values
    wsd = float(np.sqrt(0.5 * (base[lnk].var() + base[unl].var())))
    gap = float(base[lnk].mean() - base[unl].mean())
    r = N[key]
    out[key] = dict(within_class_sd=wsd, class_gap=gap,
                    relay_vs_none_sd=r["relay vs nothing, unlinked"]["mean"] / wsd, relay_vs_none_frac_gap=r["relay vs nothing, unlinked"]["mean"] / gap,
                    relay_vs_neutral_sd=r["relay vs neutral, unlinked"]["mean"] / wsd, relay_vs_neutral_frac_gap=r["relay vs neutral, unlinked"]["mean"] / gap)
    o = out[key]
    print(f"{key:36s} within-class SD {wsd:.2f} | class gap {gap:+.2f} | relay vs none {o['relay_vs_none_sd']:+.2f} SD ({o['relay_vs_none_frac_gap']:+.0%} of gap) | relay vs neutral {o['relay_vs_neutral_sd']:+.2f} SD ({o['relay_vs_neutral_frac_gap']:+.0%} of gap)")
json.dump(out, open(ROOT / "results" / "relay_sd.json", "w"), indent=1)
