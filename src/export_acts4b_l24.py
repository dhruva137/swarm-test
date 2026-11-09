"""Build the Kaggle input of kaggle/steer4b: Qwen3-4B layer-24 answer-position activations and verdict scores
(neutral and collusion-framed prompts) on the 1,500-pair v2 sample, taken from results/k_swarm-probe-acts/.
Usage: python src/export_acts4b_l24.py [output.npz]   (default: kaggle/acts4b_l24_data/acts4b_l24.npz)
The steering job also needs the v2 pairs file (kaggle/v2_data/pairs.jsonl, built by src/export_v2.py).
"""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "kaggle" / "acts4b_l24_data" / "acts4b_l24.npz"
A = {st: np.load(ROOT / "results" / "k_swarm-probe-acts" / f"Qwen3-4B_{st}_meta.npz") for st in ["neutral", "primed"]}
assert (A["neutral"]["sid"] == A["primed"]["sid"]).all()
np.savez_compressed(out, X_neutral=A["neutral"]["L24"], X_primed=A["primed"]["L24"],
                    score_neutral=A["neutral"]["score"], score_primed=A["primed"]["score"], sid=A["neutral"]["sid"])
print("wrote", out)
