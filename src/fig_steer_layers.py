"""Figure: causal steering across layers (Qwen3-1.7B). Usage = share of the verdict's own linked-vs-unlinked gap
produced by a class-gap move along a direction (src/usage.py, src/steer_layers.py). Log scale of |usage|;
hollow markers = negative sign. Layer 20 (answer position) comes from results/usage.json."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figs"
BLUE, ORANGE, INK, INK2, GRID, MUTE = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e6e5e0", "#9b9a94"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False})
SL = json.load(open(ROOT / "results" / "steer_layers.json"))["summary"]
U = json.load(open(ROOT / "results" / "usage.json"))["Qwen3-1.7B"]
CFG = [("L8/answer", "8"), ("L12/answer", "12"), ("L16/answer", "16"), ("L20/answer", "20"), ("L24/answer", "24"), ("L20/all", "20\nall pos.")]
FLOOR = 1e-4  # % ; values below are drawn at the floor


def get(cfg, st, name):
    if cfg == "L20/answer":
        if name.startswith("perm"):
            return None
        return U[st][name]["usage"]
    return SL[f"{cfg}/{st}/{name}"]["usage"]


fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), sharey=True)
for ax, st, title in [(axes[0], "neutral", "(a) Neutral prompt"), (axes[1], "primed", "(b) Collusion-framed prompt")]:
    xs = [0, 1, 2, 3, 4, 5.4]
    for name, col, lab in [("verbal", ORANGE, "verdict-predictive direction (control)"), ("linkage", BLUE, "linkage direction (what a probe reads)")]:
        v = [get(c, st, name) for c, _ in CFG]
        y = [max(abs(x) * 100, FLOOR) for x in v]
        ax.plot(xs[:5], y[:5], color=col, lw=1.3, zorder=2)
        for xx, yy, raw in zip(xs, y, v):
            ax.scatter([xx], [yy], s=30, zorder=3, edgecolor=col, lw=1.2, facecolor=col if raw >= 0 else "white",
                       marker="D" if xx > 5 else "o", label=lab if xx == 0 else None)
    nul = []
    for c, _ in CFG:
        vals = [get(c, st, f"perm{i}") for i in range(2)]
        vals = [abs(x) * 100 for x in vals if x is not None]
        nul.append(max(max(vals), FLOOR) if vals else np.nan)
    ax.scatter(xs, nul, marker="x", s=22, color=MUTE, zorder=3, label="largest of 2 shuffled-label nulls")
    ax.axvspan(-0.4, 1.4, color=GRID, alpha=0.45, zorder=0)
    ax.text(0.5, 60, "control\nineffective", ha="center", va="top", fontsize=6.5, color=INK2)
    ax.set_yscale("log"); ax.set_ylim(FLOOR * 0.6, 100)
    ax.set_xticks(xs); ax.set_xticklabels([lab for _, lab in CFG], fontsize=7.2); ax.set_xlabel("Layer", fontsize=7.5)
    ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.set_title(title, fontsize=8.5, color=INK2, loc="left")
axes[0].set_ylabel("Share of verdict's class gap (%)", fontsize=7.5)
axes[0].set_yticks([1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]); axes[0].set_yticklabels(["<0.0001", "0.001", "0.01", "0.1", "1", "10", "100"], fontsize=7)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, frameon=False, fontsize=6.8, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.01), handlelength=1.4)
fig.tight_layout(rect=(0, 0.09, 1, 1))
fig.savefig(FIG / "fig_steer_layers.png", dpi=220); fig.savefig(FIG / "fig_steer_layers.pdf")
print("saved fig_steer_layers")
