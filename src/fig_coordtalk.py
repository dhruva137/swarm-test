"""Figure: the verdict tracks how coordination sounds; the representation does not; warned swarms drop that register.
Inputs: results/topic_vs_relation.json (semantic instrument), results/coordtalk_checks.json (lexical instrument, sims).
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figs"
BLUE, ORANGE, INK, INK2, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e6e5e0"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False})
TV = json.load(open(ROOT / "results" / "topic_vs_relation.json"))
CK = json.load(open(ROOT / "results" / "coordtalk_checks.json"))
MN = json.load(open(ROOT / "results" / "coordtalk_minilm.json"))
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw=dict(width_ratios=[1.55, 1]))
ax = axes[0]
rows = [("Qwen3-1.7B", "neutral", "1.7B neutral"), ("Qwen3-1.7B", "primed", "1.7B framed"),
        ("Qwen3-4B", "neutral", "4B neutral"), ("Qwen3-4B", "primed", "4B framed")]
for i, (mdl, st, lab) in enumerate(rows):
    y = len(rows) - 1 - i
    for kind, col, off in [("llm", ORANGE, 0.2), ("probe", BLUE, -0.2)]:
        sem = TV[f"{kind}|{mdl}_{st}_meta"]
        lex = CK[f"{kind}|{mdl}"]["neutral" if st == "neutral" else "framed"]["C3_lexical"]
        mini = MN[f"{kind}|{mdl}"]["neutral" if st == "neutral" else "framed"]
        for e, yy, mk, fc in [((sem["beta_coordtalk"], sem["beta_coordtalk_ci"]), y + off + 0.09, "o", col),
                              ((mini["beta"], mini["ci"]), y + off, "^", col),
                              ((lex["beta"], lex["ci"]), y + off - 0.09, "s", "white")]:
            b, (lo, hi) = e
            ax.plot([lo, hi], [yy, yy], color=col, lw=1.2, solid_capstyle="round", zorder=2)
            ax.scatter([b], [yy], marker=mk, s=20, facecolor=fc, edgecolor=col, lw=1.0, zorder=3)
ax.axvline(0, color=INK2, lw=0.8)
ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[2] for r in rows][::-1], fontsize=8)
ax.tick_params(axis="y", length=0); ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
ax.set_xlabel("Loading on coordination-talk, beyond linkage\nand pair similarity (standardised coefficient)", fontsize=8)
ax.set_title("(a) The verdict and the probe disagree", fontsize=9, color=INK2, loc="left")
h = [plt.Line2D([], [], color=ORANGE, marker="o", lw=1.4, ms=5, label="verdict (YES$-$NO log-odds)"),
     plt.Line2D([], [], color=BLUE, marker="o", lw=1.4, ms=5, label="linear probe, same activations"),
     plt.Line2D([], [], color=INK2, marker="o", lw=0, ms=5, label="semantic (bge)"),
     plt.Line2D([], [], color=INK2, marker="^", lw=0, ms=5, label="semantic (MiniLM)"),
     plt.Line2D([], [], color=INK2, marker="s", mfc="white", lw=0, ms=5, label="keyword (TF-IDF)")]
fig.legend(handles=h, frameon=False, fontsize=6.7, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.01), handlelength=1.6)
ax = axes[1]
for x0, mdl, short in [(0, "Qwen3-8B", "8B"), (3, "Qwen3-1.7B", "1.7B")]:
    o = CK[f"C5_{mdl}_coordtalk"]
    for a, b in zip(o["per_seed_open"], o["per_seed_covert"]):
        ax.plot([x0, x0 + 1], [a, b], color=GRID, lw=1.2, zorder=1)
        ax.scatter([x0], [a], color=BLUE, s=24, zorder=3, edgecolor="white", lw=0.6)
        ax.scatter([x0 + 1], [b], color=ORANGE, s=24, zorder=3, edgecolor="white", lw=0.6)
    ax.text(x0 + 0.5, 0.468, f"lower in {o['seeds_lower']}/{o['n_seeds']} seeds", ha="center", fontsize=7, color=INK)
ax.set_xticks([0, 1, 3, 4]); ax.set_xticklabels(["8B\nnot told", "8B\ntold", "1.7B\nnot told", "1.7B\ntold"], fontsize=7.5)
ax.set_xlim(-0.6, 4.6); ax.set_ylim(0.46, 0.63); ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
ax.set_ylabel("Coordination-talk (cosine to\nrelay-sentence centroid)", fontsize=8)
ax.set_title("(b) Warned swarms drop it", fontsize=9, color=INK2, loc="left")
fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig(FIG / "fig_coordtalk.png", dpi=200); fig.savefig(FIG / "fig_coordtalk.pdf")
print("saved fig_coordtalk")
