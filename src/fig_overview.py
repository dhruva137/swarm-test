"""Figure 1: the three findings at a glance (every value read from results/*.json)."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figs"
BLUE, ORANGE, AQUA, INK, INK2, GRID, MUTE = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#e6e5e0", "#9b9a94"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False})
PR = json.load(open(ROOT / "results" / "partner_retrieval_causal.json"))
LV = json.load(open(ROOT / "results" / "llm_v2.json"))
SM = json.load(open(ROOT / "results" / "sim_eval_sim_1p7b.json"))
fig, ax = plt.subplots(1, 3, figsize=(7.3, 2.6), gridspec_kw=dict(width_ratios=[1.12, 1.2, 1.0]))
pct = lambda v: f"{v:.1f}%" if v < 1 else f"{v:.0f}%"

# (a) logs: top-1 partner retrieval among ~8,000 earlier posts
a = ax[0]
rows = [("Text (frozen encoder)", "content-emb", BLUE), ("Same page", "page", MUTE), ("Page + IP + recency", "metadata", MUTE),
        ("Agent history", "agent-history", ORANGE), ("Random", "random", MUTE)]
for i, (lab, k, col) in enumerate(rows):
    y0 = len(rows) - 1 - i
    v, vn = PR[k]["hit@1"] * 100, PR[k]["novel_hit@1"] * 100
    a.barh(y0 + 0.18, v, height=0.34, color=col)
    a.barh(y0 - 0.18, vn, height=0.34, color=col, alpha=0.38)
    a.text(v + 1.2, y0 + 0.18, pct(v), va="center", fontsize=7.3, color=INK)
a.set_yticks(range(len(rows))); a.set_yticklabels([r[0] for r in rows][::-1], fontsize=7.3); a.tick_params(axis="y", length=0)
a.set_xlim(0, 85); a.set_xlabel("True partner ranked first (%)", fontsize=7.3); a.grid(axis="x", color=GRID, lw=0.6); a.set_axisbelow(True)
a.set_title("(a) Logs: words find partners", fontsize=8.2, color=INK2, loc="left")
a.text(84, 0.05, "faded: posts with no\nnear-copy partner", fontsize=6.2, color=INK2, ha="right", va="center")

# (b) monitor: verbal answer vs probe on the same activations
b = ax[1]
labels, xs = [], []
for gi, mdl in enumerate(["Qwen3-1.7B", "Qwen3-4B"]):
    vals = [LV[f"llm|{mdl}_neutral_meta"]["auroc_hard"], LV[f"llm|{mdl}_primed_meta"]["auroc_hard"], LV[f"probe|{mdl}_primed_meta"]["auroc_hard"]]
    for j, (v, col, lab) in enumerate(zip(vals, [MUTE, ORANGE, AQUA], ["neutral verdict", "framed verdict", "framed probe"])):
        x = gi * 1.3 + (j - 1) * 0.38
        b.bar(x, v, width=0.33, color=col)
        b.text(x, v + 0.006, f"{v:.2f}", ha="center", fontsize=6.6, color=INK)
        b.text(x, 0.512, lab.replace("\n", " "), rotation=90, ha="center", va="bottom", fontsize=5.9, color="white")
    xs.append(gi * 1.3); labels.append(mdl)
b.set_xticks(xs); b.set_xticklabels(labels, fontsize=7.3); b.tick_params(axis="x", length=0); b.set_ylim(0.5, 0.9)
b.set_ylabel("Linkage AUROC (hard negatives)", fontsize=7.3); b.grid(axis="y", color=GRID, lw=0.6); b.set_axisbelow(True)
b.set_title("(b) Monitors: probe beats verdict", fontsize=8.2, color=INK2, loc="left")

# (c) swarm: told about a content monitor (Qwen3-1.7B, 4 paired seeds)
c = ax[2]
m = SM["means"]
items = [("Task\nsuccess (%)", m["task"]["open"] * 100, m["task"]["covert"] * 100),
         ("Linkability\n(AUROC x100)", m["auc_emb"]["open"] * 100, m["auc_emb"]["covert"] * 100)]
for i, (lab, o, w) in enumerate(items):
    c.bar(i - 0.17, o, width=0.32, color=BLUE); c.bar(i + 0.17, w, width=0.32, color=ORANGE)
    c.text(i - 0.17, o + 2, f"{o:.0f}", ha="center", fontsize=7, color=INK); c.text(i + 0.17, w + 2, f"{w:.0f}", ha="center", fontsize=7, color=INK)
c.set_xticks([0, 1]); c.set_xticklabels([it[0] for it in items], fontsize=6.8); c.tick_params(axis="x", length=0); c.set_ylim(0, 125)
c.set_yticks([0, 25, 50, 75, 100]); c.grid(axis="y", color=GRID, lw=0.6); c.set_axisbelow(True)
c.set_title("(c) Swarms: hiding costs", fontsize=8.2, color=INK2, loc="left")
c.legend([plt.Rectangle((0, 0), 1, 1, color=BLUE), plt.Rectangle((0, 0), 1, 1, color=ORANGE)], ["not told", "told"],
         frameon=False, fontsize=6.6, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02), handlelength=1.0, columnspacing=1.0)
fig.tight_layout(w_pad=1.0)
fig.savefig(FIG / "fig_overview.png", dpi=220); fig.savefig(FIG / "fig_overview.pdf")
print("saved fig_overview")
