"""Paper figures from results/eval.json (static PNG/PDF for the report)."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
R = json.load(open(ROOT / "results" / "eval.json"))
FIG = ROOT / "figs"; FIG.mkdir(exist_ok=True)
BLUE, MUTED, INK, INK2, GRID = "#2a78d6", "#b7d3f6", "#0b0b0b", "#52514e", "#e6e5e0"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False})


def fig_conditions(split="hard"):
    C = R["conditions"]
    groups = [("Log view", [("all fields", "All fields"), ("content only (SwarmTraces-like)", "Content only (SwarmTraces-like)"),
                            ("metadata only (gateway view)", "Metadata only (gateway view)")]),
              ("Drop one field", [(f"drop {f}", f"− {f}") for f in ["content", "time", "identity", "network", "location"]]),
              ("Only one field", [(f"only {f}", f"{f} only") for f in ["content", "time", "location", "identity", "network"]])]
    fig, ax = plt.subplots(figsize=(6.0, 4.6), dpi=200)
    y, ticks, labels = 0, [], []
    for gi, (gname, items) in enumerate(groups):
        ax.text(0.302, y + 0.15, gname.upper(), fontsize=7.5, color=INK2, fontweight="bold", va="bottom")
        y -= 0.85
        for key, lab in items:
            s = C[key][split]
            c = BLUE if gi == 0 else INK2
            ax.plot(s["ci"], [y, y], color=c, lw=2, solid_capstyle="round")
            ax.plot(s["auroc"], y, "o", ms=6, color=c, mec="white", mew=1.2, zorder=3)
            ax.text(max(s["ci"][1], s["auroc"]) + 0.012, y, f"{s['auroc']:.3f}", va="center", fontsize=8, color=INK)
            ticks.append(y); labels.append(lab); y -= 1
        y -= 0.4
    ax.axvline(0.5, color=INK2, lw=0.8, ls=":"); ax.text(0.505, y + 0.5, "chance", fontsize=7.5, color=INK2)
    ax.set_yticks(ticks); ax.set_yticklabels(labels, fontsize=8.5); ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(0.3, 0.95); ax.set_ylim(y + 0.2, 1.0)
    ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.set_xlabel(f"Linkage AUROC, {'hard' if split == 'hard' else 'all'} negatives (95% CI)")
    fig.tight_layout(); fig.savefig(FIG / f"fig_conditions_{split}.png"); fig.savefig(FIG / f"fig_conditions_{split}.pdf")


def fig_clock():
    T = R["time_resolution"]
    labs = list(T.keys())
    fig, ax = plt.subplots(figsize=(4.2, 2.6), dpi=200)
    for split, c, ls in [("hard", BLUE, "-"), ("all", INK2, "--")]:
        v = [T[l][split]["auroc"] for l in labs]
        ax.plot(range(len(labs)), v, color=c, lw=2, ls=ls, marker="o", ms=4, label=f"{split} negatives")
    ax.set_xticks(range(len(labs))); ax.set_xticklabels(labs)
    ax.set_xlabel("Timestamp resolution kept in the log"); ax.set_ylabel("Linkage AUROC")
    ax.grid(axis="y", color=GRID, lw=0.6); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "fig_clock.png"); fig.savefig(FIG / "fig_clock.pdf")


def fig_monitors():
    """Same 1,500 pairs: cheap retrieval monitor vs LLM judges of growing size (and prompt variants if present)."""
    S = json.load(open(ROOT / "results" / "scale.json"))
    rows = [("Cheap monitor (embeddings + metadata)", S["cheap"]["hard"], BLUE),
            ("Embedding cosine alone", S["emb_cos"]["hard"], BLUE)]
    for m in ["Qwen3-8B", "Qwen3-14B-AWQ", "Qwen3-32B-AWQ"]:
        rows.append((f"{m.replace('-AWQ', '')} judge, content + meta", S[f"{m} | content+meta"]["hard"], INK2))
    pp = ROOT / "results" / "prompts.json"
    if pp.exists():
        Pj = json.load(open(pp))
        for v in ["neutral", "fewshot", "thinking"]:
            if v in Pj:
                rows.append((f"Qwen3-8B judge, {v} prompt", Pj[v]["hard"], INK2))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 0.42 * len(rows) + 0.9), dpi=200, sharey=True,
                             gridspec_kw=dict(width_ratios=[1.4, 1]))
    ys = list(range(len(rows)))[::-1]
    for (lab, s, c), y in zip(rows, ys):
        axes[0].plot(s["ci"], [y, y], color=c, lw=2, solid_capstyle="round")
        axes[0].plot(s["auroc"], y, "o", ms=6, color=c, mec="white", mew=1.2, zorder=3)
        axes[0].text(s["ci"][1] + 0.01, y, f"{s['auroc']:.2f}", va="center", fontsize=8, color=INK)
        axes[1].barh(y, s["recall_at_fpr1"] * 100, color=c, height=0.55)
        axes[1].text(s["recall_at_fpr1"] * 100 + 0.8, y, f"{s['recall_at_fpr1']*100:.0f}%", va="center", fontsize=8, color=INK)
    axes[0].set_yticks(ys); axes[0].set_yticklabels([r[0] for r in rows], fontsize=8.5); axes[0].tick_params(axis="y", length=0)
    axes[0].axvline(0.5, color=INK2, lw=0.8, ls=":"); axes[0].set_xlim(0.45, 1.0)
    axes[0].set_xlabel("Linkage AUROC, hard negatives"); axes[1].set_xlabel("Recall at 1% false-positive rate")
    axes[1].set_xlim(0, 55)
    for ax in axes:
        ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True); ax.spines["left"].set_visible(False)
    fig.tight_layout(); fig.savefig(FIG / "fig_monitors.png"); fig.savefig(FIG / "fig_monitors.pdf")


if __name__ == "__main__":
    fig_conditions("hard"); fig_conditions("all"); fig_clock()
    if (ROOT / "results" / "scale.json").exists():
        fig_monitors()
    print("figs ->", FIG)
