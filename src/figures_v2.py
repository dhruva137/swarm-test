"""Figures for the v2 (audit-corrected) results."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figs"; FIG.mkdir(exist_ok=True)
BLUE, ORANGE, AQUA, INK, INK2, GRID, MUTE = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#e6e5e0", "#9b9a94"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False})


def save(fig, name):
    fig.tight_layout(); fig.savefig(FIG / f"{name}.png", dpi=200); fig.savefig(FIG / f"{name}.pdf")


def fig_fields():
    R = json.load(open(ROOT / "results" / "eval_v2.json"))
    C = R["main"]["conditions"]; A = R["main"]["agent_disjoint"]; N = R["near-copies removed (both classes)"]["conditions"]
    rows = [("All fields", C["all fields"]), ("Content only", C["content only (SwarmTraces-like)"]),
            ("Metadata only (gateway view)", C["metadata only (gateway view)"]),
            ("Time only", C["only time"]), ("Location (page) only", C["only location"]),
            ("Identity (self-chosen name) only", C["only identity"]), ("Network (/16 IP) only", C["only network"])]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True, gridspec_kw=dict(width_ratios=[1.35, 1]))
    ys = list(range(len(rows)))[::-1]
    for (lab, r), y in zip(rows, ys):
        c = BLUE if lab.startswith(("All", "Content")) else (ORANGE if lab.startswith("Metadata") else INK2)
        axes[0].plot(r["auroc_hard_ci"], [y, y], color=c, lw=2, solid_capstyle="round")
        axes[0].plot(r["auroc_hard"], y, "o", ms=6, color=c, mec="white", mew=1.2, zorder=3)
        axes[0].text(r["auroc_hard_ci"][1] + 0.008, y, f"{r['auroc_hard']:.2f}", va="center", fontsize=8, color=INK)
        axes[1].barh(y, r["tpr1_hard"] * 100, color=c, height=0.55)
        axes[1].text(r["tpr1_hard"] * 100 + 0.8, y, f"{r['tpr1_hard']*100:.0f}%", va="center", fontsize=8, color=INK)
    axes[0].set_yticks(ys); axes[0].set_yticklabels([r[0] for r in rows], fontsize=8.5); axes[0].tick_params(axis="y", length=0)
    axes[0].axvline(0.5, color=INK2, lw=0.8, ls=":"); axes[0].set_xlim(0.45, 0.9)
    axes[0].set_xlabel("Linkage AUROC (hard negatives, 95% CI)"); axes[1].set_xlabel("Recall at 1% false-positive rate")
    axes[1].set_xlim(0, 48)
    for ax in axes:
        ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True); ax.spines["left"].set_visible(False)
    save(fig, "fig_fields")


def fig_retrieval():
    R = json.load(open(ROOT / "results" / "partner_retrieval_causal.json"))
    K = [1, 5, 10, 50, 100]
    series = [("content-emb", "Content (URL-free text, frozen encoder)", BLUE, "-"), ("metadata", "Metadata (page > IP > time)", ORANGE, "-"),
              ("agent-history", "Agent history", AQUA, "-"), ("ip", "IP prefix", INK2, "--"),
              ("random", "Random", MUTE, ":")]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
    for ax, pre, title in [(axes[0], "hit", f"All coordinated posts (n={R['n_queries']:,})"),
                           (axes[1], "novel_hit", f"Posts with no near-copy partner (n={R['n_novel']:,})")]:
        for key, lab, c, ls in series:
            v = [R[key][f"{pre}@{k}"] * 100 for k in K]
            ax.plot(K, v, color=c, lw=2, ls=ls, marker="o", ms=4, label=lab)
            if key in ("content-emb", "metadata", "agent-history"):
                ax.text(K[0] * 0.82, v[0], f"{v[0]:.0f}%", ha="right", va="center", fontsize=7.5, color=c)
        ax.set_xscale("log"); ax.set_xticks(K); ax.set_xticklabels([str(k) for k in K]); ax.set_xlim(0.6, 130)
        ax.set_xlabel("Top-k candidates returned"); ax.set_title(title, fontsize=8.5, color=INK2)
        ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    axes[0].set_ylabel("% with a true partner in top k"); axes[0].set_ylim(0, 100)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=7.5, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.12, 1, 1)); fig.savefig(FIG / "fig_retrieval.png", dpi=200); fig.savefig(FIG / "fig_retrieval.pdf")


def fig_knowsay():
    R = json.load(open(ROOT / "results" / "llm_v2.json"))
    models = [m for m in ["Qwen3-1.7B", "Qwen3-4B", "Qwen3-8B"] if f"llm|{m}_primed_meta" in R]
    conds = [("neutral_meta", "neutral prompt"), ("primed_meta", "collusion-framed"), ("primed_content", "framed, no metadata")]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw=dict(width_ratios=[1.5, 1]))
    labels, y = [], 0
    for m in models:
        for c, cl in conds:
            k = f"{m}_{c}"
            if f"llm|{k}" not in R: continue
            v, p = R[f"llm|{k}"], R[f"probe|{k}"]
            axes[0].plot([v["auroc_hard"], p["auroc_hard"]], [y, y], color=GRID, lw=3, zorder=1)
            axes[0].plot(v["auroc_hard"], y, "o", ms=7, color=ORANGE, mec="white", mew=1.2, zorder=3, label="what it says (verbal)" if y == 0 else None)
            axes[0].plot(p["auroc_hard"], y, "o", ms=7, color=BLUE, mec="white", mew=1.2, zorder=3, label="what it knows (linear probe)" if y == 0 else None)
            axes[1].barh(y, R[f"yesrate_neg|{k}"] * 100, color=ORANGE, height=0.55)
            axes[1].text(R[f"yesrate_neg|{k}"] * 100 + 1.5, y, f"{R[f'yesrate_neg|{k}']*100:.0f}%", va="center", fontsize=7.5, color=INK)
            labels.append(f"{m.replace('Qwen3-', '')} · {cl}"); y += 1
    for ax in axes:
        ax.set_yticks(range(len(labels))); ax.invert_yaxis()
        ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True); ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    axes[0].set_yticklabels(labels, fontsize=8); axes[1].set_yticklabels([])
    tf = json.load(open(ROOT / "results" / "llm_v2.json"))["tfidf_1500"]["auroc_hard"]
    axes[0].axvline(tf, color=INK2, lw=0.8, ls="--"); axes[0].text(tf + 0.003, -0.6, "text similarity", fontsize=7, color=INK2)
    axes[0].set_xlabel("Linkage AUROC (hard negatives)")
    axes[1].set_xlabel("Unrelated pairs called coordinated"); axes[1].set_xlim(0, 115)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=7.5, loc="lower center", ncol=2, bbox_to_anchor=(0.35, -0.02))
    fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(FIG / "fig_knowsay.png", dpi=200); fig.savefig(FIG / "fig_knowsay.pdf")


def fig_attack_transfer():
    I = json.load(open(ROOT / "results" / "recruit_null.json"))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), gridspec_kw=dict(width_ratios=[1.25, 1]))
    order = ["Qwen3-1.7B verbal [primed prompt]", "Qwen3-4B verbal [security prompt]", "Qwen3-8B verbal [security prompt]", "Qwen3-8B verbal [neutral prompt]"]
    nice = ["1.7B · collusion-framed", "4B · collusion-framed", "8B · collusion-framed", "8B · neutral"]
    labels, y = [], 0
    for key, lab_ in zip(order, nice):
        if key not in I: continue
        for cname, c, off in [("relay vs nothing, unlinked", ORANGE, -0.18), ("relay vs neutral, unlinked", MUTE, 0.18)]:
            r = I[key][cname]
            axes[0].barh(y + off, r["mean"], height=0.34, color=c, label=("vs. no sentence" if "nothing" in cname else "vs. neutral swarm sentence") if y == 0 else None)
            axes[0].errorbar(r["mean"], y + off, xerr=[[r["mean"] - r["ci"][0]], [r["ci"][1] - r["mean"]]], color=INK, lw=0.8, capsize=2)
        labels.append(lab_); y += 1
    axes[0].axvline(0, color=INK2, lw=0.8)
    axes[0].set_yticks(range(len(labels))); axes[0].set_yticklabels(labels, fontsize=8); axes[0].invert_yaxis()
    axes[0].set_xlabel("Shift in log-odds of 'coordinated' for UNRELATED pairs\nafter one real swarm relay sentence")
    axes[0].set_title("(a) Swarm text frames unrelated agents", fontsize=9, color=INK2, loc="left")
    axes[0].legend(frameon=False, fontsize=7.5, loc="upper right")
    tf = sorted((ROOT / "results").glob("transfer_*.json"))
    if tf:
        T = json.load(open(tf[0]))
        items = [("Wiki-trained\nprobe", T["probe trained on wiki swarm (transfer)"], BLUE),
                 ("Verbal\nanswer", T["verbal judge (zero-shot)"], ORANGE),
                 ("TF-IDF", T["TF-IDF similarity (zero-shot)"], INK2),
                 ("In-swarm\nprobe", T["probe trained in-swarm (5-fold CV, upper bound)"], MUTE)]
        for i, (lab_, r, c) in enumerate(items):
            axes[1].bar(i, r["auroc"] - 0.5, bottom=0.5, color=c, width=0.62)
            axes[1].errorbar(i, r["auroc"], yerr=[[r["auroc"] - r["ci"][0]], [r["ci"][1] - r["auroc"]]], color=INK, lw=0.8, capsize=2)
            axes[1].text(i, r["ci"][1] + 0.01, f"{r['auroc']:.2f}", ha="center", fontsize=8, color=INK)
        axes[1].set_xticks(range(len(items))); axes[1].set_xticklabels([i[0] for i in items], fontsize=7.5)
        axes[1].set_ylim(0.5, 1.0); axes[1].set_ylabel("AUROC on the attack swarm")
        axes[1].set_title(f"(b) Transfer to the HF attack swarm", fontsize=9, color=INK2, loc="left")
    for ax in axes:
        ax.grid(axis="x" if ax is axes[0] else "y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    axes[0].spines["left"].set_visible(False); axes[0].tick_params(axis="y", length=0)
    save(fig, "fig_attack_transfer")


def fig_sim():
    srcs = [("results/sim_eval_k_swarm-sim.json", "Qwen3-8B"), ("results/sim_eval_sim_1p7b.json", "Qwen3-1.7B")]
    srcs = [(ROOT / p, m) for p, m in srcs if (ROOT / p).exists()]
    if not srcs: return
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=False)
    xs = {"Qwen3-8B": 0, "Qwen3-1.7B": 1}
    for path, model in srcs:
        D = {(d["cond"], d["seed"]): d for d in json.load(open(path))["per_swarm"]}
        seeds = sorted({s for _, s in D})
        x0 = xs[model] * 3
        for sd in seeds:
            if ("open", sd) in D and ("covert", sd) in D:
                for ax, key in [(axes[0], "task"), (axes[1], "leak")]:
                    a, b = D[("open", sd)][key] * 100, D[("covert", sd)][key] * 100
                    ax.plot([x0, x0 + 1], [a, b], color=GRID, lw=1.2, zorder=1)
                    ax.scatter([x0], [a], color=BLUE, s=26, zorder=3, edgecolor="white", lw=0.6)
                    ax.scatter([x0 + 1], [b], color=ORANGE, s=26, zorder=3, edgecolor="white", lw=0.6)
    for ax in axes:
        ticks, labs = [], []
        for path, model in srcs:
            x0 = xs[model] * 3; short = model.replace("Qwen3-", "")
            ticks += [x0, x0 + 1]; labs += [short + "\nnot told", short + "\ntold"]
        ax.set_xticks(ticks); ax.set_xticklabels(labs, fontsize=7.5); ax.set_xlim(-0.6, max(ticks) + 0.6)
        ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    axes[0].axhline(100 / 12, color=INK2, lw=0.8, ls=":"); axes[0].text(-0.5, 100 / 12 + 2, "floor: own code only", fontsize=7, color=INK2)
    axes[0].set_ylabel("Task success (% of codes known)"); axes[1].set_ylabel("Posts leaking a true code (%)")
    axes[0].set_title("(a) Coordination collapses", fontsize=9, color=INK2, loc="left")
    axes[1].set_title("(b) …while true codes still leak", fontsize=9, color=INK2, loc="left")
    axes[1].set_ylim(0, 105)
    save(fig, "fig_sim")


def fig_geometry():
    import numpy as np
    G = json.load(open(ROOT / "results" / "geometry.json")); Z = np.load(ROOT / "results" / "geometry_plane.npz")
    xN, xP, yN, yP, yy = Z["xN"], Z["xP"], Z["yN"], Z["yP"], Z["y"]
    # common standardization (neutral moments) so both panels share axes and units
    mx, sx, my, sy = xN.mean(), xN.std(), yN.mean(), yN.std()
    fig = plt.figure(figsize=(7.2, 3.15))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.05], wspace=0.38)
    axs = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    for ax, x, yv, title in [(axs[0], xN, yN, "(a) Neutral prompt"), (axs[1], xP, yP, "(b) Collusion-framed prompt")]:
        X, Y = (x - mx) / sx, (yv - my) / sy
        for lab, c, z in [(0, ORANGE, 2), (1, BLUE, 3)]:
            m = yy == lab
            ax.scatter(X[m], Y[m], s=3, alpha=0.35, color=c, lw=0, zorder=z, label="linked" if lab else "unlinked")
            ax.scatter(X[m].mean(), Y[m].mean(), s=46, color=c, edgecolor=INK, lw=0.9, zorder=5, marker="D")
        ax.axhline((0 - my) / sy if False else 0, color=GRID, lw=0)
        ax.set_title(title, fontsize=8.5, color=INK2, loc="left")
        ax.set_xlabel("linkage axis (SD)", fontsize=8); ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
    axs[0].set_ylabel("answer axis YES−NO (SD)", fontsize=8)
    lo = min(((yN - my) / sy).min(), ((yP - my) / sy).min()); hi = max(((yN - my) / sy).max(), ((yP - my) / sy).max())
    lx = min(((xN - mx) / sx).min(), ((xP - mx) / sx).min()); hx = max(((xN - mx) / sx).max(), ((xP - mx) / sx).max())
    for ax in axs:
        ax.set_ylim(lo - 0.3, hi + 0.3); ax.set_xlim(lx - 0.3, hx + 0.3)
    axs[0].legend(frameon=False, fontsize=7, loc="upper left", markerscale=3)
    T2 = G["T2"]
    axs[1].text(0.98, 0.04, f"linkage-axis AUROC {T2['auc_linkage_axis_primed_using_neutral_axis']:.2f}\nanswer-axis AUROC {T2['auc_answer_axis_primed']:.2f}",
                transform=axs[1].transAxes, ha="right", va="bottom", fontsize=6.8, color=INK, bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5))
    axs[0].text(0.98, 0.04, f"linkage-axis AUROC {T2['auc_linkage_axis_neutral']:.2f}\nanswer-axis AUROC {T2['auc_answer_axis_neutral']:.2f}",
                transform=axs[0].transAxes, ha="right", va="bottom", fontsize=6.8, color=INK, bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5))
    ax = fig.add_subplot(gs[0, 2])
    G2 = json.load(open(ROOT / "results" / "geometry2.json"))["Qwen3-1.7B"]["G1"]
    Ls = G2["layers"]
    ax.plot(Ls, G2["N_on_N"], color=BLUE, lw=1.8, marker="o", ms=3, label="neutral axis · neutral")
    ax.plot(Ls, G2["N_on_P"], color=BLUE, lw=1.8, ls="--", marker="o", ms=3, label="neutral axis · framed")
    ax.plot(Ls, G2["P_on_P"], color=AQUA, lw=1.4, marker="s", ms=3, label="framed axis · framed")
    ax.axhline(G2["verbal_N"], color=ORANGE, lw=1.2, label="verbal, neutral")
    ax.axhline(G2["verbal_P"], color=ORANGE, lw=1.2, ls="--", label="verbal, framed")
    ax.set_xlabel("layer", fontsize=8); ax.set_ylabel("held-out AUROC (hard)", fontsize=8)
    ax.set_title("(c) Read-out by layer", fontsize=8.5, color=INK2, loc="left")
    ax.legend(frameon=False, fontsize=5.6, loc="upper center", bbox_to_anchor=(0.42, -0.2), ncol=2, columnspacing=0.8, handlelength=1.6); ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.30, top=0.90)
    fig.savefig(FIG / "fig_geometry.png", dpi=220); fig.savefig(FIG / "fig_geometry.pdf")


def fig_mech():
    p = ROOT / "results" / "mech.json"
    if not p.exists(): return
    M = json.load(open(p))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
    for ax, model in zip(axes, ["Qwen3-1.7B", "Qwen3-4B"]):
        for style, ls in [("neutral_meta", "-"), ("primed_meta", "--")]:
            key = f"{model}|{style}"
            if key not in M: continue
            r = M[key]
            for probe, c, nm in [("linear", BLUE, "linear"), ("mlp", ORANGE, "MLP"), ("quad", INK2, "quadratic")]:
                ax.plot(r["layers"], r[probe], color=c, ls=ls, lw=1.5, marker="o", ms=2.5,
                        label=f"{nm}, {'neutral' if style.startswith('neutral') else 'framed'}")
        ax.set_title(model, fontsize=8.5, color=INK2, loc="left"); ax.set_xlabel("layer"); ax.grid(color=GRID, lw=0.5)
    axes[0].set_ylabel("held-out linkage AUROC (hard)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=7, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.14, 1, 1)); fig.savefig(FIG / "fig_mech.png", dpi=200); fig.savefig(FIG / "fig_mech.pdf")


if __name__ == "__main__":
    fig_fields(); fig_retrieval(); fig_knowsay(); fig_attack_transfer(); fig_sim(); fig_geometry(); fig_mech(); print("ok")


def fig_steer():
    import numpy as np
    p = ROOT / "results" / "steer3.json"
    if not p.exists(): return
    S = json.load(open(p))["summary"]
    fig, ax = plt.subplots(figsize=(5.6, 2.3))
    rows = [("neutral", "neutral prompt"), ("primed", "collusion-framed")]
    for i, (st, lab) in enumerate(rows):
        y0 = i * 1.0
        nulls = [abs(S[f"{st}|perm{k}|2"]["mean"]) for k in range(5)] + [abs(S[f"{st}|perm{k}|8"]["mean"]) for k in range(5)]
        ax.barh(y0 + 0.22, S[f"{st}|verbal|2"]["mean"], height=0.2, color=ORANGE, label="verdict-predictive direction" if i == 0 else None)
        ax.barh(y0, max(S[f"{st}|linkage|2"]["mean"], 1e-4), height=0.2, color=BLUE, label="linkage direction" if i == 0 else None)
        ax.barh(y0 - 0.22, max(nulls), height=0.2, color=MUTE, label="shuffled-label nulls (max |effect|)" if i == 0 else None)
        for yy, v in [(y0 + 0.22, S[f"{st}|verbal|2"]["mean"]), (y0, S[f"{st}|linkage|2"]["mean"]), (y0 - 0.22, max(nulls))]:
            ax.text(max(v, 1e-4) * 1.15, yy, f"{v:+.4f}" if abs(v) < 0.1 else f"{v:+.2f}", va="center", fontsize=7, color=INK)
    ax.set_xscale("log"); ax.set_xlim(5e-4, 8)
    ax.set_yticks([0, 1]); ax.set_yticklabels([r[1] for r in rows], fontsize=8)
    ax.set_xlabel("change in verdict log-odds per SD of push (log scale)", fontsize=8)
    ax.legend(frameon=False, fontsize=6.8, loc="lower right"); ax.grid(axis="x", color=GRID, lw=0.5); ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    fig.tight_layout(); fig.savefig(FIG / "fig_steer.png", dpi=220); fig.savefig(FIG / "fig_steer.pdf")


fig_steer()
