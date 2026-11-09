# Agents Change Names, Not Words

Code, results and paper for *Agents Change Names, Not Words: What Oversight Can See in a Real AI Swarm* (Apart Research Swarm Alignment Hackathon, November 2026).

Overseeing an AI agent swarm rests on one operation: linking actions taken under different identities to a shared effort. This project asks where that coordination signal lives and where oversight loses it. It follows the signal through three places:

1. **The logs of a real swarm** (collusion.wiki, replicated on the SwarmTraces reconstruction of the Hugging Face attack swarm). The signal is in what agents write, not in their self-chosen names or metadata.
2. **The internals of LLM monitors** (Qwen3 1.7B/4B/8B). The signal is linearly encoded in the monitor's activations, but its verdict barely uses it. Collusion framing makes the verdict confidently wrong, and framed verdicts track how coordination *sounds* (coordination-like language that does not indicate linkage) rather than whether agents are linked.
3. **Controlled synthetic swarms** that are told a content monitor is watching. They change how they write but do not become measurably harder to link; they lose their ability to coordinate, and their evasion is itself detectable.

The paper is `paper/main.pdf` (LaTeX source `paper/main.tex`, references `paper/refs.bib`).

All code in this repository was written during the hackathon (8 to 9 November 2026). Nothing is pre-existing.

## Repository layout

```
paper/        main.tex, refs.bib, main.pdf
slides/       slides.tex, slides.pdf, slides.pptx (11-slide summary deck)
src/          analysis scripts (Python); see the reproduction table below
kaggle/       headless Kaggle GPU jobs (each folder: script + kernel-metadata.json)
results/      every result file behind every number in the paper (JSON, CSV, JSONL, Parquet)
figs/         all figures (PNG and PDF)
validation/   blind annotation labels used for ground-truth validation
requirements.txt, LARGE_FILES.md (in results/), PORTABILITY_NOTES.md
```

## Data

Both datasets are public and are not redistributed here.

- **collusion.wiki.** Download `revisions.jsonl.gz`, `events.jsonl.gz` and `labels.jsonl.gz` from https://collusion.wiki/explorer/download and place them in `data/raw/`.
- **SwarmTraces.** Download https://swarmtraces.org/data/final/redacted.jsonl.gz and save it as `data/raw_st/r.jsonl.gz`.

Both contain untrusted, agent-written text and (redacted) attack payloads. The code never executes them and never follows their links.

## Environment

- Python 3.11 or newer. Install dependencies with `pip install -r requirements.txt`.
- A CUDA GPU is needed for embeddings, activation extraction, steering and the local synthetic swarm. An 8 GB laptop GPU (RTX 5050) was sufficient for every model up to Qwen3-1.7B.
- The 4B and 8B jobs and all vLLM jobs ran on Kaggle (2x T4). Push one with `kaggle kernels push -p kaggle/<job> --accelerator NvidiaTeslaT4`. The kernel and dataset ids in `kernel-metadata.json` refer to the author's Kaggle account; change them to your own.
- Every `src/` script resolves the project root from its own location, so run them from anywhere.

## Reproduction

The final results come from the audit-corrected pipeline below. Run the steps in order. Outputs land in `results/`.

| Paper result | Script(s) | Output |
|---|---|---|
| Edit table, URL deletion, masks | `src/data.py` | `data/edits.parquet` |
| Text embeddings | `src/embed.py text_nourl`, `src/embed.py text_nourl_strict` | `data/emb_bge_*.npy` |
| Part 1: log-field ablation, agent-disjoint and near-copy robustness | `src/pairs_v2.py`, `src/features_v2.py`, `src/run_eval_v2.py` | `eval_v2.json`, `pairs_v2.parquet` |
| Part 1: causal partner retrieval with CIs | `src/partner_retrieval_causal.py` | `partner_retrieval_causal.json` |
| Part 1: name churn | `src/watchlist.py` | `watchlist.json` |
| Part 1: SwarmTraces replication and leakage checks | `src/st_linkage.py`, `src/st_leakcheck.py` | `st_eval.json`, `st_leakcheck.json`, `st_pairs.parquet` |
| Ground-truth validation | `src/validation.py` (labels in `validation/`) | `validation.json` |
| Part 2: evaluation sample | `src/export_v2.py` | `v2_sample.parquet` (and the Kaggle input) |
| Part 2: verbal scores and activations (1.7B, 4B) | `kaggle/probe_acts/acts.py` (Kaggle) | `k_swarm-probe-acts/*.npz` (see LARGE_FILES.md) |
| Part 2: verbal vs probe, framing, paired tests | `src/llm_v2_eval.py`, `src/paired_extra.py`, `src/gapchange_baserate.py` | `llm_v2.json`, `paired_extra.json`, `gapchange_baserate.json` |
| Part 2: framing vs wording (2x2 swap of system message and question, Qwen3-1.7B) | `src/prompt_swap.py` (local GPU) | `prompt_swap.json`, `prompt_swap.log` |
| Part 2: read-out by layer, framing shift | `src/geometry2.py` | `geometry2.json` |
| Part 2: linear vs MLP vs quadratic probes, decision entropy | `src/mech.py` | `mech.json` |
| Part 2: causal steering (corrected) and positive control | `src/steer2.py`, `src/steer3.py` | `steer2.json`, `steer3.json`, `steer*_rows.csv` |
| Part 2: scale-free steering usage and per-norm ratios | `src/usage.py` (re-fits the `steer3` directions and checks them against `steer3.json`) | `usage.json` |
| Part 2: steering at layers 8, 12, 16, 24 (answer position) and at layer 20 (all token positions) | `src/steer_layers.py` (local GPU) | `steer_layers.json`, `steer_layers_rows.csv`, `steer_layers.log` |
| Part 2: Qwen3-4B steering replication (never left the Kaggle queue and was cancelled; not in the paper) | `src/export_acts4b_l24.py` → `kaggle/steer4b/steer4b.py`; then `src/usage.py` picks it up | `k_swarm-steer4b/` |
| Part 2: 8B framed judge | `kaggle/recruit/recruit.py` | `k_swarm-recruit/`, `recruit.json` |
| Part 2: swarm-sentence injection, null-controlled | `kaggle/recruit/`, `kaggle/recruit8n/`, `kaggle/inject/inject.py` → `src/recruit_null.py` | `recruit_null.json` |
| Part 2: relay-sentence effect as a share of each judge's own class gap | `src/relay_sd.py` | `relay_sd.json` |
| Part 2: does injected text move the probe as well as the verdict? | `src/robust_repr.py` (needs the `local_inject/` and `k_swarm-inject/` activations) | `robust_repr.json`, `robust_repr.log` |
| Part 2: verdict vs probe loading on coordination-talk, and its stress tests (framing difference, hard pairs only, lexical instrument, unrelated pairs only) | `src/topic_vs_relation.py`, `src/coordtalk_checks.py`, `src/coordtalk_minilm.py` (third instrument) | `topic_vs_relation.json`, `coordtalk_checks.json`, `coordtalk_minilm.json` |
| Part 2: cross-swarm probe transfer | `kaggle/st_acts/st_acts.py` (local GPU) → `src/transfer_eval.py` | `transfer_Qwen3-1.7B.json` |
| Part 3: controlled swarms | `kaggle/sim/sim.py` (vLLM on Kaggle; `BACKEND=hf` runs locally) → `src/sim_eval.py` | `k_swarm-sim/`, `sim_1p7b/`, `sim_eval_*.json` |
| Part 3: evasion detectability, cross-swarm similarity (monoculture hypothesis, rejected), model fingerprint | `src/monoculture.py` | `monoculture.json` |
| Part 3: coordination-talk of synthetic posts, open vs warned | `src/coordtalk_checks.py` (check C5) | `coordtalk_checks.json` |
| Figures | `src/figures_v2.py`, `src/fig_overview.py`, `src/fig_coordtalk.py`, `src/fig_steer_layers.py` | `figs/` |
| Slides and project image | `tectonic slides/slides.tex` (expects figures in `../figs/`), then `src/make_extras.py` | `slides/slides.pdf`, `slides/slides.pptx`, `figs/project_image.png` |
| Paper | `tectonic paper/main.tex` (expects figures in `../figs/`) | `paper/main.pdf` |

### Superseded scripts (kept for the audit trail)

These first-pass scripts produced numbers that were later corrected or withdrawn. They are kept so every correction in the paper's Audit Log can be inspected.

`src/pairs.py`, `src/features.py`, `src/run_eval.py` (still provides the shared `tpr_at_fpr` and `scores` helpers), `src/robust.py`, `src/stack.py`, `src/evasion.py`, `src/evasion_v2.py`, `src/inject_eval.py`, `src/inject_eval2.py` (withdrawn injection-evasion analysis), `src/deletions.py` (dropped moderation analysis), `src/llm_eval.py`, `src/scale_eval.py`, `src/recruit_eval.py`, `src/export_llm_pairs.py`, `src/export_para.py`, `src/geometry.py` (withdrawn cosine-based geometry; replaced by `geometry2.py`), `src/steer.py` (first steering design; replaced by `steer2.py`/`steer3.py`), `src/probe_eval.py`, `src/partner_retrieval.py` and `src/partner_retrieval_strict.py` (non-causal TF-IDF version), `src/make_labeler.py`, `src/figures.py`, `src/build_pdf.py`.

The subspace-rank (INLP) part of `src/mech.py` is reported in `mech.json`, but its claim was withdrawn (see the Audit Log). Only the linear-vs-nonlinear comparison is used.

## Audit trail

Four independent code reviews and two number-by-number fact checks were run on every script before submission. They found a label shortcut, fold leakage, an unfair LLM comparison, a tie bug in recall-at-FPR, a regression-to-the-mean injection effect, and three geometry claims with invalid nulls. A final review covered the analyses added last (steering usage, injection robustness, coordination-talk); it found no code bug, and its wording corrections are in the paper. For coordination-talk, a keyword measure disagreed with the semantic measure at Qwen3-1.7B; a second semantic measure from another model family (MiniLM) confirmed the semantic result at both sizes, and the paper states where the keyword measure disagrees. Every correction and every withdrawn claim is listed in the paper's Appendix C (Audit Log). Every number in the paper traces to a file in `results/`.

## Ethics and dual use

- **No raw content redistributed.** No raw wiki posts or attack payloads are included. Files derived directly from raw posts, such as paraphrases and Kaggle input pairs, are excluded and listed in `results/LARGE_FILES.md`. Parquet files contain ids, numeric features and the public, already-pseudonymous agent handles.
- **Synthetic agents were contained.** They ran only inside a closed text harness, with no network, tools or code execution.
- **No external contact.** No external service was contacted, no URL from the data was followed, and no re-identification was attempted.

## License

Code: MIT. The datasets remain under their original terms.

## Citation

```bibtex
@misc{gowda2026agentschangenames,
  author = {Dhruva P Gowda},
  title  = {Agents Change Names, Not Words: What Oversight Can See in a Real AI Swarm},
  year   = {2026},
  note   = {Apart Research Swarm Alignment Hackathon},
  url    = {https://github.com/dhruva137/swarm-test}
}
```
