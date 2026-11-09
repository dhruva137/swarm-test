# Portability notes

Scanned `src/` and `kaggle/` (*.py, *.json) for absolute Windows paths and user-specific strings
(`C:/Users`, `C:\Users`, `DHRUVA`, `AppData`, any drive-letter path). No code was edited.

Every script in `src/` resolves its project root from its own location (`ROOT = Path(__file__).resolve().parents[1]`),
so the repository can live anywhere. The Kaggle scripts use `/kaggle/working` (override with env var `OUT`) and `/kaggle/input`.

## Occurrences

| Location | Issue |
|---|---|
| `src/build_pdf.py:45` | Hard-coded Chrome path `C:\Program Files\Google\Chrome\Application\chrome.exe` (Windows-only; used to print paper.html to PDF). Only affects the superseded v1 PDF build. |

No `C:/Users`, `C:\Users` or `DHRUVA` strings were found.

## Other portability issues spotted (not path literals)

- Kaggle kernel and dataset ids are tied to the author's Kaggle account (`dhruvapgowda/...`) in every `kaggle/*/kernel-metadata.json` and `kaggle/*_data/dataset-metadata.json`, and in the `json.dump({... "id": "dhruvapgowda/..."})` calls of `src/export_llm_pairs.py:27`, `src/export_para.py:16`, `src/export_v2.py:32`, `src/st_linkage.py:71`. Replace with your own username before pushing.
- `src/transfer_eval.py:29` expects `results/k_swarm-st-acts/ST_Qwen3-4B_primed_content.npz` (Qwen3-4B on SwarmTraces); this activation file is not present in the working tree, so only the Qwen3-1.7B variant (`results/local_st/`) is reproducible as shipped (`results/transfer_Qwen3-1.7B.json`).
- `src/probe_eval.py:24` reads `results/k_swarm-probe-acts/acts_{tag}.npz` (v1 file naming) whereas the Kaggle job now writes `Qwen3-<size>_<style>_<meta>.npz`; this script is superseded by `src/llm_v2_eval.py`.
- `src/mech.py`, `src/steer.py` and `src/probe_eval.py` write `results/mech.json`, `results/steer.json`, `results/probe.json`; none of these outputs existed when this release folder was assembled.
- `src/embed.py` assumes a local CUDA GPU (documented as an RTX 5050, 8 GB).
- Scripts need `data/raw/*` and `data/raw_st/r.jsonl.gz` in place (see README, section Data); `data/` is git-ignored.
