# Discover what the Kaggle model proxy (kaggle_benchmarks) exposes. Prints names only, never key values.
import os, json, subprocess, traceback
out = {}
out["proxy_env_names"] = sorted(k for k in os.environ if "PROXY" in k or "MODEL" in k or "BENCH" in k)
try:
    import kaggle_benchmarks as kbench
    out["kbench_version"] = getattr(kbench, "__version__", "?")
    out["kbench_dir"] = [x for x in dir(kbench) if not x.startswith("_")]
    for attr in ["llm", "llms", "models", "judge_llm"]:
        if hasattr(kbench, attr):
            obj = getattr(kbench, attr)
            out[f"attr_{attr}"] = repr(obj)[:500]
            try:
                out[f"attr_{attr}_dir"] = [x for x in dir(obj) if not x.startswith("_")][:80]
            except Exception:
                pass
    try:
        out["llm_prompt"] = str(kbench.llm.prompt("Reply with exactly: OK"))[:200]
    except Exception:
        out["llm_prompt_err"] = traceback.format_exc()[-800:]
    try:
        out["llms_keys"] = list(kbench.llms.keys())[:100]
    except Exception:
        out["llms_keys_err"] = traceback.format_exc()[-500:]
except Exception:
    out["import_err"] = traceback.format_exc()[-1500:]
out["pip_kaggle"] = subprocess.run("pip list 2>/dev/null | grep -i -E 'kaggle|openai|genai'", shell=True,
                                   capture_output=True, text=True).stdout
print(json.dumps(out, indent=1))
json.dump(out, open("/kaggle/working/probe.json", "w"), indent=1)
