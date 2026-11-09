# Smoke test: confirm 2xT4, vLLM install, model load, and batched throughput.
import subprocess, sys, time, json, os

def sh(cmd):
    print(f"\n$ {cmd}", flush=True)
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(r.stdout[-4000:], r.stderr[-4000:], flush=True)
    return r.returncode

report = {}
sh("nvidia-smi")
report["torch_before"] = subprocess.run("pip show torch | head -2", shell=True, capture_output=True, text=True).stdout
t0 = time.time()
report["vllm_install_rc"] = sh("pip install -q vllm")
report["vllm_install_s"] = round(time.time() - t0)
sh("pip show vllm torch | grep -E 'Name|Version'")

try:
    from vllm import LLM, SamplingParams
    t0 = time.time()
    llm = LLM(model="Qwen/Qwen3-8B", dtype="half", tensor_parallel_size=2,
              gpu_memory_utilization=0.90, max_model_len=4096, enforce_eager=True,
              disable_custom_all_reduce=True)
    report["load_s"] = round(time.time() - t0)
    tok = llm.get_tokenizer()
    prompts = []
    for i in range(64):
        msgs = [{"role": "system", "content": f"You are agent {i} in a team of workers sharing a filesystem."},
                {"role": "user", "content": "Describe in 3 sentences how you would split a log-analysis task with your teammates."}]
        prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False))
    sp = SamplingParams(max_tokens=200, temperature=0.7)
    t0 = time.time()
    outs = llm.generate(prompts, sp)
    dt = time.time() - t0
    ntok = sum(len(o.outputs[0].token_ids) for o in outs)
    report.update(gen_s=round(dt, 1), gen_tokens=ntok, tok_per_s=round(ntok / dt, 1),
                  sample=outs[0].outputs[0].text[:500])
except Exception as e:
    import traceback
    report["error"] = traceback.format_exc()[-3000:]

print(json.dumps(report, indent=2))
os.makedirs("/kaggle/working", exist_ok=True)
json.dump(report, open("/kaggle/working/smoke_report.json", "w"), indent=2)
