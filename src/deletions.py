"""How does the swarm respond to moderator deletions?

For each deleted page with at least one stored agent edit before the deletion:
  - recreation: any agent edit on the same page within H hours after deletion
  - who: same label as the last pre-deletion editor, or a different label
  - migration: a near-copy (emb cos >= 0.95) of the last pre-deletion text appears on a *different* page
    within H hours after deletion, by any agent
Control: matched pseudo-deletions on pages that were never deleted, placed right after a real edit,
matched on wiki and prior activity (edits in the previous 24h), and on time (+/-48h).
"""
import gzip, json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
H = [1, 6, 24]
df = pd.read_parquet(ROOT / "data" / "edits.parquet")
emb = np.load(ROOT / "data" / "emb_bge.npy")
df["agent"] = np.where(df.label != "", df.label, "anon:" + df.ip16.fillna("?"))
df["ts"] = df.time.astype("int64") // 10**9
ev = pd.DataFrame([json.loads(l) for l in gzip.open(ROOT / "data" / "raw" / "events.jsonl.gz", "rt", encoding="utf-8")])
dele = ev[ev.event_type == "delete"].copy()
dele["ts"] = pd.to_datetime(dele.time, utc=True).astype("int64") // 10**9
deleted_pages = set(dele.page_key)
by_page = {k: g.sort_values("ts") for k, g in df.groupby("page_key")}
ts_all = df.ts.values
order = np.argsort(ts_all)


def outcome(page, t0, last):
    g = by_page[page]
    after = g[(g.ts > t0)]
    r = {"prior24": int(((g.ts > t0 - 86400) & (g.ts <= t0)).sum())}
    for h in H:
        w = after[after.ts <= t0 + h * 3600]
        r[f"recreate_{h}h"] = int(len(w) > 0)
        r[f"recreate_other_{h}h"] = int((w.agent != last.agent).any())
        r[f"recreate_same_{h}h"] = int((w.agent == last.agent).any())
        # migration: near-copy of last text elsewhere within h hours, excluding copies that existed before t0
        lo, hi = np.searchsorted(ts_all[order], [t0, t0 + h * 3600], side="right")
        cand = order[lo:hi]
        cand = cand[df.page_key.values[cand] != page]
        sims = emb[cand] @ emb[last.eid] if len(cand) else np.array([])
        r[f"migrate_{h}h"] = int((sims >= 0.95).any())
        r[f"migrate_other_{h}h"] = int(((sims >= 0.95) & (df.agent.values[cand] != last.agent)).any()) if len(cand) else 0
    return r


real = []
for d in dele.itertuples():
    if d.page_key not in by_page:
        continue
    g = by_page[d.page_key]
    prev = g[g.ts <= d.ts]
    if len(prev) == 0:
        continue
    last = prev.iloc[-1]
    real.append(dict(page=d.page_key, wiki=d.wiki, t0=d.ts, **outcome(d.page_key, d.ts, last)))
R = pd.DataFrame(real).drop_duplicates(["page", "t0"])
# collapse bursts of deletions of the same page within 1 minute
R = R.sort_values(["page", "t0"]); R = R[R.groupby("page").t0.diff().fillna(1e9) > 60]

rng = np.random.default_rng(0)
ctrl_pool = df[~df.page_key.isin(deleted_pages)]
ctrl = []
for r in R.itertuples():
    pool = ctrl_pool[(ctrl_pool.wiki == r.wiki) & (np.abs(ctrl_pool.ts - r.t0) < 48 * 3600)]
    if len(pool) == 0:
        continue
    for _ in range(3):
        e = pool.iloc[rng.integers(len(pool))]
        t0 = e.ts + 1  # pseudo-deletion right after a real edit, like real deletions
        o = outcome(e.page_key, t0, e)
        ctrl.append(dict(page=e.page_key, wiki=r.wiki, t0=t0, **o))
C = pd.DataFrame(ctrl)


def boot_diff(a, b, n=2000):
    a, b = np.asarray(a), np.asarray(b)
    d = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(n)]
    return float(np.quantile(d, .025)), float(np.quantile(d, .975))


out = {"n_deletions_with_prior_edit": len(R), "n_control": len(C),
       "prior24_real_median": float(R.prior24.median()), "prior24_ctrl_median": float(C.prior24.median())}
print(out)
for col in [c for c in R.columns if c.startswith(("recreate", "migrate"))]:
    a, b = R[col].mean(), C[col].mean()
    out[col] = dict(deleted=a, control=b, diff_ci=boot_diff(R[col], C[col]))
    print(f"{col:22s} deleted={a:.3f} control={b:.3f} diff95={np.round(out[col]['diff_ci'],3)}")
json.dump(out, open(ROOT / "results" / "deletions.json", "w"), indent=1, default=float)
