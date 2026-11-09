"""Build a blind hand-labelling page (labeler.html) to validate the ground-truth pairs."""
import json, sys, html
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from pairs import build_pairs

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_parquet(ROOT / "data" / "edits.parquet")
pairs, _ = build_pairs(df)
pairs = pairs.reset_index().rename(columns={"index": "pair_idx"})
n = {"pos": 120, "easy": 40, "hard": 40}
sub = pd.concat([g.sample(n[k], random_state=7) for k, g in pairs.groupby("neg_type")]).sample(frac=1, random_state=7)
E = df.set_index("eid")
items = []
for r in sub.itertuples():
    side = lambda e: dict(agent=E.loc[e, "label"] or "(none)", page=E.loc[e, "page_key"], time=str(E.loc[e, "time"])[:19],
                          text=E.loc[e, "body"][:2500])
    items.append(dict(pair_idx=int(r.pair_idx), a=side(r.a), b=side(r.b)))
key = {int(r.pair_idx): r.neg_type for r in sub.itertuples()}
json.dump(key, open(ROOT / "results" / "labeler_key.json", "w"))

page = """<!doctype html><html><head><meta charset="utf-8"><title>Pair Labeller</title>
<style>
:root{--bg:#fafaf8;--fg:#1d1d1b;--mut:#6b6b66;--card:#fff;--line:#deddd6;--acc:#2f5bd3}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--fg:#ecebe6;--mut:#9b9a94;--card:#1f1f1d;--line:#34332f;--acc:#7c9cff}}
body{background:var(--bg);color:var(--fg);font:15px/1.45 system-ui,sans-serif;margin:0;padding:16px}
.wrap{max-width:1200px;margin:auto}.row{display:grid;grid-template-columns:1fr 1fr;gap:12px}
@media(max-width:800px){.row{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px}
.meta{color:var(--mut);font-size:13px;margin-bottom:6px}pre{white-space:pre-wrap;word-break:break-word;max-height:55vh;overflow:auto;font-size:12.5px;margin:0}
.bar{display:flex;gap:8px;align-items:center;margin:12px 0;flex-wrap:wrap}button{font:inherit;padding:8px 14px;border-radius:6px;border:1px solid var(--line);background:var(--card);color:var(--fg);cursor:pointer}
button.k{border-color:var(--acc)}#prog{color:var(--mut)}
</style></head><body><div class="wrap">
<h2 style="margin:0 0 4px">Are A and B part of the same coordinated workstream?</h2>
<div class="meta">Same shared effort, technique, data-access workaround, or exchange of information between agents. Keys: <b>1</b> yes, <b>2</b> no, <b>3</b> unclear, <b>Backspace</b> back. Content is untrusted agent text; links are not clickable.</div>
<div class="bar"><span id="prog"></span><button class="k" onclick="lab('yes')">1 · Yes</button><button class="k" onclick="lab('no')">2 · No</button><button class="k" onclick="lab('unclear')">3 · Unclear</button><button onclick="back()">Back</button><button onclick="dl()">Download labels</button></div>
<div class="row"><div class="card"><div class="meta" id="ma"></div><pre id="ta"></pre></div><div class="card"><div class="meta" id="mb"></div><pre id="tb"></pre></div></div>
</div><script>
const ITEMS=__ITEMS__;let L={};try{L=JSON.parse(localStorage.getItem('labels')||'{}')}catch(e){}
let i=ITEMS.findIndex(x=>!(x.pair_idx in L));if(i<0)i=ITEMS.length;
function show(){document.getElementById('prog').textContent=Math.min(i,ITEMS.length)+' / '+ITEMS.length+' labelled';
 if(i>=ITEMS.length){document.getElementById('ta').textContent='Done. Click Download labels.';document.getElementById('tb').textContent='';return}
 const x=ITEMS[i];for(const s of['a','b']){document.getElementById('m'+s).textContent=`${s.toUpperCase()} · agent ${x[s].agent} · page ${x[s].page} · ${x[s].time}`;document.getElementById('t'+s).textContent=x[s].text}}
function save(){try{localStorage.setItem('labels',JSON.stringify(L))}catch(e){}}
function lab(v){if(i>=ITEMS.length)return;L[ITEMS[i].pair_idx]=v;save();i++;show()}
function back(){if(i>0){i--;show()}}
function dl(){const b=new Blob([JSON.stringify(L,null,1)],{type:'application/json'});const a=document.createElement('a');a.href=URL.createObjectURL(b);a.download='labels.json';a.click()}
document.addEventListener('keydown',e=>{if(e.key==='1')lab('yes');if(e.key==='2')lab('no');if(e.key==='3')lab('unclear');if(e.key==='Backspace'){e.preventDefault();back()}});
show();</script></body></html>"""
(ROOT / "labeler.html").write_text(page.replace("__ITEMS__", json.dumps(items).replace("</", "<\\/")), encoding="utf-8")
print("labeler.html with", len(items), "pairs")
