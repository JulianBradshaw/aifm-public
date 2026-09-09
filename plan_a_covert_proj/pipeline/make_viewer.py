#!/usr/bin/env python3
"""Generate a self-contained HTML browser for the covert pipeline's runs.

Scans plan_a_covert_proj/outputs/runs/*/manifest.json and writes
plan_a_covert_proj/outputs/viewer.html — a two-pane viewer (run list + per-run config,
stats, and figure gallery). Figures are referenced by RELATIVE path (runs/<id>/figures/*),
so open the file from within outputs/ (file:// works, or `python -m http.server` there).

The pipeline (plan_a_covert_proj/pipeline/run.py) calls regenerate() at the end of every
run, so the viewer auto-refreshes. Run this module directly to rebuild it on demand:

    python -m plan_a_covert_proj.pipeline.make_viewer
"""
from __future__ import annotations
import json, sys
from datetime import datetime
from pathlib import Path

# This file lives in plan_a_covert_proj/pipeline/, so parents[1] is plan_a_covert_proj/.
OUT = Path(__file__).resolve().parents[1] / "outputs"
RUNS = OUT / "runs"


def _base(p):
    return Path(str(p)).name.replace(".yaml", "") if p else "?"


def collect():
    runs = []
    for mf in sorted(RUNS.glob("*/manifest.json")):
        try:
            m = json.loads(mf.read_text())
        except Exception:
            continue
        cfg = m.get("config", {}) or {}
        s = m.get("summary", {}) or {}
        k = cfg.get("covert_knobs_fixed", {}) or {}
        figs = [f for f in (m.get("figures") or []) if f.endswith(".png")]
        runid = m.get("runid", mf.parent.name)
        v = int(mf.stat().st_mtime)   # cache-buster: changes whenever the run is rebuilt
        # Compute-acquisition tables (pipeline/tables.py) — embed the HTML fragment inline so it
        # renders in the detail pane. Detected on disk (not via the manifest) so runs whose tables
        # were generated after the run still show them.
        tbl_fp = mf.parent / "tables_compute_acquisition.html"
        tbl_html = ""
        if tbl_fp.exists():
            try:
                tbl_html = tbl_fp.read_text()
            except Exception:
                tbl_html = ""
        runs.append({
            "runid": runid,
            "name": m.get("name"),
            "mtime": mf.stat().st_mtime,
            "date": datetime.fromtimestamp(mf.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
            "model": _base(cfg.get("model_config")),
            "covert": _base(cfg.get("covert_config")),
            "input_data": _base(cfg.get("input_data")),
            "num": cfg.get("num"), "n_ok": s.get("n_ok"), "n_req": s.get("n_requested"),
            "seed": cfg.get("seed"), "legal_mode": cfg.get("legal_mode"),
            "years_before_ac": cfg.get("years_before_ac"),
            "win": (round(100 * s["covert_win_frac"], 1) if s.get("covert_win_frac") is not None else None),
            "git_sha": m.get("git_sha"),
            "legal_strategy": k.get("legal_strategy"),
            "compute_mode": k.get("compute_mode"),
            "knobs": {kk: vv for kk, vv in k.items()},
            "figures": [f"runs/{runid}/figures/{f}?v={v}" for f in figs],
            "fignames": figs,
            "has_explorer": "explorer.html" in (m.get("figures") or []),
            "explorer": f"runs/{runid}/figures/explorer.html?v={v}",
            "has_tables": bool(tbl_html),
            "tables_html": tbl_html,
            "tables_md": f"runs/{runid}/tables_compute_acquisition.md?v={v}",
            # post-hoc AC-time reweighting (ac_reweight.py --emit); None for ordinary runs
            "reweight": m.get("ac_reweight"),
        })
    runs.sort(key=lambda r: r["mtime"], reverse=True)
    return runs


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Covert pipeline — run browser</title>
<style>
:root{--bg:#FBF8F0;--panel:#fffdf7;--ink:#23211c;--muted:#6f6a5f;--line:#e3ddcd;
--navy:#1f4e9c;--sienna:#b5651d;--olive:#6b8e23;--red:#a23b3b;--gold:#e3b23c;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
font:14px/1.5 ui-sans-serif,system-ui,"Segoe UI",Roboto,sans-serif;}
.mono{font-family:"Courier New",ui-monospace,Menlo,monospace}
header{padding:14px 20px;border-bottom:1px solid var(--line);display:flex;align-items:baseline;gap:14px;flex-wrap:wrap}
header h1{font:600 17px/1.2 "Courier New",monospace;margin:0;color:var(--navy)}
header .sub{color:var(--muted);font-size:12px}
header .controls{margin-left:auto;display:flex;gap:8px;align-items:center}
input,select{font:13px "Courier New",monospace;padding:5px 8px;border:1px solid var(--line);
border-radius:6px;background:var(--panel);color:var(--ink)}
.layout{display:flex;height:calc(100vh - 53px)}
aside{width:340px;flex:0 0 340px;border-right:1px solid var(--line);overflow-y:auto;background:var(--panel)}
main{flex:1;overflow-y:auto;padding:20px 26px}
.run{padding:11px 16px;border-bottom:1px solid var(--line);cursor:pointer}
.run:hover{background:#f3eedd}
.run.sel{background:#eef3fb;box-shadow:inset 3px 0 0 var(--navy)}
.run .top{display:flex;justify-content:space-between;align-items:baseline;gap:8px}
.run .cov{font:600 13px "Courier New",monospace;color:var(--navy);word-break:break-all}
.run .meta{color:var(--muted);font-size:11.5px;margin-top:3px}
.badge{font:600 12px "Courier New",monospace;padding:2px 7px;border-radius:10px;white-space:nowrap}
.win{background:#eef6e6;color:#3f5e10;border:1px solid #cfe3b0}
.win.lo{background:#f7ecea;color:#7e2f2f;border:1px solid #e6c7c2}
.win.na{background:#efeadf;color:var(--muted);border:1px solid var(--line)}
.rwline{color:var(--sienna);font-weight:600;margin-top:3px}
.run.rw{box-shadow:inset 3px 0 0 var(--gold)}
.run.rw.sel{box-shadow:inset 3px 0 0 var(--navy)}
.rwbanner{background:#fbf1d8;border:1px solid var(--gold);color:#6b531a;border-radius:8px;
padding:9px 12px;margin:0 0 14px;font:600 13px "Courier New",monospace}
.rwform{background:#fbf1d8;border:1px solid var(--gold);border-radius:8px;padding:10px 13px;margin:0 0 16px;font-size:13px;color:#6b531a}
.rwform input{width:62px;font:13px "Courier New",monospace;padding:3px 6px;border:1px solid var(--line);border-radius:5px;margin:0 4px}
.rwform .msg{margin-left:8px;color:var(--muted);font-size:12px}
.rwin{margin:0 10px;white-space:nowrap;font-weight:600}
h2{font:600 16px/1.3 "Courier New",monospace;margin:0 0 2px;color:var(--navy);word-break:break-all}
.runsub{color:var(--muted);font-size:12.5px;margin-bottom:14px}
.stats{display:flex;flex-wrap:wrap;gap:10px;margin:0 0 16px}
.stat{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:8px 12px;min-width:96px}
.stat .l{font-size:10.5px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted)}
.stat .v{font:600 16px "Courier New",monospace;margin-top:2px}
.knobs{font-family:"Courier New",monospace;font-size:12px;color:var(--ink);background:var(--panel);
border:1px solid var(--line);border-radius:8px;padding:10px 12px;margin-bottom:18px;
column-width:230px;column-gap:22px}
.knobs b{color:var(--navy);font-weight:600}
.gallery{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:14px}
.fig{border:1px solid var(--line);border-radius:8px;overflow:hidden;background:var(--panel);cursor:zoom-in}
.fig img{width:100%;display:block;background:#fff}
.fig .cap{font:12px "Courier New",monospace;padding:6px 9px;color:var(--muted);border-top:1px solid var(--line);
white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.pill{display:inline-block;font:600 11px "Courier New",monospace;padding:2px 8px;border-radius:10px;
border:1px solid var(--line);margin-right:6px}
.explorer-btn{display:inline-block;margin:0 0 16px;padding:7px 13px;background:var(--navy);color:#fff;
border-radius:7px;text-decoration:none;font:600 13px "Courier New",monospace}
.tables{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:4px 16px 14px;margin:0 0 18px}
.tables h1{font:600 15px "Courier New",monospace;color:var(--navy);margin:12px 0 4px;border:0}
.tables h2{font:600 13px "Courier New",monospace;color:var(--sienna);margin:14px 0 4px}
.tables p em{color:var(--muted);font-size:11.5px}
.tables table{border-collapse:collapse;width:100%;margin:6px 0;font-size:12.5px}
.tables td,.tables th{border:1px solid var(--line);padding:5px 9px;text-align:left;vertical-align:top}
.tables tr:first-child td{background:#f3eedd}
.tables td p{margin:0}
#lb{position:fixed;inset:0;background:rgba(20,18,14,.86);display:none;align-items:center;justify-content:center;
z-index:50;cursor:zoom-out;padding:24px}
#lb img{max-width:97%;max-height:97%;box-shadow:0 8px 40px rgba(0,0,0,.5);background:#fff}
.empty{color:var(--muted);padding:40px;text-align:center}
.btn{font:600 13px "Courier New",monospace;padding:5px 11px;border:1px solid var(--line);border-radius:6px;
background:var(--navy);color:#fff;cursor:pointer;text-decoration:none}
.btn:hover{filter:brightness(1.08)}
</style></head><body>
<header>
  <h1>covert pipeline · run browser</h1>
  <span class="sub" id="count"></span>
  <div class="controls">
    <button id="newrun" class="btn">＋ new run</button>
    <input id="q" placeholder="filter: config / runid / strategy…" size="26">
    <select id="sort">
      <option value="mtime">newest first</option>
      <option value="win">covert win ↓</option>
      <option value="num">num sims ↓</option>
      <option value="covert">covert config</option>
    </select>
  </div>
</header>
<div class="layout">
  <aside id="list"></aside>
  <main id="detail"><div class="empty">Select a run on the left.</div></main>
</div>
<div id="lb"><img></div>
<script>
let RUNS = /*__DATA__*/[];
const $=s=>document.querySelector(s);
const winClass=w=> w==null?"na":(w>=25?"win":(w>=15?"win":"win lo"));
let sel=null;

function fmt(v){ if(v==null) return "—";
  if(typeof v==="number"){ if(Math.abs(v)>=1e5) return v.toExponential(2);
    return Number.isInteger(v)?v:(+v.toFixed(4)); } return String(v); }

function list(){
  const q=$("#q").value.toLowerCase().trim(), by=$("#sort").value;
  let rs=RUNS.filter(r=>!q || ((r.name||"")+" "+r.covert+" "+r.model+" "+r.runid+" "+(r.legal_strategy||"")+(r.reweight?" reweight reweighted acrw ac-time":"")).toLowerCase().includes(q));
  rs.sort((a,b)=> by==="mtime"? b.mtime-a.mtime : by==="win"? (b.win??-1)-(a.win??-1)
    : by==="num"? (b.num??0)-(a.num??0) : a.covert.localeCompare(b.covert));
  $("#count").textContent = rs.length+" / "+RUNS.length+" runs";
  $("#list").innerHTML = rs.map(r=>`<div class="run ${r.reweight?'rw':''} ${sel===r.runid?'sel':''}" data-id="${r.runid}">
    <div class="top"><span class="cov">${r.reweight?'⚖ ':''}${esc(r.name||r.covert)}</span>
      <span class="badge ${winClass(r.win)}">${r.win==null?'—':r.win+'%'}</span></div>
    <div class="meta">${r.name?esc(r.covert)+' · ':''}${esc(r.model)} · ${r.legal_strategy?esc(r.legal_strategy)+' · ':''}n=${r.n_ok??'?'}/${r.num??'?'} · ${r.date}</div>
    ${r.reweight?`<div class="meta rwline">⚖ AC≈${r.reweight.target} reweighted · ESS ${Math.round(100*r.reweight.ess_frac)}% · src ${esc(r.reweight.source_run)}</div>`:''}
  </div>`).join("") || `<div class="empty">no matches</div>`;
  document.querySelectorAll(".run").forEach(el=>el.onclick=()=>show(el.dataset.id));
  if(sel) document.querySelector(".run.sel")?.scrollIntoView({block:"nearest"});  // keep selection visible after sort/filter
}
function esc(s){return (s==null?"":String(s)).replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));}

function show(id){
  sel=id; const r=RUNS.find(x=>x.runid===id); if(!r) return;
  list();
  const knobs=Object.entries(r.knobs).map(([k,v])=>`<div><b>${esc(k)}</b> = ${esc(fmt(v))}</div>`).join("");
  const stat=(l,v)=>`<div class="stat"><div class="l">${l}</div><div class="v">${v}</div></div>`;
  // Reweighting is post-hoc on a base run; offer it only when served over http (needs the
  // POST endpoint) and not on a run that is itself a reweighted sibling (no dataset.json).
  const canRW = SERVED && !r.reweight;
  const rwcard = canRW ? `<div class="rwform">
      <b>⚖ reweight this run</b> — condition on AC near a target year (post-hoc; builds a sibling run, no re-sim).
      <span class="rwin">AC=<input id="rwT" value="2030"></span>
      <span class="rwin">σ=<input id="rwS" value="0.3"></span>
      <button id="rwGo" class="btn">reweight</button>
      <span id="rwMsg" class="msg"></span></div>` : "";
  $("#detail").innerHTML=`
    <h2>${esc(r.name||r.covert)}</h2>
    <div class="runsub">covert <b>${esc(r.covert)}</b> · model <b>${esc(r.model)}</b> · input <span class="mono">${esc(r.input_data)}</span>
      · runid <span class="mono">${esc(r.runid)}</span>${r.git_sha?' · git '+esc(r.git_sha):''} · ${r.date}</div>
    ${r.reweight?`<div class="rwbanner">⚖ AC-time reweighted toward <b>${r.reweight.target}</b> · σ=${r.reweight.sigma} ·
      ESS <b>${Math.round(r.reweight.ess)}/${r.reweight.n}</b> (${Math.round(100*r.reweight.ess_frac)}% kept) ·
      source run <span class="mono">${esc(r.reweight.source_run)}</span> ·
      median-scenario &amp; scatter figures are NOT reweighted (shown as-is)</div>`:''}
    <div class="stats">
      ${stat("covert win", r.win==null?'—':r.win+'%')}
      ${stat("n ok", (r.n_ok??'?')+' / '+(r.num??'?'))}
      ${stat("legal strategy", esc(r.legal_strategy||'—'))}
      ${stat("legal mode", esc(r.legal_mode||'—'))}
      ${stat("compute mode", esc(r.compute_mode||'—'))}
      ${stat("yrs pre-AC", fmt(r.years_before_ac))}
    </div>
    ${rwcard}
    ${r.has_explorer?`<a class="explorer-btn" href="${r.explorer}" target="_blank">↗ open interactive explorer</a><br>`:""}
    ${r.has_tables?`<div class="tables">${r.tables_html}<a class="btn" href="${r.tables_md}" target="_blank" style="display:inline-block;margin-top:10px">↓ download tables (.md)</a></div>`:""}
    <div class="knobs">${knobs}</div>
    <div class="gallery">${r.figures.map((f,i)=>`<div class="fig" data-src="${f}">
       <img loading="lazy" src="${f}" alt="${esc(r.fignames[i])}">
       <div class="cap">${esc(r.fignames[i])}</div></div>`).join("")}</div>`;
  document.querySelectorAll(".fig").forEach(el=>el.onclick=()=>{ $("#lb img").src=el.dataset.src; $("#lb").style.display="flex"; });
  if(canRW){ const g=$("#rwGo"); if(g) g.onclick=()=>reweight(r.runid); }
}

// Kick off ac_reweight --emit on a base run, poll the job, then refresh the list with the
// new sibling selected. Server-only (needs serve.py); no-op when opened as a static file.
async function reweight(srcid){
  const t=$("#rwT").value.trim(), s=$("#rwS").value.trim();
  const g=$("#rwGo"), msg=$("#rwMsg");
  g.disabled=true; msg.textContent="launching…";
  let job;
  try{ job=await (await fetch("/api/reweight",{method:"POST",headers:{"Content-Type":"application/json"},
        body:JSON.stringify({runid:srcid,target:t,sigma:s})})).json(); }
  catch(e){ msg.textContent="error: "+e; g.disabled=false; return; }
  if(job.error){ msg.textContent="error: "+job.error; g.disabled=false; return; }
  const want=(srcid+"__acrw"+(+t)+"_s"+(+s)).replace(/\\./g,"p");
  msg.textContent="reweighting… (job "+job.job_id+")";
  const tick=async()=>{
    let js=[]; try{ js=await (await fetch("/api/jobs")).json(); }catch(e){}
    const j=js.find(x=>x.id===job.job_id);
    if(j && j.status!=="running"){
      if(j.status==="failed"){ msg.textContent="failed — check job "+job.job_id+" log"; g.disabled=false; return; }
      RUNS=await (await fetch("/api/runs")).json();   // pull in the new sibling
      sel=(j.runid||want); list(); show(sel); return;
    }
    setTimeout(tick,1500);
  };
  setTimeout(tick,1500);
}
$("#lb").onclick=()=>$("#lb").style.display="none";
$("#q").oninput=list; $("#sort").onchange=list;

// "+ new run" opens the launcher page (needs the serve.py server, i.e. http).
const SERVED = location.protocol.startsWith("http");
$("#newrun").onclick=()=> SERVED ? (location.href="/launch")
  : alert("Launching runs needs the server. Start it with:\\n  python -m plan_a_covert_proj.pipeline.serve\\nthen open http://127.0.0.1:8765/");

list(); if(RUNS.length) show(RUNS[0].runid);
</script></body></html>"""


def regenerate():
    """Rebuild outputs/viewer.html from every run manifest. Returns (path, n_runs) or
    (None, 0) if there are no runs yet. Safe to call from the pipeline (never raises on
    an empty outputs dir)."""
    runs = collect()
    if not runs:
        return None, 0
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "viewer.html"
    out.write_text(PAGE.replace("/*__DATA__*/[]", json.dumps(runs, separators=(",", ":"))))
    return out, len(runs)


def main():
    out, n = regenerate()
    if out is None:
        sys.exit(f"no runs found under {RUNS}")
    print(f"wrote {out}  ({n} runs)")


if __name__ == "__main__":
    main()
