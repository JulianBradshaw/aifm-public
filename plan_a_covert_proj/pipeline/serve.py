#!/usr/bin/env python3
"""Serve the covert run browser AND let it kick off runs from the page.

    <project-venv>/bin/python -m plan_a_covert_proj.pipeline.serve [--port 8765]

Serves plan_a_covert_proj/outputs/ (viewer.html + each run's figures) over http so the
page is same-origin with a tiny JSON API:

  GET  /api/configs -> {covert:[...], model:[...]}   available config/*.yaml files
  GET  /api/tree    -> `parent:` inheritance tree of every config (incl. external ancestors)
  GET  /api/config_diff?name=x.yaml -> leaf-level changes x makes vs its resolved parent
  GET  /api/runs    -> the run list (same shape make_viewer bakes into the page)
  GET  /api/jobs    -> launched runs with status + log tail
  POST /api/run     -> {num, covert_config, model_config, seed, local} -> {job_id}

Each launch spawns `run.py` exactly as you would from the shell (remote build on
aifm-runpod unless `local` is set), logging to outputs/_jobs/<id>.log. Run this with a
python that has the model deps (numpy/matplotlib) — i.e. the project venv — because the
spawned run.py needs them; serve.py itself only uses the stdlib.
"""
from __future__ import annotations
import argparse, json, subprocess, sys, threading, time
from datetime import datetime
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs

ROOT = Path(__file__).resolve().parents[2]            # repo root (cwd for run.py)
PKG = Path(__file__).resolve().parents[1]             # plan_a_covert_proj/
OUT = PKG / "outputs"
RUNS = OUT / "runs"
CONFIG = PKG / "config"
JOBS_DIR = OUT / "_jobs"
RUN_PY = PKG / "pipeline" / "run.py"

sys.path.insert(0, str(ROOT))
from plan_a_covert_proj.pipeline import make_viewer  # noqa: E402  (collect() for /api/runs)

_JOBS: dict = {}
_LOCK = threading.Lock()


def _list_configs() -> dict:
    """config/*.yaml split into model* vs covert* (by filename prefix)."""
    cov, mod = [], []
    for p in sorted(CONFIG.glob("*.yaml")):
        (mod if p.name.startswith("model") else cov).append(p.name)
    return {"covert": cov, "model": mod}


def _read_config(name: str, resolved: bool, compact: bool = False) -> dict:
    """Raw text of config/<name> (+ optional parent-merged resolution, + optional compact
    one-line-per-parameter view). Path-restricted to *.yaml directly inside the config dir."""
    fp = (CONFIG / name)
    if not name.endswith(".yaml") or fp.resolve().parent != CONFIG.resolve() or not fp.exists():
        return {"error": f"config not found: {name!r}"}
    out = {"name": name, "raw": fp.read_text()}
    if resolved:
        try:
            import yaml
            from scripts.pipelines.batch_rollout import _load_config
            out["resolved"] = yaml.safe_dump(_load_config(str(fp)) or {}, sort_keys=False)
        except Exception as e:  # noqa: BLE001
            out["resolved_error"] = str(e)
    if compact:
        try:
            from plan_a_covert_proj.pipeline import show_config
            out["compact"] = show_config.render(fp)
        except Exception as e:  # noqa: BLE001
            out["compact_error"] = str(e)
    return out


def _config_tree() -> dict:
    """`parent:` inheritance tree over config/*.yaml. External ancestors (parents outside
    the config dir, e.g. ../../config/sampling_config_daniel.yaml) are included as nodes
    keyed by repo-relative path, and their own parent chains are followed too."""
    import yaml
    nodes: dict = {}

    def _key(fp: Path) -> str:
        if fp.parent == CONFIG.resolve():
            return fp.name
        try:
            return str(fp.relative_to(ROOT))
        except ValueError:
            return str(fp)

    def _add(fp: Path) -> str:
        fp = fp.resolve()
        k = _key(fp)
        if k in nodes:
            return k
        local = fp.parent == CONFIG.resolve()
        kind = ("model" if fp.name.startswith("model") else "covert") if local else "external"
        node = nodes[k] = {"name": k, "kind": kind, "parent": None, "keys": []}
        if not fp.exists():
            node["kind"] = "missing"
            return k
        try:
            data = yaml.safe_load(fp.read_text()) or {}
        except Exception as e:  # noqa: BLE001
            node["kind"] = "error"
            node["error"] = str(e)
            return k
        node["keys"] = sorted(x for x in data if x != "parent")
        if data.get("parent"):
            node["parent"] = _add(fp.parent / str(data["parent"]))
        return k

    for p in sorted(CONFIG.glob("*.yaml")):
        _add(p)
    for n in nodes.values():
        n["children"] = []
    for k, n in nodes.items():
        if n["parent"]:
            nodes[n["parent"]]["children"].append(k)
    for n in nodes.values():
        n["children"].sort()
    roots = sorted(k for k, n in nodes.items() if not n["parent"])
    return {"nodes": nodes, "roots": roots}


_MISS = object()


def _short(v) -> str:
    s = json.dumps(v, default=str) if not isinstance(v, str) else v
    return s if len(s) <= 80 else s[:77] + "…"


def _leaf_diff(base, merged, path: list, out: list) -> None:
    if isinstance(base, dict) and isinstance(merged, dict):
        for k in sorted(set(base) | set(merged)):
            _leaf_diff(base.get(k, _MISS), merged.get(k, _MISS), path + [k], out)
        return
    key = ".".join(path)
    if base is _MISS:
        out.append({"key": key, "op": "+", "new": _short(merged)})
    elif merged is _MISS:
        out.append({"key": key, "op": "-", "old": _short(base)})
    elif base != merged:
        out.append({"key": key, "op": "~", "old": _short(base), "new": _short(merged)})


def _config_diff(name: str) -> dict:
    """Leaf-level changes config/<name> makes vs its RESOLVED parent (i.e. the effective
    overrides after the whole ancestor chain is merged). Same path restriction as
    _read_config; a config with no parent diffs as all-additions vs {}."""
    fp = CONFIG / name
    if not name.endswith(".yaml") or fp.resolve().parent != CONFIG.resolve() or not fp.exists():
        return {"error": f"config not found: {name!r}"}
    try:
        import yaml
        from scripts.pipelines.batch_rollout import _load_config
        raw = yaml.safe_load(fp.read_text()) or {}
        parent = raw.get("parent")
        base = _load_config(str((fp.parent / str(parent)).resolve())) if parent else {}
        merged = _load_config(str(fp)) or {}
        changes: list = []
        _leaf_diff(base or {}, merged, [], changes)
        return {"name": name, "parent": parent, "changes": changes}
    except Exception as e:  # noqa: BLE001
        return {"name": name, "parent": None, "error": str(e)}


def _spawn(params: dict) -> dict:
    """Launch run.py with the requested knobs; return the tracked job dict."""
    cov = params.get("covert_config") or "covert_scenario_branch.yaml"
    mod = params.get("model_config") or "model_uncertain_thomas.yaml"
    num = int(params.get("num") or 500)
    if cov not in _list_configs()["covert"] and cov not in _list_configs()["model"]:
        raise ValueError(f"unknown covert config {cov!r}")
    if mod not in _list_configs()["model"] and mod not in _list_configs()["covert"]:
        raise ValueError(f"unknown model config {mod!r}")
    name = str(params.get("name") or "").strip()
    cmd = [sys.executable, "-u", str(RUN_PY), "--num", str(num),
           "--covert-config", f"plan_a_covert_proj/config/{cov}",
           "--model-config", f"plan_a_covert_proj/config/{mod}"]
    if name:
        cmd += ["--name", name]
    if str(params.get("seed", "")).strip() != "":
        cmd += ["--seed", str(int(params["seed"]))]
    if params.get("local"):
        cmd += ["--local"]
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    jid = datetime.now().strftime("%H%M%S") + "-" + cov.replace(".yaml", "")[:24]
    logp = JOBS_DIR / f"{jid}.log"
    logf = open(logp, "w")
    proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=logf, stderr=subprocess.STDOUT)
    job = {"id": jid, "cmd": " ".join(cmd), "num": num, "covert": cov, "model": mod,
           "name": name or None,
           "status": "running", "started": time.time(), "ended": None,
           "log": str(logp), "_proc": proc, "_logf": logf}
    with _LOCK:
        _JOBS[jid] = job
    return job


def _spawn_reweight(params: dict) -> dict:
    """Launch ac_reweight.py --emit on an EXISTING run (post-hoc AC-time conditioning); return
    the tracked job dict. Mirrors _spawn so the viewer's /api/jobs shows it the same way. The
    source run must be a real base run (a dataset.json on disk) — not a reweighted sibling."""
    runid = str(params.get("runid") or "").strip()
    src = RUNS / runid
    if not runid or "/" in runid or ".." in runid or not (src / "dataset.json").exists():
        raise ValueError(f"no base run {runid!r} (needs an existing dataset.json)")
    target = float(params.get("target", 2030.0))
    sigma = float(params.get("sigma", 0.3))
    if not (sigma > 0):
        raise ValueError("sigma must be > 0")
    cmd = [sys.executable, "-u", "-m", "plan_a_covert_proj.pipeline.ac_reweight", runid,
           "--emit", "--target", repr(target), "--sigma", repr(sigma)]
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    jid = datetime.now().strftime("%H%M%S") + f"-rw-{runid[:12]}"
    logp = JOBS_DIR / f"{jid}.log"
    logf = open(logp, "w")
    proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=logf, stderr=subprocess.STDOUT)
    job = {"id": jid, "cmd": " ".join(cmd), "num": None,
           "covert": f"⚖ reweight {runid}", "model": f"AC≈{target:g} σ={sigma:g}",
           "status": "running", "started": time.time(), "ended": None,
           "log": str(logp), "_proc": proc, "_logf": logf}
    with _LOCK:
        _JOBS[jid] = job
    return job


def _job_view(j: dict) -> dict:
    rc = j["_proc"].poll()
    if rc is not None and j["status"] == "running":
        j["status"] = "done" if rc == 0 else "failed"
        j["ended"] = time.time()
    tail, runid = "", None
    try:
        lines = Path(j["log"]).read_text(errors="replace").splitlines()
        tail = "\n".join(lines[-30:])
        for ln in lines:
            if ln.startswith("runid "):
                runid = ln.split()[1]
                break
            if "reweighted run -> " in ln:        # ac_reweight --emit: the sibling run dir
                runid = ln.split("runs/")[-1].strip()
                break
    except Exception:  # noqa: BLE001
        pass
    return {"id": j["id"], "num": j["num"], "covert": j["covert"], "model": j["model"],
            "name": j.get("name"), "status": j["status"], "runid": runid,
            "elapsed": int((j["ended"] or time.time()) - j["started"]), "tail": tail}


LAUNCHER_HTML = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>covert · launch a run</title>
<style>
:root{--bg:#FBF8F0;--panel:#fffdf7;--ink:#23211c;--muted:#6f6a5f;--line:#e3ddcd;--navy:#1f4e9c;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 ui-sans-serif,system-ui,"Segoe UI",Roboto,sans-serif}
.mono,pre,select,input{font-family:"Courier New",ui-monospace,Menlo,monospace}
header{padding:14px 20px;border-bottom:1px solid var(--line);display:flex;align-items:baseline;gap:14px}
header h1{font:600 17px/1.2 "Courier New",monospace;margin:0;color:var(--navy)}
a.link{margin-left:auto;color:var(--navy);font:600 13px "Courier New",monospace;text-decoration:none}
.wrap{display:flex;gap:22px;padding:20px;align-items:flex-start;flex-wrap:wrap}
.col{flex:1;min-width:420px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin-bottom:16px}
.card h2{font:600 14px "Courier New",monospace;margin:0 0 10px;color:var(--navy)}
label{display:block;font-size:12px;color:var(--muted);margin:8px 0 3px}
select,input{font-size:13px;padding:5px 8px;border:1px solid var(--line);border-radius:6px;background:#fff;color:var(--ink);width:100%}
.row{display:flex;gap:12px;align-items:flex-end;flex-wrap:wrap}
.row>div{flex:1;min-width:110px}
.ck{display:flex;gap:6px;align-items:center;font-size:12.5px;color:var(--ink);margin-top:8px}
button{font:600 13px "Courier New",monospace;padding:7px 14px;border:1px solid var(--line);border-radius:7px;
background:var(--navy);color:#fff;cursor:pointer}
button:hover{filter:brightness(1.08)} button:disabled{opacity:.5;cursor:not-allowed}
button.sec{background:var(--panel);color:var(--ink)}
pre{white-space:pre-wrap;background:#fbf7ec;border:1px solid var(--line);border-radius:6px;padding:9px 11px;
font-size:11.5px;max-height:300px;overflow:auto;margin:4px 0 0}
.msg{font-size:12px;color:var(--muted);margin-left:8px}
.job{border:1px solid var(--line);border-radius:8px;padding:8px 11px;margin-bottom:9px;background:var(--panel)}
.jh{display:flex;gap:9px;align-items:center;flex-wrap:wrap;font:12px "Courier New",monospace}
.st{font-weight:600;padding:1px 8px;border-radius:9px}
.st.running{background:#fff3d6;color:#7a5a00}.st.done{background:#eef6e6;color:#3f5e10}.st.failed{background:#f7ecea;color:#7e2f2f}
.job pre{max-height:150px;font-size:11px}
.empty{color:var(--muted);padding:14px;text-align:center}
.treewrap{display:flex;gap:16px;flex-wrap:wrap;align-items:flex-start}
#tree{flex:1.3;min-width:380px;font:12px "Courier New",monospace;line-height:1.6;max-height:380px;overflow:auto;margin:0}
.tn{white-space:pre;cursor:pointer;border-radius:4px;padding:0 4px}
.tn:hover{background:#f3eedd}.tn.sel{background:#e8effc}
.tn a{color:var(--navy);font-weight:600}
.tn[data-kind=model] a{color:#7a2e8f}
.tn[data-kind=external] a{color:var(--muted);font-weight:400}
.tn[data-kind=missing] a,.tn[data-kind=error] a{color:#a33}
.tk{color:var(--muted);font-size:10.5px}
#treediff{flex:1;min-width:340px;max-height:380px}
.d-add{color:#3f5e10}.d-chg{color:#7a5a00}.d-del{color:#7e2f2f}
.legend{font-size:11px;color:var(--muted);margin:2px 0 8px}
.legend b{font-weight:600}.legend .lm{color:#7a2e8f}.legend .lc{color:var(--navy)}
</style></head><body>
<header><h1>covert · launch a run</h1><a class="link" href="/">↗ run browser</a></header>
<div class="wrap">
  <div class="card" style="flex-basis:100%">
    <h2>config inheritance tree</h2>
    <div class="legend"><b class="lc">covert</b> · <b class="lm">model</b> · <span>external ancestor</span> — grey words after a name are the keys that file sets locally; click a config to select it above and see its effective changes vs its resolved parent</div>
    <div class="treewrap">
      <pre id="tree">…</pre>
      <pre id="treediff">click a config on the left to diff it against its parent</pre>
    </div>
  </div>
  <div class="col">
    <div class="card">
      <h2>new run</h2>
      <label>run name <span style="color:var(--muted)">(optional — label shown in the browser; not hashed)</span></label>
      <input id="name" type="text" placeholder="e.g. thomas + 0.3 min-SW sweep" maxlength="120">
      <div class="row" style="margin-top:8px">
        <div><label>num rollouts</label><input id="num" type="number" value="500" min="1" max="5000"></div>
        <div><label>seed</label><input id="seed" type="number" placeholder="(use yaml)"></div>
      </div>
      <label class="ck"><input id="local" type="checkbox"> build locally (unchecked = aifm-runpod)</label>
      <label>config preview</label>
      <select id="view" style="max-width:260px">
        <option value="compact" selected>compact params (parent-merged)</option>
        <option value="raw">raw yaml</option>
        <option value="resolved">resolved yaml</option>
      </select>
      <div style="margin-top:12px"><button id="go">launch</button><span id="msg" class="msg"></span></div>
    </div>
    <div class="card">
      <h2>covert config <button id="reload" class="sec" style="float:right;padding:2px 9px;font-size:11px">↻ rescan configs</button></h2>
      <select id="cov"></select>
      <pre id="cov_prev">…</pre>
    </div>
    <div class="card">
      <h2>model config</h2>
      <select id="mod"></select>
      <pre id="mod_prev">…</pre>
    </div>
  </div>
  <div class="col">
    <div class="card">
      <h2>in flight / recent</h2>
      <div id="jobs"><div class="empty">no runs launched yet this session</div></div>
    </div>
  </div>
</div>
<script>
const $=s=>document.querySelector(s);
const esc=s=>(s==null?"":String(s)).replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));
async function jget(u){ const r=await fetch(u); return r.json(); }
async function preview(sel, pre){
  const name=$(sel).value; if(!name) return;
  const mode=$("#view").value;
  try{ const d=await jget("/api/config?name="+encodeURIComponent(name)+"&mode="+mode);
    if(d.error){ $(pre).textContent="error: "+d.error; return; }
    let text;
    if(mode==="compact") text = d.compact || ("(compact failed: "+(d.compact_error||"?")+")\\n\\n"+(d.raw||""));
    else if(mode==="resolved") text = d.resolved || ("(resolve failed: "+(d.resolved_error||"?")+")\\n\\n"+(d.raw||""));
    else text = d.raw;
    $(pre).textContent = text;
  }catch(e){ $(pre).textContent="error: "+e; }
}
async function loadConfigs(){
  const c=await jget("/api/configs");
  const cur=sel=>($(sel)&&$(sel).value)||null;   // keep current pick across a rescan
  const opt=(arr,d)=>arr.map(n=>`<option ${n===d?"selected":""}>${esc(n)}</option>`).join("");
  $("#cov").innerHTML=opt(c.covert, cur("#cov")||"covert_scenario_branch.yaml");
  $("#mod").innerHTML=opt(c.model, cur("#mod")||"model_uncertain_thomas.yaml");
  preview("#cov","#cov_prev"); preview("#mod","#mod_prev");
}
function renderTree(t){
  const lines=[];
  function rec(k, prefix, isLast, isRoot){
    const n=t.nodes[k];
    const branch=isRoot?"":prefix+(isLast?"└─ ":"├─ ");
    const cp=isRoot?"":prefix+(isLast?"   ":"│  ");
    const keys=(n.keys||[]).join(" ");
    lines.push(`<div class="tn" data-k="${esc(k)}" data-kind="${n.kind}">${esc(branch)}<a>${esc(n.name)}</a>${keys?`  <span class="tk">${esc(keys)}</span>`:""}</div>`);
    (n.children||[]).forEach((c,i)=>rec(c,cp,i===n.children.length-1,false));
  }
  t.roots.forEach(r=>rec(r,"",true,true));
  $("#tree").innerHTML=lines.join("");
  document.querySelectorAll("#tree .tn").forEach(el=>el.onclick=()=>pickNode(el));
}
async function pickNode(el){
  document.querySelectorAll("#tree .tn.sel").forEach(x=>x.classList.remove("sel"));
  el.classList.add("sel");
  const k=el.dataset.k, kind=el.dataset.kind;
  if(kind==="covert"&&[...$("#cov").options].some(o=>o.value===k)){ $("#cov").value=k; preview("#cov","#cov_prev"); }
  if(kind==="model"&&[...$("#mod").options].some(o=>o.value===k)){ $("#mod").value=k; preview("#mod","#mod_prev"); }
  if(kind!=="covert"&&kind!=="model"){
    $("#treediff").textContent="("+kind+" config — diff shown only for configs in plan_a_covert_proj/config/)"; return;
  }
  $("#treediff").textContent="diffing vs parent…";
  try{
    const d=await jget("/api/config_diff?name="+encodeURIComponent(k));
    if(d.error){ $("#treediff").textContent="error: "+d.error; return; }
    const head=d.parent?`${esc(k)}  vs resolved parent ${esc(d.parent)}\\n\\n`
                       :`${esc(k)} has no parent — showing all settings\\n\\n`;
    const body=(d.changes||[]).map(c=>{
      if(c.op==="+") return `<span class="d-add">+ ${esc(c.key)}: ${esc(c.new)}</span>`;
      if(c.op==="-") return `<span class="d-del">- ${esc(c.key)}  (was ${esc(c.old)})</span>`;
      return `<span class="d-chg">~ ${esc(c.key)}: ${esc(c.old)} → ${esc(c.new)}</span>`;
    }).join("\\n")||"(no effective changes vs parent)";
    $("#treediff").innerHTML=head+body;
  }catch(e){ $("#treediff").textContent="error: "+e; }
}
async function loadTree(){ try{ renderTree(await jget("/api/tree")); }catch(e){ $("#tree").textContent="error: "+e; } }
$("#reload").onclick=()=>{ loadConfigs(); loadTree(); };
$("#cov")&&($("#cov").onchange=()=>preview("#cov","#cov_prev"));
$("#mod")&&($("#mod").onchange=()=>preview("#mod","#mod_prev"));
$("#view").onchange=()=>{ preview("#cov","#cov_prev"); preview("#mod","#mod_prev"); };
function jobCard(j){
  return `<div class="job"><div class="jh"><span class="st ${j.status}">${j.status}</span>
    ${j.name?`<b>${esc(j.name)}</b> · `:""}<b>${esc(j.covert)}</b> · ${esc(j.model)} · num ${j.num} · ${j.elapsed}s${j.runid?` · runid <a href="/#${esc(j.runid)}">${esc(j.runid)}</a>`:""}</div>
    <pre>${esc(j.tail||"")}</pre></div>`;
}
async function pollJobs(){
  try{ const js=await jget("/api/jobs");
    $("#jobs").innerHTML = js.length ? js.map(jobCard).join("") : `<div class="empty">no runs launched yet this session</div>`;
  }catch(e){}
}
$("#go").onclick=async()=>{
  const body={num:+$("#num").value, covert_config:$("#cov").value, model_config:$("#mod").value,
    name:$("#name").value, seed:$("#seed").value, local:$("#local").checked};
  $("#go").disabled=true; $("#msg").textContent="launching…";
  try{ const j=await (await fetch("/api/run",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)})).json();
    $("#msg").textContent=j.job_id?("started "+j.job_id):("error: "+(j.error||"?")); pollJobs();
  }catch(e){ $("#msg").textContent="error: "+e; }
  finally{ $("#go").disabled=false; }
};
loadConfigs(); loadTree(); pollJobs(); setInterval(pollJobs,2000);
</script></body></html>"""


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=str(OUT), **k)

    def _json(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def _html(self, s: str):
        b = s.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):  # noqa: N802
        p = self.path.split("?")[0]
        if p == "/api/configs":
            return self._json(_list_configs())
        if p == "/api/tree":
            return self._json(_config_tree())
        if p == "/api/config_diff":
            qs = parse_qs(urlparse(self.path).query)
            out = _config_diff((qs.get("name") or [""])[0])
            return self._json(out, 404 if out.get("error") and "changes" not in out else 200)
        if p == "/api/config":
            qs = parse_qs(urlparse(self.path).query)
            name = (qs.get("name") or [""])[0]
            mode = (qs.get("mode") or [""])[0]     # "compact" | "raw" | "resolved" (falls back to `resolved` flag)
            resolved = mode == "resolved" or (qs.get("resolved") or ["0"])[0] == "1"
            compact = mode == "compact"
            out = _read_config(name, resolved, compact)
            return self._json(out, 404 if out.get("error") else 200)
        if p == "/api/runs":
            return self._json(make_viewer.collect())
        if p == "/api/jobs":
            with _LOCK:
                js = [_job_view(j) for j in _JOBS.values()]
            js.sort(key=lambda x: x["id"], reverse=True)
            return self._json(js)
        if p.startswith("/api/"):
            return self._json({"error": "unknown endpoint"}, 404)
        if p == "/launch":
            return self._html(LAUNCHER_HTML)
        if p in ("/", "/index.html"):
            self.path = "/viewer.html"
        return super().do_GET()

    def do_POST(self):  # noqa: N802
        p = self.path.split("?")[0]
        if p in ("/api/run", "/api/reweight"):
            n = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
                job = _spawn(body) if p == "/api/run" else _spawn_reweight(body)
            except Exception as e:  # noqa: BLE001
                return self._json({"error": str(e)}, 400)
            return self._json({"job_id": job["id"]})
        return self._json({"error": "unknown endpoint"}, 404)

    def log_message(self, *a):  # quiet
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    make_viewer.regenerate()
    httpd = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    print(f"covert viewer + launcher -> http://127.0.0.1:{a.port}/   (Ctrl-C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
