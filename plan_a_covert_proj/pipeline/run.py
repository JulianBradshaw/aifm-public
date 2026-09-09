#!/usr/bin/env python3
"""Covert experiment orchestrator — one command rebuilds everything, robustly.

  python -m plan_a_covert_proj.pipeline.run [--num 500] [--legal-mode capped] [--local] [--force]
        [--model-config plan_a_covert_proj/config/model_uncertain_thomas.yaml] [--covert-config ...]

Inputs are content-hashed into a `runid`: the chosen model + covert yamls AND their full
`parent:` chains, ALL model/pipeline code (progress_model/config.py, progress_model/, sim.py,
detection.py, pipeline/mc.py, batch_rollout.py — see _code_inputs), the input CSV the
model config points at, and the run params. Outputs live in
plan_a_covert_proj/outputs/runs/<runid>/ : dataset.json, figures/*.png, manifest.json. If the
dataset for the current inputs already exists it is reused (skip --force). So: change a
config OR any model code -> new hash -> the dataset + every figure rebuild automatically;
nothing goes stale silently. `plan_a_covert_proj/outputs/latest` points at the newest run.

By default the (expensive) dataset build runs on aifm-runpod; figures always build
locally from the pulled dataset. Use --local to build the dataset locally too. Before a
remote build, run.py pushes exactly that hashed input set to the box and verifies it is
byte-identical (_assert_remote_in_sync) — the box can never run stale code behind a
valid-looking runid.
"""
from __future__ import annotations
import argparse, hashlib, json, subprocess, sys
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
COVERT = ROOT / "plan_a_covert_proj"
from plan_a_covert_proj.pipeline import figures as figmod  # noqa: E402
from plan_a_covert_proj.pipeline import tables as tabmod  # noqa: E402
from plan_a_covert_proj.pipeline import remote as rmt  # noqa: E402
from plan_a_covert_proj.pipeline import make_viewer  # noqa: E402
from scripts.pipelines.batch_rollout import _load_config  # noqa: E402

def _resolve_input_csv(model_cfg: Path) -> Path:
    """The time-series CSV the pipeline actually loads — the model config's `input_data`
    (resolved relative to the repo root, matching mc.build_dataset)."""
    rel = (_load_config(str(model_cfg)) or {}).get("input_data", "inputs/input_data.csv")
    p = Path(rel)
    return p if p.is_absolute() else (ROOT / rel)


def _resolve_plan_a_csv(covert_cfg: Path) -> Path | None:
    """The covert config's optional `plan_a_input_data` series (the Plan-A divergence the
    legal/covert follow post-pause). None when not configured."""
    rel = (_load_config(str(covert_cfg)) or {}).get("plan_a_input_data")
    if not rel:
        return None
    p = Path(rel)
    return p if p.is_absolute() else (ROOT / rel)


def _resolve_bottom_up_cfg(covert_cfg: Path) -> Path | None:
    """The covert config's optional `bottom_up_compute_config` (diversion-parameter
    distributions for compute_mode=bottom_up_estimate). None when not configured. Hashed so
    editing the bottom-up parameters invalidates dependent runs."""
    rel = (_load_config(str(covert_cfg)) or {}).get("bottom_up_compute_config")
    if not rel:
        return None
    p = Path(rel)
    return p if p.is_absolute() else (ROOT / rel)


def _config_chain(cfg: Path) -> list[Path]:
    """The config file plus every ancestor it inherits via `parent:`, leaf-first.
    Walks the RAW `parent:` field (not the merged config) so editing any base in the
    chain bumps the runid — otherwise inheritance silently introduces a stale-dataset
    trap where a base edit doesn't invalidate dependent runs."""
    chain: list[Path] = []
    seen: set[Path] = set()
    cur: Path | None = cfg.resolve()
    while cur is not None and cur not in seen:
        seen.add(cur)
        chain.append(cur)
        try:
            raw = yaml.safe_load(cur.read_text()) or {}
        except Exception:
            break
        parent = raw.get("parent")
        if not parent:
            break
        pp = Path(parent)
        cur = (pp if pp.is_absolute() else (cur.parent / pp)).resolve()
    return chain


# Code whose CONTENT changes the dataset but isn't a per-run config. Entries may be
# files or directories (dirs are walked for *.py). This is the SINGLE SOURCE OF TRUTH:
# the same set is (a) hashed into the runid and (b) pushed to the remote box before every
# build — so the box can never run different code than the runid claims. Previously only
# the plan_a_covert_proj tree was pushed/hashed, so a stale progress_model / model_config
# on the box (or a daniel-parent edit) silently produced wrong results under a valid runid.
def _code_inputs() -> list[Path]:
    return [
        ROOT / "progress_model" / "config.py",         # Parameters + milestone definitions
        ROOT / "progress_model",                        # the actual model (dir, walked for *.py)
        ROOT / "scripts" / "pipelines" / "batch_rollout.py",  # sampler / _load_config / dist fitting
        COVERT / "sim.py",
        COVERT / "detection.py",         # detection draws are baked into the dataset
        COVERT / "deal_hazard.py",       # deal-breakdown sampling is baked into the dataset
        COVERT / "pipeline" / "mc.py",   # the dataset's contents depend on its builder
    ]


def _expand(paths) -> list[Path]:
    """Expand directories to their *.py files (sorted); pass files through unchanged."""
    out: list[Path] = []
    for p in paths:
        if p.is_dir():
            out.extend(sorted(p.rglob("*.py")))
        else:
            out.append(p)
    return out


def _input_files(model_cfg: Path, covert_cfg: Path) -> list[Path]:
    """Every file whose CONTENT defines a run (hashed into the runid), deduped & ordered.
    = each config's full `parent:` inheritance chain (so editing a shared base like
    covert_scenario_branch.yaml or sampling_config_daniel.yaml invalidates dependents)
    + all model/pipeline code (_code_inputs) + the default input CSV the model config points
    at + the covert config's Plan-A series (if any)."""
    files: list[Path] = []
    seen: set[Path] = set()
    extra_csv = [c for c in (_resolve_input_csv(model_cfg), _resolve_plan_a_csv(covert_cfg),
                             _resolve_bottom_up_cfg(covert_cfg)) if c]
    for f in ([c for cfg in (model_cfg, covert_cfg) for c in _config_chain(cfg)]
              + _expand(_code_inputs())
              + extra_csv):
        if f not in seen:
            seen.add(f)
            files.append(f)
    return files


def _push_build_inputs(model_cfg: Path, covert_cfg: Path) -> None:
    """rsync everything a remote build depends on to the box, mirroring _input_files, so
    the box's code+config+data match exactly what the runid hashes. Directories go via
    push_tree; individual files via push. Cheap after the first time (rsync is incremental)."""
    pushed: set[Path] = set()
    # configs (incl. parent chains, e.g. the daniel/sampling parents outside the covert tree)
    for cfg in (model_cfg, covert_cfg):
        for f in _config_chain(cfg):
            if f not in pushed:
                pushed.add(f); rmt.push(str(f), str(f.relative_to(ROOT)))
    for csv in (c for c in (_resolve_input_csv(model_cfg), _resolve_plan_a_csv(covert_cfg),
                            _resolve_bottom_up_cfg(covert_cfg)) if c):
        if csv not in pushed:
            pushed.add(csv); rmt.push(str(csv), str(csv.relative_to(ROOT)))
    for p in _code_inputs():
        rel = str(p.relative_to(ROOT))
        if p.is_dir():
            rmt.push_tree(str(p), rel, excludes=["__pycache__", "*.pyc"])
        else:
            rmt.push(str(p), rel)


def _assert_remote_in_sync(input_files: list[Path]) -> None:
    """Belt-and-suspenders: hash the build inputs on the BOX and compare to the local
    hash. Identical => the box is byte-for-byte in sync; abort loudly otherwise (a push
    gap / exclude). This makes a silently-stale box impossible, not just unlikely."""
    rels = [str(f.relative_to(ROOT)) for f in input_files]
    local = _files_digest(input_files)
    pyc = ("import hashlib;fs=%r;h=hashlib.sha256()\n"
           "for f in fs:\n try:h.update(open(f,'rb').read())\n"
           " except FileNotFoundError:h.update(b'MISSING')\n"
           "print(h.hexdigest())" % rels)
    import shlex
    remote = rmt.ssh(f"{rmt.PY} -c {shlex.quote(pyc)}").strip().splitlines()[-1].strip()
    if remote != local:
        sys.exit(f"remote build inputs DO NOT match local (local {local[:12]} != box {remote[:12]}).\n"
                 f"The push didn't fully sync the box; refusing to build on stale inputs.")


def _git_sha():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True).stdout.strip() or None
    except Exception:
        return None


def _files_digest(files: list[Path]) -> str:
    """sha256 over the concatenated bytes of `files`, in order (missing => b'MISSING').
    Shared by the runid and the remote-sync check so both see the exact same content."""
    h = hashlib.sha256()
    for f in files:
        h.update(f.read_bytes() if f.exists() else b"MISSING")
    return h.hexdigest()


def _runid(params: dict, input_files: list[Path]) -> str:
    h = hashlib.sha256()
    h.update(bytes.fromhex(_files_digest(input_files)))
    h.update(json.dumps(params, sort_keys=True).encode())
    return h.hexdigest()[:12]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num", type=int, default=500)
    # seed / years-before-ac / legal-mode now live in the covert yaml; these flags are
    # optional OVERRIDES (None => take the value from the config).
    ap.add_argument("--seed", type=int, default=None, help="override the covert yaml `seed`")
    ap.add_argument("--years-before-ac", type=float, default=None,
                    help="override the covert yaml years_before_ac knob (pins it fixed)")
    ap.add_argument("--legal-mode", choices=["backsolve", "capped"], default=None,
                    help="override the covert yaml legal_mode knob (pins it fixed)")
    ap.add_argument("--min-survival-years", type=float, default=None,
                    help="override deal_breakdown.min_survival_years (condition on the deal "
                         "surviving at least this long; requires a deal_breakdown block)")
    ap.add_argument("--covert-exists-prior", type=float, default=None,
                    help="override the covert yaml covert_exists_prior (prior P(a covert "
                         "project exists) for the figures' Bayesian update)")
    ap.add_argument("--workers", type=int, default=30)
    ap.add_argument("--model-config", default="plan_a_covert_proj/config/model_uncertain_thomas.yaml",
                    help="model-parameters yaml (abs or repo-relative; remote builds need it under the repo tree)")
    ap.add_argument("--covert-config", default="plan_a_covert_proj/config/covert_scenario_branch.yaml",
                    help="covert-knobs yaml (abs or repo-relative; remote builds need it under the repo tree)")
    ap.add_argument("--name", default=None,
                    help="human-readable label for this run (metadata only — NOT hashed into the "
                         "runid, so it never triggers a rebuild; shown in the run browser)")
    ap.add_argument("--local", action="store_true", help="build the dataset locally (default: aifm-runpod)")
    ap.add_argument("--force", action="store_true", help="rebuild dataset even if it exists")
    ap.add_argument("--only", default=None, help="comma-separated figure names to (re)build (default: all)")
    ap.add_argument("--solve-legal-compute", action="store_true",
                    help="also solve implied inference+experiment compute PER ROLLOUT (costly; "
                         "the median scenario always computes it)")
    args = ap.parse_args()

    def _resolve_cfg(p):
        q = Path(p)
        return (q if q.is_absolute() else (ROOT / q)).resolve()
    model_cfg, covert_cfg = _resolve_cfg(args.model_config), _resolve_cfg(args.covert_config)
    for c in (model_cfg, covert_cfg):
        if not c.exists():
            ap.error(f"config not found: {c}")
    input_files = _input_files(model_cfg, covert_cfg)

    # Resolve the effective seed once (CLI override > covert yaml `seed` > model yaml > 42)
    # so the SAME value drives the runid AND the remote build.
    covert_loaded = _load_config(str(covert_cfg)) or {}
    model_loaded = _load_config(str(model_cfg)) or {}
    eff_seed = args.seed if args.seed is not None else int(
        covert_loaded.get("seed", model_loaded.get("seed", 42)))

    # years_before_ac / legal_mode live in the covert yaml (hashed via its file content).
    # A CLI override is recorded here too so it perturbs the runid.
    params = {"num": args.num, "seed": eff_seed}
    if args.years_before_ac is not None:
        params["years_before_ac"] = args.years_before_ac
    if args.legal_mode is not None:
        params["legal_mode"] = args.legal_mode
    if args.min_survival_years is not None:
        params["min_survival_years"] = args.min_survival_years
    if args.covert_exists_prior is not None:
        params["covert_exists_prior"] = args.covert_exists_prior
    if args.solve_legal_compute:   # only perturbs the runid when enabled (adds per-rollout fields)
        params["solve_legal_compute"] = True
    runid = _runid(params, input_files)
    rundir = COVERT / "outputs" / "runs" / runid
    (rundir / "figures").mkdir(parents=True, exist_ok=True)
    dataset = rundir / "dataset.json"
    print(f"runid {runid}  ->  {rundir.relative_to(ROOT)}")

    # 1. dataset (reuse unless --force) ------------------------------------------------
    if dataset.exists() and not args.force:
        print("dataset exists — reusing (use --force to rebuild)")
    elif args.local:
        from plan_a_covert_proj.pipeline.mc import build_dataset
        ds = build_dataset(str(model_cfg), str(covert_cfg),
                           args.num, eff_seed, args.years_before_ac, args.legal_mode, args.workers,
                           solve_legal_compute=args.solve_legal_compute,
                           min_survival_years=args.min_survival_years,
                           covert_exists_prior=args.covert_exists_prior)
        dataset.write_text(json.dumps(ds, separators=(",", ":")))
        print(f"built dataset locally: {ds['summary']['n_ok']}/{ds['summary']['n_requested']} ok")
    else:
        # Remote builds resolve configs relative to REMOTE_ROOT, so every input must live
        # under the repo tree. Pass repo-relative paths — they resolve where ssh() runs.
        try:
            model_rel, covert_rel = model_cfg.relative_to(ROOT), covert_cfg.relative_to(ROOT)
        except ValueError:
            sys.exit("remote build needs both configs under the repo tree; put them inside "
                     "plan_a_covert_proj (or run with --local).")
        print("building dataset on", rmt.HOST, "...")
        # Push EVERYTHING the build depends on (code + configs + data), mirroring the
        # runid's input set, then verify the box is byte-identical before building — so a
        # stale box can never silently produce wrong results under a valid-looking runid.
        rmt.push_tree(str(COVERT), "plan_a_covert_proj", excludes=["outputs", "__pycache__", "*.pyc"])
        _push_build_inputs(model_cfg, covert_cfg)
        _assert_remote_in_sync(input_files)
        remote_out = f"plan_a_covert_proj/outputs/_build_{runid}.json"
        # years_before_ac / legal_mode come from the covert yaml on the box; only pass
        # the override flags when the user actually supplied them.
        override = ""
        if args.years_before_ac is not None:
            override += f" --years-before-ac {args.years_before_ac}"
        if args.legal_mode is not None:
            override += f" --legal-mode {args.legal_mode}"
        if args.min_survival_years is not None:
            override += f" --min-survival-years {args.min_survival_years}"
        if args.covert_exists_prior is not None:
            override += f" --covert-exists-prior {args.covert_exists_prior}"
        rmt.ssh(f"{rmt.PY} -W ignore -m plan_a_covert_proj.pipeline.mc "
                f"--model-config {model_rel} --covert-config {covert_rel} "
                f"--num {args.num} --seed {eff_seed}{override} "
                f"--workers {args.workers} --out {remote_out}"
                + (" --solve-legal-compute" if args.solve_legal_compute else ""))
        rmt.pull(remote_out, str(dataset))
        print("pulled dataset")

    ds = json.loads(dataset.read_text())

    # 2. figures (always rebuilt from the dataset — they're cheap & deterministic) -----
    only = set(args.only.split(",")) if args.only else None
    built = []
    for name, fn in figmod.FIGURES:
        if only and name not in only:
            continue
        try:
            r = fn(ds, rundir / "figures")
            if r is None:                      # figure opted out (e.g. not applicable to this run)
                print(f"  figure {name} -> skipped (not applicable)")
                continue
            built.append(r)
            print(f"  figure {name} -> {r}")
        except Exception as e:  # noqa: BLE001
            print(f"  figure {name} FAILED: {e}")

    # 2b. compute-acquisition tables (markdown + HTML in the run dir) — bottom-up runs only.
    try:
        tbl_files = tabmod.generate(ds, rundir)
        if tbl_files:
            print(f"  tables -> {', '.join(tbl_files)}")
    except Exception as e:  # noqa: BLE001
        tbl_files = []
        print(f"  tables FAILED: {e}")

    # 3. manifest + latest symlink -----------------------------------------------------
    # Full resolved config snapshot (distributions + ci80 + min/max bounds + correlation
    # targets), à la the website forecast page's input_distributions.yaml. _load_config
    # merges the shared daniel parent, so this is exactly what was sampled. Drop the
    # derived latent correlation matrix (a calibration artifact, not an input).
    def _snap(cfg_path):
        c = _load_config(str(cfg_path)) or {}
        cm = c.get("correlation_matrix")
        if isinstance(cm, dict):
            c["correlation_matrix"] = {k: v for k, v in cm.items() if k != "latent_correlation_matrix"}
        return c
    def _rel(f):
        try:
            return str(f.relative_to(ROOT))
        except ValueError:
            return str(f)
    # Run name is metadata only. If none was passed, keep any name a prior build of this
    # (identical-runid) run already recorded, rather than clobbering it with None.
    name = (args.name or "").strip() or None
    if name is None and (rundir / "manifest.json").exists():
        try:
            name = (json.loads((rundir / "manifest.json").read_text()) or {}).get("name")
        except Exception:  # noqa: BLE001
            pass
    manifest = {
        "runid": runid, "name": name, "params": params, "git_sha": _git_sha(),
        "summary": ds["summary"], "config": ds["config"],
        "input_files": [_rel(f) for f in input_files],
        "config_snapshot": {"model": _snap(model_cfg), "covert": _snap(covert_cfg)},
        # all figures present in the run dir (not just this invocation's --only subset)
        "figures": sorted(p.name for p in (rundir / "figures").iterdir() if p.is_file()),
        "figures_built_now": built,
        "tables": tbl_files,
    }
    (rundir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    latest = COVERT / "outputs" / "latest"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(Path("runs") / runid)
    print(f"manifest + latest -> {runid}; {len(built)} figures in {rundir.relative_to(ROOT)}/figures")

    # Refresh the run browser (outputs/viewer.html) so it always lists the latest run.
    _vpath, _vn = make_viewer.regenerate()
    if _vpath is not None:
        print(f"viewer -> {_vpath.relative_to(ROOT)} ({_vn} runs)")


if __name__ == "__main__":
    main()
