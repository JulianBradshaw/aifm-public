#!/usr/bin/env python3
"""Compact viewer for plan_a_covert_proj configs.

The covert/model yamls are heavily commented and inherit from parents (a covert
yaml -> covert_scenario_branch.yaml; a model yaml -> ../../config/sampling_config_daniel.yaml),
so eyeballing the *effective* parameters means reading through a lot of prose and
tracing the inheritance chain. This prints the RESOLVED parameters as a compact,
one-line-per-knob list instead — each distribution rendered as e.g.

    frac_stolen              = 0.67                       (fixed)
    total_h100e_frac         = logN ci80[0.004, 0.04]     -> med 0.01265
    legal_strategy           = target_year_actual_hw      (fixed)

Usage:
    python -m plan_a_covert_proj.pipeline.show_config                       # default model+covert pair
    python -m plan_a_covert_proj.pipeline.show_config config/covert_uncertain.yaml
    python -m plan_a_covert_proj.pipeline.show_config config/model_uncertain_thomas.yaml config/covert_uncertain.yaml
    python -m plan_a_covert_proj.pipeline.show_config --all                 # every yaml under config/

Paths may be given relative to CWD, relative to plan_a_covert_proj/, or as
bare config names (config/ is searched). Add --sampled-only to hide fixed knobs.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent                       # plan_a_covert_proj/
REPO_ROOT = PROJ.parent
CONFIG_DIR = PROJ / "config"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import yaml  # noqa: E402
from scripts.pipelines.batch_rollout import (  # noqa: E402
    _load_config,
    _sample_from_dist_with_quantile,
)

# Top-level keys that hold structured blocks (rendered specially, not as scalars).
_STRUCTURED = {
    "parameters",
    "covert_parameters",
    "bottom_up_compute_parameters",
    "time_series_parameters",
    "correlation_matrix",
    "detection_grids",
    "detection_silver_bullet",
}


def _resolve_path(arg: str) -> Path:
    """Resolve a user-supplied config path against CWD, the project dir, and config/."""
    p = Path(arg)
    candidates = [p]
    if not p.is_absolute():
        candidates += [PROJ / p, CONFIG_DIR / p, CONFIG_DIR / p.name]
    for c in candidates:
        if c.exists():
            return c.resolve()
    raise FileNotFoundError(f"config not found: {arg} (looked in ., {PROJ}, {CONFIG_DIR})")


def _parent_chain(path: Path) -> list[Path]:
    """The inheritance chain (parents only, nearest first), read from raw `parent:` fields."""
    chain: list[Path] = []
    cur = path
    seen = {cur}
    while True:
        raw = yaml.safe_load(cur.read_text()) or {}
        parent = raw.get("parent")
        if not parent:
            break
        pp = Path(parent)
        pp = (cur.parent / pp).resolve() if not pp.is_absolute() else pp.resolve()
        if pp in seen:
            break
        chain.append(pp)
        seen.add(pp)
        cur = pp
    return chain


def _num(v) -> str:
    """Compact number formatting: ints as ints, floats to 5 significant figures."""
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if v != v:  # NaN
            return "nan"
        s = f"{v:.5g}"
        return s
    return str(v)


def _fmt_list(v) -> str:
    if isinstance(v, list):
        return "[" + ", ".join(_num(x) for x in v) + "]"
    return _num(v)


def _median(spec: dict, name: str):
    try:
        return _sample_from_dist_with_quantile(spec, 0.5, name)
    except Exception:
        return None


def _fmt_dist(spec, name: str) -> tuple[str, str]:
    """Return (compact_spec_string, annotation). annotation is e.g. '(fixed)' or '-> med X'."""
    if not isinstance(spec, dict):
        return _fmt_list(spec), ""

    kind = spec.get("dist", "fixed")

    if kind == "fixed":
        return _num(spec.get("value")), "(fixed)"

    # Support / clip bounds suffix, shown only when present.
    bounds = ""
    if "min" in spec or "max" in spec:
        bounds = f" in[{_num(spec.get('min'))}, {_num(spec.get('max'))}]"

    def with_med(core: str) -> tuple[str, str]:
        med = _median(spec, name)
        ann = f"-> med {_num(med)}" if med is not None else ""
        return core + bounds, ann

    if kind == "choice":
        vals = spec.get("values", [])
        ps = spec.get("p")
        if ps and len(ps) == len(vals):
            body = ", ".join(f"{_num(v)}:{_num(p)}" for v, p in zip(vals, ps))
        else:
            body = ", ".join(_num(v) for v in vals)
        return "choice{" + body + "}", ""

    if kind == "uniform":
        # min/max ARE the params here, so don't append the redundant bounds suffix.
        core = f"U[{_num(spec.get('min'))}, {_num(spec.get('max'))}]"
        med = _median(spec, name)
        return core, (f"-> med {_num(med)}" if med is not None else "")

    if kind in ("normal", "lognormal", "shifted_lognormal", "beta"):
        label = {
            "normal": "N",
            "lognormal": "logN",
            "shifted_lognormal": "shiftedLogN",
            "beta": "Beta",
        }[kind]
        if "ci80" in spec:
            ci = spec["ci80"]
            core = f"{label} ci80[{_num(ci[0])}, {_num(ci[1])}]"
        elif "ci80_low" in spec and "ci80_high" in spec:
            core = f"{label} ci80[{_num(spec['ci80_low'])}, {_num(spec['ci80_high'])}]"
        elif kind in ("normal", "lognormal") and ("mean" in spec or "mu" in spec):
            loc = spec.get("mean", spec.get("mu"))
            scale = spec.get("sd", spec.get("sigma"))
            core = f"{label}(loc={_num(loc)}, scale={_num(scale)})"
        else:
            core = label
        if kind == "shifted_lognormal" and "shift" in spec:
            core += f" +{_num(spec['shift'])}"
        return with_med(core)

    # Unknown dist kind: fall back to dumping the spec keys.
    return f"{kind}{{{', '.join(f'{k}={_num(v)}' for k, v in spec.items() if k != 'dist')}}}", ""


def _render_param_block(title: str, block, sampled_only: bool) -> list[str]:
    if not isinstance(block, dict) or not block:
        return ["", f"[{title}]  (empty -> defaults)"]
    rendered = []
    for name, spec in block.items():
        core, ann = _fmt_dist(spec, name)
        is_fixed = ann == "(fixed)" or (isinstance(spec, dict) and spec.get("dist", "fixed") == "fixed")
        if sampled_only and is_fixed:
            continue
        rendered.append((name, core, ann))
    n_hidden = len(block) - len(rendered)
    header = f"[{title}]  ({len(block)} knobs"
    if sampled_only and n_hidden:
        header += f", {n_hidden} fixed hidden"
    header += ")"
    lines = ["", header]
    if not rendered:
        lines.append("  (all fixed)")
        return lines
    w = max(len(n) for n, _, _ in rendered)
    cw = max(len(c) for _, c, _ in rendered)
    for name, core, ann in rendered:
        line = f"  {name:<{w}} = {core:<{cw}}"
        if ann:
            line += f"  {ann}"
        lines.append(line.rstrip())
    return lines


def _render_detection(cfg: dict) -> list[str]:
    grids = cfg.get("detection_grids")
    sb = cfg.get("detection_silver_bullet")
    if not grids and not sb:
        return []
    lines = ["", "[detection]"]
    if grids:
        sizes = ", ".join(f"{g.get('size_mw')}MW({len(g.get('rows', []))} rows)" for g in grids)
        lines.append(f"  detection_grids       = {len(grids)} size grids: {sizes}")
    if sb:
        parts = ", ".join(f"{k}={_num(v)}" for k, v in sb.items())
        lines.append(f"  detection_silver_bullet = {parts}")
    return lines


def _config_kind(cfg: dict) -> str:
    if "covert_parameters" in cfg:
        return "covert config"
    if "bottom_up_compute_parameters" in cfg:
        return "bottom-up compute config"
    if "parameters" in cfg:
        return "model config"
    return "config"


def render(path: Path, sampled_only: bool = False) -> str:
    """Compact one-line-per-parameter view of a config, with parent inheritance resolved.

    Returns a plain string (used by both the CLI and the /launch web preview)."""
    path = Path(path)
    cfg = _load_config(str(path)) or {}
    chain = _parent_chain(path)

    rel = path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path
    lines = [f"{rel}   [{_config_kind(cfg)}]"]
    if chain:
        pretty = " <- ".join(
            (c.relative_to(REPO_ROOT).as_posix() if c.is_relative_to(REPO_ROOT) else c.name)
            for c in chain
        )
        lines.append(f"inherits: {pretty}")

    # Top-level scalar run settings (anything not a structured block, not `parent`).
    scalars = {k: v for k, v in cfg.items() if k not in _STRUCTURED and k != "parent"}
    if scalars:
        lines.append("")
        lines.append("[run settings]")
        w = max(len(k) for k in scalars)
        for k, v in scalars.items():
            lines.append(f"  {k:<{w}} = {_fmt_list(v)}")

    if "parameters" in cfg:
        lines += _render_param_block("model parameters", cfg.get("parameters"), sampled_only)
    if "covert_parameters" in cfg:
        lines += _render_param_block("covert_parameters", cfg.get("covert_parameters"), sampled_only)
    if "bottom_up_compute_parameters" in cfg:
        lines += _render_param_block("bottom_up_compute_parameters", cfg.get("bottom_up_compute_parameters"), sampled_only)
    if "time_series_parameters" in cfg:
        lines += _render_param_block("time_series_parameters", cfg.get("time_series_parameters"), sampled_only)

    lines += _render_detection(cfg)
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("configs", nargs="*", help="config yaml paths (bare names resolved against config/)")
    ap.add_argument("--all", action="store_true", help="show every *.yaml under config/")
    ap.add_argument("--sampled-only", action="store_true", help="hide fixed knobs (show only sampled distributions)")
    args = ap.parse_args()

    if args.all:
        paths = sorted(CONFIG_DIR.glob("*.yaml"))
    elif args.configs:
        paths = [_resolve_path(a) for a in args.configs]
    else:
        # Default: the pipeline's default model + covert pair.
        paths = [
            CONFIG_DIR / "model_uncertain_thomas.yaml",
            CONFIG_DIR / "covert_scenario_branch.yaml",
        ]

    for p in paths:
        print("=" * 78)
        print(render(p, args.sampled_only))
        print()


if __name__ == "__main__":
    main()
