#!/usr/bin/env python3
"""Minimal covert-project simulator — the physics from the AI-futures "game"
engine, with none of the game scaffolding (no turns, fog-of-war, scoring,
pin-one sampling, milestones-as-UI, or deal-duration logic).

Pipeline
--------
    run_default(params, data)            reference (un-paused) run: TED-AI level,
                                         AC time, start state, rebase offset
    solve_legal_sw(..., target_year)     back-solve the legal software rate so the
                                         legal project reaches TED-AI + alignment
                                         tax at target_year (1-D root-find; no turns)
    build_legal(..., hw_rate, sw_rate)   legal project frozen at the pause, then
                                         HW/SW growing at the policy rates — the
                                         trajectory the covert steals from
    make_covert_data / integrate_covert  the covert project itself (own R&D +
                                         optional S-theft + optional dormancy)
    simulate_covert(...)                 one covert-race record

Coordinates
-----------
All internal integration runs in RAW progress coordinates, because
``progress_rate_at_time`` computes EC = TC + SW in true (un-rebased) units. A
single additive ``rebase_offset`` (= −SE(reference year)) maps raw E to the
model's rebased ``effective_compute`` metric, so comparisons against TED-AI (a
rebased milestone) are consistent. This mirrors ``aifutures_engine/physics.py``
in the monorepo, which is the correctness reference for this module.

The legal project's software-efficiency trajectory is treated as an exogenous
*policy* (a software OOM/yr rate), exactly as the game's growth-rate mode does:
SW is the input and research stock/effort are back-derived from it
(``rs = RS_0 · 10**(SW / r_software)``). The covert project couples to the legal
project only through that SE trajectory plus the at-pause start scalars.
"""

from __future__ import annotations

import copy
import logging

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import brentq

from progress_model import config as cfg

from progress_model.types import TimeSeriesData
from progress_model.parameters import Parameters  # noqa: F401  (re-exported for callers)
from progress_model._impl import ProgressModel
from progress_model.metrics_computation import compute_metrics_loop
from progress_model.progress_rate import progress_rate_at_time

logger = logging.getLogger(__name__)

LN10 = np.log(10.0)
_LOG10_FPH = np.log10(cfg.FLOP_PER_H100E_YEAR)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log_interp(t, times, values):
    """Log-space interpolation (linear fallback when any value is non-positive)."""
    values = np.asarray(values, dtype=float)
    if np.any(values <= 0):
        return float(np.interp(t, times, values))
    return float(np.exp(np.interp(t, times, np.log(values))))


def _tc_ooms(data, t):
    """Training system performance at t: raw rate log10(FLOP/yr) from raw H100e.

    CWD ``TimeSeriesData`` stores raw H100e and has no ``training_compute_ooms``
    property, so we convert at the point of use: log10(H100e) + log10(FLOP/H100e-yr).
    """
    h = _log_interp(t, data.time, data.training_H100e)
    return np.log10(max(h, 1e-300)) + _LOG10_FPH


def _crossing_time(t, e, level):
    """First time the (monotone-ish) series ``e`` reaches ``level`` (lerp), or None."""
    e = np.asarray(e, dtype=float)
    above = np.where(e >= level)[0]
    if len(above) == 0:
        return None
    i = int(above[0])
    if i == 0:
        return float(t[0])
    e0, e1, t0, t1 = e[i - 1], e[i], t[i - 1], t[i]
    if e1 == e0:
        return float(t1)
    return float(t0 + (level - e0) / (e1 - e0) * (t1 - t0))


# ---------------------------------------------------------------------------
# Bottom-up covert compute (ai-2030 covert-AI-projects supplement §2.3, Method #1)
# ---------------------------------------------------------------------------
# The diversion-parameter knobs sampled per draw (must match the keys in
# covert_compute_bottom_up.yaml's `bottom_up_compute_parameters`).
_BOTTOM_UP_KEYS = (
    "decommission_feasible", "decommission_frac", "prc_stock_frac_world_2028",
    "china_domestic_frac_world", "domestic_feasible", "domestic_concealable_frac",
    "smuggled_frac", "smuggled_unidentifiable_frac",
    # Diversion TIMING (relative to each rollout's pause). Sampled + recorded but NOT consumed by
    # bottom_up_covert_fraction (which returns pure fractions): timing enters the ABSOLUTE sizing in
    # simulate_covert, by setting the window over which world compute is accumulated.
    #   declaration_to_pause_years -- compute-declaration (books close) -> pause gap (~Jan1->Mar1);
    #                                 sets the window END = pause - this.
    #   diversion_duration         -- window length; the window is
    #                                 [declaration - diversion_duration, declaration]. The diverted
    #                                 pool = world compute PRODUCED over the window, so a longer
    #                                 window => more diverted (DRIVES the amount together with frac).
    "declaration_to_pause_years", "diversion_duration",
    # Facility POWER (supplement §3.3 "How much datacenter capacity does the diverted
    # compute require?"): per-source W/H100e is derived from these + the diversion window
    # (bottom_up_facility_w_per_h100e), so the covert's detection-relevant MW reflects the
    # sampled source mix — decommissioned/domestic chips are less power-efficient than
    # freshly smuggled US chips.
    "it_w_per_h100e_2026", "power_efficiency_per_year", "smuggled_proxy_offset_years",
    "decommissioned_age_years", "domestic_efficiency_gap", "peak_pue",
)


def bottom_up_covert_fraction(p: dict) -> dict:
    """Per-source diversion fractions, each as a fraction of its OWN NATURAL BASE (covert-AI-projects
    supplement §2.3, Method #1). simulate_covert multiplies these by the world compute over the
    diversion window: the STOCK at the window start for false decommissioning, and the FLOW produced
    during the window for smuggling + concealed-domestic. ``p`` is one sampled draw of
    ``_BOTTOM_UP_KEYS`` (feasibility Bernoullis are 1.0/0.0). Returns natural-base fractions:

      decommission_stock_frac -- of the pre-deal STOCK:
            feasible * decommission_frac * (prc_stock_frac_world * us_designed_frac_china)
      smuggled_flow_frac      -- of the window FLOW:
            smuggled_frac * smuggled_unidentifiable_frac
      domestic_flow_frac      -- of the window FLOW:
            domestic_feasible * domestic_concealable_frac * china_domestic_frac_world

      us_designed_frac_china = 1 - china_domestic_frac_world / prc_stock_frac_world_2028   (~0.45)

    The 2028.0/2029.0 conversion factors the old reconstruction used (pre2028_ratio = 1 - flow,
    flow_2028_frac) are GONE: the diversion window itself now selects the stock vs flow, so those
    fixed-year conversions are unnecessary. With a window of [2028.0, 2029.0] this reproduces the
    supplement's source magnitudes; the composition ratios prc_stock_frac_world_2028 and
    china_domestic_frac_world are treated as (time-invariant) shares of world stock/flow."""
    g = lambda k: float(p[k])
    prc = g("prc_stock_frac_world_2028")
    dom_world = g("china_domestic_frac_world")
    us_designed_frac_china = 1.0 - (dom_world / prc if prc > 0 else 0.0)
    decommission_stock_frac = (g("decommission_feasible") * g("decommission_frac")
                               * prc * us_designed_frac_china)
    smuggled_flow_frac = g("smuggled_frac") * g("smuggled_unidentifiable_frac")
    domestic_flow_frac = (g("domestic_feasible") * g("domestic_concealable_frac") * dom_world)
    return {"decommission_stock_frac": decommission_stock_frac,
            "smuggled_flow_frac": smuggled_flow_frac,
            "domestic_flow_frac": domestic_flow_frac}


def bottom_up_facility_w_per_h100e(p: dict, window_start: float, declaration: float) -> dict:
    """Per-source facility power (W per H100e) for the bottom-up sizing (supplement §3.3
    "How much datacenter capacity does the diverted compute require?"). US IT W/H100e is
    anchored at 2026.0 (it_w_per_h100e_2026, DGX GB200 SuperPOD: 833) and improves
    power_efficiency_per_year x/yr; each source is evaluated at a proxy year tied to the
    diversion window [window_start, declaration]:
      smuggled       -- window midpoint + smuggled_proxy_offset_years (chips flow during
                        the window; the small forward offset corrects for the average of an
                        exponential lying past its midpoint). Supplement: 2028.6 for the
                        [2028, 2029] window.
      decommissioned -- window_start - decommissioned_age_years (the falsely-decommissioned
                        pre-deal stock is older than the window). Supplement: 2026.5.
      domestic       -- same proxy year as smuggled, times domestic_efficiency_gap
                        (Chinese-designed chips are ~3.7x less power-efficient than
                        contemporaneous US chips).
    All are multiplied by peak_pue (facility power / IT power at max utilization), so these
    are FACILITY watts: covert facility MW = sum_i h100e_i * w_i / 1e6."""
    w26 = float(p["it_w_per_h100e_2026"])
    rate = float(p["power_efficiency_per_year"])
    pue = float(p["peak_pue"])
    gap = float(p["domestic_efficiency_gap"])
    smuggled_proxy = 0.5 * (window_start + declaration) + float(p["smuggled_proxy_offset_years"])
    decommission_proxy = window_start - float(p["decommissioned_age_years"])

    def us_w(year: float) -> float:
        return w26 / rate ** (year - 2026.0)

    return {"smuggled_w": us_w(smuggled_proxy) * pue,
            "decommission_w": us_w(decommission_proxy) * pue,
            "domestic_w": us_w(smuggled_proxy) * gap * pue}


# ---------------------------------------------------------------------------
# Reference run
# ---------------------------------------------------------------------------

# Dense (E, uplift) map resolution. The webapp's uplift axis maps each trajectory's E(t)
# through the reference run's (E, uplift) relationship. The "ref present day" uplift is a
# pure function of capability level E (it holds resources fixed at present day), but the
# model's native ODE grid is sparse in E across the steep takeoff (~1-2 month steps that
# each span a large E jump), so a map built from those samples is kinky/inaccurate. We
# instead EVALUATE uplift on a dense, uniform-E grid by re-running the same vectorized
# metrics pass — exact values at every grid point, not interpolated from sparse samples.
_UPLIFT_MAP_POINTS = 800


def _build_metrics_ctx(model, RS_0):
    """Bundle everything `compute_metrics_loop` needs from a finished reference run, so any
    project's trajectory can be re-run through the SAME canonical metrics pass (with the
    reference present-day baseline, which keeps the ref-present-day counterfactuals — uplift,
    etc. — comparable across projects). Stored on the ref dict as `metrics_ctx`."""
    anchor = ((getattr(model, 'human_only_results', None) or {}).get('anchor_stats')) or {}
    return {
        'params': model.params,
        'anchor': anchor,
        'taste_distribution': model.taste_distribution,
        'horizon_trajectory': getattr(model, 'horizon_trajectory', None),
        # Post-rebase observables from the reference run: capability (ECI) and any
        # AC-anchor metric are pure functions of E, so every project trajectory is
        # mapped through the SAME reference trend.
        'extra_observables': model.extra_observables(),
        'RS_0': float(RS_0),
    }


def _metrics_pass(times, progress, RS, data, ctx):
    """Thin wrapper over the model's vectorized `compute_metrics_loop` (NOT a reimplementation)
    for an arbitrary trajectory + its effective input series `data`, using the reference-run
    context `ctx`. Returns the full metrics dict."""
    anchor = ctx['anchor']
    progress = np.asarray(progress, dtype=float)
    metrics, _, _ = compute_metrics_loop(
        times=np.asarray(times, dtype=float),
        progress_values=progress,
        research_stock_values=np.asarray(RS, dtype=float),
        data=data,
        params=ctx['params'],
        initial_progress=float(progress[0]),
        present_day_human_labor=anchor['human_labor'],
        present_day_inference_compute=anchor['inference_H100e'],
        present_day_experiment_compute=anchor['experiment_H100e'],
        present_day_research_stock=anchor['research_stock'],
        present_day_human_only_research_effort=anchor['research_effort'],
        taste_distribution=ctx['taste_distribution'],
        horizon_trajectory=ctx['horizon_trajectory'],
        takeoff_start_human_only_stats=None,
        RS_0=ctx['RS_0'],
        extra_observables=ctx['extra_observables'],
    )
    return metrics


def project_metrics(times, logE_raw, RS, cdata, ctx):
    """Run the canonical AIFM metrics pass on a covert/legal project's trajectory.

    `times`, `logE_raw` (raw progress = un-rebased logE), `RS` are the project's own
    trajectory; `cdata` is its EFFECTIVE input series (e.g. the covert datacenter's compute
    from `make_covert_data`, or the legal's reconstructed implied compute); `ctx` is the
    reference run's `metrics_ctx`. Returns the full `compute_metrics_loop` dict.

    IMPORTANT subtlety: the metrics pass RE-DERIVES research-stock growth from the inputs, so
    the *rate* metrics (`research_efforts`, `software_progress_rates`, `progress_rates`,
    `etsp_rates`) describe the project's INDIGENOUS R&D only — they omit anything that lived
    in the ODE RHS rather than in (E, RS, inputs): e.g. the covert's stolen RS-growth + theft
    boost, or the legal's back-solved SW rate. STOCK-derived metrics (`software_efficiency`
    from RS, `effective_compute`, `automation_fractions`, `capability_ecis`, `training_*`) and
    the ref-present-day counterfactuals are faithful. Splice the sim's own rate series back in
    where the realized rate matters."""
    return _metrics_pass(times, logE_raw, RS, cdata, ctx)


# Capability-indexed metrics (all PURE functions of E / ref-present-day, so they map onto
# ANY trajectory's E(t)). client/series field name -> compute_metrics_loop output key.
_CAP_METRICS = {
    'uplift': 'ai_sw_progress_mult_ref_present_day',
    'codingLaborMult': 'ai_coding_labor_mult_ref_present_day',
    'researchTaste': 'ai_research_tastes',
}


def _dense_capability_maps(ctx, data, times, progress, RS, ec, rebase_offset):
    """High-resolution (rebased-E -> capability-indexed metric) maps for the reference run.

    Re-evaluates each metric in `_CAP_METRICS` — all pure functions of progress (logE) /
    ref-present-day — on a uniform grid of `_UPLIFT_MAP_POINTS` E values spanning the run's
    full E range, via `_metrics_pass` (the model's own `compute_metrics_loop`). The model's
    native ODE grid is sparse in E across the steep takeoff, so a map built from those samples
    is kinky; evaluating on a dense uniform-E grid is exact at every point. `times`/`RS` are
    required by the pass but do NOT affect these ref-present-day quantities, so we synthesise
    them from the native trajectory. Returns (e_grid, {name: values}) or (None, None) on any
    failure (callers fall back to the native, sparse samples for uplift).
    """
    try:
        if not ctx.get('anchor'):
            return None, None
        finite = np.isfinite(ec) & np.isfinite(progress)
        if finite.sum() < 2:
            return None, None
        emin, emax = float(np.min(ec[finite])), float(np.max(ec[finite]))
        if not (emax > emin):
            return None, None
        # Sort the native trajectory by progress so we can invert progress -> (time, RS).
        order = np.argsort(progress[finite])
        prog_n = progress[finite][order]
        time_n = np.asarray(times)[finite][order]
        rs_n = np.asarray(RS)[finite][order]

        e_grid = np.linspace(emin, emax, _UPLIFT_MAP_POINTS)
        prog_grid = e_grid - rebase_offset
        # Synthetic time / research-stock for the metrics pass (these metrics ignore both).
        t_grid = np.interp(prog_grid, prog_n, time_n)
        rs_grid = np.exp(np.interp(prog_grid, prog_n, np.log(np.maximum(rs_n, 1e-300))))

        metrics = _metrics_pass(t_grid, prog_grid, rs_grid, data, ctx)
        maps, ok = {}, np.ones(e_grid.shape, dtype=bool)
        for name, key in _CAP_METRICS.items():
            v = np.asarray(metrics.get(key, []), dtype=float)
            if v.shape != e_grid.shape:
                return None, None
            maps[name] = v
            ok &= np.isfinite(v)   # drop tail points where ANY metric is non-finite
        if ok.sum() < 2:
            return None, None
        return e_grid[ok], {name: v[ok] for name, v in maps.items()}
    except Exception as e:  # noqa: BLE001
        logger.warning(f"dense capability maps failed ({e}); falling back to native samples")
        return None, None


def run_default(params: Parameters, data: TimeSeriesData, time_range=None) -> dict:
    """Run the reference (un-paused) model once and extract what the covert sim needs."""
    if time_range is None:
        time_range = [float(data.time[0]), float(data.time[-1])]
    # ProgressModel.compute_progress_trajectory MUTATES params in place (sets
    # progress_at_aa and converts the taste / automation-efficiency slopes from
    # per-progress-year to per-progress-unit). Deep-copy so calling run_default
    # twice with the SAME Parameters object can't double-convert / corrupt it.
    model = ProgressModel(copy.deepcopy(params), copy.deepcopy(data))
    model.compute_progress_trajectory(time_range, rebase_output=True)
    r = model.results

    times = np.asarray(r['times'], dtype=float)
    progress = np.asarray(r['progress'], dtype=float)            # raw integration variable
    ec = np.asarray(r['effective_compute'], dtype=float)         # rebased (SE = 0 at ref)
    RS = np.asarray(r['research_stock'], dtype=float)
    RE = np.asarray(r['research_efforts'], dtype=float)
    # AI software R&D uplift ("ref present day"): the webapp's main metric. It is a
    # ref-present-day counterfactual, hence a pure function of capability level E —
    # so it can be mapped onto any project's E(t) via the (E, uplift) pairs below.
    uplift = np.asarray(r.get('ai_sw_progress_mult_ref_present_day', []), dtype=float)

    # rebase_offset = rebased E − raw progress = −SE(reference year). Rebasing is a pure
    # additive shift (effective_compute = progress.copy() − se_at_ref upstream), so this
    # difference is the SAME constant at every sample. Assert that before collapsing it to
    # a scalar — if it ever stops being constant the rebasing changed and the raw↔rebased
    # mapping used throughout this module is no longer valid.
    ec_minus_progress = ec - progress
    finite = np.isfinite(ec_minus_progress)
    rebase_offset = float(np.median(ec_minus_progress[finite]))
    if np.any(finite):
        spread = float(np.ptp(ec_minus_progress[finite]))
        assert spread < 1e-6, (
            f"rebase offset (effective_compute − progress) is not constant "
            f"(spread {spread:.3e} OOMs); rebasing is no longer a pure additive shift"
        )

    ms = r.get('milestones', {}) or {}

    def _ms_e(name):
        m = ms.get(name)
        return float(m['effective_compute_ooms']) if m and 'effective_compute_ooms' in m else None

    def _ms_t(name):
        m = ms.get(name)
        return float(m['time']) if m and 'time' in m else None

    # Context for re-running the canonical metrics pass on arbitrary project trajectories
    # (covert/legal) and for the dense capability maps below.
    metrics_ctx = _build_metrics_ctx(model, float(RS[0]))
    # Dense, accurate (rebased-E -> metric) maps for the webapp's capability-indexed axes
    # (uplift, coding-labor multiplier, research taste — all pure functions of E). Falls back
    # to None → callers use the native samples (uplift) / NaN (the rest).
    cap_e, cap_maps = _dense_capability_maps(metrics_ctx, model.data, times, progress, RS, ec, rebase_offset)
    uplift_map_e = cap_e
    uplift_map_u = (cap_maps['uplift'] if cap_maps else None)
    cap_map_clm = (cap_maps['codingLaborMult'] if cap_maps else None)
    cap_map_taste = (cap_maps['researchTaste'] if cap_maps else None)

    ac_time, ted_ai_time = _ms_t('AC'), _ms_t('TED-AI')
    # Per-rollout anchor progress rate (TRUE d(progress)/dt at the ~2026 anchor, post the
    # progress-rates naming reform). The model's OOM↔progress-year exchange rate (also
    # drives the taste/automation slope conversions). Used to convert tax/windfall knobs
    # specified in 2025-progress-years/months into OOMs of effective compute.
    _anchor = r.get('anchor_stats') or {}
    anchor_progress_rate = _anchor.get('progress_rate')
    if anchor_progress_rate is None:  # fallback: true progress_rate series at present_day
        pr = np.asarray(r.get('progress_rates', []), dtype=float)
        anchor_progress_rate = float(np.interp(float(params.present_day), times, pr)) if pr.size else float('nan')
    anchor_progress_rate = float(anchor_progress_rate)
    milestones = {
        name: {'e': float(m['effective_compute_ooms']), 'time': float(m['time'])}
        for name, m in ms.items()
        if isinstance(m, dict) and 'effective_compute_ooms' in m and 'time' in m
        and np.isfinite(m.get('effective_compute_ooms', np.nan))
    }
    return {
        # The model PREPARES params during the run (sets progress_at_aa, converts the
        # taste/automation slopes to per-progress-unit). Since we no longer mutate the
        # caller's params in place, expose the prepared copy here — downstream covert/legal
        # code (progress_rate_at_time, build_legal_*) needs the prepared form, not the raw
        # input params. Callers MUST pass this to simulate_covert, not the original params.
        'prepared_params': model.params,
        'milestones': milestones,   # {label: {'e': rebased OOMs, 'time': yr}}
        'times': times,
        'progress': progress,
        'effective_compute': ec,    # rebased OOMs (SE=0 at ref year)
        'uplift': uplift,           # ai_sw_progress_mult_ref_present_day, aligned with effective_compute
        # Dense, uniform-E (rebased-E -> metric) maps evaluated exactly (not interpolated from
        # the sparse native grid). `uplift_map_e` is the shared E grid; the *_u / cap_map_*
        # arrays are aligned to it. None when the dense pass failed → callers fall back to the
        # native (effective_compute, uplift) samples (uplift) / NaN (coding-labor mult, taste).
        'uplift_map_e': uplift_map_e,
        'uplift_map_u': uplift_map_u,
        'cap_map_clm': cap_map_clm,       # ai_coding_labor_mult_ref_present_day on uplift_map_e
        'cap_map_taste': cap_map_taste,   # ai_research_tastes on uplift_map_e
        # Context bundle for project_metrics() — re-run the canonical metrics pass on the
        # covert/legal trajectories. Server-side only (never serialized to the client).
        'metrics_ctx': metrics_ctx,
        'RS': RS,
        'RE': RE,
        'rebase_offset': rebase_offset,
        'RS_0': float(RS[0]),
        # PREPARED r_software: the model RESCALES r_software during the run (human-only
        # rescale, _impl.py) so the sw rate matches software_progress_rate_at_reference_year.
        # The legal build_* code reconstructs SW = r_software·log10(RS/RS_0) and accumulates
        # E from it, so it MUST use the rescaled value the model itself integrated with —
        # NOT the raw input params.r_software (the pre-calibration seed). Using the raw value
        # understated the legal's SW (and thus E), starting it OOMs below the default/covert
        # at the pause and throttling its whole trajectory. The covert code already uses the
        # prepared params.r_software directly, which is why it branched from default correctly.
        'r_software': float(model.params.r_software),
        'anchor_progress_rate': anchor_progress_rate,   # OOM/yr; OOM↔progress-year exchange rate
        'ted_ai_e': _ms_e('TED-AI'),
        'ac_e': _ms_e('AC'),
        'ac_time': ac_time,
        'ted_ai_time': ted_ai_time,
        'ac_to_ted_gap': (ted_ai_time - ac_time) if (ac_time is not None and ted_ai_time is not None) else None,
        # WORLD AI-relevant compute series (for compute_mode='bottom_up_estimate', which sizes the
        # covert as a fraction of world compute at the pause). None when the input has no such column.
        'world_h100e': getattr(data, 'world_H100e', None),
        'world_h100e_time': (data.time if getattr(data, 'world_H100e', None) is not None else None),
    }


# ---------------------------------------------------------------------------
# Legal trajectory (frozen at pause + exogenous policy SW), raw coordinates
# ---------------------------------------------------------------------------

def build_legal(ref: dict, data: TimeSeriesData, t_pause: float,
                hw_rate: float, sw_rate: float, t_end: float,
                setup_period_years: float = 0.0, setup_sw_gain_ooms: float = 0.0,
                sit_at_e_raw: float | None = None, n_fine_per_yr: int = 200):
    """Legal project frozen at ``t_pause``.

    Optional training-R&D pause over ``[t_pause, t_pause + setup_period_years]``:
    NO training occurs (TSP frozen, dE/dt = 0, so effective compute is flat), but
    ``setup_sw_gain_ooms`` of algorithmic progress accrues smoothly across the
    window — the radical-transparency windfall from labs pooling algorithms
    publicly under the deal. This raises software efficiency before training
    resumes, and (because the covert steals from the legal project's research
    stock) it feeds the covert's S-theft during the pause.

    After the pause, training resumes and HW grows at ``hw_rate`` OOM/yr, SW at
    ``sw_rate`` OOM/yr (the policy). SW is exogenous (policy + windfall); research
    stock/effort are back-derived from it. Returns ``(times, logE_raw, rs_legal,
    re_legal)`` in raw coordinates.
    """
    times_ref = ref['times']
    RS_0 = ref['RS_0']
    r_sw = ref['r_software']

    logE_pause = float(np.interp(t_pause, times_ref, ref['progress']))   # raw
    rs_at_pause = _log_interp(t_pause, times_ref, ref['RS'])
    sw_at_pause = r_sw * np.log10(max(rs_at_pause, 1e-300) / RS_0)       # raw SW
    tc_at_pause = _tc_ooms(data, t_pause)

    setup = max(0.0, float(setup_period_years))
    sw_resume = sw_at_pause + setup_sw_gain_ooms   # SW once the windfall has fully accrued

    T = max(t_end - t_pause, 1e-9)
    n = max(int(T * n_fine_per_yr) + 1, 2)
    tau = np.linspace(0.0, T, n)
    t_post = t_pause + tau
    in_setup = tau <= setup

    # Software efficiency: windfall accrues smoothly during the pause, then the
    # policy SW rate ramps from the post-windfall level.
    if setup > 0:
        sw_setup = sw_at_pause + setup_sw_gain_ooms * np.clip(tau / setup, 0.0, 1.0)
    else:
        sw_setup = np.full_like(tau, sw_at_pause)
    sw = np.where(in_setup, sw_setup, sw_resume + sw_rate * (tau - setup))

    # TSP frozen during the pause (no training), then ramps at hw_rate.
    tc = np.where(in_setup, tc_at_pause, tc_at_pause + hw_rate * (tau - setup))

    # dE/dt = 10^(TC+SW), but ZERO during the pause — no training runs, so effective
    # compute is flat across the window (the windfall lives in SW, not E).
    dEdt = np.where(in_setup, 0.0, 10.0 ** (tc + sw))
    E_lin = 10.0 ** logE_pause + np.concatenate([[0.0], cumulative_trapezoid(dEdt, t_post)])
    logE = np.log10(np.maximum(E_lin, 1e-300))

    # "Sit at TED-AI+1": once effective compute reaches sit_at_e_raw, the legal
    # project stops training AND software R&D and holds there (it sits to do
    # alignment). E, SW and TSP all freeze — so the legal's research stock stops
    # growing and the covert's S-theft source dries up from this point on.
    if sit_at_e_raw is not None:
        reached = np.where(logE >= sit_at_e_raw)[0]
        if len(reached):
            k = int(reached[0])
            logE[k:] = logE[k]
            sw[k:] = sw[k]
            tc[k:] = tc[k]

    # Research stock follows SW (incl. the windfall), so it grows during the pause
    # and the covert's S-theft draws on it. re = dRS/dt. After the legal sits, SW is
    # frozen → rs_legal flat → re_legal ≈ 0 → no further theft.
    rs_legal = RS_0 * 10.0 ** (sw / r_sw)
    re_legal = np.maximum(np.gradient(rs_legal, t_post), 1e-30)
    # tc = TSP (hardware rate, log10 FLOP/yr), sw = software efficiency (raw OOMs);
    # ETSP = tc + sw. Both returned for the ETSP decomposition.
    return t_post, logE, rs_legal, re_legal, tc, sw


def solve_legal_sw(ref: dict, data: TimeSeriesData, t_pause: float, hw_rate: float,
                   target_year: float, target_e_rebased: float, t_end: float,
                   setup_period_years: float = 0.0, setup_sw_gain_ooms: float = 0.0,
                   lo: float = 0.0, hi: float = 20.0) -> float:
    """Back-solve the legal SW rate (OOM/yr) so the legal project's *rebased*
    effective compute reaches ``target_e_rebased`` at ``target_year``.

    One monotone 1-D root-find against the analytical legal integrator — the
    turn-free replacement for the game's per-turn proportional controller.
    Clamps when even the min/max SW rate can't bracket the target. The training-R&D
    pause (``setup_period_years``/``setup_sw_gain_ooms``) is honored so the rate is
    solved against the post-pause ramp.
    """
    off = ref['rebase_offset']

    def gap(sw_rate):
        bl = build_legal(ref, data, t_pause, hw_rate, sw_rate, t_end,
                         setup_period_years, setup_sw_gain_ooms)
        t_post, logE = bl[0], bl[1]
        return float(np.interp(target_year, t_post, logE)) + off - target_e_rebased

    glo, ghi = gap(lo), gap(hi)
    if glo >= 0:    # even the floor SW rate already overshoots
        return lo
    if ghi <= 0:    # even the ceiling SW rate can't reach the target
        return hi
    return float(brentq(gap, lo, hi, xtol=1e-4))


# ---------------------------------------------------------------------------
# OPTIONAL: capped legal project (research effort bounded by AIFM physics)
# ---------------------------------------------------------------------------
# The back-solved legal above can imply research effort beyond what the project's
# resources + capability would allow (acausal). The objective is: MINIMIZE software
# progress subject to (1) the mandatory radical-transparency windfall during the
# pause and (2) reaching TED-AI+tax as close to target_year as possible, with
# effort never exceeding the AIFM ceiling.
#
# The minimal-software, on-time path is exactly the back-solve glidepath (the
# smallest steady SW ramp that hits the target at target_year). So the controller
# FORWARD-integrates while TRACKING that glidepath FROM BELOW, with per-step effort
# clamped to the ceiling (= progress_rate_at_time(...)[1], research effort for the
# project's capability + frontier resources):
#   - feasible worlds: effort needed to track <= ceiling -> reproduces back-solve
#     exactly (minimal software, no late surge, arrives at target_year);
#   - infeasible (slow-takeoff) worlds: effort clamps at the ceiling, the legal
#     falls behind the glidepath and its arrival SLIPS, but it never exceeds the
#     ceiling and never overshoots the minimal-software glidepath.

def _make_legal_data(data: TimeSeriesData, t_pause: float, hw_rate: float,
                     setup: float, t_end: float) -> TimeSeriesData:
    """The legal consortium scales its ENTIRE compute budget — inference AND
    experiment AND training — at ``hw_rate`` OOM/yr (frozen during the setup pause),
    not just training. Used to evaluate the legal's AIFM research-effort ceiling with
    its actual (growing, often > default-frontier) resources. Human labor is held at
    its at-pause level. Three knots (log-interp gives the exponential ramp)."""
    def v(c):
        return _log_interp(t_pause, data.time, getattr(data, c))
    inf0, exp0, tr0 = v("inference_H100e"), v("experiment_H100e"), v("training_H100e")
    L0 = _log_interp(t_pause, data.time, data.L_HUMAN)
    g = 10.0 ** (hw_rate * max(t_end - (t_pause + setup), 0.0))
    if setup > 0:
        t = np.array([t_pause, t_pause + setup, t_end], dtype=float)
        return TimeSeriesData(t, np.array([L0, L0, L0]), np.array([inf0, inf0, inf0 * g]),
                              np.array([exp0, exp0, exp0 * g]), np.array([tr0, tr0, tr0 * g]))
    t = np.array([t_pause, t_end], dtype=float)
    return TimeSeriesData(t, np.array([L0, L0]), np.array([inf0, inf0 * g]),
                          np.array([exp0, exp0 * g]), np.array([tr0, tr0 * g]))

def build_legal_capped(ref: dict, data: TimeSeriesData, params: Parameters, t_pause: float,
                       hw_rate: float, t_end: float, target_e_raw: float, sw_rate: float,
                       setup_period_years: float = 0.0, setup_sw_gain_ooms: float = 0.0,
                       n_fine_per_yr: int = 100):
    """Track the minimal-software back-solve glidepath (SW ramp ``sw_rate`` OOM/yr
    after the windfall) from below, clamping per-step effort at the AIFM ceiling.
    HW ramps at ``hw_rate``; the project sits once it reaches ``target_e_raw``.
    Returns ``(t, logE_raw, RS, RE, tc, sw)`` like ``build_legal``.
    """
    times_ref = ref['times']; RS_0 = ref['RS_0']; r_sw = ref['r_software']
    logE0 = float(np.interp(t_pause, times_ref, ref['progress']))
    rs0 = _log_interp(t_pause, times_ref, ref['RS'])
    sw0 = r_sw * np.log10(max(rs0, 1e-300) / RS_0)
    tc0 = _tc_ooms(data, t_pause)
    setup = max(0.0, float(setup_period_years))
    sw_resume = sw0 + setup_sw_gain_ooms
    s = 1.0 / n_fine_per_yr
    # the legal scales its whole compute budget at hw_rate -> its research-effort
    # ceiling uses these (growing) resources, not the default frontier trajectory.
    legal_data = _make_legal_data(data, t_pause, hw_rate, setup, t_end)

    def sw_glidepath(tau):  # the minimal-software target SW at elapsed time tau
        if tau < setup:
            return sw0 + setup_sw_gain_ooms * (np.clip(tau / setup, 0.0, 1.0) if setup > 0 else 1.0)
        return sw_resume + sw_rate * (tau - setup)

    ts = [t_pause]; logEs = [logE0]; RSs = [rs0]; tcs = [tc0]; sws = [sw0]
    t, logE, RS, E_lin = t_pause, logE0, rs0, 10.0 ** logE0
    reached = logE0 >= target_e_raw
    while t < t_end - 1e-9 and not reached:
        step = min(s, t_end - t)
        tau = t - t_pause
        if tau < setup - 1e-12:                      # setup: mandatory windfall, dE/dt = 0
            sw = sw0 + setup_sw_gain_ooms * (min((tau + step) / setup, 1.0) if setup > 0 else 1.0)
            tc = tc0
            RS = RS_0 * 10.0 ** (sw / r_sw)
        else:                                        # track glidepath from below, clamp at ceiling
            rs_target = RS_0 * 10.0 ** (sw_glidepath(tau + step) / r_sw)
            eff_needed = max((rs_target - RS) / step, 0.0)
            RE_ceil = max(progress_rate_at_time(
                t, [logE, max(RS, 1e-30)], legal_data, params, RS_0=RS_0)[1], 0.0)
            RS = max(RS + min(eff_needed, RE_ceil) * step, 1e-30)
            tc = tc0 + hw_rate * (tau + step - setup)
            sw = r_sw * np.log10(max(RS, 1e-300) / RS_0)
            E_lin = E_lin + 10.0 ** (tc + sw) * step
            logE = np.log10(max(E_lin, 1e-300))
        t += step
        ts.append(t); logEs.append(logE); RSs.append(RS); tcs.append(tc); sws.append(sw)
        reached = logE >= target_e_raw
    # freeze tail to t_end (legal sits: RS/SW/TC flat -> theft dries up). Insert a
    # point ONE fine step after the sit time first: the research effort re is recovered
    # below as the gradient of RS, and the covert log-interpolates it. Jumping straight
    # to t_end would leave re log-interpolated across the whole gap (a phantom, slowly
    # decaying effort the covert keeps "stealing" for years after the legal stopped);
    # the extra flat point makes the gradient — and thus theft — cliff to ~0 at the sit.
    if ts[-1] < t_end - 1e-9:
        if reached:
            ts.append(min(ts[-1] + s, t_end)); logEs.append(logEs[-1]); RSs.append(RSs[-1])
            tcs.append(tcs[-1]); sws.append(sws[-1])
        if ts[-1] < t_end - 1e-9:
            ts.append(t_end); logEs.append(logEs[-1]); RSs.append(RSs[-1])
            tcs.append(tcs[-1]); sws.append(sws[-1])
    t_post = np.array(ts); logE_a = np.array(logEs); rs_a = np.array(RSs)
    re_a = np.maximum(np.gradient(rs_a, t_post), 1e-30)
    return t_post, logE_a, rs_a, re_a, np.array(tcs), np.array(sws)


def solve_legal_capped_controls(ref: dict, data: TimeSeriesData, params: Parameters, t_pause: float,
                                max_hw_rate: float, target_year: float, target_e_rebased: float,
                                off: float, t_end: float, setup_period_years: float = 0.0,
                                setup_sw_gain_ooms: float = 0.0):
    """Choose the legal project's (hardware ramp, software rate) to arrive at the target
    as close to ``target_year`` as possible, minimizing software and not over-building
    compute. Both are bounded (hw <= max_hw_rate, sw effort <= AIFM ceiling).

    - If HW at the cap + windfall + zero software would reach the target *before*
      target_year, throttle HW down (solve hw <= max so arrival = target_year, sw=0).
    - Else HW stays at the cap and the software glidepath fills the gap.
    Returns ``(hw_rate, sw_rate)``.
    """
    target_raw = target_e_rebased - off

    def arrival_hw(hw):  # arrival with software held at the windfall floor (sw_rate=0)
        t_post, logE = build_legal_capped(ref, data, params, t_pause, hw, t_end, target_raw, 0.0,
                                          setup_period_years, setup_sw_gain_ooms)[:2]
        idx = np.where(logE >= target_raw - 1e-9)[0]
        return float(t_post[idx[0]]) if len(idx) else np.inf

    if arrival_hw(max_hw_rate) >= target_year:           # HW-max can't make the deadline alone
        sw_rate = solve_legal_sw(ref, data, t_pause, max_hw_rate, target_year, target_e_rebased,
                                 t_end, setup_period_years, setup_sw_gain_ooms)
        return max_hw_rate, sw_rate
    if arrival_hw(0.0) <= target_year:                   # windfall alone already overshoots
        return 0.0, 0.0
    hw_rate = float(brentq(lambda h: arrival_hw(h) - target_year, 0.0, max_hw_rate, xtol=2e-3))
    return hw_rate, 0.0


def build_legal_actual_hw(ref: dict, data: TimeSeriesData, params: Parameters, t_pause: float,
                          t_end: float, target_e_raw: float, sw_rate: float,
                          setup_period_years: float = 0.0, setup_sw_gain_ooms: float = 0.0,
                          post_scaling_externalities_rate: float = 0.0,
                          min_sw_frac_of_tsp: float = 0.0,
                          n_fine_per_yr: int = 100):
    """Like ``build_legal_capped`` but the legal's HARDWARE (TSP) follows the ACTUAL
    ``training_H100e`` series — ``tc(t) = _tc_ooms(data, t)`` — instead of a hw_rate ramp.
    During the setup window training is still PAUSED: TSP is frozen at the pause level and
    effective compute is flat (dE/dt = 0), with only the ``setup_sw_gain_ooms`` windfall
    accruing; after the pause the legal rejoins the real ``training_H100e`` series at the
    current calendar time (a small catch-up step at the pause boundary). The AIFM
    research-effort ceiling is taken from the REAL compute trajectory ``data``. Software
    tracks the windfall-then-``sw_rate`` glidepath from below, clamped at that ceiling, and
    the project sits at ``target_e_raw``. Returns ``(t, logE_raw, RS, RE, tc, sw, sw_floor)``
    where ``sw_floor`` is the cumulative mandatory-minimum SW guarantee (OOMs since the
    pause, = min_sw_frac_of_tsp x cumulative TSP growth accrued ONLY while scaling — it
    freezes when the track arrives, since the legal halts training then). Downstream
    decompositions consume this series rather than re-deriving the floor themselves."""
    times_ref = ref['times']; RS_0 = ref['RS_0']; r_sw = ref['r_software']
    logE0 = float(np.interp(t_pause, times_ref, ref['progress']))
    rs0 = _log_interp(t_pause, times_ref, ref['RS'])
    sw0 = r_sw * np.log10(max(rs0, 1e-300) / RS_0)
    tc0 = _tc_ooms(data, t_pause)
    setup = max(0.0, float(setup_period_years))
    tc_end = _tc_ooms(data, t_pause + setup)   # series TSP at pause-end (display ramp target)
    sw_resume = sw0 + setup_sw_gain_ooms
    s = 1.0 / n_fine_per_yr

    def sw_glidepath(tau):  # windfall over the setup window, then a flat sw_rate ramp
        if tau < setup:
            return sw0 + setup_sw_gain_ooms * (np.clip(tau / setup, 0.0, 1.0) if setup > 0 else 1.0)
        return sw_resume + sw_rate * (tau - setup)

    ts = [t_pause]; logEs = [logE0]; RSs = [rs0]; tcs = [tc0]; sws = [sw0]; floors = [0.0]
    t, logE, RS, E_lin = t_pause, logE0, rs0, 10.0 ** logE0
    reached = logE0 >= target_e_raw
    while t < t_end - 1e-9 and not reached:
        step = min(s, t_end - t)
        tau = t - t_pause
        if tau < setup - 1e-12:                      # training-R&D pause: no training, dE/dt = 0
            frac = (min((tau + step) / setup, 1.0) if setup > 0 else 1.0)
            sw = sw0 + setup_sw_gain_ooms * frac
            # TSP here is DISPLAY-ONLY (E is flat regardless of tc during the pause): ramp it
            # from the pause-date to the pause-end series value so the legal ETSP has no kink
            # at the boundary. Does not affect logE / RS / RE.
            tc = tc0 + (tc_end - tc0) * frac
            RS = RS_0 * 10.0 ** (sw / r_sw)           # only the windfall accrues; E stays flat
            floor = floors[-1]                        # no floor accrual during the setup pause
        else:                                        # post-pause: real series + tuned software
            rs_target = RS_0 * 10.0 ** (sw_glidepath(tau + step) / r_sw)
            eff_needed = max((rs_target - RS) / step, 0.0)
            RE_ceil = max(progress_rate_at_time(t, [logE, max(RS, 1e-30)], data, params, RS_0=RS_0)[1], 0.0)
            RS = max(RS + min(eff_needed, RE_ceil) * step, 1e-30)
            tc = _tc_ooms(data, t + step)            # actual training-compute series (no ramp)
            sw = r_sw * np.log10(max(RS, 1e-300) / RS_0)
            # MANDATORY MINIMUM software during scaling: SW grows at >= min_sw_frac_of_tsp x the
            # TSP (hardware) growth this step, REGARDLESS of the AIFM effort ceiling — a guaranteed
            # floor like the transparency windfall. Applies to both the used and discovered tracks.
            floor_gain = min_sw_frac_of_tsp * max(tc - tcs[-1], 0.0)
            floor = floors[-1] + floor_gain
            sw_floor = sws[-1] + floor_gain
            if sw_floor > sw:
                sw = sw_floor
                RS = max(RS_0 * 10.0 ** (sw / r_sw), 1e-30)
            E_lin = E_lin + 10.0 ** (tc + sw) * step
            logE = np.log10(max(E_lin, 1e-300))
        t += step
        ts.append(t); logEs.append(logE); RSs.append(RS); tcs.append(tc); sws.append(sw)
        floors.append(floor)
        reached = logE >= target_e_raw
    if reached and post_scaling_externalities_rate > 0.0 and ts[-1] < t_end - 1e-9:
        # POST-SCALING EXTERNALITIES: once the legal has actually reached its target, it keeps
        # DISCOVERING software efficiency at a flat rate (an unavoidable capabilities side-effect
        # of ongoing alignment research). None of this software is deployed — it only inflates
        # the DISCOVERED research stock, which the covert steals at frac_stolen exactly like the
        # normal discovered software. HARDWARE (TSP), however, keeps following the real
        # training_H100e series: the discovered ceiling benefits from the continued hardware ramp
        # even though the legal froze its own deployed run. (Applied only to the discovered
        # glidepath; the used/deployed track passes rate=0 and freezes below.)
        t_arr, sw_arr = ts[-1], sws[-1]
        E_lin = 10.0 ** logEs[-1]
        t = t_arr
        while t < t_end - 1e-9:
            step = min(s, t_end - t); t += step
            sw = sw_arr + post_scaling_externalities_rate * (t - t_arr)
            RS = RS_0 * 10.0 ** (sw / r_sw)
            tc = _tc_ooms(data, t)                      # hardware keeps following the real series
            E_lin += 10.0 ** (tc + sw) * step          # discovered EC rises from hardware + software
            ts.append(t); logEs.append(np.log10(max(E_lin, 1e-300)))
            RSs.append(RS); tcs.append(tc); sws.append(sw); floors.append(floors[-1])
    elif ts[-1] < t_end - 1e-9:
        # freeze tail to t_end (legal sits), with the same cliff-the-effort trick as capped.
        if reached:
            ts.append(min(ts[-1] + s, t_end)); logEs.append(logEs[-1]); RSs.append(RSs[-1])
            tcs.append(tcs[-1]); sws.append(sws[-1]); floors.append(floors[-1])
        if ts[-1] < t_end - 1e-9:
            ts.append(t_end); logEs.append(logEs[-1]); RSs.append(RSs[-1])
            tcs.append(tcs[-1]); sws.append(sws[-1]); floors.append(floors[-1])
    t_post = np.array(ts); logE_a = np.array(logEs); rs_a = np.array(RSs)
    re_a = np.maximum(np.gradient(rs_a, t_post), 1e-30)
    return t_post, logE_a, rs_a, re_a, np.array(tcs), np.array(sws), np.array(floors)


def solve_legal_actual_hw_sw(ref: dict, data: TimeSeriesData, params: Parameters, t_pause: float,
                             target_year: float, target_e_rebased: float, off: float, t_end: float,
                             setup_period_years: float = 0.0, setup_sw_gain_ooms: float = 0.0,
                             min_sw_frac_of_tsp: float = 0.0, hi: float = 20.0) -> float:
    """Hardware is the actual training_H100e series; tune the post-windfall software rate so
    the legal reaches the target at ``target_year``. The mandatory TSP-linked software floor
    (``min_sw_frac_of_tsp``) is included in the arrival simulation, so the solved tuned rate is
    on top of it. If the real hardware + windfall + floor alone already reaches the target by
    then, returns 0 (arrives early); if even ``hi`` can't make the deadline, returns ``hi``."""
    target_raw = target_e_rebased - off

    def arrival(sw_rate):
        t_post, logE = build_legal_actual_hw(ref, data, params, t_pause, t_end, target_raw, sw_rate,
                                             setup_period_years, setup_sw_gain_ooms,
                                             min_sw_frac_of_tsp=min_sw_frac_of_tsp)[:2]
        idx = np.where(logE >= target_raw - 1e-9)[0]
        return float(t_post[idx[0]]) if len(idx) else np.inf

    if arrival(0.0) <= target_year:      # real HW + windfall already overshoots the deadline
        return 0.0
    if arrival(hi) >= target_year:       # even max software can't reach by target_year
        return hi
    return float(brentq(lambda sw: arrival(sw) - target_year, 0.0, hi, xtol=2e-3))


# ---------------------------------------------------------------------------
# Implied R&D compute: invert the glidepath effort to a compute factor
# ---------------------------------------------------------------------------
# The glidepath throttles the legal's research effort below its physics ceiling. This
# inverts that: at each time, what common factor f on INFERENCE + EXPERIMENT compute
# (human labor held fixed) reproduces the legal's actual research effort? f=1 means the
# full real series; f<1 means the throttled effort needs only f x the real inf/exp.

def _re_at_factor(t, logE, RS, params, RS_0, inf_t, exp_t, L_t, tr_t, f):
    """Research effort dRS/dt with inference & experiment compute scaled by ``f`` (labor
    fixed), via ``progress_rate_at_time`` on a 2-point constant series at ``t`` — the same
    code path that sets the glidepath ceiling, so f=1 reproduces it."""
    fv = max(float(f), 0.0)
    tt = np.array([t - 0.5, t + 0.5], dtype=float)
    d2 = TimeSeriesData(tt, np.array([L_t, L_t]),
                        np.array([inf_t * fv, inf_t * fv]),
                        np.array([exp_t * fv, exp_t * fv]),
                        np.array([tr_t, tr_t]))
    return max(float(progress_rate_at_time(t, [logE, max(RS, 1e-30)], d2, params, RS_0=RS_0)[1]), 0.0)


def solve_implied_legal_compute(data: TimeSeriesData, params: Parameters, t_pause: float,
                                leg_t, leg_logE, leg_RS, leg_RE, RS_0: float,
                                setup: float = 0.0, f_lo: float = 1e-4, f_hi: float = 1e3):
    """For each post-setup time on the legal trajectory, solve the common factor ``f`` on
    (inference, experiment) compute — human labor held fixed — that reproduces the legal's
    research effort ``leg_RE``. Returns ``(actual_inf, actual_exp, implied_inf, implied_exp,
    factor)`` aligned to ``leg_t`` (raw H100e). Setup-window points are NaN (effort is the
    exogenous transparency windfall, not compute-driven); sit / ~zero-effort points are 0
    (no R&D compute)."""
    leg_t = np.asarray(leg_t, float)
    inf = np.array([_log_interp(t, data.time, data.inference_H100e) for t in leg_t])
    exp = np.array([_log_interp(t, data.time, data.experiment_H100e) for t in leg_t])
    Lh = np.array([_log_interp(t, data.time, data.L_HUMAN) for t in leg_t])
    trc = np.array([_log_interp(t, data.time, data.training_H100e) for t in leg_t])
    factor = np.full(leg_t.shape, np.nan)
    for i, t in enumerate(leg_t):
        if t < t_pause + setup - 1e-9:
            continue                                   # setup windfall: not compute-driven
        re = float(leg_RE[i])
        if re <= 1e-12:
            factor[i] = 0.0; continue                  # legal sits: no R&D compute
        lE, RS = float(leg_logE[i]), float(leg_RS[i])
        g = lambda ff: _re_at_factor(t, lE, RS, params, RS_0, inf[i], exp[i], Lh[i], trc[i], ff) - re  # noqa: E731,B023
        glo, ghi = g(f_lo), g(f_hi)
        if glo >= 0:                                   # human labor alone overproduces -> ~no compute
            factor[i] = 0.0
        elif ghi <= 0:                                 # even f_hi can't reach it (effort > ceiling)
            factor[i] = f_hi
        else:
            factor[i] = float(brentq(g, f_lo, f_hi, xtol=1e-4, rtol=1e-4))
    return inf, exp, inf * factor, exp * factor, factor


# ---------------------------------------------------------------------------
# Covert project
# ---------------------------------------------------------------------------

def make_covert_data(t0: float, t_end: float, total_h100e: float,
                     profile: TimeSeriesData) -> TimeSeriesData:
    """A covert datacenter: ``total_h100e`` split into inference/experiment/training
    in the ``profile`` series' mix at ``t0``, with labor scaled by the same fraction,
    held flat over ``[t0, t_end]``. (Port of ``_make_covert_data_h100e``.)
    """
    inf = _log_interp(t0, profile.time, profile.inference_H100e)
    exp = _log_interp(t0, profile.time, profile.experiment_H100e)
    tc_h100e = _log_interp(t0, profile.time, profile.training_H100e)
    total = inf + exp + tc_h100e
    frac = total_h100e / total if total > 0 else 0.0

    L = _log_interp(t0, profile.time, profile.L_HUMAN) * frac
    t = np.array([t0, t_end], dtype=float)
    return TimeSeriesData(
        t,
        np.array([L, L]),
        np.array([inf * frac, inf * frac]),
        np.array([exp * frac, exp * frac]),
        np.array([tc_h100e * frac, tc_h100e * frac]),
    )


def make_breakdown_data(t_ref: float, t0: float, t_end: float, reduction: float,
                        growth_rate: float, profile: TimeSeriesData,
                        n_per_yr: int = 20) -> TimeSeriesData:
    """The leading company's compute series after a deal breakdown: ``reduction`` × the
    ``profile`` compute mix at ``t_ref``, growing at a constant ``growth_rate`` OOM/yr from
    ``t0`` (the breakdown). ``t_ref`` encodes the breakdown_compute_mode: the pause / start
    of declarations ('frontier_at_pause' — deal-era compute is effectively destroyed) or the
    breakdown itself ('world_at_breakdown' — the leading company takes ``reduction`` of the
    compute the world has when the deal dissolves). Labor scales by the same reduction and
    stays flat — compute rebuilds; headcount doesn't follow the exponential."""
    inf = _log_interp(t_ref, profile.time, profile.inference_H100e) * reduction
    exp_ = _log_interp(t_ref, profile.time, profile.experiment_H100e) * reduction
    tc = _log_interp(t_ref, profile.time, profile.training_H100e) * reduction
    L = _log_interp(t_ref, profile.time, profile.L_HUMAN) * reduction
    t = np.linspace(t0, t_end, max(int(round((t_end - t0) * n_per_yr)) + 1, 2))
    g = 10.0 ** (growth_rate * (t - t0))
    return TimeSeriesData(t, np.full_like(t, L), inf * g, exp_ * g, tc * g)


def _freeze_track_at(tb, t_end, t, arrays, re):
    """Truncate a legal track at ``tb`` (deal breakdown — the consortium dissolves) and
    hold every series flat to ``t_end``. Research effort ``re`` is zeroed on the frozen
    tail so the covert's stealing term (frac_stolen · RE_legal/RS_legal) sees no further
    legal R&D. Returns ``(t, arrays, re)`` with all series on the truncated grid."""
    t = np.asarray(t, float)
    keep = t < tb - 1e-12
    tail_t = [tb, t_end] if tb < t_end - 1e-9 else [tb]
    new_t = np.concatenate([t[keep], tail_t])
    out = []
    for a in arrays:
        a = np.asarray(a, float)
        v = float(np.interp(tb, t, a))
        out.append(np.concatenate([a[keep], [v] * len(tail_t)]))
    re = np.asarray(re, float)
    re_out = np.concatenate([re[keep], [1e-30] * len(tail_t)])
    return new_t, out, re_out


def integrate_covert(t0, t_end, dt, logE0, RS0, data, params, RS_0,
                     leg_t, leg_RS, leg_RE, rs_at_pause,
                     frac_stolen=0.0, dormant_until=None, collect_metrics=False):
    """Forward-Euler the covert ODE from ``t0``. Same RHS (``progress_rate_at_time``)
    as the canonical model; the logE step is exact-in-linear-E. Optional S-theft
    boosts own R&D and siphons the legal project's algorithmic progress; optional
    dormancy freezes E while only stolen progress accrues. (Port of
    ``_integrate_covert``.) Returns ``(t, logE_raw, RS)``.
    """
    t, logE, RS = t0, logE0, RS0
    ts, Es, RSs = [t], [logE], [RS]
    # Software-source decomposition: cumulative natural-log growth of RS since t0,
    # attributed to three channels (cs+cb+ci = ln(RS/RS0), so r_software/ln10 times
    # each is its share of the covert's added SW in OOMs). Software efficiency is
    # logarithmic in RS, and d(ln RS)/dt = g + b·E_own/RS splits additively, so we
    # apportion each step's ln-growth by the channels' shares of dRS:
    #   cs  proportional theft       (stolen_rs_growth_rate g · RS)
    #   cb  theft-derived boost markup ((b−1)·E_own, own effort made more productive)
    #   ci  unboosted indigenous effort (E_own)
    cs, cb, ci = [0.0], [0.0], [0.0]
    while t < t_end - 1e-9:
        step = min(dt, t_end - t)
        rs_l = _log_interp(t, leg_t, leg_RS)
        re_l = _log_interp(t, leg_t, leg_RE)
        RS_prev = RS

        if dormant_until is not None and t < dormant_until:
            if frac_stolen > 0 and rs_l > 0:
                RS = max(RS + frac_stolen * re_l / rs_l * RS * step, 1e-30)
            t += step
            ts.append(t); Es.append(logE); RSs.append(RS)
            dl = np.log(RS / RS_prev) if RS_prev > 0 else 0.0  # dormancy growth is pure theft
            cs.append(cs[-1] + dl); cb.append(cb[-1]); ci.append(ci[-1])
            continue

        boost = (rs_l / rs_at_pause) ** frac_stolen if rs_at_pause > 0 else 1.0
        stolen = frac_stolen * re_l / rs_l if rs_l > 0 else 0.0
        d_logE, dRS = progress_rate_at_time(
            t, [logE, RS], data, params, RS_0=RS_0,
            boost=boost, stolen_rs_growth_rate=stolen,
        )
        logE += np.log10(1.0 + d_logE * LN10 * step)
        RS = max(RS + dRS * step, 1e-30)
        t += step
        ts.append(t); Es.append(logE); RSs.append(RS)
        # split this step's ln-growth by each channel's share of dRS (all ≥ 0):
        #   dRS = stolen·RS_prev  +  (boost−1)·E_own  +  E_own
        dl = np.log(RS / RS_prev) if RS_prev > 0 else 0.0
        d_stolen = stolen * RS_prev
        own_boosted = max(dRS - d_stolen, 0.0)               # = boost · E_own
        own_base = own_boosted / boost if boost > 0 else own_boosted
        d_markup = max(own_boosted - own_base, 0.0)          # = (boost−1) · E_own
        tot = d_stolen + d_markup + own_base
        if tot > 0 and dl != 0.0:
            cs.append(cs[-1] + dl * d_stolen / tot)
            cb.append(cb[-1] + dl * d_markup / tot)
            ci.append(ci[-1] + dl * own_base / tot)
        else:
            cs.append(cs[-1]); cb.append(cb[-1]); ci.append(ci[-1])

    ts, Es, RSs = np.array(ts), np.array(Es), np.array(RSs)
    if not collect_metrics:
        return ts, Es, RSs

    # Per-step metrics, recovered from the recorded (logE, RS) trajectory.
    # TSP = the covert datacenter's frozen FLOP/yr rate; SW = r_software·log10(RS/RS_0);
    # ETSP = TSP + SW (= the `ec` in progress_rate_at_time). Automation fraction and
    # research effort (dRS/dt) are the model's own quantities for the covert.
    tsp = np.array([_tc_ooms(data, float(tt)) for tt in ts])
    sw = params.r_software * np.log10(np.maximum(RSs, 1e-300) / RS_0)
    etsp = tsp + sw
    am = getattr(params, "automation_model", None)
    automation = (np.array([am.get_automation_fraction(float(e)) for e in Es])
                  if am is not None else np.full(len(Es), np.nan))
    re = np.gradient(RSs, ts) if len(ts) > 1 else np.zeros(len(ts))
    # software added since pause, split by source (OOMs): r_software/ln10 · cumulative ln-growth
    c_sw = params.r_software / LN10
    metrics = {"tsp": tsp, "sw": sw, "etsp": etsp, "automation": automation, "re": re,
               "sw_stolen": c_sw * np.array(cs), "sw_boost": c_sw * np.array(cb),
               "sw_indig": c_sw * np.array(ci)}
    return ts, Es, RSs, metrics


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def simulate_covert(params: Parameters, data: TimeSeriesData, t_pause: float, *,
                    alignment_tax_progress_years: float, target_year: float, max_hw_rate: float,
                    compute_mode: str, frac_stolen: float,
                    total_h100e: float | None = None, total_h100e_frac: float | None = None,
                    bottom_up_fracs: dict | None = None,
                    bottom_up_eval_offset_years: float = 0.0,
                    bottom_up_diversion_duration: float | None = None,
                    bottom_up_power: dict | None = None,
                    dormancy_years: float = 0.0, construction_years: float = 0.0,
                    setup_period_years: float = 0.0, setup_sw_gain_progress_months: float = 0.0,
                    dt: float = 0.01, t_end: float | None = None, ref: dict | None = None,
                    return_trajectories: bool = False, legal_mode: str = "backsolve",
                    legal_strategy: str = "target_year", max_sw_rate: float | None = None,
                    post_scaling_externalities_rate: float = 0.0,
                    min_sw_frac_of_tsp: float = 0.2,
                    enable_deal_breakdown: bool = False,
                    deal_breakdown_years_after_pause_end: float | None = None,
                    breakdown_compute_reduction: float | None = None,
                    breakdown_compute_mode: str = 'frontier_at_pause',
                    post_breakdown_compute_growth_rate: float = 0.0,
                    solve_legal_compute: bool = False) -> dict:
    """Simulate one covert project against a paused legal project and return a record.

    TWO legal glidepaths share the same actual-HW / pause / windfall, differing only in
    the back-solved software rate:
      - DISCOVERED: reaches TED-AI + tax at target_year — the legal's full research
        output. Its research stock is what the covert steals (the primary legal track).
      - USED:       reaches bare TED-AI at target_year — the legal's DEPLOYED capability
        (the rest of the research is spent on the alignment tax). Used for the race.
    Because each is its own proper integration back-solved to its target, discovered_EC =
    used_EC + tax holds exactly at the target (no log/integral inconsistency).

    The alignment tax and the radical-transparency windfall are specified as STOCKS in
    2025-progress units and converted to OOMs of effective compute via the per-rollout
    anchor progress rate (``ref['anchor_progress_rate']``, OOM/progress-year):
      tax_OOMs     = alignment_tax_progress_years     × anchor_rate
      windfall_OOMs = setup_sw_gain_progress_months/12 × anchor_rate
    ``ref`` may be a precomputed ``run_default`` to amortise the reference run.
    """
    if ref is None:
        ref = run_default(params, data)
    # Always use the model-PREPARED params: run_default deep-copies its input and only
    # mutates the copy (rescaled r_software, taste/automation slopes converted to
    # per-progress-unit, automation_model built), exposing it as ref['prepared_params'].
    # The caller's `params` object stays RAW, and downstream code (progress_rate_at_time,
    # build_legal_*) needs the prepared form. Normalising here makes simulate_covert correct
    # regardless of whether the caller passed raw or prepared params (and fixes the
    # ref-is-None path, which built ref from raw params then used them downstream).
    if ref.get('prepared_params') is not None:
        params = ref['prepared_params']
    ted = ref['ted_ai_e']
    if ted is None:
        return {'valid': False, 'reason': 'no_TED_AI_milestone', 't_pause': t_pause}

    t_end = float(data.time[-1]) if t_end is None else float(t_end)
    # Horizon guarantee: simulate at least 5 years past the legal's PLANNED TED-AI
    # arrival (target_year) so the would-be legal crossing and a post-arrival window
    # always exist, even when a deal breakdown truncates the realized tracks. Only
    # ever extends (never shrinks) t_end, and the input series is the hard ceiling.
    t_end = max(t_end, min(float(target_year) + 5.0, float(data.time[-1])))
    off = ref['rebase_offset']
    # tax + windfall are progress-time STOCKS -> OOMs via the per-rollout anchor rate.
    anchor_rate = float(ref['anchor_progress_rate'])
    tax_OOMs = float(alignment_tax_progress_years) * anchor_rate
    windfall_OOMs = float(setup_sw_gain_progress_months) / 12.0 * anchor_rate
    legal_target = ted + tax_OOMs                       # the DISCOVERED target (TED-AI + tax)

    # Deal breakdown (optional, gated by enable_deal_breakdown): the deal collapses
    # `deal_breakdown_years_after_pause_end` years after the setup pause ends. The
    # consortium dissolves there — both legal glidepaths freeze and covert stealing stops —
    # and the leading company continues alone as a covert-style project (no stealing:
    # there is no consortium left to steal from) sized `breakdown_compute_reduction` ×
    # a reference compute mix set by `breakdown_compute_mode` ('frontier_at_pause': the
    # frontier at the pause, i.e. deal-era compute is destroyed; 'world_at_breakdown': the
    # world's compute when the deal dissolves — nothing is destroyed and the leading
    # company runs the largest post-dissolution project on that share), growing at
    # `post_breakdown_compute_growth_rate` OOM/yr.
    t_breakdown = None
    if enable_deal_breakdown:
        if deal_breakdown_years_after_pause_end is None or breakdown_compute_reduction is None:
            raise ValueError("enable_deal_breakdown=True requires deal_breakdown_years_after_pause_end "
                             "and breakdown_compute_reduction")
        if float(deal_breakdown_years_after_pause_end) < 0:
            raise ValueError("deal_breakdown_years_after_pause_end must be >= 0 (years past the "
                             f"end of the pause), got {deal_breakdown_years_after_pause_end!r}")
        if not (0.0 < float(breakdown_compute_reduction) <= 1.0):
            raise ValueError("breakdown_compute_reduction must be in (0, 1] (fraction of the "
                             f"reference compute mix), got {breakdown_compute_reduction!r}")
        if breakdown_compute_mode not in ('frontier_at_pause', 'world_at_breakdown'):
            raise ValueError("breakdown_compute_mode must be 'frontier_at_pause' or "
                             f"'world_at_breakdown', got {breakdown_compute_mode!r}")
        t_breakdown = t_pause + setup_period_years + float(deal_breakdown_years_after_pause_end)

    # Size the covert datacenter. compute_mode is MANDATORY — there is no implicit
    # fallback, so callers must explicitly declare how the covert's compute is sized:
    #   'frontier_relative' -> total = total_h100e_frac * frontier compute at the pause
    #       (holds the covert-vs-frontier reduction constant across worlds, removing the
    #        confounder that a fixed 500k is a different reduction depending on the pause).
    #   'absolute'          -> total = total_h100e (a fixed H100e count, pause-independent).
    #   'bottom_up_estimate'-> per-source diversion over the window [declaration-duration, declaration]
    #        (declaration = pause - eval_offset): decommissioning takes a fraction of the world STOCK
    #        at the window start; smuggling + concealed-domestic take a fraction of the FLOW produced
    #        during the window. world_H100e is log-interpolated from the input series, exposed via ref.
    bottom_up_world_at_pause = None      # cumulative world stock at the declaration
    bottom_up_diverted_pool = None       # world compute produced over the window (the flow pool)
    bottom_up_stock_at_start = None      # cumulative world stock at the window start
    bottom_up_decommission_h100e = None
    bottom_up_smuggled_h100e = None
    bottom_up_domestic_h100e = None
    bottom_up_facility_mw = None         # per-source-weighted facility power (bottom_up_power)
    bottom_up_source_w = None            # per-source facility W/H100e used for the MW above
    if compute_mode == "frontier_relative":
        if total_h100e_frac is None:
            raise ValueError("compute_mode='frontier_relative' requires total_h100e_frac")
        frontier_total = sum(_log_interp(t_pause, data.time, getattr(data, c))
                             for c in ("inference_H100e", "experiment_H100e", "training_H100e"))
        total_h100e = total_h100e_frac * frontier_total
    elif compute_mode == "absolute":
        if total_h100e is None:
            raise ValueError("compute_mode='absolute' requires total_h100e")
        total_h100e_frac = None  # ignored in absolute mode; record reflects that
    elif compute_mode == "bottom_up_estimate":
        if bottom_up_fracs is None:
            raise ValueError("compute_mode='bottom_up_estimate' requires bottom_up_fracs "
                             "(from bottom_up_covert_fraction)")
        wt, w = ref.get("world_h100e_time"), ref.get("world_h100e")
        if w is None or wt is None:
            raise ValueError("compute_mode='bottom_up_estimate' requires a world_H100e column in "
                             "the input series (none found in ref)")
        # Diversion window: ENDS at the compute declaration (books close) = pause - eval_offset
        # (~Jan1->Mar1 gap), and runs back diversion_duration:  [declaration - duration, declaration].
        # Per the supplement's source structure, the bases differ:
        #   - false decommissioning  -> fraction of the world STOCK at the window START (pre-deal stock)
        #   - smuggling + concealed-domestic -> fraction of the world FLOW produced over the window
        #     (the increment in the cumulative world_H100e stock).
        # duration None/0 => no flow window (flow pool 0; decommissioning still acts on the start stock,
        # which then equals the declaration stock).
        t_declaration = t_pause - float(bottom_up_eval_offset_years)
        bottom_up_world_at_pause = _log_interp(t_declaration, wt, w)   # stock at the declaration
        if bottom_up_diversion_duration is not None and float(bottom_up_diversion_duration) > 0:
            bottom_up_stock_at_start = _log_interp(t_declaration - float(bottom_up_diversion_duration), wt, w)
        else:
            bottom_up_stock_at_start = bottom_up_world_at_pause
        bottom_up_diverted_pool = max(bottom_up_world_at_pause - bottom_up_stock_at_start, 0.0)  # flow
        f = bottom_up_fracs
        bottom_up_decommission_h100e = float(f.get("decommission_stock_frac", 0.0)) * bottom_up_stock_at_start
        bottom_up_smuggled_h100e = float(f.get("smuggled_flow_frac", 0.0)) * bottom_up_diverted_pool
        bottom_up_domestic_h100e = float(f.get("domestic_flow_frac", 0.0)) * bottom_up_diverted_pool
        total_h100e = bottom_up_decommission_h100e + bottom_up_smuggled_h100e + bottom_up_domestic_h100e
        total_h100e_frac = None  # not a frontier-relative fraction; record the world fraction instead
        # Facility power from the sampled source mix (supplement §3.3): each source has its
        # own W/H100e (decommissioned/domestic chips are older/less efficient than smuggled
        # US chips), evaluated at proxy years tied to this rollout's diversion window. When
        # bottom_up_power is absent (legacy callers), the caller falls back to the flat
        # facility_w_per_h100e knob.
        if bottom_up_power is not None:
            t_window_start = (t_declaration - float(bottom_up_diversion_duration)
                              if (bottom_up_diversion_duration is not None
                                  and float(bottom_up_diversion_duration) > 0)
                              else t_declaration)
            bottom_up_source_w = bottom_up_facility_w_per_h100e(
                bottom_up_power, t_window_start, t_declaration)
            bottom_up_facility_mw = (
                bottom_up_decommission_h100e * bottom_up_source_w["decommission_w"]
                + bottom_up_smuggled_h100e * bottom_up_source_w["smuggled_w"]
                + bottom_up_domestic_h100e * bottom_up_source_w["domestic_w"]) / 1e6
    else:
        raise ValueError(
            "compute_mode must be 'frontier_relative', 'absolute', or 'bottom_up_estimate', "
            f"got {compute_mode!r}")

    # Legal scaling strategy:
    #  - 'target_year' (default): tune HW + SW to reach the target at target_year.
    #  - 'target_year_capped_sw': identical, but cap the TUNED software rate at a flat
    #    max_sw_rate OOM/yr. The mandatory setup windfall (setup_sw_gain_ooms) is exempt;
    #    only the post-windfall tuned scaling is capped. max_sw_rate=0 => the legal makes
    #    no tuned software progress, so the covert can steal only the windfall.
    #  - 'target_year_actual_hw': hardware (TSP) follows the ACTUAL training_H100e series
    #    instead of a chosen ramp (hw_rate / max_hw_rate ignored); software is the only free
    #    control, tuned to reach the target at target_year (effort clamped at the AIFM ceiling
    #    computed from the real compute trajectory). Forward-integrated, so legal_mode is moot.
    if legal_strategy not in ("target_year", "target_year_capped_sw", "target_year_actual_hw"):
        raise ValueError("legal_strategy must be 'target_year', 'target_year_capped_sw', or "
                         f"'target_year_actual_hw', got {legal_strategy!r}")
    if legal_strategy == "target_year_capped_sw":
        if max_sw_rate is None:
            raise ValueError("legal_strategy='target_year_capped_sw' requires max_sw_rate")
        if max_sw_rate < 0:
            raise ValueError(f"max_sw_rate must be >= 0, got {max_sw_rate!r}")
    # SW-rate ceiling for the solve: the flat cap when capped_sw, else the default bound.
    sw_hi = float(max_sw_rate) if legal_strategy == "target_year_capped_sw" else 20.0
    if post_scaling_externalities_rate and post_scaling_externalities_rate > 0.0 \
            and legal_strategy != "target_year_actual_hw":
        raise NotImplementedError(
            "post_scaling_externalities_rate is only implemented for legal_strategy="
            f"'target_year_actual_hw' (the discovered glidepath); got {legal_strategy!r}.")

    # 1. legal project. Run the chosen strategy to a given rebased-EC target. legal_mode:
    #  - 'backsolve': exogenous SW rate back-solved to hit the target at target_year.
    #  - 'capped': forward-integrate effort clamped at the AIFM ceiling; arrival slips if
    #    the cap binds. (target_year_actual_hw forward-integrates either way.)
    def _build_legal_to(target_rebased, tyear, ext_rate=0.0):
        """Build the legal trajectory under the chosen strategy, reaching ``target_rebased``
        (rebased EC) at ``tyear``. The windfall enters during the setup pause. ``ext_rate``
        (post_scaling_externalities_rate) only applies to the DISCOVERED actual-hw glidepath.
        Returns (t, logE_raw, RS, RE, tsp, sw, sw_floor, sw_rate, hw_rate_used); sw_floor
        is the cumulative mandatory-minimum guarantee (zeros for the non-actual_hw
        strategies, which have no floor)."""
        tgt_raw = target_rebased - off
        if legal_strategy == "target_year_actual_hw":
            swr = solve_legal_actual_hw_sw(ref, data, params, t_pause, tyear, target_rebased,
                                           off, t_end, setup_period_years, windfall_OOMs,
                                           min_sw_frac_of_tsp=min_sw_frac_of_tsp)
            out = build_legal_actual_hw(ref, data, params, t_pause, t_end, tgt_raw, swr,
                                        setup_period_years, windfall_OOMs,
                                        post_scaling_externalities_rate=ext_rate,
                                        min_sw_frac_of_tsp=min_sw_frac_of_tsp)
            return (*out, swr, None)   # HW exogenous (real series), not a chosen ramp
        if legal_mode == "capped":
            hwr, swr = solve_legal_capped_controls(
                ref, data, params, t_pause, max_hw_rate, tyear, target_rebased, off, t_end,
                setup_period_years, windfall_OOMs)
            if legal_strategy == "target_year_capped_sw":   # obey the flat tuned-SW ceiling
                swr = min(swr, sw_hi)
            out = build_legal_capped(ref, data, params, t_pause, hwr, t_end, tgt_raw, swr,
                                     setup_period_years, windfall_OOMs)
            return (*out, np.zeros_like(out[5]), swr, hwr)
        swr = solve_legal_sw(ref, data, t_pause, max_hw_rate, tyear, target_rebased, t_end,
                             setup_period_years, windfall_OOMs, hi=sw_hi)
        out = build_legal(ref, data, t_pause, max_hw_rate, swr, t_end, setup_period_years,
                          windfall_OOMs, sit_at_e_raw=tgt_raw)
        return (*out, np.zeros_like(out[5]), swr, max_hw_rate)

    # DISCOVERED glidepath (reaches TED-AI + tax at target_year) — the legal's full research
    # output; the covert steals its research stock. This is the primary legal track.
    (leg_t, leg_logE, leg_RS, leg_RE, leg_tsp, leg_sw, leg_sw_floor,
     sw_rate, hw_rate_used) = _build_legal_to(
        legal_target, target_year, ext_rate=post_scaling_externalities_rate)
    legal_E = leg_logE + off          # discovered EC (reaches TED-AI + tax)
    # The date discovered ACTUALLY reaches TED-AI + tax — == target_year unless the cap binds
    # and it slips later. The USED glidepath targets bare TED-AI at THAT date (not target_year),
    # so used reaches TED-AI exactly when discovered finishes the tax: this preserves the
    # invariant that tax = discovered_EC − used_EC at the moment used reaches TED-AI, and
    # guarantees used never reaches TED-AI before discovered reaches TED-AI + tax.
    _disc_ttt = _crossing_time(leg_t, legal_E, legal_target)
    used_tyear = _disc_ttt if _disc_ttt is not None else target_year
    # USED glidepath (reaches bare TED-AI at used_tyear) — the legal's deployed capability.
    u_t, u_logE, u_RS, u_RE, u_tsp, u_sw, u_sw_floor, used_sw_rate, _ = _build_legal_to(ted, used_tyear)
    # Counterfactual (deal-holds) bookkeeping, captured BEFORE any breakdown truncation:
    # when the legal glidepaths WOULD have crossed their targets had the deal held. Equal
    # to the realized crossings when no breakdown occurs within the horizon. `would_tracks`
    # keeps the unfrozen trajectories for the record (only stored when a breakdown
    # actually truncated them — otherwise they'd duplicate legal_t/legal_e).
    would_disc_ttt = _disc_ttt
    would_used_ttt = _crossing_time(u_t, u_logE + off, ted)
    would_legal_ttt = (max(would_used_ttt, would_disc_ttt)
                       if (would_used_ttt is not None and would_disc_ttt is not None) else None)
    would_tracks = None
    # Deal breakdown: freeze BOTH consortium glidepaths at t_breakdown (flat to t_end,
    # research effort zeroed → the covert's stealing dries up), and recompute the crossings
    # from the frozen tracks — a consortium that hadn't arrived by the breakdown never does.
    # The glidepaths are still SOLVED for the original targets (the consortium planned for
    # the deal to hold); the breakdown just truncates the realized paths.
    if t_breakdown is not None and t_breakdown < t_end - 1e-9:
        would_tracks = {'legal_t': leg_t, 'legal_e': leg_logE + off,
                        'used_t': u_t, 'used_e': u_logE + off}
        leg_t, (leg_logE, leg_RS, leg_tsp, leg_sw, leg_sw_floor), leg_RE = _freeze_track_at(
            t_breakdown, t_end, leg_t, [leg_logE, leg_RS, leg_tsp, leg_sw, leg_sw_floor], leg_RE)
        u_t, (u_logE, u_RS, u_tsp, u_sw, u_sw_floor), u_RE = _freeze_track_at(
            t_breakdown, t_end, u_t, [u_logE, u_RS, u_tsp, u_sw, u_sw_floor], u_RE)
        legal_E = leg_logE + off
        _disc_ttt = _crossing_time(leg_t, legal_E, legal_target)
    legal_theta = None
    used_E = u_logE + off             # used/deployed EC (reaches bare TED-AI at used_tyear)
    _used_ttt = _crossing_time(u_t, used_E, ted)
    # Legal arrives when both hold; by construction they coincide at _disc_ttt (max guards
    # the degenerate case where used itself clamps and slips past _disc_ttt).
    legal_ttt = (max(_used_ttt, _disc_ttt) if (_used_ttt is not None and _disc_ttt is not None)
                 else None)

    # 2. covert project: integrate from the at-pause state (raw coords)
    logE_pause = float(np.interp(t_pause, ref['times'], ref['progress']))
    rs_at_pause = _log_interp(t_pause, ref['times'], ref['RS'])
    cparams = copy.copy(params)
    cdata = make_covert_data(t_pause, t_end, total_h100e, data)
    # Two sequential inactive phases before the covert runs openly: DORMANCY (lying low —
    # no facility exists yet, so the detection clock hasn't started) followed by
    # CONSTRUCTION (the build window — detectable). For the covert's training physics
    # they are equivalent: E frozen, only stolen software accrues. Detection treats them
    # differently (see pipeline/mc.py: the facility-frailty clock starts at the end of
    # dormancy; P(detected during dormancy) = 0).
    inactive_years = float(dormancy_years) + float(construction_years)
    dormant_until = (t_pause + inactive_years) if inactive_years > 0 else None

    cov_out = integrate_covert(
        t_pause, t_end, dt, logE_pause, rs_at_pause, cdata, cparams, ref['RS_0'],
        leg_t, leg_RS, leg_RE, rs_at_pause, frac_stolen, dormant_until,
        collect_metrics=return_trajectories,
    )
    if return_trajectories:
        cov_t, cov_logE, cov_RS, cov_metrics = cov_out
    else:
        cov_t, cov_logE, cov_RS = cov_out
        cov_metrics = None
    covert_E = cov_logE + off
    covert_ttt = _crossing_time(cov_t, covert_E, ted)   # covert pays no tax

    # 3. leading company after the deal breakdown: covert-style physics with NO stealing
    # (the consortium it would steal from is gone). Capability (E) starts from the USED /
    # deployed track — that's what it actually runs; research stock starts from the
    # DISCOVERED track — the deploy restrictions holding that research back die with the
    # deal. Compute: `breakdown_compute_reduction` × the reference mix (frontier at pause,
    # or — 'world_at_breakdown' — the world's compute at the collapse, which the input
    # series carries through the deal), growing at `post_breakdown_compute_growth_rate`
    # OOM/yr.
    lead_t = lead_logE = lead_RS = lead_metrics = None
    leading_ttt = None
    if t_breakdown is not None and t_breakdown < t_end - 1e-9:
        lead_logE0 = float(np.interp(t_breakdown, u_t, u_logE))
        lead_RS0 = _log_interp(t_breakdown, leg_t, leg_RS)
        bd_ref_t = float(t_breakdown) if breakdown_compute_mode == 'world_at_breakdown' else t_pause
        lead_data = make_breakdown_data(bd_ref_t, t_breakdown, t_end,
                                        float(breakdown_compute_reduction),
                                        float(post_breakdown_compute_growth_rate), data)
        lead_out = integrate_covert(
            t_breakdown, t_end, dt, lead_logE0, lead_RS0, lead_data, cparams, ref['RS_0'],
            leg_t, leg_RS, leg_RE, rs_at_pause, 0.0, None,
            collect_metrics=return_trajectories)
        if return_trajectories:
            lead_t, lead_logE, lead_RS, lead_metrics = lead_out
        else:
            lead_t, lead_logE, lead_RS = lead_out
        leading_ttt = _crossing_time(lead_t, lead_logE + off, ted)  # no consortium -> no tax

    covert_wins = covert_ttt is not None and (legal_ttt is None or covert_ttt < legal_ttt)
    lead_years = (legal_ttt - covert_ttt) if (covert_ttt is not None and legal_ttt is not None) else None
    ac_t, ted_t = ref['ac_time'], ref['ted_ai_time']
    # Slowdown vs the un-paused default world: both share one trajectory until the pause
    # (where the covert project splits off), so compare times from that split to TED-AI.
    covert_slowdown = ((covert_ttt - t_pause) / (ted_t - t_pause)
                       if (covert_ttt is not None and ted_t is not None
                           and ted_t - t_pause > 0) else None)

    # Realized average SW rate over the scaling window (end of setup -> that track's arrival,
    # or the last simulated point if it never arrives), EXCLUDING the setup windfall. Unlike
    # sw_rate_solved — the tuned rate ON TOP of the mandatory min_sw_frac_of_tsp floor — this
    # is the slope of the actual sw path, so it includes the floor wherever it binds (and any
    # effort-ceiling clamping). Evaluated AT arrival so post-scaling externalities (which keep
    # inflating discovered sw afterwards) don't leak in.
    def _realized_sw_rate(track_t, track_sw, arrival):
        t0 = t_pause + setup_period_years
        t1 = float(arrival) if arrival is not None else float(track_t[-1])
        if t1 - t0 <= 1e-9:            # arrived during/before setup: no scaling-phase software
            return 0.0
        s0 = float(np.interp(t0, track_t, track_sw))
        s1 = float(np.interp(t1, track_t, track_sw))
        return max((s1 - s0) / (t1 - t0), 0.0)

    # Under a deal breakdown a track that never arrived froze at t_breakdown — measure its
    # realized slope over the alive window, not diluted by the frozen tail.
    _disc_end = _disc_ttt if _disc_ttt is not None else t_breakdown
    _used_end = _used_ttt if _used_ttt is not None else t_breakdown
    sw_rate_incl_floor = _realized_sw_rate(leg_t, leg_sw, _disc_end)
    used_sw_rate_incl_floor = _realized_sw_rate(u_t, u_sw, _used_end)
    # Average rate of the mandatory-minimum guarantee alone over the same window (0 when the
    # strategy has no floor — sw_floor is all-zeros there).
    sw_floor_rate_avg = _realized_sw_rate(leg_t, leg_sw_floor, _disc_end)

    rec = {
        'valid': True,
        't_pause': t_pause,
        'setup_period_years': setup_period_years,
        'setup_sw_gain_progress_months': setup_sw_gain_progress_months,
        'setup_sw_gain_ooms': windfall_OOMs,           # converted via anchor rate
        'alignment_tax_progress_years': alignment_tax_progress_years,
        'alignment_tax_ooms': tax_OOMs,                # converted via anchor rate
        'anchor_progress_rate': anchor_rate,
        'legal_mode': legal_mode,
        'legal_strategy': legal_strategy,
        'max_sw_rate': (float(max_sw_rate) if max_sw_rate is not None else None),
        'sw_rate_solved': sw_rate,                     # discovered glidepath rate
        'used_sw_rate_solved': used_sw_rate,           # used (bare-TED-AI) glidepath rate
        # realized scaling-window SW rates (tuned rate + mandatory floor + clamps; see above)
        'sw_rate_incl_floor': float(sw_rate_incl_floor),
        'used_sw_rate_incl_floor': float(used_sw_rate_incl_floor),
        'sw_floor_rate_avg': float(sw_floor_rate_avg),
        'legal_theta_solved': legal_theta,
        'legal_hw_rate_solved': (float(hw_rate_used) if hw_rate_used is not None else None),
        'compute_mode': compute_mode,
        'total_h100e_used': float(total_h100e),
        'total_h100e_frac': total_h100e_frac,
        # bottom_up_estimate sizing (None for other compute modes). Per-source amounts are the
        # absolute H100e; the *_frac fields express each as a fraction of the world STOCK at the
        # declaration (W_end), so they remain comparable "% of world" numbers that sum to
        # bottom_up_world_frac, even though decommissioning is actually sized off the start stock and
        # smuggling/domestic off the window flow.
        #  - bottom_up_world_h100e_at_pause: cumulative world stock at the declaration (W_end).
        #  - bottom_up_stock_at_start      : cumulative world stock at the window start.
        #  - bottom_up_diverted_pool       : world compute PRODUCED over the window (flow = W_end - W_start).
        'bottom_up_world_frac': ((float(total_h100e) / bottom_up_world_at_pause)
                                 if (compute_mode == "bottom_up_estimate"
                                     and bottom_up_world_at_pause) else None),
        'bottom_up_decommission_frac': ((bottom_up_decommission_h100e / bottom_up_world_at_pause)
                                        if (bottom_up_decommission_h100e is not None
                                            and bottom_up_world_at_pause) else None),
        'bottom_up_smuggled_frac': ((bottom_up_smuggled_h100e / bottom_up_world_at_pause)
                                    if (bottom_up_smuggled_h100e is not None
                                        and bottom_up_world_at_pause) else None),
        'bottom_up_domestic_frac': ((bottom_up_domestic_h100e / bottom_up_world_at_pause)
                                    if (bottom_up_domestic_h100e is not None
                                        and bottom_up_world_at_pause) else None),
        'bottom_up_decommission_h100e': (float(bottom_up_decommission_h100e)
                                         if bottom_up_decommission_h100e is not None else None),
        'bottom_up_smuggled_h100e': (float(bottom_up_smuggled_h100e)
                                     if bottom_up_smuggled_h100e is not None else None),
        'bottom_up_domestic_h100e': (float(bottom_up_domestic_h100e)
                                     if bottom_up_domestic_h100e is not None else None),
        'bottom_up_world_h100e_at_pause': (float(bottom_up_world_at_pause)
                                           if bottom_up_world_at_pause is not None else None),
        'bottom_up_stock_at_start': (float(bottom_up_stock_at_start)
                                     if bottom_up_stock_at_start is not None else None),
        'bottom_up_diverted_pool': (float(bottom_up_diverted_pool)
                                    if bottom_up_diverted_pool is not None else None),
        # Facility power from the sampled source mix (None unless bottom_up_power given):
        # facility_mw = sum_i h100e_i * w_i / 1e6; *_w are the per-source facility W/H100e
        # (PUE included) evaluated at this rollout's diversion-window proxy years; the
        # effective W/H100e is the mix-weighted average the detection model implies.
        'bottom_up_facility_mw': (float(bottom_up_facility_mw)
                                  if bottom_up_facility_mw is not None else None),
        'bottom_up_w_per_h100e': ((float(bottom_up_facility_mw) * 1e6 / float(total_h100e))
                                  if (bottom_up_facility_mw is not None and total_h100e)
                                  else None),
        'bottom_up_smuggled_w': (float(bottom_up_source_w["smuggled_w"])
                                 if bottom_up_source_w is not None else None),
        'bottom_up_decommission_w': (float(bottom_up_source_w["decommission_w"])
                                     if bottom_up_source_w is not None else None),
        'bottom_up_domestic_w': (float(bottom_up_source_w["domestic_w"])
                                 if bottom_up_source_w is not None else None),
        # diversion-window timing (calendar): declaration = t_pause - eval_offset; the window is
        # [declaration - diversion_duration, declaration].
        'bottom_up_eval_offset_years': (float(bottom_up_eval_offset_years)
                                        if compute_mode == "bottom_up_estimate" else None),
        'bottom_up_diversion_duration': (float(bottom_up_diversion_duration)
                                         if (compute_mode == "bottom_up_estimate"
                                             and bottom_up_diversion_duration is not None) else None),
        'ted_ai_e': ted,
        'legal_target_e': legal_target,                # discovered target = TED-AI + tax
        'legal_used_target_e': ted,                    # used target = bare TED-AI
        'covert_ttt': covert_ttt,
        'legal_ttt': legal_ttt,
        # When legal USED capability first reaches bare TED-AI (deployed-capability crossing).
        # Usually == _disc_ttt == target_year; diverges only if the discovered glidepath clamps.
        # Anchors the detection silver-bullet sigmoid; None when used never reaches TED-AI.
        'legal_used_ttt': _used_ttt,
        # When legal DISCOVERED capability first reaches TED-AI + tax (full-research crossing,
        # the higher target). Usually == _used_ttt == target_year; later than used only if the
        # (steeper) discovered glidepath clamps. None when it never reaches the target.
        'legal_discovered_ttt': _disc_ttt,
        # Counterfactual (deal-holds) crossings: when each legal glidepath WOULD have
        # reached its target had no breakdown truncated it. Equal to the realized
        # crossings when the deal never breaks down within the horizon; None only when
        # the unfrozen track never crosses by t_end (e.g. the effort cap binds hard).
        'legal_would_ttt': would_legal_ttt,
        'legal_used_would_ttt': would_used_ttt,
        'legal_discovered_would_ttt': would_disc_ttt,
        'covert_wins': bool(covert_wins),
        'lead_years': lead_years,
        # covert time-to-TED-AI from the pause split, as a ratio to the default world's
        'covert_slowdown': (float(covert_slowdown) if covert_slowdown is not None else None),
        # deal breakdown (None/False when the gate is off)
        'deal_breakdown_enabled': bool(enable_deal_breakdown),
        'deal_breakdown_t': (float(t_breakdown) if t_breakdown is not None else None),
        'breakdown_compute_mode': (str(breakdown_compute_mode) if enable_deal_breakdown else None),
        'leading_ttt': (float(leading_ttt) if leading_ttt is not None else None),
        'covert_e_end': float(covert_E[-1]),
        'legal_e_end': float(legal_E[-1]),
        'used_e_end': float(used_E[-1]),
        # conditioning variables (for N-style conditional analyses)
        'ac_time': ac_t,
        'ted_ai_time': ted_t,
        'ac_to_ted_gap': (ted_t - ac_t) if (ac_t is not None and ted_t is not None) else None,
    }
    if return_trajectories:
        rec['covert_t'] = cov_t.tolist()
        rec['covert_e'] = covert_E.tolist()
        rec['legal_t'] = leg_t.tolist()
        rec['legal_e'] = legal_E.tolist()         # discovered EC (TED-AI + tax)
        # ETSP decomposition (rates, OOMs): ETSP = TSP (hardware) + SW (software).
        rec['legal_tsp'] = leg_tsp.tolist()
        rec['legal_sw'] = leg_sw.tolist()
        # used (deployed-capability) glidepath — reaches bare TED-AI
        rec['legal_used_t'] = u_t.tolist()
        rec['legal_used_e'] = used_E.tolist()
        rec['legal_used_sw'] = u_sw.tolist()
        # Mandatory-minimum SW guarantee (cumulative OOMs since pause; accrues only while
        # that track is scaling, frozen at arrival). The SIM owns this phase knowledge —
        # decompositions downstream consume these instead of re-deriving the floor.
        rec['legal_sw_floor'] = leg_sw_floor.tolist()
        rec['legal_used_sw_floor'] = u_sw_floor.tolist()
        rec['covert_tsp'] = cov_metrics['tsp'].tolist()
        rec['covert_sw'] = cov_metrics['sw'].tolist()
        # Raw research-stock / research-effort trajectories (needed to re-run the canonical
        # metrics pass per project via project_metrics, and to compare vs the ad-hoc series).
        rec['covert_rs'] = cov_RS.tolist()
        rec['covert_logE_raw'] = cov_logE.tolist()
        rec['legal_rs'] = leg_RS.tolist()
        rec['legal_re'] = leg_RE.tolist()
        rec['legal_logE_raw'] = leg_logE.tolist()
        rec['legal_used_rs'] = u_RS.tolist()
        rec['legal_used_re'] = u_RE.tolist()
        rec['legal_used_tsp'] = u_tsp.tolist()
        rec['legal_used_logE_raw'] = u_logE.tolist()
        # software added since pause, decomposed by source (OOMs; sum = sw − sw_at_pause)
        rec['covert_sw_stolen'] = cov_metrics['sw_stolen'].tolist()
        rec['covert_sw_boost'] = cov_metrics['sw_boost'].tolist()
        rec['covert_sw_indig'] = cov_metrics['sw_indig'].tolist()
        rec['covert_etsp'] = cov_metrics['etsp'].tolist()
        rec['covert_automation'] = cov_metrics['automation'].tolist()
        rec['covert_re'] = cov_metrics['re'].tolist()
        if would_tracks is not None:
            # Counterfactual (deal-holds) legal glidepaths, unfrozen — only present when a
            # breakdown actually truncated the realized tracks (otherwise identical to
            # legal_t/legal_e and legal_used_t/legal_used_e).
            rec['legal_would_t'] = would_tracks['legal_t'].tolist()
            rec['legal_would_e'] = would_tracks['legal_e'].tolist()
            rec['legal_used_would_t'] = would_tracks['used_t'].tolist()
            rec['legal_used_would_e'] = would_tracks['used_e'].tolist()
        if lead_t is not None:
            # leading company post-breakdown (same shape as the covert's series)
            rec['leading_t'] = lead_t.tolist()
            rec['leading_e'] = (lead_logE + off).tolist()
            rec['leading_logE_raw'] = lead_logE.tolist()
            rec['leading_rs'] = lead_RS.tolist()
            rec['leading_tsp'] = lead_metrics['tsp'].tolist()
            rec['leading_sw'] = lead_metrics['sw'].tolist()
            rec['leading_etsp'] = lead_metrics['etsp'].tolist()
            rec['leading_automation'] = lead_metrics['automation'].tolist()
            rec['leading_re'] = lead_metrics['re'].tolist()
        if solve_legal_compute:
            # implied inference+experiment compute (common factor f, labor fixed) that
            # reproduces the legal's research-effort glidepath. NaN in setup, 0 once it sits.
            # Done for BOTH glidepaths: DISCOVERED (full research output, incl. the tax
            # research; covert steals it) and USED (deployed-capability research only).
            a_inf, a_exp, i_inf, i_exp, fac = solve_implied_legal_compute(
                data, params, t_pause, leg_t, leg_logE, leg_RS, leg_RE, ref['RS_0'],
                setup=setup_period_years)
            rec['legal_actual_inf'] = a_inf.tolist()
            rec['legal_actual_exp'] = a_exp.tolist()
            rec['legal_implied_inf'] = i_inf.tolist()
            rec['legal_implied_exp'] = i_exp.tolist()
            rec['legal_compute_factor'] = fac.tolist()
            _, _, ui_inf, ui_exp, ufac = solve_implied_legal_compute(
                data, params, t_pause, u_t, u_logE, u_RS, u_RE, ref['RS_0'],
                setup=setup_period_years)
            rec['legal_used_implied_inf'] = ui_inf.tolist()
            rec['legal_used_implied_exp'] = ui_exp.tolist()
            rec['legal_used_compute_factor'] = ufac.tolist()
    return rec
