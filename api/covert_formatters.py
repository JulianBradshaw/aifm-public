"""
Response formatting for the covert-race explorer (/api/plan-a).

Mirrors response_formatters.py (the `_maybe_float` idiom, camelCase keys) but for the
covert experiment: turns a `sim.simulate_covert` record + a detection survival
decomposition into a JSON payload the CovertExplorerClient renders directly.

The three trajectory families (covert, legal DISCOVERED, legal USED) each have their
OWN time grid, and the detection curve is on a years-after-pause grid, so we return
parallel arrays per series rather than one unified per-point list. The frontend merges
them onto a common x-axis (lines skip nulls).
"""
from typing import Any, Dict, List, Optional, cast
from importlib import import_module

np = cast(Any, import_module('numpy'))


def _maybe_float(value: Any) -> Optional[float]:
    """Convert value to float if finite, otherwise return None (mirrors response_formatters)."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


_MAX_PLOT_POINTS = 250


def _ds_idx(n: int, m: int = _MAX_PLOT_POINTS):
    """Indices subsampling length n to <= m points (endpoints kept). Trajectories are
    for plotting only — crossing scalars come from the full-resolution sim — so this
    never affects computed outcomes (mirrors pipeline/mc.py _ds_idx)."""
    if n <= 1:
        return np.arange(n)
    return np.unique(np.linspace(0, n - 1, min(m, n)).astype(int))


def _series(*arrays: Any) -> List[List[Optional[float]]]:
    """Zip equal-length arrays into a list of [v0, v1, ...] rows of finite-or-None floats,
    downsampled to <= _MAX_PLOT_POINTS for lean payloads."""
    cols = [np.asarray(a, dtype=float) for a in arrays]
    n = min((c.size for c in cols), default=0)
    idx = _ds_idx(n)
    return [[_maybe_float(c[i]) for c in cols] for i in idx]


def build_covert_payload(rec: Dict[str, Any], ref: Dict[str, Any],
                         detection: Dict[str, Any], t_pause: float) -> Dict[str, Any]:
    """Build the /api/plan-a JSON payload from a simulate_covert record (return_trajectories=True),
    the reference run, and a detection survival decomposition.

    `detection` is the dict assembled in the endpoint:
        {'t': tg (years after pause), 'pDetected', 'sbCdf', 'sbSurvival',
         'frailtySurvival', 'combinedSurvival', 'spec', 'sizeMw', 'nSites', 'effSiteMw'}
    """
    # The model REBASES software efficiency + effective compute by se_at_ref so present-day
    # SE = 0 (progress_model/_impl.py:1788). `effective_compute` (and hence the covert/legal
    # E we report) already carries this shift via rebase_offset. But the sim's raw SW/ETSP
    # arrays (covert_sw, legal_sw, covert_etsp = r_software·log10(RS/RS_0) + TSP) are NOT
    # rebased, so they'd sit ~se_at_ref OOMs above the default's rebased software_efficiency
    # in the overlay. Add rebase_offset (= −se_at_ref) so SW/ETSP share the model's base.
    # TSP is not rebased in the model either, so it stays raw (a real project-size difference).
    off = float(ref.get('rebase_offset', 0.0))

    def _reb(arr):
        return (np.asarray(arr, dtype=float) + off)

    # --- Panel A: effective compute vs time (rows of [t, e]) -------------------------
    # `leading` = the leading company post-deal-breakdown (present only when the deal
    # breakdown is enabled and lands within the horizon).
    has_leading = rec.get('leading_t') is not None
    capability = {
        'covert': _series(rec['covert_t'], rec['covert_e']),
        'legalDiscovered': _series(rec['legal_t'], rec['legal_e']),
        'legalUsed': _series(rec['legal_used_t'], rec['legal_used_e']),
        **({'leading': _series(rec['leading_t'], rec['leading_e'])} if has_leading else {}),
    }

    # --- Panel B: ETSP decomposition (rows of [t, tsp, sw]) — SW rebased to the model base.
    etsp = {
        'covert': _series(rec['covert_t'], rec['covert_tsp'], _reb(rec['covert_sw'])),
        'legalDiscovered': _series(rec['legal_t'], rec['legal_tsp'], _reb(rec['legal_sw'])),
        # used glidepath: only SW exported by the sim (shares discovered's HW/TSP)
        'legalUsed': _series(rec['legal_used_t'], _reb(rec['legal_used_sw'])),
        **({'leading': _series(rec['leading_t'], rec['leading_tsp'], _reb(rec['leading_sw']))}
           if has_leading else {}),
    }

    # --- Stacked-area decompositions of cumulative OOMs added SINCE THE PAUSE (mirrors
    #     pipeline/figures.py oom_cumulative / oom_cumulative_tax). Deltas-since-pause, so
    #     the SW rebase offset cancels.
    def _cumdelta(arr):
        a = np.asarray(arr, dtype=float)
        return a - (a[0] if a.size else 0.0)

    # Covert: hardware (ΔTSP — ~0, frozen datacenter), indigenous (own R&D = boost + indig,
    # already cumulative-since-pause), stolen (= frac × legal SW gain, cumulative).
    covert_breakdown = _series(
        rec['covert_t'], _cumdelta(rec['covert_tsp']),
        np.asarray(rec['covert_sw_indig'], float) + np.asarray(rec['covert_sw_boost'], float),
        np.asarray(rec['covert_sw_stolen'], float))  # [t, hardware, indigenous, stolen]

    # Legal (discovered): hardware (ΔTSP), transparency windfall (one-time, over the setup
    # window, capped at the used-SW gain), used/deployed software (used gain − windfall), and
    # alignment tax (discovered − used). discovered = leg_sw, used = leg_used_sw.
    leg_t = np.asarray(rec['legal_t'], float)
    hw_l = _cumdelta(rec['legal_tsp'])
    disc_cum = _cumdelta(rec['legal_sw'])
    used_cum = np.interp(leg_t, np.asarray(rec['legal_used_t'], float),
                         _cumdelta(rec['legal_used_sw']))
    setup = float(rec.get('setup_period_years') or 0.0) or 1e-9
    wf_total = float(rec.get('setup_sw_gain_ooms') or 0.0)
    t0 = float(leg_t[0]) if leg_t.size else 0.0
    wf_cum = np.minimum(wf_total * np.clip((leg_t - t0) / setup, 0.0, 1.0), np.maximum(used_cum, 0.0))
    used_tuned = np.maximum(used_cum - wf_cum, 0.0)
    tax_cum = np.maximum(disc_cum - used_cum, 0.0)
    # Mandatory-minimum SW floor (>= frac x TSP growth while scaling, exempt from the AIFM
    # ceiling like the windfall): the sim owns the phase knowledge and emits the cumulative
    # guarantee directly (legal_used_sw_floor, frozen once the track arrives) — consume it
    # rather than re-deriving the floor here. It is a guaranteed subset of the tuned/used
    # software, so split it out of used_tuned (the min() is defensive: the sim guarantees
    # used SW gain >= floor during scaling, up to grid interpolation).
    floor_used = rec.get('legal_used_sw_floor')
    if floor_used is not None:
        floor_cum = np.interp(leg_t, np.asarray(rec['legal_used_t'], float),
                              np.maximum(np.asarray(floor_used, float), 0.0))
    else:
        floor_cum = np.zeros_like(used_tuned)
    mand_min = np.minimum(used_tuned, floor_cum)          # guaranteed minimum, capped at realized used
    used_tuned = np.maximum(used_tuned - mand_min, 0.0)   # tuned software above the floor
    legal_breakdown = _series(rec['legal_t'], hw_l, wf_cum, mand_min, used_tuned,
                              tax_cum)  # [t, hw, windfall, mandatoryMin, used, tax]

    # --- Panel C: detection (rows of [t, pDetected, sbCdf, sbSurvival, frailtySurvival, combinedSurvival])
    det_points = _series(detection['t'], detection['pDetected'], detection['sbCdf'],
                         detection['sbSurvival'], detection['frailtySurvival'],
                         detection['combinedSurvival'])
    spec = detection.get('spec')
    detection_out = {
        'points': det_points,
        'silverBullet': ({'C': _maybe_float(spec['C']), 'k': _maybe_float(spec['k']),
                          'tMid': _maybe_float(spec['t_mid'])} if spec else None),
        'sizeMw': _maybe_float(detection.get('sizeMw')),
        'nSites': _maybe_float(detection.get('nSites')),
        'effSiteMw': _maybe_float(detection.get('effSiteMw')),
    }

    # --- Per-metric covert series for the mini-charts (overlaid on the reference/default
    #     project, which the client fetches from /api/compute). Columns:
    #     [t, effectiveCompute, automationFraction, etsp, softwareEfficiency,
    #      trainingSystemPerformance, researchEffort, aiSwProgressMultRefPresentDay].
    #     The uplift (AI software R&D multiplier) is a ref-present-day counterfactual — a
    #     pure function of capability level E — so it maps onto the covert's E(t) via the
    #     reference (E, uplift) pairs, exactly as pipeline/mc.py _e_to_uplift does.
    covert_e_arr = np.asarray(rec['covert_e'], dtype=float)
    # Prefer the reference run's DENSE (E, uplift) map — uplift evaluated exactly on a
    # uniform-E grid (sim._dense_uplift_map), which is smooth/accurate across the steep
    # takeoff. Fall back to the native (effective_compute, uplift) samples, which are sparse
    # in E during takeoff and yield a kinky mapping, only if the dense map is unavailable.
    _me, _mu = ref.get('uplift_map_e'), ref.get('uplift_map_u')
    map_e = np.asarray(_me if _me is not None else [], dtype=float)
    map_u = np.asarray(_mu if _mu is not None else [], dtype=float)
    if map_e.size < 2 or map_u.size < 2:
        map_e = np.asarray(ref.get('effective_compute', []), dtype=float)
        map_u = np.asarray(ref.get('uplift', []), dtype=float)
    uplift_map: List[List[Optional[float]]] = []
    if map_e.size and map_u.size:
        order = np.argsort(map_e)
        ecs, ups = map_e[order], map_u[order]
        # The client maps ANY trajectory's E(t) (and each milestone) through this map; ship it
        # as-is and interp the covert's E(t) through the same map for the per-metric series.
        covert_uplift = np.interp(covert_e_arr, ecs, ups)
        uplift_map = [[_maybe_float(e), _maybe_float(u)] for e, u in zip(ecs, ups)]
    else:
        covert_uplift = np.full(covert_e_arr.shape, np.nan)

    # Other capability-indexed metrics (dense E -> metric maps from the reference run): the
    # ref-present-day coding-labor multiplier and AI research taste. Both are pure functions of
    # E, so — exactly like uplift — they map onto ANY trajectory's E(t) (covert AND legal).
    _cap_e = np.asarray(ref.get('uplift_map_e') if ref.get('uplift_map_e') is not None else [], dtype=float)
    _cap_order = np.argsort(_cap_e) if _cap_e.size else None
    _cap_ecs = _cap_e[_cap_order] if _cap_e.size else None

    def _map_cap(e_arr, key):
        e_arr = np.asarray(e_arr, dtype=float)
        v = ref.get(key)
        v = np.asarray(v if v is not None else [], dtype=float)
        if _cap_e.size and v.size == _cap_e.size:
            return np.interp(e_arr, _cap_ecs, v[_cap_order])
        return np.full(e_arr.shape, np.nan)

    covert_clm = _map_cap(covert_e_arr, 'cap_map_clm')
    covert_taste = _map_cap(covert_e_arr, 'cap_map_taste')

    # SW + ETSP rebased to the model base (see `off` above); E/TSP/automation/RE/uplift as-is.
    # Covert INDIGENOUS research effort + software-progress rate (theft/boost stripped — the
    # covert's own R&D output). Computed in the request path (api/index.py) via project_metrics;
    # NaN-filled if absent so the column count stays stable.
    _n_cov = len(rec['covert_t'])
    _re_indig = rec.get('covert_re_indigenous')
    _swr_indig = rec.get('covert_sw_rate_indigenous')
    _re_indig = _re_indig if _re_indig is not None else [float('nan')] * _n_cov
    _swr_indig = _swr_indig if _swr_indig is not None else [float('nan')] * _n_cov
    covert_metrics = _series(
        rec['covert_t'], rec['covert_e'], rec['covert_automation'], _reb(rec['covert_etsp']),
        _reb(rec['covert_sw']), rec['covert_tsp'], rec['covert_re'], covert_uplift,
        _re_indig, _swr_indig, covert_clm, covert_taste)

    # --- Legal USED per-metric series (SAME columns as covert) so the mini-charts can show a
    #     legal line alongside default + covert. The legal project pays NO theft, so its
    #     realized R&D == its indigenous R&D: research effort = the sim's u_RE, and the
    #     indigenous SW rate = d(SW)/dt. automation is a pure function of progress; uplift maps
    #     E(t) through the same reference map. (No project_metrics / implied-compute pass needed.)
    def _uplift_of(e_arr):
        e_arr = np.asarray(e_arr, dtype=float)
        if map_e.size and map_u.size:
            return np.interp(e_arr, ecs, ups)
        return np.full(e_arr.shape, np.nan)

    _params = (ref.get('metrics_ctx') or {}).get('params') or ref.get('prepared_params')
    _am = getattr(_params, 'automation_model', None)

    def _automation_of(logE_arr):
        a = np.asarray(logE_arr, dtype=float)
        if _am is None:
            return np.full(a.shape, np.nan)
        return np.array([_am.get_automation_fraction(float(e)) for e in a])

    lu_t = rec['legal_used_t']
    lu_e = np.asarray(rec['legal_used_e'], dtype=float)
    lu_logE = np.asarray(rec['legal_used_logE_raw'], dtype=float)
    lu_tsp = np.asarray(rec['legal_used_tsp'], dtype=float)
    lu_sw = np.asarray(rec['legal_used_sw'], dtype=float)
    lu_re = np.asarray(rec['legal_used_re'], dtype=float)
    lu_tt = np.asarray(lu_t, dtype=float)
    lu_swrate = np.gradient(lu_sw, lu_tt) if lu_sw.size > 1 else np.zeros(lu_sw.shape)
    legal_used_metrics = _series(
        lu_t, lu_e, _automation_of(lu_logE), _reb(lu_tsp + lu_sw),
        _reb(lu_sw), lu_tsp, lu_re, _uplift_of(lu_e),
        lu_re, lu_swrate,  # realized RE == indigenous RE (no theft)
        _map_cap(lu_e, 'cap_map_clm'), _map_cap(lu_e, 'cap_map_taste'))

    # --- Legal DISCOVERED per-metric series (SAME columns): the consortium's full research
    #     output (target = TED-AI + tax; `legal_e` in the record). Like used, it pays no
    #     theft, so realized RE == indigenous RE and the indigenous SW rate is d(SW)/dt.
    ldis_t = rec['legal_t']
    ldis_e = np.asarray(rec['legal_e'], dtype=float)
    ldis_logE = np.asarray(rec['legal_logE_raw'], dtype=float)
    ldis_tsp = np.asarray(rec['legal_tsp'], dtype=float)
    ldis_sw = np.asarray(rec['legal_sw'], dtype=float)
    ldis_re = np.asarray(rec['legal_re'], dtype=float)
    ldis_tt = np.asarray(ldis_t, dtype=float)
    ldis_swrate = np.gradient(ldis_sw, ldis_tt) if ldis_sw.size > 1 else np.zeros(ldis_sw.shape)
    legal_discovered_metrics = _series(
        ldis_t, ldis_e, _automation_of(ldis_logE), _reb(ldis_tsp + ldis_sw),
        _reb(ldis_sw), ldis_tsp, ldis_re, _uplift_of(ldis_e),
        ldis_re, ldis_swrate,  # realized RE == indigenous RE (no theft)
        _map_cap(ldis_e, 'cap_map_clm'), _map_cap(ldis_e, 'cap_map_taste'))

    # --- Leading company (post-deal-breakdown) per-metric series, SAME columns — its OWN
    #     series (rendered olive by the client), starting at the breakdown. It steals
    #     nothing (no consortium left), so like the legal its realized R&D == indigenous.
    #     The consortium's own series is truncated AT the breakdown (it ceases to exist);
    #     its sim grid provides a point exactly at t_breakdown, so the navy line holds its
    #     frozen value up to the breakdown and stops.
    leading_metrics = None
    if has_leading:
        ld_t = rec['leading_t']
        ld_e = np.asarray(rec['leading_e'], dtype=float)
        ld_sw = np.asarray(rec['leading_sw'], dtype=float)
        ld_tsp = np.asarray(rec['leading_tsp'], dtype=float)
        ld_re = np.asarray(rec['leading_re'], dtype=float)
        ld_tt = np.asarray(ld_t, dtype=float)
        ld_swrate = np.gradient(ld_sw, ld_tt) if ld_sw.size > 1 else np.zeros(ld_sw.shape)
        leading_metrics = _series(
            ld_t, ld_e, rec['leading_automation'], _reb(np.asarray(rec['leading_etsp'], float)),
            _reb(ld_sw), ld_tsp, ld_re, _uplift_of(ld_e),
            ld_re, ld_swrate,
            _map_cap(ld_e, 'cap_map_clm'), _map_cap(ld_e, 'cap_map_taste'))
        _tb = float(rec['deal_breakdown_t'])
        legal_used_metrics = [r for r in legal_used_metrics
                              if r[0] is not None and r[0] <= _tb]
        legal_discovered_metrics = [r for r in legal_discovered_metrics
                                    if r[0] is not None and r[0] <= _tb]
        # The downsampler can drop the frozen-tail grid point at exactly t_breakdown; hold
        # the consortium's (frozen) state to the breakdown so the navy line doesn't end at
        # its arrival with a gap before the olive leading series starts.
        if legal_used_metrics and legal_used_metrics[-1][0] < _tb - 1e-9:
            legal_used_metrics.append([_tb] + list(legal_used_metrics[-1][1:]))
        if legal_discovered_metrics and legal_discovered_metrics[-1][0] < _tb - 1e-9:
            legal_discovered_metrics.append([_tb] + list(legal_discovered_metrics[-1][1:]))

    scalars = {
        'covertWins': bool(rec['covert_wins']),
        'leadYears': _maybe_float(rec['lead_years']),
        # covert time-to-TED-AI from the pause split / default world's (see simulate_covert)
        'covertSlowdown': _maybe_float(rec.get('covert_slowdown')),
        'covertTtt': _maybe_float(rec['covert_ttt']),
        'legalTtt': _maybe_float(rec['legal_ttt']),
        'legalUsedTtt': _maybe_float(rec['legal_used_ttt']),
        'tedAiE': _maybe_float(rec['ted_ai_e']),
        'legalTargetE': _maybe_float(rec['legal_target_e']),   # TED-AI + alignment tax (discovered)
        'acTime': _maybe_float(rec['ac_time']),
        'tedAiTime': _maybe_float(rec['ted_ai_time']),
        'pauseYear': _maybe_float(t_pause),
        'anchorProgressRate': _maybe_float(rec['anchor_progress_rate']),
        'alignmentTaxOoms': _maybe_float(rec['alignment_tax_ooms']),
        'setupSwGainOoms': _maybe_float(rec['setup_sw_gain_ooms']),
        'totalH100eUsed': _maybe_float(rec['total_h100e_used']),
        'totalH100eFrac': _maybe_float(rec.get('total_h100e_frac')),
        'swRateSolved': _maybe_float(rec.get('sw_rate_solved')),
        'usedSwRateSolved': _maybe_float(rec.get('used_sw_rate_solved')),
        'legalHwRateSolved': _maybe_float(rec.get('legal_hw_rate_solved')),
        # deal breakdown (enabled flag, absolute breakdown year, leading-co TED-AI crossing)
        'dealBreakdownEnabled': bool(rec.get('deal_breakdown_enabled')),
        'dealBreakdownT': _maybe_float(rec.get('deal_breakdown_t')),
        'leadingTtt': _maybe_float(rec.get('leading_ttt')),
    }

    return {
        'success': True,
        'scalars': scalars,
        'capability': capability,
        'etsp': etsp,
        'detection': detection_out,
        # Under a deal breakdown legalUsed is truncated at the breakdown and `leading` is
        # the leading company's own series from there on.
        'metrics': {'covert': covert_metrics, 'legalUsed': legal_used_metrics,
                    'legalDiscovered': legal_discovered_metrics,
                    **({'leading': leading_metrics} if leading_metrics is not None else {})},
        # Cumulative-since-pause stacked-area decompositions.
        'breakdown': {
            'covert': covert_breakdown,  # [t, hardware, indigenous, stolen]
            'legal': legal_breakdown,    # [t, hardware, windfall, used, tax]
        },
        # Dense (E, uplift) map from the reference run — the client maps each trajectory's
        # E(t) and each milestone through this to plot the AI-software-R&D uplift axis.
        'upliftMap': uplift_map,
        # Milestone anchors from the SIM'S OWN reference run ({label: {'time', 'e'}}, E in
        # the same rebased space as `capability`). The client places per-trajectory
        # milestone dots against these: /api/compute's milestone E can drift ~0.01 OOMs
        # from this run's, which near the consortium's asymptotic TED-AI plateau would turn
        # into decades of crossing-time error.
        'milestones': {
            str(name): {'time': _maybe_float(m.get('time')), 'e': _maybe_float(m.get('e'))}
            for name, m in (ref.get('milestones') or {}).items()
        },
    }
