from flask import Flask, request, jsonify
from werkzeug.exceptions import HTTPException
import numpy as np
import io
import csv
import json
import hashlib
import logging
import time
from collections import OrderedDict
from dataclasses import fields as dataclass_fields

from progress_model import (
    ProgressModel, Parameters, TimeSeriesData,
)
from progress_model.parameters import (
    resolve_swe_multiplier_at_present_day,
)
from progress_model.integration import collect_ode_diagnostics
from api.response_formatters import build_time_series_payload
from progress_model.time_series import generate_time_series_from_dict
from progress_model.config import FLOP_PER_H100E_YEAR

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

_KNOWN_PARAM_KEYS = frozenset(f.name for f in dataclass_fields(Parameters) if f.init)

def drop_unknown_params(params_dict):
    """Drop parameter keys the current model doesn't accept, with a warning.

    Clients that loaded the frontend before a deploy keep POSTing the previous
    frontend's full parameter dict; when a deploy removes or renames a model
    parameter, Parameters(**...) would raise TypeError and 500 every request
    from those stale pages. They get current-model results for the parameters
    that still exist, which is the best available answer until they reload.
    """
    unknown = sorted(k for k in params_dict if k not in _KNOWN_PARAM_KEYS)
    if unknown:
        logger.warning(f"Ignoring unknown model parameters (stale client?): {unknown}")
    return {k: v for k, v in params_dict.items() if k in _KNOWN_PARAM_KEYS}

# The model raises ValueError for parameter sets it refuses to run (validation
# failures, unsolvable uplift schedules from Monte Carlo samples, ...). Those are
# client errors, expected under rejection sampling: answer 400 with the message
# and skip the traceback — the solver already logged its own diagnostics.
@app.errorhandler(ValueError)
def handle_value_error(e):
    logger.warning(f"Rejected model request: {e}")
    return jsonify({'success': False, 'error': str(e)}), 400

# Anything else unhandled is a genuine server bug: keep the traceback, but give
# the client parseable JSON instead of Werkzeug's HTML error page.
@app.errorhandler(Exception)
def handle_uncaught_exception(e):
    if isinstance(e, HTTPException):
        return e
    logger.exception(f"Unhandled error serving {request.path}")
    return jsonify({'success': False, 'error': str(e)}), 500

def create_default_time_series():
    """Load default time series data from inputs/input_data.csv"""
    try:
        # Try to load from inputs/input_data.csv
        import os
        csv_path = os.path.join(os.path.dirname(__file__), '../inputs/input_data.csv')
        
        if os.path.exists(csv_path):
            logger.info("Loading default time series data from inputs/input_data.csv")
            with open(csv_path, 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                data = list(reader)
            
            time = np.array([float(row['time']) for row in data])
            L_HUMAN = np.array([float(row['L_HUMAN']) for row in data])
            inference_H100e = np.array([float(row['inference_H100e']) for row in data])
            experiment_H100e = np.array([float(row['experiment_H100e']) for row in data])
            training_H100e = np.array([float(row['training_H100e']) for row in data])
            # Optional display-only columns (world compute, leading AI company total)
            world_H100e = (np.array([float(row['world_H100e']) for row in data])
                           if data and 'world_H100e' in data[0] else None)
            total_ai_company_H100e = (np.array([float(row['total_ai_company_H100e']) for row in data])
                                      if data and 'total_ai_company_H100e' in data[0] else None)

            logger.info(f"Loaded time series data: {len(data)} points from {time[0]} to {time[-1]}")
            return TimeSeriesData(time, L_HUMAN, inference_H100e, experiment_H100e, training_H100e,
                                  world_H100e=world_H100e,
                                  total_ai_company_H100e=total_ai_company_H100e)
        else:
            logger.warning("input_data.csv not found, falling back to synthetic data")
            raise FileNotFoundError("input_data.csv not found")
            
    except Exception as e:
        logger.warning(f"Error loading input_data.csv: {e}, falling back to synthetic data")
        # Fallback to synthetic data if CSV loading fails
        time = np.linspace(2029, 2030, 12)
        L_HUMAN = np.ones_like(time) * 1e6
        inference_H100e = np.logspace(3, 8, len(time))
        experiment_H100e = np.logspace(6, 10, len(time))  # Use exponential growth as fallback
        training_H100e = np.logspace(3, 5, len(time))  # raw H100e, synthetic fallback

        logger.info("Using synthetic fallback data")
        return TimeSeriesData(time, L_HUMAN, inference_H100e, experiment_H100e, training_H100e)

def create_default_parameters():
    """Create default model parameters"""
    return Parameters()

def load_time_series_from_csv_path(csv_path):
    """Load time series data from a given CSV file path"""
    with open(csv_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        data = list(reader)
    time = np.array([float(row['time']) for row in data])
    L_HUMAN = np.array([float(row['L_HUMAN']) for row in data])
    inference_H100e = np.array([float(row['inference_H100e']) for row in data])
    experiment_H100e = np.array([float(row['experiment_H100e']) for row in data])
    training_H100e = np.array([float(row['training_H100e']) for row in data])
    world_H100e = (np.array([float(row['world_H100e']) for row in data])
                   if data and 'world_H100e' in data[0] else None)
    total_ai_company_H100e = (np.array([float(row['total_ai_company_H100e']) for row in data])
                              if data and 'total_ai_company_H100e' in data[0] else None)
    return TimeSeriesData(time, L_HUMAN, inference_H100e, experiment_H100e, training_H100e,
                          world_H100e=world_H100e,
                          total_ai_company_H100e=total_ai_company_H100e)

def time_series_to_dict(data: TimeSeriesData):
    """Convert TimeSeriesData to dictionary for JSON serialization"""
    result = {
        'time': data.time.tolist(),
        'L_HUMAN': data.L_HUMAN.tolist(),
        'inference_H100e': data.inference_H100e.tolist(),
        'experiment_H100e': data.experiment_H100e.tolist(),
        'training_H100e': data.training_H100e.tolist(),
    }
    # Optional display-only series (present when the source CSV carries them)
    world = getattr(data, 'world_H100e', None)
    if world is not None:
        result['world_H100e'] = world.tolist()
    company_total = getattr(data, 'total_ai_company_H100e', None)
    if company_total is not None:
        result['total_ai_company_H100e'] = company_total.tolist()
    return result

def params_to_dict(params: Parameters):
    """Convert parameters to dictionary for JSON serialization""" 
    param_dict = {
        'rho_coding_labor': params.rho_coding_labor,
        'rho_experiment_capacity': params.rho_experiment_capacity,
        'alpha_experiment_capacity': params.alpha_experiment_capacity,
        'r_software': params.r_software,
        'automation_fraction_at_coding_automation_anchor': params.automation_fraction_at_coding_automation_anchor,
        'automation_interp_type': getattr(params, 'automation_interp_type', 'exponential'),

        'coding_labor_normalization': params.coding_labor_normalization,
        'coding_labor_mode': getattr(params, 'coding_labor_mode', 'simple_ces'),
        'coding_automation_efficiency_slope': getattr(params, 'coding_automation_efficiency_slope', 1.0),
        'optimal_ces_eta_init': getattr(params, 'optimal_ces_eta_init', 1.0),
        'optimal_ces_grid_size': getattr(params, 'optimal_ces_grid_size', 4096),
        'optimal_ces_frontier_tail_eps': getattr(params, 'optimal_ces_frontier_tail_eps', 1e-6),
        'optimal_ces_frontier_cap': getattr(params, 'optimal_ces_frontier_cap', None),
        'max_serial_coding_labor_multiplier': getattr(params, 'max_serial_coding_labor_multiplier', None),
        'experiment_compute_exponent': params.experiment_compute_exponent,
            'ai_research_taste_at_coding_automation_anchor': params.ai_research_taste_at_coding_automation_anchor,
            'ai_research_taste_at_coding_automation_anchor_sd': params.ai_research_taste_at_coding_automation_anchor_sd,
            'ai_research_taste_slope': params.ai_research_taste_slope,
        'taste_schedule_type': params.taste_schedule_type,
        'progress_at_aa': params.progress_at_aa,
        'ac_time_horizon_minutes': params.ac_time_horizon_minutes,
        'horizon_extrapolation_type': params.horizon_extrapolation_type,
        # AC anchor mode + revenue/ECI-mode parameters
        'ac_anchor_mode': getattr(params, 'ac_anchor_mode', 'time-horizon'),
        'present_revenue': getattr(params, 'present_revenue', None),
        'present_revenue_growth_rate': getattr(params, 'present_revenue_growth_rate', None),
        'revenue_required_for_ac': getattr(params, 'revenue_required_for_ac', None),
        'eci_at_present_day': getattr(params, 'eci_at_present_day', None),
        'eci_growth_rate': getattr(params, 'eci_growth_rate', None),
        'eci_required_for_ac': getattr(params, 'eci_required_for_ac', None),
        # Raw values (None = "derive it") plus the resolved values the model
        # actually used, so the frontend can show what is in force without
        # re-implementing the None conventions.
        'serial_uplift_at_reference_date': getattr(params, 'serial_uplift_at_reference_date', None),
        'serial_uplift_at_reference_date_resolved': getattr(params, 'serial_uplift_at_reference_date', None),
        # swe_multiplier_at_present_day is the DERIVED, legacy view of the same
        # belief. Its raw is None unless the caller actually stated it.
        'swe_multiplier_at_present_day': getattr(params, 'swe_multiplier_at_present_day', None),
        'swe_multiplier_at_present_day_resolved': resolve_swe_multiplier_at_present_day(params),
        'excess_uplift_doubling_time': getattr(params, 'excess_uplift_doubling_time', None),
        'uplift_reference_date': getattr(params, 'uplift_reference_date', None),
        'uplift_reference_date_resolved': (
            getattr(params, 'uplift_reference_date', None)
            if getattr(params, 'uplift_reference_date', None) is not None
            else getattr(params, 'present_day', None)
        ),
        # Manual horizon fitting parameters
        'present_day': params.present_day,
        'present_horizon': params.present_horizon,
        'present_doubling_time': params.present_doubling_time,
        'doubling_difficulty_growth_factor': params.doubling_difficulty_growth_factor,
        # Gap mode selection
        'include_gap': getattr(params, 'include_gap', 'no gap'),
        'gap_years': float(getattr(params, 'gap_years', 0.0)),
        # Research taste distribution parameter
        'median_to_top_taste_multiplier': getattr(params, 'median_to_top_taste_multiplier', 3.25),
        # Median-to-best multipliers for general capability milestones
        'strat_ai_m2b': getattr(params, 'strat_ai_m2b', 2.0),
        'ted_ai_m2b': getattr(params, 'ted_ai_m2b', 4.0),
    }

    return param_dict

@app.route('/api/compute', methods=['POST'])
def compute_model():
    """Compute model with given parameters"""
    data = request.json

    # Parse parameters
    all_params_dict = data.get('parameters', {})

    # Extract time series generation parameters
    time_series_param_keys = ['constant_training_system_performance_growth_rate', 'slowdown_year', 'post_slowdown_training_system_performance_growth_rate']
    time_series_params = {k: all_params_dict[k] for k in time_series_param_keys if k in all_params_dict}

    # Filter out time series parameters from model parameters
    params_dict = {k: v for k, v in all_params_dict.items() if k not in time_series_param_keys}

    # Check if profiling is requested
    enable_profiling = data.get('enable_profiling', False)

    # Initialize profiling variables
    total_time = 0.0
    total_cpu_time = 0.0
    timing_breakdown = {}
    cpu_timing_breakdown = {}
    ode_diagnostics = None

    if enable_profiling:
        start_time = time.perf_counter()
        start_cpu_time = time.process_time()

    # Time: Parameters creation (includes TasteDistribution init)
    _t0 = time.perf_counter()
    _c0 = time.process_time()
    params = Parameters(**drop_unknown_params(params_dict))
    timing_breakdown['params_creation'] = time.perf_counter() - _t0
    cpu_timing_breakdown['params_creation'] = time.process_time() - _c0

    # Generate time series data using parameters if provided, otherwise use default CSV
    _t0 = time.perf_counter()
    _c0 = time.process_time()

    # Check if exponential growth inputs should be used (when compute/labor growth slowdown is disabled)
    use_exponential_growth_inputs = data.get('use_exponential_growth_inputs', False)

    # Check if custom time series data is provided inline (from the CSV editor sidebar).
    # These values round-trip the model-internal representation returned by
    # /api/default-data (via time_series_to_dict), so training_compute here is
    # already OOMs of FLOP -- do NOT re-apply the H100e->FLOP conversion.
    custom_ts = data.get('custom_time_series')

    if custom_ts and isinstance(custom_ts, dict):
        time_arr = np.array(custom_ts['time'], dtype=float)
        L_HUMAN_arr = np.array(custom_ts['L_HUMAN'], dtype=float)
        inference_arr = np.array(custom_ts['inference_H100e'], dtype=float)
        experiment_arr = np.array(custom_ts['experiment_H100e'], dtype=float)
        training_arr = np.array(custom_ts['training_H100e'], dtype=float)
        # Optional display-only series (world compute, leading AI company total); only
        # honored when they align with the time grid so a partial edit can't desync them.
        def _optional_series(key):
            vals = custom_ts.get(key)
            if isinstance(vals, list) and len(vals) == len(time_arr):
                return np.array(vals, dtype=float)
            return None
        time_series = TimeSeriesData(time_arr, L_HUMAN_arr, inference_arr, experiment_arr, training_arr,
                                     world_H100e=_optional_series('world_H100e'),
                                     total_ai_company_H100e=_optional_series('total_ai_company_H100e'))
        logger.info(f"Using custom inline time series: {len(time_arr)} points")
    elif use_exponential_growth_inputs:
        # The no-slowdown series (inputs/2200_exponential.csv) was dropped in the H100e
        # input migration, and the "Compute and Human Labor Growth Slowdown" checkbox is
        # removed from the UI until a converted series exists. Stale clients and old
        # share links can still send the flag — ignore it rather than 500 on the
        # missing file, falling through to the same series selection as everyone else.
        logger.warning("Ignoring use_exponential_growth_inputs: the exponential input series no longer exists")
        if time_series_params and len(time_series_params) == 3:
            import os
            base_csv_path = os.path.join(os.path.dirname(__file__), '../inputs/input_data.csv')
            time_series = generate_time_series_from_dict(time_series_params, base_csv_path)
        else:
            time_series = create_default_time_series()
    elif time_series_params and len(time_series_params) == 3:
        # Use time series generator with the provided parameters
        import os
        base_csv_path = os.path.join(os.path.dirname(__file__), '../inputs/input_data.csv')
        time_series = generate_time_series_from_dict(time_series_params, base_csv_path)
        logger.info(f"Generated time series with params: {time_series_params}")
    else:
        # Fall back to default CSV
        time_series = create_default_time_series()
        logger.info("Using default time series from CSV")
    timing_breakdown['time_series_load'] = time.perf_counter() - _t0
    cpu_timing_breakdown['time_series_load'] = time.process_time() - _c0

    # Get time range
    time_range = data.get('time_range', [2029, 2030])
    initial_progress = data.get('initial_progress', 0.0)

    # Time: Model initialization
    _t0 = time.perf_counter()
    _c0 = time.process_time()
    model = ProgressModel(params, time_series)
    timing_breakdown['model_init'] = time.perf_counter() - _t0
    cpu_timing_breakdown['model_init'] = time.process_time() - _c0

    # Time: Trajectory computation
    _t0 = time.perf_counter()
    _c0 = time.process_time()
    if enable_profiling:
        with collect_ode_diagnostics() as ode_diagnostics:
            times, progress_values, research_stock_values = model.compute_progress_trajectory(
                time_range, initial_progress
            )
    else:
        times, progress_values, research_stock_values = model.compute_progress_trajectory(
            time_range, initial_progress
        )
    timing_breakdown['compute_trajectory'] = time.perf_counter() - _t0
    cpu_timing_breakdown['compute_trajectory'] = time.process_time() - _c0

    # Extract timing from model results if available
    if hasattr(model, 'results') and model.results:
        if 'timing' in model.results:
            timing_breakdown['model_internal'] = model.results['timing']

    if enable_profiling:
        end_time = time.perf_counter()
        total_time = end_time - start_time
        total_cpu_time = time.process_time() - start_cpu_time

    # All metrics are now computed by ProgressModel - use them directly
    all_metrics = model.results

    # Prepare summary focusing on SC metrics
    summary = {
        'time_range': time_range
    }
    
    # Add anchor taste slope metrics if available
    if model.results.get('ai_research_taste_slope_per_anchor_progress_year') is not None:
        summary['ai_research_taste_slope_per_anchor_progress_year'] = float(model.results['ai_research_taste_slope_per_anchor_progress_year'])
    if model.results.get('ai_research_taste_slope_per_effective_oom') is not None:
        summary['ai_research_taste_slope_per_effective_oom'] = float(model.results['ai_research_taste_slope_per_effective_oom'])

    # Add anchor progress rate if available
    if model.results.get('anchor_progress_rate') is not None:
        summary['anchor_progress_rate'] = float(model.results['anchor_progress_rate'])

    # Add r_software (calibrated value) if available
    if model.results.get('r_software') is not None:
        summary['r_software'] = float(model.results['r_software'])
    
    # Add beta_software (inverse of r_software) if available
    if model.results.get('beta_software') is not None:
        summary['beta_software'] = float(model.results['beta_software'])
    
    # Add top taste percentile metrics if available
    if model.results.get('top_taste_percentile') is not None:
        summary['top_taste_percentile'] = float(model.results['top_taste_percentile'])
    if model.results.get('top_taste_num_sds') is not None:
        summary['top_taste_num_sds'] = float(model.results['top_taste_num_sds'])
    if model.results.get('f_multiplier_per_sd') is not None:
        summary['f_multiplier_per_sd'] = float(model.results['f_multiplier_per_sd'])
    if model.results.get('slope_times_log_f') is not None:
        summary['slope_times_log_f'] = float(model.results['slope_times_log_f'])

    # Add instantaneous anchor doubling time (years) if available
    if model.results.get('instantaneous_anchor_doubling_time_years') is not None:
        summary['instantaneous_anchor_doubling_time_years'] = float(model.results['instantaneous_anchor_doubling_time_years'])

    # Add SC timing information if available
    if model.results.get('sc_progress_level') is not None and model.results.get('sc_sw_multiplier') is not None:
        summary['aa_time'] = float(model.results['aa_time'])
        summary['sc_progress_level'] = float(model.results['sc_progress_level'])
        summary['sc_sw_multiplier'] = float(model.results['sc_sw_multiplier']) 
        logger.info(f"SC time: {summary['aa_time']}, SC progress level: {summary['sc_progress_level']}, SC SW multiplier: {summary['sc_sw_multiplier']}")
    # Add AI2027 SC time if computed
    if model.results.get('ai2027_sc_time') is not None:
        try:
            summary['ai2027_sc_time'] = float(model.results.get('ai2027_sc_time'))
        except Exception:
            summary['ai2027_sc_time'] = None
    # Extract horizon params if available
    horizon_params = None
    if 'horizon_params' in model.results:
        hp = model.results['horizon_params']
        horizon_params = {
            'uses_shifted_form': bool(hp.get('uses_shifted_form', False)),
            'anchor_progress': float(hp['anchor_progress']) if hp.get('anchor_progress') is not None else None
        }

    # Log milestone times for debugging
    milestones = model.results['milestones']
    print(f"[API] Milestone times: AC={milestones.get('AC', {}).get('time')}, SAR={milestones.get('SAR-level-experiment-selection-skill', {}).get('time')}")

    response_data = {
        'success': True,
        # uplift_solve and td_correction ride inside summary because summary is already
        # threaded to AdvancedSections at every call site; a separate top-level key would
        # need a new prop through four components for a couple of scalars.
        'summary': {**(summary or {}),
                    'uplift_solve': model.results.get('uplift_solve'),
                    'td_correction': model.results.get('td_correction')},
        'time_series': build_time_series_payload(model.results),
        'milestones': milestones,
        'horizon_params': horizon_params,
        # Surface computed exp capacity params so UI can display when using pseudoparameters
        'exp_capacity_params': {
            'rho': float(all_metrics.get('exp_capacity_params', {}).get('rho')) if all_metrics.get('exp_capacity_params') else None,
            'alpha': float(all_metrics.get('exp_capacity_params', {}).get('alpha')) if all_metrics.get('exp_capacity_params') else None,
            'experiment_compute_exponent': float(all_metrics.get('exp_capacity_params', {}).get('experiment_compute_exponent')) if all_metrics.get('exp_capacity_params') else None,
        }
    }

    # Add profiling data if enabled
    if enable_profiling:
        response_data['profiling'] = {
            'enabled': True,
            'total_time_seconds': total_time,
            'total_cpu_time_seconds': total_cpu_time,
            'timing_breakdown': timing_breakdown,
            'cpu_timing_breakdown': cpu_timing_breakdown,
            'ode_diagnostics': ode_diagnostics or [],
            'stats': '(cProfile disabled to avoid overhead)'
        }

    return jsonify(response_data)

# ---------------------------------------------------------------------------
# Plan A explorer (/api/plan-a) — a single interactive covert-vs-legal race.
# Calls the pure plan_a_covert_proj.sim / detection functions (NOT the MC pipeline
# orchestrator). The reference (un-paused) run is the expensive part and is independent
# of the covert knobs, so it is cached per (model params + time_range); covert-knob
# tweaks then only pay for simulate_covert + detection.
# ---------------------------------------------------------------------------

# Curated covert knobs that are ALWAYS required in the request. Conditional knobs
# (total_h100e / total_h100e_frac per compute_mode; max_sw_rate for capped_sw) are
# validated by simulate_covert itself (single source of truth) — passed through as given.
_COVERT_REQUIRED_KNOBS = (
    'years_before_ac', 'setup_period_years', 'setup_sw_gain_progress_months',
    'alignment_tax_progress_years', 'legal_scaling_duration', 'legal_strategy',
    'legal_mode', 'frac_stolen', 'dormancy_years', 'compute_mode',
    'facility_w_per_h100e', 'optimal_site_mw', 'max_hw_rate',
)
# Default time range and how far past AC to integrate the race. The covert race resolves
# a few years after AC (covert/legal cross TED-AI well before this), so we don't integrate
# the meaningless post-takeoff tail — keeps each interactive recompute affordable. Crossing
# times are unchanged vs the full 2100 horizon (verified).
_COVERT_TIME_RANGE = [2017.0, 2100.0]
_COVERT_HORIZON_AFTER_AC = 25.0
_COVERT_DETECTION_HORIZON = 30.0   # max years-after-pause for the detection curve

# Bounded LRU cache of reference runs, keyed on a stable hash of (model params, time_range).
_REF_CACHE: "OrderedDict[str, dict]" = OrderedDict()
_REF_CACHE_MAX = 16

# Bounded LRU cache of frailty fits, keyed on a stable hash of the detection grids. The fit
# is a Nelder-Mead minimization (~seconds) that depends ONLY on the grids — which come from
# the config defaults and almost never change between covert-knob tweaks — so caching it keeps
# each interactive recompute affordable.
_FRAILTY_CACHE: "OrderedDict[str, object]" = OrderedDict()
_FRAILTY_CACHE_MAX = 16


def _get_cached_frailty(grids):
    """The joint detection fit for ``grids``: per-size frailty + the shared per-site
    hazard — the same fit the MC pipeline bakes into its datasets.

    Resolution order: in-process LRU -> the BAKED fit committed to the repo
    (detection_baked.load_baked_fit, fingerprint-checked) -> a live
    detection.fit_detection_model. The live fit takes MINUTES — far beyond the lambda's
    maxDuration, so a cold start that misses the baked fit will time out (the 504 of
    2026-07). In practice the baked fit must cover the config's grids; the live path
    only serves non-default grids (and local dev, where there is no time limit)."""
    from plan_a_covert_proj import detection, detection_baked
    key = hashlib.sha256(
        json.dumps(grids, sort_keys=True, default=str).encode()).hexdigest()
    fit = _FRAILTY_CACHE.get(key)
    if fit is not None:
        _FRAILTY_CACHE.move_to_end(key)
        return fit
    fit = detection_baked.load_baked_fit(grids)
    if fit is None:
        logger.warning("no baked detection fit for these grids "
                       "(stale config/detection_fit_baked.json? re-run "
                       "pipeline/bake_detection_fit.py); running the live joint fit "
                       "— this takes minutes")
        fit = detection.fit_detection_model(grids)
    _FRAILTY_CACHE[key] = fit
    _FRAILTY_CACHE.move_to_end(key)
    while len(_FRAILTY_CACHE) > _FRAILTY_CACHE_MAX:
        _FRAILTY_CACHE.popitem(last=False)
    return fit


def _covert_config_path():
    import os
    return os.path.join(os.path.dirname(__file__),
                        '../plan_a_covert_proj/config/covert_scenario_branch.yaml')


# Plan-A divergence series (the covert config's `plan_a_input_data`). The MC pipeline runs
# the legal/covert on the DEFAULT series up to the pause, then Plan A's compute ramp
# relativized to the pause (build_plan_a_series). Loaded once and reused.
_PLAN_A_CACHE = {}


def _get_plan_a_series():
    """The Plan-A input TimeSeriesData named by the covert config, or None if unset/missing."""
    if 'series' in _PLAN_A_CACHE:
        return _PLAN_A_CACHE['series']
    series = None
    try:
        import os
        from progress_model import load_time_series_data
        rel = (_load_covert_config() or {}).get('plan_a_input_data')
        if rel:
            path = rel if os.path.isabs(rel) else os.path.join(os.path.dirname(__file__), '..', rel)
            if os.path.exists(path):
                series = load_time_series_data(path)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"could not load plan_a_input_data ({e})")
    _PLAN_A_CACHE['series'] = series
    return series


def _load_covert_config():
    # Resolve `parent:` inheritance (covert_scenario_branch.yaml is a specialization of
    # covert_uncertain.yaml) — a raw yaml.safe_load would see only the child's overrides.
    from scripts.pipelines.batch_rollout import _load_config
    return _load_config(_covert_config_path()) or {}


def _ref_cache_key(params_dict, time_range):
    blob = json.dumps({'p': params_dict, 'tr': time_range}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


def _get_cached_ref(params, params_dict, data, time_range):
    """run_default cached on (model params, time_range). data is the fixed default series."""
    from plan_a_covert_proj import sim
    key = _ref_cache_key(params_dict, time_range)
    ref = _REF_CACHE.get(key)
    if ref is not None:
        _REF_CACHE.move_to_end(key)
        return ref
    ref = sim.run_default(params, data, time_range)
    _REF_CACHE[key] = ref
    _REF_CACHE.move_to_end(key)
    while len(_REF_CACHE) > _REF_CACHE_MAX:
        _REF_CACHE.popitem(last=False)
    return ref


# The explorer is published at /plan-a. The original /api/covert path is gone: the embed
# loads this app's own page, so its JS ships from this deployment and calls /api/plan-a,
# and shared links are page URLs served by that same JS — neither pinned the API path.
@app.route('/api/plan-a/defaults', methods=['GET'])
def get_plan_a_defaults():
    """Seed the explorer UI from the covert config: median knob values, the detection
    grids, and the silver-bullet block. The compute path (/api/plan-a) takes no implicit
    defaults — the client echoes these back (and may override) on every request."""
    try:
        cfg = _load_covert_config()
        cov = cfg.get('covert_parameters', {}) or {}

        def _median(spec):
            # fixed dist -> its value; otherwise the central tendency the pipeline would use.
            if not isinstance(spec, dict):
                return None
            if spec.get('dist') == 'fixed':
                return spec.get('value')
            if spec.get('dist') == 'cdf_points':
                # Elicited CDF anchors: the exact median is the pipeline's quantile-0.5 draw.
                from scripts.pipelines.batch_rollout import _sample_from_dist_with_quantile
                return _sample_from_dist_with_quantile(spec, 0.5)
            for k in ('value', 'median', 'mean', 'mu'):
                if spec.get(k) is not None:
                    return spec[k]
            lo, hi = spec.get('min'), spec.get('max')
            if lo is not None and hi is not None:
                return (float(lo) + float(hi)) / 2.0
            return None

        knobs = {k: _median(v) for k, v in cov.items()}
        # The explorer only supports frontier_relative / absolute sizing (it has no UI for
        # the bottom-up diversion fractions and never sends them), so coerce any other
        # configured compute_mode — the MC pipeline reads the yaml directly and is unaffected.
        if knobs.get('compute_mode') not in ('frontier_relative', 'absolute'):
            knobs['compute_mode'] = 'frontier_relative'
        grids = cfg.get('detection_grids')
        sizes = sorted(float(g['size_mw']) for g in (grids or []))

        # Seed the MODEL parameters from the covert experiment's model config (the same
        # central values the MC pipeline uses), NOT the website's homepage preset — so the
        # explorer's defaults reproduce the covert experiment. ac_time_horizon_minutes is
        # returned in LOG10(minutes) because the frontend ParametersType stores it that way
        # (and converts back to minutes before POSTing).
        import os
        import math
        from scripts.pipelines.batch_rollout import _load_config as _load_model_cfg, _get_median_parameter_dict
        model_cfg_path = os.path.join(os.path.dirname(__file__),
                                      '../plan_a_covert_proj/config/model_scenario_branch.yaml')
        model_params = None
        try:
            import inspect
            mcfg = _load_model_cfg(model_cfg_path) or {}
            central = _get_median_parameter_dict(mcfg.get('parameters', {}) or {})
            # Match the MC pipeline exactly: it runs Parameters(**config_params). To make the
            # client reproduce that — and NOT leak the website's homepage-preset values for the
            # many fields the covert config leaves to calibration (r_software, the CES
            # exponents, automation params, …) — return the FULLY RESOLVED parameter set:
            # construct Parameters(**config) and read back every constructor kwarg. The client
            # spreads this over its homepage defaults, so every model field is overridden with
            # the experiment's resolved value (e.g. the documented single ~300 MW facility).
            _sig = inspect.signature(Parameters.__init__)
            _p0 = Parameters(**central)
            resolved = {}
            for n in _sig.parameters:
                if n == 'self' or not hasattr(_p0, n):
                    continue
                v = getattr(_p0, n)
                if v is None or isinstance(v, (bool, int, float, str)):
                    resolved[n] = v
            # The frontend's convertParametersToAPIFormat derives parallel_penalty from
            # coding_labor_exponent and include_gap from benchmarks_and_gaps_mode, then drops
            # the originals — so seed those frontend-only keys consistently or the homepage
            # values would win on the round-trip.
            if 'parallel_penalty' in resolved:
                resolved['coding_labor_exponent'] = resolved['parallel_penalty']
            if 'include_gap' in resolved:
                resolved['benchmarks_and_gaps_mode'] = (str(resolved['include_gap']).lower() == 'gap')
            ac = resolved.get('ac_time_horizon_minutes')
            if ac is not None and float(ac) > 0:
                resolved['ac_time_horizon_minutes'] = math.log10(float(ac))  # UI stores log10(min)

            def _coerce(v):
                if isinstance(v, bool):
                    return v
                return float(v) if isinstance(v, (int, float)) else v
            model_params = {k: _coerce(v) for k, v in resolved.items() if v is not None}
        except Exception as e:  # noqa: BLE001
            logger.warning(f"covert defaults: could not seed model params ({e})")

        return jsonify({
            'success': True,
            'knobs': knobs,
            'detection_grids': grids,
            'silver_bullet': cfg.get('detection_silver_bullet'),
            'site_mw_range': ([sizes[0], sizes[-1]] if sizes else None),
            'time_range': cfg.get('time_range', _COVERT_TIME_RANGE),
            'model_params': model_params,
        })
    except Exception as e:
        logger.error(f"Error loading covert defaults: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/plan-a', methods=['POST'])
def compute_plan_a():
    """Simulate one covert-vs-legal race + detection decomposition and return chart data."""
    from plan_a_covert_proj import sim, detection
    from api.covert_formatters import build_covert_payload

    body = request.json or {}
    params_dict = body.get('parameters', {}) or {}
    knobs = body.get('knobs')
    grids = body.get('detection_grids')
    sb_block = body.get('silver_bullet')   # None/absent -> silver bullet disabled
    time_range = body.get('time_range') or _COVERT_TIME_RANGE

    # --- validate inputs (no silent defaults: missing/invalid -> 400) ----------------
    if not isinstance(knobs, dict):
        return jsonify({'success': False, 'error': "request must include a 'knobs' object"}), 400
    missing = [k for k in _COVERT_REQUIRED_KNOBS if knobs.get(k) is None]
    if missing:
        return jsonify({'success': False,
                        'error': f"missing required knob(s): {missing}"}), 400
    if not grids:
        return jsonify({'success': False,
                        'error': "request must include non-empty 'detection_grids'"}), 400

    def _f(name):
        return float(knobs[name])

    # The covert race uses a FIXED input series (input_data.csv), so the time-series
    # generation params (used by /api/compute to synthesize a series) are not applicable
    # here — drop them before constructing Parameters (mirrors /api/compute's split).
    _ts_keys = ('constant_training_system_performance_growth_rate', 'slowdown_year',
                'post_slowdown_training_system_performance_growth_rate')
    params_dict = {k: v for k, v in params_dict.items() if k not in _ts_keys}

    try:
        # Reference (un-paused) run — cached on model params + time_range.
        params_dict = drop_unknown_params(params_dict)
        params = Parameters(**params_dict)
        data = create_default_time_series()
        ref = _get_cached_ref(params, params_dict, data, time_range)
        if ref.get('ac_time') is None or ref.get('ted_ai_e') is None:
            return jsonify({'success': False,
                            'error': 'model never reaches AC / TED-AI for these parameters'}), 400

        t_pause = float(ref['ac_time']) - _f('years_before_ac')
        setup = _f('setup_period_years')
        t_end = min(float(time_range[1]), float(ref['ac_time']) + _COVERT_HORIZON_AFTER_AC)
        target_year = t_pause + setup + _f('legal_scaling_duration')

        # The legal/covert run on the DEFAULT series up to the pause, then Plan A's compute
        # ramp relativized to the pause — exactly as the MC pipeline (build_plan_a_series).
        # Self-anchored (pause_med = this run's pause), which reproduces the pipeline's
        # median-rollout behaviour. AC/milestones still come from the DEFAULT series (ref).
        lc_data = data
        plan_a = _get_plan_a_series()
        if plan_a is not None:
            try:
                from plan_a_covert_proj.pipeline.mc import build_plan_a_series
                lc_data = build_plan_a_series(data, plan_a, t_pause, t_pause + setup,
                                              t_pause, t_pause + setup, float(time_range[1]))
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Plan-A relativization failed ({e}); using default series")
                lc_data = data

        _min_sw_frac = (float(knobs['min_sw_frac_of_tsp'])
                        if knobs.get('min_sw_frac_of_tsp') is not None else 0.2)
        rec = sim.simulate_covert(
            params, lc_data, t_pause,
            alignment_tax_progress_years=_f('alignment_tax_progress_years'),
            target_year=target_year, max_hw_rate=_f('max_hw_rate'),
            compute_mode=knobs['compute_mode'], frac_stolen=_f('frac_stolen'),
            total_h100e=(float(knobs['total_h100e']) if knobs.get('total_h100e') is not None else None),
            total_h100e_frac=(float(knobs['total_h100e_frac']) if knobs.get('total_h100e_frac') is not None else None),
            dormancy_years=_f('dormancy_years'),
            # construction follows dormancy in sequence (optional; older clients omit it)
            construction_years=(float(knobs['construction_years'])
                                if knobs.get('construction_years') is not None else 0.0),
            setup_period_years=setup,
            setup_sw_gain_progress_months=_f('setup_sw_gain_progress_months'),
            t_end=t_end, ref=ref, return_trajectories=True,
            legal_mode=knobs['legal_mode'], legal_strategy=knobs['legal_strategy'],
            max_sw_rate=(float(knobs['max_sw_rate']) if knobs.get('max_sw_rate') is not None else None),
            post_scaling_externalities_rate=(float(knobs['post_scaling_externalities_rate'])
                                             if knobs.get('post_scaling_externalities_rate') is not None else 0.0),
            min_sw_frac_of_tsp=_min_sw_frac,
            # deal breakdown (optional, gated): consortium dissolves at pause-end + N years,
            # leading company continues alone (validated by simulate_covert).
            enable_deal_breakdown=bool(knobs.get('enable_deal_breakdown')),
            deal_breakdown_years_after_pause_end=(float(knobs['deal_breakdown_years_after_pause_end'])
                                                  if knobs.get('deal_breakdown_years_after_pause_end') is not None else None),
            breakdown_compute_reduction=(float(knobs['breakdown_compute_reduction'])
                                         if knobs.get('breakdown_compute_reduction') is not None else None),
            breakdown_compute_mode=(str(knobs['breakdown_compute_mode'])
                                    if knobs.get('breakdown_compute_mode') else 'frontier_at_pause'),
            post_breakdown_compute_growth_rate=(float(knobs['post_breakdown_compute_growth_rate'])
                                                if knobs.get('post_breakdown_compute_growth_rate') is not None else 0.0))
        rec['min_sw_frac_of_tsp'] = _min_sw_frac
        if not rec.get('valid', False):
            return jsonify({'success': False,
                            'error': f"simulation invalid: {rec.get('reason', 'unknown')}"}), 400

        # Covert INDIGENOUS metrics: re-run the canonical AIFM metrics pass on the covert's
        # own trajectory + its effective compute (make_covert_data), which yields the covert's
        # research effort / software-progress rate from its OWN R&D — i.e. with the stolen
        # legal RS-growth and theft boost stripped out (those live in the ODE RHS, not in
        # (E, RS, inputs)). This is computed in the request path, NOT in simulate_covert, so
        # the MC pipeline (which also calls simulate_covert) is unaffected. Best-effort.
        ctx = ref.get('metrics_ctx')
        if ctx is not None and rec.get('covert_logE_raw') is not None:
            try:
                cdata = sim.make_covert_data(t_pause, t_end, float(rec['total_h100e_used']), lc_data)
                ind = sim.project_metrics(rec['covert_t'], rec['covert_logE_raw'],
                                          rec['covert_rs'], cdata, ctx)
                rec['covert_re_indigenous'] = np.asarray(ind['research_efforts'], dtype=float).tolist()
                rec['covert_sw_rate_indigenous'] = np.asarray(ind['software_progress_rates'], dtype=float).tolist()
            except Exception as e:  # noqa: BLE001
                logger.warning(f"covert indigenous metrics failed ({e})")

        # --- detection: fit frailty from grids, build the silver bullet exactly as the
        #     MC pipeline (_run_sample), and decompose survival over a years-after-pause grid.
        fits, det_r_fit = _get_cached_frailty(grids)
        gmin, gmax = min(float(g['size_mw']) for g in grids), max(float(g['size_mw']) for g in grids)
        site_mw = _f('optimal_site_mw')
        if not (gmin <= site_mw <= gmax):
            return jsonify({'success': False,
                            'error': f"optimal_site_mw {site_mw} must lie within the detection_grids "
                                     f"range [{gmin:g}, {gmax:g}] MW (else the frailty interpolation "
                                     f"extrapolates)"}), 400
        size_mw = float(rec['total_h100e_used']) * _f('facility_w_per_h100e') / 1e6
        # per-site slip-up hazard comes from the joint fit, not a knob (mirrors mc.py)
        det_r = det_r_fit
        n_sites, eff_site = detection.deploy(size_mw, site_mw)

        sb_spec = None
        if sb_block:
            used_ttt = rec.get('legal_used_ttt')
            if used_ttt is not None and np.isfinite(used_ttt):
                t_used_ted = float(used_ttt) - t_pause     # legal used -> TED-AI (years after pause)
                t0 = 0.5 * (setup + t_used_ted)            # time-midpoint(end-of-pause, used->TED-AI)
                t1 = t_used_ted + float(sb_block['delta_years'])
                sb_spec = detection.build_silver_bullet(
                    float(sb_block['p_mid_pause_to_ted']), t0,
                    float(sb_block['p_used_ted_plus_d']), t1, float(sb_block['ceiling']))

        tg = np.linspace(0.0, min(t_end - t_pause, _COVERT_DETECTION_HORIZON), 200)
        # Dormancy shifts the detection clock: nothing is detectable before construction
        # starts, so the frailty clock runs from the end of dormancy and combined survival
        # is 1 throughout it. The silver-bullet curve stays on the pause clock.
        dorm = _f('dormancy_years')
        sb_survival = detection.g_no_silver_bullet(sb_spec, tg)              # G(t) = 1 - P_SB(t)
        frailty_survival = detection.survival(detection.interp_frailty(eff_site, fits), n_sites,
                                              np.maximum(tg - dorm, 0.0), det_r)
        combined_survival = detection.combined_survival(
            detection.interp_frailty(eff_site, fits), n_sites, tg, det_r,
            silver_bullet=sb_spec, dormancy_years=dorm)
        p_detected = 1.0 - combined_survival
        det = {
            't': tg, 'pDetected': p_detected, 'sbCdf': 1.0 - sb_survival,
            'sbSurvival': sb_survival, 'frailtySurvival': frailty_survival,
            'combinedSurvival': combined_survival, 'spec': sb_spec,
            'sizeMw': size_mw, 'nSites': n_sites, 'effSiteMw': eff_site,
        }
        payload = build_covert_payload(rec, ref, det, t_pause)

        # Input compute series the two trajectories run on: the DEFAULT series (drives the
        # reference/default project) and the Plan-A-relativized series (drives legal/covert).
        # Sampled (log-interp) on a shared grid so the UI can show how the inputs diverge.
        def _sample_inputs(ts, t0=2024.0):
            grid = np.linspace(t0, t_end, 120)
            tt = np.asarray(ts.time, dtype=float)

            def col(name):
                v = np.asarray(getattr(ts, name), dtype=float)
                return 10.0 ** np.interp(grid, tt, np.log10(np.maximum(v, 1e-300)))
            train = col('training_H100e')
            total = train + col('inference_H100e') + col('experiment_H100e')
            return [[float(grid[i]), float(train[i]), float(total[i])] for i in range(grid.size)]

        # Under a deal breakdown the consortium's input series is truncated at the breakdown
        # (no consortium left to use it) and the leading company's post-breakdown compute
        # (reduction × the mode's reference mix — frontier at pause, or the world's compute
        # at the collapse — growing at a constant OOM/yr) ships as its own 'leading' series,
        # which the client renders in olive. Must mirror simulate_covert's bd_ref_t choice.
        legal_rows = _sample_inputs(lc_data)
        leading_rows = None
        if rec.get('deal_breakdown_t') is not None:
            try:
                _bd_t = float(rec['deal_breakdown_t'])
                _bd_ref_t = (_bd_t if (knobs.get('breakdown_compute_mode') or 'frontier_at_pause')
                             == 'world_at_breakdown' else t_pause)
                _bd = sim.make_breakdown_data(
                    _bd_ref_t, _bd_t, t_end,
                    float(knobs['breakdown_compute_reduction']),
                    (float(knobs['post_breakdown_compute_growth_rate'])
                     if knobs.get('post_breakdown_compute_growth_rate') is not None else 0.0),
                    lc_data)
                leading_rows = _sample_inputs(_bd, t0=_bd_t)
                legal_rows = [r for r in legal_rows if r[0] is not None and r[0] <= _bd_t]
            except Exception as e:  # noqa: BLE001
                logger.warning(f"breakdown leading input series failed ({e})")
        payload['inputs'] = {  # rows: [year, training_H100e, total_rd_H100e]
            'default': _sample_inputs(data),
            'legal': legal_rows,
            **({'leading': leading_rows} if leading_rows is not None else {}),
        }
        # The covert datacenter holds its compute FLAT over [t_pause, t_end] (make_covert_data),
        # so it's two points at constant training / total-R&D H100e — a flat line starting at
        # the pause. total R&D = training + inference + experiment == total_h100e_used.
        try:
            _cd = sim.make_covert_data(t_pause, t_end, float(rec['total_h100e_used']), lc_data)
            _ct = float(_cd.training_H100e[0])
            _ctot = _ct + float(_cd.inference_H100e[0]) + float(_cd.experiment_H100e[0])
            # Emit the flat covert compute on a grid over [t_pause, t_end] (not just the two
            # endpoints) so points fall inside the chart's cropped display window — otherwise
            # only the t_pause point survives the crop and no line is drawn.
            _gcov = np.linspace(float(t_pause), float(t_end), 120)
            payload['inputs']['covert'] = [[float(t), _ct, _ctot] for t in _gcov]
        except Exception as e:  # noqa: BLE001
            logger.warning(f"covert input series failed ({e})")

        # Safety tax paid (DUMB version), in OOMs: log10( SE_disc * cumulative_training_FLOP
        # / tedai_EC ) = (TC_stock + SE_rebased) - ted_ai_e. The sim tracks TSP (the rate),
        # not the cumulative FLOP stock, so reconstruct TC_stock = log10( pre-pause cumulative
        # FLOP from the default series + ∫10^(legal discovered TSP) post-pause ). Unlike the
        # integrated effective compute, SE*TC ignores the integration lag — hence "dumb".
        try:
            from scipy.integrate import cumulative_trapezoid
            leg_t = np.asarray(rec['legal_t'], dtype=float)
            leg_tsp = np.asarray(rec['legal_tsp'], dtype=float)   # log10 FLOP/yr (rate)
            leg_sw = np.asarray(rec['legal_sw'], dtype=float)     # raw SW OOMs
            off = float(ref['rebase_offset']); ted = float(rec['ted_ai_e'])
            gpre = np.linspace(float(data.time[0]), t_pause, 500)
            rate_pre = 10.0 ** np.array([sim._tc_ooms(data, float(t)) for t in gpre])
            flop_pre = float(cumulative_trapezoid(rate_pre, gpre)[-1])  # cumulative training FLOP to the pause
            flop_post = (np.concatenate([[0.0], cumulative_trapezoid(10.0 ** leg_tsp, leg_t)])
                         if leg_t.size > 1 else np.zeros_like(leg_t))
            tc_stock = np.log10(np.maximum(flop_pre + flop_post, 1e-300))   # log10 cumulative FLOP
            dumb = tc_stock + (leg_sw + off) - ted                 # (TC + SE_rebased) - ted_ai_e
            # Deal breakdown: the consortium is gone, nobody pays the tax anymore -> 0 from the
            # breakdown on. Densify the zero tail — the frozen legal grid has no samples between
            # the breakdown and t_end, and the chart drops out-of-domain endpoints, so a two-point
            # tail would never draw.
            if rec.get('deal_breakdown_t') is not None:
                _tb = float(rec['deal_breakdown_t'])
                rows = [[float(leg_t[i]), float(dumb[i])] for i in range(leg_t.size) if leg_t[i] < _tb]
                rows += [[float(t), 0.0] for t in np.arange(_tb, float(leg_t[-1]) + 1e-9, 0.5)]
                payload['safetyTaxDumb'] = rows
            else:
                payload['safetyTaxDumb'] = [[float(leg_t[i]), float(dumb[i])] for i in range(leg_t.size)]
        except Exception as e:  # noqa: BLE001
            logger.warning(f"safety tax (dumb) failed ({e})")
        return jsonify(payload)
    except (ValueError, KeyError, TypeError) as e:
        # Model-level validation errors (compute_mode, silver-bullet ordering, bad knob types,
        # poor frailty fit, ...) surface to the UI as a 400 rather than a 500.
        return jsonify({'success': False, 'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Error computing covert race: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/parameter-config', methods=['GET'])
def get_parameter_config():
    """Get parameter configuration including bounds, defaults, and metadata"""
    try:
        from progress_model import config as cfg
        
        parameter_config = {
            'bounds': cfg.PARAMETER_BOUNDS,
            'defaults': cfg.DEFAULT_PARAMETERS,
            'validation_thresholds': cfg.PARAM_VALIDATION_THRESHOLDS,
            'taste_schedule_types': cfg.TASTE_SCHEDULE_TYPES,
            'taste_slope_defaults': cfg.TASTE_SLOPE_DEFAULTS,
            'horizon_extrapolation_types': cfg.HORIZON_EXTRAPOLATION_TYPES,
            'ac_anchor_modes': cfg.AC_ANCHOR_MODES,
            'automation_interp_types': ['exponential', 'linear', 'logistic'],
            'coding_labor_modes': ['simple_ces', 'optimal_ces'],
            'sos_start_milestones': cfg.SOS_START_MILESTONES,
            'modelConstants': {
                'baseForSoftwareLOM': cfg.BASE_FOR_SOFTWARE_LOM,
                'medianToTopGap': cfg.MEDIAN_TO_TOP_TASTE_MULTIPLIER,
                'topPercentile': cfg.TOP_PERCENTILE,
                'aggregateResearchTasteBaseline': cfg.AGGREGATE_RESEARCH_TASTE_BASELINE,
                'aiResearchTasteMaxSD': cfg.AI_RESEARCH_TASTE_MAX_SD,
                'aiResearchTasteMin': cfg.AI_RESEARCH_TASTE_MIN,
                'aiResearchTasteMax': cfg.AI_RESEARCH_TASTE_MAX,
                'automationFractionClipMin': cfg.AUTOMATION_FRACTION_CLIP_MIN,
                'trainingComputeReferenceYear': cfg.TRAINING_COMPUTE_REFERENCE_YEAR,
            },
            'descriptions': {
                'rho_coding_labor': {
                    'name': 'Cognitive Elasticity (ρ_cognitive)',
                    'description': 'Elasticity of substitution between AI and human cognitive labor',
                    'units': 'dimensionless'
                },
                'rho_experiment_capacity': {
                    'name': 'Progress Elasticity (ρ_progress)', 
                    'description': 'Elasticity of substitution in progress production function',
                    'units': 'dimensionless'
                },
                'alpha_experiment_capacity': {
                    'name': 'Compute Weight (α)',
                    'description': 'Weight of compute vs cognitive output in progress production',
                    'units': 'dimensionless'
                },
                'r_software': {
                    'name': 'Software Scale',
                    'description': 'Scale factor for software progress',
                    'units': 'dimensionless'
                },
                'automation_fraction_at_coding_automation_anchor': {
                    'name': 'Max Automation',
                    'description': 'Automation fraction when AI reaches superhuman coding ability',
                    'units': 'fraction'
                },
                'automation_interp_type': {
                    'name': 'Automation Interpolation Type',
                    'description': 'Method for interpolating automation between anchors',
                    'units': 'categorical'
                },
                
                'swe_multiplier_at_present_day': {
                    'name': 'SWE Multiplier at Anchor Time',
                    'description': 'Software engineering productivity multiplier at the anchor time',
                    'units': 'dimensionless'
                },
                'coding_labor_normalization': {
                    'name': 'Cognitive Output Normalization',
                    'description': 'Normalization factor for cognitive output',
                    'units': 'dimensionless'
                },
                'coding_labor_mode': {
                    'name': 'Coding Labor Mode',
                    'description': 'Select between simple CES and optimal frontier CES',
                    'units': 'categorical'
                },
                'coding_automation_efficiency_slope': {
                    'name': 'Automation Efficiency Slope',
                    'description': 'Slope of automation multiplier above threshold (η ∝ E^θ)',
                    'units': 'dimensionless'
                },
                'optimal_ces_eta_init': {
                    'name': 'Optimal CES η_init',
                    'description': 'Automation multiplier at the threshold',
                    'units': 'dimensionless'
                },
                'optimal_ces_grid_size': {
                    'name': 'Optimal CES Grid Size',
                    'description': 'Number of grid points for frontier precompute',
                    'units': 'count'
                },
                'optimal_ces_frontier_tail_eps': {
                    'name': 'Optimal CES Frontier Tail ε',
                    'description': 'Controls how close the precomputed grid approaches full automation (smaller ε allows larger AI labor multipliers)',
                    'units': 'dimensionless'
                },
                'optimal_ces_frontier_cap': {
                    'name': 'Optimal CES Frontier Cap',
                    'description': 'Targets a specific ceiling on the AI coding labor multiplier; internally mapped to the frontier grid tail ε',
                    'units': 'multiplier'
                },
                'max_serial_coding_labor_multiplier': {
                    'name': 'Max Serial Coding Labor Multiplier',
                    'description': 'Cap on serial coding labor; the model raises this to parallel_penalty to produce the internal optimal CES frontier cap',
                    'units': 'multiplier'
                },
                'experiment_compute_exponent': {
                    'name': 'Experiment Compute Discounting (ζ)',
                    'description': 'Diminishing returns factor for experiment compute',
                    'units': 'dimensionless'
                },

                'ai_research_taste_at_coding_automation_anchor': {
                    'name': 'Max AI Research Taste',
                    'description': 'AI research taste when AI reaches superhuman coding ability',
                    'units': 'fraction'
                },
                'ai_research_taste_at_coding_automation_anchor_sd': {
                    'name': 'Max AI Research Taste (SD)',
                    'description': 'AI research taste at superhuman coder specified in human-range standard deviations',
                    'units': 'SD'
                },
                
                'ai_research_taste_slope': {
                    'name': 'Automated Research Taste Slope',
                    'description': 'Steepness of AI research taste curve',
                    'units': 'dimensionless'
                },
                'taste_schedule_type': {
                    'name': 'AI Research Taste Schedule Type',
                    'description': 'Unit convention for SD-based schedule',
                    'units': 'categorical'
                },
                'progress_at_aa': {
                    'name': 'Progress at AC',
                    'description': 'Progress level where AI reaches superhuman coding ability (exponential mode)',
                    'units': 'dimensionless'
                },
                'ac_time_horizon_minutes': {
                    'name': 'Time Horizon to AC',
                    'description': 'Time horizon length corresponding to superhuman coder achievement',
                    'units': 'minutes'
                },
                'include_gap': {
                    'name': 'Include Gap',
                    'description': 'Choose whether to include a gap after pre-gap SC horizon',
                    'units': 'choice (gap | no gap)'
                },
                'gap_years': {
                    'name': 'Gap Size (YEAR-progress-years)',
                    'description': 'Additional gap after saturation horizon to reach SC, measured in YEAR-progress-years (converted using anchor_progress_rate). UI displays YEAR based on Anchor Time.',
                    'units': 'YEAR-progress-years'
                },
                'horizon_extrapolation_type': {
                    'name': 'Horizon Extrapolation Type',
                    'description': 'Method for extrapolating progress beyond the time horizon',
                    'units': 'categorical'
                },
                'parallel_penalty': {
                    'name': 'Parallel Penalty',
                    'description': 'Penalty applied to parallel coding labor contribution in experiment capacity calculations',
                    'units': 'dimensionless'
                },
                # Manual horizon fitting parameters
                'present_day': {
                    'name': 'Anchor Time',
                    'description': 'Reference time point for manual horizon fitting',
                    'units': 'year'
                },
                'present_horizon': {
                    'name': 'Anchor Horizon',
                    'description': 'Time horizon length at the anchor time (leave empty for auto-fit)',
                    'units': 'minutes'
                },
                'present_doubling_time': {
                    'name': 'Anchor Doubling Time',
                    'description': 'Doubling time parameter at the anchor point (leave empty for auto-fit)',
                    'units': 'progress units'
                },
                'doubling_difficulty_growth_factor': {
                    'name': 'Doubling Difficulty Growth Rate',
                    'description': 'Rate of growth for doubling difficulty (1 - decay rate, leave empty for auto-fit)',
                    'units': 'dimensionless'
                },
                # exp capacity pseudoparameters
                'inf_labor_asymptote': {
                    'name': 'Infinite Labor Asymptote',
                    'description': 'Asymptotic experiment capacity as labor → ∞ (holding compute finite)',
                    'units': 'capacity units'
                },
                'inf_compute_asymptote': {
                    'name': 'Infinite Compute Asymptote',
                    'description': 'Asymptotic experiment capacity as compute → ∞ (holding labor finite)',
                    'units': 'capacity units'
                },
                'labor_anchor_exp_cap': {
                    'name': 'Labor Anchor Experiment Capacity',
                    'description': 'Experiment capacity at the labor anchor used to infer CES parameters',
                    'units': 'capacity units'
                },
                'compute_anchor_exp_cap': {
                    'name': 'Compute Anchor Experiment Capacity',
                    'description': 'Experiment capacity at the compute anchor used to infer CES parameters (direct form, deprecated in UI)',
                    'units': 'capacity units'
                },
                'inv_compute_anchor_exp_cap': {
                    'name': 'Exp. Cap. Reduction with 0.1x compute',
                    'description': 'Inverse form: capacity at 0.1x compute relative to baseline (i.e., reduction factor). The model uses 1 / value internally for the direct anchor.',
                    'units': 'x relative to baseline'
                },
                'median_to_top_taste_multiplier': {
                    'name': 'Median to Top Taste Multiplier',
                    'description': 'Ratio of top percentile researcher taste to median researcher taste',
                    'units': 'ratio'
                },
                'strat_ai_m2b': {
                    'name': 'STRAT-AI M2Bs',
                    'description': 'Number of median-to-bests above SAR for STRAT-AI milestone',
                    'units': 'M2Bs'
                },
                'ted_ai_m2b': {
                    'name': 'TED-AI M2Bs',
                    'description': 'Number of median-to-best jumps above SAR for TED-AI milestone (ASI is always 2 median-to-bests above TED-AI). Median-to-best jumps are measured by SDs within the underlying human distribution (i.e., the population we would get if we sampled humans across many possible Earths). Percentiles are computed using the standard normal CDF—i.e., a position z SDs above the mean corresponds to the Φ(z) percentile.',
                    'units': 'M2Bs'
                }
            }
        }

        return jsonify({
            'success': True,
            'config': parameter_config
        })
        
    except Exception as e:
        logger.error(f"Error getting parameter config: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/upload-data', methods=['POST'])
def upload_data():
    """Upload custom time series data"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        # Read CSV data
        stream = io.StringIO(file.stream.read().decode("UTF8"), newline=None)
        csv_input = csv.DictReader(stream)
        data = list(csv_input)
        
        # Parse data
        time = np.array([float(row['time']) for row in data])
        L_HUMAN = np.array([float(row['L_HUMAN']) for row in data])
        inference_H100e = np.array([float(row['inference_H100e']) for row in data])
        experiment_H100e = np.array([float(row['experiment_H100e']) for row in data])
        training_H100e = np.array([float(row['training_H100e']) for row in data])

        time_series = TimeSeriesData(time, L_HUMAN, inference_H100e, experiment_H100e, training_H100e)

        return jsonify({
            'success': True,
            'data_summary': {
                'time_range': [float(time.min()), float(time.max())],
                'data_points': len(time),
                'preview': time_series_to_dict(time_series)
            }
        })
        
    except Exception as e:
        logger.error(f"Error uploading data: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/select-data', methods=['POST'])
def select_data():
    """Select a built-in dataset by actor and update session time series"""
    try:
        data = request.json or {}
        dataset = data.get('dataset', 'default')

        import os
        base_dir = os.path.dirname(__file__)

        # Map datasets to absolute file paths. The "default" CSV lives in inputs/ at the repo root.
        dataset_to_file = {
            'default': os.path.join(base_dir, '..', 'inputs', 'input_data.csv'),
        }

        if dataset not in dataset_to_file:
            return jsonify({'success': False, 'error': f"Unknown dataset: {dataset}"}), 400

        csv_path = dataset_to_file[dataset]

        if not os.path.exists(csv_path):
            return jsonify({'success': False, 'error': f"Dataset file not found: {os.path.basename(csv_path)}"}), 400

        time_series = load_time_series_from_csv_path(csv_path)

        time_array = time_series.time
        return jsonify({
            'success': True,
            'data_summary': {
                'time_range': [float(time_array.min()), float(time_array.max())],
                'data_points': len(time_array),
                'preview': time_series_to_dict(time_series)
            }
        })
    except Exception as e:
        logger.error(f"Error selecting dataset: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/sampling-config')
def get_sampling_config():
    """Get the sampling configuration for parameter uncertainty"""
    try:
        import yaml
        import os
        config_path = os.path.join(os.path.dirname(__file__), '../config/sampling_config.yaml')
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return jsonify({
            'success': True,
            'config': config
        })
    except Exception as e:
        logger.error(f"Error loading sampling config: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/default-data')
def get_default_data():
    """Get default time series data and parameters"""
    time_series = create_default_time_series()
    params = create_default_parameters()
    
    return jsonify({
        'time_series': time_series_to_dict(time_series),
        'parameters': {
            'rho_coding_labor': params.rho_coding_labor,
            'rho_experiment_capacity': params.rho_experiment_capacity,
            'alpha_experiment_capacity': params.alpha_experiment_capacity,
            'r_software': params.r_software,
            'automation_fraction_at_coding_automation_anchor': params.automation_fraction_at_coding_automation_anchor,
            'coding_labor_normalization': params.coding_labor_normalization,
            'experiment_compute_exponent': params.experiment_compute_exponent,
            'ai_research_taste_at_coding_automation_anchor': params.ai_research_taste_at_coding_automation_anchor,
            'ai_research_taste_slope': params.ai_research_taste_slope,
            'taste_schedule_type': params.taste_schedule_type,
            'progress_at_aa': params.progress_at_aa,
            'ac_time_horizon_minutes': params.ac_time_horizon_minutes,
            'horizon_extrapolation_type': params.horizon_extrapolation_type,
            # AC anchor mode + revenue/ECI-mode parameters
            'ac_anchor_mode': getattr(params, 'ac_anchor_mode', 'time-horizon'),
            'present_revenue': getattr(params, 'present_revenue', None),
            'present_revenue_growth_rate': getattr(params, 'present_revenue_growth_rate', None),
            'revenue_required_for_ac': getattr(params, 'revenue_required_for_ac', None),
            'eci_at_present_day': getattr(params, 'eci_at_present_day', None),
            'eci_growth_rate': getattr(params, 'eci_growth_rate', None),
            'eci_required_for_ac': getattr(params, 'eci_required_for_ac', None),
            # Raw values (None = "derive it") plus the resolved values the model
            # actually used, so the frontend can show what is in force without
            # re-implementing the None conventions.
            'serial_uplift_at_reference_date': getattr(params, 'serial_uplift_at_reference_date', None),
            'serial_uplift_at_reference_date_resolved': resolve_serial_uplift_at_reference_date(params),
            # swe_multiplier_at_present_day is the DERIVED, legacy view of the
            # same belief. Its raw is None unless the caller actually stated it.
            'swe_multiplier_at_present_day': getattr(params, 'swe_multiplier_at_present_day', None),
            'swe_multiplier_at_present_day_resolved': resolve_swe_multiplier_at_present_day(params),
            'excess_uplift_doubling_time': getattr(params, 'excess_uplift_doubling_time', None),
            'uplift_reference_date': getattr(params, 'uplift_reference_date', None),
            'uplift_reference_date_resolved': (
                getattr(params, 'uplift_reference_date', None)
                if getattr(params, 'uplift_reference_date', None) is not None
                else getattr(params, 'present_day', None)
            ),
            # Manual horizon fitting parameters
            'present_day': params.present_day,
            'present_horizon': params.present_horizon,
            'present_doubling_time': params.present_doubling_time,
            'doubling_difficulty_growth_factor': params.doubling_difficulty_growth_factor,
            # Research taste distribution parameter
            'median_to_top_taste_multiplier': params.median_to_top_taste_multiplier,
            # Median-to-best multipliers for general capability milestones
            'strat_ai_m2b': params.strat_ai_m2b,
            'ted_ai_m2b': params.ted_ai_m2b
        }
    })

import os
if __name__ == '__main__':
    # Get port from environment variable for deployment platforms
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    
    app.run(debug=debug, host='0.0.0.0', port=port)
