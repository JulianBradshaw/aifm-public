"""
Response formatting utilities for API endpoints.
Converts internal model results to frontend-compatible formats.
"""

from typing import Dict, List, Any, Optional, cast
from importlib import import_module

np = cast(Any, import_module('numpy'))

# H100e <-> FLOP-OOM conversion factor (inverse of the load-time conversion).
FLOP_PER_H100E_YEAR = import_module('progress_model.config').FLOP_PER_H100E_YEAR


def _maybe_float(value: Any) -> Optional[float]:
    """Convert value to float if valid, otherwise return None."""
    if value is None:
        return None
    try:
        cast = float(value)
    except (TypeError, ValueError):
        return None
    return cast if np.isfinite(cast) else None


def build_time_series_payload(results: Dict[str, Any]) -> List[Dict[str, Optional[float]]]:
    """
    Build frontend-compatible time series payload from model results.

    Converts model.results dictionary (snake_case arrays) to array of objects
    with camelCase field names expected by the frontend.

    Args:
        results: Dictionary containing model computation results with keys:
            - 'times': array of time points
            - 'progress': cumulative progress values
            - 'effective_compute': effective compute in OOMs
            - 'horizon_lengths': time horizon lengths
            - 'research_stock': research stock values
            - 'automation_fraction': automation fraction at each time
            - 'training_compute': training compute in OOMs
            - 'experiment_capacity': experiment capacity values
            - 'ai_research_taste': AI research taste values
            - 'ai_software_progress_multipliers': software progress multipliers
            - 'serial_coding_labor_multipliers': serial coding labor multipliers vs human-only

    Returns:
        List of dictionaries with camelCase keys for frontend consumption
    """
    times = np.asarray(results.get('times', []), dtype=float)
    progress = np.asarray(results.get('progress', []), dtype=float)
    effective_compute = np.asarray(results.get('effective_compute', []), dtype=float)
    horizon_lengths = np.asarray(results.get('horizon_lengths', []), dtype=float)
    research_stock = np.asarray(results.get('research_stock', []), dtype=float)
    automation_fraction = np.asarray(results.get('automation_fraction', []), dtype=float)
    training_compute = np.asarray(results.get('training_compute', []), dtype=float)  # stock: cumulative hardware FLOP (OOMs)
    training_system_performance = np.asarray(results.get('training_system_performance', []), dtype=float)  # rate: log10(FLOP/yr)
    experiment_capacity = np.asarray(results.get('experiment_capacity', []), dtype=float)
    ai_research_taste = np.asarray(results.get('ai_research_taste', []), dtype=float)
    ai_software_progress_multipliers = np.asarray(results.get('ai_software_progress_multipliers', []), dtype=float)
    ai_sw_progress_mult_ref_present_day = np.asarray(results.get('ai_sw_progress_mult_ref_present_day', []), dtype=float)
    serial_coding_labor_multipliers = np.asarray(results.get('serial_coding_labor_multipliers', []), dtype=float)
    ai_coding_labor_multipliers = np.asarray(results.get('ai_coding_labor_multipliers', []), dtype=float)
    ai_coding_labor_mult_ref_present_day = np.asarray(results.get('ai_coding_labor_mult_ref_present_day', []), dtype=float)

    # Additional metrics
    research_efforts = np.asarray(results.get('research_efforts', []), dtype=float)
    software_progress_rates = np.asarray(results.get('software_progress_rates', []), dtype=float)
    software_efficiency = np.asarray(results.get('software_efficiency', []), dtype=float)
    training_system_performance_growth_rate = np.asarray(results.get('training_system_performance_growth_rate', []), dtype=float)
    progress_rates = np.asarray(results.get('progress_rates', []), dtype=float)  # d(progress)/dt (true)
    etsp_rates = np.asarray(results.get('etsp_rates', []), dtype=float)          # d(ETSP)/dt = sw + TSP
    predicted_revenue = np.asarray(results.get('predicted_revenue', []), dtype=float)  # $/yr, anchored revenue trend
    capability_ecis = np.asarray(results.get('capability_ecis', []), dtype=float)      # ECI points, anchored ECI trend
    capability_eci_rates = np.asarray(results.get('capability_eci_rates', []), dtype=float)  # d(ECI)/dt, points/yr
    excess_serial_uplift = np.asarray(results.get('excess_serial_coding_labor_uplift_ref_present_day', []), dtype=float)  # (M - 1), the model's OWN serial ref-present-day uplift

    # Input time series (may be at different sampling) -> interpolate to model times
    input_ts = results.get('input_time_series', {}) or {}
    input_time = np.asarray(input_ts.get('time', []), dtype=float)
    input_human_labor = np.asarray(input_ts.get('L_HUMAN', []), dtype=float)
    input_inference_compute = np.asarray(input_ts.get('inference_H100e', []), dtype=float)
    input_experiment_compute = np.asarray(input_ts.get('experiment_H100e', []), dtype=float)
    input_training_H100e = np.asarray(input_ts.get('training_H100e', []), dtype=float)
    # Optional display-only series (None/absent when the input CSV lacks the columns)
    _input_world = input_ts.get('world_H100e')
    input_world_H100e = np.asarray(_input_world if _input_world is not None else [], dtype=float)
    _input_company_total = input_ts.get('total_ai_company_H100e')
    input_company_total_H100e = np.asarray(_input_company_total if _input_company_total is not None else [], dtype=float)

    def _interp_to_times(src_x: Any, src_y: Any, dst_x: Any) -> Any:
        if src_x.size > 0 and src_y.size > 0 and dst_x.size > 0:
            try:
                finite_y = np.asarray(src_y, dtype=float)

                if np.all(np.isfinite(finite_y)) and np.all(finite_y > 0):
                    # Mirror the model solver: interpolate exponentially when series stays positive.
                    log_y = np.log(finite_y)
                    left_log = float(log_y[0])
                    right_log = float(log_y[-1])
                    interp_log = np.interp(dst_x, src_x, log_y, left=left_log, right=right_log)
                    return np.exp(interp_log)

                # Clamp to bounds by extending edge values (linear fallback)
                left_val = float(finite_y[0]) if np.isfinite(finite_y[0]) else 0.0
                right_val = float(finite_y[-1]) if np.isfinite(finite_y[-1]) else left_val
                return np.interp(dst_x, src_x, finite_y, left=left_val, right=right_val)
            except Exception:
                pass
        return np.full_like(dst_x, np.nan, dtype=float)

    human_labor_series = _interp_to_times(input_time, input_human_labor, times)
    inference_compute_series = _interp_to_times(input_time, input_inference_compute, times)
    experiment_compute_series = _interp_to_times(input_time, input_experiment_compute, times)

    # Training-run instantaneous H100e: the input training_H100e series is already raw H100e
    # (the instantaneous frontier-run capacity), so interpolate it directly (log space).
    training_compute_h100e = _interp_to_times(input_time, input_training_H100e, times)

    # Total R&D compute (H100e): training + internal inference + experiment, all H100e.
    total_rd_compute_h100e = (
        training_compute_h100e + inference_compute_series + experiment_compute_series
    )

    # World compute and the leading AI company's total compute (both H100e, display only).
    # _interp_to_times yields all-NaN when the source series is absent, so every derived
    # field below degrades to None per-point via _maybe_float.
    world_compute_h100e = _interp_to_times(input_time, input_world_H100e, times)
    company_total_h100e = _interp_to_times(input_time, input_company_total_H100e, times)

    with np.errstate(invalid='ignore', divide='ignore'):
        # Commercial (serving customers) = company total minus R&D, floored at 0.
        commercial_compute_h100e = np.maximum(company_total_h100e - total_rd_compute_h100e, 0.0)
        # Spend shares of the company's total compute, as percentages. The denominator is
        # max(total, R&D) so a user-edited series where R&D exceeds the stated total still
        # yields shares that sum to 100 instead of overflowing.
        spend_denom = np.maximum(company_total_h100e, total_rd_compute_h100e)
        spend_pct_inference = 100.0 * inference_compute_series / spend_denom
        spend_pct_training = 100.0 * training_compute_h100e / spend_denom
        spend_pct_experiment = 100.0 * experiment_compute_series / spend_denom
        spend_pct_commercial = 100.0 * commercial_compute_h100e / spend_denom
        # Cumulative shares, bottom band first, for stacked-area rendering.
        spend_pct_cum_inference = spend_pct_inference
        spend_pct_cum_training = spend_pct_cum_inference + spend_pct_training
        spend_pct_cum_experiment = spend_pct_cum_training + spend_pct_experiment
        spend_pct_cum_commercial = spend_pct_cum_experiment + spend_pct_commercial
        # Share of world compute operated by the frontier AI company, as a percentage.
        company_share_of_world_pct = 100.0 * company_total_h100e / world_compute_h100e
        # Constant-100 top band for the share-of-world stacked chart (the area between
        # the company's share and 100 renders as "rest of world"). Derived from the
        # share so it is NaN -> None exactly where the share itself is unavailable.
        company_share_of_world_pct_total = company_share_of_world_pct * 0.0 + 100.0

    payload: List[Dict[str, Optional[float]]] = []

    for idx, year in enumerate(times):
        payload.append({
            'year': float(year),
            'progress': _maybe_float(progress[idx]) if idx < progress.size else None,
            'effectiveCompute': _maybe_float(effective_compute[idx]) if idx < effective_compute.size else None,
            'horizonLength': _maybe_float(horizon_lengths[idx]) if idx < horizon_lengths.size else None,
            'researchStock': _maybe_float(research_stock[idx]) if idx < research_stock.size else None,
            'automationFraction': _maybe_float(automation_fraction[idx]) if idx < automation_fraction.size else None,
            'trainingCompute': _maybe_float(training_compute[idx]) if idx < training_compute.size else None,
            'trainingSystemPerformance': _maybe_float(training_system_performance[idx]) if idx < training_system_performance.size else None,
            'trainingComputeH100e': _maybe_float(training_compute_h100e[idx]) if idx < training_compute_h100e.size else None,
            'totalRdComputeH100e': _maybe_float(total_rd_compute_h100e[idx]) if idx < total_rd_compute_h100e.size else None,
            'worldComputeH100e': _maybe_float(world_compute_h100e[idx]) if idx < world_compute_h100e.size else None,
            'totalAiCompanyComputeH100e': _maybe_float(company_total_h100e[idx]) if idx < company_total_h100e.size else None,
            'commercialComputeH100e': _maybe_float(commercial_compute_h100e[idx]) if idx < commercial_compute_h100e.size else None,
            'companyShareOfWorldPct': _maybe_float(company_share_of_world_pct[idx]) if idx < company_share_of_world_pct.size else None,
            'companyShareOfWorldPctTotal': _maybe_float(company_share_of_world_pct_total[idx]) if idx < company_share_of_world_pct_total.size else None,
            'spendPctInference': _maybe_float(spend_pct_inference[idx]) if idx < spend_pct_inference.size else None,
            'spendPctTraining': _maybe_float(spend_pct_training[idx]) if idx < spend_pct_training.size else None,
            'spendPctExperiment': _maybe_float(spend_pct_experiment[idx]) if idx < spend_pct_experiment.size else None,
            'spendPctCommercial': _maybe_float(spend_pct_commercial[idx]) if idx < spend_pct_commercial.size else None,
            'spendPctCumInference': _maybe_float(spend_pct_cum_inference[idx]) if idx < spend_pct_cum_inference.size else None,
            'spendPctCumTraining': _maybe_float(spend_pct_cum_training[idx]) if idx < spend_pct_cum_training.size else None,
            'spendPctCumExperiment': _maybe_float(spend_pct_cum_experiment[idx]) if idx < spend_pct_cum_experiment.size else None,
            'spendPctCumCommercial': _maybe_float(spend_pct_cum_commercial[idx]) if idx < spend_pct_cum_commercial.size else None,
            'experimentCapacity': _maybe_float(experiment_capacity[idx]) if idx < experiment_capacity.size else None,
            'aiResearchTaste': _maybe_float(ai_research_taste[idx]) if idx < ai_research_taste.size else None,
            'aiSoftwareProgressMultiplier': _maybe_float(ai_software_progress_multipliers[idx]) if idx < ai_software_progress_multipliers.size else None,
            'aiSwProgressMultRefPresentDay': _maybe_float(ai_sw_progress_mult_ref_present_day[idx]) if idx < ai_sw_progress_mult_ref_present_day.size else None,
            'serialCodingLaborMultiplier': _maybe_float(serial_coding_labor_multipliers[idx]) if idx < serial_coding_labor_multipliers.size else None,
            'aiCodingLaborMultiplier': _maybe_float(ai_coding_labor_multipliers[idx]) if idx < ai_coding_labor_multipliers.size else None,
            'aiCodingLaborMultRefPresentDay': _maybe_float(ai_coding_labor_mult_ref_present_day[idx]) if idx < ai_coding_labor_mult_ref_present_day.size else None,
            # New metrics
            'researchEffort': _maybe_float(research_efforts[idx]) if idx < research_efforts.size else None,
            'softwareProgressRate': _maybe_float(software_progress_rates[idx]) if idx < software_progress_rates.size else None,
            'softwareEfficiency': _maybe_float(software_efficiency[idx]) if idx < software_efficiency.size else None,
            # ETSP = log10 instantaneous effective-compute rate = training compute + software efficiency.
            # ETSP (effective training system performance) = TSP + SE, both RATES (log10 FLOP/yr).
            'etsp': _maybe_float(training_system_performance[idx] + software_efficiency[idx]) if (idx < training_system_performance.size and idx < software_efficiency.size) else None,
            'trainingSystemPerformanceGrowthRate': _maybe_float(training_system_performance_growth_rate[idx]) if idx < training_system_performance_growth_rate.size else None,
            # progressRate = d(progress)/dt (true derivative of logE); etspRate = d(ETSP)/dt = sw + TSP.
            'progressRate': _maybe_float(progress_rates[idx]) if idx < progress_rates.size else None,
            'etspRate': _maybe_float(etsp_rates[idx]) if idx < etsp_rates.size else None,
            'humanLabor': _maybe_float(human_labor_series[idx]) if idx < human_labor_series.size else None,
            'inferenceCompute': _maybe_float(inference_compute_series[idx]) if idx < inference_compute_series.size else None,
            'experimentCompute': _maybe_float(experiment_compute_series[idx]) if idx < experiment_compute_series.size else None,
            'predictedRevenue': _maybe_float(predicted_revenue[idx]) if idx < predicted_revenue.size else None,
            'capabilityEci': _maybe_float(capability_ecis[idx]) if idx < capability_ecis.size else None,
            'capabilityEciRate': _maybe_float(capability_eci_rates[idx]) if idx < capability_eci_rates.size else None,
            'excessSerialCodingLaborUplift': _maybe_float(excess_serial_uplift[idx]) if idx < excess_serial_uplift.size else None,
        })

    return payload
