/**
 * Parameter bounds overrides
 * 
 * These bounds take the highest priority when determining slider min/max values.
 * Priority order: overrides > customMin/customMax props > API bounds > fallbackMin/fallbackMax
 * 
 * Set min/max to override the bounds for a parameter, or leave undefined to use other sources.
 * 
 * Example usage:
 * ```typescript
 * export const PARAMETER_BOUNDS_OVERRIDES: ParameterBoundsOverrides = {
 *   present_doubling_time: { min: 0.05, max: 1.5 },  // Override both bounds
 *   ai_research_taste_slope: { min: 0.5 },          // Override only min
 *   median_to_top_taste_multiplier: undefined,       // No override
 * };
 * ```
 */

import type { ParametersType } from './parameters';

export interface BoundsOverride {
  min?: number;
  max?: number;
}

export type ParameterBoundsOverrides = {
  [K in keyof ParametersType]?: BoundsOverride;
};

export const PARAMETER_BOUNDS_OVERRIDES: ParameterBoundsOverrides = {
  // Main UI Parameters (ProgressChartClient.tsx)
  present_doubling_time: undefined,
  ac_time_horizon_minutes: undefined,
  doubling_difficulty_growth_factor: undefined,
  ai_research_taste_slope: undefined,
  median_to_top_taste_multiplier: undefined,

  // Advanced Parameters (AdvancedSections.tsx)

  // Time Horizon & Progress
  gap_years: { max: 50 },

  // Coding Automation
  swe_multiplier_at_present_day: { max: 100 },
	automation_logistic_asymptote: { min: 1.0001, max: 5 },
  coding_automation_efficiency_slope: undefined,
  rho_coding_labor: undefined,
  max_serial_coding_labor_multiplier: undefined,
  
  // Experiment Throughput Production
  rho_experiment_capacity: undefined,
  alpha_experiment_capacity: undefined,
  experiment_compute_exponent: undefined,
  coding_labor_exponent: undefined,
  inf_labor_asymptote: undefined,
  inf_compute_asymptote: undefined,
  inv_compute_anchor_exp_cap: undefined,
  
  // Experiment Selection Automation
  ai_research_taste_at_coding_automation_anchor_sd: undefined,
  taste_limit: undefined,
  taste_limit_smoothing: undefined,
  
  // General Capabilities
  ted_ai_m2b: undefined,

  // Effective Compute
  software_progress_rate_at_reference_year: undefined,
  
  // Training Compute Growth
  constant_training_system_performance_growth_rate: { min: 0, max: 10 },  // Pre-Slowdown
  post_slowdown_training_system_performance_growth_rate: { min: 0, max: 10 },  // Post-Slowdown

  // Extra Parameters
  present_day: undefined,
  present_horizon: { max: 2000 * 60 },  // 1 work year in minutes
  automation_fraction_at_coding_automation_anchor: undefined,
  optimal_ces_eta_init: undefined,
  top_percentile: undefined,
};

