'use client';

import { createContext, useContext, useState, type ReactNode } from 'react';

type ParameterName = string | null;

interface ParameterHoverContextValue {
  hoveredParameter: ParameterName;
  setHoveredParameter: (param: ParameterName) => void;
}

const ParameterHoverContext = createContext<ParameterHoverContextValue | undefined>(undefined);

export function useParameterHover() {
  const context = useContext(ParameterHoverContext);
  if (!context) {
    throw new Error('useParameterHover must be used within a ParameterHoverProvider');
  }
  return context;
}

export function ParameterHoverProvider({ children }: { children: ReactNode }) {
  const [hoveredParameter, setHoveredParameter] = useState<ParameterName>(null);

  return (
    <ParameterHoverContext.Provider value={{ hoveredParameter, setHoveredParameter }}>
      {children}
    </ParameterHoverContext.Provider>
  );
}

// Mapping from parameter names to SVG text node labels that should be highlighted
export const PARAMETER_TO_SVG_NODES: Record<string, string[]> = {
  // Main sliders
  present_doubling_time: ['Effective compute', 'Automated coder'],
  ac_time_horizon_minutes: ['Effective compute', 'Automated coder'],
  doubling_difficulty_growth_factor: ['Effective compute', 'Automated coder'],

  // Coding automation parameters
  coding_automation_efficiency_slope: ['Effective compute', 'Coding automation fraction and efficiency', 'Aggregate Labor', 'Aggregate coding labor'],
  rho_coding_labor: ['Human coding labor', 'Automation compute', 'Coding automation fraction and efficiency', 'Aggregate coding labor'],
  max_serial_coding_labor_multiplier: ['Human coding labor', 'Automation compute', 'Coding automation fraction and efficiency', 'Aggregate coding labor'],

  // Experiment throughput parameters
  rho_experiment_capacity: ['Aggregate coding labor', 'Experiment compute', 'Experiment throughput'],
  alpha_experiment_capacity: ['Aggregate coding labor', 'Experiment compute', 'Experiment throughput'],
  experiment_compute_exponent: ['Experiment compute', 'Experiment throughput'],
  coding_labor_exponent: ['Aggregate coding labor', 'Experiment throughput'],
  inf_labor_asymptote: ['Aggregate coding labor', 'Experiment throughput'],
  inf_compute_asymptote: ['Experiment compute', 'Experiment throughput'],
  inv_compute_anchor_exp_cap: ['Experiment compute', 'Experiment throughput'],

  // AI research taste parameters
  ai_research_taste_at_coding_automation_anchor_sd: ['Automated coder', 'Automated experiment selection skill'],
  ai_research_taste_slope: ['Effective compute', 'Automated experiment selection skill'],
  median_to_top_taste_multiplier: ['Effective compute', 'Automated experiment selection skill', 'Human experiment selection skill'],

  // Software progress parameters
  software_progress_rate_at_reference_year: ['Software research effort', 'Software efficiency'],

  // Gap/horizon parameters
  gap_years: ['Effective compute', 'Automated coder'],

  // Extra parameters
  present_day: ['Inputs'], // TODO think more about what this should highlight
  present_horizon: ['Effective compute', 'Automated coder'],
  automation_fraction_at_coding_automation_anchor: ['Effective compute', 'Coding automation fraction and efficiency', 'Automated coder'],

  // Additional parameters not currently highlighted in diagram
  taste_limit: [],
  taste_limit_smoothing: [],
  ted_ai_m2b: [],
  optimal_ces_eta_init: [],
  top_percentile: [],
};

// Rationales for parameter default values
export const PARAMETER_RATIONALES: Record<string, string> = {
  // Input parameters
  constant_training_system_performance_growth_rate: 'Recent compute scaling trends.',
  slowdown_year: 'Investment and fab capacity constraints.',
  post_slowdown_training_system_performance_growth_rate: 'Investment and fab capacity constraints.',

  // Main sliders
  present_doubling_time: 'In between the doubling time of the past long-term trend and the potential recent speedup.',
  ac_time_horizon_minutes: 'We considered what tasks are required to automate coding in AGI projects, then adjusted our estimate upward to make up for only requiring 80% reliability and also to account for a gap between the time horizon benchmark and real-world tasks.',
  doubling_difficulty_growth_factor: 'Set to <1 because once an AIs are more capable than humans at tasks of any length, time horizon should be infinite, implying superexopnential growth before that point. Specific value is based on intuitions about how the difficulty of time horizon doubling changes over time, and is informed by past data.',
  ai_research_taste_slope: 'Based on data regarding how quickly AIs have moved through the human range for a variety of tasks, and preliminary results from P-Zero Research.',
  median_to_top_taste_multiplier: 'Based on surveys of frontier AI researchers and AI experts.',

  // Coding time horizon parameters
  gap_years: 'Intuitions regarding how large the gap between the METR time horizon and the real world tasks might be at tasks that have very high time horizons (which under our conception, determines the size of an effective compute gap)',

  // Coding automation parameters
  coding_automation_efficiency_slope: 'Data regarding how quickly coding automation efficiency has increased over time.',
  rho_coding_labor: 'Intuition about the level of substitutability between coding tasks.',
  max_serial_coding_labor_multiplier: 'Intuitive estimates of the maximum thinking speed and actions relative to humans, and the efficiency of these actions.',
  automation_logistic_asymptote: 'Allows the logistic curve to overshoot 100% before being clipped, enabling smoother transitions in automation fraction.',

  // Experiment throughput parameters
  rho_experiment_capacity: 'Set via the experiment throughput increase from infinite coding labor and the increase from experiment compute.',
  alpha_experiment_capacity: 'Set via the experiment throughput constraints',
  experiment_compute_exponent: 'Set via the experiment throughput constraint',
  coding_labor_exponent: 'Surveys of frontier AI researchers and AI experts as well as our intuitive estimates.',
  inf_labor_asymptote: 'Estimates of what extent extremely fast coding could speed things up despite compute bottlenecks.',
  inf_compute_asymptote: 'Estimating how quickly AGI projects could fully utilize infinite compute, and sanity checking against more marginal constraints that we aren\'t explicitly modeling',
  inv_compute_anchor_exp_cap: 'Surveys of frontier AI researchers and AI experts.',

  // AI research taste parameters
  ai_research_taste_at_coding_automation_anchor_sd: 'Surveys of frontier AI researchers and AI experts, plus a minor intuitive adjustment.',
  taste_limit: 'Data on how superhuman Chess and Go AIs are, plus an intuitive adjustment based on how research taste differs from these cases.',
  taste_limit_smoothing: 'Intuition and sanity checking that it seems to give reasonable results.',

  // General capabilities parameters
  ted_ai_m2b: 'Intuitions regarding the spikiness of AI capability profiles (ASI is always 2 M2Bs above TED-AI).',

  // Revenue anchor parameters
  present_revenue: 'Anthropic\'s most recent reported annualized revenue.',
  present_revenue_growth_rate: 'Based on recent revenue growth, adjusted downward due to expectations of slowing soon.',
  revenue_required_for_ac: 'In between current revenue and the revenue that would be obtained by automating a huge fraction of white-collar labor.',
  revenue_reference_date: 'The date of the most recent reported annualized revenue.',

  // Coding uplift parameters
  serial_uplift_at_reference_date: 'Based on researcher estimates and uplift studies.',
  excess_uplift_doubling_time: 'Based primarily on researcher estimates, and also based on analysis of Anthropic\'s publicly reported data on how much code is written by AIs.',
  uplift_reference_date: 'The date our uplift estimates were elicited.',
  serial_uplift_at_ac: 'Based on AC\'s estimated uplift in the full model.',

  // Effective compute parameters
  software_progress_rate_at_reference_year: 'Analysis of data from the Epoch Capabiltiles Index.',

  // Extra parameters
  present_day: 'The date Opus 4.6 was released.',
  present_horizon: 'Claude Mythos Preview\'s time horizon.',
  automation_fraction_at_coding_automation_anchor: 'With our current modeling decisions, 100% automation is required for Automated Coder.',
  optimal_ces_eta_init: 'The value that, if it were the same for all tasks, would be required to completely replace humans in present day conditions.',
  top_percentile: 'A rough estimate of the amount of people who do experiment seleciton at current AGI projects.',
};

