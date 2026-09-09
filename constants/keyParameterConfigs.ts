import { formatSCHorizon, formatTimeDuration, yearsToMinutes } from '@/utils/formatting';

/**
 * Google Doc ID for parameter definitions document
 */
export const PARAMETER_DOC_ID = '1ru6Okbxb6XuH18Cz8439sdQJazMV39hNxsWDokh97r0';

/**
 * Build a Google Doc link with an anchor
 */
export const buildDocLink = (anchor: string) => 
  `https://docs.google.com/document/d/${PARAMETER_DOC_ID}/edit#heading=${anchor}`;

/**
 * Shared configuration for key parameters used in both PlaygroundClient and AdvancedSections.
 * This ensures consistent naming, tooltips, and bounds across the UI.
 */

export const KEY_PARAMETER_CONFIGS = {
  present_doubling_time: {
    label: 'Time Horizon Doubling Time',
    description: "How long it takes the coding time horizon to double, at the horizon reference date (considered as a long-term trend)",
    docAnchor: 'h.6s2m9fxy3w25',
    fallbackMin: 0.01,
    fallbackMax: 2.0,
    step: 0.01,
    decimalPlaces: 2,
    formatValue: (years: number) => formatTimeDuration(yearsToMinutes(years)),
    useLogScale: true,
  },
  ac_time_horizon_minutes: {
    label: 'AC Time Horizon Requirement',
    description: "If AIs can do 80% of tasks that take a skilled human this long (plus an optional effective compute gap), then an Automated Coder is reached.",
    docAnchor: 'h.ga01t71wyiv7',
    step: 0.1,
    formatValue: formatSCHorizon,
    useLogScale: true,
  },
  doubling_difficulty_growth_factor: {
    label: 'Doubling Difficulty Growth Factor',
    description:
      'If this parameter equals 0.9, this would mean each time horizon doubling requires an increase of log(effective compute) ' +
      'that is 0.9× what the previous doubling required. To make time horizon exponential in log(effective compute) and thus ' +
      'roughly exponential in time if compute and human labor trends stay the same as in the recent past, set this parameter ' +
      'to 1.00x. To make the trend superexponential, set its value to below 1, or set it to above 1 for subexponential.',
    docAnchor: 'h.qaz4ru2yoi5b',
    customMin: 0.5,
    customMax: 1.5,
    step: 0.01,
    formatValue: (val: number) => `${val.toFixed(2)}x/doubling`,
  },
  ai_research_taste_slope: {
    label: 'Automated Research Taste Slope',
    description: "For each amount of effective compute OOMs crossed in the present year, by how many SDs in the human range is AIs' research taste increased?",
    docAnchor: 'h.y0yy6iou4a4q',
    step: 0.1,
    fallbackMin: 0.1,
    fallbackMax: 10.0,
    decimalPlaces: 1,
    formatValue: (v: number) => `${v.toFixed(1)} SDs/2025-effective-FLOP-growth`,
    useLogScale: true,
  },
  median_to_top_taste_multiplier: {
    label: 'Median-to-Top-Human Research Taste Multiplier',
    description: "Ratio between the top human researcher's research taste and the median human researcher's research taste",
    docAnchor: 'h.k53tugksut89',
    step: 0.1,
    fallbackMin: 1.1,
    fallbackMax: 20.0,
    decimalPlaces: 2,
    formatValue: (v: number) => `${v.toFixed(2)}x`,
    useLogScale: true,
  },
} as const;

/**
 * Mapping of all parameter names to their Google Doc anchors
 */
export const PARAMETER_DOC_ANCHORS: Record<string, string> = {
  // Time horizon parameters
  present_doubling_time: 'h.6s2m9fxy3w25',
  ac_time_horizon_minutes: 'h.ga01t71wyiv7',
  doubling_difficulty_growth_factor: 'h.qaz4ru2yoi5b',
  gap_years: 'h.e1zpd6153zmb',
  
  // Training compute
  constant_training_system_performance_growth_rate: 'h.244pgjpxqbq3',
  slowdown_year: 'h.35jr6dkikh6y',
  post_slowdown_training_system_performance_growth_rate: 'h.5f4shb5zjmtl',
  
  // Coding automation. Some of these are off by one to two because of a weird isue with heading links I'm getting.
  rho_coding_labor: 'h.wqcz0ulbaih6',
  coding_automation_efficiency_slope: 'h.endn64m955cs',
  swe_multiplier_at_present_day: 'h.yxoz7suuy7kc',
  parallel_penalty: 'h.wqcz0ulbaih6',
  coding_labor_exponent: 'h.wqcz0ulbaih6',
  automation_logistic_asymptote: 'h.wqcz0ulbaih6',
  max_serial_coding_labor_multiplier: 'h.wqcz0ulbaih6b',
  automation_fraction_at_coding_automation_anchor: 'h.2mpyhiqhyws5',
  optimal_ces_eta_init: 'h.pmv4sli0dm5m',
  
  // Experiment throughput
  inv_compute_anchor_exp_cap: 'h.uuwga1loww0q',
  inf_labor_asymptote: 'h.m7lqknwrnfi8',
  inf_compute_asymptote: 'h.bmf9ewebxrww',
  
  // Research taste
  ai_research_taste_at_coding_automation_anchor_sd: 'h.kbgu2lmpzrn8',
  median_to_top_taste_multiplier: 'h.k53tugksut89',
  top_percentile: 'h.j8mlcgoxpv1m',
  ai_research_taste_slope: 'h.y0yy6iou4a4q',
  taste_limit: 'h.791kyhzi61xv',
  taste_limit_smoothing: 'h.s67ck98xrba6',
  
  // Software efficiency
  software_progress_rate_at_reference_year: 'h.5zk7qhk1zfpk',
  r_software: 'h.ibptwgw3jl2z',
  
  // AC anchors: Annualized revenue
  present_revenue: 'h.y5tksy1e7utc',
  present_revenue_growth_rate: 'h.qe1537eqca8g',
  revenue_required_for_ac: 'h.v5eytuphirl4',

  // AC anchors: Coding uplift. The serial uplift rows' own heading IDs are duplicated
  // elsewhere in the doc (links would land on the first duplicate), so those two point
  // at the section heading instead.
  serial_uplift_at_reference_date: 'h.dm95fh6gqk3j',
  excess_uplift_doubling_time: 'h.mkq2jl9g8ves',
  serial_uplift_at_ac: 'h.dm95fh6gqk3j',

  // Milestones
  strat_ai_m2b: 'h.orjv6qp5erhp',
  ted_ai_m2b: 'h.263egfydciqe',
  
  // Present day
  present_day: 'h.dsi9htafxp1s',
  present_horizon: 'h.jsnholg4ydnv',
};

export type KeyParameterName = keyof typeof KEY_PARAMETER_CONFIGS;
