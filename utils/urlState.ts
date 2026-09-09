import { DEFAULT_PARAMETERS, ParametersType, PRESET_BASELINES, LEGACY_BASELINE_ID, CURRENT_DEFAULT_BASELINE_ID } from '@/constants/parameters';

type CheckboxStateKey =
  | 'enableExperimentAutomation'
  | 'useExperimentThroughputCES'
  | 'enableSoftwareProgress'
  | 'useComputeLaborGrowthSlowdown';

export type PauseTrigger = 'year' | 'milestone';

export interface PauseState {
  pauseEnabled: boolean;
  // How the pause point is chosen: at a fixed calendar year, or when a milestone is reached.
  pauseMode: PauseTrigger;
  pauseYear: number;
  // Milestone key (e.g. 'AC', 'AI2027-SC', 'ASI') used when pauseMode === 'milestone'.
  pauseMilestone: string;
  experimentReduction: number;
  inferenceReduction: number;
  laborReduction: number;
  trainingReduction: number;
}

export const DEFAULT_PAUSE_STATE: PauseState = {
  pauseEnabled: false,
  pauseMode: 'milestone',
  pauseYear: 2029.1,
  pauseMilestone: 'AC',
  experimentReduction: 10,
  inferenceReduction: 10,
  laborReduction: 10,
  trainingReduction: 10,
};

export interface FullUIState {
  parameters: ParametersType;
  presetId: string;
  enableExperimentAutomation: boolean;
  useExperimentThroughputCES: boolean;
  enableSoftwareProgress: boolean;
  useComputeLaborGrowthSlowdown: boolean;
  horizonPercentile: 'p50' | 'p80';
  pause: PauseState;
}

export const DEFAULT_CHECKBOX_STATES: Record<CheckboxStateKey, boolean> = {
  enableExperimentAutomation: true,
  useExperimentThroughputCES: true,
  enableSoftwareProgress: true,
  useComputeLaborGrowthSlowdown: true,
};

const PARAM_ABBREVIATIONS: Record<string, keyof ParametersType> = {
  'tst': 'taste_schedule_type',
  'pdt': 'present_doubling_time',
  'acth': 'ac_time_horizon_minutes',
  'ddgf': 'doubling_difficulty_growth_factor',
  'rcl': 'rho_coding_labor',
  'rec': 'rho_experiment_capacity',
  'aec': 'alpha_experiment_capacity',
  'diec': 'direct_input_exp_cap_ces_params',
  'rsw': 'r_software',
  'spry': 'software_progress_rate_at_reference_year',
  'cln': 'coding_labor_normalization',
  'ece': 'experiment_compute_exponent',
  'cle': 'coding_labor_exponent',
  'afca': 'automation_fraction_at_coding_automation_anchor',
  'ait': 'automation_interp_type',
  'aa': 'automation_anchors',
  'artc': 'ai_research_taste_at_coding_automation_anchor_sd',
  'arts': 'ai_research_taste_slope',
  'paa': 'progress_at_aa',
  'pd': 'present_day',
  'ph': 'present_horizon',
  'het': 'horizon_extrapolation_type',
  'acam': 'ac_anchor_mode',
  'prev': 'present_revenue',
  'prgr': 'present_revenue_growth_rate',
  'rrac': 'revenue_required_for_ac',
  'suard': 'serial_uplift_at_reference_date',
  'eudt': 'excess_uplift_doubling_time',
  'urd': 'uplift_reference_date',
  'suac': 'serial_uplift_at_ac',
  'hrd': 'horizon_reference_date',
  'rvrd': 'revenue_reference_date',
  'ecrd': 'eci_reference_date',
  'eapd': 'eci_at_present_day',
  'egr': 'eci_growth_rate',
  'erac': 'eci_required_for_ac',
  'ila': 'inf_labor_asymptote',
  'ica': 'inf_compute_asymptote',
  'laec': 'labor_anchor_exp_cap',
  'caec': 'compute_anchor_exp_cap',
  'icaec': 'inv_compute_anchor_exp_cap',
  'bgm': 'benchmarks_and_gaps_mode',
  'gy': 'gap_years',
  'caes': 'coding_automation_efficiency_slope',
  'mscl': 'max_serial_coding_labor_multiplier',
  'mttm': 'median_to_top_taste_multiplier',
  'tp': 'top_percentile',
  'sam': 'strat_ai_m2b',
  'tam': 'ted_ai_m2b',
  'ocei': 'optimal_ces_eta_init',
  'ctcgr': 'constant_training_system_performance_growth_rate',
  'sdy': 'slowdown_year',
  'pstcgr': 'post_slowdown_training_system_performance_growth_rate',
  'ala': 'automation_logistic_asymptote',
  'tl': 'taste_limit',
  'tls': 'taste_limit_smoothing',
};

const PARAM_ABBREVIATIONS_REVERSE: Record<string, string> = Object.fromEntries(
  Object.entries(PARAM_ABBREVIATIONS).map(([short, long]) => [long, short])
);

// NOTE: 'eca' (enableCodingAutomation) and 'uvhd' (useVariableHorizonDifficulty) are
// absent here deliberately; links carrying them are ignored by this codec. They are
// still named in legacyLinkRouting.ts's OLD_ERA_KEYS, which is a historical snapshot
// used to route Dec-2025-era links (no 'v' key) to the frozen legacy deployment --
// do NOT prune them from that list.
const CHECKBOX_ABBREVIATIONS: Record<string, string> = {
  'eea': 'enableExperimentAutomation',
  'uetc': 'useExperimentThroughputCES',
  'esp': 'enableSoftwareProgress',
  'uclgs': 'useComputeLaborGrowthSlowdown',
};

const PAUSE_ABBREVIATIONS: Record<string, keyof PauseState> = {
  'pe': 'pauseEnabled',
  'pmo': 'pauseMode',
  'py': 'pauseYear',
  'pms': 'pauseMilestone',
  'per': 'experimentReduction',
  'pir': 'inferenceReduction',
  'plr': 'laborReduction',
  'ptr': 'trainingReduction',
};

const PAUSE_ABBREVIATIONS_REVERSE: Record<string, string> = Object.fromEntries(
  Object.entries(PAUSE_ABBREVIATIONS).map(([short, long]) => [long, short])
);

const CHECKBOX_ABBREVIATIONS_REVERSE: Record<string, string> = Object.fromEntries(
  Object.entries(CHECKBOX_ABBREVIATIONS).map(([short, long]) => [long, short])
);

const sanitizeParameterValue = (
  key: keyof ParametersType,
  value: string | null
): ParametersType[typeof key] | undefined => {
  if (value === null) {
    return undefined;
  }

  const defaultValue = DEFAULT_PARAMETERS[key];

  if (typeof defaultValue === 'number') {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) {
      return parsed as ParametersType[typeof key];
    }
    return undefined;
  }

  if (typeof defaultValue === 'string') {
    return value as ParametersType[typeof key];
  }

  if (typeof defaultValue === 'boolean') {
    if (value === 'true') {
      return true as ParametersType[typeof key];
    }
    if (value === 'false') {
      return false as ParametersType[typeof key];
    }
    return undefined;
  }

  if (defaultValue === null) {
    if (value === 'null') {
      return null as ParametersType[typeof key];
    }
    const parsed = Number(value);
    if (Number.isFinite(parsed)) {
      return parsed as ParametersType[typeof key];
    }
    return value as ParametersType[typeof key];
  }

  return undefined;
};

// Link-format generation marker, minted into every share link.
//
// `v` is bumped ONLY when model semantics change enough that older links can
// no longer be faithfully interpreted by the current model (the frozen-preset
// mechanism already absorbs non-breaking preset/default changes within a
// generation). Absent `v` ⇔ generation 1 ⇔ the Dec-2025 model: the apex
// middleware routes marker-less share links to the frozen legacy deployment
// at v1.aifuturesmodel.com (see utils/legacyLinkRouting.ts). The decoder
// deliberately ignores `v`.
export const LINK_FORMAT_VERSION = '2';

export const encodeFullStateToParams = (state: FullUIState): URLSearchParams => {
  const params = new URLSearchParams();

  const presetId = state.presetId || CURRENT_DEFAULT_BASELINE_ID;
  const baseline = PRESET_BASELINES[presetId] ?? DEFAULT_PARAMETERS;

  // Generation marker first, then base, so URLs are self-describing
  params.set('v', LINK_FORMAT_VERSION);
  params.set('base', presetId);

  // Add all parameters that differ from the preset baseline (using short names)
  (Object.keys(DEFAULT_PARAMETERS) as Array<keyof ParametersType>).forEach((paramKey) => {
    const value = state.parameters[paramKey];
    const baselineValue = baseline[paramKey];

    if (value !== baselineValue) {
      const shortName = PARAM_ABBREVIATIONS_REVERSE[paramKey];
      if (shortName) {
        params.set(shortName, String(value));
      }
    }
  });

  // Add checkbox states that differ from defaults (using short names)
  if (state.enableExperimentAutomation !== DEFAULT_CHECKBOX_STATES.enableExperimentAutomation) {
    params.set(CHECKBOX_ABBREVIATIONS_REVERSE.enableExperimentAutomation, String(state.enableExperimentAutomation));
  }
  if (state.useExperimentThroughputCES !== DEFAULT_CHECKBOX_STATES.useExperimentThroughputCES) {
    params.set(CHECKBOX_ABBREVIATIONS_REVERSE.useExperimentThroughputCES, String(state.useExperimentThroughputCES));
  }
  if (state.enableSoftwareProgress !== DEFAULT_CHECKBOX_STATES.enableSoftwareProgress) {
    params.set(CHECKBOX_ABBREVIATIONS_REVERSE.enableSoftwareProgress, String(state.enableSoftwareProgress));
  }
  if (state.useComputeLaborGrowthSlowdown !== DEFAULT_CHECKBOX_STATES.useComputeLaborGrowthSlowdown) {
    params.set(CHECKBOX_ABBREVIATIONS_REVERSE.useComputeLaborGrowthSlowdown, String(state.useComputeLaborGrowthSlowdown));
  }
  if (state.horizonPercentile !== 'p80') {
    params.set('hp', state.horizonPercentile);
  }

  // Only encode pause params when pause is enabled (keeps URLs clean by default)
  if (state.pause.pauseEnabled) {
    params.set(PAUSE_ABBREVIATIONS_REVERSE.pauseEnabled, 'true');
    if (state.pause.pauseMode !== DEFAULT_PAUSE_STATE.pauseMode) {
      params.set(PAUSE_ABBREVIATIONS_REVERSE.pauseMode, state.pause.pauseMode);
    }
    if (state.pause.pauseYear !== DEFAULT_PAUSE_STATE.pauseYear) {
      params.set(PAUSE_ABBREVIATIONS_REVERSE.pauseYear, String(state.pause.pauseYear));
    }
    if (state.pause.pauseMilestone !== DEFAULT_PAUSE_STATE.pauseMilestone) {
      params.set(PAUSE_ABBREVIATIONS_REVERSE.pauseMilestone, state.pause.pauseMilestone);
    }
    if (state.pause.experimentReduction !== DEFAULT_PAUSE_STATE.experimentReduction) {
      params.set(PAUSE_ABBREVIATIONS_REVERSE.experimentReduction, String(state.pause.experimentReduction));
    }
    if (state.pause.inferenceReduction !== DEFAULT_PAUSE_STATE.inferenceReduction) {
      params.set(PAUSE_ABBREVIATIONS_REVERSE.inferenceReduction, String(state.pause.inferenceReduction));
    }
    if (state.pause.laborReduction !== DEFAULT_PAUSE_STATE.laborReduction) {
      params.set(PAUSE_ABBREVIATIONS_REVERSE.laborReduction, String(state.pause.laborReduction));
    }
    if (state.pause.trainingReduction !== DEFAULT_PAUSE_STATE.trainingReduction) {
      params.set(PAUSE_ABBREVIATIONS_REVERSE.trainingReduction, String(state.pause.trainingReduction));
    }
  }

  return params;
};

export const decodeFullStateFromParams = (searchParams: URLSearchParams): FullUIState => {
  // Determine baseline: explicit `base` param, legacy fallback, or current default
  const baseParam = searchParams.get('base');
  const hasParamDiffs = Array.from(searchParams.keys()).some(
    k => k !== 'base' && (k in PARAM_ABBREVIATIONS || k in CHECKBOX_ABBREVIATIONS)
  );

  let presetId: string;
  if (baseParam && PRESET_BASELINES[baseParam]) {
    presetId = baseParam;
  } else if (!baseParam && hasParamDiffs) {
    // Legacy URL without base param — apply diffs against old baseline
    presetId = LEGACY_BASELINE_ID;
  } else {
    presetId = CURRENT_DEFAULT_BASELINE_ID;
  }

  const baseline = PRESET_BASELINES[presetId] ?? DEFAULT_PARAMETERS;
  const parameters: ParametersType = { ...baseline };

  Object.entries(PARAM_ABBREVIATIONS).forEach(([shortName, paramKey]) => {
    const value = searchParams.get(shortName);

    if (value !== null) {
      const sanitized = sanitizeParameterValue(paramKey, value);
      if (sanitized !== undefined) {
        parameters[paramKey] = sanitized;
      }
    }
  });

  const enableExperimentAutomation = searchParams.has(CHECKBOX_ABBREVIATIONS_REVERSE.enableExperimentAutomation)
    ? searchParams.get(CHECKBOX_ABBREVIATIONS_REVERSE.enableExperimentAutomation) === 'true'
    : DEFAULT_CHECKBOX_STATES.enableExperimentAutomation;

  const useExperimentThroughputCES = searchParams.has(CHECKBOX_ABBREVIATIONS_REVERSE.useExperimentThroughputCES)
    ? searchParams.get(CHECKBOX_ABBREVIATIONS_REVERSE.useExperimentThroughputCES) === 'true'
    : DEFAULT_CHECKBOX_STATES.useExperimentThroughputCES;

  const enableSoftwareProgress = searchParams.has(CHECKBOX_ABBREVIATIONS_REVERSE.enableSoftwareProgress)
    ? searchParams.get(CHECKBOX_ABBREVIATIONS_REVERSE.enableSoftwareProgress) === 'true'
    : DEFAULT_CHECKBOX_STATES.enableSoftwareProgress;

  const useComputeLaborGrowthSlowdown = searchParams.has(CHECKBOX_ABBREVIATIONS_REVERSE.useComputeLaborGrowthSlowdown)
    ? searchParams.get(CHECKBOX_ABBREVIATIONS_REVERSE.useComputeLaborGrowthSlowdown) === 'true'
    : DEFAULT_CHECKBOX_STATES.useComputeLaborGrowthSlowdown;

  const hpRaw = searchParams.get('hp');
  const horizonPercentile: 'p50' | 'p80' = hpRaw === 'p50' ? 'p50' : 'p80';

  // Decode pause state
  const pauseEnabled = searchParams.get(PAUSE_ABBREVIATIONS_REVERSE.pauseEnabled) === 'true';
  const pauseModeRaw = searchParams.get(PAUSE_ABBREVIATIONS_REVERSE.pauseMode);
  const pauseMode: PauseTrigger = pauseModeRaw === 'milestone' ? 'milestone' : 'year';
  const pauseMilestoneRaw = searchParams.get(PAUSE_ABBREVIATIONS_REVERSE.pauseMilestone);
  const pauseMilestone = pauseMilestoneRaw !== null && pauseMilestoneRaw !== ''
    ? pauseMilestoneRaw
    : DEFAULT_PAUSE_STATE.pauseMilestone;
  const pauseYearRaw = searchParams.get(PAUSE_ABBREVIATIONS_REVERSE.pauseYear);
  const pauseYear = pauseYearRaw !== null && Number.isFinite(Number(pauseYearRaw))
    ? Number(pauseYearRaw)
    : DEFAULT_PAUSE_STATE.pauseYear;
  const expRedRaw = searchParams.get(PAUSE_ABBREVIATIONS_REVERSE.experimentReduction);
  const experimentReduction = expRedRaw !== null && Number.isFinite(Number(expRedRaw))
    ? Number(expRedRaw)
    : DEFAULT_PAUSE_STATE.experimentReduction;
  const infRedRaw = searchParams.get(PAUSE_ABBREVIATIONS_REVERSE.inferenceReduction);
  const inferenceReduction = infRedRaw !== null && Number.isFinite(Number(infRedRaw))
    ? Number(infRedRaw)
    : DEFAULT_PAUSE_STATE.inferenceReduction;
  const labRedRaw = searchParams.get(PAUSE_ABBREVIATIONS_REVERSE.laborReduction);
  const laborReduction = labRedRaw !== null && Number.isFinite(Number(labRedRaw))
    ? Number(labRedRaw)
    : DEFAULT_PAUSE_STATE.laborReduction;
  const trainRedRaw = searchParams.get(PAUSE_ABBREVIATIONS_REVERSE.trainingReduction);
  const trainingReduction = trainRedRaw !== null && Number.isFinite(Number(trainRedRaw))
    ? Number(trainRedRaw)
    : DEFAULT_PAUSE_STATE.trainingReduction;

  return {
    parameters,
    presetId,
    enableExperimentAutomation,
    useExperimentThroughputCES,
    enableSoftwareProgress,
    useComputeLaborGrowthSlowdown,
    horizonPercentile,
    pause: { pauseEnabled, pauseMode, pauseYear, pauseMilestone, experimentReduction, inferenceReduction, laborReduction, trainingReduction },
  };
};

// ---------------------------------------------------------------------------
// Covert-race explorer URL state.
//
// Reuses the homepage's PARAM_ABBREVIATIONS short-name scheme for ParametersType
// (so the model-param encoding matches the homepage), but instead of diffing
// against a preset baseline + `base` token, it diffs against a caller-supplied
// seeded baseline (the covert experiment's config defaults). Covert-specific
// state (knobs, silver bullet, sbEnabled) is namespaced with distinct prefixes
// so it can't collide with the model-param short names:
//   - knobs:        `k.<knobKey>`   (e.g. k.years_before_ac)
//   - silver bullet: `sb.<field>`   (e.g. sb.delta_years)
//   - sb enabled:    `sbe`          ('true' | 'false')
// ---------------------------------------------------------------------------

const COVERT_KNOB_PREFIX = 'k.';
const COVERT_SB_PREFIX = 'sb.';
const COVERT_SB_ENABLED_KEY = 'sbe';
const COVERT_HIDDEN_SERIES_KEY = 'hide';
// Trajectory keys the explorer can hide; the URL only ever carries these values.
export const COVERT_TRAJECTORY_KEYS = ['default', 'consortium', 'covert', 'leading'] as const;
export type PlanATrajectoryKey = (typeof COVERT_TRAJECTORY_KEYS)[number];
// Hidden by default (no `hide` param): the covert trajectory. `hide=none` means all visible.
export const COVERT_DEFAULT_HIDDEN: readonly PlanATrajectoryKey[] = ['covert'];
const COVERT_HIDDEN_NONE = 'none';

export interface PlanAUrlInput {
  // Current model params and the seeded baseline they should be diffed against.
  parameters: ParametersType;
  baselineParameters: ParametersType;
  // Covert scenario knobs (Record<string, number|string|boolean|null>) and their seeded baseline.
  knobs: Record<string, number | string | boolean | null>;
  baselineKnobs: Record<string, number | string | boolean | null>;
  // Silver bullet block (numeric fields) and whether it's enabled.
  silverBullet: Record<string, number> | null;
  baselineSilverBullet: Record<string, number> | null;
  sbEnabled: boolean;
  baselineSbEnabled: boolean;
  // Trajectories hidden from the charts (display-only; empty = all visible).
  hiddenSeries: PlanATrajectoryKey[];
}

export const encodePlanAStateToParams = (state: PlanAUrlInput): URLSearchParams => {
  const params = new URLSearchParams();

  // Model params: emit short-name diffs vs the seeded baseline (no `base` token —
  // the covert baseline comes from the experiment config, not a preset).
  (Object.keys(DEFAULT_PARAMETERS) as Array<keyof ParametersType>).forEach((paramKey) => {
    const value = state.parameters[paramKey];
    const baselineValue = state.baselineParameters[paramKey];
    if (value !== baselineValue) {
      const shortName = PARAM_ABBREVIATIONS_REVERSE[paramKey];
      if (shortName) {
        params.set(shortName, String(value));
      }
    }
  });

  // Knobs: emit only values that differ from the seeded knob baseline.
  for (const [key, value] of Object.entries(state.knobs)) {
    if (value === null || value === undefined) continue;
    if (state.baselineKnobs[key] === value) continue;
    params.set(`${COVERT_KNOB_PREFIX}${key}`, String(value));
  }

  // Silver bullet: only when enabled (keeps URLs clean by default).
  if (state.sbEnabled !== state.baselineSbEnabled) {
    params.set(COVERT_SB_ENABLED_KEY, String(state.sbEnabled));
  }
  if (state.sbEnabled && state.silverBullet) {
    const base = state.baselineSilverBullet ?? {};
    for (const [key, value] of Object.entries(state.silverBullet)) {
      if (!Number.isFinite(value)) continue;
      if (base[key] === value) continue;
      params.set(`${COVERT_SB_PREFIX}${key}`, String(value));
    }
  }

  // Hidden trajectories: absent = the default hidden set; `hide=none` = all visible.
  const sortedHidden = [...state.hiddenSeries].sort();
  if (sortedHidden.join(',') !== [...COVERT_DEFAULT_HIDDEN].sort().join(',')) {
    params.set(COVERT_HIDDEN_SERIES_KEY, sortedHidden.join(',') || COVERT_HIDDEN_NONE);
  }

  return params;
};

export interface PlanADecodeOverrides {
  // Sparse overrides parsed from the URL — only keys actually present in the URL.
  parameterOverrides: Partial<ParametersType>;
  knobOverrides: Record<string, number | string | boolean>;
  silverBulletOverrides: Record<string, number>;
  // null when the URL says nothing about sb-enabled; otherwise the explicit value.
  sbEnabled: boolean | null;
  // Trajectories the URL hides from the charts (unknown values dropped; empty = all visible).
  hiddenSeries: PlanATrajectoryKey[];
}

// Decode covert state from URL params into sparse overrides. The caller applies
// these on top of its seeded defaults so the precedence is
//   URL value > config-seeded default > DEFAULT_PARAMETERS.
export const decodePlanAStateFromParams = (searchParams: URLSearchParams): PlanADecodeOverrides => {
  const parameterOverrides: Partial<ParametersType> = {};
  Object.entries(PARAM_ABBREVIATIONS).forEach(([shortName, paramKey]) => {
    const value = searchParams.get(shortName);
    if (value !== null) {
      const sanitized = sanitizeParameterValue(paramKey, value);
      if (sanitized !== undefined) {
        parameterOverrides[paramKey] = sanitized;
      }
    }
  });

  const knobOverrides: Record<string, number | string | boolean> = {};
  const silverBulletOverrides: Record<string, number> = {};
  for (const [rawKey, rawValue] of searchParams.entries()) {
    if (rawKey.startsWith(COVERT_KNOB_PREFIX)) {
      const knobKey = rawKey.slice(COVERT_KNOB_PREFIX.length);
      if (!knobKey) continue;
      const num = Number(rawValue);
      // Numeric strings restore as numbers; booleans (gate knobs) as booleans — never leave
      // "false" as a string (it's truthy); non-numeric (e.g. select modes) stay strings.
      knobOverrides[knobKey] = rawValue !== '' && Number.isFinite(num) ? num
        : rawValue === 'true' ? true : rawValue === 'false' ? false : rawValue;
    } else if (rawKey.startsWith(COVERT_SB_PREFIX)) {
      const sbKey = rawKey.slice(COVERT_SB_PREFIX.length);
      if (!sbKey) continue;
      const num = Number(rawValue);
      if (Number.isFinite(num)) {
        silverBulletOverrides[sbKey] = num;
      }
    }
  }

  const sbeRaw = searchParams.get(COVERT_SB_ENABLED_KEY);
  const sbEnabled = sbeRaw === null ? null : sbeRaw === 'true';

  const hiddenRaw = searchParams.get(COVERT_HIDDEN_SERIES_KEY);
  const hiddenSeries = hiddenRaw === null
    ? [...COVERT_DEFAULT_HIDDEN]
    : hiddenRaw.split(',').filter((v): v is PlanATrajectoryKey =>
        (COVERT_TRAJECTORY_KEYS as readonly string[]).includes(v));

  return { parameterOverrides, knobOverrides, silverBulletOverrides, sbEnabled, hiddenSeries };
};

// Helper for server-side usage with Next.js searchParams object
export const decodeFullStateFromSearchParamsRecord = (
  searchParams: Record<string, string | string[] | undefined>
): FullUIState => {
  const urlSearchParams = new URLSearchParams();
  for (const [key, value] of Object.entries(searchParams)) {
    if (typeof value === 'string') {
      urlSearchParams.set(key, value);
    } else if (Array.isArray(value) && value.length > 0) {
      urlSearchParams.set(key, value[0]);
    }
  }
  return decodeFullStateFromParams(urlSearchParams);
};

