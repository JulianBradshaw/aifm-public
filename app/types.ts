export interface TooltipProps {
  active?: boolean;
  payload?: Array<{
      value: number;
      color: string;
      dataKey: string;
      name?: string;
  }>;
  label?: number;
}

export interface ChartDataPoint {
    year: number;
    horizonLength: number;
    horizonFormatted: string;
    effectiveCompute?: number | null;
    automationFraction?: number | null;
    trainingCompute?: number | null;
    trainingSystemPerformance?: number | null;
    experimentCapacity?: number | null;
    aiResearchTaste?: number | null;
    aiSoftwareProgressMultiplier?: number | null;
    aiSwProgressMultRefPresentDay?: number | null;
    serialCodingLaborMultiplier?: number | null;
    // New small-chart metrics
    humanLabor?: number | null;
    inferenceCompute?: number | null;
    experimentCompute?: number | null;
    researchEffort?: number | null;
    researchStock?: number | null;
    softwareProgressRate?: number | null;
    softwareEfficiency?: number | null;
    etsp?: number | null;
    trainingSystemPerformanceGrowthRate?: number | null;
    etspRate?: number | null;
    progressRate?: number | null;
    aiCodingLaborMultiplier?: number | null;
    aiCodingLaborMultRefPresentDay?: number | null;
    predictedRevenue?: number | null;
    capabilityEci?: number | null;
    capabilityEciRate?: number | null;
    excessSerialCodingLaborUplift?: number | null;
    // World compute / leading-AI-company compute decomposition (display only)
    worldComputeH100e?: number | null;
    totalAiCompanyComputeH100e?: number | null;
    commercialComputeH100e?: number | null;
    companyShareOfWorldPct?: number | null;
    companyShareOfWorldPctTotal?: number | null;
    spendPctInference?: number | null;
    spendPctTraining?: number | null;
    spendPctExperiment?: number | null;
    spendPctCommercial?: number | null;
    spendPctCumInference?: number | null;
    spendPctCumTraining?: number | null;
    spendPctCumExperiment?: number | null;
    spendPctCumCommercial?: number | null;
    benchmarkHorizon?: number;
    benchmarkLabel?: string;
    // Monte Carlo trajectory data
    [key: string]: number | string | null | undefined; // Allow trajectory_0, trajectory_1, etc.
}

export interface ShapeProps {
    cx: number;
    cy: number;
    onMouseEnter?: (e: React.MouseEvent) => void;
    onMouseLeave?: () => void;
    style?: React.CSSProperties;
}
export interface BenchmarkPoint {
  year: number;
  horizonLength: number;
  label: string;
  company: string;
  releaseDate: string;
}
export interface TooltipProps {
    active?: boolean;
    payload?: Array<{
        value: number;
        color: string;
        dataKey: string;
        name?: string;
    }>;
    label?: number;
}

// ---------------------------------------------------------------------------
// Plan A explorer (/api/plan-a) request + response shapes. The type names say PlanA
// because they are the explorer's API contract; the FIELD names keep the model's
// covert/legal vocabulary, mirroring api/covert_formatters.py.
// Mirrors api/covert_formatters.py: each trajectory family has its OWN time grid,
// so series are returned as parallel-array rows rather than one merged point list.
// ---------------------------------------------------------------------------

/** Knob values are numbers, except a few categorical selects (legal_strategy,
 * legal_mode, compute_mode) which are strings. */
export type PlanAKnobs = Record<string, number | string | boolean | null>;

export interface PlanADetectionGrid {
  size_mw: number;
  rows: Array<[number, number | null, number | null]>;
}

export interface PlanASilverBullet {
  p_mid_pause_to_ted: number;
  p_used_ted_plus_d: number;
  delta_years: number;
  ceiling: number;
}

export interface PlanADefaultsResponse {
  success: boolean;
  knobs: PlanAKnobs;
  detection_grids: PlanADetectionGrid[];
  silver_bullet: PlanASilverBullet | null;
  site_mw_range: [number, number] | null;
  time_range: [number, number];
  // Central model parameters from the covert experiment's model config (model_scenario_branch.yaml),
  // ac_time_horizon_minutes in log10(minutes). Seeds the Key Parameters so the explorer's
  // defaults reproduce the experiment rather than the website's homepage preset.
  model_params?: Record<string, number | string | boolean> | null;
  error?: string;
}

export interface PlanAComputeRequest {
  parameters?: Record<string, number | string | boolean>;
  knobs: PlanAKnobs;
  detection_grids: PlanADetectionGrid[];
  silver_bullet: PlanASilverBullet | null;
}

export interface PlanAScalars {
  covertWins: boolean;
  leadYears: number | null;
  // covert time-to-TED-AI from the pause split, as a ratio to the default world's
  covertSlowdown: number | null;
  covertTtt: number | null;
  legalTtt: number | null;
  legalUsedTtt: number | null;
  tedAiE: number | null;
  legalTargetE: number | null;
  acTime: number | null;
  tedAiTime: number | null;
  pauseYear: number | null;
  anchorProgressRate: number | null;
  alignmentTaxOoms: number | null;
  setupSwGainOoms: number | null;
  totalH100eUsed: number | null;
  totalH100eFrac: number | null;
  swRateSolved: number | null;
  usedSwRateSolved: number | null;
  legalHwRateSolved: number | null;
  // Deal breakdown (all absent/null when the enable_deal_breakdown gate is off).
  dealBreakdownEnabled?: boolean;
  dealBreakdownT?: number | null;   // absolute breakdown year (pause end + N)
  leadingTtt?: number | null;       // leading company reaches bare TED-AI
}

/** A row of parallel values; columns are documented per-field in covert_formatters.py. */
export type PlanASeriesRow = Array<number | null>;

export interface PlanAResponse {
  success: boolean;
  error?: string;
  scalars: PlanAScalars;
  capability: {
    covert: PlanASeriesRow[];          // [t, e]
    legalDiscovered: PlanASeriesRow[];  // [t, e]
    legalUsed: PlanASeriesRow[];        // [t, e]
    leading?: PlanASeriesRow[];         // [t, e] — leading company post-deal-breakdown
  };
  // Safety tax paid (dumb version): [t, OOMs] = (SE_disc + cumulative training compute) − TED-AI EC.
  safetyTaxDumb?: PlanASeriesRow[];
  etsp: {
    covert: PlanASeriesRow[];          // [t, tsp, sw]
    legalDiscovered: PlanASeriesRow[];  // [t, tsp, sw]
    legalUsed: PlanASeriesRow[];        // [t, sw]
    leading?: PlanASeriesRow[];         // [t, tsp, sw] — leading company post-deal-breakdown
  };
  detection: {
    points: PlanASeriesRow[];          // [t, pDetected, sbCdf, sbSurvival, frailtySurvival, combinedSurvival]
    silverBullet: { C: number | null; k: number | null; tMid: number | null } | null;
    sizeMw: number | null;
    nSites: number | null;
    effSiteMw: number | null;
  };
  // Per-metric covert series for the mini-charts (overlaid on the default project).
  // Columns: [t, effectiveCompute, automationFraction, etsp, softwareEfficiency,
  //           trainingSystemPerformance, researchEffort, aiSwProgressMultRefPresentDay,
  //           researchEffortIndigenous, swRateIndigenous,
  //           aiCodingLaborMultRefPresentDay, aiResearchTaste].
  // cols 8-9 are the project's OWN-R&D research effort / software-progress rate (theft + boost
  // stripped; `researchEffort` col 6 is the REALIZED rate incl. stolen growth). cols 10-11 are
  // capability-indexed (pure functions of E), mapped through the reference run like uplift.
  metrics: {
    covert: PlanASeriesRow[];
    // Same columns; legal pays no theft so realized == indigenous. Under a deal breakdown
    // this series is truncated at the breakdown (the consortium ceases to exist).
    legalUsed: PlanASeriesRow[];
    // Legal DISCOVERED track (full research output, target TED-AI + tax) — same columns
    // and the same no-theft/truncation conventions as legalUsed.
    legalDiscovered: PlanASeriesRow[];
    // Leading company post-deal-breakdown (same columns; no theft) — rendered olive.
    leading?: PlanASeriesRow[];
  };
  // Input compute series each trajectory runs on. Columns: [year, training_H100e, total_rd_H100e].
  // `default` drives the reference/default project; `legal` is the Plan-A-relativized series
  // that drives the legal + covert projects.
  inputs: {
    default: PlanASeriesRow[];
    // Under a deal breakdown, truncated at the breakdown (the consortium ceases to exist).
    legal: PlanASeriesRow[];
    covert?: PlanASeriesRow[];   // flat covert datacenter compute over [pause, t_end]
    leading?: PlanASeriesRow[];  // leading-co post-breakdown compute — rendered olive
  };
  // Cumulative-OOMs-since-pause stacked-area decompositions.
  // covert: [t, hardware, indigenous, stolen];
  // legal: [t, hardware, windfall, mandatoryMin, used, tax].
  breakdown: {
    covert: PlanASeriesRow[];
    legal: PlanASeriesRow[];
  };
  // Dense, E-sorted (effectiveCompute, uplift) pairs from the reference run, used client-side
  // to map any trajectory's E → AI-software-R&D uplift for the headline chart.
  upliftMap?: PlanASeriesRow[];
  // Milestone anchors from the sim's own reference run (E in the same rebased space as
  // `capability`), used to place per-trajectory milestone dots self-consistently.
  milestones?: Record<string, { time?: number | null; e?: number | null }>;
}
