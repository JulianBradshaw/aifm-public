import {
  ParametersType,
  TRAINING_COMPUTE_REFERENCE_OOMS
} from '@/constants/parameters';

/**
 * Configuration for SC horizon calculations containing all model constants
 * Values are kept in step with Python progress_model/config.py by hand, or passed explicitly
 */
export interface SCHorizonConfig {
  // Core model constants
  baseForSoftwareLOM: number;  // BASE_FOR_SOFTWARE_LOM (typically 10.0)
  trainingComputeReferenceOOMs: number; // TRAINING_COMPUTE_REFERENCE_OOMS (see constants/parameters.ts)

  // Horizon parameters for progress_at_aa calculation
  presentHorizon: number;       // H_0 in minutes
  presentDoublingTime: number;  // T_0 in progress units
  doublingDifficultyGrowthRate: number; // for A_0 = 1 - this
  anchorProgress?: number;       // For shifted form calculations

  // Automation model parameters
  automationAnchors: {
    lowerProgress: number;       // Progress at lower anchor (present day)
    lowerAutomation: number;     // Automation fraction at lower anchor
    upperProgress: number;       // progress_at_aa
    upperAutomation: number;     // automation_fraction_at_coding_automation_anchor
  };

  // Taste distribution parameters
  medianToTopGap: number;       // MEDIAN_TO_TOP_TASTE_MULTIPLIER
  topPercentile: number;        // TOP_PERCENTILE
  aggregateResearchTasteBaseline: number; // Baseline mean

  // AI research taste schedule parameters
  aiResearchTasteSlope: number; // SDs per progress unit
  aiResearchTasteAtSuperhuman_SD: number; // Target SD at SC
  aiResearchTasteMaxSD: number; // Max allowed SD
  aiResearchTasteMin: number;   // Min taste value
  aiResearchTasteMax: number;   // Max taste value
}

/**
 * Computes the horizon length based on progress using the decaying doubling time formula
 * @param progress - The progress value (in effective OOMs)
 * @param uiParameters - The UI parameters containing horizon settings
 * @param anchorProgress - Optional anchor progress for shifted form (from backend)
 */
export function computeHorizonFromProgress(
  progress: number,
  uiParameters: ParametersType,
  anchorProgress?: number
): number {
  const H_0 = uiParameters.present_horizon; // Horizon at present_day (anchor year)
  const difficultyGrowth = uiParameters.doubling_difficulty_growth_factor;
  const A_0 = 1 - difficultyGrowth; // Decay parameter derived from growth rate
  const T_0 = uiParameters.present_doubling_time; // Time constant

  // Handle special case where A_0 is zero (no decay)
  if (A_0 === 0) {
    return H_0 * Math.pow(2, progress / T_0);
  }

  // Safety checks (matching Python model)
  if (T_0 <= 0 || H_0 <= 0 || A_0 >= 1 || A_0 === 0) {
    return H_0; // fallback to base horizon
  }

  // Apply progress shifting when anchor is provided (shifted form from backend)
  // When manual horizon parameters are provided, Python uses: H(p) = H_0 * (1 - A_0 * (p - anchor_progress) / T_0)^exponent
  const progressAdjusted = anchorProgress !== undefined ? progress - anchorProgress : progress;

  // Calculate base term: (1 - A_0 * progressAdjusted / T_0)
  const base_term = Math.max(1 - A_0 * progressAdjusted / T_0, 1e-12);

  // Calculate log denominator: ln(1 - A_0)
  const log_denominator = Math.log(Math.max(1 - A_0, 1e-12));

  // Calculate exponent: ln(2) / ln(1 - A_0)
  const exponent = Math.log(2) / log_denominator;

  // Final horizon calculation: H_0 * (base_term)^exponent
  const result = H_0 * Math.pow(base_term, exponent);

  return (isFinite(result) && result > 0) ? result : H_0;
}

/**
 * Generates data for doubling time and difficulty growth rate charts
 * @param uiParameters - The UI parameters
 * @param startOOM - Starting OOM value
 * @param length - Number of points to generate
 * @param anchorProgress - Optional anchor progress from backend for shifted form
 */
export function generateHorizonData(
  uiParameters: ParametersType,
  startOOM: number,
  length: number = 20,
  anchorProgress?: number
): Array<{ x: number; y: number }> {
  return Array.from({ length }, (_, i) => {
    const ec = startOOM + i;
    const progress = ec - TRAINING_COMPUTE_REFERENCE_OOMS;
    const horizonLength = computeHorizonFromProgress(progress, uiParameters, anchorProgress);
    return {
      x: ec,
      y: horizonLength
    };
  });
}

/**
 * Calculate progress_at_aa from ac_time_horizon_minutes using the inverse horizon equation
 * Follows the logic from progress_model.py lines 2768-2826
 *
 * @param scHorizonMinutes - Target SC time horizon in minutes
 * @param config - Configuration containing horizon parameters
 * @returns progress_at_aa value or null if calculation fails
 */
function calculateProgressAtSC(
  scHorizonMinutes: number,
  config: SCHorizonConfig
): number | null {
  const H_0 = config.presentHorizon;
  const T_0 = config.presentDoublingTime;
  const A_0 = 1.0 - config.doublingDifficultyGrowthRate;
  const anchorProgress = config.anchorProgress;

  // Validation checks (matching Python model lines 2770-2776)
  if (!isFinite(T_0) || T_0 <= 0 || !isFinite(H_0) || H_0 <= 0) {
    return null;
  }
  if (!isFinite(A_0) || Math.abs(A_0) >= 1 || A_0 === 0) {
    return null;
  }
  if (!isFinite(scHorizonMinutes) || scHorizonMinutes <= 0) {
    return null;
  }

  try {
    // Compute ratio term (lines 2777-2788)
    const ratio = scHorizonMinutes / H_0;
    if (!isFinite(ratio) || ratio <= 0) {
      return null;
    }

    const log_ratio = Math.log(1 - A_0) / Math.log(2);
    if (!isFinite(log_ratio)) {
      return null;
    }

    const ratio_term = Math.pow(ratio, log_ratio);
    if (!isFinite(ratio_term)) {
      return null;
    }

    // Apply shifted or unshifted form (lines 2793-2801)
    let calculated_progress_at_aa: number;
    if (anchorProgress !== undefined) {
      // Shifted form: progress = anchor_progress + T_0 * (1 - ratio_term) / A_0
      calculated_progress_at_aa = anchorProgress + T_0 * (1 - ratio_term) / A_0;
    } else {
      // Original form: progress = T_0 * (1 - ratio_term) / A_0
      calculated_progress_at_aa = T_0 * (1 - ratio_term) / A_0;
    }

    // Final validation (lines 2821-2826)
    if (!isFinite(calculated_progress_at_aa)) {
      return null;
    }

    return calculated_progress_at_aa;
  } catch (e) {
    console.warn('Could not calculate progress_at_aa:', e);
    return null;
  }
}

/**
 * TasteDistribution class - models research taste as a log-normal distribution
 * Follows the implementation from progress_model.py TasteDistribution class
 */
class TasteDistribution {
  private mu: number;
  private sigma: number;

  /**
   * Initialize the taste distribution from empirical anchors
   * @param medianToTopGap - Ratio of threshold taste to median taste
   * @param topPercentile - Fraction of researchers classified as "top"
   * @param baselineMean - Company-wide mean taste
   */
  constructor(
    medianToTopGap: number,
    topPercentile: number,
    baselineMean: number
  ) {
    // Step 1: Find z-score for top_percentile using inverse normal CDF approximation
    // For high percentiles (> 0.5), we use a polynomial approximation
    const z_p = this.inverseCDF(topPercentile);

    // Step 2: Compute sigma from the gap ratio
    // sigma = log(median_to_top_gap) / z_p
    this.sigma = Math.log(medianToTopGap) / z_p;

    // Step 3: Compute mu to match baseline mean
    // For LogNormal(mu, sigma^2): mean = exp(mu + sigma^2/2)
    // Therefore: mu = log(baseline_mean) - 0.5 * sigma^2
    this.mu = Math.log(baselineMean) - 0.5 * this.sigma * this.sigma;
  }

  /**
   * Get taste value at a given number of standard deviations
   * Formula: exp(mu + num_sds * sigma)
   * @param numSDs - Number of standard deviations (can be negative)
   * @returns Taste value
   */
  getTasteAtSD(numSDs: number): number {
    return Math.exp(this.mu + numSDs * this.sigma);
  }

  /**
   * Approximation of the inverse standard normal CDF
   * Uses Abramowitz & Stegun approximation for high accuracy
   */
  private inverseCDF(p: number): number {
    if (p <= 0 || p >= 1) {
      throw new Error('Percentile must be between 0 and 1');
    }

    // For percentiles > 0.5, compute using symmetry
    const sign = p > 0.5 ? 1 : -1;
    const q = p > 0.5 ? 1 - p : p;

    // Rational approximation for lower tail
    const t = Math.sqrt(-2 * Math.log(q));

    // Coefficients for Abramowitz & Stegun approximation
    const c0 = 2.515517;
    const c1 = 0.802853;
    const c2 = 0.010328;
    const d1 = 1.432788;
    const d2 = 0.189269;
    const d3 = 0.001308;

    const numerator = c0 + c1 * t + c2 * t * t;
    const denominator = 1 + d1 * t + d2 * t * t + d3 * t * t * t;

    return sign * (t - numerator / denominator);
  }
}

/**
 * Compute automation fraction using linear interpolation between anchors
 * Follows AutomationModel.get_automation_fraction() from progress_model.py
 *
 * @param progress - Progress value
 * @param anchors - Automation anchor points
 * @returns Automation fraction clipped to [0, 1]
 */
function getAutomationFraction(
  progress: number,
  anchors: SCHorizonConfig['automationAnchors']
): number {
  const { lowerProgress, lowerAutomation, upperProgress, upperAutomation } = anchors;

  // Linear interpolation formula:
  // automation = intercept + slope * progress
  // where slope = (aut_2 - aut_1) / (prog_2 - prog_1)
  // and intercept = aut_1 - slope * prog_1
  const slope = (upperAutomation - lowerAutomation) / (upperProgress - lowerProgress);
  const intercept = lowerAutomation - slope * lowerProgress;

  const automation = intercept + slope * progress;

  // Clip to [0, 1] as in Python model
  return Math.max(0.0, Math.min(1.0, automation));
}

/**
 * Create a default SCHorizonConfig from UI parameters
 * Uses model constants synced from Python via JSON config
 */
export function createDefaultSCHorizonConfig(uiParameters: ParametersType): SCHorizonConfig {
  // Model constants are kept in step with Python progress_model/config.py by hand (see constants/parameters.ts)
  const progress_at_aa = 4.0; // Fallback value, will be recalculated

  return {
    baseForSoftwareLOM: 10.0,  // cfg.BASE_FOR_SOFTWARE_LOM
    trainingComputeReferenceOOMs: TRAINING_COMPUTE_REFERENCE_OOMS,
    presentHorizon: uiParameters.present_horizon,
    presentDoublingTime: uiParameters.present_doubling_time,
    doublingDifficultyGrowthRate: uiParameters.doubling_difficulty_growth_factor,
    anchorProgress: undefined,  // Will be provided by backend if using shifted form
    automationAnchors: {
      lowerProgress: 0,  // Present day - simplified assumption
      lowerAutomation: 0.01,  // Small starting automation
      upperProgress: progress_at_aa,
      upperAutomation: uiParameters.automation_fraction_at_coding_automation_anchor || 1.0,
    },
    medianToTopGap: 3.25,  // cfg.MEDIAN_TO_TOP_TASTE_MULTIPLIER
    topPercentile: 0.999,  // cfg.TOP_PERCENTILE
    aggregateResearchTasteBaseline: 1.0,  // cfg.AGGREGATE_RESEARCH_TASTE_BASELINE
    aiResearchTasteSlope: uiParameters.ai_research_taste_slope || 2.5,
    aiResearchTasteAtSuperhuman_SD: uiParameters.ai_research_taste_at_coding_automation_anchor_sd || 0.0,
    aiResearchTasteMaxSD: 100,  // cfg.AI_RESEARCH_TASTE_MAX_SD
    aiResearchTasteMin: 0.0,  // cfg.AI_RESEARCH_TASTE_MIN
    aiResearchTasteMax: 1e30,  // cfg.AI_RESEARCH_TASTE_MAX
  };
}

/**
 * Generates data for SC horizon charts (automation fraction and AI research taste)
 * Rewritten to follow progress_model.py logic exactly without hardcoded constants
 *
 * @param uiParameters - UI parameters containing ac_time_horizon_minutes
 * @param config - Configuration with all model constants from API (or default if omitted)
 * @param startOOM - Starting OOM value
 * @param length - Number of points to generate
 */
export function generateSCHorizonData(
  uiParameters: ParametersType,
  configOrStartOOM: SCHorizonConfig | number,
  startOOMOrLength?: number,
  lengthParam?: number
): {
  effectiveComputes: number[];
  automationFraction: number[];
  aiResearchTaste: number[];
  progressAtSc: number;
  scOom: number;
} {
  // Handle overloaded function signature for backward compatibility
  let config: SCHorizonConfig;
  let startOOM: number;
  let length: number;

  if (typeof configOrStartOOM === 'number') {
    // Old signature: generateSCHorizonData(params, startOOM, length?)
    config = createDefaultSCHorizonConfig(uiParameters);
    startOOM = configOrStartOOM;
    length = startOOMOrLength ?? 20;
  } else {
    // New signature: generateSCHorizonData(params, config, startOOM, length?)
    config = configOrStartOOM;
    startOOM = startOOMOrLength ?? 0;
    length = lengthParam ?? 20;
  }

  // Step 1: Calculate progress_at_aa from ac_time_horizon_minutes
  const scHorizonMinutes = Math.pow(10, uiParameters.ac_time_horizon_minutes);
  const progress_at_aa = calculateProgressAtSC(scHorizonMinutes, config);

  // Use fallback if calculation failed
  if (progress_at_aa === null) {
    console.warn('Failed to calculate progress_at_aa, using fallback value 4.0');
    const fallbackProgress = 4.0;

    return {
      effectiveComputes: Array.from({ length }, (_, i) => startOOM + i),
      automationFraction: Array(length).fill(0.5),
      aiResearchTaste: Array(length).fill(1.0),
      progressAtSc: fallbackProgress,
      scOom: fallbackProgress + config.trainingComputeReferenceOOMs
    };
  }

  const automationAnchors = {
    ...config.automationAnchors,
    upperProgress: progress_at_aa
  };

  config = {
    ...config,
    automationAnchors
  };

  // Step 2: Generate effective compute range (in total OOMs)
  const effectiveComputes = Array.from({ length }, (_, i) => startOOM + i);

  // Step 3: Convert OOMs to progress
  // progress and effectiveCompute are related by a simple offset, which is the
  // convention THIS client-side model uses:
  //     progress = effectiveCompute_OOM - trainingComputeReferenceOOMs
  // Note the backend does not produce that offset: it rebases effective_compute by
  // subtracting software_efficiency at the reference year (so ETSP == TSP there) and
  // leaves progress in raw ODE coordinates. See progress_model/_impl.py's rebase and
  // docs/input-series-provenance.md before assuming the two conventions agree.
  const progress_offset = config.trainingComputeReferenceOOMs;

  // Step 4: Create TasteDistribution for AI research taste calculations
  let tasteDistribution: TasteDistribution;
  try {
    tasteDistribution = new TasteDistribution(
      config.medianToTopGap,
      config.topPercentile,
      config.aggregateResearchTasteBaseline
    );
  } catch (e) {
    console.warn('Error creating TasteDistribution, using fallback:', e);
    // Return simple fallback data
    return {
      effectiveComputes,
      automationFraction: effectiveComputes.map(oom => {
        const progress = oom - progress_offset;
        return getAutomationFraction(progress, automationAnchors);
      }),
      aiResearchTaste: Array(length).fill(1.0),
      progressAtSc: progress_at_aa,
      scOom: progress_at_aa + progress_offset
    };
  }

  // Step 5: Calculate automation fraction for each OOM
  // Using linear interpolation between anchor points (Python AutomationModel)
  const automationFraction = effectiveComputes.map(oom => {
    const progress = oom - progress_offset;

    if (!isFinite(progress)) {
      return automationAnchors.lowerAutomation;
    }

    return getAutomationFraction(progress, automationAnchors);
  });

  // Step 6: Calculate AI research taste for each OOM
  // Following _compute_ai_research_taste_sd_per_progress logic
  const slope = config.aiResearchTasteSlope;
  const target_sd = config.aiResearchTasteAtSuperhuman_SD;

  // Compute offset so curve passes through (progress_at_aa, target_sd)
  const offset = target_sd - slope * progress_at_aa;

  const aiResearchTaste = effectiveComputes.map(oom => {
    const progress = oom - progress_offset;

    if (!isFinite(progress)) {
      return config.aggregateResearchTasteBaseline;
    }

    try {
      // Compute SD at current progress: current_sd = slope * progress + offset
      const current_sd = slope * progress + offset;

      // Clamp SD to maximum allowed value
      const current_sd_clamped = Math.min(current_sd, config.aiResearchTasteMaxSD);

      // Convert SD to taste value using log-normal distribution
      const taste = tasteDistribution.getTasteAtSD(current_sd_clamped);

      // Clamp taste to valid range
      const result = Math.max(
        config.aiResearchTasteMin,
        Math.min(config.aiResearchTasteMax, taste)
      );

      return isFinite(result) ? result : config.aggregateResearchTasteBaseline;
    } catch (e) {
      console.warn('Error computing AI research taste at progress', progress, ':', e);
      return config.aggregateResearchTasteBaseline;
    }
  });

  // Step 7: Convert progress_at_aa back to OOMs for reference
  const sc_oom = progress_at_aa + progress_offset;

  return {
    effectiveComputes,
    automationFraction,
    aiResearchTaste,
    progressAtSc: progress_at_aa,
    scOom: sc_oom
  };
}