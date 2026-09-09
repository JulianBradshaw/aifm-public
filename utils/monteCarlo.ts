/**
 * Monte Carlo sampling utilities for parameter uncertainty
 */

// Distribution configuration types
export interface DistributionConfig {
  dist: 'fixed' | 'uniform' | 'normal' | 'lognormal' | 'beta' | 'choice' | 'custom_negative_lognormal';
  value?: number | string;
  min?: number;
  max?: number;
  mean?: number;
  sd?: number;
  sigma?: number;
  mu?: number;
  alpha?: number;
  beta?: number;
  values?: (string | number)[];
  p?: number[];
  clip_to_bounds?: boolean;
  // Additional fields for custom distributions
  original_sign?: number;
  shift?: number; // For shifted lognormal distributions
}

export interface SamplingConfig {
  parameters: Record<string, DistributionConfig>;
  seed?: number;
}

// Simple PRNG for reproducible sampling
class SeededRandom {
  private seed: number;

  constructor(seed: number = Date.now()) {
    this.seed = seed;
  }

  random(): number {
    const a = 1664525;
    const c = 1013904223;
    const m = Math.pow(2, 32);
    this.seed = (a * this.seed + c) % m;
    return this.seed / m;
  }

  normal(mean: number = 0, stdDev: number = 1): number {
    // Box-Muller transform
    const u1 = this.random();
    const u2 = this.random();
    const z0 = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    return z0 * stdDev + mean;
  }

  lognormal(mu: number, sigma: number): number {
    return Math.exp(this.normal(mu, sigma));
  }

  beta(alpha: number, beta: number): number {
    // Simple beta distribution using rejection sampling
    const gamma1 = this.gamma(alpha);
    const gamma2 = this.gamma(beta);
    return gamma1 / (gamma1 + gamma2);
  }

  private gamma(shape: number): number {
    // Simple gamma distribution approximation
    if (shape < 1) {
      return this.gamma(shape + 1) * Math.pow(this.random(), 1 / shape);
    }
    
    const d = shape - 1 / 3;
    const c = 1 / Math.sqrt(9 * d);
    
    while (true) {
      let x, v;
      do {
        x = this.normal();
        v = 1 + c * x;
      } while (v <= 0);
      
      v = v * v * v;
      const u = this.random();
      
      if (u < 1 - 0.0331 * x * x * x * x) {
        return d * v;
      }
      
      if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) {
        return d * v;
      }
    }
  }

  choice<T>(values: T[], probabilities?: number[]): T {
    if (!probabilities) {
      const index = Math.floor(this.random() * values.length);
      return values[index];
    }
    
    const cumulative = probabilities.reduce((acc, p, i) => {
      acc[i] = (acc[i - 1] || 0) + p;
      return acc;
    }, [] as number[]);
    
    const r = this.random() * cumulative[cumulative.length - 1];
    const index = cumulative.findIndex(cum => r <= cum);
    return values[index] || values[0];
  }
}


// Convert log scale to actual minutes for ac_time_horizon_minutes
export function convertLogMinutesToMinutes(logValue: number): number {
  return Math.pow(10, logValue);
}

// Convert the sampled parameters to the format expected by the API
export type ParameterRecord = Record<string, number | string | boolean | null | undefined>;

/**
 * `serial_uplift_at_reference_date` and `swe_multiplier_at_present_day` are two
 * containers for ONE quantity (the current AI coding uplift) — serial space at
 * the reference date, parallel space at the present day. The model treats the
 * serial one as the source of truth and derives the other from it, and it
 * REFUSES a request that states both to values which disagree through the
 * conversion (rather than silently picking one).
 *
 * The serial container is the UI's uplift control (the parallel one no longer
 * has a slider), but legacy presets and old URLs still state the parallel
 * container. Whenever the serial container is stated, drop the parallel one
 * from the payload — that is the UI's precedence, not a guess: the serial
 * value is the one the user (or the preset) actually stated.
 */
function applyUpliftContainerPrecedence(converted: ParameterRecord): void {
  const serial = converted.serial_uplift_at_reference_date;
  if (serial != null && converted.swe_multiplier_at_present_day !== undefined) {
    delete converted.swe_multiplier_at_present_day;
  }
}

export function convertParametersToAPIFormat(original: ParameterRecord): ParameterRecord {
  const converted: ParameterRecord = { ...original };

  if (typeof converted.ac_time_horizon_minutes === 'number') {
    converted.ac_time_horizon_minutes = convertLogMinutesToMinutes(converted.ac_time_horizon_minutes);
  }

  applyUpliftContainerPrecedence(converted);

  if (converted.benchmarks_and_gaps_mode !== undefined) {
    const includeGap = converted.benchmarks_and_gaps_mode === true || converted.benchmarks_and_gaps_mode === 'gap';
    converted.include_gap = includeGap ? 'gap' : 'no gap';
    delete converted.benchmarks_and_gaps_mode;
  }

  if (typeof converted.coding_labor_exponent === 'number') {
    converted.parallel_penalty = converted.coding_labor_exponent;
    delete converted.coding_labor_exponent;
  }

  return converted;
}

export function convertSampledParametersToAPIFormat(sampled: ParameterRecord): ParameterRecord {
  // For sampled parameters, we don't want to convert ac_time_horizon_minutes from log to linear
  // because the YAML config already has it in linear (actual minutes)
  const converted: ParameterRecord = { ...sampled };

  applyUpliftContainerPrecedence(converted);

  // Handle benchmarks_and_gaps_mode -> include_gap conversion
  if (converted.benchmarks_and_gaps_mode !== undefined) {
    const includeGap = converted.benchmarks_and_gaps_mode === true || converted.benchmarks_and_gaps_mode === 'gap';
    converted.include_gap = includeGap ? 'gap' : 'no gap';
    delete converted.benchmarks_and_gaps_mode;
  }

  // Handle coding_labor_exponent -> parallel_penalty conversion
  if (typeof converted.coding_labor_exponent === 'number') {
    converted.parallel_penalty = converted.coding_labor_exponent;
    delete converted.coding_labor_exponent;
  }

  return converted;
}

// Create dynamic sampling configuration based on slider values
export function createDynamicSamplingConfig(sliderParams: {
  present_doubling_time: number;
  ac_time_horizon_minutes: number; 
  doubling_difficulty_growth_factor: number;
}): SamplingConfig {
  const { present_doubling_time, ac_time_horizon_minutes, doubling_difficulty_growth_factor } = sliderParams;
  
  return {
    parameters: {
      // Present doubling time with uncertainty scaling based on distance from default
      present_doubling_time: {
        dist: 'lognormal',
        mu: Math.log(present_doubling_time),
        sigma: 0.3 * (1 + Math.abs(present_doubling_time - 0.46) * 0.5),
        min: 0.01,
        max: 2.0,
        clip_to_bounds: true
      },
      
      // SC time horizon (already in log scale) with adaptive uncertainty
      ac_time_horizon_minutes: {
        dist: 'normal',
        mean: ac_time_horizon_minutes,
        sd: 0.3 + 0.2 * Math.abs(ac_time_horizon_minutes - 5.1),
        min: 3.0,
        max: 6.0,
        clip_to_bounds: true
      },
      
      // Doubling difficulty growth rate sampled with adaptive variance
      doubling_difficulty_growth_factor: {
        dist: 'normal',
        mean: doubling_difficulty_growth_factor,
        sd: 0.05 + 0.05 * Math.abs(doubling_difficulty_growth_factor - 1),
        min: 0.5,
        max: 1.5,
        clip_to_bounds: true
      },
      
      // Fixed parameters that don't vary with sliders
      horizon_extrapolation_type: {
        dist: 'choice',
        values: ['exponential', 'decaying doubling time'],
        p: [0.25, 0.75]
      },
      present_horizon: {
        dist: 'fixed',
        value: 15.0
      },
      present_day: {
        dist: 'fixed',
        value: 2025.25
      }
    }
  };
}

// Enhanced sampling function that handles all distribution types
export function sampleFromDynamicDistribution(config: DistributionConfig, rng: SeededRandom): number | string {
  const { dist } = config;

  switch (dist) {
    case 'fixed':
      return config.value!;

    case 'uniform':
      const a = config.min!;
      const b = config.max!;
      return a + rng.random() * (b - a);

    case 'normal':
      const mean = config.mean!;
      const sd = config.sd || config.sigma || 1;
      let value = rng.normal(mean, sd);
      if (config.clip_to_bounds && config.min !== undefined && config.max !== undefined) {
        value = Math.max(config.min, Math.min(config.max, value));
      }
      return value;

    case 'lognormal':
      const mu = config.mu!;
      const sigma = config.sigma!;
      let lnValue = rng.lognormal(mu, sigma);
      // Apply shift if present (for shifted_lognormal distributions)
      if (config.shift !== undefined) {
        lnValue += config.shift;
      }
      if (config.clip_to_bounds && config.min !== undefined && config.max !== undefined) {
        lnValue = Math.max(config.min, Math.min(config.max, lnValue));
      }
      return lnValue;

    case 'beta':
      const alpha = config.alpha!;
      const beta = config.beta!;
      const lo = config.min || 0;
      const hi = config.max || 1;
      const betaValue = rng.beta(alpha, beta);
      return lo + (hi - lo) * betaValue;

    case 'choice':
      return rng.choice(config.values!, config.p);

    case 'custom_negative_lognormal':
      // Sample from lognormal and apply original sign
      const muNeg = config.mu!;
      const sigmaNeg = config.sigma!;
      const originalSign = (config as typeof config & { original_sign?: number }).original_sign || 1;

      let lnValueNeg = rng.lognormal(muNeg, sigmaNeg) * originalSign;
      if (config.clip_to_bounds && config.min !== undefined && config.max !== undefined) {
        lnValueNeg = Math.max(config.min, Math.min(config.max, lnValueNeg));
      }
      return lnValueNeg;

    default:
      throw new Error(`Unknown distribution type: ${dist}`);
  }
}

// Generate parameter set with dynamic distributions
export function sampleDynamicParameterSet(config: SamplingConfig, seed?: number): ParameterRecord {
  const rng = new SeededRandom(seed);
  const sampled: ParameterRecord = {};
  
  for (const [paramName, paramConfig] of Object.entries(config.parameters)) {
    sampled[paramName] = sampleFromDynamicDistribution(paramConfig, rng);
  }
  
  return sampled;
}

// Generate Monte Carlo batch with dynamic configuration
export function generateDynamicMonteCarloBatch(
  allParams: ParameterRecord,
  numSamples: number = 10, // Reduced for faster updates
  baseSeed?: number
): ParameterRecord[] {
  // Extract the 3 main parameters for sampling
  const sliderParams = {
    present_doubling_time: allParams.present_doubling_time as number,
    ac_time_horizon_minutes: allParams.ac_time_horizon_minutes as number,
    doubling_difficulty_growth_factor: allParams.doubling_difficulty_growth_factor as number,
  };
  
  const config = createDynamicSamplingConfig(sliderParams);
  const samples: ParameterRecord[] = [];
  const baseRandomSeed = baseSeed || Date.now();
  
  for (let i = 0; i < numSamples; i++) {
    const seed = baseRandomSeed + i;
    const sampledVariableParams = sampleDynamicParameterSet(config, seed);

    const fullSample = {
      ...allParams,
      ...sampledVariableParams
    };
    
    samples.push(fullSample);
  }
  
  return samples;
}
