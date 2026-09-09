// Sampling utilities for parameter uncertainty

// Seeded random number generator (mulberry32)
export class SeededRandom {
  private state: number;

  constructor(seed: number) {
    this.state = seed;
  }

  // Returns a random number between 0 and 1
  next(): number {
    this.state |= 0;
    this.state = (this.state + 0x6D2B79F5) | 0;
    let t = Math.imul(this.state ^ (this.state >>> 15), 1 | this.state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  // Box-Muller transform for normal distribution
  nextNormal(mean: number = 0, std: number = 1): number {
    const u1 = this.next();
    const u2 = this.next();
    const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    return mean + std * z;
  }

  // Gamma distribution using Marsaglia and Tsang's method
  nextGamma(shape: number): number {
    if (shape < 1) {
      return this.nextGamma(shape + 1) * Math.pow(this.next(), 1 / shape);
    }

    const d = shape - 1/3;
    const c = 1 / Math.sqrt(9 * d);

    while (true) {
      let x: number, v: number;
      do {
        x = this.nextNormal();
        v = 1 + c * x;
      } while (v <= 0);

      v = v * v * v;
      const u = this.next();

      if (u < 1 - 0.0331 * (x * x) * (x * x)) {
        return d * v;
      }

      if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) {
        return d * v;
      }
    }
  }

  // Beta distribution
  nextBeta(alpha: number, beta: number): number {
    const gammaA = this.nextGamma(alpha);
    const gammaB = this.nextGamma(beta);
    return gammaA / (gammaA + gammaB);
  }
}

// Global RNG instance (can be replaced with seeded version)
let currentRng: SeededRandom | null = null;

// Get random value - uses seeded RNG if set, otherwise Math.random
function getRandom(): number {
  return currentRng ? currentRng.next() : Math.random();
}

function getRandomNormal(mean: number = 0, std: number = 1): number {
  if (currentRng) {
    return currentRng.nextNormal(mean, std);
  }
  const u1 = Math.random();
  const u2 = Math.random();
  const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  return mean + std * z;
}

function getRandomGamma(shape: number): number {
  if (currentRng) {
    return currentRng.nextGamma(shape);
  }
  return randomGamma(shape);
}

function getRandomBeta(alpha: number, beta: number): number {
  if (currentRng) {
    return currentRng.nextBeta(alpha, beta);
  }
  return randomBeta(alpha, beta);
}

// Set the seeded RNG for subsequent sampling operations
export function setSamplingRng(rng: SeededRandom | null): void {
  currentRng = rng;
}

// Distribution type definitions
export type DistributionConfig = 
  | { dist: 'fixed'; value: number | string }
  | { dist: 'normal'; ci80: [number, number]; min?: number; max?: number; clip_to_bounds?: boolean }
  | { dist: 'lognormal'; ci80: [number, number]; min?: number; max?: number; clip_to_bounds?: boolean }
  | { dist: 'shifted_lognormal'; ci80: [number, number]; shift: number; min?: number; max?: number; clip_to_bounds?: boolean }
  | { dist: 'uniform'; min: number; max: number; clip_to_bounds?: boolean }
  | { dist: 'beta'; alpha: number; beta: number; min?: number; max?: number; clip_to_bounds?: boolean }
  | { dist: 'choice'; values: (string | number)[]; p: number[] };

export interface CorrelationConfig {
  parameters: string[];
  correlation_matrix: number[][];
  correlation_type?: 'spearman' | 'pearson';
  latent_correlation_matrix?: number[][];
}

export interface SamplingConfig {
  parameters: Record<string, DistributionConfig>;
  time_series_parameters?: Record<string, DistributionConfig>;
  correlation_matrix?: CorrelationConfig;
  num_rollouts?: number;
}

// Standard normal CDF inverse (approximation using Rational approximation)
function normalCdfInverse(p: number): number {
  if (p <= 0) return -Infinity;
  if (p >= 1) return Infinity;
  if (p === 0.5) return 0;

  const a = [
    -3.969683028665376e+01,
    2.209460984245205e+02,
    -2.759285104469687e+02,
    1.383577518672690e+02,
    -3.066479806614716e+01,
    2.506628277459239e+00
  ];
  const b = [
    -5.447609879822406e+01,
    1.615858368580409e+02,
    -1.556989798598866e+02,
    6.680131188771972e+01,
    -1.328068155288572e+01
  ];
  const c = [
    -7.784894002430293e-03,
    -3.223964580411365e-01,
    -2.400758277161838e+00,
    -2.549732539343734e+00,
    4.374664141464968e+00,
    2.938163982698783e+00
  ];
  const d = [
    7.784695709041462e-03,
    3.224671290700398e-01,
    2.445134137142996e+00,
    3.754408661907416e+00
  ];

  const pLow = 0.02425;
  const pHigh = 1 - pLow;

  let q: number, r: number;

  if (p < pLow) {
    q = Math.sqrt(-2 * Math.log(p));
    return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
           ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1);
  } else if (p <= pHigh) {
    q = p - 0.5;
    r = q * q;
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q /
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1);
  } else {
    q = Math.sqrt(-2 * Math.log(1 - p));
    return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
            ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1);
  }
}

// Box-Muller transform for generating normal random numbers
function randomNormal(mean: number = 0, std: number = 1): number {
  return getRandomNormal(mean, std);
}

// Beta distribution sampling using rejection sampling
function randomBeta(alpha: number, beta: number): number {
  if (currentRng) {
    return currentRng.nextBeta(alpha, beta);
  }
  // Use gamma distribution relationship: Beta(a,b) = Gamma(a) / (Gamma(a) + Gamma(b))
  const gammaA = randomGammaInternal(alpha);
  const gammaB = randomGammaInternal(beta);
  return gammaA / (gammaA + gammaB);
}

// Gamma distribution using Marsaglia and Tsang's method
function randomGamma(shape: number): number {
  if (currentRng) {
    return currentRng.nextGamma(shape);
  }
  return randomGammaInternal(shape);
}

// Internal gamma without seeding check (used by randomBeta when not seeded)
function randomGammaInternal(shape: number): number {
  if (shape < 1) {
    // For shape < 1, use the relation: Gamma(a) = Gamma(a+1) * U^(1/a)
    return randomGammaInternal(shape + 1) * Math.pow(getRandom(), 1 / shape);
  }

  const d = shape - 1/3;
  const c = 1 / Math.sqrt(9 * d);

  while (true) {
    let x: number, v: number;
    do {
      x = getRandomNormal();
      v = 1 + c * x;
    } while (v <= 0);

    v = v * v * v;
    const u = getRandom();

    if (u < 1 - 0.0331 * (x * x) * (x * x)) {
      return d * v;
    }

    if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) {
      return d * v;
    }
  }
}

// Convert CI80 to normal distribution parameters
function ci80ToNormalParams(ci80: [number, number]): { mean: number; std: number } {
  const [low, high] = ci80;
  // CI80 corresponds to 10th and 90th percentiles
  // For normal distribution: mean ± 1.28155 * std
  const z = 1.28155; // normalCdfInverse(0.9)
  const mean = (low + high) / 2;
  const std = (high - low) / (2 * z);
  return { mean, std };
}

// Convert CI80 to lognormal distribution parameters
function ci80ToLognormalParams(ci80: [number, number]): { mu: number; sigma: number } {
  const [low, high] = ci80;
  // For lognormal: log(X) ~ Normal(mu, sigma)
  // CI80 means: exp(mu - 1.28155*sigma) = low, exp(mu + 1.28155*sigma) = high
  const z = 1.28155;
  const logLow = Math.log(low);
  const logHigh = Math.log(high);
  const mu = (logLow + logHigh) / 2;
  const sigma = (logHigh - logLow) / (2 * z);
  return { mu, sigma };
}

// Standard normal CDF using error function approximation
function normalCDF(x: number): number {
  // Using the relationship: Phi(x) = 0.5 * (1 + erf(x / sqrt(2)))
  return 0.5 * (1 + erf(x / Math.sqrt(2)));
}

// Error function approximation (Abramowitz and Stegun)
function erf(x: number): number {
  const sign = x >= 0 ? 1 : -1;
  x = Math.abs(x);
  
  const a1 =  0.254829592;
  const a2 = -0.284496736;
  const a3 =  1.421413741;
  const a4 = -1.453152027;
  const a5 =  1.061405429;
  const p  =  0.3275911;
  
  const t = 1.0 / (1.0 + p * x);
  const y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);
  
  return sign * y;
}

// Inverse error function (using Newton-Raphson refinement)
function erfinv(x: number): number {
  if (x === 0) return 0;
  if (x >= 1) return Infinity;
  if (x <= -1) return -Infinity;
  
  const sign = x < 0 ? -1 : 1;
  x = Math.abs(x);
  
  // Initial approximation using Winitzki's approximation
  const a = 0.147;
  const ln1mx2 = Math.log(1 - x * x);
  const term1 = 2 / (Math.PI * a) + ln1mx2 / 2;
  const term2 = ln1mx2 / a;
  
  let result = sign * Math.sqrt(Math.sqrt(term1 * term1 - term2) - term1);
  
  // Newton-Raphson refinement
  for (let i = 0; i < 3; i++) {
    const erfResult = erf(result);
    const derivative = (2 / Math.sqrt(Math.PI)) * Math.exp(-result * result);
    result = result - (erfResult - sign * x) / derivative;
  }
  
  return result;
}

// Cholesky decomposition for positive definite matrices
function choleskyDecomposition(matrix: number[][]): number[][] {
  const n = matrix.length;
  const L: number[][] = Array.from({ length: n }, () => Array(n).fill(0));
  
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let sum = 0;
      for (let k = 0; k < j; k++) {
        sum += L[i][k] * L[j][k];
      }
      
      if (i === j) {
        const diag = matrix[i][i] - sum;
        if (diag <= 0) {
          // Matrix is not positive definite; add small jitter
          L[i][j] = Math.sqrt(Math.max(diag + 1e-8, 1e-10));
        } else {
          L[i][j] = Math.sqrt(diag);
        }
      } else {
        L[i][j] = (matrix[i][j] - sum) / L[j][j];
      }
    }
  }
  
  return L;
}

// Convert Spearman rank correlation to latent Gaussian Pearson correlation
// Using the formula: rho_latent = 2 * sin(pi * rho_spearman / 6)
function spearmanToLatentGaussian(spearmanMatrix: number[][]): number[][] {
  const n = spearmanMatrix.length;
  const latent: number[][] = Array.from({ length: n }, () => Array(n).fill(0));
  
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      if (i === j) {
        latent[i][j] = 1.0;
      } else {
        const rhoS = spearmanMatrix[i][j];
        // Clamp to valid range
        let rhoLatent = 2.0 * Math.sin(Math.PI * rhoS / 6.0);
        rhoLatent = Math.max(-0.9999, Math.min(0.9999, rhoLatent));
        latent[i][j] = rhoLatent;
      }
    }
  }
  
  return latent;
}

// Sample from multivariate normal with given correlation matrix
function sampleMultivariateNormal(L: number[][]): number[] {
  const n = L.length;
  // Generate independent standard normal samples
  const z: number[] = [];
  for (let i = 0; i < n; i++) {
    z.push(randomNormal(0, 1));
  }
  
  // Multiply by Cholesky factor to get correlated samples
  const result: number[] = [];
  for (let i = 0; i < n; i++) {
    let sum = 0;
    for (let j = 0; j <= i; j++) {
      sum += L[i][j] * z[j];
    }
    result.push(sum);
  }
  
  return result;
}

// Inverse beta CDF using numerical approximation
function betaInverseCDF(p: number, alpha: number, beta: number): number {
  // Use bisection method for inverse beta CDF
  if (p <= 0) return 0;
  if (p >= 1) return 1;
  
  // Initial guess
  let low = 0;
  let high = 1;
  const tol = 1e-10;
  const maxIter = 100;
  
  for (let iter = 0; iter < maxIter; iter++) {
    const mid = (low + high) / 2;
    const cdfVal = incompleteBeta(mid, alpha, beta);
    
    if (Math.abs(cdfVal - p) < tol) {
      return mid;
    }
    
    if (cdfVal < p) {
      low = mid;
    } else {
      high = mid;
    }
  }
  
  return (low + high) / 2;
}

// Incomplete beta function using continued fraction
function incompleteBeta(x: number, a: number, b: number): number {
  if (x === 0) return 0;
  if (x === 1) return 1;
  
  // Use the regularized incomplete beta function
  const bt = x === 0 || x === 1 ? 0 :
    Math.exp(lgamma(a + b) - lgamma(a) - lgamma(b) + a * Math.log(x) + b * Math.log(1 - x));
  
  if (x < (a + 1) / (a + b + 2)) {
    return bt * betaContinuedFraction(x, a, b) / a;
  } else {
    return 1 - bt * betaContinuedFraction(1 - x, b, a) / b;
  }
}

// Log gamma function approximation
function lgamma(x: number): number {
  const cof = [
    76.18009172947146, -86.50532032941677, 24.01409824083091,
    -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5
  ];
  
  let y = x;
  let tmp = x + 5.5;
  tmp -= (x + 0.5) * Math.log(tmp);
  let ser = 1.000000000190015;
  
  for (let j = 0; j < 6; j++) {
    ser += cof[j] / ++y;
  }
  
  return -tmp + Math.log(2.5066282746310005 * ser / x);
}

// Continued fraction for incomplete beta
function betaContinuedFraction(x: number, a: number, b: number): number {
  const maxIter = 200;
  const eps = 1e-14;
  
  const qab = a + b;
  const qap = a + 1;
  const qam = a - 1;
  
  let c = 1;
  let d = 1 - qab * x / qap;
  if (Math.abs(d) < 1e-30) d = 1e-30;
  d = 1 / d;
  let h = d;
  
  for (let m = 1; m <= maxIter; m++) {
    const m2 = 2 * m;
    let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
    d = 1 + aa * d;
    if (Math.abs(d) < 1e-30) d = 1e-30;
    c = 1 + aa / c;
    if (Math.abs(c) < 1e-30) c = 1e-30;
    d = 1 / d;
    h *= d * c;
    
    aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
    d = 1 + aa * d;
    if (Math.abs(d) < 1e-30) d = 1e-30;
    c = 1 + aa / c;
    if (Math.abs(c) < 1e-30) c = 1e-30;
    d = 1 / d;
    const del = d * c;
    h *= del;
    
    if (Math.abs(del - 1) < eps) break;
  }
  
  return h;
}

// Sample from a distribution config
export function sampleFromDistribution(config: DistributionConfig): number | string {
  switch (config.dist) {
    case 'fixed':
      return config.value;

    case 'normal': {
      const { mean, std } = ci80ToNormalParams(config.ci80);
      let value = randomNormal(mean, std);
      if (config.min !== undefined) value = Math.max(value, config.min);
      if (config.max !== undefined) value = Math.min(value, config.max);
      return value;
    }

    case 'lognormal': {
      const { mu, sigma } = ci80ToLognormalParams(config.ci80);
      let value = Math.exp(randomNormal(mu, sigma));
      if (config.min !== undefined) value = Math.max(value, config.min);
      if (config.max !== undefined) value = Math.min(value, config.max);
      return value;
    }

    case 'shifted_lognormal': {
      // For shifted lognormal: X = shift + Y where Y ~ Lognormal
      // The CI80 specifies the 10th and 90th percentiles of Y (the lognormal part BEFORE shift)
      // Then we add the shift to get the final value
      const shift = config.shift;
      const { mu, sigma } = ci80ToLognormalParams(config.ci80);
      const normalSample = randomNormal(mu, sigma);
      const lognormalPart = Math.exp(normalSample);
      let value = lognormalPart + shift;
      if (config.min !== undefined) value = Math.max(value, config.min);
      if (config.max !== undefined) value = Math.min(value, config.max);
      return value;
    }

    case 'uniform': {
      const value = config.min + getRandom() * (config.max - config.min);
      return value;
    }

    case 'beta': {
      const min = config.min ?? 0;
      const max = config.max ?? 1;
      const betaSample = randomBeta(config.alpha, config.beta);
      return min + betaSample * (max - min);
    }

    case 'choice': {
      const r = getRandom();
      let cumulative = 0;
      for (let i = 0; i < config.values.length; i++) {
        cumulative += config.p[i];
        if (r < cumulative) {
          return config.values[i];
        }
      }
      return config.values[config.values.length - 1];
    }

    default:
      throw new Error(`Unknown distribution type: ${(config as DistributionConfig).dist}`);
  }
}

// Sample from a distribution using a uniform quantile (for correlated sampling via Gaussian copula)
export function sampleFromDistributionWithQuantile(config: DistributionConfig, quantile: number): number | string {
  // Clamp quantile to valid range
  quantile = Math.max(1e-10, Math.min(1 - 1e-10, quantile));
  
  switch (config.dist) {
    case 'fixed':
      return config.value;

    case 'normal': {
      const { mean, std } = ci80ToNormalParams(config.ci80);
      // Inverse normal CDF: mean + std * sqrt(2) * erfinv(2*q - 1)
      let value = mean + std * Math.sqrt(2) * erfinv(2 * quantile - 1);
      if (config.min !== undefined) value = Math.max(value, config.min);
      if (config.max !== undefined) value = Math.min(value, config.max);
      return value;
    }

    case 'lognormal': {
      const { mu, sigma } = ci80ToLognormalParams(config.ci80);
      // Inverse lognormal CDF: exp(mu + sigma * sqrt(2) * erfinv(2*q - 1))
      let value = Math.exp(mu + sigma * Math.sqrt(2) * erfinv(2 * quantile - 1));
      if (config.min !== undefined) value = Math.max(value, config.min);
      if (config.max !== undefined) value = Math.min(value, config.max);
      return value;
    }

    case 'shifted_lognormal': {
      const shift = config.shift;
      const { mu, sigma } = ci80ToLognormalParams(config.ci80);
      // Sample the lognormal part, then add shift
      const lognormalPart = Math.exp(mu + sigma * Math.sqrt(2) * erfinv(2 * quantile - 1));
      let value = lognormalPart + shift;
      if (config.min !== undefined) value = Math.max(value, config.min);
      if (config.max !== undefined) value = Math.min(value, config.max);
      return value;
    }

    case 'uniform': {
      // Inverse uniform CDF: min + quantile * (max - min)
      return config.min + quantile * (config.max - config.min);
    }

    case 'beta': {
      const min = config.min ?? 0;
      const max = config.max ?? 1;
      // Use inverse beta CDF
      const x01 = betaInverseCDF(quantile, config.alpha, config.beta);
      return min + x01 * (max - min);
    }

    case 'choice': {
      // Weighted choice using quantile
      const p = config.p;
      if (!p) {
        // Uniform choice
        const idx = Math.min(Math.floor(quantile * config.values.length), config.values.length - 1);
        return config.values[idx];
      } else {
        // Weighted choice - find cumulative sum
        let cumsum = 0;
        for (let i = 0; i < p.length; i++) {
          cumsum += p[i];
          if (quantile < cumsum) {
            return config.values[i];
          }
        }
        return config.values[config.values.length - 1];
      }
    }

    default:
      throw new Error(`Unknown distribution type: ${(config as DistributionConfig).dist}`);
  }
}

// Cached Cholesky decomposition for correlation matrix
let cachedCholeskyL: number[][] | null = null;
let cachedCorrelatedParams: string[] | null = null;

// Initialize correlation sampling (call once when config changes)
export function initializeCorrelationSampling(correlationConfig: CorrelationConfig | undefined): void {
  if (!correlationConfig) {
    cachedCholeskyL = null;
    cachedCorrelatedParams = null;
    return;
  }
  
  const corrMatrix = correlationConfig.correlation_matrix;
  const corrType = correlationConfig.correlation_type ?? 'spearman';
  
  // Convert to latent Gaussian correlation if Spearman
  let latentMatrix: number[][];
  if (correlationConfig.latent_correlation_matrix) {
    // Use pre-computed latent correlation matrix if available
    latentMatrix = correlationConfig.latent_correlation_matrix;
  } else if (corrType === 'spearman') {
    latentMatrix = spearmanToLatentGaussian(corrMatrix);
  } else {
    latentMatrix = corrMatrix;
  }
  
  // Compute Cholesky decomposition
  cachedCholeskyL = choleskyDecomposition(latentMatrix);
  cachedCorrelatedParams = correlationConfig.parameters;
}

// Generate a complete parameter sample from a sampling config
export function generateParameterSample(
  samplingConfig: SamplingConfig,
  baseParameters: Record<string, number | string | boolean>
): Record<string, number | string | boolean> {
  const sample = { ...baseParameters };
  const sampledParams = new Set<string>();
  
  // Handle correlated parameters if correlation matrix is configured and initialized
  if (samplingConfig.correlation_matrix && cachedCholeskyL && cachedCorrelatedParams) {
    const correlatedParams = cachedCorrelatedParams;
    const n = correlatedParams.length;
    
    // Generate correlated multivariate normal samples
    const mvSamples = sampleMultivariateNormal(cachedCholeskyL);
    
    // Convert to uniform marginals using normal CDF
    const uniformSamples = mvSamples.map(x => normalCDF(x));
    
    // Sample each correlated parameter using its distribution with the uniform quantile
    for (let i = 0; i < n; i++) {
      const paramName = correlatedParams[i];
      // Check both parameters and time_series_parameters
      const distConfig = samplingConfig.parameters[paramName]
        ?? samplingConfig.time_series_parameters?.[paramName];

      if (!distConfig) {
        console.warn(`Correlated parameter '${paramName}' not found in parameter distributions`);
        continue;
      }
      
      const value = sampleFromDistributionWithQuantile(distConfig, uniformSamples[i]);
      if (typeof value === 'number' || typeof value === 'string') {
        sample[paramName] = value;
        sampledParams.add(paramName);
      }
    }
  }

  // Sample remaining independent parameters (those not already sampled via correlation)
  for (const [paramName, distConfig] of Object.entries(samplingConfig.parameters)) {
    if (sampledParams.has(paramName)) continue;
    
    const value = sampleFromDistribution(distConfig);
    if (typeof value === 'number' || typeof value === 'string') {
      sample[paramName] = value;
    }
  }

  // Sample from time series parameters if present (skip those already sampled via correlation)
  if (samplingConfig.time_series_parameters) {
    for (const [paramName, distConfig] of Object.entries(samplingConfig.time_series_parameters)) {
      if (sampledParams.has(paramName)) continue;

      const value = sampleFromDistribution(distConfig);
      if (typeof value === 'number' || typeof value === 'string') {
        sample[paramName] = value;
      }
    }
  }

  return sample;
}

// Generate multiple samples
export function generateMultipleSamples(
  samplingConfig: SamplingConfig,
  baseParameters: Record<string, number | string | boolean>,
  count: number
): Record<string, number | string | boolean>[] {
  const samples: Record<string, number | string | boolean>[] = [];
  for (let i = 0; i < count; i++) {
    samples.push(generateParameterSample(samplingConfig, baseParameters));
  }
  return samples;
}

// Extract distribution metadata for a parameter (useful for shifting distributions later)
export interface DistributionMetadata {
  type: 'normal' | 'lognormal' | 'shifted_lognormal' | 'beta' | 'uniform' | 'choice' | 'fixed';
  mean?: number;
  std?: number;
  mu?: number;
  sigma?: number;
  shift?: number;
  alpha?: number;
  beta?: number;
  min?: number;
  max?: number;
  values?: (string | number)[];
  p?: number[];
  isLogSpace?: boolean;
}

export function getDistributionMetadata(config: DistributionConfig): DistributionMetadata {
  switch (config.dist) {
    case 'fixed':
      return { type: 'fixed' };

    case 'normal': {
      const { mean, std } = ci80ToNormalParams(config.ci80);
      return { type: 'normal', mean, std, min: config.min, max: config.max, isLogSpace: false };
    }

    case 'lognormal': {
      const { mu, sigma } = ci80ToLognormalParams(config.ci80);
      return { type: 'lognormal', mu, sigma, min: config.min, max: config.max, isLogSpace: true };
    }

    case 'shifted_lognormal': {
      // CI80 describes the lognormal part BEFORE shift
      const shift = config.shift;
      const { mu, sigma } = ci80ToLognormalParams(config.ci80);
      return { type: 'shifted_lognormal', mu, sigma, shift, min: config.min, max: config.max, isLogSpace: true };
    }

    case 'uniform':
      return { type: 'uniform', min: config.min, max: config.max };

    case 'beta':
      return { type: 'beta', alpha: config.alpha, beta: config.beta, min: config.min, max: config.max };

    case 'choice':
      return { type: 'choice', values: config.values, p: config.p };

    default:
      return { type: 'fixed' };
  }
}

// Get the bounds (min/max) of a distribution
export function getDistributionBounds(config: DistributionConfig): { min?: number; max?: number } {
  switch (config.dist) {
    case 'fixed':
      // Fixed distributions don't have meaningful bounds for sliders
      return {};

    case 'normal':
    case 'lognormal':
    case 'shifted_lognormal':
    case 'beta':
      return { min: config.min, max: config.max };

    case 'uniform':
      return { min: config.min, max: config.max };

    case 'choice':
      // Choice distributions don't have numeric bounds
      return {};

    default:
      return {};
  }
}

// Mapping from UI parameter names to sampling config parameter names
const UI_TO_SAMPLING_PARAM_MAP: Record<string, string> = {
  'coding_labor_exponent': 'parallel_penalty',
};

// Reverse mapping from sampling config names to UI names
const SAMPLING_TO_UI_PARAM_MAP: Record<string, string> = Object.fromEntries(
  Object.entries(UI_TO_SAMPLING_PARAM_MAP).map(([ui, sampling]) => [sampling, ui])
);

// Extract all parameter bounds from a sampling config
// Returns bounds in the format expected by sliders, with special handling for log-scale parameters
export function extractSamplingConfigBounds(
  samplingConfig: SamplingConfig,
  logScaleParams: Set<string> = new Set(['ac_time_horizon_minutes'])
): Record<string, { min?: number; max?: number }> {
  const bounds: Record<string, { min?: number; max?: number }> = {};

  // Process regular parameters
  for (const [paramName, distConfig] of Object.entries(samplingConfig.parameters)) {
    const rawBounds = getDistributionBounds(distConfig);
    
    // Determine the UI parameter name (may be different from sampling config name)
    const uiParamName = SAMPLING_TO_UI_PARAM_MAP[paramName] || paramName;
    
    if (logScaleParams.has(paramName) || logScaleParams.has(uiParamName)) {
      // Convert to log10 for UI display
      const logBounds = {
        min: rawBounds.min !== undefined && rawBounds.min > 0 ? Math.log10(rawBounds.min) : undefined,
        max: rawBounds.max !== undefined && rawBounds.max > 0 ? Math.log10(rawBounds.max) : undefined,
      };
      bounds[paramName] = logBounds;
      // Also add under UI param name if different
      if (uiParamName !== paramName) {
        bounds[uiParamName] = logBounds;
      }
    } else {
      bounds[paramName] = rawBounds;
      // Also add under UI param name if different
      if (uiParamName !== paramName) {
        bounds[uiParamName] = rawBounds;
      }
    }
  }

  // Process time series parameters
  if (samplingConfig.time_series_parameters) {
    for (const [paramName, distConfig] of Object.entries(samplingConfig.time_series_parameters)) {
      const rawBounds = getDistributionBounds(distConfig);
      const uiParamName = SAMPLING_TO_UI_PARAM_MAP[paramName] || paramName;
      
      if (logScaleParams.has(paramName) || logScaleParams.has(uiParamName)) {
        const logBounds = {
          min: rawBounds.min !== undefined && rawBounds.min > 0 ? Math.log10(rawBounds.min) : undefined,
          max: rawBounds.max !== undefined && rawBounds.max > 0 ? Math.log10(rawBounds.max) : undefined,
        };
        bounds[paramName] = logBounds;
        if (uiParamName !== paramName) {
          bounds[uiParamName] = logBounds;
        }
      } else {
        bounds[paramName] = rawBounds;
        if (uiParamName !== paramName) {
          bounds[uiParamName] = rawBounds;
        }
      }
    }
  }

  return bounds;
}

// Get the median value of a distribution (useful for fixing parameters)
export function getDistributionMedian(config: DistributionConfig): number | string {
  switch (config.dist) {
    case 'fixed':
      return config.value;

    case 'normal': {
      const { mean } = ci80ToNormalParams(config.ci80);
      // Apply bounds if specified
      let value = mean;
      if (config.min !== undefined) value = Math.max(value, config.min);
      if (config.max !== undefined) value = Math.min(value, config.max);
      return value;
    }

    case 'lognormal': {
      const { mu } = ci80ToLognormalParams(config.ci80);
      // Median of lognormal is exp(mu)
      let value = Math.exp(mu);
      if (config.min !== undefined) value = Math.max(value, config.min);
      if (config.max !== undefined) value = Math.min(value, config.max);
      return value;
    }

    case 'shifted_lognormal': {
      // CI80 describes the lognormal part BEFORE shift
      // Median of lognormal is exp(mu), then add shift
      const { mu } = ci80ToLognormalParams(config.ci80);
      let value = Math.exp(mu) + config.shift;
      if (config.min !== undefined) value = Math.max(value, config.min);
      if (config.max !== undefined) value = Math.min(value, config.max);
      return value;
    }

    case 'uniform': {
      return (config.min + config.max) / 2;
    }

    case 'beta': {
      const min = config.min ?? 0;
      const max = config.max ?? 1;
      // Median of beta distribution - use mean as approximation
      const mean = config.alpha / (config.alpha + config.beta);
      return min + mean * (max - min);
    }

    case 'choice': {
      // Return the most probable value, or first value if equal probability
      let maxIdx = 0;
      let maxP = config.p[0];
      for (let i = 1; i < config.p.length; i++) {
        if (config.p[i] > maxP) {
          maxP = config.p[i];
          maxIdx = i;
        }
      }
      return config.values[maxIdx];
    }

    default:
      throw new Error(`Unknown distribution type: ${(config as DistributionConfig).dist}`);
  }
}

// Generate a parameter sample with some parameters fixed to median
export function generateParameterSampleWithFixedParams(
  samplingConfig: SamplingConfig,
  baseParameters: Record<string, number | string | boolean>,
  enabledParams: Set<string>
): Record<string, number | string | boolean> {
  const sample = { ...baseParameters };
  const sampledParams = new Set<string>();

  // Handle correlated parameters if correlation matrix is configured and initialized
  // Only sample correlated parameters that are enabled
  if (samplingConfig.correlation_matrix && cachedCholeskyL && cachedCorrelatedParams) {
    const correlatedParams = cachedCorrelatedParams;
    
    // Find which correlated params are enabled
    const enabledCorrelatedIndices: number[] = [];
    const enabledCorrelatedNames: string[] = [];
    for (let i = 0; i < correlatedParams.length; i++) {
      const paramName = correlatedParams[i];
      if (enabledParams.has(paramName)) {
        enabledCorrelatedIndices.push(i);
        enabledCorrelatedNames.push(paramName);
      }
    }
    
    if (enabledCorrelatedIndices.length > 0) {
      // Generate full correlated samples, but only use enabled ones
      const mvSamples = sampleMultivariateNormal(cachedCholeskyL);
      const uniformSamples = mvSamples.map(x => normalCDF(x));
      
      for (let i = 0; i < enabledCorrelatedIndices.length; i++) {
        const origIdx = enabledCorrelatedIndices[i];
        const paramName = enabledCorrelatedNames[i];
        const distConfig = samplingConfig.parameters[paramName];
        
        if (distConfig) {
          const value = sampleFromDistributionWithQuantile(distConfig, uniformSamples[origIdx]);
          if (typeof value === 'number' || typeof value === 'string') {
            sample[paramName] = value;
            sampledParams.add(paramName);
          }
        }
      }
    }
    
    // Fix non-enabled correlated params to median
    for (const paramName of correlatedParams) {
      if (!enabledParams.has(paramName) && !sampledParams.has(paramName)) {
        const distConfig = samplingConfig.parameters[paramName];
        if (distConfig) {
          const value = getDistributionMedian(distConfig);
          if (typeof value === 'number' || typeof value === 'string') {
            sample[paramName] = value;
            sampledParams.add(paramName);
          }
        }
      }
    }
  }

  // Sample from remaining parameter distributions
  for (const [paramName, distConfig] of Object.entries(samplingConfig.parameters)) {
    if (sampledParams.has(paramName)) continue;
    
    if (enabledParams.has(paramName)) {
      // Sample from the distribution
      const value = sampleFromDistribution(distConfig);
      if (typeof value === 'number' || typeof value === 'string') {
        sample[paramName] = value;
      }
    } else {
      // Use median value
      const value = getDistributionMedian(distConfig);
      if (typeof value === 'number' || typeof value === 'string') {
        sample[paramName] = value;
      }
    }
  }

  // Sample from time series parameters if present
  if (samplingConfig.time_series_parameters) {
    for (const [paramName, distConfig] of Object.entries(samplingConfig.time_series_parameters)) {
      if (enabledParams.has(paramName)) {
        const value = sampleFromDistribution(distConfig);
        if (typeof value === 'number' || typeof value === 'string') {
          sample[paramName] = value;
        }
      } else {
        const value = getDistributionMedian(distConfig);
        if (typeof value === 'number' || typeof value === 'string') {
          sample[paramName] = value;
        }
      }
    }
  }

  return sample;
}

// Sample from a shifted distribution (for when user changes a parameter)
// For normal: preserve std, shift mean
// For lognormal: preserve sigma (log-space std), shift to new median
// For beta: preserve concentration (α+β), shift mean
// For choice/fixed: collapse to user's value
export function sampleWithShiftedMean(
  metadata: DistributionMetadata,
  newMean: number
): number {
  switch (metadata.type) {
    case 'normal': {
      const std = metadata.std ?? 0;
      let value = randomNormal(newMean, std);
      if (metadata.min !== undefined) value = Math.max(value, metadata.min);
      if (metadata.max !== undefined) value = Math.min(value, metadata.max);
      return value;
    }

    case 'lognormal': {
      // For lognormal, preserve sigma (uncertainty in log space)
      // Shift so the new median equals newMean
      // Median of lognormal is exp(mu), so newMu = log(newMean)
      const sigma = metadata.sigma ?? 0;
      if (newMean <= 0) return metadata.min ?? 0;
      const newMu = Math.log(newMean);
      let value = Math.exp(randomNormal(newMu, sigma));
      if (metadata.min !== undefined) value = Math.max(value, metadata.min);
      if (metadata.max !== undefined) value = Math.min(value, metadata.max);
      return value;
    }

    case 'shifted_lognormal': {
      const shift = metadata.shift ?? 0;
      const sigma = metadata.sigma ?? 0;
      const targetUnshifted = newMean - shift;
      if (targetUnshifted <= 0) return metadata.min ?? shift;
      const newMu = Math.log(targetUnshifted);
      let value = Math.exp(randomNormal(newMu, sigma)) + shift;
      if (metadata.min !== undefined) value = Math.max(value, metadata.min);
      if (metadata.max !== undefined) value = Math.min(value, metadata.max);
      return value;
    }

    case 'beta': {
      // For beta distribution, preserve the concentration (α+β) which controls peakedness
      // Shift the mean to the user's value
      const origAlpha = metadata.alpha ?? 1;
      const origBeta = metadata.beta ?? 1;
      const min = metadata.min ?? 0;
      const max = metadata.max ?? 1;
      const range = max - min;
      
      // Convert newMean to [0,1] scale
      const normalizedMean = Math.max(0.001, Math.min(0.999, (newMean - min) / range));
      
      // Preserve concentration
      const concentration = origAlpha + origBeta;
      const newAlpha = normalizedMean * concentration;
      const newBeta = (1 - normalizedMean) * concentration;
      
      // Sample from the shifted beta
      const betaSample = randomBeta(Math.max(0.01, newAlpha), Math.max(0.01, newBeta));
      return min + betaSample * range;
    }

    case 'uniform': {
      // For uniform, we could shrink the range around the new mean
      // But for simplicity, just return the user's value (collapse uncertainty)
      return newMean;
    }

    case 'choice':
    case 'fixed':
    default:
      return newMean;
  }
}

// Sample from a shifted distribution using a quantile (for correlated sampling)
export function sampleWithShiftedMeanAndQuantile(
  metadata: DistributionMetadata,
  newMean: number,
  quantile: number
): number {
  // Clamp quantile to valid range
  quantile = Math.max(1e-10, Math.min(1 - 1e-10, quantile));
  
  switch (metadata.type) {
    case 'normal': {
      const std = metadata.std ?? 0;
      let value = newMean + std * Math.sqrt(2) * erfinv(2 * quantile - 1);
      if (metadata.min !== undefined) value = Math.max(value, metadata.min);
      if (metadata.max !== undefined) value = Math.min(value, metadata.max);
      return value;
    }

    case 'lognormal': {
      const sigma = metadata.sigma ?? 0;
      if (newMean <= 0) return metadata.min ?? 0;
      const newMu = Math.log(newMean);
      let value = Math.exp(newMu + sigma * Math.sqrt(2) * erfinv(2 * quantile - 1));
      if (metadata.min !== undefined) value = Math.max(value, metadata.min);
      if (metadata.max !== undefined) value = Math.min(value, metadata.max);
      return value;
    }

    case 'shifted_lognormal': {
      const shift = metadata.shift ?? 0;
      const sigma = metadata.sigma ?? 0;
      const targetUnshifted = newMean - shift;
      if (targetUnshifted <= 0) return metadata.min ?? shift;
      const newMu = Math.log(targetUnshifted);
      let value = Math.exp(newMu + sigma * Math.sqrt(2) * erfinv(2 * quantile - 1)) + shift;
      if (metadata.min !== undefined) value = Math.max(value, metadata.min);
      if (metadata.max !== undefined) value = Math.min(value, metadata.max);
      return value;
    }

    case 'beta': {
      const origAlpha = metadata.alpha ?? 1;
      const origBeta = metadata.beta ?? 1;
      const min = metadata.min ?? 0;
      const max = metadata.max ?? 1;
      const range = max - min;
      
      const normalizedMean = Math.max(0.001, Math.min(0.999, (newMean - min) / range));
      const concentration = origAlpha + origBeta;
      const newAlpha = normalizedMean * concentration;
      const newBeta = (1 - normalizedMean) * concentration;
      
      const x01 = betaInverseCDF(quantile, Math.max(0.01, newAlpha), Math.max(0.01, newBeta));
      return min + x01 * range;
    }

    case 'uniform':
    case 'choice':
    case 'fixed':
    default:
      return newMean;
  }
}

// Generate a parameter sample with distributions shifted to user's current values
// This preserves uncertainty but centers it around the user's parameter choices
export function generateParameterSampleWithUserValues(
  samplingConfig: SamplingConfig,
  userParameters: Record<string, number | string | boolean>,
  enabledParams: Set<string>
): Record<string, number | string | boolean> {
  const sample = { ...userParameters };
  const sampledParams = new Set<string>();

  // Pre-compute metadata for all parameters
  const paramMetadata: Record<string, DistributionMetadata> = {};
  for (const [paramName, distConfig] of Object.entries(samplingConfig.parameters)) {
    paramMetadata[paramName] = getDistributionMetadata(distConfig);
  }
  if (samplingConfig.time_series_parameters) {
    for (const [paramName, distConfig] of Object.entries(samplingConfig.time_series_parameters)) {
      paramMetadata[paramName] = getDistributionMetadata(distConfig);
    }
  }

  // Handle correlated parameters if correlation matrix is configured and initialized
  if (samplingConfig.correlation_matrix && cachedCholeskyL && cachedCorrelatedParams) {
    const correlatedParams = cachedCorrelatedParams;
    
    // Find which correlated params are enabled
    const enabledCorrelatedIndices: number[] = [];
    const enabledCorrelatedNames: string[] = [];
    for (let i = 0; i < correlatedParams.length; i++) {
      const paramName = correlatedParams[i];
      if (enabledParams.has(paramName)) {
        enabledCorrelatedIndices.push(i);
        enabledCorrelatedNames.push(paramName);
      }
    }
    
    if (enabledCorrelatedIndices.length > 0) {
      // Generate correlated uniform samples via Gaussian copula
      const mvSamples = sampleMultivariateNormal(cachedCholeskyL);
      const uniformSamples = mvSamples.map(x => normalCDF(x));
      
      for (let i = 0; i < enabledCorrelatedIndices.length; i++) {
        const origIdx = enabledCorrelatedIndices[i];
        const paramName = enabledCorrelatedNames[i];
        const metadata = paramMetadata[paramName];
        const userValue = userParameters[paramName];
        
        if (metadata && typeof userValue === 'number') {
          // Sample with distribution shifted to user's value
          const value = sampleWithShiftedMeanAndQuantile(metadata, userValue, uniformSamples[origIdx]);
          sample[paramName] = value;
          sampledParams.add(paramName);
        }
      }
    }
    
    // Non-enabled correlated params use user's value directly (collapsed uncertainty)
    for (const paramName of correlatedParams) {
      if (!enabledParams.has(paramName) && !sampledParams.has(paramName)) {
        // Keep the user's value (already in sample from spread)
        sampledParams.add(paramName);
      }
    }
  }

  // Sample from remaining parameter distributions
  for (const [paramName, distConfig] of Object.entries(samplingConfig.parameters)) {
    if (sampledParams.has(paramName)) continue;
    
    const userValue = userParameters[paramName];
    const metadata = paramMetadata[paramName];
    
    if (enabledParams.has(paramName) && typeof userValue === 'number' && metadata) {
      // Sample from distribution shifted to user's value
      const value = sampleWithShiftedMean(metadata, userValue);
      sample[paramName] = value;
    }
    // If not enabled, keep the user's value (already in sample from spread)
    
    sampledParams.add(paramName);
  }

  // Sample from time series parameters if present
  if (samplingConfig.time_series_parameters) {
    for (const [paramName, distConfig] of Object.entries(samplingConfig.time_series_parameters)) {
      if (sampledParams.has(paramName)) continue;
      
      const userValue = userParameters[paramName];
      const metadata = paramMetadata[paramName];
      
      if (enabledParams.has(paramName) && typeof userValue === 'number' && metadata) {
        const value = sampleWithShiftedMean(metadata, userValue);
        sample[paramName] = value;
      }
      // If not enabled, keep the user's value
    }
  }

  return sample;
}


