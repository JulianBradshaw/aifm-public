/**
 * TypeScript types for sampling configuration used in Monte Carlo simulations.
 * These types correspond to the input_distributions.yaml format.
 */

export type DistributionType =
  | 'fixed'
  | 'choice'
  | 'uniform'
  | 'normal'
  | 'lognormal'
  | 'shifted_lognormal'
  | 'beta';

export interface ParameterDistribution {
  dist: DistributionType;
  // For fixed distributions
  value?: number | string;
  // For choice distributions
  values?: (string | number)[];
  p?: number[];
  // For continuous distributions with CI80
  ci80?: [number, number];
  // Bounds
  min?: number;
  max?: number;
  // For shifted_lognormal
  shift?: number;
  // For beta distribution
  alpha?: number;
  beta?: number;
  // Clipping behavior
  clip_to_bounds?: boolean;
}

export interface CorrelationMatrix {
  parameters: string[];
  correlation_matrix: number[][];
  correlation_type: string;
}

export interface SamplingConfig {
  // Main model parameters
  parameters: Record<string, ParameterDistribution>;
  // Time series generation parameters
  time_series_parameters?: Record<string, ParameterDistribution>;
  // Parameter correlations
  correlation_matrix?: CorrelationMatrix;
  // Run settings
  seed?: number;
  num_samples?: number;
  num_rollouts?: number;
  initial_progress?: number;
  time_range?: [number, number];
  input_data?: string;
  per_sample_timeout?: number;
}

/**
 * Compute the median value from a distribution specification.
 * Returns null for discrete distributions where median doesn't apply.
 */
export function computeMedian(dist: ParameterDistribution): number | null {
  switch (dist.dist) {
    case 'fixed':
      return typeof dist.value === 'number' ? dist.value : null;

    case 'normal':
      // For normal distribution, median = mean
      // If ci80 is provided, the median is the center of the interval
      if (dist.ci80) {
        return (dist.ci80[0] + dist.ci80[1]) / 2;
      }
      return null;

    case 'lognormal':
      // For lognormal, the median is the geometric mean of the CI80 bounds
      if (dist.ci80) {
        return Math.sqrt(dist.ci80[0] * dist.ci80[1]);
      }
      return null;

    case 'shifted_lognormal':
      // For shifted lognormal, compute geometric mean and add shift
      if (dist.ci80 && dist.shift !== undefined) {
        const unshiftedMedian = Math.sqrt(dist.ci80[0] * dist.ci80[1]);
        return unshiftedMedian + dist.shift;
      }
      return null;

    case 'uniform':
      // Median of uniform is the midpoint
      if (dist.min !== undefined && dist.max !== undefined) {
        return (dist.min + dist.max) / 2;
      }
      return null;

    case 'beta':
      // Beta distribution median approximation
      // For symmetric beta (alpha = beta), median = 0.5
      // For general case, use approximation: (alpha - 1/3) / (alpha + beta - 2/3)
      if (dist.alpha !== undefined && dist.beta !== undefined) {
        const a = dist.alpha;
        const b = dist.beta;
        if (a > 1 && b > 1) {
          const medianNormalized = (a - 1 / 3) / (a + b - 2 / 3);
          // Scale to [min, max] if provided
          const min = dist.min ?? 0;
          const max = dist.max ?? 1;
          return min + medianNormalized * (max - min);
        } else if (a === b) {
          const min = dist.min ?? 0;
          const max = dist.max ?? 1;
          return (min + max) / 2;
        }
      }
      return null;

    case 'choice':
      // Discrete distribution - no meaningful median
      return null;

    default:
      return null;
  }
}

/**
 * Format a CI80 interval for display.
 */
export function formatCI80(ci80: [number, number] | undefined): string {
  if (!ci80) return '-';

  const [low, high] = ci80;

  // Use scientific notation for very large or very small numbers
  const formatNum = (n: number): string => {
    if (n === 0) return '0';
    const absN = Math.abs(n);
    if (absN >= 1e6 || (absN < 0.01 && absN > 0)) {
      return n.toExponential(1);
    }
    if (Number.isInteger(n)) return n.toString();
    return n.toPrecision(3);
  };

  return `[${formatNum(low)}, ${formatNum(high)}]`;
}

/**
 * Format a number for display (handles scientific notation).
 */
export function formatNumber(n: number | null | undefined): string {
  if (n === null || n === undefined) return '-';
  if (n === 0) return '0';
  const absN = Math.abs(n);
  if (absN >= 1e6 || (absN < 0.01 && absN > 0)) {
    return n.toExponential(2);
  }
  if (Number.isInteger(n)) return n.toString();
  return n.toPrecision(3);
}

/**
 * Get a human-readable name for a distribution type.
 */
export function getDistributionTypeName(dist: DistributionType): string {
  switch (dist) {
    case 'fixed':
      return 'Fixed';
    case 'choice':
      return 'Choice';
    case 'uniform':
      return 'Uniform';
    case 'normal':
      return 'Normal';
    case 'lognormal':
      return 'Log-normal';
    case 'shifted_lognormal':
      return 'Shifted log-normal';
    case 'beta':
      return 'Beta';
    default:
      return dist;
  }
}
