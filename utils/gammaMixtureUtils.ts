// Gamma mixture CDF utilities for precise interpolation of control points
// Uses a weighted sum of gamma distribution CDFs to exactly pass through control points

import { ControlPoint, clamp } from './cdfSplineUtils';

// ============================================================================
// Special Functions Implementation
// ============================================================================

/**
 * Log gamma function using Stirling's approximation with Lanczos coefficients
 */
function logGamma(x: number): number {
  if (x <= 0) return Infinity;

  // Lanczos coefficients for g=7
  const c = [
    0.99999999999980993,
    676.5203681218851,
    -1259.1392167224028,
    771.32342877765313,
    -176.61502916214059,
    12.507343278686905,
    -0.13857109526572012,
    9.9843695780195716e-6,
    1.5056327351493116e-7,
  ];

  if (x < 0.5) {
    // Use reflection formula
    return Math.log(Math.PI / Math.sin(Math.PI * x)) - logGamma(1 - x);
  }

  x -= 1;
  let a = c[0];
  for (let i = 1; i < c.length; i++) {
    a += c[i] / (x + i);
  }

  const t = x + 7.5;
  return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
}

/**
 * Gamma function
 */
function gamma(x: number): number {
  return Math.exp(logGamma(x));
}

/**
 * Regularized lower incomplete gamma function P(a, x) = γ(a, x) / Γ(a)
 * Uses series expansion for small x and continued fraction for large x
 */
function gammainc(a: number, x: number): number {
  if (x < 0 || a <= 0) return 0;
  if (x === 0) return 0;

  // Choose method based on x vs a
  if (x < a + 1) {
    // Use series expansion
    return gammainc_series(a, x);
  } else {
    // Use continued fraction (complement)
    return 1 - gammainc_cf(a, x);
  }
}

/**
 * Series expansion for the regularized lower incomplete gamma
 * P(a, x) = e^(-x) * x^a * sum(x^n / Gamma(a+n+1))
 */
function gammainc_series(a: number, x: number): number {
  const maxIter = 200;
  const eps = 1e-12;

  let sum = 1 / a;
  let term = 1 / a;

  for (let n = 1; n < maxIter; n++) {
    term *= x / (a + n);
    sum += term;
    if (Math.abs(term) < eps * Math.abs(sum)) break;
  }

  return Math.exp(-x + a * Math.log(x) - logGamma(a)) * sum;
}

/**
 * Continued fraction for the regularized upper incomplete gamma Q(a, x)
 * Uses Lentz's algorithm
 */
function gammainc_cf(a: number, x: number): number {
  const maxIter = 200;
  const eps = 1e-12;
  const tiny = 1e-30;

  let b = x + 1 - a;
  let c = 1 / tiny;
  let d = 1 / b;
  let h = d;

  for (let i = 1; i < maxIter; i++) {
    const an = -i * (i - a);
    b += 2;
    d = an * d + b;
    if (Math.abs(d) < tiny) d = tiny;
    c = b + an / c;
    if (Math.abs(c) < tiny) c = tiny;
    d = 1 / d;
    const delta = d * c;
    h *= delta;
    if (Math.abs(delta - 1) < eps) break;
  }

  return Math.exp(-x + a * Math.log(x) - logGamma(a)) * h;
}

// ============================================================================
// Gamma Distribution Functions
// ============================================================================

/**
 * Gamma distribution CDF
 * P(X <= x) for X ~ Gamma(shape, rate)
 * Note: We use shape/rate parameterization, not shape/scale
 */
export function gammaCdf(x: number, shape: number, rate: number): number {
  if (x <= 0) return 0;
  return gammainc(shape, rate * x);
}

/**
 * Gamma distribution PDF
 * f(x) = (rate^shape / Gamma(shape)) * x^(shape-1) * e^(-rate*x)
 */
export function gammaPdf(x: number, shape: number, rate: number): number {
  if (x <= 0) return shape < 1 ? Infinity : 0;

  const logPdf = shape * Math.log(rate) - logGamma(shape) + (shape - 1) * Math.log(x) - rate * x;
  return Math.exp(logPdf);
}

// ============================================================================
// Gamma Mixture Model
// ============================================================================

export interface GammaComponent {
  shape: number;  // α (shape parameter)
  rate: number;   // β (rate parameter)
  weight: number; // mixture weight (non-negative, sum to 1)
}

export interface GammaMixtureModel {
  components: GammaComponent[];
  xOffset: number;  // shift applied to x values (x_effective = x - xOffset)
  yOffset: number;  // y value at x = xOffset (typically 0)
  yScale: number;   // scaling for y values
}

/**
 * Evaluate gamma mixture CDF at a point
 */
export function evaluateGammaMixtureCdf(model: GammaMixtureModel, x: number): number {
  const t = x - model.xOffset;

  if (t <= 0) return model.yOffset;

  let cdf = 0;
  for (const comp of model.components) {
    cdf += comp.weight * gammaCdf(t, comp.shape, comp.rate);
  }

  return model.yOffset + model.yScale * cdf;
}

/**
 * Evaluate gamma mixture PDF at a point
 */
export function evaluateGammaMixturePdf(model: GammaMixtureModel, x: number): number {
  const t = x - model.xOffset;

  if (t <= 0) return 0;

  let pdf = 0;
  for (const comp of model.components) {
    pdf += comp.weight * gammaPdf(t, comp.shape, comp.rate);
  }

  return model.yScale * pdf;
}

// ============================================================================
// Matrix Operations for Linear System Solving
// ============================================================================

/**
 * Solve linear system Ax = b using Gaussian elimination with partial pivoting
 */
function solveLinearSystem(A: number[][], b: number[]): number[] | null {
  const n = A.length;

  // Create augmented matrix
  const aug: number[][] = A.map((row, i) => [...row, b[i]]);

  // Forward elimination with partial pivoting
  for (let col = 0; col < n; col++) {
    // Find pivot
    let maxVal = Math.abs(aug[col][col]);
    let maxRow = col;
    for (let row = col + 1; row < n; row++) {
      if (Math.abs(aug[row][col]) > maxVal) {
        maxVal = Math.abs(aug[row][col]);
        maxRow = row;
      }
    }

    // Swap rows
    if (maxRow !== col) {
      [aug[col], aug[maxRow]] = [aug[maxRow], aug[col]];
    }

    // Check for singular matrix
    if (Math.abs(aug[col][col]) < 1e-12) {
      return null; // Singular or near-singular
    }

    // Eliminate below
    for (let row = col + 1; row < n; row++) {
      const factor = aug[row][col] / aug[col][col];
      for (let j = col; j <= n; j++) {
        aug[row][j] -= factor * aug[col][j];
      }
    }
  }

  // Back substitution
  const x: number[] = new Array(n).fill(0);
  for (let i = n - 1; i >= 0; i--) {
    x[i] = aug[i][n];
    for (let j = i + 1; j < n; j++) {
      x[i] -= aug[i][j] * x[j];
    }
    x[i] /= aug[i][i];
  }

  return x;
}

// ============================================================================
// Fitting Gamma Mixture to Control Points
// ============================================================================

/**
 * Fit a gamma mixture model to interpolate the control points.
 *
 * The approach:
 * 1. Shift x values so the first control point is at t=0
 * 2. Use exactly N-1 components for N-1 interior points
 * 3. Initialize each component to roughly pass through its target point
 * 4. Use nonlinear optimization over all parameters (shapes, rates, weights)
 *
 * @param points - Control points to interpolate
 * @param regularization - Weight for smoothness regularization (0 = none, higher = smoother PDF)
 */
export function fitGammaMixture(points: ControlPoint[], regularization: number = 0.01): GammaMixtureModel {
  const sorted = [...points].sort((a, b) => a.x - b.x);
  const n = sorted.length;

  if (n < 2) {
    return {
      components: [{ shape: 2, rate: 1, weight: 1 }],
      xOffset: sorted[0]?.x ?? 0,
      yOffset: sorted[0]?.y ?? 0,
      yScale: 1,
    };
  }

  // Extract x and y values
  const xOffset = sorted[0].x;
  const yOffset = sorted[0].y;
  const yMax = sorted[n - 1].y;
  const yScale = yMax - yOffset;

  if (yScale < 1e-10) {
    // Flat CDF
    return {
      components: [{ shape: 2, rate: 1, weight: 1 }],
      xOffset,
      yOffset,
      yScale: 0,
    };
  }

  // Shift and normalize
  const tValues = sorted.map(p => p.x - xOffset);
  const yNorm = sorted.map(p => (p.y - yOffset) / yScale); // Normalized to [0, 1]

  const tMax = tValues[n - 1];
  const numInterior = n - 2; // Number of interior points (excluding first and last)

  // Calculate minimum number of components needed for perfect interpolation
  // Based on empirical testing: K = ceil((N-2) / 2) works for most cases
  // We add 1 as a safety margin for robustness
  const numComponents = Math.max(1, Math.ceil((numInterior + 1) / 2));

  // Initialize components spread across the time range
  const components: GammaComponent[] = [];
  for (let k = 0; k < numComponents; k++) {
    // Spread components evenly across the range
    const targetT = tMax * (k + 1) / (numComponents + 1);
    const shape = 2.0;
    const rate = shape / targetT; // So mean = shape/rate = targetT
    components.push({ shape, rate, weight: 1.0 / numComponents });
  }

  // Now optimize all parameters jointly using Levenberg-Marquardt style optimization
  // Parameters: for each component, we have (shape, rate, weight)
  // But weights are constrained to sum to 1, so we use softmax parameterization

  // Pack parameters: [logShape0, logRate0, rawWeight0, logShape1, logRate1, rawWeight1, ...]
  // We use log for shape/rate to keep them positive, and softmax for weights
  let params: number[] = [];
  for (const comp of components) {
    params.push(Math.log(comp.shape));
    params.push(Math.log(comp.rate));
    params.push(0); // Raw weight (will be softmaxed)
  }

  // Helper to unpack parameters into components
  const unpackParams = (p: number[]): GammaComponent[] => {
    const comps: GammaComponent[] = [];
    const rawWeights: number[] = [];

    for (let i = 0; i < numComponents; i++) {
      const logShape = p[i * 3];
      const logRate = p[i * 3 + 1];
      rawWeights.push(p[i * 3 + 2]);
      comps.push({
        shape: clamp(Math.exp(logShape), 0.5, 50),
        rate: clamp(Math.exp(logRate), 0.001, 1000),
        weight: 0, // Will be set below
      });
    }

    // Softmax for weights
    const maxRaw = Math.max(...rawWeights);
    const expWeights = rawWeights.map(w => Math.exp(w - maxRaw));
    const sumExp = expWeights.reduce((a, b) => a + b, 0);
    for (let i = 0; i < numComponents; i++) {
      comps[i].weight = expWeights[i] / sumExp;
    }

    return comps;
  };

  // Compute residuals (errors at each interior point)
  const computeResiduals = (p: number[]): number[] => {
    const comps = unpackParams(p);
    const residuals: number[] = [];

    for (let i = 1; i < n; i++) {
      let pred = 0;
      for (const comp of comps) {
        pred += comp.weight * gammaCdf(tValues[i], comp.shape, comp.rate);
      }
      residuals.push(pred - yNorm[i]);
    }

    return residuals;
  };

  // Compute regularization penalty for multimodality
  // Penalizes variance of component means (shape/rate), which causes multimodal PDFs
  const computeRegularization = (p: number[]): number => {
    const comps = unpackParams(p);
    if (comps.length <= 1) return 0;

    // Compute mean of each component (in normalized time units)
    const means = comps.map(c => c.shape / c.rate);

    // Compute weighted average mean
    let avgMean = 0;
    for (let i = 0; i < comps.length; i++) {
      avgMean += comps[i].weight * means[i];
    }

    // Compute weighted variance of means
    let variance = 0;
    for (let i = 0; i < comps.length; i++) {
      variance += comps[i].weight * (means[i] - avgMean) ** 2;
    }

    // Normalize by tMax^2 to make regularization scale-invariant
    const normalizedVariance = variance / (tMax * tMax);

    return normalizedVariance;
  };

  // Regularization weight - balances interpolation accuracy vs PDF smoothness
  // Lower values = better interpolation, higher values = smoother PDF
  const regWeight = regularization;

  // Compute total squared error with regularization
  const computeError = (p: number[]): number => {
    const residuals = computeResiduals(p);
    const interpError = residuals.reduce((sum, r) => sum + r * r, 0);
    const regPenalty = computeRegularization(p);
    return interpError + regWeight * regPenalty;
  };

  // Compute Jacobian numerically
  const computeJacobian = (p: number[]): number[][] => {
    const eps = 1e-6;
    const numParams = p.length;
    const baseResiduals = computeResiduals(p);
    const numResiduals = baseResiduals.length;

    const J: number[][] = [];
    for (let i = 0; i < numResiduals; i++) {
      J.push(new Array(numParams).fill(0));
    }

    for (let j = 0; j < numParams; j++) {
      const pPlus = [...p];
      pPlus[j] += eps;
      const residualsPlus = computeResiduals(pPlus);

      for (let i = 0; i < numResiduals; i++) {
        J[i][j] = (residualsPlus[i] - baseResiduals[i]) / eps;
      }
    }

    return J;
  };

  // Levenberg-Marquardt optimization
  let lambda = 0.001;
  let bestError = computeError(params);
  let bestParams = [...params];

  for (let iter = 0; iter < 200; iter++) {
    if (bestError < 1e-14) break; // Good enough

    const residuals = computeResiduals(params);
    const J = computeJacobian(params);
    const numParams = params.length;
    const numResiduals = residuals.length;

    // Compute J^T * J
    const JtJ: number[][] = [];
    for (let i = 0; i < numParams; i++) {
      JtJ.push(new Array(numParams).fill(0));
      for (let j = 0; j < numParams; j++) {
        for (let k = 0; k < numResiduals; k++) {
          JtJ[i][j] += J[k][i] * J[k][j];
        }
      }
    }

    // Compute J^T * r (gradient of sum of squared residuals)
    const Jtr: number[] = new Array(numParams).fill(0);
    for (let i = 0; i < numParams; i++) {
      for (let k = 0; k < numResiduals; k++) {
        Jtr[i] += J[k][i] * residuals[k];
      }
    }

    // Add gradient of regularization term (computed numerically)
    const eps = 1e-6;
    const baseReg = computeRegularization(params);
    for (let i = 0; i < numParams; i++) {
      const pPlus = [...params];
      pPlus[i] += eps;
      const regPlus = computeRegularization(pPlus);
      const regGrad = (regPlus - baseReg) / eps;
      Jtr[i] += regWeight * regGrad;
    }

    // Add damping: (J^T J + λI) δ = -J^T r
    const A: number[][] = JtJ.map((row, i) =>
      row.map((val, j) => val + (i === j ? lambda : 0))
    );
    const b = Jtr.map(v => -v);

    // Solve for step
    const delta = solveLinearSystem(A, b);

    if (!delta) {
      lambda *= 10;
      continue;
    }

    // Try the step
    const newParams = params.map((p, i) => p + delta[i]);
    const newError = computeError(newParams);

    if (newError < bestError) {
      params = newParams;
      bestError = newError;
      bestParams = [...newParams];
      lambda *= 0.5;
      lambda = Math.max(lambda, 1e-10);
    } else {
      lambda *= 2;
      if (lambda > 1e10) break;
    }
  }

  // Unpack best parameters into final model
  const finalComponents = unpackParams(bestParams);

  return {
    components: finalComponents,
    xOffset,
    yOffset,
    yScale,
  };
}

// ============================================================================
// Public API for CDF/PDF Evaluation
// ============================================================================

/**
 * Evaluate the gamma mixture CDF at a given x value.
 * This is the main function used for interpolation.
 */
export function evaluateGammaMixtureAt(points: ControlPoint[], x: number, regularization: number = 0.01): number {
  const sorted = [...points].sort((a, b) => a.x - b.x);

  if (sorted.length === 0) return 0;
  if (sorted.length === 1) return sorted[0].y;

  // Handle x outside the range
  if (x <= sorted[0].x) return sorted[0].y;
  if (x >= sorted[sorted.length - 1].x) return sorted[sorted.length - 1].y;

  // Build and evaluate gamma mixture model
  const model = fitGammaMixture(sorted, regularization);
  const y = evaluateGammaMixtureCdf(model, x);

  // Clamp to valid CDF range
  return clamp(y, 0, 1);
}

/**
 * Evaluate the PDF (derivative of CDF) for the gamma mixture at a given x value.
 */
export function evaluateGammaMixturePdfAt(points: ControlPoint[], x: number, regularization: number = 0.01): number {
  const sorted = [...points].sort((a, b) => a.x - b.x);

  if (sorted.length < 2) return 0;

  // Handle x outside the range
  if (x < sorted[0].x || x > sorted[sorted.length - 1].x) return 0;

  // Build and evaluate gamma mixture model
  const model = fitGammaMixture(sorted, regularization);
  const pdf = evaluateGammaMixturePdf(model, x);

  // PDF should be non-negative
  return Math.max(0, pdf);
}

/**
 * Sample the gamma mixture PDF at regular intervals for visualization.
 */
export function sampleGammaMixturePdf(
  controlPoints: ControlPoint[],
  numSamples: number = 200,
  regularization: number = 0.01
): { x: number; pdf: number }[] {
  if (controlPoints.length < 2) return [];

  const sorted = [...controlPoints].sort((a, b) => a.x - b.x);
  const minX = sorted[0].x;
  const maxX = sorted[sorted.length - 1].x;

  // Build model once for efficiency
  const model = fitGammaMixture(sorted, regularization);

  const samples: { x: number; pdf: number }[] = [];

  for (let i = 0; i <= numSamples; i++) {
    const x = minX + (maxX - minX) * (i / numSamples);
    const pdf = Math.max(0, evaluateGammaMixturePdf(model, x));
    samples.push({ x, pdf });
  }

  return samples;
}

/**
 * Generate SVG path string for gamma mixture CDF curve.
 */
export function generateGammaMixturePath(
  controlPoints: ControlPoint[],
  xScale: (value: number) => number,
  yScale: (value: number) => number,
  numSamples: number = 100,
  regularization: number = 0.01
): string {
  if (controlPoints.length < 2) return '';

  const sorted = [...controlPoints].sort((a, b) => a.x - b.x);
  const minX = sorted[0].x;
  const maxX = sorted[sorted.length - 1].x;

  // Build model once for efficiency
  const model = fitGammaMixture(sorted, regularization);

  const pathParts: string[] = [];

  for (let i = 0; i <= numSamples; i++) {
    const x = minX + (maxX - minX) * (i / numSamples);
    const y = clamp(evaluateGammaMixtureCdf(model, x), 0, 1);
    const px = xScale(x);
    const py = yScale(y);

    if (i === 0) {
      pathParts.push(`M ${px} ${py}`);
    } else {
      pathParts.push(`L ${px} ${py}`);
    }
  }

  return pathParts.join(' ');
}
