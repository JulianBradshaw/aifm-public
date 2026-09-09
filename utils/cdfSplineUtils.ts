// CDF Spline utilities for the ATC Distribution Editor

import { ScaleConfig } from './chartUtils';

export type InterpolationMethod = 'ispline' | 'gamma';

export interface ControlPoint {
  id: string;
  x: number;        // decimal year
  y: number;        // CDF value (0-1)
  isLocked: boolean; // endpoints locked vertically
}

export interface CdfPoint {
  year: number;
  cdf: number;
}

export interface TakeoffCdfPoint {
  timeFromAC: number;
  cdf: number;
}

/**
 * Create an inverse scale function that maps pixel values back to domain values
 */
export function createInverseScale(config: ScaleConfig): (pixel: number) => number {
  const { domain, range, type } = config;
  const [domainMin, domainMax] = domain;
  const [rangeMin, rangeMax] = range;

  if (type === 'log') {
    const logDomainMin = Math.log10(domainMin);
    const logDomainMax = Math.log10(domainMax);
    return (pixel: number) => {
      const t = (pixel - rangeMin) / (rangeMax - rangeMin);
      return Math.pow(10, logDomainMin + t * (logDomainMax - logDomainMin));
    };
  }

  // Linear inverse scale
  return (pixel: number) => {
    const t = (pixel - rangeMin) / (rangeMax - rangeMin);
    return domainMin + t * (domainMax - domainMin);
  };
}

/**
 * Interpolate the CDF value at a given year using linear interpolation
 */
function interpolateCdfAtYear(sorted: CdfPoint[], year: number): number {
  if (sorted.length === 0) return 0;
  if (year <= sorted[0].year) return sorted[0].cdf;
  if (year >= sorted[sorted.length - 1].year) return sorted[sorted.length - 1].cdf;

  for (let i = 0; i < sorted.length - 1; i++) {
    if (year >= sorted[i].year && year <= sorted[i + 1].year) {
      const t = (year - sorted[i].year) / (sorted[i + 1].year - sorted[i].year);
      return sorted[i].cdf + t * (sorted[i + 1].cdf - sorted[i].cdf);
    }
  }
  return sorted[sorted.length - 1].cdf;
}

/**
 * Find the year where the CDF reaches a target value using linear interpolation
 */
function findYearForCdf(sorted: CdfPoint[], targetCdf: number): number | null {
  if (sorted.length < 2) return null;
  if (targetCdf < sorted[0].cdf || targetCdf > sorted[sorted.length - 1].cdf) return null;

  for (let i = 0; i < sorted.length - 1; i++) {
    if (sorted[i].cdf <= targetCdf && sorted[i + 1].cdf >= targetCdf) {
      const cdfRange = sorted[i + 1].cdf - sorted[i].cdf;
      if (cdfRange === 0) return sorted[i].year;
      const t = (targetCdf - sorted[i].cdf) / cdfRange;
      return sorted[i].year + t * (sorted[i + 1].year - sorted[i].year);
    }
  }
  return null;
}

/**
 * Initialize control points from empirical CDF data.
 * Creates 5 meaningful points:
 *   1. Zero anchor at 2026.1
 *   2. Early point at 2028 (value from empirical CDF)
 *   3. Median (year where CDF = 0.5)
 *   4. Late point at 2050 (value from empirical CDF)
 *   5. 90th percentile (year where CDF = 0.9)
 */
export function initializeControlPoints(
  cdfData: CdfPoint[],
): ControlPoint[] {
  if (cdfData.length === 0) {
    return [];
  }

  const sorted = [...cdfData].sort((a, b) => a.year - b.year);
  const now = Date.now();
  const points: ControlPoint[] = [];

  // 1. Zero anchor at 2026.1
  points.push({ id: `cp-0-${now}`, x: 2026.1, y: 0, isLocked: true });

  // 2. Point at 2028 from empirical CDF
  const cdfAt2028 = interpolateCdfAtYear(sorted, 2028);
  points.push({ id: `cp-1-${now}`, x: 2028, y: cdfAt2028, isLocked: false });

  // 3. Median (year where CDF = 0.5)
  const medianYear = findYearForCdf(sorted, 0.5);
  if (medianYear !== null) {
    points.push({ id: `cp-2-${now}`, x: medianYear, y: 0.5, isLocked: false });
  }

  // 4. Point at 2050 from empirical CDF
  const cdfAt2050 = interpolateCdfAtYear(sorted, 2050);
  points.push({ id: `cp-3-${now}`, x: 2050, y: cdfAt2050, isLocked: false });

  // 5. 90th percentile (year where CDF = 0.9)
  const p90Year = findYearForCdf(sorted, 0.9);
  if (p90Year !== null) {
    points.push({ id: `cp-4-${now}`, x: p90Year, y: 0.9, isLocked: false });
  }

  return points;
}

/**
 * Initialize control points from takeoff speed CDF data (time from AC to milestone)
 * Similar to initializeControlPoints but uses timeFromAC instead of year
 */
export function initializeControlPointsFromTakeoff(
  cdfData: TakeoffCdfPoint[],
  numPoints: number = 7,
  maxTime: number = 25
): ControlPoint[] {
  if (cdfData.length === 0) {
    return [];
  }

  // Sort by time and filter to visible range
  const sorted = [...cdfData].sort((a, b) => a.timeFromAC - b.timeFromAC);
  const visibleData = sorted.filter(p => p.timeFromAC <= maxTime);

  if (visibleData.length === 0) {
    return [];
  }

  // Get the CDF range within the visible data
  const minCdf = visibleData[0].cdf;
  const maxCdf = visibleData[visibleData.length - 1].cdf;

  // Define quantiles for initial control points, scaled to the visible CDF range
  const baseQuantiles = numPoints === 7
    ? [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1]
    : Array.from({ length: numPoints }, (_, i) => i / (numPoints - 1));

  // Scale quantiles to the actual CDF range in visible data
  const scaledQuantiles = baseQuantiles.map(q => minCdf + q * (maxCdf - minCdf));

  // Helper to find closest point to target CDF
  const findClosest = (targetCdf: number): TakeoffCdfPoint => {
    let closest = visibleData[0];
    let minDiff = Math.abs(visibleData[0].cdf - targetCdf);
    for (const point of visibleData) {
      const diff = Math.abs(point.cdf - targetCdf);
      if (diff < minDiff) {
        minDiff = diff;
        closest = point;
      }
    }
    return closest;
  };

  return scaledQuantiles.map((targetCdf, i) => {
    const closest = findClosest(targetCdf);
    return {
      id: `cp-${i}-${Date.now()}`,
      x: closest.timeFromAC,
      y: closest.cdf,
      isLocked: i === 0, // Only lock the first point at 0%
    };
  });
}

/**
 * Enforce monotonicity constraint on control points
 * Ensures CDF is always non-decreasing
 */
export function enforceMonotonicity(points: ControlPoint[]): ControlPoint[] {
  const sorted = [...points].sort((a, b) => a.x - b.x);
  const result: ControlPoint[] = [];

  let minY = 0;
  for (const point of sorted) {
    const constrainedY = Math.max(point.y, minY);
    result.push({ ...point, y: Math.min(constrainedY, 1) });
    minY = constrainedY;
  }

  return result;
}

/**
 * Clamp a value between min and max
 */
export function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

/**
 * Constrain a control point's X position between its neighbors
 */
export function constrainXBetweenNeighbors(
  points: ControlPoint[],
  pointId: string,
  newX: number,
  minSpacing: number = 0.1
): number {
  const sorted = [...points].sort((a, b) => a.x - b.x);
  const index = sorted.findIndex(p => p.id === pointId);

  if (index === -1) return newX;

  const prevX = index > 0 ? sorted[index - 1].x + minSpacing : -Infinity;
  const nextX = index < sorted.length - 1 ? sorted[index + 1].x - minSpacing : Infinity;

  return clamp(newX, prevX, nextX);
}

// ============================================================================
// I-Spline Implementation for Monotone C² Continuous CDF Interpolation
// ============================================================================
//
// I-splines are integrated M-splines (normalized B-splines). They are:
// - Monotonically non-decreasing by construction
// - C² continuous for cubic (order 4) splines
// - Perfect for CDF fitting
//
// The CDF is represented as: F(x) = a + b * sum(w_i * I_i(x))
// where w_i >= 0 ensures monotonicity, and a, b are scaling parameters.

/**
 * Evaluate a single B-spline basis function using Cox-de Boor recursion
 * @param knots - knot vector
 * @param i - basis function index
 * @param k - order (degree + 1)
 * @param x - evaluation point
 */
function bsplineBasis(knots: number[], i: number, k: number, x: number): number {
  if (k === 1) {
    // Order 1 (degree 0): step function
    return (x >= knots[i] && x < knots[i + 1]) ? 1 : 0;
  }

  const denom1 = knots[i + k - 1] - knots[i];
  const denom2 = knots[i + k] - knots[i + 1];

  let term1 = 0;
  let term2 = 0;

  if (denom1 > 1e-10) {
    term1 = ((x - knots[i]) / denom1) * bsplineBasis(knots, i, k - 1, x);
  }
  if (denom2 > 1e-10) {
    term2 = ((knots[i + k] - x) / denom2) * bsplineBasis(knots, i + 1, k - 1, x);
  }

  return term1 + term2;
}

/**
 * Evaluate M-spline basis function (normalized B-spline)
 * M_i,k(x) = k * B_i,k(x) / (knots[i+k] - knots[i])
 */
function msplineBasis(knots: number[], i: number, k: number, x: number): number {
  const denom = knots[i + k] - knots[i];
  if (denom < 1e-10) return 0;
  return (k / denom) * bsplineBasis(knots, i, k, x);
}

/**
 * Numerical integration for I-spline basis
 */
function isplineBasisNumerical(knots: number[], i: number, k: number, x: number): number {
  if (x <= knots[i]) return 0;
  if (x >= knots[i + k]) return 1;

  // Integrate M-spline from knots[i] to x using Simpson's rule
  const a = knots[i];
  const b = x;
  const n = 50; // number of intervals (must be even for Simpson's)
  const h = (b - a) / n;

  let sum = msplineBasis(knots, i, k, a) + msplineBasis(knots, i, k, b);

  for (let j = 1; j < n; j++) {
    const xj = a + j * h;
    const coef = (j % 2 === 0) ? 2 : 4;
    sum += coef * msplineBasis(knots, i, k, xj);
  }

  return (h / 3) * sum;
}

/**
 * Create knot vector for cubic I-splines
 * Adds midpoint knots between data points to provide enough basis functions
 * for exact interpolation with non-negative weights.
 */
function createKnotVector(xValues: number[], order: number = 4): number[] {
  const n = xValues.length;
  const xMin = xValues[0];
  const xMax = xValues[n - 1];

  const knots: number[] = [];

  // Repeated boundary knots at start
  for (let i = 0; i < order; i++) {
    knots.push(xMin);
  }

  // Interior knots: at data points plus midpoints between consecutive points
  // This provides more basis functions for better interpolation
  for (let i = 1; i < n - 1; i++) {
    // Add midpoint before this data point
    const midpoint = (xValues[i - 1] + xValues[i]) / 2;
    knots.push(midpoint);
    // Add knot at data point
    knots.push(xValues[i]);
  }
  // Add final midpoint
  knots.push((xValues[n - 2] + xValues[n - 1]) / 2);

  // Repeated boundary knots at end
  for (let i = 0; i < order; i++) {
    knots.push(xMax);
  }

  return knots;
}

/**
 * Compute number of I-spline basis functions for given knots and order
 */
function numBasisFunctions(knots: number[], order: number): number {
  return knots.length - order;
}

// ============================================================================
// Lawson-Hanson NNLS (Non-Negative Least Squares) Solver
// ============================================================================

/**
 * Solve a linear system using QR decomposition (for small systems)
 * Returns null if the system is singular
 */
function solveLinearSystemQR(A: number[][], b: number[]): number[] | null {
  const m = A.length;
  const n = A[0]?.length ?? 0;

  if (m === 0 || n === 0) return null;

  // Make copies
  const Q: number[][] = A.map(row => [...row]);
  const R: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));

  // Modified Gram-Schmidt QR decomposition
  for (let j = 0; j < n; j++) {
    // Compute column j of Q
    let norm = 0;
    for (let i = 0; i < m; i++) {
      norm += Q[i][j] * Q[i][j];
    }
    norm = Math.sqrt(norm);

    if (norm < 1e-12) {
      // Singular column
      R[j][j] = 0;
      continue;
    }

    R[j][j] = norm;
    for (let i = 0; i < m; i++) {
      Q[i][j] /= norm;
    }

    // Orthogonalize remaining columns
    for (let k = j + 1; k < n; k++) {
      let dot = 0;
      for (let i = 0; i < m; i++) {
        dot += Q[i][j] * Q[i][k];
      }
      R[j][k] = dot;
      for (let i = 0; i < m; i++) {
        Q[i][k] -= dot * Q[i][j];
      }
    }
  }

  // Compute Q^T * b
  const QtB: number[] = new Array(n).fill(0);
  for (let j = 0; j < n; j++) {
    for (let i = 0; i < m; i++) {
      QtB[j] += Q[i][j] * b[i];
    }
  }

  // Back substitution to solve R * x = Q^T * b
  const x: number[] = new Array(n).fill(0);
  for (let i = n - 1; i >= 0; i--) {
    if (Math.abs(R[i][i]) < 1e-12) {
      x[i] = 0; // Singular, set to 0
      continue;
    }
    let sum = QtB[i];
    for (let j = i + 1; j < n; j++) {
      sum -= R[i][j] * x[j];
    }
    x[i] = sum / R[i][i];
  }

  return x;
}

/**
 * Lawson-Hanson algorithm for Non-Negative Least Squares (NNLS)
 * Solves: min ||Ax - b||^2 subject to x >= 0
 *
 * This is the standard algorithm from "Solving Least Squares Problems"
 * by Lawson & Hanson (1974), Chapter 23.
 */
function nnls(A: number[][], b: number[]): number[] {
  const m = A.length;
  const n = A[0]?.length ?? 0;

  if (m === 0 || n === 0) return [];

  // Initialize
  const x: number[] = new Array(n).fill(0);
  const P: Set<number> = new Set(); // Positive set (can be > 0)
  const Z: Set<number> = new Set(); // Zero set (must be 0)

  // All variables start in zero set
  for (let j = 0; j < n; j++) {
    Z.add(j);
  }

  const maxIter = 3 * n; // Standard bound for NNLS

  for (let iter = 0; iter < maxIter; iter++) {
    // Compute gradient w = A^T * (b - Ax)
    const residual: number[] = [...b];
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < n; j++) {
        residual[i] -= A[i][j] * x[j];
      }
    }

    const w: number[] = new Array(n).fill(0);
    for (let j = 0; j < n; j++) {
      for (let i = 0; i < m; i++) {
        w[j] += A[i][j] * residual[i];
      }
    }

    // Find maximum gradient among zero set variables
    let maxW = -Infinity;
    let maxJ = -1;
    for (const j of Z) {
      if (w[j] > maxW) {
        maxW = w[j];
        maxJ = j;
      }
    }

    // Check termination: all gradients in Z are <= 0
    if (maxW <= 1e-10 || maxJ === -1) {
      break;
    }

    // Move variable with maximum gradient from Z to P
    Z.delete(maxJ);
    P.add(maxJ);

    // Inner loop: solve unconstrained problem and handle negative values
    for (let innerIter = 0; innerIter < maxIter; innerIter++) {
      // Build subproblem: solve for variables in P only
      const pIndices = Array.from(P);
      const nP = pIndices.length;

      if (nP === 0) break;

      // Extract sub-matrix A_P and solve A_P * s_P = b
      const Ap: number[][] = [];
      for (let i = 0; i < m; i++) {
        const row: number[] = [];
        for (const j of pIndices) {
          row.push(A[i][j]);
        }
        Ap.push(row);
      }

      const sP = solveLinearSystemQR(Ap, b);

      if (!sP) {
        // Singular system, break inner loop
        break;
      }

      // Check if all s_P values are positive
      let allPositive = true;
      for (let k = 0; k < nP; k++) {
        if (sP[k] <= 1e-12) {
          allPositive = false;
          break;
        }
      }

      if (allPositive) {
        // Accept the solution
        for (let k = 0; k < nP; k++) {
          x[pIndices[k]] = sP[k];
        }
        break;
      }

      // Find alpha: step length to first zero crossing
      let alpha = 1.0;
      let blockingVar = -1;

      for (let k = 0; k < nP; k++) {
        const j = pIndices[k];
        if (sP[k] <= 1e-12) {
          const ratio = x[j] / (x[j] - sP[k] + 1e-12);
          if (ratio < alpha) {
            alpha = ratio;
            blockingVar = j;
          }
        }
      }

      // Update x = x + alpha * (s - x)
      for (let k = 0; k < nP; k++) {
        const j = pIndices[k];
        x[j] = x[j] + alpha * (sP[k] - x[j]);
      }

      // Move blocking variables from P to Z
      for (const j of Array.from(P)) {
        if (x[j] <= 1e-12) {
          x[j] = 0;
          P.delete(j);
          Z.add(j);
        }
      }

      if (P.size === 0) break;
    }
  }

  return x;
}

/**
 * Fit I-spline to data points using NNLS with smoothness regularization
 * Returns coefficients w_i >= 0
 *
 * @param smoothWeight - Regularization weight for second-difference penalty (0 = no regularization)
 */
function fitIspline(
  xData: number[],
  yData: number[],
  knots: number[],
  order: number,
  smoothWeight: number = 0.05
): number[] {
  const nBasis = numBasisFunctions(knots, order);
  const nData = xData.length;

  // Build design matrix: A[i][j] = I_j(x_i)
  const A: number[][] = [];
  for (let i = 0; i < nData; i++) {
    const row: number[] = [];
    for (let j = 0; j < nBasis; j++) {
      row.push(isplineBasisNumerical(knots, j, order, xData[i]));
    }
    A.push(row);
  }

  const a = yData[0];
  const b = yData[nData - 1];
  const range = b - a;

  if (range < 1e-10) {
    return new Array(nBasis).fill(1 / nBasis);
  }

  const z: number[] = yData.map(y => (y - a) / range);

  // Add smoothness regularization to reduce PDF oscillations
  const augA: number[][] = [...A];
  const augB: number[] = [...z];

  if (smoothWeight > 0) {
    // Second-difference penalty on weights
    for (let i = 0; i < nBasis - 2; i++) {
      const row = new Array(nBasis).fill(0);
      row[i] = smoothWeight;
      row[i + 1] = -2 * smoothWeight;
      row[i + 2] = smoothWeight;
      augA.push(row);
      augB.push(0);
    }
  }

  // Solve with NNLS
  const w = nnls(augA, augB);

  // Normalize
  const sumW = w.reduce((s, v) => s + v, 0);
  if (sumW > 1e-10) {
    for (let j = 0; j < w.length; j++) {
      w[j] /= sumW;
    }
  } else {
    return new Array(nBasis).fill(1 / nBasis);
  }

  return w;
}

/**
 * I-spline model that encapsulates knots, weights, and scaling
 */
interface ISplineModel {
  knots: number[];
  order: number;
  weights: number[];
  yMin: number;
  yMax: number;
}

/**
 * Build I-spline model from control points
 */
function buildISplineModel(sorted: ControlPoint[], smoothWeight: number = 0.05): ISplineModel {
  const n = sorted.length;
  const order = 4; // cubic

  if (n < 2) {
    return {
      knots: [0, 1],
      order: 4,
      weights: [1],
      yMin: 0,
      yMax: 1,
    };
  }

  const xData = sorted.map(p => p.x);
  const yData = sorted.map(p => p.y);

  const knots = createKnotVector(xData, order);
  const weights = fitIspline(xData, yData, knots, order, smoothWeight);

  return {
    knots,
    order,
    weights,
    yMin: yData[0],
    yMax: yData[n - 1],
  };
}

/**
 * Evaluate I-spline model at a given x
 */
function evaluateISplineModel(model: ISplineModel, x: number): number {
  const { knots, order, weights, yMin, yMax } = model;
  const nBasis = weights.length;

  // Clamp x to the domain
  const xMin = knots[order - 1];
  const xMax = knots[knots.length - order];

  if (x <= xMin) return yMin;
  if (x >= xMax) return yMax;

  // Compute weighted sum of I-spline basis functions
  let sum = 0;
  for (let i = 0; i < nBasis; i++) {
    sum += weights[i] * isplineBasisNumerical(knots, i, order, x);
  }

  // Scale to [yMin, yMax]
  return yMin + (yMax - yMin) * sum;
}

/**
 * Evaluate I-spline model derivative (PDF) at a given x
 */
function evaluateISplineModelDerivative(model: ISplineModel, x: number): number {
  const { knots, order, weights, yMin, yMax } = model;
  const nBasis = weights.length;

  // Clamp x to the domain
  const xMin = knots[order - 1];
  const xMax = knots[knots.length - order];

  if (x <= xMin || x >= xMax) return 0;

  // Compute weighted sum of M-spline (I-spline derivative) basis functions
  let sum = 0;
  for (let i = 0; i < nBasis; i++) {
    sum += weights[i] * msplineBasis(knots, i, order, x);
  }

  // Scale by the range
  return (yMax - yMin) * sum;
}

/**
 * Monotone C² continuous CDF interpolation using I-splines.
 *
 * I-splines (Integrated M-splines) are:
 * - Monotonically non-decreasing by construction (when weights are non-negative)
 * - C² continuous for cubic (order 4) splines
 * - Perfect for CDF fitting
 */
export function evaluateSplineAt(points: ControlPoint[], x: number, smoothWeight: number = 0.05): number {
  const sorted = [...points].sort((a, b) => a.x - b.x);

  if (sorted.length === 0) return 0;
  if (sorted.length === 1) return sorted[0].y;

  // Handle x outside the range
  if (x <= sorted[0].x) return sorted[0].y;
  if (x >= sorted[sorted.length - 1].x) return sorted[sorted.length - 1].y;

  // Build and evaluate I-spline model
  const model = buildISplineModel(sorted, smoothWeight);
  const y = evaluateISplineModel(model, x);

  // Clamp to valid CDF range
  return clamp(y, 0, 1);
}

/**
 * Evaluate the PDF (derivative of the CDF) at a given x value.
 * This computes the derivative of the I-spline, which is a linear combination of M-splines.
 * M-splines are non-negative, so the PDF is guaranteed non-negative when weights are non-negative.
 */
export function evaluatePdfAt(points: ControlPoint[], x: number, smoothWeight: number = 0.05): number {
  const sorted = [...points].sort((a, b) => a.x - b.x);

  if (sorted.length < 2) return 0;

  // Handle x outside the range - PDF is 0 outside
  if (x < sorted[0].x || x > sorted[sorted.length - 1].x) return 0;

  // Build and evaluate I-spline model derivative
  const model = buildISplineModel(sorted, smoothWeight);
  const pdf = evaluateISplineModelDerivative(model, x);

  // PDF should be non-negative (guaranteed by I-spline construction, but clamp for safety)
  return Math.max(0, pdf);
}

/**
 * Sample the PDF at regular intervals for visualization
 */
export function samplePdf(
  controlPoints: ControlPoint[],
  numSamples: number = 200,
  smoothWeight: number = 0.05
): { x: number; pdf: number }[] {
  if (controlPoints.length < 2) return [];

  const sorted = [...controlPoints].sort((a, b) => a.x - b.x);
  const minX = sorted[0].x;
  const maxX = sorted[sorted.length - 1].x;

  const samples: { x: number; pdf: number }[] = [];

  for (let i = 0; i <= numSamples; i++) {
    const x = minX + (maxX - minX) * (i / numSamples);
    samples.push({
      x,
      pdf: evaluatePdfAt(sorted, x, smoothWeight),
    });
  }

  return samples;
}

/**
 * Sample the spline at regular intervals for export
 */
export function sampleSplineForExport(
  controlPoints: ControlPoint[],
  timePoints: number[]
): CdfPoint[] {
  return timePoints.map(year => ({
    year,
    cdf: evaluateSplineAt(controlPoints, year),
  }));
}

// The CDFs this operates on are exported on a monthly grid
const MONTH_IN_YEARS = 1 / 12;

/**
 * Compute PDF from empirical CDF data using numerical differentiation.
 * Uses centered differences for interior points and forward/backward differences at boundaries.
 *
 * Samples on a monthly step, which is the resolution these CDFs are stored at.
 * A fixed sample *count* spread across the extent made the step - and with it
 * the width of the centered difference, and so how much the resulting PDF is
 * smoothed - depend on how far the curve's tail happened to reach: an ATC CDF
 * running to 2182 was differentiated over 3.7 months, flattening the part of
 * the curve on screen, while a short curve got a fine step. Monthly is the same
 * everywhere and matches the data.
 */
export function computePdfFromCdf(cdfData: CdfPoint[]): { x: number; pdf: number }[] {
  if (cdfData.length < 2) return [];

  // Sort by year
  const sorted = [...cdfData].sort((a, b) => a.year - b.year);
  const minX = sorted[0].year;
  const maxX = sorted[sorted.length - 1].year;

  // Helper to interpolate CDF value at any x
  function interpolateCdf(x: number): number {
    if (x <= sorted[0].year) return sorted[0].cdf;
    if (x >= sorted[sorted.length - 1].year) return sorted[sorted.length - 1].cdf;

    // Binary search for the interval
    let lo = 0;
    let hi = sorted.length - 1;
    while (hi - lo > 1) {
      const mid = Math.floor((lo + hi) / 2);
      if (sorted[mid].year <= x) {
        lo = mid;
      } else {
        hi = mid;
      }
    }

    // Linear interpolation
    const x0 = sorted[lo].year;
    const x1 = sorted[hi].year;
    const y0 = sorted[lo].cdf;
    const y1 = sorted[hi].cdf;
    const t = (x - x0) / (x1 - x0);
    return y0 + t * (y1 - y0);
  }

  const samples: { x: number; pdf: number }[] = [];
  // Whole months across the extent, so the step lands on ~1/12 without leaving
  // a ragged final interval. At least one step for extents under a month.
  const numSamples = Math.max(1, Math.round((maxX - minX) / MONTH_IN_YEARS));
  const h = (maxX - minX) / numSamples;

  for (let i = 0; i <= numSamples; i++) {
    const x = minX + (maxX - minX) * (i / numSamples);

    // Compute derivative using centered differences
    const epsilon = h * 0.5;
    const cdfPlus = interpolateCdf(x + epsilon);
    const cdfMinus = interpolateCdf(x - epsilon);
    const pdf = Math.max(0, (cdfPlus - cdfMinus) / (2 * epsilon));

    samples.push({ x, pdf });
  }

  return samples;
}

/**
 * Add a new control point at the end, extending the x range
 */
export function addControlPoint(points: ControlPoint[]): ControlPoint[] {
  if (points.length < 1) return points;

  const sorted = [...points].sort((a, b) => a.x - b.x);
  const lastPoint = sorted[sorted.length - 1];
  const secondLastPoint = sorted.length > 1 ? sorted[sorted.length - 2] : null;

  let newX: number;
  let newY: number;

  if (secondLastPoint) {
    // Extend by the same step as the last two points
    const step = lastPoint.x - secondLastPoint.x;
    newX = lastPoint.x + step;
    // Increase y slightly, but cap at 1.0
    newY = Math.min(1.0, lastPoint.y + (lastPoint.y - secondLastPoint.y) * 0.5);
    // Ensure y is at least as high as the last point (CDF must be monotonic)
    newY = Math.max(lastPoint.y, newY);
  } else {
    // Only one point exists, add some reasonable offset
    newX = lastPoint.x + 5;
    newY = Math.min(1.0, lastPoint.y + 0.1);
  }

  const newPoint: ControlPoint = {
    id: `cp-new-${Date.now()}`,
    x: Math.round(newX * 100) / 100,
    y: Math.round(newY * 1000) / 1000,
    isLocked: false,
  };

  return [...points, newPoint];
}

/**
 * Remove a control point by ID (cannot remove locked points)
 */
export function removeControlPoint(points: ControlPoint[], pointId: string): ControlPoint[] {
  const point = points.find(p => p.id === pointId);
  if (!point || point.isLocked) return points;

  return points.filter(p => p.id !== pointId);
}

/**
 * Generate SVG path string for a spline through control points
 */
export function generateSplinePath(
  controlPoints: ControlPoint[],
  xScale: (value: number) => number,
  yScale: (value: number) => number,
  numSamples: number = 100,
  smoothWeight: number = 0.05
): string {
  if (controlPoints.length < 2) return '';

  const sorted = [...controlPoints].sort((a, b) => a.x - b.x);
  const minX = sorted[0].x;
  const maxX = sorted[sorted.length - 1].x;

  const pathParts: string[] = [];

  for (let i = 0; i <= numSamples; i++) {
    const x = minX + (maxX - minX) * (i / numSamples);
    const y = evaluateSplineAt(sorted, x, smoothWeight);
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
