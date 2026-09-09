// Chart utilities for custom SVG charts

export interface ChartMargin {
  top: number;
  right: number;
  bottom: number;
  left: number;
}

export interface ScaleConfig {
  domain: [number, number];
  range: [number, number];
  type: 'linear' | 'log';
}

/**
 * Create a scale function that maps domain values to range values
 */
export function createScale(config: ScaleConfig): (value: number) => number {
  const { domain, range, type } = config;
  const [domainMin, domainMax] = domain;
  const [rangeMin, rangeMax] = range;

  if (type === 'log') {
    return (value: number) => {
      if (value <= 0) return rangeMin;
      const logDomainMin = Math.log10(domainMin);
      const logDomainMax = Math.log10(domainMax);
      const logValue = Math.log10(value);
      const t = (logValue - logDomainMin) / (logDomainMax - logDomainMin);
      return rangeMin + t * (rangeMax - rangeMin);
    };
  }

  // Linear scale
  return (value: number) => {
    const t = (value - domainMin) / (domainMax - domainMin);
    return rangeMin + t * (rangeMax - rangeMin);
  };
}

/**
 * Generate SVG path string for a line chart
 */
export function generateLinePath(
  data: Array<{ x: number; y: number | null | undefined }>,
  xScale: (value: number) => number,
  yScale: (value: number) => number,
  smooth: boolean = false
): string {
  if (data.length === 0) return '';

  const validPoints = data.filter(d => d.y != null && !isNaN(d.y));
  if (validPoints.length === 0) return '';

  if (!smooth) {
    // Linear interpolation (existing behavior)
    const pathParts: string[] = [];

    validPoints.forEach((point, i) => {
      const x = xScale(point.x);
      const y = yScale(point.y!);

      if (i === 0) {
        pathParts.push(`M ${x} ${y}`);
      } else {
        pathParts.push(`L ${x} ${y}`);
      }
    });

    return pathParts.join(' ');
  }

  // Smooth curve interpolation using Catmull-Rom splines
  const scaledPoints = validPoints.map(point => ({
    x: xScale(point.x),
    y: yScale(point.y!),
  }));

  if (scaledPoints.length === 1) {
    return `M ${scaledPoints[0].x} ${scaledPoints[0].y}`;
  }

  if (scaledPoints.length === 2) {
    return `M ${scaledPoints[0].x} ${scaledPoints[0].y} L ${scaledPoints[1].x} ${scaledPoints[1].y}`;
  }

  const pathParts: string[] = [];
  pathParts.push(`M ${scaledPoints[0].x} ${scaledPoints[0].y}`);

  // Use cubic bezier curves for smooth interpolation
  for (let i = 0; i < scaledPoints.length - 1; i++) {
    const p0 = i > 0 ? scaledPoints[i - 1] : scaledPoints[i];
    const p1 = scaledPoints[i];
    const p2 = scaledPoints[i + 1];
    const p3 = i < scaledPoints.length - 2 ? scaledPoints[i + 2] : p2;

    // Catmull-Rom to Bezier conversion
    const tension = 0.5; // Controls curve smoothness (0.5 is standard Catmull-Rom)

    const cp1x = p1.x + (p2.x - p0.x) / 6 * tension;
    const cp1y = p1.y + (p2.y - p0.y) / 6 * tension;
    const cp2x = p2.x - (p3.x - p1.x) / 6 * tension;
    const cp2y = p2.y - (p3.y - p1.y) / 6 * tension;

    pathParts.push(`C ${cp1x} ${cp1y}, ${cp2x} ${cp2y}, ${p2.x} ${p2.y}`);
  }

  return pathParts.join(' ');
}

/**
 * Calculate nice tick values for an axis
 */
export function calculateTicks(
  domain: [number, number],
  config: { targetCount: number } | { stepSize: number },
  type: 'linear' | 'log',
): number[] {

  const [min, max] = domain;

  if (type === 'log') {
    const logMin = Math.log10(min);
    const logMax = Math.log10(max);
    const ticks: number[] = [];

    // Generate ticks at powers of 10
    for (let i = Math.floor(logMin); i <= Math.ceil(logMax); i++) {
      const value = Math.pow(10, i);
      if (value >= min && value <= max) {
        ticks.push(value);
      }
    }

    return ticks;
  }

  // Linear ticks
  const range = max - min;
  const step = 'stepSize' in config ? config.stepSize : (range / config.targetCount);

  const ticks: number[] = [];
  const start = Math.ceil(min / step) * step;

  for (let value = start; value <= max; value += step) {
    ticks.push(value);
  }

  return ticks;
}

/**
 * Thin x-axis ticks so their rendered labels don't overlap.
 *
 * Labels are drawn centered on each tick, so two adjacent labels collide when the
 * gap between tick positions is smaller than the sum of their half-widths. Widths
 * are estimated from character count (~0.6em per character for typical UI fonts).
 * Returns an evenly-strided subset (every nth tick) so label spacing stays regular.
 */
export function thinTicksToFitLabels(
  ticks: number[],
  xScale: (value: number) => number,
  formatter: (value: number) => string,
  fontSize: number = 12,
  minGap: number = 12
): number[] {
  if (ticks.length <= 2) return ticks;

  const n = ticks.length;
  const halfWidths = ticks.map(tick => (formatter(tick).length * fontSize * 0.6) / 2);

  // The first and last ticks are always kept: they anchor the axis range,
  // which matters most on exactly the narrow charts where thinning engages.
  // Only the interior is thinned (every stride-th tick), so try each stride
  // and keep the densest subset whose labels don't collide.
  for (let stride = 1; stride < n; stride++) {
    const kept: number[] = [];
    for (let i = 0; i < n - 1; i += stride) kept.push(i);
    kept.push(n - 1);

    let fits = true;
    for (let k = 1; k < kept.length; k++) {
      const a = kept[k - 1];
      const b = kept[k];
      const gap = xScale(ticks[b]) - xScale(ticks[a]);
      if (gap - halfWidths[a] - halfWidths[b] < minGap) {
        fits = false;
        break;
      }
    }
    if (fits) {
      return kept.map(i => ticks[i]);
    }
  }

  // Nothing denser fits: fall back to the two endpoints, which sit at
  // opposite edges and (re-anchored inward by the axis renderer) can
  // essentially always coexist.
  return [ticks[0], ticks[n - 1]];
}

/**
 * Calculate log-scale ticks using 1-2-5 subdivision for better coverage within a domain.
 * This keeps tick marks aligned with human-friendly values even when the domain spans
 * partial decades.
 */
export function calculateLogTicksWithSubdivisions(
  domain: [number, number],
  targetCount: number = 5
): number[] {
  const [min, max] = domain;

  if (min <= 0 || max <= 0 || min >= max) {
    return [];
  }

  const logMin = Math.floor(Math.log10(min));
  const logMax = Math.ceil(Math.log10(max));
  const rawTicks: number[] = [];

  for (let exponent = logMin; exponent <= logMax; exponent++) {
    const decadeBase = Math.pow(10, exponent);
    const candidates = [1, 2, 5];

    for (const multiple of candidates) {
      const tickValue = decadeBase * multiple;
      if (tickValue >= min && tickValue <= max) {
        rawTicks.push(tickValue);
      }
    }
  }

  rawTicks.sort((a, b) => a - b);

  if (rawTicks.length <= targetCount) {
    return rawTicks;
  }

  const selected = new Set<number>();
  const step = (rawTicks.length - 1) / (targetCount - 1);

  for (let i = 0; i < targetCount; i++) {
    const index = Math.round(i * step);
    selected.add(rawTicks[index]);
  }

  return Array.from(selected).sort((a, b) => a - b);
}

/**
 * Find the closest data point to mouse coordinates
 */
export function findClosestPoint<T extends { x: number; y?: number | null }>(
  data: T[],
  mouseX: number,
  mouseY: number,
  xScale: (value: number) => number,
  yScale: (value: number) => number,
  maxDistance: number = 50
): T | null {
  let closest: T | null = null;
  let minDistance = maxDistance;

  for (const point of data) {
    if (point.y == null || isNaN(point.y)) continue;

    const px = xScale(point.x);
    const py = yScale(point.y);
    const distance = Math.sqrt(Math.pow(mouseX - px, 2) + Math.pow(mouseY - py, 2));

    if (distance < minDistance) {
      minDistance = distance;
      closest = point;
    }
  }

  return closest;
}

/**
 * Clamp a value between min and max
 */
export function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

/**
 * Calculate nice time-based tick values for log scale (in minutes)
 * Returns approximately targetCount tick values at interpretable time intervals
 */
export function calculateTimeLogTicks(
  domain: [number, number],
  targetCount: number = 5,
  workNormalized: boolean = false
): number[] {
  const [min, max] = domain;

  // Define nice time values in minutes
  const niceTimeValues = workNormalized ? [
    1,                      // 1 minute
    5,                      // 5 minutes
    15,                     // 15 minutes
    30,                     // 30 minutes
    60,                     // 1 hour
    180,                    // 3 hours
    360,                    // 6 hours
    720,                    // 12 hours
    8 * 60,                 // 1 day
    24 * 60,                // 3 days
    40 * 60,                // 1 week
    80 * 60,                // 2 weeks
    (2000 / 12) * 60,       // 1 month (30 days)
    (2000 / 12) * 60 * 3,   // 3 months
    (2000 / 12) * 60 * 6,   // 6 months
    2000 * 60,              // 1 year
    2000 * 60 * 3,          // 3 years
    2000 * 60 * 10,         // 10 years
    2000 * 60 * 30,         // 30 years
    2000 * 60 * 100,        // 100 years
    2000 * 60 * 1000,       // 1000 years
    2000 * 60 * 10000       // 10000 years
  ] : [
    1,           // 1 minute
    5,           // 5 minutes
    15,          // 15 minutes
    30,          // 30 minutes
    60,          // 1 hour
    180,         // 3 hours
    360,         // 6 hours
    720,         // 12 hours
    1440,        // 1 day
    4320,        // 3 days
    10080,       // 1 week
    20160,       // 2 weeks
    43200,       // 1 month (30 days)
    129600,      // 3 months
    259200,      // 6 months
    525600,      // 1 year
    1576800,     // 3 years
    5256000,     // 10 years
    15768000,    // 30 years
    52560000,    // 100 years
    525600000,   // 1000 years
    5256000000   // 10000 years
  ];

  // Filter values within domain
  const validValues = niceTimeValues.filter(v => v >= min && v <= max);

  if (validValues.length <= targetCount) {
    return validValues;
  }

  const result: number[] = [];
  const step = validValues.length / targetCount;

  for (let i = 0; i < targetCount && i * step < validValues.length; i++) {
    const index = Math.round(i * step);
    if (index < validValues.length) {
      result.push(validValues[index]);
    }
  }

  return result;
}

export interface DynamicDomainOptions {
  intersectionPoints?: (number | null)[];
  minPadding?: number;
  maxPadding?: number;
  minRange?: number;
  maxDomain?: number; // Hard maximum constraint for domain
}

/**
 * Calculate dynamic X-axis domain that automatically expands to show intersection points
 * with appropriate padding for readability
 */
export function calculateDynamicXDomain(
  dataValues: number[],
  options: DynamicDomainOptions = {}
): [number, number] {
  const {
    intersectionPoints = [],
    minPadding = 0.1,
    maxPadding = 0.2,
    minRange = 1
  } = options;

  // Filter to valid data values
  const validDataValues = dataValues.filter(x => isFinite(x) && !isNaN(x));

  // Filter to valid intersection points
  const validIntersections = intersectionPoints
    .filter((x): x is number => x != null && isFinite(x) && !isNaN(x));

  // Combine all relevant points
  const allPoints = [...validDataValues, ...validIntersections];

  // Fallback if no valid data
  if (allPoints.length === 0) {
    return [0, 100];
  }

  const min = Math.min(...allPoints);
  const rawMax = Math.max(...allPoints);

  // Clamp against the hard maximum BEFORE deriving the range that padding is a
  // fraction of. These charts routinely hold data far past the window they display:
  // the solver integrates to 2200 while maxDomain is the AC/ASI-driven display end
  // year (~2030). Taking a percentage of the *raw* range therefore produced an
  // enormous left margin -- 5% of 181 years pushed the horizon chart's left edge back
  // to ~2010 -- while the right edge stayed pinned at maxDomain, so the visible data
  // was squeezed into the right of the plot behind a wide blank strip. Padding off the
  // displayed range keeps both edges tight. When the data ends before maxDomain this
  // is a no-op, so the right-edge behaviour is unchanged.
  const max = options.maxDomain !== undefined
    ? Math.max(min, Math.min(rawMax, options.maxDomain))
    : rawMax;
  const range = max - min;

  // Enforce minimum range
  if (range < minRange) {
    const center = (max + min) / 2;
    const halfRange = minRange / 2;
    return [center - halfRange, center + halfRange];
  }

  // Calculate adaptive padding
  let paddingPercent = minPadding;

  // Increase padding if any intersection is near boundary
  if (validIntersections.length > 0) {
    const nearBoundary = validIntersections.some(ix => {
      const threshold = range * 0.05; // Within 5% of boundary
      return Math.abs(ix - min) < threshold || Math.abs(ix - max) < threshold;
    });

    if (nearBoundary) {
      paddingPercent = Math.min(maxPadding, minPadding * 1.5);
    }
  }

  const padding = range * paddingPercent;

  const finalMin = min - padding;
  let finalMax = max + padding;

  // Apply hard maximum constraint if provided
  if (options.maxDomain !== undefined) {
    finalMax = Math.min(finalMax, options.maxDomain);
  }

  return [finalMin, finalMax];
}
