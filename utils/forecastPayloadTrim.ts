/**
 * Trims the forecast series down to what the charts can actually draw, applied
 * at the point where data is handed to the client component.
 *
 * The exported CSVs cover the full support of each distribution - the milestone
 * PDF grid runs to ~2182 and the takeoff grid to ~188 years after AC, both at
 * monthly resolution - while the charts cap their display at 2050 and at 10
 * years after AC respectively. Shipped verbatim, ~80% of the timeline points and
 * ~95% of the takeoff points are off-screen, and every density arrives at full
 * float64 precision (20 characters to describe a sub-pixel quantity).
 *
 * So: full resolution inside the displayed window, one point per year beyond it,
 * and numbers rounded to what a pixel can distinguish. The CSVs stay
 * authoritative - this only shapes the payload.
 *
 * IMPORTANT: trim last. Several server-side derivations read these arrays as a
 * uniform grid (the takeoff y-domain sums raw densities and locates an
 * "empirical median" by cumulative fraction) or interpolate them for displayed
 * statistics (computeAtcStatistics, the arrival/takeoff probability tables).
 * Decimating before those run would shift their results; decimating after is
 * invisible to them.
 */

import { TIMELINE_DISPLAY_MAX_YEAR, TAKEOFF_DISPLAY_MAX_YEARS } from '@/constants/forecastDisplay';

// Beyond the ceiling, keep roughly this many points per year. The tail is kept
// rather than clipped so the shipped series still describes the whole
// distribution - it costs ~1% of the savings - and so lifting a display ceiling
// doesn't silently truncate a curve.
const TAIL_POINTS_PER_YEAR = 1;

// A density of 8.399142664507132e-3 renders identically to 8.39914e-3.
const DENSITY_SIG_DIGITS = 6;
// CDFs live in [0, 1]; 1e-6 absolute is far below a pixel, and the client
// derives PDFs from CDFs by finite difference, where this leaves ~1e-5 of
// density noise against values of order 1e-2.
const CDF_DECIMALS = 6;
// ~1 hour of calendar precision on a monthly grid.
const YEAR_DECIMALS = 4;
const TIME_FROM_AC_DECIMALS = 5;

function roundSig(value: number, digits: number): number {
  if (!Number.isFinite(value) || value === 0) return value;
  return Number(value.toPrecision(digits));
}

function roundFixed(value: number, decimals: number): number {
  if (!Number.isFinite(value)) return value;
  return Number(value.toFixed(decimals));
}

/**
 * Keeps every point at or below `cap`, then thins what follows to
 * TAIL_POINTS_PER_YEAR. The last point is always kept so the series still ends
 * where the distribution does.
 */
function decimateTail<T>(points: T[], cap: number, xOf: (point: T) => number): T[] {
  if (points.length === 0) return points;

  const kept: T[] = [];
  let lastTailX = -Infinity;
  const tailSpacing = 1 / TAIL_POINTS_PER_YEAR;

  for (let i = 0; i < points.length; i += 1) {
    const x = xOf(points[i]);
    if (x <= cap) {
      kept.push(points[i]);
      continue;
    }
    // Always keep the final point, so the tail doesn't end early
    if (i === points.length - 1 || x - lastTailX >= tailSpacing) {
      kept.push(points[i]);
      lastTailX = x;
    }
  }

  return kept;
}

type YearPdfPoint = { year: number; probabilityDensity: number };
type YearCdfPoint = { year: number; cdf: number };
type TimePdfPoint = { timeFromAC: number; probabilityDensity: number };
type TimeCdfPoint = { timeFromAC: number; cdf: number };

function trimYearPdf(points: YearPdfPoint[]): YearPdfPoint[] {
  return decimateTail(points, TIMELINE_DISPLAY_MAX_YEAR, p => p.year).map(p => ({
    year: roundFixed(p.year, YEAR_DECIMALS),
    probabilityDensity: roundSig(p.probabilityDensity, DENSITY_SIG_DIGITS),
  }));
}

function trimYearCdf(points: YearCdfPoint[]): YearCdfPoint[] {
  return decimateTail(points, TIMELINE_DISPLAY_MAX_YEAR, p => p.year).map(p => ({
    year: roundFixed(p.year, YEAR_DECIMALS),
    cdf: roundFixed(p.cdf, CDF_DECIMALS),
  }));
}

function trimTimePdf(points: TimePdfPoint[]): TimePdfPoint[] {
  return decimateTail(points, TAKEOFF_DISPLAY_MAX_YEARS, p => p.timeFromAC).map(p => ({
    timeFromAC: roundFixed(p.timeFromAC, TIME_FROM_AC_DECIMALS),
    probabilityDensity: roundSig(p.probabilityDensity, DENSITY_SIG_DIGITS),
  }));
}

function trimTimeCdf(points: TimeCdfPoint[]): TimeCdfPoint[] {
  return decimateTail(points, TAKEOFF_DISPLAY_MAX_YEARS, p => p.timeFromAC).map(p => ({
    timeFromAC: roundFixed(p.timeFromAC, TIME_FROM_AC_DECIMALS),
    cdf: roundFixed(p.cdf, CDF_DECIMALS),
  }));
}

function mapSeries<T>(
  series: { [key: string]: T[] } | undefined,
  trim: (points: T[]) => T[]
): { [key: string]: T[] } | undefined {
  if (!series) return series;
  return Object.fromEntries(Object.entries(series).map(([key, points]) => [key, trim(points)]));
}

/** Milestone-arrival PDFs/CDFs, keyed by calendar year. */
export function trimMilestoneDistributions<
  T extends { milestones: { [key: string]: YearPdfPoint[] }; cdf?: { [key: string]: YearCdfPoint[] } }
>(data: T): T {
  return {
    ...data,
    milestones: mapSeries(data.milestones, trimYearPdf)!,
    cdf: mapSeries(data.cdf, trimYearCdf),
  };
}

/** Takeoff PDFs/CDFs, keyed by years after AC. */
export function trimTimingDistributions<
  T extends { milestones: { [key: string]: TimePdfPoint[] }; cdf?: { [key: string]: TimeCdfPoint[] } }
>(data: T): T {
  return {
    ...data,
    milestones: mapSeries(data.milestones, trimTimePdf)!,
    cdf: mapSeries(data.cdf, trimTimeCdf),
  };
}

/** All-things-considered CDF overlay on the milestone chart. */
export function trimAtcCdf<T extends { [key: string]: YearCdfPoint[] }>(data: T | null): T | null {
  if (!data) return data;
  return mapSeries(data, trimYearCdf) as T;
}

/** All-things-considered CDF overlay on the takeoff chart. */
export function trimAtcTakeoffCdf<T extends { [key: string]: TimeCdfPoint[] }>(data: T | null): T | null {
  if (!data) return data;
  return mapSeries(data, trimTimeCdf) as T;
}

/** Per-AC-anchor-mode breakout: the same milestone series, once per mode. */
export function trimModeBreakout<
  M extends { milestones: { [key: string]: YearPdfPoint[] }; cdf: { [key: string]: YearCdfPoint[] } },
  T extends { perMode: { [mode: string]: M } }
>(data: T | null): T | null {
  if (!data) return data;
  return {
    ...data,
    perMode: Object.fromEntries(
      Object.entries(data.perMode).map(([mode, modeData]) => [
        mode,
        {
          ...modeData,
          milestones: mapSeries(modeData.milestones, trimYearPdf)!,
          cdf: mapSeries(modeData.cdf, trimYearCdf)!,
        },
      ])
    ),
  };
}
