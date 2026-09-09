'use client';

import { memo, useMemo, useRef, useCallback, ReactNode } from 'react';
import { CustomLineChart, DataPoint } from './CustomLineChart';
import { PdfLineChart } from './PdfLineChart';
import { ChartSyncProvider } from './ChartSyncContext';
import { ForecastChartDownloadButton } from './ForecastChartDownloadButton';
import { buildExportFilename, formatForecastDateLabel } from '@/utils/forecastChartExport';
import { formatTo3SigFigs, formatYearMonth, formatYearMonthShort } from '@/utils/formatting';
import { tooltipBoxStyle, tooltipHeaderStyle, tooltipValueStyle } from './chartTooltipStyle';
import type { AtcCdfData, AtcStatistics } from '@/app/forecast/[[...simulationId]]/page';
import { computePdfFromCdf, CdfPoint } from '@/utils/cdfSplineUtils';
import { TIMELINE_DISPLAY_MAX_YEAR } from '@/constants/forecastDisplay';
import { ANCHOR_MODE_DISPLAY_NAMES, ANCHOR_MODE_DASH_PATTERNS } from '@/constants/acAnchorModes';

export interface MilestoneDistributionPoint {
  year: number;
  probabilityDensity: number;
}

export interface MilestoneCdfPoint {
  year: number;
  cdf: number;
}

export interface MilestoneDistributionData {
  milestones: { [key: string]: MilestoneDistributionPoint[] };
  statistics: {
    [key: string]: {
      achievementRate: number;
      mode: number;
      p10: number;
      p50: number;
      p90: number;
    };
  };
  cdf?: { [key: string]: MilestoneCdfPoint[] };
}

// Per-AC-anchor-mode breakout data (only present for mixture simulation runs)
export interface ModeBreakoutModeData {
  milestones: { [key: string]: MilestoneDistributionPoint[] };
  cdf: { [key: string]: MilestoneCdfPoint[] };
  statistics: {
    [key: string]: {
      achievementRate: number;
      mode: number;
      p10: number;
      p50: number;
      p90: number;
    };
  };
  // Successful rollouts sampled with this mode (the denominator for this
  // mode's PDF/CDF; failed rollouts are excluded, as in the combined data)
  numRollouts: number;
  // All rollouts sampled with this mode, including failed ones
  numSampled: number;
  numFailed: number;
  // Realized mixture weight: share of ALL sampled rollouts (incl. failed)
  weightPct: number;
}

export interface ModeBreakoutData {
  modes: string[];
  perMode: { [mode: string]: ModeBreakoutModeData };
}

// Display names and dash patterns for AC-anchor modes are shared with the
// /atc breakout overlay via constants/acAnchorModes.ts.

// Comparison data from another forecaster
interface ComparisonDataItem {
  forecaster: string;
  label: string;
  data: MilestoneDistributionData;
  atcCdfData?: AtcCdfData | null;
  atcStatistics?: AtcStatistics | null;
}

interface MilestoneDistributionChartProps {
  data: MilestoneDistributionData;
  className?: string;
  title?: string;
  height?: number;
  showCdf?: boolean;
  atcCdfData?: AtcCdfData | null;
  atcStatistics?: AtcStatistics | null;
  showAtcOverlay?: boolean;
  showModelOutput?: boolean;
  visibleMilestones?: Set<string>;
  showGridlines?: boolean;
  forecasterName?: string;
  forecasterDate?: string;
  comparisonData?: ComparisonDataItem[];
  compareMode?: 'forecaster' | 'model-vs-atc';
  forecasterShowType?: 'model' | 'atc';
  modeBreakoutData?: ModeBreakoutData | null;
  showModeBreakout?: boolean;
  /** Identifies the forecast in the exported PNG's filename. */
  simulationId?: string;
  /** User-selected x-axis window; overrides the auto-computed domain. */
  zoomDomain?: [number, number] | null;
}

// Dash patterns distinguishing multiple comparison forecasts, which share the
// milestone's color. Index 0 keeps the original style (solid for model output,
// standard dashes for ATC); later comparisons get progressively tighter dashes.
export const COMPARISON_MODEL_DASH_PATTERNS: (string | undefined)[] = [undefined, '8 4', '2 3', '10 4 2 4'];
export const COMPARISON_ATC_DASH_PATTERNS: string[] = ['6 3', '2 3', '10 4 2 4', '1 3'];

// Color palette for different milestones
const MILESTONE_COLORS: { [key: string]: string } = {
  'AC': '#2A623D',
  'AI2027-SC': '#af1e86ff',
  'SAR-level-experiment-selection-skill': '#000090',
  'SIAR-level-experiment-selection-skill': '#900000',
  'TED-AI': '#D62728',
  'ASI': '#8C564B',
};

// Compute nice round tick values for y-axis
function computeNiceTicks(maxValue: number, targetTickCount: number = 4): number[] {
  if (maxValue <= 0) return [0];

  // Find a nice step size
  const roughStep = maxValue / targetTickCount;
  const magnitude = Math.pow(10, Math.floor(Math.log10(roughStep)));
  const normalized = roughStep / magnitude;

  // Choose a nice step: 1, 2, 2.5, 5, or 10
  let niceStep: number;
  if (normalized <= 1) niceStep = magnitude;
  else if (normalized <= 2) niceStep = 2 * magnitude;
  else if (normalized <= 2.5) niceStep = 2.5 * magnitude;
  else if (normalized <= 5) niceStep = 5 * magnitude;
  else niceStep = 10 * magnitude;

  // Generate ticks from 0 up to (and including) a value >= maxValue
  const ticks: number[] = [0];
  let tick = niceStep;
  while (tick < maxValue * 1.01) { // Small buffer to ensure we include the max
    ticks.push(tick);
    tick += niceStep;
  }
  // Ensure we have at least one tick above maxValue for headroom
  if (ticks[ticks.length - 1] < maxValue) {
    ticks.push(tick);
  }

  return ticks;
}

// Display names for milestones
const MILESTONE_DISPLAY_NAMES: { [key: string]: string } = {
  'AC': 'AC (Automated Coder)',
  'AI2027-SC': 'SC (Superhuman Coder)',
  'SAR-level-experiment-selection-skill': 'SAR (Superhuman AI Researcher)',
  'SIAR-level-experiment-selection-skill': 'SIAR (Superintelligent AI Researcher)',
  'TED-AI': 'TED-AI (Top-Expert-Dominating AI)',
  'ASI': 'ASI (Artificial Superintelligence)',
};

const MilestoneDistributionChart = memo(({ data, className, title = 'Milestone Probability Density', height = 400, showCdf = false, atcCdfData, atcStatistics, showAtcOverlay = false, showModelOutput = true, visibleMilestones, showGridlines = false, forecasterName, forecasterDate, comparisonData = [], compareMode = 'model-vs-atc', forecasterShowType = 'model', modeBreakoutData = null, showModeBreakout = false, simulationId, zoomDomain = null }: MilestoneDistributionChartProps) => {
  // Refs feeding the PNG export: the plot itself plus the HTML legend, which is
  // redrawn as SVG (it lives outside the chart's <svg>).
  const exportSvgRef = useRef<SVGSVGElement | null>(null);
  const legendRef = useRef<HTMLDivElement | null>(null);

  // Filter milestones based on visibility prop
  const filteredMilestoneKeys = useMemo(() => {
    const allKeys = Object.keys(data.milestones);
    if (!visibleMilestones) return allKeys;
    return allKeys.filter(key => visibleMilestones.has(key));
  }, [data.milestones, visibleMilestones]);

  // AC-anchor-mode breakout is active when toggled on and data is available
  const breakoutModes = useMemo(
    () => (showModeBreakout && modeBreakoutData ? modeBreakoutData.modes : []),
    [showModeBreakout, modeBreakoutData]
  );
  const breakoutActive = breakoutModes.length > 0;

  const { chartData, lines } = useMemo(() => {
    if (showCdf && data.cdf) {
      // Use CDF data
      const allYears = new Set<number>();

      // Filter CDF milestone keys based on visibility
      const cdfMilestoneKeys = filteredMilestoneKeys.filter(key => data.cdf![key]);

      // Build lookup maps for O(1) access instead of O(n) find() calls
      const cdfLookups = new Map<string, Map<number, number>>();
      for (const milestoneName of cdfMilestoneKeys) {
        const lookup = new Map<number, number>();
        for (const point of data.cdf![milestoneName]) {
          lookup.set(point.year, point.cdf);
          allYears.add(point.year);
        }
        cdfLookups.set(milestoneName, lookup);
      }

      // Build ATC CDF lookup maps
      const atcCdfLookups = new Map<string, Map<number, number>>();
      if (showAtcOverlay && atcCdfData) {
        for (const milestoneName of cdfMilestoneKeys) {
          if (atcCdfData[milestoneName]) {
            const lookup = new Map<number, number>();
            for (const point of atcCdfData[milestoneName]) {
              lookup.set(point.year, point.cdf);
              allYears.add(point.year);
            }
            atcCdfLookups.set(milestoneName, lookup);
          }
        }
      }

      // Build per-anchor-mode breakout CDF lookup maps
      const breakoutCdfLookups = new Map<string, Map<number, number>>();  // key: `${mode}|${milestone}`
      if (breakoutActive && modeBreakoutData) {
        for (const mode of breakoutModes) {
          const modeData = modeBreakoutData.perMode[mode];
          if (!modeData) continue;
          for (const milestoneName of cdfMilestoneKeys) {
            const points = modeData.cdf[milestoneName];
            if (!points || points.length === 0) continue;
            const lookup = new Map<number, number>();
            for (const point of points) {
              lookup.set(point.year, point.cdf);
              allYears.add(point.year);
            }
            breakoutCdfLookups.set(`${mode}|${milestoneName}`, lookup);
          }
        }
      }

      // Build comparison CDF lookup maps
      const comparisonCdfLookups: Array<Map<string, Map<number, number>>> = [];
      for (const comparison of comparisonData) {
        const compCdfSource = (compareMode === 'forecaster' && forecasterShowType === 'atc')
          ? comparison.atcCdfData  // null if no ATC data — don't fall back to model
          : comparison.data.cdf;
        const compLookups = new Map<string, Map<number, number>>();
        if (compCdfSource) {
          for (const milestoneName of cdfMilestoneKeys) {
            if (compCdfSource[milestoneName]) {
              const lookup = new Map<number, number>();
              for (const point of compCdfSource[milestoneName]) {
                lookup.set(point.year, point.cdf);
                allYears.add(point.year);
              }
              compLookups.set(milestoneName, lookup);
            }
          }
        }
        comparisonCdfLookups.push(compLookups);
      }

      // Create a data point for each year with all milestone CDF values
      const sortedYears = Array.from(allYears).sort((a, b) => a - b);
      const combinedData = sortedYears.map(year => {
        const dataPoint: DataPoint = { x: year };

        // Add empirical CDF values using O(1) lookup
        for (const milestoneName of cdfMilestoneKeys) {
          const lookup = cdfLookups.get(milestoneName);
          dataPoint[milestoneName] = lookup?.get(year) ?? null;
        }

        // Add ATC CDF values with "_atc" suffix key using O(1) lookup
        if (showAtcOverlay && atcCdfData) {
          for (const milestoneName of cdfMilestoneKeys) {
            const lookup = atcCdfLookups.get(milestoneName);
            if (lookup) {
              dataPoint[`${milestoneName}_atc`] = lookup.get(year) ?? null;
            }
          }
        }

        // Add per-anchor-mode breakout CDF values with "mode_{i}_{milestone}" keys
        breakoutModes.forEach((mode, modeIndex) => {
          for (const milestoneName of cdfMilestoneKeys) {
            const lookup = breakoutCdfLookups.get(`${mode}|${milestoneName}`);
            if (lookup) {
              dataPoint[`mode_${modeIndex}_${milestoneName}`] = lookup.get(year) ?? null;
            }
          }
        });

        // Add comparison data with "comp_{index}_{milestone}" keys using O(1) lookup
        comparisonData.forEach((_, compIndex) => {
          const compLookups = comparisonCdfLookups[compIndex];
          for (const milestoneName of cdfMilestoneKeys) {
            const lookup = compLookups.get(milestoneName);
            if (lookup) {
              dataPoint[`comp_${compIndex}_${milestoneName}`] = lookup.get(year) ?? null;
            }
          }
        });

        return dataPoint;
      });

      // Create line configurations for each milestone (empirical)
      const lineConfigs: Array<{
        dataKey: string;
        stroke: string;
        strokeWidth: number;
        strokeDasharray?: string;
        strokeOpacity?: number;
        name: string;
      }> = showModelOutput ? cdfMilestoneKeys.map(milestoneName => ({
        dataKey: milestoneName,
        stroke: MILESTONE_COLORS[milestoneName] || '#666666',
        strokeWidth: 2.5,
        name: MILESTONE_DISPLAY_NAMES[milestoneName] || milestoneName,
      })) : [];

      // Add ATC line configs (dashed) - only for visible milestones
      if (showAtcOverlay && atcCdfData) {
        for (const milestoneName of cdfMilestoneKeys) {
          if (atcCdfData[milestoneName]) {
            lineConfigs.push({
              dataKey: `${milestoneName}_atc`,
              stroke: MILESTONE_COLORS[milestoneName] || '#666666',
              strokeWidth: 2,
              strokeDasharray: '6 3',
              name: (MILESTONE_DISPLAY_NAMES[milestoneName] || milestoneName) + ' (all-things-considered)',
            });
          }
        }
      }

      // Add per-anchor-mode breakout line configs (dashed per mode)
      breakoutModes.forEach((mode, modeIndex) => {
        for (const milestoneName of cdfMilestoneKeys) {
          if (breakoutCdfLookups.has(`${mode}|${milestoneName}`)) {
            lineConfigs.push({
              dataKey: `mode_${modeIndex}_${milestoneName}`,
              stroke: MILESTONE_COLORS[milestoneName] || '#666666',
              strokeWidth: 1.75,
              strokeOpacity: 0.85,
              strokeDasharray: ANCHOR_MODE_DASH_PATTERNS[modeIndex % ANCHOR_MODE_DASH_PATTERNS.length],
              name: `${ANCHOR_MODE_DISPLAY_NAMES[mode] || mode}: ${MILESTONE_DISPLAY_NAMES[milestoneName] || milestoneName}`,
            });
          }
        }
      });

      // Add comparison line configs (solid with reduced opacity in forecaster mode)
      // In forecaster mode, use either model CDF or ATC CDF based on forecasterShowType
      comparisonData.forEach((comparison, compIndex) => {
        const compCdfSource = (compareMode === 'forecaster' && forecasterShowType === 'atc')
          ? comparison.atcCdfData  // null if no ATC data — don't fall back to model
          : comparison.data.cdf;
        if (compCdfSource) {
          for (const milestoneName of cdfMilestoneKeys) {
            if (compCdfSource[milestoneName]) {
              const showingAtc = compareMode === 'forecaster' && forecasterShowType === 'atc';
              const suffix = showingAtc ? ' (ATC)' : '';
              lineConfigs.push({
                dataKey: `comp_${compIndex}_${milestoneName}`,
                stroke: MILESTONE_COLORS[milestoneName] || '#666666',
                strokeWidth: 2,
                strokeOpacity: compareMode === 'forecaster' ? 0.5 : 1,
                strokeDasharray: showingAtc
                  ? COMPARISON_ATC_DASH_PATTERNS[compIndex % COMPARISON_ATC_DASH_PATTERNS.length]
                  : COMPARISON_MODEL_DASH_PATTERNS[compIndex % COMPARISON_MODEL_DASH_PATTERNS.length],
                name: `${comparison.forecaster}${suffix}: ${MILESTONE_DISPLAY_NAMES[milestoneName] || milestoneName}`,
              });
            }
          }
        }
      });

      return { chartData: combinedData, lines: lineConfigs };
    }

    // PDF mode: Combine all milestone data into a single dataset
    const allYears = new Set<number>();

    // Build lookup maps for O(1) access instead of O(n) find() calls
    const pdfLookups = new Map<string, Map<number, number>>();
    for (const milestoneName of filteredMilestoneKeys) {
      const lookup = new Map<number, number>();
      for (const point of data.milestones[milestoneName]) {
        lookup.set(point.year, point.probabilityDensity);
        allYears.add(point.year);
      }
      pdfLookups.set(milestoneName, lookup);
    }

    // Build per-anchor-mode breakout PDF lookup maps
    const breakoutPdfLookups = new Map<string, Map<number, number>>();  // key: `${mode}|${milestone}`
    if (breakoutActive && modeBreakoutData) {
      for (const mode of breakoutModes) {
        const modeData = modeBreakoutData.perMode[mode];
        if (!modeData) continue;
        for (const milestoneName of filteredMilestoneKeys) {
          const points = modeData.milestones[milestoneName];
          if (!points || points.length === 0) continue;
          const lookup = new Map<number, number>();
          for (const point of points) {
            lookup.set(point.year, point.probabilityDensity);
            allYears.add(point.year);
          }
          breakoutPdfLookups.set(`${mode}|${milestoneName}`, lookup);
        }
      }
    }

    // Build comparison PDF lookup maps
    const comparisonPdfLookups: Array<Map<string, Map<number, number>>> = [];
    // In forecaster mode with model type, use model PDF data
    // In forecaster mode with ATC type, we'll compute PDF from ATC CDF (don't add years here - interpolate later)
    if (!(compareMode === 'forecaster' && forecasterShowType === 'atc')) {
      for (const comparison of comparisonData) {
        const compLookups = new Map<string, Map<number, number>>();
        for (const milestoneName of filteredMilestoneKeys) {
          if (comparison.data.milestones[milestoneName]) {
            const lookup = new Map<number, number>();
            for (const point of comparison.data.milestones[milestoneName]) {
              lookup.set(point.year, point.probabilityDensity);
              allYears.add(point.year);
            }
            compLookups.set(milestoneName, lookup);
          }
        }
        comparisonPdfLookups.push(compLookups);
      }
    }

    // Compute ATC PDFs from CDF data if showing overlay
    // We compute dense samples but only evaluate at the empirical data points
    const atcPdfData: { [milestoneName: string]: { x: number; pdf: number }[] } = {};
    // Store scaling factors for each milestone (ratio of empirical max to ATC max)
    const atcScaleFactors: { [milestoneName: string]: number } = {};
    if (showAtcOverlay && atcCdfData) {
      for (const milestoneName of filteredMilestoneKeys) {
        if (atcCdfData[milestoneName]) {
          const cdfPoints: CdfPoint[] = atcCdfData[milestoneName].map(p => ({
            year: p.year,
            cdf: p.cdf,
          }));
          const pdfSamples = computePdfFromCdf(cdfPoints);
          atcPdfData[milestoneName] = pdfSamples;
          // Don't add ATC x-values to allYears - we'll interpolate at empirical points only

          // The ATC PDF from computePdfFromCdf is already in density units (per year).
          // The empirical PDF is also now in density units (per year) after proper normalization.
          // No scaling needed - both are directly comparable.
          atcScaleFactors[milestoneName] = 1;
        }
      }
    }

    // Compute comparison ATC PDFs from CDF data when in forecaster mode with ATC type
    const comparisonAtcPdfData: Array<{ [milestoneName: string]: { x: number; pdf: number }[] }> = [];
    if (compareMode === 'forecaster' && forecasterShowType === 'atc') {
      for (const comparison of comparisonData) {
        const compAtcPdf: { [milestoneName: string]: { x: number; pdf: number }[] } = {};
        if (comparison.atcCdfData) {
          for (const milestoneName of filteredMilestoneKeys) {
            if (comparison.atcCdfData[milestoneName]) {
              const cdfPoints: CdfPoint[] = comparison.atcCdfData[milestoneName].map(p => ({
                year: p.year,
                cdf: p.cdf,
              }));
              const pdfSamples = computePdfFromCdf(cdfPoints);
              compAtcPdf[milestoneName] = pdfSamples;
            }
          }
        }
        comparisonAtcPdfData.push(compAtcPdf);
      }
    }

    // Create a data point for each year with all milestone densities
    const sortedYears = Array.from(allYears).sort((a, b) => a - b);
    const combinedData = sortedYears.map(year => {
      const dataPoint: DataPoint = { x: year };

      // Add empirical PDF values using O(1) lookup
      for (const milestoneName of filteredMilestoneKeys) {
        const lookup = pdfLookups.get(milestoneName);
        dataPoint[milestoneName] = lookup?.get(year) ?? null;
      }

      // Add ATC PDF values with "_atc" suffix - interpolate from dense samples
      // The ATC PDF is the derivative of the CDF, which gives probability density per year.
      // We scale by bin width to convert density to mass, so area ratios match probability ratios.
      if (showAtcOverlay && atcCdfData) {
        for (const milestoneName of Object.keys(atcPdfData)) {
          const pdfSamples = atcPdfData[milestoneName];
          const scaleFactor = atcScaleFactors[milestoneName] ?? 1;
          // Interpolate to get value at this year
          const sortedSamples = pdfSamples; // Already sorted from computePdfFromCdf
          const lowerIdx = sortedSamples.findIndex(s => s.x > year) - 1;
          let rawPdf: number;
          if (lowerIdx < 0) {
            rawPdf = sortedSamples[0]?.pdf ?? 0;
          } else if (lowerIdx >= sortedSamples.length - 1) {
            rawPdf = sortedSamples[sortedSamples.length - 1]?.pdf ?? 0;
          } else {
            const lower = sortedSamples[lowerIdx];
            const upper = sortedSamples[lowerIdx + 1];
            const ratio = (year - lower.x) / (upper.x - lower.x);
            rawPdf = lower.pdf + ratio * (upper.pdf - lower.pdf);
          }
          // Scale ATC PDF by bin width so area reflects total probability (max CDF value)
          dataPoint[`${milestoneName}_atc`] = rawPdf * scaleFactor;
        }
      }

      // Add per-anchor-mode breakout PDF values with "mode_{i}_{milestone}" keys
      breakoutModes.forEach((mode, modeIndex) => {
        for (const milestoneName of filteredMilestoneKeys) {
          const lookup = breakoutPdfLookups.get(`${mode}|${milestoneName}`);
          if (lookup) {
            dataPoint[`mode_${modeIndex}_${milestoneName}`] = lookup.get(year) ?? null;
          }
        }
      });

      // Add comparison PDF data with "comp_{index}_{milestone}" keys
      // In forecaster mode with ATC type, use interpolated PDF from comparison's ATC CDF
      comparisonData.forEach((_, compIndex) => {
        if (compareMode === 'forecaster' && forecasterShowType === 'atc') {
          // Use comparison ATC PDF data (interpolated)
          const compAtcPdf = comparisonAtcPdfData[compIndex];
          for (const milestoneName of filteredMilestoneKeys) {
            if (compAtcPdf && compAtcPdf[milestoneName]) {
              const pdfSamples = compAtcPdf[milestoneName];
              // Interpolate to get value at this year
              const lowerIdx = pdfSamples.findIndex(s => s.x > year) - 1;
              let rawPdf: number;
              if (lowerIdx < 0) {
                rawPdf = pdfSamples[0]?.pdf ?? 0;
              } else if (lowerIdx >= pdfSamples.length - 1) {
                rawPdf = pdfSamples[pdfSamples.length - 1]?.pdf ?? 0;
              } else {
                const lower = pdfSamples[lowerIdx];
                const upper = pdfSamples[lowerIdx + 1];
                const ratio = (year - lower.x) / (upper.x - lower.x);
                rawPdf = lower.pdf + ratio * (upper.pdf - lower.pdf);
              }
              dataPoint[`comp_${compIndex}_${milestoneName}`] = rawPdf;
            }
          }
        } else {
          // Use comparison model PDF data with O(1) lookup
          const compLookups = comparisonPdfLookups[compIndex];
          if (compLookups) {
            for (const milestoneName of filteredMilestoneKeys) {
              const lookup = compLookups.get(milestoneName);
              if (lookup) {
                dataPoint[`comp_${compIndex}_${milestoneName}`] = lookup.get(year) ?? null;
              }
            }
          }
        }
      });

      return dataPoint;
    });

    // Create line configurations for each milestone (empirical)
    const lineConfigs: Array<{
      dataKey: string;
      stroke: string;
      strokeWidth: number;
      strokeDasharray?: string;
      strokeOpacity?: number;
      name: string;
    }> = showModelOutput ? filteredMilestoneKeys.map(milestoneName => ({
      dataKey: milestoneName,
      stroke: MILESTONE_COLORS[milestoneName] || '#666666',
      strokeWidth: 2.5,
      name: MILESTONE_DISPLAY_NAMES[milestoneName] || milestoneName,
    })) : [];

    // Add ATC PDF line configs (dashed)
    if (showAtcOverlay && atcCdfData) {
      for (const milestoneName of Object.keys(atcPdfData)) {
        lineConfigs.push({
          dataKey: `${milestoneName}_atc`,
          stroke: MILESTONE_COLORS[milestoneName] || '#666666',
          strokeWidth: 2,
          strokeDasharray: '6 3',
          name: (MILESTONE_DISPLAY_NAMES[milestoneName] || milestoneName) + ' (all-things-considered)',
        });
      }
    }

    // Add per-anchor-mode breakout PDF line configs (dashed per mode)
    breakoutModes.forEach((mode, modeIndex) => {
      for (const milestoneName of filteredMilestoneKeys) {
        if (breakoutPdfLookups.has(`${mode}|${milestoneName}`)) {
          lineConfigs.push({
            dataKey: `mode_${modeIndex}_${milestoneName}`,
            stroke: MILESTONE_COLORS[milestoneName] || '#666666',
            strokeWidth: 1.75,
            strokeOpacity: 0.85,
            strokeDasharray: ANCHOR_MODE_DASH_PATTERNS[modeIndex % ANCHOR_MODE_DASH_PATTERNS.length],
            name: `${ANCHOR_MODE_DISPLAY_NAMES[mode] || mode}: ${MILESTONE_DISPLAY_NAMES[milestoneName] || milestoneName}`,
          });
        }
      }
    });

    // Add comparison PDF line configs (solid with reduced opacity in forecaster mode)
    // In forecaster mode with ATC type, show comparison ATC data with dashed styling
    comparisonData.forEach((comparison, compIndex) => {
      if (compareMode === 'forecaster' && forecasterShowType === 'atc') {
        // Use comparison ATC PDF data
        const compAtcPdf = comparisonAtcPdfData[compIndex];
        for (const milestoneName of filteredMilestoneKeys) {
          if (compAtcPdf && compAtcPdf[milestoneName]) {
            lineConfigs.push({
              dataKey: `comp_${compIndex}_${milestoneName}`,
              stroke: MILESTONE_COLORS[milestoneName] || '#666666',
              strokeWidth: 2,
              strokeOpacity: 0.5,
              strokeDasharray: COMPARISON_ATC_DASH_PATTERNS[compIndex % COMPARISON_ATC_DASH_PATTERNS.length],
              name: `${comparison.forecaster} (ATC): ${MILESTONE_DISPLAY_NAMES[milestoneName] || milestoneName}`,
            });
          }
        }
      } else {
        // Use comparison model PDF data
        for (const milestoneName of filteredMilestoneKeys) {
          if (comparison.data.milestones[milestoneName]) {
            lineConfigs.push({
              dataKey: `comp_${compIndex}_${milestoneName}`,
              stroke: MILESTONE_COLORS[milestoneName] || '#666666',
              strokeWidth: 2,
              strokeOpacity: compareMode === 'forecaster' ? 0.5 : 1,
              strokeDasharray: COMPARISON_MODEL_DASH_PATTERNS[compIndex % COMPARISON_MODEL_DASH_PATTERNS.length],
              name: `${comparison.forecaster}: ${MILESTONE_DISPLAY_NAMES[milestoneName] || milestoneName}`,
            });
          }
        }
      }
    });

    return { chartData: combinedData, lines: lineConfigs };
  }, [data, showCdf, atcCdfData, showAtcOverlay, showModelOutput, filteredMilestoneKeys, comparisonData, compareMode, forecasterShowType, breakoutActive, modeBreakoutData, breakoutModes]);

  const xDomain = useMemo<[number, number]>(() => {
    // A user-selected zoom window takes precedence over the auto domain
    if (zoomDomain && zoomDomain[1] > zoomDomain[0]) {
      return zoomDomain;
    }

    if (chartData.length === 0) {
      return [2020, 2030];
    }

    const years = chartData.map(point => point.x);
    const minYear = Math.min(...years);
    const dataMaxYear = Math.max(...years);

    // Cap the display, but use actual data range if it's smaller
    const maxYear = Math.min(dataMaxYear, TIMELINE_DISPLAY_MAX_YEAR);
    const span = Math.max(maxYear - minYear, 1);
    const padding = span * 0.02;

    return [minYear - padding, Math.min(maxYear + padding, TIMELINE_DISPLAY_MAX_YEAR)];
  }, [chartData, zoomDomain]);

  // Put ticks and gridlines on calendar-year boundaries (Jan 2027, Jan 2031, …).
  // Letting the generic tick generator choose the step landed every label on
  // whatever month the data happened to start in, so the axis read as a row of
  // repeated "September"s that meant nothing.
  //
  // The run starts at the first whole year in the domain rather than at a
  // multiple of the step: anchoring to multiples left a gap wider than the tick
  // spacing at the left edge, which is exactly where the curves do their work.
  const xTicks = useMemo(() => {
    const [start, end] = xDomain;
    const span = Math.max(end - start, 1e-6);

    if (span >= 4) {
      const step = Math.max(1, Math.round(span / 6));
      const ticks: number[] = [];
      for (let year = Math.ceil(start); year <= end; year += step) {
        ticks.push(year);
      }
      return ticks;
    }

    // Zoomed-in windows narrower than ~4 years would get one or two yearly
    // ticks at most, so switch to month boundaries there instead.
    const monthStep = Math.max(1, Math.round((span * 12) / 6));
    const ticks: number[] = [];
    for (let m = Math.ceil(start * 12 - 1e-6); m / 12 <= end + 1e-6; m += monthStep) {
      ticks.push(m / 12);
    }
    return ticks;
  }, [xDomain]);

  const yDomain = useMemo<[number, number]>(() => {
    // CDF mode uses 0-1 range with padding for label visibility
    if (showCdf) {
      return [-0.08, 1.05];
    }

    if (chartData.length === 0) {
      return [0, 1];
    }

    let maxDensity = 0;

    // Check empirical PDF values
    for (const point of chartData) {
      for (const milestoneName of filteredMilestoneKeys) {
        const density = point[milestoneName] as number;
        if (Number.isFinite(density) && density > maxDensity) {
          maxDensity = density;
        }
      }
    }

    // Also check ATC PDF values if overlay is enabled
    if (showAtcOverlay && atcCdfData) {
      for (const point of chartData) {
        for (const milestoneName of filteredMilestoneKeys) {
          const atcKey = `${milestoneName}_atc`;
          const density = point[atcKey] as number;
          if (Number.isFinite(density) && density > maxDensity) {
            maxDensity = density;
          }
        }
      }
    }

    // Also check comparison PDF values
    for (const point of chartData) {
      comparisonData.forEach((_, compIndex) => {
        for (const milestoneName of filteredMilestoneKeys) {
          const compKey = `comp_${compIndex}_${milestoneName}`;
          const density = point[compKey] as number;
          if (Number.isFinite(density) && density > maxDensity) {
            maxDensity = density;
          }
        }
      });
    }

    // Also check per-anchor-mode breakout PDF values
    if (breakoutActive) {
      for (const point of chartData) {
        breakoutModes.forEach((_, modeIndex) => {
          for (const milestoneName of filteredMilestoneKeys) {
            const modeKey = `mode_${modeIndex}_${milestoneName}`;
            const density = point[modeKey] as number;
            if (Number.isFinite(density) && density > maxDensity) {
              maxDensity = density;
            }
          }
        });
      }
    }

    const headroom = maxDensity === 0 ? 0.1 : maxDensity * 0.1;

    return [0, maxDensity + headroom];
  }, [chartData, filteredMilestoneKeys, showCdf, showAtcOverlay, atcCdfData, comparisonData, breakoutActive, breakoutModes]);

  const formatProbabilityPercent = (value: number): string => {
    if (!Number.isFinite(value)) {
      return '0.00';
    }

    const text = value.toFixed(2);

    return text === '-0.00' ? '0.00' : text;
  };

  // Interpolate ATC CDF value at a given year
  const getAtcCdfAtYear = (milestoneName: string, year: number): number | null => {
    if (!atcCdfData || !atcCdfData[milestoneName]) return null;
    const points = atcCdfData[milestoneName];
    if (points.length === 0) return null;

    // Find surrounding points for interpolation
    if (year <= points[0].year) return points[0].cdf;
    if (year >= points[points.length - 1].year) return points[points.length - 1].cdf;

    for (let i = 1; i < points.length; i++) {
      if (year <= points[i].year) {
        const prev = points[i - 1];
        const curr = points[i];
        const ratio = (year - prev.year) / (curr.year - prev.year);
        return prev.cdf + ratio * (curr.cdf - prev.cdf);
      }
    }
    return null;
  };

  // Interpolate comparison CDF value at a given year
  // Uses ATC CDF when in forecaster comparison mode with ATC type selected
  const getComparisonCdfAtYear = (comparisonIndex: number, milestoneName: string, year: number): number | null => {
    const comparison = comparisonData[comparisonIndex];
    if (!comparison) return null;

    // Use ATC CDF when in forecaster mode with ATC type
    const useAtc = compareMode === 'forecaster' && forecasterShowType === 'atc';
    const cdfSource = useAtc ? comparison.atcCdfData : comparison.data.cdf;

    if (!cdfSource?.[milestoneName]) return null;
    const points = cdfSource[milestoneName];
    if (points.length === 0) return null;

    // Find surrounding points for interpolation
    if (year <= points[0].year) return points[0].cdf;
    if (year >= points[points.length - 1].year) return points[points.length - 1].cdf;

    for (let i = 1; i < points.length; i++) {
      if (year <= points[i].year) {
        const prev = points[i - 1];
        const curr = points[i];
        const ratio = (year - prev.year) / (curr.year - prev.year);
        return prev.cdf + ratio * (curr.cdf - prev.cdf);
      }
    }
    return null;
  };

  // Interpolate primary CDF value at a given year (for tooltip when exact year not in lookup)
  const getPrimaryCdfAtYear = (milestoneName: string, year: number): number | null => {
    // Use ATC CDF when in forecaster mode with ATC type
    const useAtc = compareMode === 'forecaster' && forecasterShowType === 'atc';
    const cdfSource = useAtc ? atcCdfData : data.cdf;

    if (!cdfSource?.[milestoneName]) return null;
    const points = cdfSource[milestoneName];
    if (points.length === 0) return null;

    // Find surrounding points for interpolation
    if (year <= points[0].year) return points[0].cdf;
    if (year >= points[points.length - 1].year) return points[points.length - 1].cdf;

    for (let i = 1; i < points.length; i++) {
      if (year <= points[i].year) {
        const prev = points[i - 1];
        const curr = points[i];
        const ratio = (year - prev.year) / (curr.year - prev.year);
        return prev.cdf + ratio * (curr.cdf - prev.cdf);
      }
    }
    return null;
  };

  // Interpolate a breakout mode's CDF value at a given year
  const getModeCdfAtYear = (mode: string, milestoneName: string, year: number): number | null => {
    const points = modeBreakoutData?.perMode[mode]?.cdf?.[milestoneName];
    if (!points || points.length === 0) return null;

    if (year <= points[0].year) return points[0].cdf;
    if (year >= points[points.length - 1].year) return points[points.length - 1].cdf;

    for (let i = 1; i < points.length; i++) {
      if (year <= points[i].year) {
        const prev = points[i - 1];
        const curr = points[i];
        const ratio = (year - prev.year) / (curr.year - prev.year);
        return prev.cdf + ratio * (curr.cdf - prev.cdf);
      }
    }
    return null;
  };

  const tooltip = (point: DataPoint): ReactNode => {
    return (
      <div style={tooltipBoxStyle}>
        <span style={tooltipHeaderStyle}>{formatYearMonth(point.x)}</span>
        <div style={{ fontSize: '10px', color: '#666', marginBottom: '4px' }}>
          Cumulative Probability
        </div>
        {compareMode === 'forecaster' ? (
          // Forecaster comparison mode: show primary forecaster then comparison forecaster
          <>
            {forecasterName && (
              <div style={{ fontSize: '11px', fontWeight: 600, color: '#333', marginBottom: '4px' }}>
                {forecasterName}:
              </div>
            )}
            {filteredMilestoneKeys.map(milestoneName => {
              const displayName = MILESTONE_DISPLAY_NAMES[milestoneName] || milestoneName;
              const color = MILESTONE_COLORS[milestoneName] || '#666666';
              const probability = getPrimaryCdfAtYear(milestoneName, point.x);
              if (probability === null || probability === undefined) return null;
              return (
                <div key={milestoneName} style={{ ...tooltipValueStyle, color, marginLeft: forecasterName ? '8px' : '0' }}>
                  {`${displayName}: ${formatProbabilityPercent(probability * 100)}%`}
                </div>
              );
            })}
            {comparisonData.length > 0 && comparisonData.map((comparison, compIndex) => (
              <div key={compIndex}>
                <div style={{ fontSize: '11px', fontWeight: 600, color: '#333', marginTop: '8px', marginBottom: '4px' }}>
                  {comparison.forecaster}:
                </div>
                {filteredMilestoneKeys.map(milestoneName => {
                  const displayName = MILESTONE_DISPLAY_NAMES[milestoneName] || milestoneName;
                  const color = MILESTONE_COLORS[milestoneName] || '#666666';
                  const compProbability = getComparisonCdfAtYear(compIndex, milestoneName, point.x);
                  if (compProbability === null) return null;
                  return (
                    <div key={milestoneName} style={{ ...tooltipValueStyle, color, marginLeft: '8px', opacity: 0.8 }}>
                      {`${displayName}: ${formatProbabilityPercent(compProbability * 100)}%`}
                    </div>
                  );
                })}
              </div>
            ))}
          </>
        ) : (
          // Model vs ATC mode: show milestone with Model and ATC labels
          filteredMilestoneKeys.map(milestoneName => {
            const displayName = MILESTONE_DISPLAY_NAMES[milestoneName] || milestoneName;
            const color = MILESTONE_COLORS[milestoneName] || '#666666';
            const probability = getPrimaryCdfAtYear(milestoneName, point.x);

            // Get ATC CDF value if overlay is shown
            const atcProbability = showAtcOverlay ? getAtcCdfAtYear(milestoneName, point.x) : null;

            // If both model and ATC are shown, use labeled format
            const showBothLabels = showModelOutput && atcProbability !== null;

            return (
              <div key={milestoneName}>
                {showBothLabels ? (
                  <>
                    <div style={{ ...tooltipValueStyle, color, fontWeight: 500 }}>
                      {displayName}:
                    </div>
                    {showModelOutput && (
                      <div style={{ ...tooltipValueStyle, color, marginLeft: '8px' }}>
                        {`Model: ${formatProbabilityPercent((probability ?? 0) * 100)}%`}
                      </div>
                    )}
                    <div style={{ ...tooltipValueStyle, color, marginLeft: '8px', opacity: 0.7 }}>
                      {`All-things-considered: ${formatProbabilityPercent(atcProbability * 100)}%`}
                    </div>
                  </>
                ) : (
                  <>
                    {showModelOutput && (
                      <div style={{ ...tooltipValueStyle, color }}>
                        {`${displayName}: ${formatProbabilityPercent((probability ?? 0) * 100)}%`}
                      </div>
                    )}
                    {atcProbability !== null && !showModelOutput && (
                      <div style={{ ...tooltipValueStyle, color }}>
                        {`${displayName} (ATC): ${formatProbabilityPercent(atcProbability * 100)}%`}
                      </div>
                    )}
                  </>
                )}
              </div>
            );
          })
        )}
        {breakoutActive && (
          <div style={{ marginTop: '6px' }}>
            <div style={{ fontSize: '10px', fontWeight: 600, color: '#333', marginBottom: '2px' }}>
              By AC-anchor method:
            </div>
            {filteredMilestoneKeys.map(milestoneName => {
              const color = MILESTONE_COLORS[milestoneName] || '#666666';
              const rows = breakoutModes
                .map(mode => ({ mode, cdf: getModeCdfAtYear(mode, milestoneName, point.x) }))
                .filter((r): r is { mode: string; cdf: number } => r.cdf !== null);
              if (rows.length === 0) return null;
              return (
                <div key={`bk-${milestoneName}`}>
                  {filteredMilestoneKeys.length > 1 && (
                    <div style={{ ...tooltipValueStyle, color, fontWeight: 500 }}>
                      {MILESTONE_DISPLAY_NAMES[milestoneName] || milestoneName}:
                    </div>
                  )}
                  {rows.map(({ mode, cdf }) => (
                    <div key={mode} style={{ ...tooltipValueStyle, color, marginLeft: '8px', opacity: 0.8 }}>
                      {`${ANCHOR_MODE_DISPLAY_NAMES[mode] || mode}: ${formatProbabilityPercent(cdf * 100)}%`}
                    </div>
                  ))}
                </div>
              );
            })}
          </div>
        )}
      </div>
    );
  };

  // Export metadata reflects the chart's current interactive state. The title
  // is the site's brand line, so the subtitle carries what the chart shows.
  const exportSubtitle = [
    'Milestone arrival dates',
    forecasterName
      ? `${forecasterName}'s forecast${forecasterDate ? ` (${formatForecastDateLabel(forecasterDate)})` : ''}`
      : null,
    showCdf ? 'Cumulative probability' : 'Probability density',
    showModeBreakout && breakoutActive ? 'Broken out by AC-anchor method' : null,
  ].filter(Boolean).join('  ·  ');

  const getExportFilename = useCallback(
    () => buildExportFilename(simulationId, `milestone-arrival-${showCdf ? 'cdf' : 'pdf'}`),
    [simulationId, showCdf]
  );

  return (
    <div className={className || 'flex-1'}>
      <div className="flex gap-2 items-center mb-4">
        <span className="text-[10px] font-semibold text-primary text-left font-system-mono">{title}</span>
        <div className="flex-1 border-t border-gray-500/30" />
        <ForecastChartDownloadButton
          chartSvgRef={exportSvgRef}
          legendRef={legendRef}
          title="AI Futures Model: Timelines Forecast"
          subtitle={exportSubtitle}
          getFilename={getExportFilename}
        />
      </div>

      {/* Legend. NOTE: the PNG export reads this markup via
          `[data-chart-legend] > div > div` — keep the column/row nesting intact. */}
      <div ref={legendRef} data-chart-legend className="flex flex-wrap gap-4 mb-4 text-xs">
        {filteredMilestoneKeys.map(milestoneName => {
          const displayName = MILESTONE_DISPLAY_NAMES[milestoneName] || milestoneName;
          const color = MILESTONE_COLORS[milestoneName] || '#666666';
          const stats = data.statistics[milestoneName];
          const hasAtcData = showAtcOverlay && atcCdfData?.[milestoneName]?.length;
          const showingCompAtc = compareMode === 'forecaster' && forecasterShowType === 'atc';
          const hasComparisonEntry = comparisonData.some(comparison => {
            return showingCompAtc
              ? (comparison.atcCdfData?.[milestoneName]?.length ?? 0) > 0
              : (comparison.data.milestones[milestoneName]?.length ?? 0) > 0;
          });
          const milestoneLabel = displayName;
          const forecasterPrefix = forecasterName && forecasterDate
            ? `${forecasterName}'s ${forecasterDate}`
            : forecasterName;
          const modelLabel = forecasterPrefix
            ? `${forecasterPrefix} model-based forecast for ${milestoneLabel}`
            : `Model-based forecast for ${milestoneLabel}`;
          const atcLabel = forecasterPrefix
            ? `${forecasterPrefix} all-things-considered forecast for ${milestoneLabel}`
            : `All-things-considered forecast for ${milestoneLabel}`;

          const hasBreakoutEntry = breakoutActive && breakoutModes.some(mode =>
            (modeBreakoutData?.perMode[mode]?.milestones[milestoneName]?.length ?? 0) > 0
          );

          if (!showModelOutput && !hasAtcData && !hasComparisonEntry && !hasBreakoutEntry) {
            return null;
          }

          return (
            <div key={milestoneName} className="flex flex-col gap-1">
              {/* Empirical line - only show when showModelOutput is true */}
              {showModelOutput && (
                <div className="flex items-center gap-2">
                  <div className="w-8 h-0.5" style={{ backgroundColor: color }} />
                  <span className="font-medium min-w-0" style={{ color }}>
                    {modelLabel}
                    {stats && (
                      <span className="text-gray-500">
                        {` (p10: ${formatYearMonth(stats.p10)}, p50: ${formatYearMonth(stats.p50)}, p90: ${formatYearMonth(stats.p90)})`}
                      </span>
                    )}
                  </span>
                </div>
              )}
              {/* All-things-considered line - show when overlay is active and data exists for this milestone */}
              {hasAtcData && (
                <div className="flex items-center gap-2">
                  <div
                    className="w-8 h-0"
                    style={{
                      borderTop: `2px dashed ${color}`,
                    }}
                  />
                  <span className="font-medium text-gray-500 min-w-0">
                    {atcLabel}
                    {atcStatistics?.[milestoneName] && (
                      <span className="text-gray-500">
                        {` (p10: ${formatYearMonth(atcStatistics[milestoneName].p10)}, p50: ${formatYearMonth(atcStatistics[milestoneName].p50)}, p90: ${formatYearMonth(atcStatistics[milestoneName].p90)})`}
                      </span>
                    )}
                  </span>
                </div>
              )}
              {/* Comparison forecaster lines - dashed when showing ATC, solid when showing model */}
              {comparisonData.map((comparison, compIndex) => {
                const compStats = showingCompAtc
                  ? comparison.atcStatistics?.[milestoneName]
                  : comparison.data.statistics[milestoneName];
                const hasCompData = showingCompAtc
                  ? (comparison.atcCdfData?.[milestoneName]?.length ?? 0) > 0
                  : (comparison.data.milestones[milestoneName]?.length ?? 0) > 0;
                if (!hasCompData) return null;
                const comparisonPrefix = `${comparison.forecaster}'s ${comparison.label.split(' - ')[1] || ''}`.trim();
                const comparisonLabel = showingCompAtc
                  ? `${comparisonPrefix} all-things-considered forecast for ${milestoneLabel}`
                  : `${comparisonPrefix} model-based forecast for ${milestoneLabel}`;
                const compDash = showingCompAtc
                  ? COMPARISON_ATC_DASH_PATTERNS[compIndex % COMPARISON_ATC_DASH_PATTERNS.length]
                  : COMPARISON_MODEL_DASH_PATTERNS[compIndex % COMPARISON_MODEL_DASH_PATTERNS.length];
                return (
                  <div key={`comp-${compIndex}`} className="flex items-center gap-2">
                    <svg width="32" height="4" className="shrink-0" aria-hidden="true">
                      <line x1="0" y1="2" x2="32" y2="2" stroke={color} strokeWidth="2" strokeDasharray={compDash} opacity={0.5} />
                    </svg>
                    <span className="font-medium min-w-0" style={{ color, opacity: 0.7 }}>
                      {comparisonLabel}
                      {compStats && (
                        <span className="text-gray-500">
                          {` (p10: ${formatYearMonth(compStats.p10)}, p50: ${formatYearMonth(compStats.p50)}, p90: ${formatYearMonth(compStats.p90)})`}
                        </span>
                      )}
                    </span>
                  </div>
                );
              })}
              {/* Per-AC-anchor-mode breakout lines */}
              {breakoutActive && breakoutModes.map((mode, modeIndex) => {
                const modeData = modeBreakoutData?.perMode[mode];
                if (!modeData || (modeData.milestones[milestoneName]?.length ?? 0) === 0) return null;
                const modeStats = modeData.statistics[milestoneName];
                const dash = ANCHOR_MODE_DASH_PATTERNS[modeIndex % ANCHOR_MODE_DASH_PATTERNS.length];
                return (
                  <div key={`mode-${mode}`} className="flex items-center gap-2">
                    <svg width="32" height="4" className="shrink-0" aria-hidden="true">
                      <line x1="0" y1="2" x2="32" y2="2" stroke={color} strokeWidth="2" strokeDasharray={dash} opacity={0.85} />
                    </svg>
                    <span className="font-medium min-w-0" style={{ color, opacity: 0.85 }}>
                      {`${ANCHOR_MODE_DISPLAY_NAMES[mode] || mode}: ${milestoneLabel}`}
                      <span className="text-gray-500">
                        {` (mixture weight ${modeData.weightPct.toFixed(1)}%`}
                        {modeStats ? `; p10: ${formatYearMonth(modeStats.p10)}, p50: ${formatYearMonth(modeStats.p50)}, p90: ${formatYearMonth(modeStats.p90)}` : ''}
                        {')'}
                      </span>
                    </span>
                  </div>
                );
              })}
            </div>
          );
        })}
      </div>

      <ChartSyncProvider>
        {showCdf ? (
          <CustomLineChart
            data={chartData}
            svgRef={exportSvgRef}
            height={height}
            margin={{ top: 20, right: 40, left: 60, bottom: 60 }}
            xDomain={xDomain}
            xTicks={xTicks}
            xTickFormatter={(value) => formatYearMonthShort(value)}
            yDomain={yDomain}
            yTickFormatter={(value) => `${Math.round(value * 100)}%`}
            showYAxis={true}
            yTicks={[0, 0.25, 0.5, 0.75, 1]}
            xPadding={{ start: 50 }}
            lines={lines}
            tooltip={tooltip}
            showGridlines={showGridlines}
            yAxisLabelsOutside
          />
        ) : (
          <PdfLineChart
            data={chartData}
            svgRef={exportSvgRef}
            height={height}
            margin={{ top: 15, right: 40, left: 60, bottom: 60 }}
            xDomain={xDomain}
            xTicks={xTicks}
            xTickFormatter={(value) => formatYearMonthShort(value)}
            yDomain={yDomain}
            yTicks={computeNiceTicks(yDomain[1])}
            yTickFormatter={(value) => `${formatTo3SigFigs(value * 100)}%/yr`}
            showYAxis={true}
            lines={lines}
            tooltip={tooltip}
            showGridlines={showGridlines}
          />
        )}
      </ChartSyncProvider>
    </div>
  );
});

MilestoneDistributionChart.displayName = 'MilestoneDistributionChart';

export default MilestoneDistributionChart;
