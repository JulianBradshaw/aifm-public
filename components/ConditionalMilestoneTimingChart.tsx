'use client';

import { memo, useMemo, useRef, useCallback, ReactNode } from 'react';
import { CustomLineChart, DataPoint } from './CustomLineChart';
import { PdfLineChart } from './PdfLineChart';
import { ChartSyncProvider } from './ChartSyncContext';
import { ForecastChartDownloadButton } from './ForecastChartDownloadButton';
import { buildExportFilename, formatForecastDateLabel } from '@/utils/forecastChartExport';
import { formatTo3SigFigs } from '@/utils/formatting';
import { tooltipBoxStyle, tooltipHeaderStyle, tooltipValueStyle } from './chartTooltipStyle';
import { AtcTakeoffCdfData } from '@/app/forecast/[[...simulationId]]/page';
import { computePdfFromCdf, CdfPoint } from '@/utils/cdfSplineUtils';
import { COMPARISON_MODEL_DASH_PATTERNS, COMPARISON_ATC_DASH_PATTERNS } from './MilestoneDistributionChart';

export interface ConditionalTimingPoint {
  timeFromAC: number;
  probabilityDensity: number;
}

export interface ConditionalCdfPoint {
  timeFromAC: number;
  cdf: number;
}

export interface ConditionalTimingData {
  milestones: { [key: string]: ConditionalTimingPoint[] };
  statistics: {
    [key: string]: {
      achievementRate: number;
      mode: number;
      p10: number;
      p50: number;
      p90: number;
    };
  };
  conditionDescription: string;
  cdf?: { [key: string]: ConditionalCdfPoint[] };
}

// Comparison data from another forecaster
interface TakeoffComparisonDataItem {
  forecaster: string;
  label: string;
  data: ConditionalTimingData;
  atcTakeoffCdfData?: AtcTakeoffCdfData | null;
}

interface ConditionalMilestoneTimingChartProps {
  data: ConditionalTimingData;
  className?: string;
  title?: string;
  height?: number;
  maxTimeYears?: number;
  sharedYDomain?: [number, number];
  showLegend?: boolean;
  showCdf?: boolean;
  atcTakeoffCdfData?: AtcTakeoffCdfData | null;
  showModelOutput?: boolean;
  visibleMilestones?: Set<string>;
  showGridlines?: boolean;
  forecasterName?: string;
  forecasterDate?: string;
  comparisonData?: TakeoffComparisonDataItem[];
  compareMode?: 'forecaster' | 'model-vs-atc';
  forecasterShowType?: 'model' | 'atc';
  /** Identifies the forecast in the exported PNG's filename. */
  simulationId?: string;
  /** User-selected x-axis window; overrides the auto-computed domain. */
  zoomDomain?: [number, number] | null;
}

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

// Color palette for different milestones (matching MilestoneDistributionChart)
const MILESTONE_COLORS: { [key: string]: string } = {
  'AI2027-SC': '#af1e86ff',
  'AIR-5x': '#FF6B35',
  'AIR-25x': '#FF8C42',
  'AIR-250x': '#FFAD5A',
  'AIR-2000x': '#FFCE73',
  'AIR-10000x': '#FFE98C',
  'SAR-level-experiment-selection-skill': '#000090',
  'SIAR-level-experiment-selection-skill': '#900000',
  'STRAT-AI': '#00A896',
  'TED-AI': '#D62728',
  'ASI': '#8C564B',
};

// Display names for milestones
const MILESTONE_DISPLAY_NAMES: { [key: string]: string } = {
  'AI2027-SC': 'SC (Superhuman Coder)',
  'AIR-5x': 'AIR 5×',
  'AIR-25x': 'AIR 25×',
  'AIR-250x': 'AIR 250×',
  'AIR-2000x': 'AIR 2000×',
  'AIR-10000x': 'AIR 10000×',
  'SAR-level-experiment-selection-skill': 'SAR (Superhuman AI Researcher)',
  'SIAR-level-experiment-selection-skill': 'SIAR (Superintelligent AI Researcher)',
  'STRAT-AI': 'STRAT-AI',
  'TED-AI': 'TED-AI (Top Expert Dominating AI)',
  'ASI': 'ASI (Artificial Superintelligence)',
};

// Statistics files encode "milestone not reached" percentiles with a huge sentinel value.
// Rollouts are censored at the simulation horizon (2200), so a censored percentile means
// "not reached by 2200", not a genuine never — see scripts/pipelines/summarize_rollouts.py.
const NEVER_REACHED_THRESHOLD_YEARS = 1000;
const CENSORED_PERCENTILE_LABEL = 'not by 2200';

// Format a duration in decimal years for the legend summary stats
function formatDurationYears(value: number | null): string {
  if (value === null || !Number.isFinite(value) || value >= NEVER_REACHED_THRESHOLD_YEARS) {
    return CENSORED_PERCENTILE_LABEL;
  }
  if (value < 1) {
    const months = (value * 12).toFixed(1);
    return `${months.endsWith('.0') ? months.slice(0, -2) : months} mo`;
  }
  const years = value.toFixed(1);
  return `${years.endsWith('.0') ? years.slice(0, -2) : years} yrs`;
}

interface TakeoffPercentiles {
  p10: number | null;
  p50: number | null;
  p90: number | null;
}

function formatStatsSuffix(stats: TakeoffPercentiles): string {
  return ` (p10: ${formatDurationYears(stats.p10)}, p50: ${formatDurationYears(stats.p50)}, p90: ${formatDurationYears(stats.p90)})`;
}

// Compute p10/p50/p90 from ATC takeoff CDF points by interpolating the time at each target
// probability. A percentile the CDF never reaches (curve plateaus below the target) is null.
function computeTakeoffCdfPercentiles(points: AtcTakeoffCdfData[string]): TakeoffPercentiles | null {
  if (!points || points.length === 0) return null;

  const sorted = [...points].sort((a, b) => a.timeFromAC - b.timeFromAC);
  const maxCdf = sorted[sorted.length - 1].cdf;

  const findTimeAtCdf = (targetCdf: number): number | null => {
    if (maxCdf < targetCdf) return null;
    if (targetCdf <= sorted[0].cdf) return sorted[0].timeFromAC;

    for (let i = 0; i < sorted.length - 1; i++) {
      if (sorted[i].cdf <= targetCdf && sorted[i + 1].cdf >= targetCdf) {
        const cdfDiff = sorted[i + 1].cdf - sorted[i].cdf;
        if (cdfDiff === 0) return sorted[i].timeFromAC;
        const t = (targetCdf - sorted[i].cdf) / cdfDiff;
        return sorted[i].timeFromAC + t * (sorted[i + 1].timeFromAC - sorted[i].timeFromAC);
      }
    }

    return sorted[sorted.length - 1].timeFromAC;
  };

  return {
    p10: findTimeAtCdf(0.10),
    p50: findTimeAtCdf(0.50),
    p90: findTimeAtCdf(0.90),
  };
}

const ConditionalMilestoneTimingChart = memo(({
  data,
  className,
  title = 'Time Until Milestones (Conditional)',
  height = 400,
  maxTimeYears,
  sharedYDomain,
  showLegend = true,
  showCdf = false,
  atcTakeoffCdfData,
  showModelOutput = true,
  visibleMilestones,
  showGridlines = false,
  forecasterName,
  forecasterDate,
  comparisonData = [],
  compareMode = 'model-vs-atc',
  forecasterShowType = 'model',
  simulationId,
  zoomDomain = null,
}: ConditionalMilestoneTimingChartProps) => {
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

  const { chartData, lines } = useMemo(() => {
    if (showCdf && data.cdf) {
      // Use CDF data directly
      const allTimes = new Set<number>();

      // Filter CDF milestone keys based on visibility
      const cdfMilestoneKeys = filteredMilestoneKeys.filter(key => data.cdf![key]);

      // Build lookup maps for O(1) access instead of O(n) find() calls
      const cdfLookups = new Map<string, Map<number, number>>();
      for (const milestoneName of cdfMilestoneKeys) {
        const lookup = new Map<number, number>();
        for (const point of data.cdf![milestoneName]) {
          lookup.set(point.timeFromAC, point.cdf);
          allTimes.add(point.timeFromAC);
        }
        cdfLookups.set(milestoneName, lookup);
      }

      // Build ATC CDF lookup maps
      const atcCdfLookups = new Map<string, Map<number, number>>();
      if (atcTakeoffCdfData) {
        for (const milestoneName of cdfMilestoneKeys) {
          if (atcTakeoffCdfData[milestoneName]) {
            const lookup = new Map<number, number>();
            for (const point of atcTakeoffCdfData[milestoneName]) {
              lookup.set(point.timeFromAC, point.cdf);
              allTimes.add(point.timeFromAC);
            }
            atcCdfLookups.set(milestoneName, lookup);
          }
        }
      }

      // Build comparison CDF lookup maps
      const comparisonCdfLookups: Array<Map<string, Map<number, number>>> = [];
      for (const comparison of comparisonData) {
        const compCdfSource = (compareMode === 'forecaster' && forecasterShowType === 'atc')
          ? comparison.atcTakeoffCdfData  // null if no ATC takeoff data — don't fall back to model
          : comparison.data.cdf;
        const compLookups = new Map<string, Map<number, number>>();
        if (compCdfSource) {
          for (const milestoneName of cdfMilestoneKeys) {
            if (compCdfSource[milestoneName]) {
              const lookup = new Map<number, number>();
              for (const point of compCdfSource[milestoneName]) {
                lookup.set(point.timeFromAC, point.cdf);
                allTimes.add(point.timeFromAC);
              }
              compLookups.set(milestoneName, lookup);
            }
          }
        }
        comparisonCdfLookups.push(compLookups);
      }

      // Create a data point for each time with all milestone CDF values
      const sortedTimes = Array.from(allTimes).sort((a, b) => a - b);
      const combinedData = sortedTimes.map(time => {
        const dataPoint: DataPoint = { x: time };

        // Add empirical CDF values using O(1) lookup
        for (const milestoneName of cdfMilestoneKeys) {
          const lookup = cdfLookups.get(milestoneName);
          dataPoint[milestoneName] = lookup?.get(time) ?? null;
        }

        // Add ATC data points with prefixed keys using O(1) lookup
        if (atcTakeoffCdfData) {
          for (const milestoneName of cdfMilestoneKeys) {
            const lookup = atcCdfLookups.get(milestoneName);
            if (lookup) {
              dataPoint[`${milestoneName}_atc`] = lookup.get(time) ?? null;
            }
          }
        }

        // Add comparison data with "comp_{index}_{milestone}" keys using O(1) lookup
        comparisonData.forEach((_, compIndex) => {
          const compLookups = comparisonCdfLookups[compIndex];
          for (const milestoneName of cdfMilestoneKeys) {
            const lookup = compLookups.get(milestoneName);
            if (lookup) {
              dataPoint[`comp_${compIndex}_${milestoneName}`] = lookup.get(time) ?? null;
            }
          }
        });

        return dataPoint;
      });

      // Create line configurations for each milestone
      const lineConfigs: Array<{
        dataKey: string;
        stroke: string;
        strokeWidth: number;
        name: string;
        smooth: boolean;
        strokeDasharray?: string;
        strokeOpacity?: number;
      }> = showModelOutput ? cdfMilestoneKeys.map(milestoneName => ({
        dataKey: milestoneName,
        stroke: MILESTONE_COLORS[milestoneName] || '#666666',
        strokeWidth: 2.5,
        name: MILESTONE_DISPLAY_NAMES[milestoneName] || milestoneName,
        smooth: true,
      })) : [];

      // Add all-things-considered lines as dashed overlays (only for visible milestones)
      if (atcTakeoffCdfData) {
        for (const milestoneName of cdfMilestoneKeys) {
          if (atcTakeoffCdfData[milestoneName]) {
            lineConfigs.push({
              dataKey: `${milestoneName}_atc`,
              stroke: MILESTONE_COLORS[milestoneName] || '#666666',
              strokeWidth: 2,
              name: `${MILESTONE_DISPLAY_NAMES[milestoneName] || milestoneName} (all-things-considered)`,
              smooth: true,
              strokeDasharray: '6 3',
            });
          }
        }
      }

      // Add comparison line configs (solid with reduced opacity in forecaster mode)
      // In forecaster mode, use either model CDF or ATC CDF based on forecasterShowType
      comparisonData.forEach((comparison, compIndex) => {
        const compCdfSource = (compareMode === 'forecaster' && forecasterShowType === 'atc')
          ? comparison.atcTakeoffCdfData  // null if no ATC takeoff data — don't fall back to model
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
                name: `${comparison.forecaster}${suffix}: ${MILESTONE_DISPLAY_NAMES[milestoneName] || milestoneName}`,
                smooth: true,
                strokeOpacity: compareMode === 'forecaster' ? 0.5 : 1,
                strokeDasharray: showingAtc
                  ? COMPARISON_ATC_DASH_PATTERNS[compIndex % COMPARISON_ATC_DASH_PATTERNS.length]
                  : COMPARISON_MODEL_DASH_PATTERNS[compIndex % COMPARISON_MODEL_DASH_PATTERNS.length],
              });
            }
          }
        }
      });

      return { chartData: combinedData, lines: lineConfigs };
    }

    // PDF mode: Calculate scaled densities for each milestone to match p50 medians
    const scaledDensities: Record<string, Map<number, number>> = {};

    for (const milestoneName of filteredMilestoneKeys) {
      const stats = data.statistics[milestoneName];
      const points = [...data.milestones[milestoneName]].sort((a, b) => a.timeFromAC - b.timeFromAC);

      if (points.length === 0) {
        scaledDensities[milestoneName] = new Map();
        continue;
      }

      const totalDensity = points.reduce((sum, point) => sum + point.probabilityDensity, 0);

      if (!Number.isFinite(totalDensity) || totalDensity <= 0) {
        scaledDensities[milestoneName] = new Map();
        continue;
      }

      // Calculate bin width from data spacing (assumes uniform spacing)
      const binWidth = points.length > 1
        ? (points[points.length - 1].timeFromAC - points[0].timeFromAC) / (points.length - 1)
        : 1;

      // Find the empirical median from the raw PDF
      let running = 0;
      let empiricalMedianTime = points[0].timeFromAC;
      for (const point of points) {
        running += point.probabilityDensity / totalDensity;
        if (running >= 0.5) {
          empiricalMedianTime = point.timeFromAC;
          break;
        }
      }

      // Calculate scaling factor to shift the median
      const targetMedian = stats?.p50;
      const hasMedian = typeof targetMedian === 'number' && Number.isFinite(targetMedian);

      // Get achievement rate to scale the PDF area
      const achievementRate = stats?.achievementRate ?? 1;

      if (!hasMedian || empiricalMedianTime === 0) {
        // No time scaling needed, just normalize and scale by achievementRate
        // Divide by binWidth to get proper density units (per year)
        const densityMap = new Map<number, number>();
        for (const point of points) {
          densityMap.set(point.timeFromAC, (point.probabilityDensity / totalDensity) * achievementRate / binWidth);
        }
        scaledDensities[milestoneName] = densityMap;
        continue;
      }

      // Scale the time axis so empirical median becomes target median
      const timeScaleFactor = targetMedian / empiricalMedianTime;

      // Create new scaled density map
      // Note: When time is scaled, bin width also scales, so we divide by (binWidth * timeScaleFactor)
      // The / timeScaleFactor at the end accounts for the density transformation
      const densityMap = new Map<number, number>();
      for (const point of points) {
        const scaledTime = point.timeFromAC * timeScaleFactor;
        // Scale density by inverse of time scale to preserve shape, and by achievementRate for area
        // Also divide by binWidth to get proper density units (per year)
        const scaledDensity = (point.probabilityDensity / totalDensity) * achievementRate / binWidth / timeScaleFactor;
        densityMap.set(scaledTime, scaledDensity);
      }

      scaledDensities[milestoneName] = densityMap;
    }

    // Compute ATC PDFs from CDF data if available
    // We compute dense samples but only evaluate at the empirical data points
    const atcPdfData: { [milestoneName: string]: { x: number; pdf: number }[] } = {};
    const atcScaleFactors: { [milestoneName: string]: number } = {};
    if (atcTakeoffCdfData) {
      for (const milestoneName of filteredMilestoneKeys) {
        if (atcTakeoffCdfData[milestoneName]) {
          // Convert timeFromAC to year format for computePdfFromCdf
          const cdfPoints: CdfPoint[] = atcTakeoffCdfData[milestoneName].map(p => ({
            year: p.timeFromAC,
            cdf: p.cdf,
          }));
          const pdfSamples = computePdfFromCdf(cdfPoints);
          atcPdfData[milestoneName] = pdfSamples;
          // Don't add ATC x-values to allTimes - we'll interpolate at empirical points only

          // The ATC PDF from computePdfFromCdf is already in density units (per year).
          // The empirical PDF is also now in density units (per year) after proper normalization.
          // No scaling needed - both are directly comparable.
          atcScaleFactors[milestoneName] = 1;
        }
      }
    }

    // Combine all milestone data into a single dataset
    const allTimes = new Set<number>();
    for (const milestoneName of filteredMilestoneKeys) {
      const densityMap = scaledDensities[milestoneName];
      if (densityMap) {
        for (const time of densityMap.keys()) {
          allTimes.add(time);
        }
      }
    }

    // Build comparison PDF lookup maps and collect times
    // In forecaster mode with ATC type, we'll compute PDF from ATC CDF (don't add times here - interpolate later)
    const comparisonPdfLookups: Array<Map<string, Map<number, number>>> = [];
    if (!(compareMode === 'forecaster' && forecasterShowType === 'atc')) {
      for (const comparison of comparisonData) {
        const compLookups = new Map<string, Map<number, number>>();
        for (const milestoneName of filteredMilestoneKeys) {
          if (comparison.data.milestones[milestoneName]) {
            const lookup = new Map<number, number>();
            for (const point of comparison.data.milestones[milestoneName]) {
              lookup.set(point.timeFromAC, point.probabilityDensity);
              allTimes.add(point.timeFromAC);
            }
            compLookups.set(milestoneName, lookup);
          }
        }
        comparisonPdfLookups.push(compLookups);
      }
    }

    // Compute comparison ATC PDFs from CDF data when in forecaster mode with ATC type
    const comparisonAtcPdfData: Array<{ [milestoneName: string]: { x: number; pdf: number }[] }> = [];
    if (compareMode === 'forecaster' && forecasterShowType === 'atc') {
      for (const comparison of comparisonData) {
        const compAtcPdf: { [milestoneName: string]: { x: number; pdf: number }[] } = {};
        if (comparison.atcTakeoffCdfData) {
          for (const milestoneName of filteredMilestoneKeys) {
            if (comparison.atcTakeoffCdfData[milestoneName]) {
              const cdfPoints: CdfPoint[] = comparison.atcTakeoffCdfData[milestoneName].map(p => ({
                year: p.timeFromAC,
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

    // Create a data point for each time with scaled densities
    const sortedTimes = Array.from(allTimes).sort((a, b) => a - b);
    const combinedData = sortedTimes.map(time => {
      const dataPoint: DataPoint = { x: time };

      for (const milestoneName of filteredMilestoneKeys) {
        const densityMap = scaledDensities[milestoneName];
        const value = densityMap?.get(time);
        // Only set the value if it exists, otherwise leave as null
        // This allows the smooth curve to only render where there's actual data
        dataPoint[milestoneName] = value !== undefined ? value : null;
      }

      // Add ATC PDF values with prefixed keys - interpolate from dense samples
      // Scale by bin width so area reflects total probability (max CDF value)
      if (atcTakeoffCdfData) {
        for (const milestoneName of Object.keys(atcPdfData)) {
          const pdfSamples = atcPdfData[milestoneName];
          const scaleFactor = atcScaleFactors[milestoneName] ?? 1;
          // Interpolate to get value at this time
          const sortedSamples = pdfSamples; // Already sorted from computePdfFromCdf
          const lowerIdx = sortedSamples.findIndex(s => s.x > time) - 1;
          let rawPdf: number;
          if (lowerIdx < 0) {
            rawPdf = sortedSamples[0]?.pdf ?? 0;
          } else if (lowerIdx >= sortedSamples.length - 1) {
            rawPdf = sortedSamples[sortedSamples.length - 1]?.pdf ?? 0;
          } else {
            const lower = sortedSamples[lowerIdx];
            const upper = sortedSamples[lowerIdx + 1];
            const ratio = (time - lower.x) / (upper.x - lower.x);
            rawPdf = lower.pdf + ratio * (upper.pdf - lower.pdf);
          }
          dataPoint[`${milestoneName}_atc`] = rawPdf * scaleFactor;
        }
      }

      // Add comparison PDF data with "comp_{index}_{milestone}" keys
      // In forecaster mode with ATC type, use interpolated PDF from comparison's ATC CDF
      comparisonData.forEach((_, compIndex) => {
        if (compareMode === 'forecaster' && forecasterShowType === 'atc') {
          // Use comparison ATC PDF data (interpolated)
          const compAtcPdf = comparisonAtcPdfData[compIndex];
          for (const milestoneName of filteredMilestoneKeys) {
            if (compAtcPdf && compAtcPdf[milestoneName]) {
              const pdfSamples = compAtcPdf[milestoneName];
              // Interpolate to get value at this time
              const lowerIdx = pdfSamples.findIndex(s => s.x > time) - 1;
              let rawPdf: number;
              if (lowerIdx < 0) {
                rawPdf = pdfSamples[0]?.pdf ?? 0;
              } else if (lowerIdx >= pdfSamples.length - 1) {
                rawPdf = pdfSamples[pdfSamples.length - 1]?.pdf ?? 0;
              } else {
                const lower = pdfSamples[lowerIdx];
                const upper = pdfSamples[lowerIdx + 1];
                const ratio = (time - lower.x) / (upper.x - lower.x);
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
                dataPoint[`comp_${compIndex}_${milestoneName}`] = lookup.get(time) ?? null;
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
      name: string;
      smooth: boolean;
      strokeDasharray?: string;
      strokeOpacity?: number;
    }> = showModelOutput ? filteredMilestoneKeys.map(milestoneName => ({
      dataKey: milestoneName,
      stroke: MILESTONE_COLORS[milestoneName] || '#666666',
      strokeWidth: 2.5,
      name: MILESTONE_DISPLAY_NAMES[milestoneName] || milestoneName,
      smooth: true,
    })) : [];

    // Add all-things-considered PDF line configs (dashed)
    if (atcTakeoffCdfData) {
      for (const milestoneName of Object.keys(atcPdfData)) {
        lineConfigs.push({
          dataKey: `${milestoneName}_atc`,
          stroke: MILESTONE_COLORS[milestoneName] || '#666666',
          strokeWidth: 2,
          name: `${MILESTONE_DISPLAY_NAMES[milestoneName] || milestoneName} (all-things-considered)`,
          smooth: true,
          strokeDasharray: '6 3',
        });
      }
    }

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
              name: `${comparison.forecaster} (ATC): ${MILESTONE_DISPLAY_NAMES[milestoneName] || milestoneName}`,
              smooth: true,
              strokeOpacity: 0.5,
              strokeDasharray: COMPARISON_ATC_DASH_PATTERNS[compIndex % COMPARISON_ATC_DASH_PATTERNS.length],
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
              name: `${comparison.forecaster}: ${MILESTONE_DISPLAY_NAMES[milestoneName] || milestoneName}`,
              smooth: true,
              strokeOpacity: compareMode === 'forecaster' ? 0.5 : 1,
              strokeDasharray: COMPARISON_MODEL_DASH_PATTERNS[compIndex % COMPARISON_MODEL_DASH_PATTERNS.length],
            });
          }
        }
      }
    });

    return { chartData: combinedData, lines: lineConfigs };
  }, [data, showCdf, atcTakeoffCdfData, showModelOutput, filteredMilestoneKeys, comparisonData, compareMode, forecasterShowType]);

  const xDomain = useMemo<[number, number]>(() => {
    // A user-selected zoom window takes precedence over the auto domain
    if (zoomDomain && zoomDomain[1] > zoomDomain[0]) {
      return zoomDomain;
    }

    if (chartData.length === 0) {
      return [0, maxTimeYears ?? 10];
    }

    const times = chartData.map(point => point.x);
    const minTime = Math.min(...times);
    const maxTime = Math.max(...times);

    // Cap the display at specified max or 50 years by default
    const cappedMaxTime = Math.min(maxTime, maxTimeYears ?? 50);
    const span = Math.max(cappedMaxTime - minTime, 1);
    const padding = span * 0.02;

    return [Math.max(minTime - padding, 0), cappedMaxTime + padding];
  }, [chartData, maxTimeYears, zoomDomain]);

  // The default even-year ticks stay as-is; a zoom window gets ticks computed
  // for its span, on steps that stay exact at one decimal place.
  const xTicks = useMemo<number[]>(() => {
    if (!zoomDomain) return [0, 2, 4, 6, 8, 10];

    const [start, end] = xDomain;
    const span = Math.max(end - start, 1e-6);
    const rough = span / 5;
    const magnitude = Math.pow(10, Math.floor(Math.log10(rough)));
    const normalized = rough / magnitude;
    const niceStep =
      normalized <= 1 ? magnitude :
      normalized <= 2 ? 2 * magnitude :
      normalized <= 5 ? 5 * magnitude :
      10 * magnitude;

    const ticks: number[] = [];
    for (let t = Math.ceil(start / niceStep - 1e-9) * niceStep; t <= end + 1e-9; t += niceStep) {
      ticks.push(Number(t.toFixed(6)));
    }
    return ticks;
  }, [zoomDomain, xDomain]);

  const yDomain = useMemo<[number, number]>(() => {
    // CDF mode uses 0-1 range with padding for label visibility
    if (showCdf) {
      return [-0.08, 1.05];
    }

    if (chartData.length === 0) {
      return sharedYDomain ?? [0, 1];
    }

    // Every series that gets drawn has to fit, or it is silently painted above
    // the frame: the ATC overlay is far more concentrated than the model here
    // (its density peaked near 58%/yr against a model-derived ceiling of 12%),
    // so most of it used to be invisible with no indication anything was there.
    const seriesKeys: string[] = [];
    for (const milestoneName of filteredMilestoneKeys) {
      seriesKeys.push(milestoneName);
      if (atcTakeoffCdfData) {
        seriesKeys.push(`${milestoneName}_atc`);
      }
      comparisonData.forEach((_, compIndex) => {
        seriesKeys.push(`comp_${compIndex}_${milestoneName}`);
      });
    }

    let maxDensity = 0;
    for (const point of chartData) {
      for (const key of seriesKeys) {
        const density = point[key] as number;
        if (Number.isFinite(density) && density > maxDensity) {
          maxDensity = density;
        }
      }
    }

    const headroom = maxDensity === 0 ? 0.1 : maxDensity * 0.1;
    const localMax = maxDensity + headroom;

    // sharedYDomain keeps the scale steady as milestones are toggled, so treat
    // it as a floor rather than an answer: grow past it only when something
    // drawn would not otherwise fit. Hiding the ATC overlay therefore returns
    // the chart to the model's own scale.
    return sharedYDomain ? [sharedYDomain[0], Math.max(sharedYDomain[1], localMax)] : [0, localMax];
  }, [chartData, filteredMilestoneKeys, sharedYDomain, showCdf, atcTakeoffCdfData, comparisonData]);

  const formatYears = (value: number): string => {
    if (!Number.isFinite(value)) {
      return '0';
    }

    // Show one decimal place, trimming trailing zeros
    const formatted = value.toFixed(1);
    return formatted.endsWith('.0') ? formatted.slice(0, -2) : formatted;
  };

  const formatProbabilityPercent = (value: number): string => {
    if (!Number.isFinite(value)) {
      return '0.00';
    }

    const text = value.toFixed(2);

    return text === '-0.00' ? '0.00' : text;
  };

  // Interpolate ATC CDF value at a given time
  const getAtcCdfAtTime = (milestoneName: string, time: number): number | null => {
    if (!atcTakeoffCdfData || !atcTakeoffCdfData[milestoneName]) return null;
    const points = atcTakeoffCdfData[milestoneName];
    if (points.length === 0) return null;

    // Find surrounding points for interpolation
    if (time <= points[0].timeFromAC) return points[0].cdf;
    if (time >= points[points.length - 1].timeFromAC) return points[points.length - 1].cdf;

    for (let i = 1; i < points.length; i++) {
      if (time <= points[i].timeFromAC) {
        const prev = points[i - 1];
        const curr = points[i];
        const ratio = (time - prev.timeFromAC) / (curr.timeFromAC - prev.timeFromAC);
        return prev.cdf + ratio * (curr.cdf - prev.cdf);
      }
    }
    return null;
  };

  // Interpolate comparison CDF value at a given time
  // Uses ATC CDF when in forecaster comparison mode with ATC type selected
  const getComparisonCdfAtTime = (comparisonIndex: number, milestoneName: string, time: number): number | null => {
    const comparison = comparisonData[comparisonIndex];
    if (!comparison) return null;

    // Use ATC CDF when in forecaster mode with ATC type
    const useAtc = compareMode === 'forecaster' && forecasterShowType === 'atc';
    const cdfSource = useAtc ? comparison.atcTakeoffCdfData : comparison.data.cdf;

    if (!cdfSource?.[milestoneName]) return null;
    const points = cdfSource[milestoneName];
    if (points.length === 0) return null;

    // Find surrounding points for interpolation
    if (time <= points[0].timeFromAC) return points[0].cdf;
    if (time >= points[points.length - 1].timeFromAC) return points[points.length - 1].cdf;

    for (let i = 1; i < points.length; i++) {
      if (time <= points[i].timeFromAC) {
        const prev = points[i - 1];
        const curr = points[i];
        const ratio = (time - prev.timeFromAC) / (curr.timeFromAC - prev.timeFromAC);
        return prev.cdf + ratio * (curr.cdf - prev.cdf);
      }
    }
    return null;
  };

  // Interpolate primary CDF value at a given time (for tooltip when exact time not in lookup)
  const getPrimaryCdfAtTime = (milestoneName: string, time: number): number | null => {
    const useAtc = compareMode === 'forecaster' && forecasterShowType === 'atc';
    const cdfSource = useAtc ? atcTakeoffCdfData : data.cdf;

    if (!cdfSource?.[milestoneName]) return null;
    const points = cdfSource[milestoneName];
    if (points.length === 0) return null;

    if (time <= points[0].timeFromAC) return points[0].cdf;
    if (time >= points[points.length - 1].timeFromAC) return points[points.length - 1].cdf;

    for (let i = 1; i < points.length; i++) {
      if (time <= points[i].timeFromAC) {
        const prev = points[i - 1];
        const curr = points[i];
        const ratio = (time - prev.timeFromAC) / (curr.timeFromAC - prev.timeFromAC);
        return prev.cdf + ratio * (curr.cdf - prev.cdf);
      }
    }

    return null;
  };

  const tooltip = (point: DataPoint): ReactNode => {
    return (
      <div style={tooltipBoxStyle}>
        <span style={tooltipHeaderStyle}>{formatYears(point.x)} years from AC</span>
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
              const probability = getPrimaryCdfAtTime(milestoneName, point.x);
              if (probability === null) return null;
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
                  const compProbability = getComparisonCdfAtTime(compIndex, milestoneName, point.x);
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
            const probability = getPrimaryCdfAtTime(milestoneName, point.x);

            // Get ATC CDF value if data is available
            const atcProbability = atcTakeoffCdfData ? getAtcCdfAtTime(milestoneName, point.x) : null;

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
                    {showModelOutput && probability !== null && (
                      <div style={{ ...tooltipValueStyle, color }}>
                        {`${displayName}: ${formatProbabilityPercent(probability * 100)}%`}
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
      </div>
    );
  };

  // Export metadata reflects the chart's current interactive state. The title
  // is the site's brand line, so the subtitle carries what the chart shows.
  const exportSubtitle = [
    'Time from coding automation to future milestones',
    forecasterName
      ? `${forecasterName}'s forecast${forecasterDate ? ` (${formatForecastDateLabel(forecasterDate)})` : ''}`
      : null,
    showCdf ? 'Cumulative probability' : 'Probability density',
  ].filter(Boolean).join('  ·  ');

  const getExportFilename = useCallback(
    () => buildExportFilename(simulationId, `takeoff-duration-${showCdf ? 'cdf' : 'pdf'}`),
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
          title="AI Futures Model: Takeoff Forecast"
          subtitle={exportSubtitle}
          getFilename={getExportFilename}
        />
      </div>

      {/* Legend. NOTE: the PNG export reads this markup via
          `[data-chart-legend] > div > div` — keep the column/row nesting intact. */}
      {showLegend && (
        <div ref={legendRef} data-chart-legend className="flex flex-wrap gap-4 mb-4 text-xs">
          {filteredMilestoneKeys.map(milestoneName => {
            const displayName = MILESTONE_DISPLAY_NAMES[milestoneName] || milestoneName;
            const color = MILESTONE_COLORS[milestoneName] || '#666666';
            const stats = data.statistics[milestoneName];
            const atcStats = atcTakeoffCdfData?.[milestoneName]
              ? computeTakeoffCdfPercentiles(atcTakeoffCdfData[milestoneName])
              : null;
            const hasAtcData = atcTakeoffCdfData?.[milestoneName]?.length;

            const showingCompAtc = compareMode === 'forecaster' && forecasterShowType === 'atc';
            const hasComparisonEntry = comparisonData.some(comparison => {
              return showingCompAtc
                ? (comparison.atcTakeoffCdfData?.[milestoneName]?.length ?? 0) > 0
                : (comparison.data.milestones[milestoneName]?.length ?? 0) > 0;
            });
            const milestoneLabel = `AC to ${displayName}`;
            const forecasterPrefix = forecasterName && forecasterDate
              ? `${forecasterName}'s ${forecasterDate}`
              : forecasterName;
            const modelLabel = forecasterPrefix
              ? `${forecasterPrefix} model-based forecast for ${milestoneLabel}`
              : `Model-based forecast for ${milestoneLabel}`;
            const atcLabel = forecasterPrefix
              ? `${forecasterPrefix} all-things-considered forecast for ${milestoneLabel}`
              : `All-things-considered forecast for ${milestoneLabel}`;

            if (!showModelOutput && !hasAtcData && !hasComparisonEntry) {
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
                          {formatStatsSuffix(stats)}
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
                      {atcStats && (
                        <span className="text-gray-500">
                          {formatStatsSuffix(atcStats)}
                        </span>
                      )}
                    </span>
                  </div>
                )}
                {/* Comparison forecaster lines (solid with reduced opacity, dashed if ATC) */}
                {comparisonData.map((comparison, compIndex) => {
                  const hasCompData = comparison.data.milestones[milestoneName]?.length > 0;
                  if (!hasCompData) return null;
                  const compStats = showingCompAtc
                    ? (comparison.atcTakeoffCdfData?.[milestoneName]
                        ? computeTakeoffCdfPercentiles(comparison.atcTakeoffCdfData[milestoneName])
                        : null)
                    : comparison.data.statistics[milestoneName];
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
                            {formatStatsSuffix(compStats)}
                          </span>
                        )}
                      </span>
                    </div>
                  );
                })}
              </div>
            );
          })}
        </div>
      )}

      <div style={{ overflow: 'hidden' }}>
        <ChartSyncProvider>
          {showCdf ? (
            <CustomLineChart
              data={chartData}
              svgRef={exportSvgRef}
              height={height}
              margin={{ top: 20, right: 40, left: 60, bottom: 60 }}
              xDomain={xDomain}
              xTicks={xTicks}
              xTickFormatter={(value) => `${formatYears(value)} yrs`}
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
              xTickFormatter={(value) => `${formatYears(value)} yrs`}
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
    </div>
  );
});

ConditionalMilestoneTimingChart.displayName = 'ConditionalMilestoneTimingChart';

export default ConditionalMilestoneTimingChart;
