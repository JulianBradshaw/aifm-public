'use client';
import { ChartDataPoint } from '@/app/types';
import { memo, useMemo, useEffect, useRef, useState, useCallback, ReactNode } from 'react';
import { CustomLineChart, DataPoint, CustomElementsContext } from './CustomLineChart';
import { formatTo3SigFigs, formatAsPowerOfTenText, formatCompactNumberNode } from '@/utils/formatting';
import { calculateDynamicXDomain, clamp, ChartMargin } from '@/utils/chartUtils';
import { CHART_LAYOUT } from '@/constants/chartLayout';
import { tooltipBoxStyle, tooltipHeaderStyle, tooltipValueStyle } from './chartTooltipStyle';
import { ChartMilestoneMarkers } from './ChartMilestoneMarkers';
import {
  ObservedMarkShape,
  observedMarkFor,
  observedMarkPoints,
  OBSERVED_MARK_BY_ORGANIZATION,
} from './ChartShapes';
import { EPOCH_REVENUE_ATTRIBUTION, loadObservedRevenueData } from '@/utils/revenueLoader';
import { EPOCH_ECI_ATTRIBUTION, loadEciFrontierData } from '@/utils/eciLoader';
import {
  XYPoint,
  PixelSegment,
  interpolateAtX,
  getVisibleEndpoints,
  estimateLabelWidth,
  selectLabelPosition,
  pointsToPixelSegments,
  LABEL_HEIGHT,
  SEGMENTS_TO_CHECK,
  computeLabelBounds,
  type LabelBounds,
} from '@/utils/chartLabelUtils';

// Stacked-area decompositions applied automatically by dataKey (unless an explicit
// stackAreas prop is passed). ETSP = training compute (hardware) + software efficiency,
// drawn back-to-front so the bands read TC (below) and SW efficiency (above).
//
// Colours are lifted wholesale from CombinedComputeChart, which draws this same
// effective-vs-training decomposition: #2A623D for the total (also this component's
// default `color` and AIRnDProgressMultiplierChart's series) and #888 for training.
// One decomposition, one colour language, no palette of its own.
//
// The fills sit close together once blended -- #2A623D at 35% over white is ~#B4C8BB,
// and #888 at 35% on top of that is ~#A5B2A9, a difference of lightness rather than
// hue. What separates the bands is the solid stroke on each edge (#2A623D vs #888 at
// full opacity), not the fills. Flagged rather than fixed: brightening the grey would
// mean inventing a colour neither chart uses.
type StackAreaDef = {
  key: string;
  fill: string;
  stroke: string;
  name?: string;
  /** Area fill opacity; defaults to the translucent 0.35 the ETSP decomposition uses.
   *  1 gives solid Epoch-style bands (each cumulative area fully covers the ones
   *  painted before it, so the visible band is exactly this fill). */
  fillOpacity?: number;
};

const STACK_DECOMPOSITIONS: Record<string, StackAreaDef[]> = {
  // ETSP (rate) = training system performance (hardware) + software efficiency.
  etsp: [
    { key: 'etsp', fill: '#2A623D', stroke: '#2A623D', name: 'software efficiency' },
    { key: 'trainingSystemPerformance', fill: '#888', stroke: '#888', name: 'training system performance' },
  ],
  // AI company compute spend shares (% of company total), a 100%-stacked area chart:
  // bottom-to-top the bands are automation inference / training / experiment /
  // commercial. Series are CUMULATIVE percentages (commercial's cumulative = 100),
  // drawn largest-first with OPAQUE fills so each band reads as a solid color.
  // Hues match the per-series colors used elsewhere (inference red, training amber,
  // experiment green); commercial is neutral grey.
  spendPctCumCommercial: [
    { key: 'spendPctCumCommercial', fill: '#a8bec4', stroke: '#a8bec4', name: 'commercial', fillOpacity: 1 },
    { key: 'spendPctCumExperiment', fill: '#0d9376', stroke: '#0d9376', name: 'experiment', fillOpacity: 1 },
    { key: 'spendPctCumTraining', fill: '#e8a33d', stroke: '#e8a33d', name: 'training', fillOpacity: 1 },
    { key: 'spendPctCumInference', fill: '#e0654f', stroke: '#e0654f', name: 'automation inference', fillOpacity: 1 },
  ],
  // Share of world compute, as a 100%-stacked plot: the frontier AI company's slice
  // at the bottom, the rest of the world filling to 100% on top (same convention as
  // the spend decomposition: cumulative series, largest painted first, opaque fills).
  companyShareOfWorldPct: [
    { key: 'companyShareOfWorldPctTotal', fill: '#a8bec4', stroke: '#a8bec4', name: 'rest of world', fillOpacity: 1 },
    { key: 'companyShareOfWorldPct', fill: '#12968c', stroke: '#12968c', name: 'frontier AI company', fillOpacity: 1 },
  ],
};

/** A measured, real-world datapoint drawn on top of a modelled series. */
export interface ObservedMetricPoint {
  /** Decimal year. */
  x: number;
  /** Value in the same units as the series' dataKey. */
  y: number;
  /** Tooltip header, e.g. the company or model the figure belongs to. */
  label: string;
  /** Tooltip second line, e.g. the date the figure describes. */
  dateLabel: string;
  /** Tooltip provenance line, e.g. "Annualized run rate · Epoch AI". */
  detail: string;
  /** Optional tooltip line between the header and the date, e.g. the lab. */
  subLabel?: string;
  /** Marker fill; defaults to the Anthropic red the revenue overlay uses. */
  fill?: string;
  /** Marker shape; defaults to the diamond the revenue overlay uses. */
  shape?: ObservedMarkShape;
}

// Anthropic red + diamond: the same mark this site already uses for Claude points on
// the METR horizon chart (see CustomHorizonChart), so an observed Anthropic datapoint
// reads the same wherever it appears. Multi-lab overlays vary fill and shape per
// organization using the same vocabulary (see OBSERVED_MARK_BY_ORGANIZATION).
const OBSERVED_POINT_FILL = OBSERVED_MARK_BY_ORGANIZATION.anthropic.fill;
const OBSERVED_POINT_SHAPE: ObservedMarkShape = OBSERVED_MARK_BY_ORGANIZATION.anthropic.shape;

const MONTH_YEAR = new Intl.DateTimeFormat('en-US', {
  month: 'short',
  year: 'numeric',
  timeZone: 'UTC',
});

const formatIsoMonthYear = (isoDate: string) =>
  MONTH_YEAR.format(new Date(`${isoDate}T00:00:00Z`));

// Observed-datapoint overlays applied automatically by dataKey, in the same spirit as
// STACK_DECOMPOSITIONS above: the chart component owns the mapping so every call site
// (primary chart and the small-chart grid alike) picks it up without extra wiring.
//
// These are DISPLAY ONLY. Each anchored trend is fit purely from the user's parameters
// (progress_model/ac_anchor.py fit_revenue_trend / fit_eci_trend); nothing here is fit
// to the observed points.
const OBSERVED_POINTS_BY_KEY: Record<string, ObservedMetricPoint[]> = {
  // Multi-lab overlay: the marker varies by company (Anthropic red diamond,
  // OpenAI green square) using the same per-organization vocabulary as the
  // METR horizon and ECI charts.
  predictedRevenue: loadObservedRevenueData().map(point => ({
    x: point.year,
    y: point.revenueUsd,
    label: point.company,
    dateLabel: formatIsoMonthYear(point.date),
    detail: `${point.revenueType} · ${EPOCH_REVENUE_ATTRIBUTION}`,
    ...observedMarkFor(point.company),
  })),
  // The ECI frontier: every lab competes, so the marker varies by organization rather
  // than being uniformly Anthropic red.
  capabilityEci: loadEciFrontierData().map(point => ({
    x: point.year,
    y: point.eciScore,
    label: point.displayName,
    subLabel: point.organization,
    dateLabel: formatIsoMonthYear(point.releaseDate),
    detail: `ECI frontier · ${EPOCH_ECI_ATTRIBUTION}`,
    ...observedMarkFor(point.organization),
  })),
};

export interface CustomMetricChartProps {
  chartData: ChartDataPoint[];
  tooltip: (point: DataPoint) => ReactNode;
  className?: string;
  title: string;
  dataKey: string;
  yFormatter?: (value: number) => string | ReactNode;
  yScale?: 'linear' | 'log';
  color?: string;
  domainOverride?: [number, number] | ((observed: number[]) => [number, number]);
  verticalReferenceLine?: {
    x: number;
    stroke: string;
    strokeDasharray?: string;
    strokeWidth?: number;
    label?: string;
    strokeOpacity?: number;
  };
  /** Horizontal threshold line at a y value (in data units, same space as dataKey values) */
  horizontalReferenceLine?: {
    y: number;
    stroke: string;
    strokeDasharray?: string;
    strokeWidth?: number;
    label?: string;
    strokeOpacity?: number;
    // Which end of the line the label sits at. Defaults to 'right', which is
    // where a monotonically rising series is highest -- fine when the series
    // stays below the line, but it collides with the curve once the series
    // climbs past it. Use 'left' for lines the series is expected to cross.
    labelPosition?: 'left' | 'right';
  };
  // An arbitrary reference curve drawn in the same scales as the series. Used
  // where the meaningful reference is a RATE rather than a level -- e.g. the
  // uplift mode pins a doubling rate at a reference date, which is a sloped
  // line on a log axis, not a horizontal threshold.
  referenceCurve?: {
    at: (x: number) => number;
    stroke: string;
    strokeDasharray?: string;
    strokeWidth?: number;
    label?: string;
    markerX?: number;
  };
  // Short horizontal tick sitting ON the series at a given x, matching the AC
  // marker style used by CustomHorizonChart. y is interpolated from the series
  // so the tick is always on the curve.
  seriesMarkerAtX?: {
    x: number;
    stroke: string;
    halfWidthPx?: number;
  };
  displayEndYear: number;
  height?: number;
  logSuffix?: string;
  width?: number;
  showXAxis?: boolean;
  xDomainOverride?: [number, number];
  xTickCount?: number;
  xTickFormatter?: (value: number) => string;
  xLabel?: string;
  margin?: ChartMargin;
  showTitle?: boolean;
  /** If true, the data values are already in log form (OOMs) and should be converted via Math.pow(10, v) for display */
  isDataInLogForm?: boolean;
  /** Sample trajectories for Monte Carlo visualization */
  sampleTrajectories?: ChartDataPoint[][];
  /** If set, render a stacked-area decomposition: each entry is a cumulative series
   *  (drawn back-to-front) filled to the axis bottom, producing visible bands.
   *  e.g. [{key:'etsp', fill:SE}, {key:'trainingCompute', fill:TC}] => TC band + SE band. */
  stackAreas?: StackAreaDef[];
  /** Observed datapoints scattered over the series. Defaults to OBSERVED_POINTS_BY_KEY[dataKey]. */
  observedPoints?: ObservedMetricPoint[];
  /** Milestone dots drawn ON the series (y read off the curve), labelled and styled
   *  like the takeoff-period chart's markers. The data point nearest each milestone
   *  also gets `milestoneLabels`, which SmallChartMetricTooltip renders as the
   *  milestone's name and explanation. */
  milestoneMarkers?: Array<{ x: number; label: string }>;
}

function CustomMetricChartComponent({
  chartData,
  tooltip,
  className,
  title,
  dataKey,
  yFormatter,
  yScale = 'linear',
  color = '#2A623D',
  domainOverride,
  verticalReferenceLine,
  horizontalReferenceLine,
  referenceCurve,
  seriesMarkerAtX,
  displayEndYear,
  height,
  logSuffix,
  width,
  showXAxis,
  xDomainOverride,
  xTickCount,
  xTickFormatter,
  xLabel,
  margin,
  showTitle = true,
  isDataInLogForm = false,
  sampleTrajectories = [],
  stackAreas,
  observedPoints,
  milestoneMarkers,
}: CustomMetricChartProps) {
  const resolvedStackAreas = stackAreas ?? STACK_DECOMPOSITIONS[dataKey];
  // Opaque stacks (Epoch-style solid bands) fill the whole plot rect, so they get
  // extra top margin to clear the overlaid title, and no endpoint dots/labels
  // (a 100%-stacked series' endpoints are just the axis limits, not information).
  const stackIsOpaque = !!resolvedStackAreas?.some(a => (a.fillOpacity ?? 0.35) >= 1);
  const resolvedObservedPoints = observedPoints ?? OBSERVED_POINTS_BY_KEY[dataKey];
  const resolvedYFormatter = useMemo<(value: number) => string | ReactNode>(() => {
    if (yFormatter) {
      return yFormatter;
    }

    return yScale === 'log'
      ? (value: number) => formatAsPowerOfTenText(value)
      : (value: number) => formatTo3SigFigs(value);
  }, [yFormatter, yScale]);

  const containerRef = useRef<HTMLDivElement>(null);
  const headerRef = useRef<HTMLDivElement>(null);
  // When `height` is provided it's the source of truth; otherwise we measure the
  // container. Deriving chartHeight (rather than mirroring the prop into state in
  // an effect) keeps it reactive to prop changes without a set-state-in-effect.
  const [measuredHeight, setMeasuredHeight] = useState<number>(0);
  const chartHeight = height ?? measuredHeight;

  const [hoveredObserved, setHoveredObserved] = useState<ObservedMetricPoint | null>(null);
  const [observedMousePos, setObservedMousePos] = useState<{ x: number; y: number } | null>(null);

  useEffect(() => {
    if (height != null) {
      return;
    }

    const updateHeight = () => {
      if (!containerRef.current) return;

      const containerHeight = containerRef.current.clientHeight;
      setMeasuredHeight(containerHeight);
    };

    updateHeight();

    const observer = new ResizeObserver(updateHeight);
    if (containerRef.current) {
      observer.observe(containerRef.current);
    }

    return () => {
      observer.disconnect();
    };
  }, [height]);

  const yDomain = useMemo(() => {
    const observed: number[] = [];

    // Collect all values for this metric (or, for a stacked decomposition, across
    // every band key so the y-domain spans from the lowest band to the top total).
    const domainKeys = resolvedStackAreas ? resolvedStackAreas.map(a => a.key) : [dataKey];
    chartData.forEach(point => {
      if (point.year > displayEndYear) return;
      for (const k of domainKeys) {
        const value = point[k as keyof ChartDataPoint];
        if (typeof value === 'number' && Number.isFinite(value)) {
          observed.push(value);
        }
      }
    });

    if (observed.length === 0) {
      return [0, 1] as [number, number];
    }

    const minObserved = Math.min(...observed);
    const maxObserved = Math.max(...observed);

    let domain: [number, number];

    if (domainOverride) {
      domain = typeof domainOverride === 'function' ? domainOverride(observed) : domainOverride;
    } else if (yScale === 'log' && isDataInLogForm) {
      // Data is already in log form (OOMs), so use linear domain on the log values
      // Add some padding in log space
      const range = maxObserved - minObserved;
      const padding = Math.max(range * 0.1, 0.5); // At least 0.5 OOM padding
      domain = [minObserved - padding, maxObserved + padding];
    } else if (yScale === 'log') {
      const positiveValues = observed.filter(v => v > 0);
      if (positiveValues.length === 0) {
        domain = [1, 10];
      } else {
        const minPositive = Math.min(...positiveValues);
        const maxPositive = Math.max(...positiveValues);
        let lower = minPositive * 0.9;
        const upper = maxPositive * 1.1;

        if (lower <= 0) {
          lower = minPositive * 0.1;
        }

        const safeLower = Math.max(lower, Number.MIN_VALUE);
        let safeUpper = Math.max(upper, safeLower * 1.1);

        if (!Number.isFinite(safeUpper) || safeUpper <= safeLower) {
          safeUpper = safeLower * 10;
        }

        domain = [safeLower, safeUpper];
      }
    } else {
      const suggestedMax = maxObserved > 1 ? Math.floor(Math.max(1, Math.ceil(maxObserved * 1.1))) : 1;
      domain = [Math.min(0, minObserved), suggestedMax];
    }

    return domain;
  }, [chartData, dataKey, yScale, domainOverride, displayEndYear, isDataInLogForm, resolvedStackAreas]);

  const xDomain = useMemo((): [number, number] => {
    if (xDomainOverride && xDomainOverride[1] > xDomainOverride[0]) {
      return xDomainOverride;
    }

    const years = chartData
      .filter(d => {
        const value = d[dataKey as keyof ChartDataPoint];
        return value != null && !isNaN(value as number);
      })
      .map(d => d.year)
      .filter(y => !isNaN(y));

    if (years.length === 0) return [2020, 2030];

    // Use dynamic domain calculation that considers vertical reference line (intersection)
    const intersectionX = verticalReferenceLine?.x ?? null;
    return calculateDynamicXDomain(years, {
      intersectionPoints: [intersectionX],
      minPadding: 0.01,
      maxPadding: 0.15,
      minRange: 3, // Minimum 3 years visible
      maxDomain: displayEndYear // Constrain to smart display end year
    });
  }, [chartData, dataKey, verticalReferenceLine, displayEndYear, xDomainOverride]);

  // Prepare trajectory lines
  const trajectoryKeys = useMemo(() => {
    if (chartData.length === 0) return [];
    const sample = chartData[0];
    return Object.keys(sample).filter(key => key.startsWith(`${dataKey}_trajectory_`));
  }, [chartData, dataKey]);

  // Generate keys for sample trajectories (Monte Carlo)
  const sampleKeys = useMemo(() => {
    return sampleTrajectories.map((_, index) => `sample_${dataKey}_${index}`);
  }, [sampleTrajectories, dataKey]);

  // Add x property to chartData and merge in sample trajectories
  // The points mergedData is built from: everything up to displayEndYear, falling back
  // to the full series when that window is empty.
  const sourceData = useMemo(() => {
    const withinDisplay = chartData.filter(point => point.year <= displayEndYear);
    return withinDisplay.length > 0 ? withinDisplay : chartData;
  }, [chartData, displayEndYear]);

  // Each milestone is tagged onto the single data point closest to it, so hovering
  // near the dot surfaces the milestone in the tooltip. Same construction as the
  // takeoff-period chart's overlay; a year may carry more than one milestone.
  const milestoneLabelsByYear = useMemo(() => {
    const byYear = new Map<number, string[]>();
    if (!milestoneMarkers || milestoneMarkers.length === 0) {
      return byYear;
    }

    const years = sourceData
      .map(point => point.year)
      .filter(year => typeof year === 'number' && Number.isFinite(year));

    milestoneMarkers.forEach(marker => {
      if (!Number.isFinite(marker.x)) return;
      let closestYear: number | null = null;
      let minDiff = Infinity;
      years.forEach(year => {
        const diff = Math.abs(year - marker.x);
        if (diff < minDiff) {
          minDiff = diff;
          closestYear = year;
        }
      });
      if (closestYear !== null) {
        const existing = byYear.get(closestYear);
        if (existing) {
          existing.push(marker.label);
        } else {
          byYear.set(closestYear, [marker.label]);
        }
      }
    });

    return byYear;
  }, [milestoneMarkers, sourceData]);

  const mergedData = useMemo<(ChartDataPoint & { x: number } & Record<string, unknown>)[]>(() => {
    return sourceData.map(point => {
      const newPoint: ChartDataPoint & { x: number } & Record<string, unknown> = { ...point, x: point.year };
      
      // Add sample trajectory values
      sampleTrajectories.forEach((trajectory, sampleIndex) => {
        const key = `sample_${dataKey}_${sampleIndex}`;
        const samplePoint = trajectory.find(p => p.year === point.year);
        // Get the value for this dataKey from the sample trajectory
        const value = samplePoint ? samplePoint[dataKey as keyof ChartDataPoint] : null;
        newPoint[key] = value ?? null;
      });

      const labels = milestoneLabelsByYear.get(point.year);
      if (labels) {
        // ChartDataPoint's index signature is scalar-only; the tooltip reads this
        // back through the same widened view (see MilestoneTooltipSection).
        (newPoint as Record<string, unknown>)['milestoneLabels'] = labels;
      }

      return newPoint;
    });
  }, [sourceData, sampleTrajectories, dataKey, milestoneLabelsByYear]);

  // Build line configs
  const lines = useMemo(() => {
    const result = [];

    // Add sample trajectory lines first (so they're behind everything)
    for (const key of sampleKeys) {
      result.push({
        dataKey: key,
        stroke: color,
        strokeWidth: 1,
        strokeOpacity: 0.15,
      });
    }

    // Add trajectory lines (behind main line)
    for (const key of trajectoryKeys) {
      result.push({
        dataKey: key,
        stroke: color,
        strokeWidth: 1,
        strokeOpacity: 0.3,
      });
    }

    if (resolvedStackAreas && resolvedStackAreas.length) {
      // Stacked-area decomposition: cumulative series back-to-front, each filled to
      // the axis bottom. Drawing the largest (e.g. ETSP) first and smaller ones on
      // top yields visible bands (e.g. TC band below, SE band above).
      for (const a of resolvedStackAreas) {
        const opaque = (a.fillOpacity ?? 0.35) >= 1;
        result.push({
          dataKey: a.key,
          stroke: a.stroke,
          // Opaque bands separate themselves; a heavy stroke would just outline them.
          strokeWidth: opaque ? 1 : 2,
          strokeOpacity: 1,
          fill: a.fill,
          fillOpacity: a.fillOpacity ?? 0.35,
          name: a.name,
        });
      }
    } else {
      // Add main line
      result.push({
        dataKey: dataKey,
        stroke: color,
        strokeWidth: 3,
        strokeOpacity: 1,
        name: title,
      });
    }

    return result;
  }, [trajectoryKeys, sampleKeys, dataKey, color, title, resolvedStackAreas]);

  type EndpointLabelRenderer = ((context: CustomElementsContext) => ReactNode) & { displayName?: string };

  const formatLogEndpoint = useCallback((value: number): ReactNode => {
    // If data is already in log form (OOMs), convert to actual value first
    if (isDataInLogForm) {
      const actualValue = Math.pow(10, value);
      return formatCompactNumberNode(actualValue, {
        renderMode: 'svg',
        mantissaOpacity: 0.7,
        mantissaFontSize: 12,
        exponentFontSize: 12,
        suffix: logSuffix,
      });
    }

    if (!Number.isFinite(value) || value <= 0) {
      return '';
    }

    // Use formatCompactNumberNode to properly handle mantissas (e.g., 3×10^-8)
    return formatCompactNumberNode(value, {
      renderMode: 'svg',
      mantissaOpacity: 0.7,
      mantissaFontSize: 12,
      exponentFontSize: 12,
      suffix: logSuffix,
    });

  }, [logSuffix, isDataInLogForm]);

  const formatEndpointValue = useCallback((value: number): ReactNode => {
    if (!Number.isFinite(value)) {
      return '';
    }

    // For log-scale charts (including those with data already in log form)
    if (yScale === 'log') {
      return formatLogEndpoint(value);
    }
    
    // For data in log form but not explicitly marked as log scale
    if (isDataInLogForm) {
      return formatLogEndpoint(value);
    }

    if (yFormatter) {
      const formatted = yFormatter(value);
      if (typeof formatted === 'string' || typeof formatted === 'number') {
        return formatted;
      }
      return formatted;
    }

    return formatTo3SigFigs(value);
  }, [yFormatter, yScale, formatLogEndpoint, isDataInLogForm]);

  const customElements = useMemo(() => {
    const seriesPoints: XYPoint[] = mergedData
      .filter(point => {
        const value = point[dataKey as keyof ChartDataPoint];
        return typeof value === 'number' && Number.isFinite(value) && point.x <= displayEndYear;
      })
      .map(point => ({ x: point.x as number, y: point[dataKey as keyof ChartDataPoint] as number }))
      .sort((a, b) => a.x - b.x);

    if (seriesPoints.length === 0) {
      return undefined;
    }

    const EndpointLabels: EndpointLabelRenderer = (context) => {
      const { chartWidth, chartHeight, xScale, yScale: yScaleFunc, xDomain: renderedXDomain } = context;

      const { left, right } = getVisibleEndpoints(seriesPoints, renderedXDomain[0], renderedXDomain[1]);
      const elements: ReactNode[] = [];

      if (horizontalReferenceLine) {
        const refY = yScaleFunc(horizontalReferenceLine.y);
        if (Number.isFinite(refY) && refY >= 0 && refY <= chartHeight) {
          elements.push(
            <g key="metric-horizontal-reference">
              <line
                x1={0}
                x2={chartWidth}
                y1={refY}
                y2={refY}
                stroke={horizontalReferenceLine.stroke}
                strokeWidth={horizontalReferenceLine.strokeWidth ?? 1.5}
                strokeDasharray={horizontalReferenceLine.strokeDasharray ?? '5 5'}
                strokeOpacity={horizontalReferenceLine.strokeOpacity ?? 0.9}
              />
              {horizontalReferenceLine.label && (
                <text
                  x={horizontalReferenceLine.labelPosition === 'left' ? 6 : chartWidth - 6}
                  y={refY - 6}
                  textAnchor={horizontalReferenceLine.labelPosition === 'left' ? 'start' : 'end'}
                  fontSize={11}
                  fontWeight={600}
                  fill={horizontalReferenceLine.stroke}
                  stroke="var(--background, #ffffff)"
                  strokeWidth={3}
                  paintOrder="stroke"
                >
                  {horizontalReferenceLine.label}
                </text>
              )}
            </g>
          );
        }
      }

      if (referenceCurve) {
        const [rx0, rx1] = renderedXDomain;
        const N = 160;
        const pts: string[] = [];
        for (let i = 0; i <= N; i += 1) {
          const xv = rx0 + ((rx1 - rx0) * i) / N;
          const yv = referenceCurve.at(xv);
          if (!Number.isFinite(yv) || yv <= 0) continue;
          const px = xScale(xv);
          const py = yScaleFunc(yv);
          if (!Number.isFinite(px) || !Number.isFinite(py)) continue;
          if (py < -2 || py > chartHeight + 2) continue;   // clip to the plot area
          pts.push(`${px},${py}`);
        }
        if (pts.length > 1) {
          elements.push(
            <polyline
              key="metric-reference-curve"
              points={pts.join(' ')}
              fill="none"
              stroke={referenceCurve.stroke}
              strokeWidth={referenceCurve.strokeWidth ?? 1.5}
              strokeDasharray={referenceCurve.strokeDasharray ?? '6 3'}
              strokeOpacity={0.9}
            />
          );
        }
        if (referenceCurve.markerX != null) {
          const mx = xScale(referenceCurve.markerX);
          const my = yScaleFunc(referenceCurve.at(referenceCurve.markerX));
          if (Number.isFinite(mx) && Number.isFinite(my) && my >= 0 && my <= chartHeight) {
            elements.push(
              <circle key="metric-reference-marker" cx={mx} cy={my} r={4}
                      fill={referenceCurve.stroke} stroke="none" />
            );
          }
        }
        if (referenceCurve.label && pts.length > 1) {
          const [lx, ly] = pts[Math.min(12, pts.length - 1)].split(',').map(Number);
          elements.push(
            <text key="metric-reference-label" x={lx + 6} y={ly - 6}
                  textAnchor="start" fontSize={11} fontWeight={600}
                  fill={referenceCurve.stroke}
                  stroke="var(--background, #ffffff)" strokeWidth={3} paintOrder="stroke">
              {referenceCurve.label}
            </text>
          );
        }
      }

      let visibleSeries = seriesPoints.filter(
        point => point.x >= renderedXDomain[0] - 1e-6 && point.x <= renderedXDomain[1] + 1e-6
      );
      if (visibleSeries.length === 0) {
        visibleSeries = [...seriesPoints];
      }
      if (left && (visibleSeries.length === 0 || Math.abs(visibleSeries[0].x - left.x) > 1e-6)) {
        visibleSeries = [left, ...visibleSeries];
      }
      if (right && (visibleSeries.length === 0 || Math.abs(visibleSeries[visibleSeries.length - 1].x - right.x) > 1e-6)) {
        visibleSeries = [...visibleSeries, right];
      }

      if (seriesMarkerAtX && seriesPoints.length > 1) {
        const mx = seriesMarkerAtX.x;
        let yAt: number | null = null;
        for (let i = 1; i < seriesPoints.length; i += 1) {
          const a = seriesPoints[i - 1];
          const b = seriesPoints[i];
          if (mx >= a.x && mx <= b.x) {
            const t = b.x === a.x ? 0 : (mx - a.x) / (b.x - a.x);
            yAt = a.y + (b.y - a.y) * t;
            break;
          }
        }
        if (yAt != null && Number.isFinite(yAt)) {
          const px = xScale(mx);
          const py = yScaleFunc(yAt);
          if (Number.isFinite(px) && Number.isFinite(py) && py >= 0 && py <= chartHeight) {
            const hw = seriesMarkerAtX.halfWidthPx ?? 16;
            elements.push(
              <g key="metric-series-marker" style={{ pointerEvents: 'none' }}>
                <line x1={px - hw} y1={py} x2={px + hw} y2={py}
                      stroke={seriesMarkerAtX.stroke} strokeWidth={2} strokeLinecap="round" />
                <circle cx={px} cy={py} r={3} fill={seriesMarkerAtX.stroke} />
              </g>
            );
          }
        }
      }

      // Observed datapoints. Only those inside the rendered window are drawn: the
      // series can run many orders of magnitude past them on a log axis, and a marker
      // clamped to the edge would read as a real measurement sitting at the axis limit.
      if (resolvedObservedPoints && resolvedObservedPoints.length > 0) {
        const [ox0, ox1] = renderedXDomain;
        const placed = resolvedObservedPoints
          .filter(point => point.x >= ox0 && point.x <= ox1)
          .map(point => ({ point, px: xScale(point.x), py: yScaleFunc(point.y) }))
          .filter(({ px, py }) =>
            Number.isFinite(px) && Number.isFinite(py) && py >= 0 && py <= chartHeight
          );

        if (placed.length > 0) {
          // The small-chart grid is a fraction of the primary chart's height; a marker
          // sized for the big chart would swamp it.
          const radius = chartHeight >= 240 ? 6 : 3.5;
          const strokeWidth = chartHeight >= 240 ? 2 : 1;
          const hitRadius = Math.max(12, radius * 2.5);

          const handleObservedMove = (e: React.MouseEvent<SVGRectElement>) => {
            // The overlay rect spans exactly the plot area, so measuring against it
            // gives coordinates in the same space as xScale/yScale output.
            const overlay = e.currentTarget.getBoundingClientRect();
            const mouseX = e.clientX - overlay.left;
            const mouseY = e.clientY - overlay.top;

            let closest: ObservedMetricPoint | null = null;
            let minDistance = Infinity;
            for (const { point, px, py } of placed) {
              const distance = Math.hypot(mouseX - px, mouseY - py);
              if (distance < minDistance) {
                minDistance = distance;
                closest = point;
              }
            }

            if (closest && minDistance <= hitRadius) {
              // The tooltip is absolutely positioned within the component's outer div,
              // so translate the cursor into that box rather than the plot area's.
              const container = containerRef.current?.getBoundingClientRect();
              setObservedMousePos({
                x: e.clientX - (container?.left ?? overlay.left),
                y: e.clientY - (container?.top ?? overlay.top),
              });
              setHoveredObserved(closest);
            } else {
              setObservedMousePos(null);
              setHoveredObserved(null);
            }
          };

          const handleObservedLeave = () => {
            setObservedMousePos(null);
            setHoveredObserved(null);
          };

          elements.push(
            <rect
              key="metric-observed-hit-area"
              x={0}
              y={0}
              width={chartWidth}
              height={chartHeight}
              fill="transparent"
              onMouseMove={handleObservedMove}
              onMouseLeave={handleObservedLeave}
              style={{ pointerEvents: 'all' }}
            />
          );

          elements.push(
            <g key="metric-observed-points" style={{ pointerEvents: 'none' }}>
              {placed.map(({ point, px, py }) => {
                const key = `metric-observed-${point.label}-${point.dateLabel}-${point.x}`;
                const fill = point.fill ?? OBSERVED_POINT_FILL;
                const stroke = hoveredObserved === point ? '#000000' : '#ffffff';
                const shapePoints = observedMarkPoints(
                  point.shape ?? OBSERVED_POINT_SHAPE,
                  px,
                  py,
                  radius
                );
                return shapePoints === null ? (
                  <circle
                    key={key}
                    cx={px}
                    cy={py}
                    r={radius * (5 / 6)}
                    fill={fill}
                    stroke={stroke}
                    strokeWidth={strokeWidth}
                  />
                ) : (
                  <polygon
                    key={key}
                    points={shapePoints}
                    fill={fill}
                    stroke={stroke}
                    strokeWidth={strokeWidth}
                  />
                );
              })}
            </g>
          );
        }
      }

      const segments = pointsToPixelSegments(visibleSeries, xScale, yScaleFunc, chartHeight);
      const leftSegments = segments.slice(0, SEGMENTS_TO_CHECK);
      const rightSegments = segments.slice(Math.max(segments.length - SEGMENTS_TO_CHECK, 0));

      // Boxes the milestone labels below must not overprint.
      const endpointLabelBounds: LabelBounds[] = [];

      const renderEndpoint = (endpoint: XYPoint | null, align: 'left' | 'right', key: string) => {
        if (!endpoint) {
          return;
        }

        const xPos = xScale(endpoint.x);
        const yPosRaw = yScaleFunc(endpoint.y);

        if (!Number.isFinite(xPos) || !Number.isFinite(yPosRaw)) {
          return;
        }

        // An endpoint outside the plot area (possible when a domainOverride caps
        // the y-axis below the series, as the uplift chart does past AC) would be
        // clamped onto the chart edge -- a dot marking the corner, not the data.
        if (yPosRaw < 0 || yPosRaw > chartHeight) {
          return;
        }

        const yPos = clamp(yPosRaw, 0, chartHeight);

        const label = formatEndpointValue(endpoint.y);
        if (label == null || label === '') {
          return;
        }

        const labelWidth = estimateLabelWidth(label);
        const placement = selectLabelPosition(
          align,
          xPos,
          yPos,
          labelWidth,
          LABEL_HEIGHT,
          chartWidth,
          chartHeight,
          align === 'left' ? leftSegments : rightSegments
        );

        endpointLabelBounds.push(
          computeLabelBounds(placement.textX, placement.textY, placement.anchor, labelWidth, LABEL_HEIGHT)
        );

        elements.push(
          <g key={key}>
            <circle cx={xPos} cy={yPos} r={3} fill={color} />
            <text
              x={placement.textX}
              y={placement.textY}
              textAnchor={placement.anchor}
              alignmentBaseline="middle"
              fontSize={11}
              fill={color}
              fontWeight="500"
            >
              {label}
            </text>
          </g>
        );
      };

      if (!stackIsOpaque) {
        renderEndpoint(left, 'left', 'metric-left-endpoint');
        renderEndpoint(right, 'right', 'metric-right-endpoint');
      }

      // Milestone dots, sitting ON the series (y read off the curve at the milestone's
      // date) and labelled like the takeoff-period chart's markers. Milestones outside
      // the rendered window -- or outside the series' own span -- are dropped rather
      // than clamped to an edge, where they would read as landing there.
      //
      // Drawn last, after the endpoint labels, so their boxes can be handed over as
      // obstacles: the x-axis ends half a year past ASI, which puts the ASI label
      // right where the series' endpoint value sits.
      if (milestoneMarkers && milestoneMarkers.length > 0 && seriesPoints.length > 1) {
        const [mx0, mx1] = renderedXDomain;
        const seriesStart = seriesPoints[0].x;
        const seriesEnd = seriesPoints[seriesPoints.length - 1].x;

        const positioned = milestoneMarkers
          .filter(marker =>
            Number.isFinite(marker.x)
            && marker.x >= Math.max(mx0, seriesStart)
            && marker.x <= Math.min(mx1, seriesEnd)
          )
          .map(marker => {
            const yAt = interpolateAtX(seriesPoints, marker.x);
            return {
              label: marker.label,
              xPos: xScale(marker.x),
              yPos: yAt == null ? NaN : yScaleFunc(yAt),
            };
          })
          .filter(m => Number.isFinite(m.xPos) && Number.isFinite(m.yPos));

        if (positioned.length > 0) {
          elements.push(
            <g key="metric-milestone-markers" style={{ pointerEvents: 'none' }}>
              <ChartMilestoneMarkers
                markers={positioned}
                chartWidth={chartWidth}
                chartHeight={chartHeight}
                avoidBounds={endpointLabelBounds}
              />
            </g>
          );
        }
      }

      return <g>{elements}</g>;
    };

    EndpointLabels.displayName = 'CustomMetricChartEndpointLabels';

    return EndpointLabels;
  }, [mergedData, dataKey, displayEndYear, formatEndpointValue, color, horizontalReferenceLine, referenceCurve, seriesMarkerAtX, resolvedObservedPoints, hoveredObserved, stackIsOpaque, milestoneMarkers]);

  return (
    <div
      ref={containerRef}
      className={`flex min-h-0 flex-col relative ${className ? className : ''}`.trim()}
      style={{ height: '100%' }}
    >
      {showTitle ? (
        <h3
          ref={headerRef}
          className="text-[9px] uppercase font-light text-gray-700 font-system-mono absolute left-0 max-w-40"
        >
          {title}
        </h3>
      ) : null}
      <div className="flex-1 min-h-0">
        <CustomLineChart
          data={mergedData}
          height={chartHeight}
          margin={margin ?? { top: stackIsOpaque ? 26 : 3, right: 10, left: 4, bottom: 3 }}
          xDomain={xDomain}
          xTickCount={xTickCount ?? 5}
          xTickFormatter={xTickFormatter ?? ((value) => Math.round(value).toString())}
          yDomain={yDomain}
          // When data is already in log form, use linear scale on the y-axis
          // (the data values are already logarithms, so linear spacing = log spacing of actual values)
          yScale={isDataInLogForm ? 'linear' : yScale}
          yTickFormatter={resolvedYFormatter}
          lines={lines}
          tooltip={tooltip}
          suppressTooltip={hoveredObserved !== null}
          width={width}
          showXAxis={showXAxis ?? false}
          xLabel={xLabel}
          customElements={customElements}
        />
      </div>
      {hoveredObserved && observedMousePos && (
        <div
          style={{
            position: 'absolute',
            left: observedMousePos.x + 10,
            top: observedMousePos.y + 10,
            pointerEvents: 'none',
            zIndex: 1000,
          }}
        >
          <div style={tooltipBoxStyle}>
            <span style={tooltipHeaderStyle}>{hoveredObserved.label}</span>
            {hoveredObserved.subLabel && <span>{hoveredObserved.subLabel}</span>}
            <span>{hoveredObserved.dateLabel}</span>
            <span style={tooltipValueStyle}>{resolvedYFormatter(hoveredObserved.y)}</span>
            <span style={{ opacity: 0.7 }}>{hoveredObserved.detail}</span>
          </div>
        </div>
      )}
    </div>
  );
}

export const CustomMetricChart = memo(CustomMetricChartComponent);

CustomMetricChart.displayName = 'CustomMetricChart';
