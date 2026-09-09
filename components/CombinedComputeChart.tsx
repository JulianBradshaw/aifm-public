'use client';

import { memo, useMemo, ReactNode, useState } from 'react';
import { CustomLineChart, DataPoint, CustomElementsContext } from './CustomLineChart';
import { ChartDataPoint } from '@/app/types';
import { calculateDynamicXDomain, calculateTicks, clamp } from '@/utils/chartUtils';
import { formatYearMonth, formatPowerOfTenNode, formatEffectiveComputeValue, formatOOMSuperscriptText } from '@/utils/formatting';
import { tooltipBoxStyle, tooltipHeaderStyle, tooltipValueStyle } from './chartTooltipStyle';
import { CHART_LAYOUT } from '@/constants/chartLayout';
import { COMPUTE_CHART_EXPLANATION } from '@/constants/chartExplanations';
import { WithChartTooltip } from './ChartTitleTooltip';
import { XYPoint, interpolateAtX, getVisibleEndpoints } from '@/utils/chartLabelUtils';

export interface CombinedComputeChartProps {
  chartData: ChartDataPoint[];
  className?: string;
  title?: string;
  height?: number;
  width?: number;
  domainOverride?: [number, number];
  verticalReferenceLine?: {
    x: number;
    stroke: string;
    strokeDasharray?: string;
    strokeWidth?: number;
    label?: string;
    strokeOpacity?: number;
  };
  displayEndYear: number;
  showTrainingSeries?: boolean;
  sampleTrajectories?: ChartDataPoint[][];
  isLoading?: boolean;
  /** Replaces the title text, e.g. with a chart selector. Mirrors CustomHorizonChart. */
  titleControl?: ReactNode;
}

export const CombinedComputeChart = memo<CombinedComputeChartProps>(({ chartData, className, title = 'Effective & Training Compute', height, width, domainOverride, verticalReferenceLine, displayEndYear, showTrainingSeries = true, sampleTrajectories = [], isLoading = false, titleControl }) => {

  const yDomain = useMemo((): [number, number] => {
    const recentData: ChartDataPoint[] = chartData.filter(p => p.year <= displayEndYear);
    const observed: number[] = [
        ...recentData.map(p => p.effectiveCompute).filter((c: number | null | undefined): c is number => c != null && Number.isFinite(c)),
    ];
    if (showTrainingSeries) {
      observed.push(
        ...recentData.map(p => p.trainingCompute).filter((c: number | null | undefined): c is number => c != null && Number.isFinite(c)),
      );
    }

    const maxObserved = observed.length > 0 ? Math.max(...observed) : 0;
    const minObserved = observed.length > 0 ? Math.min(...observed) : 0;
    const suggestedMax = maxObserved + 2;
    const suggestedMin = minObserved - 1;
    if (domainOverride) {
      return domainOverride;
    }
    return [suggestedMin, suggestedMax];
  }, [chartData, displayEndYear, domainOverride, showTrainingSeries]);

  const xDomain = useMemo((): [number, number] => {
    const years = chartData
      .filter(d => {
        const hasEC = d.effectiveCompute != null && !isNaN(d.effectiveCompute as number);
        const hasTC = showTrainingSeries && d.trainingCompute != null && !isNaN(d.trainingCompute as number);
        return hasEC || hasTC;
      })
      .map(d => d.year)
      .filter(y => !isNaN(y));

    if (years.length === 0) return [2020, displayEndYear];

    const intersectionX = verticalReferenceLine?.x ?? null;
    return calculateDynamicXDomain(years, {
      intersectionPoints: [intersectionX],
      minPadding: 0.05,
      maxPadding: 0.15,
      minRange: 3,
      maxDomain: displayEndYear,
    });
  }, [chartData, verticalReferenceLine, displayEndYear, showTrainingSeries]);

  // Generate x-axis ticks at whole year boundaries, limited to ~4 ticks
  const xTicks = useMemo(() => {
    const [start, end] = xDomain;
    const targetTickCount = 4;
    
    // Find all whole years in range
    const firstYear = Math.ceil(start);
    const lastYear = Math.floor(end);
    const yearCount = lastYear - firstYear + 1;
    
    if (yearCount <= 0) {
      return [Math.round(start), Math.round(end)];
    }
    
    if (yearCount <= targetTickCount) {
      // If few enough years, show them all
      const ticks: number[] = [];
      for (let year = firstYear; year <= lastYear; year++) {
        ticks.push(year);
      }
      return ticks;
    }
    
    // Otherwise, sample evenly to get ~targetTickCount ticks
    const step = Math.ceil(yearCount / targetTickCount);
    const ticks: number[] = [];
    for (let year = firstYear; year <= lastYear; year += step) {
      ticks.push(year);
    }
    // Always include the last year if not already included
    if (ticks[ticks.length - 1] !== lastYear) {
      ticks.push(lastYear);
    }
    return ticks;
  }, [xDomain]);

  // Generate keys for sample trajectories
  const sampleKeys = useMemo(() => {
    return sampleTrajectories.map((_, index) => `sample_compute_${index}`);
  }, [sampleTrajectories]);

  const mergedData = useMemo(() => {
    // Only use years from main chart data (don't add years from samples)
    return chartData.map(point => {
      const newPoint: DataPoint = { ...point, x: point.year };
      
      // Initialize all sample keys and add their values if available
      sampleTrajectories.forEach((trajectory, sampleIndex) => {
        const key = `sample_compute_${sampleIndex}`;
        const samplePoint = trajectory.find(p => p.year === point.year);
        // Always set the key, use null if no matching point found
        newPoint[key] = samplePoint?.effectiveCompute ?? null;
      });
      
      return newPoint;
    });
  }, [chartData, sampleTrajectories]);

  const lines = useMemo(() => {
    const result: Array<{ dataKey: string; stroke: string; strokeWidth?: number; strokeOpacity?: number; name?: string }> = [];

    // Add sample trajectory lines first (behind everything)
    // They fade in on hover
    for (const key of sampleKeys) {
      result.push({ dataKey: key, stroke: '#2A623D', strokeWidth: 1, strokeOpacity: 0.15 });
    }

    // Add main lines on top
    result.push({ dataKey: 'effectiveCompute', stroke: '#2A623D', strokeWidth: 3, strokeOpacity: 1, name: 'Effective' });
    if (showTrainingSeries) {
      result.push({ dataKey: 'trainingCompute', stroke: '#888', strokeWidth: 3, strokeOpacity: 1, name: 'Training' });
    }

    return result;
  }, [showTrainingSeries, sampleKeys]);

  const yTicks = useMemo(() => {
    return calculateTicks(yDomain, { targetCount: 6 }, 'linear').filter(t => t !== 0);
  }, [yDomain]);

  const tooltip = (point: DataPoint): ReactNode => {
    const year = point['year'] as number | undefined;
    const ec = typeof point['effectiveCompute'] === 'number' ? (point['effectiveCompute'] as number) : null;
    const tc = typeof point['trainingCompute'] === 'number' ? (point['trainingCompute'] as number) : null;

    return (
      <div style={tooltipBoxStyle}>
        {typeof year === 'number' && (
          <span style={tooltipHeaderStyle}>{formatYearMonth(year)}</span>
        )}
        {ec != null && (
          <span style={{ display: 'inline-flex', alignItems: 'center', gap: '6px' }}>
            <span style={{ width: '6px', height: '6px', borderRadius: '9999px', backgroundColor: '#2A623D' }} />
            <span style={{ ...tooltipValueStyle, color: '#2A623D' }}>{formatEffectiveComputeValue(ec)} FLOP</span>
          </span>
        )}
        {showTrainingSeries && tc != null && (
          <span style={{ display: 'inline-flex', alignItems: 'center', gap: '6px' }}>
            <span style={{ width: '6px', height: '6px', borderRadius: '9999px', backgroundColor: '#888' }} />
            <span style={{ ...tooltipValueStyle, color: '#888' }}>{formatEffectiveComputeValue(tc)} FLOP</span>
          </span>
        )}
      </div>
    );
  };

  // Helper utilities for label positioning and values
  type SeriesKey = 'effectiveCompute' | 'trainingCompute';

  const getSeriesPoints = (key: SeriesKey): XYPoint[] => {
    return mergedData
      .filter(d => d[key] != null && !isNaN(d[key] as number))
      .map(d => ({ x: d.x as number, y: d[key] as number }))
      .sort((a, b) => a.x - b.x);
  };

  const getSeriesEndpoints = (key: SeriesKey, xMin: number, xMax: number) => {
    const pts = getSeriesPoints(key);
    return getVisibleEndpoints(pts, xMin, xMax);
  };

  const findIntersection = (): { x: number; y: number } | null => {
    if (!showTrainingSeries) return null;
    const ecPts = getSeriesPoints('effectiveCompute');
    const tcPts = getSeriesPoints('trainingCompute');
    if (ecPts.length < 2 || tcPts.length < 2) return null;

    // Build unified sorted x grid across both series
    const xs = Array.from(new Set([...ecPts.map(p => p.x), ...tcPts.map(p => p.x)])).sort((a, b) => a - b);
    if (xs.length < 2) return null;

    // Walk segments for sign change of (ec - tc)
    let prevX: number | null = null;
    let prevDiff: number | null = null;
    for (const x of xs) {
      const ec = interpolateAtX(ecPts, x);
      const tc = interpolateAtX(tcPts, x);
      if (ec == null || tc == null) continue;
      const diff = ec - tc;
      if (prevX != null && prevDiff != null && diff !== prevDiff && (diff === 0 || prevDiff === 0 || diff * prevDiff < 0)) {
        // Linear interpolate zero crossing between prevX and x
        const x1 = prevX;
        const x2 = x;
        const d1 = prevDiff;
        const d2 = diff;
        const denom = (d2 - d1);
        if (denom === 0) continue;
        const t = (0 - d1) / denom;
        const xi = x1 + t * (x2 - x1);
        const eci = interpolateAtX(ecPts, xi);
        const tci = interpolateAtX(tcPts, xi);
        if (eci != null && tci != null) {
          return { x: xi, y: (eci + tci) / 2 };
        }
      }
      prevX = x;
      prevDiff = diff;
    }

    // Fallback to vertical reference line if provided inside domain
    const fallbackX = verticalReferenceLine?.x;
    if (fallbackX != null && isFinite(fallbackX)) {
      const eci = interpolateAtX(ecPts, fallbackX);
      const tci = interpolateAtX(tcPts, fallbackX);
      if (eci != null && tci != null) {
        return { x: fallbackX, y: (eci + tci) / 2 };
      }
    }
    return null;
  };

  return (
    <div 
      className={className || 'flex-1'} 
      style={{ position: 'relative' }}
    >
      <div className="flex gap-2 items-center">
        {titleControl ?? <span className="text-[12px] font-semibold text-primary text-left font-system-mono">{title}</span>}
        <div className="flex-1 border-t border-gray-500/30" />
        <WithChartTooltip explanation={COMPUTE_CHART_EXPLANATION} className="!gap-0" isLoading={isLoading}><></></WithChartTooltip>
      </div>
      <CustomLineChart
        data={mergedData}
        height={height ?? 239}
        width={width}
        margin={{ top: 0, right: 20, left: 48, bottom: 35 }}
        xDomain={xDomain}
        xTicks={xTicks}
        xTickFormatter={(value) => Math.round(value).toString()}
        animateXDomain={false}
        yDomain={yDomain}
        showYAxis={false}
        yTicks={yTicks}
        yTickFormatter={(v) => formatOOMSuperscriptText(v)}
        lines={lines}
        tooltip={tooltip}
        customElements={(context) => {
          const { chartWidth, chartHeight, xDomain: renderedXDomain, xScale: xScalePx, yScale: yScalePx } = context;

          const ecEndpoints = getSeriesEndpoints('effectiveCompute', renderedXDomain[0], renderedXDomain[1]);
          const tcEndpoints = showTrainingSeries
            ? getSeriesEndpoints('trainingCompute', renderedXDomain[0], renderedXDomain[1])
            : { left: null, right: null };
          const intersection = findIntersection();

          const baseFontSize = 12;
          const superscriptFontSize = 12;
          const tensFontSize = 12;

          // Compute label positions and avoid overlap with basic nudge
          const elements: ReactNode[] = [];

          // Left labels: two-line label (series name on first line, value on second)
          const labelValueOptions = {
            renderMode: 'svg' as const,
            mantissaOpacity: 0.5,
            mantissaFontSize: tensFontSize,
            exponentFontSize: superscriptFontSize,
            suffix: ' FLOP',
            baselineShift: '0.3em' as const,
          };

          const leftLabels: Array<{ pt: { x: number; y: number }; color: string; label: string; value: ReactNode }> = [];
          if (ecEndpoints.left) {
            leftLabels.push({
              pt: ecEndpoints.left,
              color: '#2A623D',
              label: 'Effective',
              value: formatPowerOfTenNode(ecEndpoints.left.y, labelValueOptions)
            });
          }
          if (showTrainingSeries && tcEndpoints.left) {
            leftLabels.push({
              pt: tcEndpoints.left,
              color: '#666',
              label: 'Training',
              value: formatPowerOfTenNode(tcEndpoints.left.y, labelValueOptions)
            });
          }

          let lastLeftY: number | null = null;
          leftLabels.forEach((lbl, idx) => {
            const x = xScalePx(lbl.pt.x);
            let y = yScalePx(lbl.pt.y);
            // Clamp within chart area
            y = clamp(y, 0, chartHeight);
            // Basic anti-overlap
            if (lastLeftY != null && Math.abs(y - lastLeftY) < 16) y += 16 * (idx + 1);
            lastLeftY = y;
            const nearLeft = x < 12;
            const tx = nearLeft ? x + 6 : x - 6;
            const anchor = nearLeft ? 'start' : 'end';
            elements.push(
              <circle key={`left-dot-${idx}`} cx={x} cy={y} r={3} fill={lbl.color} />,
              <text key={`left-${idx}`} x={tx} y={y} textAnchor={anchor} alignmentBaseline="middle" fontSize={baseFontSize} fill={lbl.color}>
                <tspan x={tx} dy={-4}>{lbl.label}</tspan>
                <tspan x={tx} dy={16}>{lbl.value}</tspan>
              </text>
            );
          });

          // Right labels: value only
          const rightLabels: Array<{ pt: { x: number; y: number }; color: string; text: ReactNode }> = [];
          if (ecEndpoints.right) {
            rightLabels.push({
              pt: ecEndpoints.right,
              color: '#2A623D',
              text: formatPowerOfTenNode(ecEndpoints.right.y, labelValueOptions)
            });
          }
          if (showTrainingSeries && tcEndpoints.right) {
            rightLabels.push({
              pt: tcEndpoints.right,
              color: '#666',
              text: formatPowerOfTenNode(tcEndpoints.right.y, labelValueOptions)
            });
          }

          let lastRightY: number | null = null;
          rightLabels.forEach((lbl, idx) => {
            const x = xScalePx(lbl.pt.x);
            let y = yScalePx(lbl.pt.y);
            y = clamp(y, 0, chartHeight);
            if (lastRightY != null && Math.abs(y - lastRightY) < 16) y += 16 * (idx + 1);
            lastRightY = y;
            const nearRight = x > chartWidth - 12;
            const tx = nearRight ? x - 6 : x + 6;
            const anchor = nearRight ? 'end' : 'start';
            elements.push(
              <circle key={`right-dot-${idx}`} cx={x} cy={y} r={3} fill={lbl.color} />,
              <text key={`right-${idx}`} x={tx} y={y} textAnchor={anchor} alignmentBaseline="middle" fontSize={baseFontSize} fill={lbl.color}>
                {lbl.text}
              </text>
            );
          });

          // Crossover label: value only + small marker
          if (showTrainingSeries && intersection) {
            const xi = xScalePx(intersection.x);
            let yi = yScalePx(intersection.y);
            yi = clamp(yi, 0, chartHeight);
            elements.push(
              <g key="crossover">
                <circle cx={xi} cy={yi} r={3} fill="#555" />
                {/* Place label above-left of the point to avoid line overlap */}
                <text x={xi - 6} y={yi - 10} textAnchor="end" fontSize={baseFontSize} fill="#555">
                  {formatPowerOfTenNode(intersection.y, {
                  ...labelValueOptions,
                  mantissaFontSize: tensFontSize,
                  exponentFontSize: superscriptFontSize,
                  suffix: ' FLOP'
                  })}
                </text>
              </g>
            );
          }

          return <g>{elements}</g>;
        }}
      />
    </div>
  );
});

CombinedComputeChart.displayName = 'CombinedComputeChart';

export default CombinedComputeChart;


