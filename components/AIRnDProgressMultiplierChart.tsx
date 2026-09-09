'use client';

import { memo, useMemo, useCallback, useRef } from 'react';
import { CustomLineChart } from './CustomLineChart';
import type { ChartDataPoint } from '@/app/types';
import type { DataPoint } from './CustomLineChart';
import { formatAsPowerOfTenText } from '@/utils/formatting';
import { CHART_LAYOUT } from '@/constants/chartLayout';
import type { MilestoneMap } from '@/types/milestones';
import { createScale, calculateDynamicXDomain } from '@/utils/chartUtils';
import { AI_R_D_PROGRESS_MULTIPLIER_EXPLANATION } from '@/constants/chartExplanations';
import { WithChartTooltip } from './ChartTitleTooltip';
import { ChartMilestoneMarkers } from './ChartMilestoneMarkers';

const MINIMUM_VISIBLE_YEAR = 2019;

export interface AIRnDProgressMultiplierChartProps {
  chartData: ChartDataPoint[];
  tooltip: (point: DataPoint) => React.ReactNode;
  milestones: MilestoneMap | null;
  displayEndYear: number;
  verticalReferenceLine?: {
    x: number;
    stroke: string;
    strokeDasharray?: string;
    strokeWidth?: number;
    label?: string;
    strokeOpacity?: number;
  };
  className?: string;
  width?: number;
  height?: number;
  sampleTrajectories?: ChartDataPoint[][];
  isLoading?: boolean;
}

const milestoneSequence: Array<{ key: string; label: string }> = [
  { key: 'AC', label: 'AC' },
  { key: 'SAR-level-experiment-selection-skill', label: 'SAR' },
  { key: 'SIAR-level-experiment-selection-skill', label: 'SIAR' },
  // { key: 'STRAT-AI', label: 'STRAT-AI' },
  { key: 'TED-AI', label: 'TED-AI' },
  { key: 'ASI', label: 'ASI' },
];

export const AIRnDProgressMultiplierChart = memo(({
  chartData,
  tooltip,
  milestones,
  displayEndYear,
  verticalReferenceLine,
  className,
  width,
  height = CHART_LAYOUT.primary.height,
  sampleTrajectories = [],
  isLoading = false,
}: AIRnDProgressMultiplierChartProps) => {
  const svgRef = useRef<SVGSVGElement>(null);

  const seriesPoints = useMemo(() => {
    return chartData
      .filter(point =>
        typeof point.year === 'number' &&
        Number.isFinite(point.year) &&
        typeof point.aiSwProgressMultRefPresentDay === 'number' &&
        Number.isFinite(point.aiSwProgressMultRefPresentDay as number)
      )
      .map(point => ({
        x: point.year as number,
        y: point.aiSwProgressMultRefPresentDay as number,
      }))
      .sort((a, b) => a.x - b.x);
  }, [chartData]);

  const milestoneDomain = useMemo<[number, number] | undefined>(() => {
    if (!milestones || seriesPoints.length === 0) {
      return undefined;
    }

    // Find all achieved milestones (those with valid time values) from the sequence
    const achievedTimes = milestoneSequence
      .map(({ key }) => milestones[key]?.time)
      .filter((time): time is number => typeof time === 'number' && Number.isFinite(time));

    // Need at least 2 milestones (including AC) to create a meaningful domain
    if (achievedTimes.length < 2) {
      return undefined;
    }

    const start = Math.min(...achievedTimes);
    const end = Math.max(...achievedTimes);

    // Find the first data point >= end to ensure we include milestone values in y-domain
    const nextDataPoint = seriesPoints.find(p => p.x >= end);
    const effectiveEnd = nextDataPoint ? nextDataPoint.x : end;

    const range = effectiveEnd - start;
    const padding = range * 0.04;
    const clampedStart = Math.max(start - padding, MINIMUM_VISIBLE_YEAR);
    const clampedEnd = effectiveEnd + padding;

    return [clampedStart, clampedEnd];
  }, [milestones, seriesPoints]);

  const xDomain = useMemo<[number, number]>(() => {
    // Use milestone domain if available (zoomed-in view)
    if (milestoneDomain) {
      return milestoneDomain;
    }

    // Fallback to data-based domain
    const years = seriesPoints.map(point => point.x);
    if (years.length === 0) {
      return [2020, displayEndYear];
    }

    const intersectionX = verticalReferenceLine?.x ?? null;
    return calculateDynamicXDomain(years, {
      intersectionPoints: [intersectionX],
      minPadding: 0.1,
      maxPadding: 0.15,
      minRange: 3,
      maxDomain: displayEndYear,
    });
  }, [seriesPoints, milestoneDomain, verticalReferenceLine, displayEndYear]);

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

  const xTickFormatter = useCallback((value: number) => Math.round(value).toString(), []);

  const milestoneMarkers = useMemo(() => {
    if (!milestones) {
      return [] as Array<{ x: number; label: string; value: number }>;
    }

    return milestoneSequence
      .map(({ key, label }) => {
        const rawTime = milestones[key]?.time;
        const year = typeof rawTime === 'number' && Number.isFinite(rawTime) ? rawTime : null;
        const value = milestones[key]?.progress_multiplier;
        return {
          x: year ?? NaN,
          label,
          value,
        };
      })
      .filter(marker => Number.isFinite(marker.x) && marker.value != null && Number.isFinite(marker.value) && (marker.value as number) > 0)
      .map(marker => ({ x: marker.x as number, label: marker.label, value: marker.value as number }));
  }, [milestones]);

  const yDomain = useMemo<[number, number]>(() => {
    if (seriesPoints.length === 0) {
      return [0.1, 10];
    }

    const [domainStart, domainEnd] = xDomain;

    const inDomain = seriesPoints.filter(point => point.x >= domainStart && point.x <= domainEnd && Number.isFinite(point.y) && point.y > 0);

    // Include milestone values in the y-domain calculation
    const milestoneValues = milestoneMarkers
      .filter(m => m.x >= domainStart && m.x <= domainEnd)
      .map(m => m.value);

    const allValues = [...inDomain.map(point => point.y), ...milestoneValues];

    if (allValues.length === 0) {
      return [0.1, 10];
    }

    const minValue = Math.min(...allValues);
    const maxValue = Math.max(...allValues);

    if (!Number.isFinite(minValue) || !Number.isFinite(maxValue) || minValue <= 0) {
      return [0.1, 10];
    }

    const lower = Math.max(minValue * 0.7, Number.MIN_VALUE);
    let upper = Math.max(maxValue * 2, lower * 1.1);

    if (!Number.isFinite(upper) || upper <= lower) {
      upper = lower * 10;
    }

    return [lower, upper];
  }, [seriesPoints, xDomain, milestoneMarkers]);

  // Map each milestone to its closest data point year
  const milestoneToClosestYear = useMemo(() => {
    const map = new Map<string, number>();

    milestoneMarkers.forEach(marker => {
      // Find the data point year that is closest to this milestone
      let closestYear: number | null = null;
      let minDiff = Infinity;

      chartData.forEach(point => {
        if (typeof point.year === 'number' && Number.isFinite(point.year)) {
          const diff = Math.abs(point.year - marker.x);
          if (diff < minDiff) {
            minDiff = diff;
            closestYear = point.year;
          }
        }
      });

      if (closestYear !== null) {
        map.set(marker.label, closestYear);
      }
    });

    return map;
  }, [milestoneMarkers, chartData]);

  const findMilestoneLabels = useCallback((year: number): string[] => {
    const labels: string[] = [];

    // Check if any milestone has this year as its closest data point
    milestoneToClosestYear.forEach((closestYear, label) => {
      if (Math.abs(closestYear - year) < 0.001) {
        labels.push(label);
      }
    });

    return labels;
  }, [milestoneToClosestYear]);

  // Generate keys for sample trajectories
  const sampleKeys = useMemo(() => {
    return sampleTrajectories.map((_, index) => `sample_progress_${index}`);
  }, [sampleTrajectories]);

  const mergedData = useMemo(() => {
    const seriesLookup = seriesPoints.length > 0 
      ? new Map(seriesPoints.map(point => [point.x, point.y]))
      : null;

    // Create merged data with all sample keys initialized
    return chartData
      .filter(point => typeof point.year === 'number' && Number.isFinite(point.year))
      .map(point => {
        const labels = findMilestoneLabels(point.year as number);
        const newPoint = {
          ...point,
          x: point.year,
          aiSwProgressMultRefPresentDay: seriesLookup?.get(point.year as number) ?? point.aiSwProgressMultRefPresentDay,
          milestoneLabel: labels.length > 0 ? labels.join(' & ') : undefined,
          milestoneLabels: labels.length > 0 ? labels : undefined,
        } as DataPoint & { milestoneLabels?: string[] };

        // Initialize all sample keys and add their values if available
        sampleTrajectories.forEach((trajectory, sampleIndex) => {
          const key = `sample_progress_${sampleIndex}`;
          const samplePoint = trajectory.find(p => p.year === point.year);
          // Always set the key, use null if no matching point found
          (newPoint as Record<string, unknown>)[key] = samplePoint?.aiSwProgressMultRefPresentDay ?? null;
        });

        return newPoint as DataPoint;
      });
  }, [chartData, seriesPoints, findMilestoneLabels, sampleTrajectories]);

  // Filter merged data to only include points within the xDomain
  // This ensures the visual content matches the x-axis labels
  const filteredMergedData = useMemo(() => {
    const [domainStart, domainEnd] = xDomain;
    // Small margin to ensure we don't cut off at the boundary
    const margin = (domainEnd - domainStart) * 0.01;
    return mergedData.filter(point => {
      if (typeof point.x !== 'number') return false;
      return point.x >= domainStart - margin && point.x <= domainEnd + margin;
    });
  }, [mergedData, xDomain]);

  const customElements = useMemo(() => {
    if (milestoneMarkers.length === 0) {
      return undefined;
    }

    function MilestoneOverlay(context: { chartWidth: number; chartHeight: number, xScale: (value: number) => number, yScale: (value: number) => number }) {
      const { chartWidth, chartHeight, xScale, yScale } = context;

      const positionedMarkers = milestoneMarkers.map(marker => ({
        label: marker.label,
        xPos: xScale(marker.x),
        yPos: yScale(marker.value),
      }));

      return (
        <ChartMilestoneMarkers
          markers={positionedMarkers}
          chartWidth={chartWidth}
          chartHeight={chartHeight}
        />
      );
    }

    return MilestoneOverlay;
  }, [milestoneMarkers]);

  const lines = useMemo(() => {
    const result = [];

    // Add sample trajectory lines first (so they're behind)
    for (const key of sampleKeys) {
      result.push({
        dataKey: key,
        stroke: '#2A623D',
        strokeWidth: 1,
        strokeOpacity: 0.15,
      });
    }

    // Add main line on top
    result.push({
      dataKey: 'aiSwProgressMultRefPresentDay',
      stroke: '#2A623D',
      strokeWidth: 3,
      strokeOpacity: 1,
      name: 'AI Software R&D Uplift',
    });

    return result;
  }, [sampleKeys]);

  return (
    <div className={className ? `flex flex-col ${className}` : 'flex flex-col'} style={{ position: 'relative' }}>
      <div className="flex gap-2 items-center">
        <span className="text-[12px] font-semibold text-primary text-left font-system-mono">
          Takeoff Period: AI Software R&amp;D Uplift
        </span>
        <div className="flex-1 border-t border-gray-500/30" />
        <WithChartTooltip explanation={AI_R_D_PROGRESS_MULTIPLIER_EXPLANATION} className="!gap-0" isLoading={isLoading}><></></WithChartTooltip>
      </div>
      <CustomLineChart
        data={filteredMergedData}
        height={height}
        width={width}
        margin={{ top: 0, right: 10, left: 20, bottom: 35 }}
        svgRef={svgRef}
        xDomain={xDomain}
        xTicks={xTicks}
        xTickFormatter={xTickFormatter}
        yDomain={yDomain}
        yScale="log"
        yTickFormatter={(value) => formatAsPowerOfTenText(value, { suffix: 'x' })}
        lines={lines}
        tooltip={tooltip}
        customElements={customElements}
        showXAxis
        animateXDomain={false}
      />
    </div>
  );
});

AIRnDProgressMultiplierChart.displayName = 'AIRnDProgressMultiplierChart';


