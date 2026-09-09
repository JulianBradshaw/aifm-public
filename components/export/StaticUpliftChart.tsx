'use client';

import { useMemo } from 'react';
import type { ChartDataPoint } from '@/app/types';
import type { MilestoneMap } from '@/types/milestones';

export interface StaticUpliftChartProps {
  chartData: ChartDataPoint[];
  milestones: MilestoneMap | null;
  displayEndYear: number;
  width: number;
  height: number;
  title?: string;
  showMilestoneIndicators?: boolean;
  presentDay?: number;
}

// Multiplier ticks for Y-axis (logarithmic scale)
const MULTIPLIER_TICKS = [
  { value: 1, label: '1x' },
  { value: 10, label: '10x' },
  { value: 100, label: '100x' },
  { value: 1000, label: '1,000x' },
  { value: 10000, label: '10,000x' },
  { value: 100000, label: '100,000x' },
  { value: 1000000, label: '1M x' },
];

const COLORS = {
  background: '#fffff8',
  foreground: '#0D0D0D',
  graphLine: '#2A623D',
  gridLine: '#e0e0e0',
  axisLine: '#333333',
  milestoneMarker: '#666666',
};

const FONTS = {
  title: 'et-book, Georgia, serif',
  axis: 'Menlo, Consolas, monospace',
};

// Milestone display names (map full keys to display labels)
const MILESTONE_LABELS: Record<string, string> = {
  'AC': 'AC',
  'SC': 'SC',
  'SAR-level-experiment-selection-skill': 'SAR',
  'SIAR': 'SIAR',
  'TED-AI': 'TED-AI',
  'ASI': 'ASI',
};

/**
 * Static uplift chart for PNG export with full axes
 */
export function StaticUpliftChart({
  chartData,
  milestones,
  displayEndYear,
  width,
  height,
  title = 'AI Software R&D Uplift',
  showMilestoneIndicators = true,
  presentDay,
}: StaticUpliftChartProps) {
  // Reduce top margin when no title is shown
  const hasTitle = title && title.length > 0;
  const margin = { top: hasTitle ? 60 : 20, right: 20, bottom: 50, left: 90 };
  const chartWidth = width - margin.left - margin.right;
  const chartHeight = height - margin.top - margin.bottom;

  // Filter valid data points
  const validData = useMemo(() => {
    return chartData.filter(
      (d) =>
        typeof d.year === 'number' &&
        Number.isFinite(d.year) &&
        typeof d.aiSwProgressMultRefPresentDay === 'number' &&
        Number.isFinite(d.aiSwProgressMultRefPresentDay) &&
        (d.aiSwProgressMultRefPresentDay as number) > 0
    );
  }, [chartData]);

  // Calculate X domain - start at present day (default to 2026 if not provided)
  const xDomain = useMemo((): [number, number] => {
    const startYear = presentDay ?? 2026;
    return [startYear, displayEndYear];
  }, [presentDay, displayEndYear]);

  // Calculate Y domain (log scale)
  const yDomain = useMemo((): [number, number] => {
    if (validData.length === 0) return [1, 1000];
    const multipliers = validData.map((d) => d.aiSwProgressMultRefPresentDay as number);
    const minMult = Math.max(0.5, Math.min(...multipliers) * 0.5);
    const maxMult = Math.max(...multipliers) * 2;
    return [minMult, maxMult];
  }, [validData]);

  // Scale functions
  const xScale = (year: number): number => {
    return margin.left + ((year - xDomain[0]) / (xDomain[1] - xDomain[0])) * chartWidth;
  };

  const yScale = (value: number): number => {
    const logMin = Math.log(yDomain[0]);
    const logMax = Math.log(yDomain[1]);
    const logValue = Math.log(Math.max(value, yDomain[0]));
    const normalized = (logValue - logMin) / (logMax - logMin);
    return margin.top + chartHeight - normalized * chartHeight;
  };

  // Generate X-axis ticks (years)
  const xTicks = useMemo(() => {
    const [start, end] = xDomain;
    const years: number[] = [];
    const range = end - start;
    const step = range <= 10 ? 1 : range <= 20 ? 2 : 5;
    for (let year = Math.ceil(start); year <= end; year += step) {
      years.push(year);
    }
    return years;
  }, [xDomain]);

  // Filter Y-axis ticks to those within domain
  const yTicks = useMemo(() => {
    return MULTIPLIER_TICKS.filter(
      (t) => t.value >= yDomain[0] * 0.5 && t.value <= yDomain[1] * 2
    );
  }, [yDomain]);

  // Generate path for trajectory line
  const trajectoryPath = useMemo(() => {
    if (validData.length === 0) return '';
    const sortedData = [...validData].sort((a, b) => a.year - b.year);
    return sortedData
      .map((d, i) => {
        const x = xScale(d.year);
        const y = yScale(d.aiSwProgressMultRefPresentDay as number);
        return `${i === 0 ? 'M' : 'L'} ${x} ${y}`;
      })
      .join(' ');
  }, [validData, xScale, yScale]);

  // Get milestone markers
  const milestoneMarkers = useMemo(() => {
    if (!milestones) return [];
    const markers: Array<{ key: string; year: number; label: string }> = [];

    // Key milestones to show (using full keys from data)
    const keysToShow = ['AC', 'TED-AI', 'ASI'];

    for (const key of keysToShow) {
      const milestone = milestones[key];
      if (milestone && milestone.time != null && Number.isFinite(milestone.time)) {
        const year = milestone.time as number;
        if (year >= xDomain[0] && year <= xDomain[1]) {
          markers.push({
            key,
            year,
            label: MILESTONE_LABELS[key] || key,
          });
        }
      }
    }

    return markers;
  }, [milestones, xDomain]);

  return (
    <svg
      width={width}
      height={height}
      viewBox={`0 0 ${width} ${height}`}
      style={{ backgroundColor: COLORS.background }}
    >
      {/* Background */}
      <rect x="0" y="0" width={width} height={height} fill={COLORS.background} />

      {/* Clip path for chart area */}
      <defs>
        <clipPath id="uplift-chart-area-clip">
          <rect x={margin.left} y={margin.top} width={chartWidth} height={chartHeight} />
        </clipPath>
      </defs>

      {/* Title */}
      {hasTitle && (
        <text
          x={width / 2}
          y={30}
          textAnchor="middle"
          fontSize="20"
          fontWeight="bold"
          fontFamily={FONTS.axis}
          fill={COLORS.foreground}
        >
          {title}
        </text>
      )}

      {/* Grid lines - horizontal */}
      {yTicks.map((tick) => (
        <line
          key={`hgrid-${tick.value}`}
          x1={margin.left}
          y1={yScale(tick.value)}
          x2={width - margin.right}
          y2={yScale(tick.value)}
          stroke={COLORS.gridLine}
          strokeWidth="1"
        />
      ))}

      {/* Grid lines - vertical */}
      {xTicks.map((year) => (
        <line
          key={`vgrid-${year}`}
          x1={xScale(year)}
          y1={margin.top}
          x2={xScale(year)}
          y2={height - margin.bottom}
          stroke={COLORS.gridLine}
          strokeWidth="1"
        />
      ))}

      {/* Milestone markers */}
      {showMilestoneIndicators && milestoneMarkers.map((marker) => {
        const x = xScale(marker.year);
        return (
          <g key={`milestone-${marker.key}`}>
            <line
              x1={x}
              y1={margin.top}
              x2={x}
              y2={height - margin.bottom}
              stroke={COLORS.milestoneMarker}
              strokeWidth="1.5"
              strokeDasharray="4,3"
            />
            <text
              x={x}
              y={margin.top - 5}
              textAnchor="middle"
              fontSize="14"
              fontWeight="bold"
              fontFamily={FONTS.axis}
              fill={COLORS.milestoneMarker}
            >
              {marker.label}
            </text>
          </g>
        );
      })}

      {/* Y-axis */}
      <line
        x1={margin.left}
        y1={margin.top}
        x2={margin.left}
        y2={height - margin.bottom}
        stroke={COLORS.axisLine}
        strokeWidth="1"
      />

      {/* Y-axis ticks and labels */}
      {yTicks.map((tick) => {
        const y = yScale(tick.value);
        return (
          <g key={`ytick-${tick.value}`}>
            <line
              x1={margin.left - 5}
              y1={y}
              x2={margin.left}
              y2={y}
              stroke={COLORS.axisLine}
              strokeWidth="1"
            />
            <text
              x={margin.left - 8}
              y={y + 4}
              textAnchor="end"
              fontSize="14"
              fontFamily={FONTS.axis}
              fill={COLORS.foreground}
            >
              {tick.label}
            </text>
          </g>
        );
      })}

      {/* X-axis */}
      <line
        x1={margin.left}
        y1={height - margin.bottom}
        x2={width - margin.right}
        y2={height - margin.bottom}
        stroke={COLORS.axisLine}
        strokeWidth="1"
      />

      {/* X-axis ticks and labels */}
      {xTicks.map((year) => {
        const x = xScale(year);
        return (
          <g key={`xtick-${year}`}>
            <line
              x1={x}
              y1={height - margin.bottom}
              x2={x}
              y2={height - margin.bottom + 5}
              stroke={COLORS.axisLine}
              strokeWidth="1"
            />
            <text
              x={x}
              y={height - margin.bottom + 20}
              textAnchor="middle"
              fontSize="15"
              fontFamily={FONTS.axis}
              fill={COLORS.foreground}
            >
              {year}
            </text>
          </g>
        );
      })}

      {/* Trajectory line - clipped to chart area */}
      <path
        d={trajectoryPath}
        fill="none"
        stroke={COLORS.graphLine}
        strokeWidth="2.5"
        strokeLinecap="round"
        strokeLinejoin="round"
        clipPath="url(#uplift-chart-area-clip)"
      />
    </svg>
  );
}
