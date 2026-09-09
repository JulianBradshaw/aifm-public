'use client';

import { useMemo } from 'react';
import type { ChartDataPoint, BenchmarkPoint } from '@/app/types';

export interface StaticHorizonChartProps {
  chartData: ChartDataPoint[];
  scHorizonMinutes: number;
  displayEndYear: number;
  width: number;
  height: number;
  title?: string;
  benchmarkData?: BenchmarkPoint[];
  showMilestoneIndicators?: boolean;
}

// Time values in minutes for Y-axis (logarithmic scale)
// Work time: 8 hours/day, 5 days/week, ~21.6 days/month, ~260 days/year
const WORK_DAY_MINUTES = 480;        // 8 hours
const WORK_WEEK_MINUTES = 5 * WORK_DAY_MINUTES;      // 5 work-days
const WORK_MONTH_MINUTES = 4 * WORK_WEEK_MINUTES;    // ~21.6 work-days
const WORK_YEAR_MINUTES = 52 * WORK_WEEK_MINUTES;    // ~260 work-days

const TIME_TICKS = [
  { minutes: 0.1, label: '6 sec' },
  { minutes: 1, label: '1 min' },
  { minutes: 15, label: '15 min' },
  { minutes: 60, label: '1 hr' },
  { minutes: WORK_DAY_MINUTES, label: '1 work-day' },
  { minutes: WORK_WEEK_MINUTES, label: '1 work-week' },
  { minutes: WORK_MONTH_MINUTES, label: '1 work-month' },
  { minutes: WORK_YEAR_MINUTES, label: '1 work-year' },
  { minutes: WORK_YEAR_MINUTES * 10, label: '10 work-years' },
  { minutes: WORK_YEAR_MINUTES * 100, label: '100 work-years' },
];

const COLORS = {
  background: '#fffff8',
  foreground: '#0D0D0D',
  graphLine: '#2A623D',
  gridLine: '#e0e0e0',
  axisLine: '#333333',
  scHorizonLine: '#666666',
};

const FONTS = {
  title: 'et-book, Georgia, serif',
  axis: 'Menlo, Consolas, monospace',
};

// Helper to get shape path for benchmark points
function getBenchmarkShape(label: string, cx: number, cy: number): { path: string; fill: string } {
  const lowerLabel = label.toLowerCase();

  if (lowerLabel.includes('claude')) {
    // Diamond
    return {
      path: `M${cx},${cy - 5} L${cx + 5},${cy} L${cx},${cy + 5} L${cx - 5},${cy} Z`,
      fill: '#dc2626',
    };
  } else if (lowerLabel.includes('gpt') || lowerLabel.includes('o1') || lowerLabel.includes('o3') || lowerLabel.includes('o4') || lowerLabel.includes('grok')) {
    // Square
    return {
      path: `M${cx - 4},${cy - 4} L${cx + 4},${cy - 4} L${cx + 4},${cy + 4} L${cx - 4},${cy + 4} Z`,
      fill: '#059669',
    };
  } else if (lowerLabel.includes('gemini')) {
    // Triangle
    return {
      path: `M${cx},${cy - 5} L${cx + 5},${cy + 5} L${cx - 5},${cy + 5} Z`,
      fill: '#2563eb',
    };
  } else if (lowerLabel.includes('deepseek')) {
    // Star (simplified to 5-point)
    const r1 = 5, r2 = 2.5;
    const points = [];
    for (let i = 0; i < 10; i++) {
      const r = i % 2 === 0 ? r1 : r2;
      const angle = (i * Math.PI) / 5 - Math.PI / 2;
      points.push(`${cx + r * Math.cos(angle)},${cy + r * Math.sin(angle)}`);
    }
    return {
      path: `M${points.join(' L')} Z`,
      fill: '#7c3aed',
    };
  } else {
    // Circle (approximated with many points)
    const r = 4;
    const points = [];
    for (let i = 0; i < 12; i++) {
      const angle = (i * Math.PI * 2) / 12;
      points.push(`${cx + r * Math.cos(angle)},${cy + r * Math.sin(angle)}`);
    }
    return {
      path: `M${points.join(' L')} Z`,
      fill: '#6b7280',
    };
  }
}

/**
 * Static horizon chart for PNG export with full axes
 */
export function StaticHorizonChart({
  chartData,
  scHorizonMinutes,
  displayEndYear,
  width,
  height,
  title = 'Coding Time Horizon',
  benchmarkData = [],
  showMilestoneIndicators = true,
}: StaticHorizonChartProps) {
  // Reduce top margin when no title is shown
  const hasTitle = title && title.length > 0;
  const margin = { top: hasTitle ? 50 : 20, right: 20, bottom: 50, left: 140 };
  const chartWidth = width - margin.left - margin.right;
  const chartHeight = height - margin.top - margin.bottom;

  // Filter valid data points
  const validData = useMemo(() => {
    return chartData.filter(
      (d) =>
        typeof d.year === 'number' &&
        Number.isFinite(d.year) &&
        typeof d.horizonLength === 'number' &&
        Number.isFinite(d.horizonLength) &&
        d.horizonLength > 0
    );
  }, [chartData]);

  // Calculate X domain - always start at 2022
  const xDomain = useMemo((): [number, number] => {
    return [2022, displayEndYear];
  }, [displayEndYear]);

  // Calculate Y domain (log scale) - cap at slightly above AC horizon
  const yDomain = useMemo((): [number, number] => {
    if (validData.length === 0) return [0.1, scHorizonMinutes * 1.2];
    const horizons = validData.map((d) => d.horizonLength);
    const minHorizon = Math.max(0.1, Math.min(...horizons) * 0.5);
    // Cap at slightly above AC horizon requirement
    const maxHorizon = scHorizonMinutes * 1.2;
    return [minHorizon, maxHorizon];
  }, [validData, scHorizonMinutes]);

  // Scale functions
  const xScale = (year: number): number => {
    return margin.left + ((year - xDomain[0]) / (xDomain[1] - xDomain[0])) * chartWidth;
  };

  const yScale = (minutes: number): number => {
    const logMin = Math.log(yDomain[0]);
    const logMax = Math.log(yDomain[1]);
    const logValue = Math.log(Math.max(minutes, yDomain[0]));
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
    return TIME_TICKS.filter(
      (t) => t.minutes >= yDomain[0] * 0.9 && t.minutes <= yDomain[1] * 1.1
    );
  }, [yDomain]);

  // Generate path for trajectory line, capping at scHorizonMinutes
  const trajectoryPath = useMemo(() => {
    if (validData.length === 0) return '';
    const sortedData = [...validData].sort((a, b) => a.year - b.year);

    // Track if we've exceeded the horizon threshold (stop drawing after first capped point)
    let exceededThreshold = false;
    const pathPoints: string[] = [];

    for (let i = 0; i < sortedData.length; i++) {
      if (exceededThreshold) break;

      const d = sortedData[i];
      const x = xScale(d.year);
      // Cap the horizon length at scHorizonMinutes so green line aligns with gray dashed line
      const clampedHorizon = Math.min(d.horizonLength, scHorizonMinutes);
      const y = yScale(clampedHorizon);

      pathPoints.push(`${i === 0 ? 'M' : 'L'} ${x} ${y}`);

      // If this point was capped, mark that we've exceeded and include this point but stop after
      if (d.horizonLength >= scHorizonMinutes) {
        exceededThreshold = true;
      }
    }

    return pathPoints.join(' ');
  }, [validData, xScale, yScale, scHorizonMinutes]);

  // SC Horizon line Y position
  const scHorizonY = yScale(scHorizonMinutes);
  const scHorizonVisible =
    scHorizonMinutes >= yDomain[0] && scHorizonMinutes <= yDomain[1];

  // Filter visible benchmarks
  const visibleBenchmarks = useMemo(() => {
    const [xMin, xMax] = xDomain;
    const [yMin, yMax] = yDomain;

    return benchmarkData.filter(benchmark => {
      const { year, horizonLength } = benchmark;
      return (
        Number.isFinite(year) &&
        Number.isFinite(horizonLength) &&
        horizonLength > 0 &&
        year >= xMin &&
        year <= xMax &&
        horizonLength >= yMin &&
        horizonLength <= yMax
      );
    });
  }, [benchmarkData, xDomain, yDomain]);

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
        <clipPath id="chart-area-clip">
          <rect x={margin.left} y={margin.top} width={chartWidth} height={chartHeight} />
        </clipPath>
      </defs>

      {/* Title */}
      {hasTitle && (
        <text
          x={width / 2}
          y={30}
          textAnchor="middle"
          fontSize="22"
          fontWeight="bold"
          fontFamily={FONTS.title}
          fill={COLORS.foreground}
        >
          {title}
        </text>
      )}

      {/* Grid lines - horizontal */}
      {yTicks.map((tick) => (
        <line
          key={`hgrid-${tick.minutes}`}
          x1={margin.left}
          y1={yScale(tick.minutes)}
          x2={width - margin.right}
          y2={yScale(tick.minutes)}
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

      {/* SC Horizon reference line */}
      {showMilestoneIndicators && scHorizonVisible && (
        <g>
          <line
            x1={margin.left}
            y1={scHorizonY}
            x2={width - margin.right}
            y2={scHorizonY}
            stroke={COLORS.scHorizonLine}
            strokeWidth="1.5"
            strokeDasharray="6,4"
          />
          <text
            x={width - margin.right - 5}
            y={scHorizonY - 5}
            textAnchor="end"
            fontSize="14"
            fontFamily={FONTS.axis}
            fill={COLORS.scHorizonLine}
          >
            Automated Coder Horizon (full coding automation)
          </text>
        </g>
      )}

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
        const y = yScale(tick.minutes);
        return (
          <g key={`ytick-${tick.minutes}`}>
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
              fontSize="15"
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
        clipPath="url(#chart-area-clip)"
      />

      {/* Benchmark points */}
      {visibleBenchmarks.map((benchmark, index) => {
        const cx = xScale(benchmark.year);
        const cy = yScale(benchmark.horizonLength);
        const { path, fill } = getBenchmarkShape(benchmark.label, cx, cy);
        return (
          <path
            key={`benchmark-${index}`}
            d={path}
            fill={fill}
            stroke="#ffffff"
            strokeWidth="1.5"
          />
        );
      })}
    </svg>
  );
}
