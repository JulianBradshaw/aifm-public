'use client';

import { useState, useMemo, useCallback, useRef } from 'react';

export interface TimeSeriesDataPoint {
  time: number;
  L_HUMAN: number;
  inference_H100e: number;
  experiment_H100e: number;
  training_H100e: number;
}

interface InputTimeSeriesChartProps {
  data: TimeSeriesDataPoint[];
  timeRange?: [number, number];
}

// Configuration for each time series
const SERIES_CONFIG = [
  {
    key: 'training_H100e' as const,
    label: 'Training Compute',
    unit: 'H100e',
    color: '#6366f1',
    description: 'H100-equivalents used in the frontier training run',
  },
  {
    key: 'experiment_H100e' as const,
    label: 'Experiment Compute',
    unit: 'H100e',
    color: '#22c55e',
    description: 'Compute available for running experiments',
  },
  {
    key: 'inference_H100e' as const,
    label: 'Inference Compute',
    unit: 'H100e',
    color: '#f59e0b',
    description: 'Compute available for inference',
  },
  {
    key: 'L_HUMAN' as const,
    label: 'Human Labor',
    unit: 'FTE',
    color: '#ec4899',
    description: 'Full-time equivalent human researchers',
  },
] as const;

type SeriesKey = typeof SERIES_CONFIG[number]['key'];

interface SingleSeriesChartProps {
  data: TimeSeriesDataPoint[];
  seriesKey: SeriesKey;
  label: string;
  unit: string;
  color: string;
  timeRange?: [number, number];
}

function formatValue(value: number): string {
  if (!Number.isFinite(value) || value <= 0) return '-';

  // For H100e and FTE, format with appropriate suffixes
  if (value >= 1e12) return `${(value / 1e12).toFixed(2)}T`;
  if (value >= 1e9) return `${(value / 1e9).toFixed(2)}B`;
  if (value >= 1e6) return `${(value / 1e6).toFixed(2)}M`;
  if (value >= 1e3) return `${(value / 1e3).toFixed(2)}K`;
  if (value >= 1) return value.toFixed(1);
  return value.toFixed(3);
}

function formatAxisValue(value: number, isLogScale: boolean): string {
  if (isLogScale) {
    const log10 = Math.log10(value);
    if (Number.isInteger(log10)) {
      return `10^${log10.toFixed(0)}`;
    }
    return value.toExponential(0);
  }
  if (value >= 1e9) return `${(value / 1e9).toFixed(0)}B`;
  if (value >= 1e6) return `${(value / 1e6).toFixed(0)}M`;
  if (value >= 1e3) return `${(value / 1e3).toFixed(0)}K`;
  return value.toFixed(0);
}

function SingleSeriesChart({
  data,
  seriesKey,
  label,
  unit,
  color,
  timeRange,
}: SingleSeriesChartProps) {
  const [hoveredPoint, setHoveredPoint] = useState<TimeSeriesDataPoint | null>(null);
  const svgRef = useRef<SVGSVGElement>(null);

  const { filteredData, xDomain, yDomain, useLogScale } = useMemo(() => {
    // Filter data by time range if provided
    const filtered = timeRange
      ? data.filter(d => d.time >= timeRange[0] && d.time <= timeRange[1])
      : data;

    if (filtered.length === 0) {
      return {
        filteredData: [],
        xDomain: [2017, 2050] as [number, number],
        yDomain: [1, 100] as [number, number],
        useLogScale: false,
      };
    }

    const times = filtered.map(d => d.time);
    const xMin = Math.min(...times);
    const xMax = Math.max(...times);

    const values = filtered.map(d => d[seriesKey]).filter(v => v > 0);
    if (values.length === 0) {
      return {
        filteredData: filtered,
        xDomain: [xMin, xMax] as [number, number],
        yDomain: [1, 100] as [number, number],
        useLogScale: false,
      };
    }

    const yMin = Math.min(...values);
    const yMax = Math.max(...values);

    // Use log scale if the range spans more than 2 orders of magnitude.
    // training_H100e is raw H100e (exponential), so it log-scales like the other compute series.
    const useLog = yMax / yMin > 100;

    let yDomainMin: number;
    let yDomainMax: number;

    if (useLog) {
      const logMin = Math.floor(Math.log10(yMin));
      const logMax = Math.ceil(Math.log10(yMax));
      yDomainMin = Math.pow(10, logMin);
      yDomainMax = Math.pow(10, logMax);
    } else {
      // Add some padding for linear scale
      const padding = (yMax - yMin) * 0.1;
      yDomainMin = Math.max(0, yMin - padding);
      yDomainMax = yMax + padding;
    }

    return {
      filteredData: filtered,
      xDomain: [xMin, xMax] as [number, number],
      yDomain: [yDomainMin, yDomainMax] as [number, number],
      useLogScale: useLog,
    };
  }, [data, timeRange, seriesKey]);

  const padding = { top: 10, right: 15, bottom: 25, left: 55 };
  const height = 120;
  const chartHeight = height - padding.top - padding.bottom;

  // We'll use viewBox for responsive width
  const viewBoxWidth = 400;
  const chartWidth = viewBoxWidth - padding.left - padding.right;

  const scaleX = useCallback((x: number) => {
    return ((x - xDomain[0]) / (xDomain[1] - xDomain[0])) * chartWidth;
  }, [xDomain, chartWidth]);

  const scaleY = useCallback((y: number) => {
    if (y <= 0) return chartHeight;
    if (useLogScale) {
      const logY = Math.log10(y);
      const logMin = Math.log10(yDomain[0]);
      const logMax = Math.log10(yDomain[1]);
      const normalized = (logY - logMin) / (logMax - logMin);
      return chartHeight - normalized * chartHeight;
    }
    const normalized = (y - yDomain[0]) / (yDomain[1] - yDomain[0]);
    return chartHeight - normalized * chartHeight;
  }, [yDomain, chartHeight, useLogScale]);

  // Generate path
  const path = useMemo(() => {
    const points = filteredData
      .map(d => {
        const val = d[seriesKey];
        if (val <= 0) return null;
        return { x: scaleX(d.time), y: scaleY(val) };
      })
      .filter((p): p is { x: number; y: number } => p !== null);

    if (points.length === 0) return '';

    return points
      .map((p, i) => (i === 0 ? `M ${p.x} ${p.y}` : `L ${p.x} ${p.y}`))
      .join(' ');
  }, [filteredData, seriesKey, scaleX, scaleY]);

  // Generate y-axis ticks
  const yTicks = useMemo(() => {
    const ticks: number[] = [];
    if (useLogScale) {
      const logMin = Math.log10(yDomain[0]);
      const logMax = Math.log10(yDomain[1]);
      for (let i = logMin; i <= logMax; i += Math.max(1, Math.floor((logMax - logMin) / 4))) {
        ticks.push(Math.pow(10, i));
      }
    } else {
      const range = yDomain[1] - yDomain[0];
      const step = range / 3;
      for (let i = 0; i <= 3; i++) {
        ticks.push(yDomain[0] + step * i);
      }
    }
    return ticks;
  }, [yDomain, useLogScale]);

  // Generate x-axis ticks
  const xTicks = useMemo(() => {
    const range = xDomain[1] - xDomain[0];
    const step = range <= 20 ? 5 : range <= 50 ? 10 : 25;
    const ticks: number[] = [];
    const start = Math.ceil(xDomain[0] / step) * step;
    for (let t = start; t <= xDomain[1]; t += step) {
      ticks.push(t);
    }
    return ticks;
  }, [xDomain]);

  const handleMouseMove = useCallback((e: React.MouseEvent<SVGSVGElement>) => {
    if (!svgRef.current || filteredData.length === 0) return;

    const svg = svgRef.current;
    const ctm = svg.getScreenCTM();
    if (!ctm) return;

    // Transform client coordinates to SVG coordinates using the CTM
    // This properly handles viewBox scaling and preserveAspectRatio centering
    const svgPoint = svg.createSVGPoint();
    svgPoint.x = e.clientX;
    svgPoint.y = e.clientY;
    const transformed = svgPoint.matrixTransform(ctm.inverse());

    // Get the x position in chart coordinates (accounting for padding)
    const chartX = transformed.x - padding.left;

    // Find closest point
    let closest: TimeSeriesDataPoint | null = null;
    let minDist = Infinity;

    for (const pt of filteredData) {
      const px = scaleX(pt.time);
      const dist = Math.abs(chartX - px);
      if (dist < minDist) {
        minDist = dist;
        closest = pt;
      }
    }

    setHoveredPoint(closest);
  }, [filteredData, scaleX, padding.left]);

  const handleMouseLeave = useCallback(() => {
    setHoveredPoint(null);
  }, []);

  if (filteredData.length === 0) {
    return (
      <div className="bg-gray-50 rounded p-4 text-center text-sm text-gray-500">
        No data available for {label}
      </div>
    );
  }

  return (
    <div className="relative">
      {/* Header */}
      <div className="flex items-baseline justify-between mb-1">
        <div className="flex items-baseline gap-2">
          <span className="text-sm font-medium text-gray-700">{label}</span>
          <span className="text-xs text-gray-400">({unit})</span>
        </div>
        {hoveredPoint && (
          <div className="text-xs text-gray-600 font-mono">
            {hoveredPoint.time.toFixed(1)}: {formatValue(hoveredPoint[seriesKey])} {unit}
          </div>
        )}
      </div>

      {/* Chart */}
      <svg
        ref={svgRef}
        width="100%"
        height={height}
        viewBox={`0 0 ${viewBoxWidth} ${height}`}
        preserveAspectRatio="xMidYMid meet"
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
        className="bg-gray-50 rounded"
      >
        <g transform={`translate(${padding.left}, ${padding.top})`}>
          {/* Grid lines */}
          {yTicks.map((tick, i) => (
            <line
              key={`grid-y-${i}`}
              x1={0}
              y1={scaleY(tick)}
              x2={chartWidth}
              y2={scaleY(tick)}
              stroke="#e5e7eb"
              strokeWidth={0.5}
            />
          ))}

          {/* Line */}
          <path
            d={path}
            fill="none"
            stroke={color}
            strokeWidth={2}
            strokeOpacity={0.9}
          />

          {/* Hover indicator */}
          {hoveredPoint && (
            <>
              <line
                x1={scaleX(hoveredPoint.time)}
                y1={0}
                x2={scaleX(hoveredPoint.time)}
                y2={chartHeight}
                stroke="#9ca3af"
                strokeWidth={1}
                strokeDasharray="3 3"
              />
              <circle
                cx={scaleX(hoveredPoint.time)}
                cy={scaleY(hoveredPoint[seriesKey])}
                r={4}
                fill={color}
                stroke="white"
                strokeWidth={2}
              />
            </>
          )}

          {/* Y-axis labels */}
          {yTicks.map((tick, i) => (
            <text
              key={`y-label-${i}`}
              x={-5}
              y={scaleY(tick)}
              textAnchor="end"
              alignmentBaseline="middle"
              fontSize={9}
              fill="#9ca3af"
            >
              {formatAxisValue(tick, useLogScale)}
            </text>
          ))}

          {/* X-axis */}
          <g transform={`translate(0, ${chartHeight})`}>
            {xTicks.map((tick, i) => (
              <text
                key={`x-label-${i}`}
                x={scaleX(tick)}
                y={14}
                textAnchor="middle"
                fontSize={9}
                fill="#9ca3af"
              >
                {tick}
              </text>
            ))}
          </g>
        </g>
      </svg>
    </div>
  );
}

export function InputTimeSeriesChart({
  data,
  timeRange,
}: InputTimeSeriesChartProps) {
  return (
    <div className="space-y-4">
      {SERIES_CONFIG.map(series => (
        <SingleSeriesChart
          key={series.key}
          data={data}
          seriesKey={series.key}
          label={series.label}
          unit={series.unit}
          color={series.color}
          timeRange={timeRange}
        />
      ))}
    </div>
  );
}

export default InputTimeSeriesChart;
