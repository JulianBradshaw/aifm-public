'use client';

import { useState, useRef, useCallback, ReactNode, useEffect, useMemo, useId } from 'react';
import { createScale, generateLinePath, calculateTicks, thinTicksToFitLabels, ChartMargin, clamp } from '@/utils/chartUtils';
import { useChartSync } from './ChartSyncContext';

const DEFAULT_X_DOMAIN_ANIMATION_DURATION_MS = 450;

export interface DataPoint {
  x: number;
  y?: number | null;
  [key: string]: number | null | undefined | string;
}

export interface CustomElementsContext {
  chartWidth: number;
  chartHeight: number;
  xDomain: [number, number];
  yDomain: [number, number];
  xScale: (value: number) => number;
  yScale: (value: number) => number;
}

export interface PdfLineChartProps {
  data: DataPoint[];
  width?: number;
  height: number;
  margin?: ChartMargin;
  /** Optional ref to expose the SVG element for export */
  svgRef?: React.RefObject<SVGSVGElement | null>;

  // X axis config
  xDomain: [number, number];
  showXAxis?: boolean;
  xTicks?: number[];
  xTickStepSize?: number;
  xTickCount?: number;
  xTickFormatter?: (value: number) => string;
  /** Shorter label variant used when full labels don't fit (e.g. narrow mobile charts) */
  xTickFormatterCompact?: (value: number) => string;
  xLabel?: string;
  xPadding?: {
    start?: number;
    end?: number;
  };

  // Y axis config
  yDomain: [number, number];
  yScale?: 'linear' | 'log';
  yTicks?: number[];
  yTickFormatter?: (value: number, workTimeLabel?: boolean) => string | ReactNode;
  showYAxis?: boolean;
  yLabel?: string;

  // Line configs
  lines: Array<{
    dataKey: string;
    stroke: string;
    strokeWidth?: number;
    strokeOpacity?: number;
    strokeDasharray?: string;
    name?: string;
    smooth?: boolean;
  }>;

  // Optional features
  verticalReferenceLine?: {
    x: number;
    stroke: string;
    strokeDasharray?: string;
    strokeWidth?: number;
    label?: string;
    strokeOpacity?: number;
  };
  customElements?: (context: CustomElementsContext) => ReactNode;
  tooltip?: (point: DataPoint) => ReactNode;
  animateXDomain?: boolean;
  xDomainAnimationDurationMs?: number;
  /** When true, displays horizontal and vertical gridlines */
  showGridlines?: boolean;
}

/**
 * Generate SVG path for an area fill under a line, from xMin to a specific xCutoff
 */
function generateAreaPath(
  data: Array<{ x: number; y: number | null | undefined }>,
  xScale: (value: number) => number,
  yScale: (value: number) => number,
  chartHeight: number,
  xCutoff: number,
  smooth: boolean = false
): string {
  if (data.length === 0) return '';

  // Filter to valid points that are at or before the cutoff
  const validPoints = data.filter(d => d.y != null && !isNaN(d.y) && d.x <= xCutoff);
  if (validPoints.length === 0) return '';

  // Get the y value at the cutoff point by interpolation if needed
  const sortedData = [...data].filter(d => d.y != null && !isNaN(d.y)).sort((a, b) => a.x - b.x);
  let cutoffY: number | null = null;

  // Find interpolated y at cutoff
  for (let i = 0; i < sortedData.length - 1; i++) {
    if (sortedData[i].x <= xCutoff && sortedData[i + 1].x >= xCutoff) {
      const x0 = sortedData[i].x;
      const x1 = sortedData[i + 1].x;
      const y0 = sortedData[i].y!;
      const y1 = sortedData[i + 1].y!;
      const t = (xCutoff - x0) / (x1 - x0);
      cutoffY = y0 + t * (y1 - y0);
      break;
    }
  }

  // If cutoff is beyond data, use last point
  if (cutoffY === null && sortedData.length > 0) {
    const lastPoint = sortedData[sortedData.length - 1];
    if (xCutoff >= lastPoint.x) {
      cutoffY = lastPoint.y!;
    }
  }

  // Build path with baseline
  const baseline = chartHeight; // Bottom of chart (y=0 in data space)
  const pathParts: string[] = [];

  // Include points up to cutoff, plus the interpolated cutoff point
  const pointsToRender = validPoints.filter(p => p.x < xCutoff);
  if (cutoffY !== null) {
    pointsToRender.push({ x: xCutoff, y: cutoffY });
  }

  if (pointsToRender.length === 0) return '';

  if (!smooth) {
    // Start at baseline
    const startX = xScale(pointsToRender[0].x);
    pathParts.push(`M ${startX} ${baseline}`);

    // Draw up to first point
    pathParts.push(`L ${startX} ${yScale(pointsToRender[0].y!)}`);

    // Draw line along curve
    for (let i = 1; i < pointsToRender.length; i++) {
      const x = xScale(pointsToRender[i].x);
      const y = yScale(pointsToRender[i].y!);
      pathParts.push(`L ${x} ${y}`);
    }

    // Draw back down to baseline and close
    const endX = xScale(pointsToRender[pointsToRender.length - 1].x);
    pathParts.push(`L ${endX} ${baseline}`);
    pathParts.push('Z');
  } else {
    // Smooth curve version
    const scaledPoints = pointsToRender.map(point => ({
      x: xScale(point.x),
      y: yScale(point.y!),
    }));

    if (scaledPoints.length === 1) {
      const p = scaledPoints[0];
      return `M ${p.x} ${baseline} L ${p.x} ${p.y} L ${p.x} ${baseline} Z`;
    }

    // Start at baseline
    pathParts.push(`M ${scaledPoints[0].x} ${baseline}`);
    pathParts.push(`L ${scaledPoints[0].x} ${scaledPoints[0].y}`);

    // Use cubic bezier curves for smooth interpolation
    for (let i = 0; i < scaledPoints.length - 1; i++) {
      const p0 = i > 0 ? scaledPoints[i - 1] : scaledPoints[i];
      const p1 = scaledPoints[i];
      const p2 = scaledPoints[i + 1];
      const p3 = i < scaledPoints.length - 2 ? scaledPoints[i + 2] : p2;

      const tension = 0.5;

      const cp1x = p1.x + (p2.x - p0.x) / 6 * tension;
      const cp1y = p1.y + (p2.y - p0.y) / 6 * tension;
      const cp2x = p2.x - (p3.x - p1.x) / 6 * tension;
      const cp2y = p2.y - (p3.y - p1.y) / 6 * tension;

      pathParts.push(`C ${cp1x} ${cp1y}, ${cp2x} ${cp2y}, ${p2.x} ${p2.y}`);
    }

    // Close back to baseline
    const lastPoint = scaledPoints[scaledPoints.length - 1];
    pathParts.push(`L ${lastPoint.x} ${baseline}`);
    pathParts.push('Z');
  }

  return pathParts.join(' ');
}

export function PdfLineChart({
  data,
  width,
  height,
  margin = { top: 20, right: 0, left: 0, bottom: 60 },
  svgRef: externalSvgRef,
  xDomain,
  xTicks: explicitXTicks,
  xTickStepSize,
  xTickCount,
  xTickFormatter = (v) => v.toString(),
  xTickFormatterCompact,
  xLabel,
  yDomain,
  yScale = 'linear',
  yTicks,
  yTickFormatter,
  showYAxis = false,
  yLabel,
  showXAxis = true,
  lines,
  verticalReferenceLine,
  customElements,
  tooltip,
  xPadding,
  animateXDomain = false,
  xDomainAnimationDurationMs = DEFAULT_X_DOMAIN_ANIMATION_DURATION_MS,
  showGridlines = false,
}: PdfLineChartProps) {
  const [hoveredPoint, setHoveredPoint] = useState<DataPoint | null>(null);
  const [mousePosition, setMousePosition] = useState<{ x: number; y: number } | null>(null);
  const [containerWidth, setContainerWidth] = useState(width || 250);
  const containerRef = useRef<HTMLDivElement>(null);
  const svgRef = useRef<SVGSVGElement>(null);

  // Callback ref to assign to both internal and external refs
  const svgRefCallback = useCallback((node: SVGSVGElement | null) => {
    (svgRef as React.MutableRefObject<SVGSVGElement | null>).current = node;
    if (externalSvgRef) {
      (externalSvgRef as React.MutableRefObject<SVGSVGElement | null>).current = node;
    }
  }, [externalSvgRef]);

  const clipId = useId();
  const { hoveredX, setHoveredX } = useChartSync();
  const [animatedXDomain, setAnimatedXDomain] = useState<[number, number]>(xDomain);
  const animatedXDomainRef = useRef(animatedXDomain);
  const xDomainAnimationFrameRef = useRef<number | null>(null);
  const xDomainAnimationStartRef = useRef(0);
  const xDomainFromRef = useRef<[number, number]>(xDomain);
  const xDomainToRef = useRef<[number, number]>(xDomain);
  const [animatedXTicks, setAnimatedXTicks] = useState<number[]>(() =>
    calculateTicks(
      xDomain,
      xTickCount ? { targetCount: xTickCount } : xTickStepSize ? { stepSize: xTickStepSize } : { targetCount: 3 },
      'linear'
    )
  );
  const animatedXTicksRef = useRef(animatedXTicks);
  const xTicksFromRef = useRef<number[]>(animatedXTicks);
  const xTicksToRef = useRef<number[]>(animatedXTicks);

  useEffect(() => {
    animatedXDomainRef.current = animatedXDomain;
  }, [animatedXDomain]);

  useEffect(() => {
    animatedXTicksRef.current = animatedXTicks;
  }, [animatedXTicks]);

  const xTickOptions = useMemo(() => {
    if (xTickCount) {
      return { targetCount: xTickCount };
    }
    if (xTickStepSize) {
      return { stepSize: xTickStepSize };
    }
    return { targetCount: 3 };
  }, [xTickCount, xTickStepSize]);

  const generateTicksForDomain = useCallback(
    (domain: [number, number]) => calculateTicks(domain, xTickOptions, 'linear'),
    [xTickOptions]
  );

  const sampleTickArray = useCallback((ticks: number[], normalizedIndex: number) => {
    if (!ticks.length) {
      return NaN;
    }
    if (ticks.length === 1) {
      return ticks[0];
    }
    const clampedNorm = clamp(normalizedIndex, 0, 1);
    const scaledIndex = clampedNorm * (ticks.length - 1);
    const lowerIndex = Math.floor(scaledIndex);
    const upperIndex = Math.min(ticks.length - 1, Math.ceil(scaledIndex));
    if (lowerIndex === upperIndex) {
      return ticks[lowerIndex];
    }
    const fraction = scaledIndex - lowerIndex;
    const lowerValue = ticks[lowerIndex];
    const upperValue = ticks[upperIndex];
    return lowerValue + (upperValue - lowerValue) * fraction;
  }, []);

  const interpolateTickArrays = useCallback(
    (from: number[], to: number[], progress: number) => {
      if (!from.length && !to.length) {
        return [];
      }

      if (progress >= 1) {
        return [...to];
      }

      if (progress <= 0) {
        return [...from];
      }

      const maxLength = Math.max(from.length, to.length, 1);
      const result: number[] = [];

      for (let i = 0; i < maxLength; i += 1) {
        const normalizedIndex = maxLength === 1 ? 0 : i / (maxLength - 1);
        const startValueRaw = sampleTickArray(from, normalizedIndex);
        const endValueRaw = sampleTickArray(to, normalizedIndex);
        const fallback = Number.isNaN(startValueRaw) ? endValueRaw : startValueRaw;
        const startValue = Number.isNaN(startValueRaw) ? fallback : startValueRaw;
        const endValue = Number.isNaN(endValueRaw) ? fallback : endValueRaw;
        const safeStart = Number.isFinite(startValue) ? startValue : 0;
        const safeEnd = Number.isFinite(endValue) ? endValue : safeStart;
        result.push(safeStart + (safeEnd - safeStart) * progress);
      }

      return result;
    },
    [sampleTickArray]
  );

  useEffect(() => {
    const arraysEqual = (a: number[], b: number[]) =>
      a.length === b.length && a.every((value, idx) => value === b[idx]);

    const nextTicks = generateTicksForDomain(xDomain);

    if (!animateXDomain) {
      if (
        animatedXDomainRef.current[0] !== xDomain[0] ||
        animatedXDomainRef.current[1] !== xDomain[1]
      ) {
        setAnimatedXDomain(xDomain);
        animatedXDomainRef.current = xDomain;
      }
      xDomainFromRef.current = xDomain;
      xDomainToRef.current = xDomain;
      if (xDomainAnimationFrameRef.current !== null) {
        cancelAnimationFrame(xDomainAnimationFrameRef.current);
        xDomainAnimationFrameRef.current = null;
      }
      if (!arraysEqual(animatedXTicksRef.current, nextTicks)) {
        setAnimatedXTicks(nextTicks);
        animatedXTicksRef.current = nextTicks;
        xTicksFromRef.current = nextTicks;
        xTicksToRef.current = nextTicks;
      }
      return;
    }

    const fromDomain = animatedXDomainRef.current;
    const domainUnchanged =
      fromDomain[0] === xDomain[0] && fromDomain[1] === xDomain[1];

    if (domainUnchanged) {
      if (!arraysEqual(animatedXTicksRef.current, nextTicks)) {
        setAnimatedXTicks(nextTicks);
        animatedXTicksRef.current = nextTicks;
        xTicksFromRef.current = nextTicks;
        xTicksToRef.current = nextTicks;
      }
      return;
    }

    xDomainFromRef.current = fromDomain;
    xDomainToRef.current = xDomain;
    xDomainAnimationStartRef.current = performance.now();
    xTicksFromRef.current = animatedXTicksRef.current;
    xTicksToRef.current = nextTicks;

    if (xDomainAnimationFrameRef.current !== null) {
      cancelAnimationFrame(xDomainAnimationFrameRef.current);
    }

    const duration = Math.max(0, xDomainAnimationDurationMs);

    const step = (timestamp: number) => {
      const elapsed = timestamp - xDomainAnimationStartRef.current;
      const progress = duration === 0 ? 1 : Math.min(1, elapsed / duration);
      const nextDomain: [number, number] = [
        xDomainFromRef.current[0] + (xDomainToRef.current[0] - xDomainFromRef.current[0]) * progress,
        xDomainFromRef.current[1] + (xDomainToRef.current[1] - xDomainFromRef.current[1]) * progress,
      ];
      setAnimatedXDomain(nextDomain);
      animatedXDomainRef.current = nextDomain;

      const interpolatedTicks = interpolateTickArrays(
        xTicksFromRef.current,
        xTicksToRef.current,
        progress
      );
      setAnimatedXTicks(interpolatedTicks);
      animatedXTicksRef.current = interpolatedTicks;

      if (progress < 1) {
        xDomainAnimationFrameRef.current = requestAnimationFrame(step);
      } else {
        setAnimatedXTicks(xTicksToRef.current);
        animatedXTicksRef.current = xTicksToRef.current;
        xDomainAnimationFrameRef.current = null;
      }
    };

    xDomainAnimationFrameRef.current = requestAnimationFrame(step);

    return () => {
      if (xDomainAnimationFrameRef.current !== null) {
        cancelAnimationFrame(xDomainAnimationFrameRef.current);
        xDomainAnimationFrameRef.current = null;
      }
    };
  }, [xDomain, animateXDomain, xDomainAnimationDurationMs, generateTicksForDomain, interpolateTickArrays]);

  // Measure container width for responsive sizing
  useEffect(() => {
    if (!containerRef.current || width) return; // Skip if width is explicitly provided

    const resizeObserver = new ResizeObserver(entries => {
      for (const entry of entries) {
        setContainerWidth(entry.contentRect.width);
      }
    });

    resizeObserver.observe(containerRef.current);
    return () => resizeObserver.disconnect();
  }, [width]);

  const actualWidth = width || containerWidth;
  const chartWidth = actualWidth - margin.left - margin.right;
  const chartHeight = height - margin.top - margin.bottom;



  const currentXDomain = animateXDomain ? animatedXDomain : xDomain;


  // Create scales
  const paddingStart = clamp(xPadding?.start ?? 0, 0, chartWidth);
  const paddingEnd = clamp(xPadding?.end ?? 0, 0, chartWidth - paddingStart);
  const xRangeStart = paddingStart;
  const xRangeEnd = Math.max(xRangeStart + 1, chartWidth - paddingEnd);

  const xScaleFunc = createScale({
    domain: currentXDomain,
    range: [xRangeStart, xRangeEnd],
    type: 'linear',
  });

  const yScaleFunc = createScale({
    domain: yDomain,
    range: [chartHeight, 0], // SVG Y is inverted
    type: yScale,
  });

  const customElementGroup = customElements
    ? customElements({
          chartWidth,
          chartHeight,
          xDomain: currentXDomain,
          yDomain,
          xScale: xScaleFunc,
          yScale: yScaleFunc,
      })
    : null;



  // Calculate ticks - use explicit ticks if provided
  const xTicks = explicitXTicks ?? (animateXDomain ? animatedXTicks : generateTicksForDomain(currentXDomain));

  // On narrow (mobile) charts, drop tick labels that would overlap; gridlines keep all ticks.
  // If full labels don't all fit and a compact formatter exists, switch to it before thinning.
  let xLabelFormatter = xTickFormatter;
  let xLabelTicks = thinTicksToFitLabels(xTicks, xScaleFunc, xLabelFormatter);
  if (xTickFormatterCompact && xLabelTicks.length < xTicks.length) {
    xLabelFormatter = xTickFormatterCompact;
    xLabelTicks = thinTicksToFitLabels(xTicks, xScaleFunc, xLabelFormatter);
  }

  const handleMouseMove = useCallback((e: React.MouseEvent<SVGSVGElement>) => {
    if (!svgRef.current) return;

    const rect = svgRef.current.getBoundingClientRect();
    const mouseX = e.clientX - rect.left - margin.left;

    setMousePosition({ x: e.clientX, y: e.clientY });

    // Find closest point based on X-axis distance only (better for time-series)
    const mainDataKey = lines[lines.length - 1]?.dataKey; // Use last line (main line)
    if (!mainDataKey) return;

    // Filter to points within domain bounds
    const dataWithMainLine = data.filter(d =>
      d[mainDataKey] != null &&
      d.x >= currentXDomain[0] &&
      d.x <= currentXDomain[1]
    );

    // Find closest point by X-axis distance only
    let closest: DataPoint | null = null;
    let minXDistance = Infinity;

    for (const point of dataWithMainLine) {
      const px = xScaleFunc(point.x);
      const xDistance = Math.abs(mouseX - px);

      if (xDistance < minXDistance) {
        minXDistance = xDistance;
        closest = point;
      }
    }

    if (tooltip) {
      setHoveredPoint(closest);
    }

    // Update shared hover X position for crosshair sync
    if (closest) {
      setHoveredX(closest.x);
    }
  }, [data, lines, margin, xScaleFunc, tooltip, setHoveredX, currentXDomain]);

  const handleMouseLeave = useCallback(() => {
    setHoveredPoint(null);
    setMousePosition(null);
    setHoveredX(null);
  }, [setHoveredX]);

  return (
    <div ref={containerRef} style={{ position: 'relative', width: width || '100%', height }}>
      <svg
        ref={svgRefCallback}
        width={actualWidth}
        height={height}
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
        style={{ userSelect: 'none' }}
      >
        <defs>
          <clipPath id={clipId}>
            <rect x={0} y={0} width={xRangeEnd} height={chartHeight} />
          </clipPath>
        </defs>
        <g transform={`translate(${margin.left},${margin.top})`}>
          {/* Gridlines */}
          {showGridlines && (
            <g className="gridlines" opacity={0.3}>
              {/* Horizontal gridlines at y-tick positions */}
              {yTicks && yTicks.map((tick, i) => (
                <line
                  key={`y-grid-${i}`}
                  x1={0}
                  y1={yScaleFunc(tick)}
                  x2={chartWidth}
                  y2={yScaleFunc(tick)}
                  stroke="#666"
                  strokeWidth={1}
                  strokeDasharray="4 4"
                />
              ))}
              {/* Vertical gridlines at x-tick positions */}
              {xTicks.map((tick, i) => {
                const xPos = xScaleFunc(tick);
                if (xPos < 0 || xPos > chartWidth) return null;
                return (
                  <line
                    key={`x-grid-${i}`}
                    x1={xPos}
                    y1={0}
                    x2={xPos}
                    y2={chartHeight}
                    stroke="#666"
                    strokeWidth={1}
                    strokeDasharray="4 4"
                  />
                );
              })}
            </g>
          )}

          {/* Vertical reference line */}
          {verticalReferenceLine && (
            <g>
              <line
                x1={xScaleFunc(verticalReferenceLine.x)}
                y1={0}
                x2={xScaleFunc(verticalReferenceLine.x)}
                y2={chartHeight}
                stroke={verticalReferenceLine.stroke}
                strokeDasharray={verticalReferenceLine.strokeDasharray || '2 10'}
                strokeWidth={verticalReferenceLine.strokeWidth || 1}
                strokeOpacity={verticalReferenceLine.strokeOpacity ?? 0.5}
                strokeLinecap="round"
              />
              {verticalReferenceLine.label && (
                <text
                  x={xScaleFunc(verticalReferenceLine.x) + 5}
                  y={15}
                  textAnchor="start"
                  fontSize={11}
                  fill={verticalReferenceLine.stroke}
                  transform={`rotate(-90, ${xScaleFunc(verticalReferenceLine.x) + 5}, 15)`}
                >
                  {verticalReferenceLine.label}
                </text>
              )}
            </g>
          )}

          {/* Clipped group for lines and area fills */}
          <g clipPath={`url(#${clipId})`}>
            {/* Area fills for hover shading (render before lines so they appear behind) */}
            {hoveredX !== null && lines.map((line, i) => {
              const lineData = data.map(d => ({
                x: d.x,
                y: d[line.dataKey] as number | null | undefined,
              }));

              const areaPath = generateAreaPath(
                lineData,
                xScaleFunc,
                yScaleFunc,
                chartHeight,
                hoveredX,
                line.smooth
              );

              if (!areaPath) return null;

              return (
                <path
                  key={`area-${i}-${line.dataKey}`}
                  data-export-hide="true"
                  d={areaPath}
                  fill={line.stroke}
                  fillOpacity={0.15}
                  stroke="none"
                  pointerEvents="none"
                />
              );
            })}

            {/* Lines */}
            {lines.map((line, i) => {
              const lineData = data.map(d => ({
                x: d.x,
                y: d[line.dataKey] as number | null | undefined,
              }));

              const path = generateLinePath(lineData, xScaleFunc, yScaleFunc, line.smooth);

              return (
                <path
                  id={`line-${i}-${line.dataKey}`}
                  key={`line-${i}-${line.dataKey}`}
                  d={path}
                  fill="none"
                  stroke={line.stroke}
                  strokeWidth={line.strokeWidth ?? 2}
                  strokeOpacity={line.strokeOpacity ?? 1}
                  strokeDasharray={line.strokeDasharray}
                  style={{ transition: 'stroke-opacity 0.6s ease-in-out' }}
                />
              );
            })}
          </g>

          {customElementGroup}

          {/* Synchronized crosshair */}
          {hoveredX !== null && (
            <line
              data-export-hide="true"
              x1={xScaleFunc(hoveredX)}
              y1={0}
              x2={xScaleFunc(hoveredX)}
              y2={chartHeight}
              stroke="#999"
              strokeWidth={1}
              strokeDasharray="4 4"
              opacity={0.6}
              pointerEvents="none"
            />
          )}

          {/* Y Axis */}
          {showYAxis && yTicks && yTicks.length > 0 && (
            <g>
              {yTicks.map((tick, i) => {
                const yPos = yScaleFunc(tick);
                const labelContent = yTickFormatter ? yTickFormatter(tick) : tick.toString();
                const isReactNode = typeof labelContent !== 'string';
                return (
                  <g key={`y-tick-${i}`} transform={`translate(0,${yPos})`}>
                    {isReactNode ? (
                      <foreignObject x={-margin.left + 5} y={-10} width={margin.left - 10} height={20} style={{ overflow: 'visible' }}>
                        <div style={{ fontSize: '11px', color: '#666', fontWeight: 500, fontFamily: 'inherit', textAlign: 'right' }}>
                          {labelContent as unknown as ReactNode}
                        </div>
                      </foreignObject>
                    ) : (
                      <text
                        x={-8}
                        textAnchor="end"
                        alignmentBaseline="middle"
                        fontSize={11}
                        fill="#666"
                      >
                        {labelContent as string}
                      </text>
                    )}
                  </g>
                );
              })}
            </g>
          )}

          {/* Y Axis Label */}
          {yLabel && (
            <text
              x={-chartHeight / 2}
              y={-margin.left + 12}
              textAnchor="middle"
              fontSize={12}
              fill="#666"
              transform="rotate(-90)"
            >
              {yLabel}
            </text>
          )}

          {/* X Axis */}
          {showXAxis && (
          <g transform={`translate(0,${chartHeight})`}>
            {xLabelTicks.map((tick, i) => {
              const xPos = xScaleFunc(tick);
              // Skip ticks that are too close to edges to be centered properly
              // A year label like "2024" is ~30px wide, so need ~15px clearance on each side
              // A centered label needs clearance on both sides; near the chart
              // edges re-anchor it inward (start/end) instead of dropping it,
              // so the axis range stays readable on narrow charts.
              const minEdgeDistance = 20;
              const anchor = xPos < minEdgeDistance ? 'start'
                : xPos > chartWidth - minEdgeDistance ? 'end'
                : 'middle';

              return (
                <g key={`x-tick-${i}`} transform={`translate(${xPos},0)`}>
                <text
                  y={20}
                  textAnchor={anchor}
                  fontSize={12}
                  fill="#666"
                >
                  {xLabelFormatter(tick)}
                </text>
              </g>
              );
            })}
            {xLabel && (
              <text
                x={chartWidth / 2}
                y={50}
                textAnchor="middle"
                fontSize={14}
                fill="#666"
              >
                {xLabel}
                </text>
              )}
            </g>
          )}

        </g>
      </svg>

      {/* Tooltip */}
      {tooltip && hoveredPoint && mousePosition && (
        <div
          style={{
            position: 'fixed',
            left: mousePosition.x + 12,
            top: mousePosition.y + 12,
            pointerEvents: 'none',
            zIndex: 1000,
          }}
        >
          {tooltip(hoveredPoint)}
        </div>
      )}
    </div>
  );
}
