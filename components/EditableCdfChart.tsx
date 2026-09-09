'use client';

import { useRef, useState, useCallback, useMemo, useEffect, useId } from 'react';
import { createScale } from '@/utils/chartUtils';
import {
  ControlPoint,
  CdfPoint,
  InterpolationMethod,
  createInverseScale,
  clamp,
  constrainXBetweenNeighbors,
  enforceMonotonicity,
  addControlPoint,
  removeControlPoint,
  generateSplinePath,
  evaluateSplineAt,
} from '@/utils/cdfSplineUtils';
import {
  evaluateGammaMixtureAt,
  generateGammaMixturePath,
} from '@/utils/gammaMixtureUtils';

// Find the x value where the CDF equals a target y value (e.g., 0.5 for median)
function findXForY(
  controlPoints: ControlPoint[],
  targetY: number,
  method: InterpolationMethod = 'ispline'
): number | null {
  if (controlPoints.length < 2) return null;

  const sorted = [...controlPoints].sort((a, b) => a.x - b.x);
  const minX = sorted[0].x;
  const maxX = sorted[sorted.length - 1].x;

  // Choose evaluation function based on method
  const evaluate = method === 'gamma' ? evaluateGammaMixtureAt : evaluateSplineAt;

  // Check if target is within the CDF range
  const minY = evaluate(sorted, minX);
  const maxY = evaluate(sorted, maxX);

  if (targetY < minY || targetY > maxY) return null;

  // Binary search to find x where CDF(x) ≈ targetY
  let low = minX;
  let high = maxX;
  const tolerance = 0.001;

  for (let i = 0; i < 50; i++) {
    const mid = (low + high) / 2;
    const midY = evaluate(sorted, mid);

    if (Math.abs(midY - targetY) < tolerance) {
      return mid;
    }

    if (midY < targetY) {
      low = mid;
    } else {
      high = mid;
    }
  }

  return (low + high) / 2;
}

// Format a decimal year as "Mon YYYY" (e.g., 2032.5 -> "Jul 2032")
function formatDecimalYearAsDate(decimalYear: number): string {
  const year = Math.floor(decimalYear);
  const fraction = decimalYear - year;
  const monthIndex = Math.round(fraction * 12);
  const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
  return `${months[Math.min(monthIndex, 11)]} ${year}`;
}

// Color palette for milestones
const MILESTONE_COLORS: { [key: string]: string } = {
  'AC': '#2A623D',
  'AI2027-SC': '#af1e86ff',
  'SAR-level-experiment-selection-skill': '#000090',
  'SIAR-level-experiment-selection-skill': '#900000',
};

export interface CdfOverlayLine {
  /** Pre-sampled CDF points to render as a polyline */
  points: Array<{ x: number; y: number }>;
  color: string;
  name: string;
  strokeDasharray?: string;
  strokeOpacity?: number;
  strokeWidth?: number;
}

interface EditableCdfChartProps {
  empiricalCdf: CdfPoint[];
  controlPoints: ControlPoint[];
  onControlPointsChange: (points: ControlPoint[]) => void;
  milestoneName: string;
  interpolationMethod?: InterpolationMethod;
  gammaRegularization?: number;
  isplineRegularization?: number;
  width?: number;
  height?: number;
  xAxisMode?: 'absoluteYear' | 'relativeYears';
  xAxisLabel?: string;
  xTickFormatter?: (value: number) => string;
  xDomain?: [number, number];
  /** Optional overlay CDF lines from other milestones */
  overlayLines?: CdfOverlayLine[];
}

// CDF probability axis is fixed [0,1] with a little padding; hoisted to module
// scope so it's a stable reference the scale memos don't need to depend on.
const Y_DOMAIN: [number, number] = [-0.05, 1.05];

export function EditableCdfChart({
  empiricalCdf,
  controlPoints,
  onControlPointsChange,
  milestoneName,
  interpolationMethod = 'ispline',
  gammaRegularization = 0.01,
  isplineRegularization = 0.05,
  width = 900,
  height = 400,
  xAxisMode = 'absoluteYear',
  xAxisLabel,
  xTickFormatter,
  xDomain: xDomainProp,
  overlayLines,
}: EditableCdfChartProps) {
  // Choose evaluation and path generation functions based on method
  const evaluateCdf = useMemo(() => {
    if (interpolationMethod === 'gamma') {
      return (pts: ControlPoint[], x: number) => evaluateGammaMixtureAt(pts, x, gammaRegularization);
    }
    return (pts: ControlPoint[], x: number) => evaluateSplineAt(pts, x, isplineRegularization);
  }, [interpolationMethod, gammaRegularization, isplineRegularization]);

  const generatePath = useMemo(() => {
    if (interpolationMethod === 'gamma') {
      return (pts: ControlPoint[], xScale: (v: number) => number, yScale: (v: number) => number, samples?: number) =>
        generateGammaMixturePath(pts, xScale, yScale, samples, gammaRegularization);
    }
    return (pts: ControlPoint[], xScale: (v: number) => number, yScale: (v: number) => number, samples?: number) =>
      generateSplinePath(pts, xScale, yScale, samples, isplineRegularization);
  }, [interpolationMethod, gammaRegularization, isplineRegularization]);
  const svgRef = useRef<SVGSVGElement>(null);
  const clipId = useId();
  const [activePointId, setActivePointId] = useState<string | null>(null);
  const [hoveredPointId, setHoveredPointId] = useState<string | null>(null);
  const [hoveredX, setHoveredX] = useState<number | null>(null);

  const margin = { top: 20, right: 40, bottom: 60, left: 60 };
  const chartWidth = width - margin.left - margin.right;
  const chartHeight = height - margin.top - margin.bottom;

  // Calculate domains - use prop if provided, otherwise auto-calculate
  const xDomainAuto = useMemo<[number, number]>(() => {
    if (empiricalCdf.length === 0) {
      return xAxisMode === 'relativeYears' ? [0, 25] : [2025, 2050];
    }
    const years = empiricalCdf.map(p => p.year);
    const min = Math.min(...years);
    const max = Math.max(...years);
    const padding = (max - min) * 0.02;
    const paddedMin = xAxisMode === 'relativeYears' ? Math.max(0, min - padding) : min - padding;
    return [paddedMin, max + padding];
  }, [empiricalCdf, xAxisMode]);

  const xDomain = xDomainProp ?? xDomainAuto;

  // Create scales
  const xScale = useMemo(() => createScale({
    domain: xDomain,
    range: [0, chartWidth],
    type: 'linear',
  }), [xDomain, chartWidth]);

  const yScale = useMemo(() => createScale({
    domain: Y_DOMAIN,
    range: [chartHeight, 0],
    type: 'linear',
  }), [chartHeight]);

  // Create inverse scales for drag handling
  const inverseXScale = useMemo(() => createInverseScale({
    domain: xDomain,
    range: [0, chartWidth],
    type: 'linear',
  }), [xDomain, chartWidth]);

  const inverseYScale = useMemo(() => createInverseScale({
    domain: Y_DOMAIN,
    range: [chartHeight, 0],
    type: 'linear',
  }), [chartHeight]);

  // Generate empirical CDF path
  const empiricalPath = useMemo(() => {
    if (empiricalCdf.length === 0) return '';
    const sorted = [...empiricalCdf].sort((a, b) => a.year - b.year);
    const pathParts: string[] = [];

    for (let i = 0; i < sorted.length; i++) {
      const x = xScale(sorted[i].year);
      const y = yScale(sorted[i].cdf);
      if (i === 0) {
        pathParts.push(`M ${x} ${y}`);
      } else {
        pathParts.push(`L ${x} ${y}`);
      }
    }

    return pathParts.join(' ');
  }, [empiricalCdf, xScale, yScale]);

  // Generate edited CDF path (using selected interpolation method)
  const splinePath = useMemo(() => {
    if (controlPoints.length < 2) return '';
    return generatePath(controlPoints, xScale, yScale, 200);
  }, [controlPoints, xScale, yScale, generatePath]);

  // Calculate median (x where CDF = 0.5)
  const medianYear = useMemo(() => {
    return findXForY(controlPoints, 0.5, interpolationMethod);
  }, [controlPoints, interpolationMethod]);

  // Get empirical CDF value at a given x (linear interpolation)
  const getEmpiricalCdfAt = useCallback((x: number): number | null => {
    if (empiricalCdf.length === 0) return null;
    const sorted = [...empiricalCdf].sort((a, b) => a.year - b.year);
    if (x <= sorted[0].year) return sorted[0].cdf;
    if (x >= sorted[sorted.length - 1].year) return sorted[sorted.length - 1].cdf;

    for (let i = 0; i < sorted.length - 1; i++) {
      if (x >= sorted[i].year && x <= sorted[i + 1].year) {
        const t = (x - sorted[i].year) / (sorted[i + 1].year - sorted[i].year);
        return sorted[i].cdf + t * (sorted[i + 1].cdf - sorted[i].cdf);
      }
    }
    return null;
  }, [empiricalCdf]);

  // Get milestone color
  const milestoneColor = MILESTONE_COLORS[milestoneName] ?? '#2563eb';

  // Mouse event handlers
  const handleMouseDown = useCallback((pointId: string) => (e: React.MouseEvent) => {
    e.preventDefault();
    const point = controlPoints.find(p => p.id === pointId);
    if (point?.isLocked) return;
    setActivePointId(pointId);
  }, [controlPoints]);

  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    if (!svgRef.current) return;

    const rect = svgRef.current.getBoundingClientRect();
    const mouseX = e.clientX - rect.left - margin.left;
    const mouseY = e.clientY - rect.top - margin.top;

    // Update hover position (for crosshair)
    if (mouseX >= 0 && mouseX <= chartWidth && mouseY >= 0 && mouseY <= chartHeight) {
      const hoverDataX = inverseXScale(mouseX);
      setHoveredX(clamp(hoverDataX, xDomain[0], xDomain[1]));
    }

    // Handle dragging
    if (!activePointId) return;

    // Convert to data coordinates
    let dataX = inverseXScale(mouseX);
    let dataY = inverseYScale(mouseY);

    // Clamp Y to valid CDF range
    dataY = clamp(dataY, 0, 1);

    // Constrain X between neighbors
    dataX = constrainXBetweenNeighbors(controlPoints, activePointId, dataX);

    // Also constrain X to the data range
    dataX = clamp(dataX, xDomain[0], xDomain[1]);

    // Update the point
    const newPoints = controlPoints.map(p => {
      if (p.id === activePointId) {
        return { ...p, x: dataX, y: dataY };
      }
      return p;
    });

    // Enforce monotonicity
    const monotonic = enforceMonotonicity(newPoints);
    onControlPointsChange(monotonic);
  }, [activePointId, controlPoints, inverseXScale, inverseYScale, xDomain, margin, chartWidth, chartHeight, onControlPointsChange]);

  const handleMouseUp = useCallback(() => {
    setActivePointId(null);
  }, []);

  const handleMouseLeave = useCallback(() => {
    setHoveredX(null);
    setActivePointId(null);
  }, []);

  // Global mouse up handler
  useEffect(() => {
    if (activePointId) {
      const handleGlobalMouseUp = () => setActivePointId(null);
      window.addEventListener('mouseup', handleGlobalMouseUp);
      return () => window.removeEventListener('mouseup', handleGlobalMouseUp);
    }
  }, [activePointId]);

  // Add control point
  const handleAddPoint = useCallback(() => {
    const newPoints = addControlPoint(controlPoints);
    onControlPointsChange(newPoints);
  }, [controlPoints, onControlPointsChange]);

  // Remove control point
  const handleRemovePoint = useCallback((pointId: string) => {
    const newPoints = removeControlPoint(controlPoints, pointId);
    onControlPointsChange(newPoints);
  }, [controlPoints, onControlPointsChange]);

  // Generate X-axis ticks
  const xTicks = useMemo(() => {
    const [min, max] = xDomain;
    const ticks: number[] = [];
    if (xAxisMode === 'relativeYears') {
      // For relative years, use nice round numbers starting from 0
      const step = Math.ceil((max - min) / 6);
      const niceStep = step <= 1 ? 1 : step <= 2 ? 2 : step <= 5 ? 5 : Math.ceil(step / 5) * 5;
      for (let t = 0; t <= max + 0.001; t += niceStep) {
        ticks.push(t);
      }
    } else {
      const startYear = Math.ceil(min);
      const step = Math.ceil((max - min) / 6);
      for (let year = startYear; year <= max; year += step) {
        ticks.push(year);
      }
    }
    return ticks;
  }, [xDomain, xAxisMode]);

  // Format x-axis tick value
  const formatXTick = useCallback((value: number): string => {
    if (xTickFormatter) return xTickFormatter(value);
    if (xAxisMode === 'relativeYears') {
      return value < 1 ? value.toFixed(1) : value.toFixed(0);
    }
    return value.toFixed(0);
  }, [xAxisMode, xTickFormatter]);

  // Get x-axis label
  const xAxisLabelText = xAxisLabel ?? (xAxisMode === 'relativeYears' ? 'Years from AC' : 'Year');

  // Generate Y-axis ticks
  const yTicks = [0, 0.25, 0.5, 0.75, 1];

  return (
    <div className="bg-white rounded-lg shadow p-4">
      {/* Control Point buttons */}
      <div className="flex gap-2 mb-4">
        <button
          onClick={handleAddPoint}
          className="px-3 py-1 text-sm font-medium text-gray-700 bg-gray-100 border border-gray-300 rounded hover:bg-gray-200"
        >
          + Add Point
        </button>
        <span className="text-sm text-gray-500 self-center">
          {controlPoints.length} control points
        </span>
      </div>

      {/* Legend */}
      <div className="flex gap-6 mb-4 text-xs">
        <div className="flex items-center gap-2">
          <div className="w-8 h-0.5 opacity-30" style={{ backgroundColor: milestoneColor }} />
          <span className="text-gray-600">Empirical CDF (reference)</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-8 h-0.5" style={{ backgroundColor: milestoneColor }} />
          <span className="text-gray-600">Your All-Things-Considered CDF</span>
        </div>
        <div className="flex items-center gap-2">
          <div
            className="w-3 h-3 rounded-full border-2"
            style={{ borderColor: milestoneColor, backgroundColor: 'white' }}
          />
          <span className="text-gray-600">Draggable control point (shift+click to remove)</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-8 h-0.5 border-t-2 border-dashed border-orange-500" />
          <span className="text-gray-600">Median (50%)</span>
        </div>
      </div>

      <svg
        ref={svgRef}
        width={width}
        height={height}
        className="select-none"
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseLeave}
        style={{ cursor: activePointId ? 'grabbing' : 'default' }}
      >
        <defs>
          <clipPath id={clipId}>
            <rect x={0} y={0} width={chartWidth} height={chartHeight} />
          </clipPath>
        </defs>
        <g transform={`translate(${margin.left}, ${margin.top})`}>
          {/* Grid lines */}
          {yTicks.map(tick => (
            <line
              key={`y-grid-${tick}`}
              x1={0}
              x2={chartWidth}
              y1={yScale(tick)}
              y2={yScale(tick)}
              stroke="#e5e7eb"
              strokeWidth={1}
            />
          ))}
          {xTicks.map(tick => (
            <line
              key={`x-grid-${tick}`}
              x1={xScale(tick)}
              x2={xScale(tick)}
              y1={0}
              y2={chartHeight}
              stroke="#e5e7eb"
              strokeWidth={1}
            />
          ))}

          {/* Clipped group for curves */}
          <g clipPath={`url(#${clipId})`}>
            {/* Overlay CDF lines from other milestones */}
            {overlayLines?.map((line, i) => {
              const validPoints = line.points.filter(p => p.x >= xDomain[0] && p.x <= xDomain[1]);
              if (validPoints.length < 2) return null;
              const d = validPoints
                .map((p, j) => `${j === 0 ? 'M' : 'L'} ${xScale(p.x)} ${yScale(p.y)}`)
                .join(' ');
              return (
                <path
                  key={`overlay-${i}-${line.name}`}
                  d={d}
                  fill="none"
                  stroke={line.color}
                  strokeWidth={line.strokeWidth ?? 1.5}
                  strokeOpacity={line.strokeOpacity ?? 0.5}
                  strokeDasharray={line.strokeDasharray ?? '4 2'}
                  pointerEvents="none"
                />
              );
            })}

            {/* Empirical CDF (faded) */}
            <path
              d={empiricalPath}
              fill="none"
              stroke={milestoneColor}
              strokeWidth={2}
              opacity={0.3}
            />

            {/* Edited spline */}
            <path
              d={splinePath}
              fill="none"
              stroke={milestoneColor}
              strokeWidth={2.5}
            />

            {/* Median line (inside clip so it doesn't extend beyond chart) */}
            {medianYear !== null && (
              <line
                x1={xScale(medianYear)}
                x2={xScale(medianYear)}
                y1={0}
                y2={chartHeight}
                stroke="#f97316"
                strokeWidth={2}
                strokeDasharray="6,4"
              />
          )}

          {/* Hover crosshair with values */}
          {hoveredX !== null && !activePointId && (() => {
            // Interpolate overlay CDF values at hover position
            const overlayValues = overlayLines?.map(line => {
              const pts = line.points;
              if (pts.length < 2) return { name: line.name, color: line.color, value: null };
              if (hoveredX <= pts[0].x) return { name: line.name, color: line.color, value: pts[0].y };
              if (hoveredX >= pts[pts.length - 1].x) return { name: line.name, color: line.color, value: pts[pts.length - 1].y };
              for (let i = 0; i < pts.length - 1; i++) {
                if (hoveredX >= pts[i].x && hoveredX <= pts[i + 1].x) {
                  const t = (hoveredX - pts[i].x) / (pts[i + 1].x - pts[i].x);
                  return { name: line.name, color: line.color, value: pts[i].y + t * (pts[i + 1].y - pts[i].y) };
                }
              }
              return { name: line.name, color: line.color, value: null };
            }) ?? [];

            const tooltipRows = 3 + overlayValues.length; // header + empirical + ATC + overlays
            const tooltipHeight = 12 + tooltipRows * 16;
            const tooltipWidth = overlayValues.length > 0 ? 180 : 130;

            return (
            <g pointerEvents="none">
              <line
                x1={xScale(hoveredX)}
                x2={xScale(hoveredX)}
                y1={0}
                y2={chartHeight}
                stroke="#666"
                strokeWidth={1}
                strokeDasharray="4,4"
                opacity={0.7}
              />
              {/* Tooltip box */}
              <g transform={`translate(${Math.min(xScale(hoveredX) + 10, chartWidth - tooltipWidth - 10)}, 10)`}>
                <rect
                  x={0}
                  y={0}
                  width={tooltipWidth}
                  height={tooltipHeight}
                  fill="white"
                  stroke="#e5e7eb"
                  strokeWidth={1}
                  rx={4}
                />
                <text x={8} y={16} fontSize="11" fontWeight="600" fill="#374151">
                  {xAxisMode === 'relativeYears' ? `${hoveredX.toFixed(2)} yr` : hoveredX.toFixed(1)}
                </text>
                <text x={8} y={32} fontSize="10" fill="#666">
                  Empirical: {((getEmpiricalCdfAt(hoveredX) ?? 0) * 100).toFixed(1)}%
                </text>
                <text x={8} y={48} fontSize="10" fill={milestoneColor} fontWeight="500">
                  {milestoneName}: {(evaluateCdf(controlPoints, hoveredX) * 100).toFixed(1)}%
                </text>
                {overlayValues.map((ov, i) => (
                  ov.value !== null && (
                    <text key={ov.name} x={8} y={64 + i * 16} fontSize="10" fill={ov.color} opacity={0.8}>
                      {ov.name}: {(ov.value * 100).toFixed(1)}%
                    </text>
                  )
                ))}
              </g>
              {/* Dots on the lines */}
              {getEmpiricalCdfAt(hoveredX) !== null && (
                <circle
                  cx={xScale(hoveredX)}
                  cy={yScale(getEmpiricalCdfAt(hoveredX)!)}
                  r={4}
                  fill={milestoneColor}
                  opacity={0.4}
                />
              )}
              {controlPoints.length >= 2 && (
                <circle
                  cx={xScale(hoveredX)}
                  cy={yScale(evaluateCdf(controlPoints, hoveredX))}
                  r={5}
                  fill={milestoneColor}
                  stroke="white"
                  strokeWidth={2}
                />
              )}
              {/* Dots on overlay lines */}
              {overlayValues.map(ov => (
                ov.value !== null && (
                  <circle
                    key={`dot-${ov.name}`}
                    cx={xScale(hoveredX)}
                    cy={yScale(ov.value)}
                    r={3}
                    fill={ov.color}
                    opacity={0.6}
                  />
                )
              ))}
            </g>
            );
          })()}

          {/* Control points */}
          {controlPoints.map(point => {
            const cx = xScale(point.x);
            const cy = yScale(point.y);
            const isActive = point.id === activePointId;
            const isHovered = point.id === hoveredPointId;

            return (
              <g key={point.id}>
                {/* Hit area (larger invisible circle for easier clicking) */}
                <circle
                  cx={cx}
                  cy={cy}
                  r={16}
                  fill="transparent"
                  style={{ cursor: point.isLocked ? 'not-allowed' : (isActive ? 'grabbing' : 'grab') }}
                  onMouseDown={handleMouseDown(point.id)}
                  onMouseEnter={() => setHoveredPointId(point.id)}
                  onMouseLeave={() => setHoveredPointId(null)}
                  onClick={(e) => {
                    if (!point.isLocked && e.shiftKey) {
                      e.preventDefault();
                      handleRemovePoint(point.id);
                    }
                  }}
                />
                {/* Visible point */}
                <circle
                  cx={cx}
                  cy={cy}
                  r={isActive || isHovered ? 10 : 8}
                  fill={point.isLocked ? '#9ca3af' : (isActive ? '#f97316' : 'white')}
                  stroke={point.isLocked ? '#6b7280' : milestoneColor}
                  strokeWidth={2}
                  style={{ pointerEvents: 'none' }}
                />
                {/* Point label on hover */}
                {isHovered && (
                  <text
                    x={cx}
                    y={cy - 16}
                    textAnchor="middle"
                    fontSize="10"
                    fill="#374151"
                  >
                    {xAxisMode === 'relativeYears' ? `${point.x.toFixed(2)} yr` : point.x.toFixed(1)}, {(point.y * 100).toFixed(0)}%
                    {point.isLocked ? ' (locked)' : ' (shift+click to remove)'}
                  </text>
                )}
              </g>
            );
          })}
          </g>

          {/* Median label (outside clip group so it's visible above the chart) */}
          {medianYear !== null && (
            <text
              x={xScale(medianYear)}
              y={-6}
              textAnchor="middle"
              fontSize="11"
              fontWeight="600"
              fill="#f97316"
            >
              Median: {xAxisMode === 'relativeYears' ? `${medianYear.toFixed(2)} yr` : formatDecimalYearAsDate(medianYear)}
            </text>
          )}

          {/* X-axis */}
          <line
            x1={0}
            x2={chartWidth}
            y1={chartHeight}
            y2={chartHeight}
            stroke="#374151"
            strokeWidth={1}
          />
          {xTicks.map(tick => (
            <g key={`x-tick-${tick}`} transform={`translate(${xScale(tick)}, ${chartHeight})`}>
              <line y2={6} stroke="#374151" />
              <text y={20} textAnchor="middle" fontSize="12" fill="#374151">
                {formatXTick(tick)}
              </text>
            </g>
          ))}
          <text
            x={chartWidth / 2}
            y={chartHeight + 45}
            textAnchor="middle"
            fontSize="12"
            fill="#374151"
          >
            {xAxisLabelText}
          </text>

          {/* Y-axis */}
          <line
            x1={0}
            x2={0}
            y1={0}
            y2={chartHeight}
            stroke="#374151"
            strokeWidth={1}
          />
          {yTicks.map(tick => (
            <g key={`y-tick-${tick}`} transform={`translate(0, ${yScale(tick)})`}>
              <line x2={-6} stroke="#374151" />
              <text x={-10} textAnchor="end" dominantBaseline="middle" fontSize="12" fill="#374151">
                {(tick * 100).toFixed(0)}%
              </text>
            </g>
          ))}
          <text
            transform={`translate(-45, ${chartHeight / 2}) rotate(-90)`}
            textAnchor="middle"
            fontSize="12"
            fill="#374151"
          >
            Cumulative Probability
          </text>
        </g>
      </svg>

      <p className="text-xs text-gray-500 mt-2">
        Drag control points to adjust the distribution. Shift+click on a point to remove it.
        The spline is constrained to be a valid CDF (monotonically increasing from 0 to 1).
      </p>
    </div>
  );
}
