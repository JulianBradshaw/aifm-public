'use client';

import { useCallback, useState, useEffect, useRef, useMemo } from 'react';
import { ControlPoint, InterpolationMethod, evaluateSplineAt } from '@/utils/cdfSplineUtils';
import { evaluateGammaMixtureAt } from '@/utils/gammaMixtureUtils';
import { CHART_DOMAIN_CONFIG } from '@/constants/chartDomainConfig';

interface CdfPoint {
  year: number;
  cdf: number;
}

interface ControlPointsTableProps {
  controlPoints: ControlPoint[];
  onControlPointsChange: (points: ControlPoint[]) => void;
  xAxisLabel?: string;
  xMin?: number;
  xMax?: number;
  // For showing actual interpolated CDF values
  interpolationMethod?: InterpolationMethod;
  gammaRegularization?: number;
  isplineRegularization?: number;
  // Empirical CDF data for comparison
  empiricalCdf?: CdfPoint[];
}

// Input component that uses local state and commits on blur/Enter
// Uses refs for editing state to avoid race conditions
function EditableNumberInput({
  value,
  onChange,
  min,
  max,
  className,
  formatValue = (v: number) => v.toString(),
  parseValue = (s: string) => parseFloat(s),
}: {
  value: number;
  onChange: (value: number) => void;
  min?: number;
  max?: number;
  className?: string;
  formatValue?: (value: number) => string;
  parseValue?: (str: string) => number;
}) {
  const [displayValue, setDisplayValue] = useState(() => formatValue(value));
  const isEditingRef = useRef(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const lastExternalValue = useRef(value);

  // Sync display value from props only when not editing and value actually changed
  useEffect(() => {
    if (!isEditingRef.current && value !== lastExternalValue.current) {
      setDisplayValue(formatValue(value));
      lastExternalValue.current = value;
    }
  }, [value, formatValue]);

  const commitValue = useCallback(() => {
    const parsed = parseValue(displayValue);
    if (!isNaN(parsed)) {
      const clamped = Math.max(min ?? -Infinity, Math.min(max ?? Infinity, parsed));
      lastExternalValue.current = clamped;
      onChange(clamped);
      // Update display to formatted version of clamped value
      setDisplayValue(formatValue(clamped));
    } else {
      // Invalid input - revert to last known good value
      setDisplayValue(formatValue(value));
    }
  }, [displayValue, parseValue, min, max, onChange, formatValue, value]);

  const handleFocus = useCallback(() => {
    isEditingRef.current = true;
  }, []);

  const handleBlur = useCallback(() => {
    isEditingRef.current = false;
    commitValue();
  }, [commitValue]);

  const handleChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    setDisplayValue(e.target.value);
  }, []);

  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      commitValue();
      inputRef.current?.blur();
    } else if (e.key === 'Escape') {
      setDisplayValue(formatValue(value));
      isEditingRef.current = false;
      inputRef.current?.blur();
    }
  }, [commitValue, formatValue, value]);

  return (
    <input
      ref={inputRef}
      type="text"
      inputMode="decimal"
      value={displayValue}
      onChange={handleChange}
      onFocus={handleFocus}
      onBlur={handleBlur}
      onKeyDown={handleKeyDown}
      className={className}
    />
  );
}

export function ControlPointsTable({
  controlPoints,
  onControlPointsChange,
  xAxisLabel = 'Year',
  xMin = CHART_DOMAIN_CONFIG.timeline.absoluteMin,
  xMax = CHART_DOMAIN_CONFIG.timeline.absoluteMax,
  interpolationMethod = 'ispline',
  gammaRegularization = 0.01,
  isplineRegularization = 0.05,
  empiricalCdf = [],
}: ControlPointsTableProps) {
  const sorted = [...controlPoints].sort((a, b) => a.x - b.x);

  // Helper to interpolate empirical CDF at a given x
  const getEmpiricalCdfAt = useCallback((x: number): number | null => {
    if (empiricalCdf.length === 0) return null;
    const sortedCdf = [...empiricalCdf].sort((a, b) => a.year - b.year);
    if (x <= sortedCdf[0].year) return sortedCdf[0].cdf;
    if (x >= sortedCdf[sortedCdf.length - 1].year) return sortedCdf[sortedCdf.length - 1].cdf;

    for (let i = 0; i < sortedCdf.length - 1; i++) {
      if (x >= sortedCdf[i].year && x <= sortedCdf[i + 1].year) {
        const t = (x - sortedCdf[i].year) / (sortedCdf[i + 1].year - sortedCdf[i].year);
        return sortedCdf[i].cdf + t * (sortedCdf[i + 1].cdf - sortedCdf[i].cdf);
      }
    }
    return null;
  }, [empiricalCdf]);

  // Compute empirical CDF values at each control point
  const empiricalCdfValues = useMemo(() => {
    const values = new Map<string, number | null>();
    for (const point of controlPoints) {
      values.set(point.id, getEmpiricalCdfAt(point.x));
    }
    return values;
  }, [controlPoints, getEmpiricalCdfAt]);

  // Compute actual CDF values at each control point
  const actualCdfValues = useMemo(() => {
    if (controlPoints.length < 2) return new Map<string, number>();
    const values = new Map<string, number>();
    for (const point of controlPoints) {
      const actualCdf = interpolationMethod === 'gamma'
        ? evaluateGammaMixtureAt(controlPoints, point.x, gammaRegularization)
        : evaluateSplineAt(controlPoints, point.x, isplineRegularization);
      values.set(point.id, actualCdf);
    }
    return values;
  }, [controlPoints, interpolationMethod, gammaRegularization, isplineRegularization]);

  const updatePoint = useCallback(
    (id: string, updates: Partial<ControlPoint>) => {
      const newPoints = controlPoints.map(p =>
        p.id === id ? { ...p, ...updates } : p
      );
      onControlPointsChange(newPoints);
    },
    [controlPoints, onControlPointsChange]
  );

  const deletePoint = useCallback(
    (id: string) => {
      const newPoints = controlPoints.filter(p => p.id !== id);
      onControlPointsChange(newPoints);
    },
    [controlPoints, onControlPointsChange]
  );

  const addPoint = useCallback(() => {
    // Add a new point at the end, extending the x range
    const lastPoint = sorted.length > 0 ? sorted[sorted.length - 1] : null;
    const secondLastPoint = sorted.length > 1 ? sorted[sorted.length - 2] : null;

    let newX: number;
    let newY: number;

    if (lastPoint && secondLastPoint) {
      // Extend by the same step as the last two points
      const step = lastPoint.x - secondLastPoint.x;
      newX = lastPoint.x + step;
      // Increase y slightly, but cap at 1.0
      newY = Math.min(1.0, lastPoint.y + (lastPoint.y - secondLastPoint.y) * 0.5);
      // Ensure y is at least as high as the last point (CDF must be monotonic)
      newY = Math.max(lastPoint.y, newY);
    } else if (lastPoint) {
      // Only one point exists, add some reasonable offset
      newX = lastPoint.x + 5;
      newY = Math.min(1.0, lastPoint.y + 0.1);
    } else {
      // No points exist
      newX = xMin;
      newY = 0.1;
    }

    // Clamp x to valid range
    newX = Math.min(xMax, Math.max(xMin, newX));

    const newPoint: ControlPoint = {
      id: `cp-${Date.now()}-${Math.random().toString(36).substring(2, 11)}`,
      x: Math.round(newX * 100) / 100,
      y: Math.round(newY * 1000) / 1000,
      isLocked: false,
    };

    onControlPointsChange([...controlPoints, newPoint]);
  }, [controlPoints, onControlPointsChange, sorted, xMin, xMax]);

  // Format/parse for Year input
  const formatX = useCallback((x: number) => x.toFixed(2), []);
  const parseX = useCallback((s: string) => {
    const val = parseFloat(s);
    return Math.max(xMin, Math.min(xMax, val));
  }, [xMin, xMax]);

  // Format/parse for CDF % input (convert 0-1 to 0-100 for display)
  const formatY = useCallback((y: number) => (y * 100).toFixed(1), []);
  const parseY = useCallback((s: string) => {
    const val = parseFloat(s);
    // Always treat as percentage (0-100 scale), clamp to valid range
    return Math.max(0, Math.min(1, val / 100));
  }, []);

  return (
    <div className="bg-white rounded-lg shadow p-3 h-fit">
      <div className="flex items-center justify-between mb-2">
        <h4 className="text-sm font-medium text-primary/70">Control Points</h4>
        <button
          onClick={addPoint}
          className="px-2 py-1 text-xs font-medium text-accent hover:bg-accent/10 rounded transition-colors"
          title="Add new control point"
        >
          + Add
        </button>
      </div>

      <div className="overflow-auto max-h-[400px]">
        <table className="w-full text-xs">
          <thead className="sticky top-0 bg-white">
            <tr className="border-b border-gray-200">
              <th className="text-left py-1.5 px-1 font-medium text-primary/60 w-[4.5rem]">
                {xAxisLabel}
              </th>
              <th className="text-left py-1.5 px-1 font-medium text-primary/60 w-[3.2rem]">
                CDF %
              </th>
              <th className="text-left py-1.5 px-0.5 font-medium text-primary/60 w-[2.5rem]" title="Empirical CDF value at this point">
                Emp.
              </th>
              <th className="text-left py-1.5 px-0.5 font-medium text-primary/60 w-[2.5rem]" title="Actual interpolated ATC CDF value">
                ATC
              </th>
              <th className="text-center py-1.5 px-0.5 font-medium text-primary/60 w-7">
                Lock
              </th>
              <th className="w-5"></th>
            </tr>
          </thead>
          <tbody>
            {sorted.map((point, index) => (
              <tr
                key={point.id}
                className="border-b border-gray-100 hover:bg-gray-50"
              >
                <td className="py-0.5 px-1">
                  <EditableNumberInput
                    value={point.x}
                    onChange={x => updatePoint(point.id, { x })}
                    formatValue={formatX}
                    parseValue={parseX}
                    className="w-full px-1 py-0.5 text-[11px] border border-gray-200 rounded focus:outline-none focus:ring-1 focus:ring-accent"
                  />
                </td>
                <td className="py-0.5 px-1">
                  <EditableNumberInput
                    value={point.y}
                    onChange={y => updatePoint(point.id, { y })}
                    formatValue={formatY}
                    parseValue={parseY}
                    className="w-full px-1 py-0.5 text-[11px] border border-gray-200 rounded focus:outline-none focus:ring-1 focus:ring-accent"
                  />
                </td>
                <td className="py-0.5 px-0.5 text-[10px] text-primary/50">
                  {(() => {
                    const empirical = empiricalCdfValues.get(point.id);
                    if (empirical === null || empirical === undefined) return '-';
                    return <span>{(empirical * 100).toFixed(1)}</span>;
                  })()}
                </td>
                <td className="py-0.5 px-0.5 text-[10px] text-primary/50">
                  {(() => {
                    const actual = actualCdfValues.get(point.id);
                    if (actual === undefined) return '-';
                    const diff = Math.abs(actual - point.y) * 100;
                    const color = diff > 1 ? 'text-amber-600' : 'text-primary/50';
                    return <span className={color}>{(actual * 100).toFixed(1)}</span>;
                  })()}
                </td>
                <td className="py-0.5 px-0.5 text-center">
                  <input
                    type="checkbox"
                    checked={point.isLocked}
                    onChange={e =>
                      updatePoint(point.id, { isLocked: e.target.checked })
                    }
                    className="w-3 h-3 accent-accent cursor-pointer"
                    title={point.isLocked ? 'Unlock point' : 'Lock point'}
                  />
                </td>
                <td className="py-0.5 px-0.5 text-center">
                  {/* Don't allow deleting first/last points */}
                  {index > 0 && index < sorted.length - 1 ? (
                    <button
                      onClick={() => deletePoint(point.id)}
                      className="text-red-400 hover:text-red-600 transition-colors"
                      title="Delete point"
                    >
                      &times;
                    </button>
                  ) : (
                    <span className="text-gray-300" title="Cannot delete endpoint">
                      &times;
                    </span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {sorted.length === 0 && (
        <div className="text-xs text-primary/40 text-center py-4">
          No control points
        </div>
      )}
    </div>
  );
}
