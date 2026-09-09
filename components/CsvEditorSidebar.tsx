'use client';

import { useState, useCallback, useRef, useMemo, useEffect, memo } from 'react';
import { createScale, generateLinePath } from '@/utils/chartUtils';
import { DEFAULT_PAUSE_STATE } from '@/utils/urlState';
import { applyPause, type InputTimeSeries } from '@/utils/pauseTransform';

// ── Types ──────────────────────────────────────────────────────────────────────

// InputTimeSeries now lives in @/utils/pauseTransform alongside the pause logic.
// Re-export it here so existing importers (e.g. ProgressChartClient) keep working.
export type { InputTimeSeries } from '@/utils/pauseTransform';

const SERIES_KEYS = ['L_HUMAN', 'inference_H100e', 'experiment_H100e', 'training_H100e'] as const;
type SeriesKey = (typeof SERIES_KEYS)[number];

const SERIES_META: Record<SeriesKey, { label: string; color: string; unit: string }> = {
  L_HUMAN:            { label: 'Human Labor',        color: '#2563eb', unit: 'FTEs' },
  inference_H100e:  { label: 'Inference Compute',  color: '#dc2626', unit: 'H100e' },
  experiment_H100e: { label: 'Experiment Compute',  color: '#059669', unit: 'H100e' },
  training_H100e:   { label: 'Training Compute',    color: '#d97706', unit: 'H100e' },
};

// Optional CSV columns: world compute and the leading AI company's total compute
// (R&D + commercial serving). Display-only — the core model never reads them.
const OPTIONAL_SERIES_KEYS = ['world_H100e', 'total_ai_company_H100e'] as const;
type OptionalSeriesKey = (typeof OPTIONAL_SERIES_KEYS)[number];

// ── Zoomable chart ─────────────────────────────────────────────────────────────

const CHART_H = 130;
const CHART_MARGIN = { top: 8, right: 12, bottom: 24, left: 56 };

// Observe the width of a chart container so charts stay responsive.
const useContainerWidth = (): [React.RefObject<HTMLDivElement | null>, number] => {
  const containerRef = useRef<HTMLDivElement>(null);
  const [containerWidth, setContainerWidth] = useState(320);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const obs = new ResizeObserver(([entry]) => {
      setContainerWidth(entry.contentRect.width);
    });
    obs.observe(el);
    return () => obs.disconnect();
  }, []);

  return [containerRef, containerWidth];
};

// Shared x-axis zoom (wheel) + pan (drag) behavior for the mini charts.
const useXZoomPan = (time: number[], chartWidth: number) => {
  const [xRange, setXRange] = useState<[number, number] | null>(null);
  const isDragging = useRef(false);
  const dragStart = useRef<{ x: number; range: [number, number] } | null>(null);

  const fullXRange: [number, number] = useMemo(() => {
    if (time.length === 0) return [2017, 2050];
    return [time[0], time[time.length - 1]];
  }, [time]);

  const activeXRange: [number, number] = useMemo(
    () => xRange ?? [fullXRange[0], Math.min(fullXRange[1], fullXRange[0] + 40)],
    [xRange, fullXRange]
  );

  const handleWheel = useCallback((e: React.WheelEvent) => {
    e.preventDefault();
    const rect = (e.currentTarget as SVGSVGElement).getBoundingClientRect();
    const mouseX = e.clientX - rect.left - CHART_MARGIN.left;
    const frac = Math.max(0, Math.min(1, mouseX / chartWidth));

    const cur = xRange ?? activeXRange;
    const span = cur[1] - cur[0];
    const zoomFactor = e.deltaY > 0 ? 1.15 : 1 / 1.15;
    const newSpan = Math.max(3, Math.min(fullXRange[1] - fullXRange[0], span * zoomFactor));

    const pivot = cur[0] + frac * span;
    let newMin = pivot - frac * newSpan;
    let newMax = pivot + (1 - frac) * newSpan;

    // Clamp
    if (newMin < fullXRange[0]) { newMax += fullXRange[0] - newMin; newMin = fullXRange[0]; }
    if (newMax > fullXRange[1]) { newMin -= newMax - fullXRange[1]; newMax = fullXRange[1]; }
    newMin = Math.max(fullXRange[0], newMin);
    newMax = Math.min(fullXRange[1], newMax);

    setXRange([newMin, newMax]);
  }, [xRange, activeXRange, chartWidth, fullXRange]);

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    isDragging.current = true;
    dragStart.current = { x: e.clientX, range: [...(xRange ?? activeXRange)] as [number, number] };
    e.preventDefault();
  }, [xRange, activeXRange]);

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (!isDragging.current || !dragStart.current) return;
      const dx = e.clientX - dragStart.current.x;
      const span = dragStart.current.range[1] - dragStart.current.range[0];
      const yearsPx = span / chartWidth;
      const shift = -dx * yearsPx;

      let newMin = dragStart.current.range[0] + shift;
      let newMax = dragStart.current.range[1] + shift;

      if (newMin < fullXRange[0]) { newMax += fullXRange[0] - newMin; newMin = fullXRange[0]; }
      if (newMax > fullXRange[1]) { newMin -= newMax - fullXRange[1]; newMax = fullXRange[1]; }
      newMin = Math.max(fullXRange[0], newMin);
      newMax = Math.min(fullXRange[1], newMax);

      setXRange([newMin, newMax]);
    };

    const handleMouseUp = () => {
      isDragging.current = false;
      dragStart.current = null;
    };

    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleMouseUp);
    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    };
  }, [chartWidth, fullXRange]);

  const handleReset = useCallback(() => setXRange(null), []);

  return { xRange, activeXRange, handleWheel, handleMouseDown, handleReset };
};

// X axis tick positions for a given visible range
const computeXTicks = (activeXRange: [number, number]): number[] => {
  const span = activeXRange[1] - activeXRange[0];
  let step = 5;
  if (span > 80) step = 20;
  else if (span > 40) step = 10;
  else if (span > 15) step = 5;
  else if (span > 5) step = 2;
  else step = 1;
  const ticks: number[] = [];
  const start = Math.ceil(activeXRange[0] / step) * step;
  for (let t = start; t <= activeXRange[1]; t += step) {
    ticks.push(t);
  }
  return ticks;
};

interface MiniChartProps {
  time: number[];
  values: number[];
  label: string;
  color: string;
  unit: string;
}

const MiniChart = memo(({ time, values, label, color, unit }: MiniChartProps) => {
  const [containerRef, containerWidth] = useContainerWidth();

  const chartWidth = containerWidth - CHART_MARGIN.left - CHART_MARGIN.right;
  const chartHeight = CHART_H - CHART_MARGIN.top - CHART_MARGIN.bottom;

  const { xRange, activeXRange, handleWheel, handleMouseDown, handleReset } = useXZoomPan(time, chartWidth);

  // Compute y domain from visible data
  const { yMin, yMax } = useMemo(() => {
    let lo = Infinity, hi = -Infinity;
    for (let i = 0; i < time.length; i++) {
      if (time[i] >= activeXRange[0] && time[i] <= activeXRange[1]) {
        const v = values[i];
        if (v > 0) {
          lo = Math.min(lo, v);
          hi = Math.max(hi, v);
        }
      }
    }
    if (!isFinite(lo)) { lo = 1; hi = 100; }
    // Give some padding in log space
    const logLo = Math.log10(lo);
    const logHi = Math.log10(hi);
    const pad = Math.max(0.2, (logHi - logLo) * 0.08);
    return { yMin: Math.pow(10, logLo - pad), yMax: Math.pow(10, logHi + pad) };
  }, [time, values, activeXRange]);

  const xScale = useMemo(() => createScale({
    domain: activeXRange as [number, number],
    range: [0, chartWidth],
    type: 'linear',
  }), [activeXRange, chartWidth]);

  const yScale = useMemo(() => createScale({
    domain: [yMin, yMax],
    range: [chartHeight, 0],
    type: 'log',
  }), [yMin, yMax, chartHeight]);

  const linePath = useMemo(() => {
    const pts = time.map((t, i) => ({ x: t, y: values[i] > 0 ? values[i] : null }));
    return generateLinePath(pts, xScale, yScale);
  }, [time, values, xScale, yScale]);

  // Y axis ticks (log)
  const yTicks = useMemo(() => {
    const logMin = Math.floor(Math.log10(yMin));
    const logMax = Math.ceil(Math.log10(yMax));
    const ticks: number[] = [];
    for (let e = logMin; e <= logMax; e++) {
      ticks.push(Math.pow(10, e));
    }
    // Limit to 6 ticks
    if (ticks.length > 6) {
      const step = Math.ceil(ticks.length / 6);
      return ticks.filter((_, i) => i % step === 0);
    }
    return ticks;
  }, [yMin, yMax]);

  // X axis ticks
  const xTicks = useMemo(() => computeXTicks(activeXRange), [activeXRange]);

  const formatYTick = (v: number) => {
    if (v === 0) return '0';
    const exp = Math.log10(v);
    if (Number.isInteger(exp)) {
      if (exp === 0) return '1';
      return `10^${exp}`;
    }
    if (v >= 1e6) return `${(v / 1e6).toFixed(0)}M`;
    if (v >= 1e3) return `${(v / 1e3).toFixed(0)}K`;
    return v.toFixed(1);
  };

  return (
    <div ref={containerRef} className="w-full">
      <div className="flex items-center justify-between mb-0.5">
        <span className="text-[10px] font-semibold" style={{ color }}>{label}</span>
        <div className="flex items-center gap-1">
          <span className="text-[9px] text-gray-400">{unit}</span>
          {xRange && (
            <button
              onClick={handleReset}
              className="text-[9px] text-gray-400 hover:text-gray-600 underline ml-1"
            >
              reset zoom
            </button>
          )}
        </div>
      </div>
      <svg
        width={containerWidth}
        height={CHART_H}
        className="select-none cursor-grab active:cursor-grabbing"
        onWheel={handleWheel}
        onMouseDown={handleMouseDown}
      >
        <g transform={`translate(${CHART_MARGIN.left},${CHART_MARGIN.top})`}>
          {/* Gridlines */}
          {yTicks.map((tick) => {
            const y = yScale(tick);
            if (y < 0 || y > chartHeight) return null;
            return (
              <line key={tick} x1={0} x2={chartWidth} y1={y} y2={y}
                stroke="#e5e7eb" strokeWidth={0.5} />
            );
          })}
          {xTicks.map((tick) => {
            const x = xScale(tick);
            if (x < 0 || x > chartWidth) return null;
            return (
              <line key={tick} x1={x} x2={x} y1={0} y2={chartHeight}
                stroke="#e5e7eb" strokeWidth={0.5} />
            );
          })}

          {/* Line */}
          <path d={linePath} fill="none" stroke={color} strokeWidth={1.5} strokeLinejoin="round" />

          {/* Y axis ticks */}
          {yTicks.map((tick) => {
            const y = yScale(tick);
            if (y < -2 || y > chartHeight + 2) return null;
            return (
              <text key={tick} x={-4} y={y} textAnchor="end" dominantBaseline="middle"
                className="text-[9px] fill-gray-500 font-mono">
                {formatYTick(tick)}
              </text>
            );
          })}

          {/* X axis ticks */}
          {xTicks.map((tick) => {
            const x = xScale(tick);
            if (x < 0 || x > chartWidth) return null;
            return (
              <text key={tick} x={x} y={chartHeight + 14} textAnchor="middle"
                className="text-[9px] fill-gray-500 font-mono">
                {tick}
              </text>
            );
          })}
        </g>
      </svg>
    </div>
  );
});

MiniChart.displayName = 'MiniChart';

// ── Editable Table ─────────────────────────────────────────────────────────────

interface EditableTableProps {
  data: InputTimeSeries;
  onChange: (data: InputTimeSeries) => void;
}

const COL_HEADERS: { key: 'time' | SeriesKey; label: string; width: string }[] = [
  { key: 'time',               label: 'Year',     width: 'w-[60px]' },
  { key: 'L_HUMAN',            label: 'Human L.', width: 'w-[80px]' },
  { key: 'inference_H100e',  label: 'Inf. Comp', width: 'w-[80px]' },
  { key: 'experiment_H100e', label: 'Exp. Comp', width: 'w-[80px]' },
  { key: 'training_H100e',   label: 'Train. Comp', width: 'w-[80px]' },
];

const formatCellValue = (val: number, key: string): string => {
  if (key === 'time') return val.toFixed(1);
  if (val >= 1e9) return val.toExponential(3);
  if (val >= 1e6) return (val / 1e6).toFixed(2) + 'M';
  if (val >= 1e3) return (val / 1e3).toFixed(2) + 'K';
  return val.toFixed(2);
};

const EditableTable = memo(({ data, onChange }: EditableTableProps) => {
  const [editCell, setEditCell] = useState<{ row: number; col: string } | null>(null);
  const [editValue, setEditValue] = useState('');
  const inputRef = useRef<HTMLInputElement>(null);

  const rowCount = data.time.length;

  const startEdit = (row: number, col: string) => {
    // COL_HEADERS only lists the required columns, so the array is always present.
    const arr = data[col as keyof InputTimeSeries] as number[];
    setEditCell({ row, col });
    setEditValue(String(arr[row]));
    setTimeout(() => inputRef.current?.select(), 0);
  };

  const commitEdit = () => {
    if (!editCell) return;
    const num = parseFloat(editValue);
    if (isNaN(num)) {
      setEditCell(null);
      return;
    }
    const key = editCell.col as keyof InputTimeSeries;
    const newArr = [...(data[key] as number[])];
    newArr[editCell.row] = num;
    onChange({ ...data, [key]: newArr });
    setEditCell(null);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') { commitEdit(); }
    else if (e.key === 'Escape') { setEditCell(null); }
    else if (e.key === 'Tab') {
      e.preventDefault();
      commitEdit();
      if (!editCell) return;
      // Move to next column
      const colIdx = COL_HEADERS.findIndex(c => c.key === editCell.col);
      if (e.shiftKey) {
        if (colIdx > 0) startEdit(editCell.row, COL_HEADERS[colIdx - 1].key);
        else if (editCell.row > 0) startEdit(editCell.row - 1, COL_HEADERS[COL_HEADERS.length - 1].key);
      } else {
        if (colIdx < COL_HEADERS.length - 1) startEdit(editCell.row, COL_HEADERS[colIdx + 1].key);
        else if (editCell.row < rowCount - 1) startEdit(editCell.row + 1, COL_HEADERS[0].key);
      }
    }
  };

  return (
    <div className="overflow-auto max-h-[300px] border border-gray-200 rounded">
      <table className="text-[10px] font-mono w-full border-collapse">
        <thead className="sticky top-0 z-10 bg-gray-100">
          <tr>
            <th className="px-1 py-0.5 text-left font-semibold text-gray-600 border-b border-gray-200 w-[28px]">#</th>
            {COL_HEADERS.map(h => (
              <th key={h.key} className={`px-1 py-0.5 text-right font-semibold text-gray-600 border-b border-gray-200 ${h.width}`}>
                {h.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {Array.from({ length: rowCount }, (_, i) => (
            <tr key={i} className="hover:bg-blue-50/50 border-b border-gray-100">
              <td className="px-1 py-0 text-gray-400">{i + 1}</td>
              {COL_HEADERS.map(h => {
                const val = (data[h.key as keyof InputTimeSeries] as number[])[i];
                const isEditing = editCell?.row === i && editCell.col === h.key;
                return (
                  <td key={h.key} className="px-1 py-0 text-right">
                    {isEditing ? (
                      <input
                        ref={inputRef}
                        type="text"
                        value={editValue}
                        onChange={(e) => setEditValue(e.target.value)}
                        onBlur={commitEdit}
                        onKeyDown={handleKeyDown}
                        className="w-full text-right text-[10px] font-mono bg-blue-50 border border-blue-300 rounded px-0.5 outline-none"
                        autoFocus
                      />
                    ) : (
                      <span
                        className="cursor-text hover:bg-blue-100 rounded px-0.5 inline-block w-full"
                        onClick={() => startEdit(i, h.key)}
                        title={String(val)}
                      >
                        {formatCellValue(val, h.key)}
                      </span>
                    )}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
});

EditableTable.displayName = 'EditableTable';

// ── Add/Delete Row Controls ────────────────────────────────────────────────────

interface RowControlsProps {
  data: InputTimeSeries;
  onChange: (data: InputTimeSeries) => void;
}

const RowControls = memo(({ data, onChange }: RowControlsProps) => {
  const [newYear, setNewYear] = useState('');

  const addRow = () => {
    const year = parseFloat(newYear);
    if (isNaN(year)) return;

    // Find insert position
    const idx = data.time.findIndex(t => t > year);
    const insertIdx = idx === -1 ? data.time.length : idx;

    // Interpolate values from neighbors
    const interpolate = (arr: number[]) => {
      if (insertIdx === 0) return arr[0] ?? 1;
      if (insertIdx >= arr.length) return arr[arr.length - 1] ?? 1;
      const lo = arr[insertIdx - 1];
      const hi = arr[insertIdx];
      const tLo = data.time[insertIdx - 1];
      const tHi = data.time[insertIdx];
      if (tHi === tLo) return lo;
      const frac = (year - tLo) / (tHi - tLo);
      // Log interpolate for non-time columns
      if (lo > 0 && hi > 0) {
        return Math.exp(Math.log(lo) + frac * (Math.log(hi) - Math.log(lo)));
      }
      return lo + frac * (hi - lo);
    };

    const insertInto = (arr: number[] | undefined): number[] | undefined =>
      arr ? [...arr.slice(0, insertIdx), interpolate(arr), ...arr.slice(insertIdx)] : undefined;

    const newData: InputTimeSeries = {
      time: [...data.time.slice(0, insertIdx), year, ...data.time.slice(insertIdx)],
      L_HUMAN: insertInto(data.L_HUMAN)!,
      inference_H100e: insertInto(data.inference_H100e)!,
      experiment_H100e: insertInto(data.experiment_H100e)!,
      training_H100e: insertInto(data.training_H100e)!,
      world_H100e: insertInto(data.world_H100e),
      total_ai_company_H100e: insertInto(data.total_ai_company_H100e),
    };
    onChange(newData);
    setNewYear('');
  };

  const deleteLastRow = () => {
    if (data.time.length <= 2) return;
    const newData: InputTimeSeries = {
      time: data.time.slice(0, -1),
      L_HUMAN: data.L_HUMAN.slice(0, -1),
      inference_H100e: data.inference_H100e.slice(0, -1),
      experiment_H100e: data.experiment_H100e.slice(0, -1),
      training_H100e: data.training_H100e.slice(0, -1),
      world_H100e: data.world_H100e?.slice(0, -1),
      total_ai_company_H100e: data.total_ai_company_H100e?.slice(0, -1),
    };
    onChange(newData);
  };

  return (
    <div className="flex items-center gap-2 text-[10px]">
      <input
        type="number"
        placeholder="Year"
        value={newYear}
        onChange={e => setNewYear(e.target.value)}
        className="w-16 px-1 py-0.5 border border-gray-300 rounded text-[10px] font-mono"
        step={1}
      />
      <button
        onClick={addRow}
        className="px-2 py-0.5 bg-gray-100 hover:bg-gray-200 border border-gray-300 rounded text-[10px] font-semibold"
      >
        Add Row
      </button>
      <button
        onClick={deleteLastRow}
        disabled={data.time.length <= 2}
        className="px-2 py-0.5 bg-gray-100 hover:bg-gray-200 border border-gray-300 rounded text-[10px] font-semibold disabled:opacity-40"
      >
        Delete Last
      </button>
      <span className="text-gray-400 ml-auto">{data.time.length} rows</span>
    </div>
  );
});

RowControls.displayName = 'RowControls';

// ── Pause Controls UI ──────────────────────────────────────────────────────────

interface PauseControlsProps {
  enabled: boolean;
  onToggle: (v: boolean) => void;
  pauseMode: 'year' | 'milestone';
  onPauseModeChange: (v: 'year' | 'milestone') => void;
  pauseYear: number;
  onPauseYearChange: (v: number) => void;
  pauseMilestone: string;
  onPauseMilestoneChange: (v: string) => void;
  pauseMilestoneOptions: PauseMilestoneOption[];
  effectivePauseYear: number | null;
  experimentReduction: number;
  onExperimentReductionChange: (v: number) => void;
  inferenceReduction: number;
  onInferenceReductionChange: (v: number) => void;
  laborReduction: number;
  onLaborReductionChange: (v: number) => void;
  trainingReduction: number;
  onTrainingReductionChange: (v: number) => void;
}

const LogSlider = memo(({
  label,
  value,
  onChange,
  min,
  max,
  disabled,
}: {
  label: string;
  value: number;
  onChange: (v: number) => void;
  min: number;
  max: number;
  disabled?: boolean;
}) => {
  const logMin = Math.log10(min);
  const logMax = Math.log10(max);
  const logVal = Math.log10(value);
  const pct = ((logVal - logMin) / (logMax - logMin)) * 100;

  return (
    <label className={`flex flex-col gap-0.5 ${disabled ? 'opacity-40' : ''}`}>
      <div className="flex items-center justify-between">
        <span className="text-[10px] text-gray-600">{label}</span>
        <span className="text-[10px] font-mono text-gray-800 font-semibold">{value >= 10 ? Math.round(value).toLocaleString() : value.toFixed(1)}x</span>
      </div>
      <input
        type="range"
        min={logMin * 1000}
        max={logMax * 1000}
        value={logVal * 1000}
        onChange={(e) => onChange(Math.pow(10, parseInt(e.target.value) / 1000))}
        disabled={disabled}
        className="w-full h-1 rounded-full appearance-none cursor-pointer accent-red-600"
        style={{
          background: disabled
            ? '#e5e7eb'
            : `linear-gradient(to right, #dc2626 0%, #dc2626 ${pct}%, #e5e7eb ${pct}%, #e5e7eb 100%)`,
        }}
      />
    </label>
  );
});

LogSlider.displayName = 'LogSlider';

const PauseControls = memo(({
  enabled,
  onToggle,
  pauseMode,
  onPauseModeChange,
  pauseYear,
  onPauseYearChange,
  pauseMilestone,
  onPauseMilestoneChange,
  pauseMilestoneOptions,
  effectivePauseYear,
  experimentReduction,
  onExperimentReductionChange,
  inferenceReduction,
  onInferenceReductionChange,
  laborReduction,
  onLaborReductionChange,
  trainingReduction,
  onTrainingReductionChange,
}: PauseControlsProps) => {
  const yearPct = ((pauseYear - 2025) / (2035 - 2025)) * 100;
  const selectedReached = pauseMode === 'milestone' ? effectivePauseYear !== null : true;

  return (
    <div className={`rounded border ${enabled ? 'border-red-300 bg-red-50/50' : 'border-gray-200 bg-gray-50/50'} transition-colors`}>
      {/* Toggle header */}
      <label className="flex items-center gap-2 px-3 py-2 cursor-pointer select-none">
        <input
          type="checkbox"
          checked={enabled}
          onChange={(e) => onToggle(e.target.checked)}
          className="accent-red-600 w-3.5 h-3.5"
        />
        <span className="text-[11px] font-semibold text-gray-800">Pause Mode</span>
        {enabled && <span className="text-[9px] text-red-600 font-semibold ml-auto">ACTIVE</span>}
      </label>

      {/* Sliders */}
      {enabled && (
        <div className="px-3 pb-3 space-y-2 border-t border-red-200/60 pt-2">
          {/* Pause trigger mode toggle */}
          <div className="flex rounded border border-gray-300 overflow-hidden text-[10px] font-semibold">
            <button
              type="button"
              onClick={() => onPauseModeChange('year')}
              className={`flex-1 px-2 py-1 transition-colors ${pauseMode === 'year' ? 'bg-red-600 text-white' : 'bg-white text-gray-600 hover:bg-gray-50'}`}
            >
              At year
            </button>
            <button
              type="button"
              onClick={() => onPauseModeChange('milestone')}
              className={`flex-1 px-2 py-1 border-l border-gray-300 transition-colors ${pauseMode === 'milestone' ? 'bg-red-600 text-white' : 'bg-white text-gray-600 hover:bg-gray-50'}`}
            >
              At milestone
            </button>
          </div>

          {pauseMode === 'year' ? (
            /* Pause year */
            <label className="flex flex-col gap-0.5">
              <div className="flex items-center justify-between">
                <span className="text-[10px] text-gray-600">Pause year</span>
                <span className="text-[10px] font-mono text-gray-800 font-semibold">{pauseYear.toFixed(1)}</span>
              </div>
              <input
                type="range"
                min={2025 * 10}
                max={2035 * 10}
                value={Math.round(pauseYear * 10)}
                onChange={(e) => onPauseYearChange(parseInt(e.target.value) / 10)}
                className="w-full h-1 rounded-full appearance-none cursor-pointer accent-red-600"
                style={{
                  background: `linear-gradient(to right, #dc2626 0%, #dc2626 ${yearPct}%, #e5e7eb ${yearPct}%, #e5e7eb 100%)`,
                }}
              />
            </label>
          ) : (
            /* Pause at milestone */
            <label className="flex flex-col gap-0.5">
              <div className="flex items-center justify-between">
                <span className="text-[10px] text-gray-600">Pause at milestone</span>
                <span className="text-[10px] font-mono text-gray-800 font-semibold">
                  {effectivePauseYear !== null ? `~${effectivePauseYear.toFixed(1)}` : 'not reached'}
                </span>
              </div>
              <select
                value={pauseMilestone}
                onChange={(e) => onPauseMilestoneChange(e.target.value)}
                className="w-full text-[11px] border border-gray-300 rounded px-1.5 py-1 bg-white text-gray-800 cursor-pointer focus:outline-none focus:ring-1 focus:ring-red-400"
              >
                {pauseMilestoneOptions.map((m) => (
                  <option key={m.key} value={m.key} disabled={m.year === null}>
                    {m.name} — {m.fullName}{m.year === null ? ' (not reached)' : ` (${m.year.toFixed(1)})`}
                  </option>
                ))}
              </select>
              {!selectedReached && (
                <span className="text-[9px] text-amber-600">
                  This milestone isn&apos;t reached in the current run — no pause applied. Pick an earlier milestone.
                </span>
              )}
            </label>
          )}

          <LogSlider
            label="Training compute reduction"
            value={trainingReduction}
            onChange={onTrainingReductionChange}
            min={1}
            max={10000}
          />
          <LogSlider
            label="Experiment compute reduction"
            value={experimentReduction}
            onChange={onExperimentReductionChange}
            min={1}
            max={10000}
          />
          <LogSlider
            label="Inference compute reduction"
            value={inferenceReduction}
            onChange={onInferenceReductionChange}
            min={1}
            max={10000}
          />
          <LogSlider
            label="Human labor reduction"
            value={laborReduction}
            onChange={onLaborReductionChange}
            min={1}
            max={10000}
          />
        </div>
      )}
    </div>
  );
});

PauseControls.displayName = 'PauseControls';

// ── Main Sidebar Component ─────────────────────────────────────────────────────

// A milestone the user can pause at, with its year from the current trajectory
// (null when the milestone is not reached within the simulation).
export interface PauseMilestoneOption {
  key: string;
  name: string;
  fullName: string;
  year: number | null;
}

interface CsvEditorSidebarProps {
  isOpen: boolean;
  onClose: () => void;
  data: InputTimeSeries | null;
  defaultData: InputTimeSeries | null;
  onChange: (data: InputTimeSeries) => void;
  onReset: () => void;
  pauseEnabled: boolean;
  onPauseEnabledChange: (v: boolean) => void;
  pauseMode: 'year' | 'milestone';
  onPauseModeChange: (v: 'year' | 'milestone') => void;
  pauseYear: number;
  onPauseYearChange: (v: number) => void;
  pauseMilestone: string;
  onPauseMilestoneChange: (v: string) => void;
  pauseMilestoneOptions: PauseMilestoneOption[];
  // Solver output times (years) of the current trajectory, used to snap a
  // milestone pause to the first datapoint after the milestone is reached.
  trajectoryTimes: number[];
  experimentReduction: number;
  onExperimentReductionChange: (v: number) => void;
  inferenceReduction: number;
  onInferenceReductionChange: (v: number) => void;
  laborReduction: number;
  onLaborReductionChange: (v: number) => void;
  trainingReduction: number;
  onTrainingReductionChange: (v: number) => void;
}

type ViewTab = 'charts' | 'table';

export const CsvEditorSidebar = memo(({
  isOpen,
  onClose,
  data,
  defaultData,
  onChange,
  onReset,
  pauseEnabled,
  onPauseEnabledChange,
  pauseMode,
  onPauseModeChange,
  pauseYear,
  onPauseYearChange,
  pauseMilestone,
  onPauseMilestoneChange,
  pauseMilestoneOptions = [],
  trajectoryTimes = [],
  experimentReduction,
  onExperimentReductionChange,
  inferenceReduction,
  onInferenceReductionChange,
  laborReduction,
  onLaborReductionChange,
  trainingReduction,
  onTrainingReductionChange,
}: CsvEditorSidebarProps) => {
  const [activeTab, setActiveTab] = useState<ViewTab>('charts');
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Base data is the data before pause is applied.
  // When pause is off, baseData IS the data the parent passed, so it is derived
  // rather than mirrored into state by an effect. Only while pause is on do we
  // hold our own copy: `data` is then the paused output, so the pre-pause source
  // has to be frozen locally (seeded in handlePauseToggle below).
  const [pausedBaseData, setPausedBaseData] = useState<InputTimeSeries | null>(null);
  const baseData = pauseEnabled ? pausedBaseData : data;

  const selectedMilestone = useMemo(
    () => pauseMilestoneOptions.find(m => m.key === pauseMilestone) ?? null,
    [pauseMilestoneOptions, pauseMilestone]
  );

  // The year at which the pause actually takes effect. In 'year' mode this is the
  // slider value. In 'milestone' mode we snap to the first solver output datapoint
  // STRICTLY AFTER the milestone's achievement time, so the milestone is reached
  // under original (un-reduced) inputs and its date is preserved exactly. Because
  // the milestone's time is therefore invariant to the pause, this is a one-step
  // fixed point — no chasing or oscillation. null => milestone not reached (or no
  // later datapoint) => no pause applied.
  const effectivePauseYear: number | null = useMemo(() => {
    if (pauseMode !== 'milestone') return pauseYear;
    const my = selectedMilestone?.year ?? null;
    if (my === null) return null;
    let next: number | null = null;
    for (const t of trajectoryTimes) {
      if (t > my) { next = t; break; }
    }
    return next;
  }, [pauseMode, pauseYear, selectedMilestone, trajectoryTimes]);

  // Apply pause whenever pause params or baseData change
  useEffect(() => {
    if (!pauseEnabled || !baseData) return;
    // Milestone mode with an unreached milestone: leave inputs unpaused.
    if (effectivePauseYear === null) {
      onChange(baseData);
      return;
    }
    const paused = applyPause(baseData, effectivePauseYear, experimentReduction, inferenceReduction, laborReduction, trainingReduction);
    onChange(paused);
  }, [pauseEnabled, effectivePauseYear, experimentReduction, inferenceReduction, laborReduction, trainingReduction, baseData, onChange]);

  // Freeze the pre-pause source on the way in; restore it on the way out.
  const handlePauseToggle = useCallback((enabled: boolean) => {
    if (enabled) {
      // Capture what the parent currently holds before the apply-pause effect
      // starts replacing `data` with paused output. Batched with the parent's
      // state change, so the next render sees both at once.
      setPausedBaseData(data);
    } else if (baseData) {
      // Restore the pre-pause data
      onChange(baseData);
    }
    onPauseEnabledChange(enabled);
  }, [data, baseData, onChange, onPauseEnabledChange]);

  // When the user edits cells or imports a CSV while pause is on,
  // we update baseData (not the paused output) so the pause can be reapplied.
  const handleBaseDataChange = useCallback((newData: InputTimeSeries) => {
    if (pauseEnabled) {
      // The apply-pause effect will recompute the paused version from this.
      setPausedBaseData(newData);
    } else {
      // Pause is off, so baseData tracks the parent; just propagate.
      onChange(newData);
    }
  }, [pauseEnabled, onChange]);

  const handleImportCSV = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
      const text = event.target?.result as string;
      if (!text) return;

      const lines = text.trim().split('\n');
      if (lines.length < 2) return;

      const headers = lines[0].split(',').map(h => h.trim());
      const requiredCols = ['time', 'L_HUMAN', 'inference_H100e', 'experiment_H100e', 'training_H100e'];
      const colIdx: Record<string, number> = {};
      for (const col of requiredCols) {
        const idx = headers.indexOf(col);
        if (idx === -1) {
          alert(`CSV missing required column: ${col}`);
          return;
        }
        colIdx[col] = idx;
      }

      const result: InputTimeSeries = {
        time: [], L_HUMAN: [], inference_H100e: [], experiment_H100e: [], training_H100e: [],
      };

      // Optional columns (world compute, leading AI company total): imported when
      // present AND parseable on every kept row, silently dropped otherwise.
      const optIdx: Partial<Record<OptionalSeriesKey, number>> = {};
      const optVals: Record<OptionalSeriesKey, number[]> = { world_H100e: [], total_ai_company_H100e: [] };
      let optValid: Record<OptionalSeriesKey, boolean> = { world_H100e: true, total_ai_company_H100e: true };
      for (const col of OPTIONAL_SERIES_KEYS) {
        const idx = headers.indexOf(col);
        if (idx !== -1) optIdx[col] = idx;
      }

      for (let i = 1; i < lines.length; i++) {
        const vals = lines[i].split(',');
        if (vals.length < headers.length) continue;
        let valid = true;
        for (const col of requiredCols) {
          const v = parseFloat(vals[colIdx[col]]);
          if (isNaN(v)) { valid = false; break; }
        }
        if (!valid) continue;
        for (const col of requiredCols) {
          (result[col as keyof InputTimeSeries] as number[]).push(parseFloat(vals[colIdx[col]]));
        }
        for (const col of OPTIONAL_SERIES_KEYS) {
          const idx = optIdx[col];
          if (idx === undefined) continue;
          const v = parseFloat(vals[idx]);
          if (isNaN(v)) optValid = { ...optValid, [col]: false };
          else optVals[col].push(v);
        }
      }

      if (result.time.length < 2) {
        alert('CSV must have at least 2 data rows');
        return;
      }

      for (const col of OPTIONAL_SERIES_KEYS) {
        if (optIdx[col] !== undefined && optValid[col] && optVals[col].length === result.time.length) {
          result[col] = optVals[col];
        }
      }

      handleBaseDataChange(result);
    };
    reader.readAsText(file);
    // Reset so the same file can be imported again
    e.target.value = '';
  }, [handleBaseDataChange]);

  const handleExportCSV = useCallback(() => {
    // Always compute the effective data fresh to avoid stale-closure issues
    const exportData = pauseEnabled && baseData && effectivePauseYear !== null
      ? applyPause(baseData, effectivePauseYear, experimentReduction, inferenceReduction, laborReduction, trainingReduction)
      : (baseData ?? data);
    if (!exportData) return;
    const optionalKeys = OPTIONAL_SERIES_KEYS.filter(
      (k) => exportData[k] && exportData[k]!.length === exportData.time.length
    );
    const headers = ['time', 'L_HUMAN', 'inference_H100e', 'experiment_H100e', 'training_H100e', ...optionalKeys].join(',');
    const rows = exportData.time.map((t: number, i: number) =>
      [t, exportData.L_HUMAN[i], exportData.inference_H100e[i], exportData.experiment_H100e[i], exportData.training_H100e[i],
       ...optionalKeys.map((k) => exportData[k]![i])].join(',')
    );
    const csv = [headers, ...rows].join('\n');
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `input_data_custom_${Date.now()}.csv`;
    link.click();
    URL.revokeObjectURL(url);
  }, [data, baseData, pauseEnabled, effectivePauseYear, experimentReduction, inferenceReduction, laborReduction, trainingReduction]);

  const handleReset = useCallback(() => {
    onPauseEnabledChange(DEFAULT_PAUSE_STATE.pauseEnabled);
    onPauseModeChange(DEFAULT_PAUSE_STATE.pauseMode);
    onPauseYearChange(DEFAULT_PAUSE_STATE.pauseYear);
    onPauseMilestoneChange(DEFAULT_PAUSE_STATE.pauseMilestone);
    onExperimentReductionChange(DEFAULT_PAUSE_STATE.experimentReduction);
    onInferenceReductionChange(DEFAULT_PAUSE_STATE.inferenceReduction);
    onLaborReductionChange(DEFAULT_PAUSE_STATE.laborReduction);
    onTrainingReductionChange(DEFAULT_PAUSE_STATE.trainingReduction);
    // Keeps parity if DEFAULT_PAUSE_STATE ever defaults pause on; with pause off
    // baseData follows the parent, which onReset() returns to defaultData.
    setPausedBaseData(defaultData);
    onReset();
  }, [defaultData, onReset, onPauseEnabledChange, onPauseModeChange, onPauseYearChange, onPauseMilestoneChange, onExperimentReductionChange, onInferenceReductionChange, onLaborReductionChange, onTrainingReductionChange]);

  const isModified = useMemo(() => {
    if (!data || !defaultData) return false;
    if (data.time.length !== defaultData.time.length) return true;
    for (let i = 0; i < data.time.length; i++) {
      if (data.time[i] !== defaultData.time[i]) return true;
      if (data.L_HUMAN[i] !== defaultData.L_HUMAN[i]) return true;
      if (data.inference_H100e[i] !== defaultData.inference_H100e[i]) return true;
      if (data.experiment_H100e[i] !== defaultData.experiment_H100e[i]) return true;
      if (data.training_H100e[i] !== defaultData.training_H100e[i]) return true;
    }
    return false;
  }, [data, defaultData]);

  if (!isOpen) return null;

  // The data shown in the table/charts is the effective (possibly paused) data
  const displayData = data;

  return (
    <div
      className="fixed inset-y-0 right-0 z-[60] flex"
      style={{ pointerEvents: 'none' }}
    >
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/30"
        style={{ pointerEvents: 'auto' }}
        onClick={onClose}
      />

      {/* Panel */}
      <div
        className="relative ml-auto h-full w-[28rem] max-w-[95vw] bg-white shadow-2xl flex flex-col overflow-hidden"
        style={{
          pointerEvents: 'auto',
          animation: 'slideInRight 0.2s ease-out',
        }}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-gray-200 bg-gray-50 shrink-0">
          <div>
            <h2 className="text-sm font-bold text-gray-900">Input Time Series</h2>
            <p className="text-[10px] text-gray-500 mt-0.5">
              Edit the data driving the model
              {isModified && <span className="ml-1 text-amber-600 font-semibold">(modified)</span>}
            </p>
          </div>
          <button
            onClick={onClose}
            className="p-1 hover:bg-gray-200 rounded transition-colors text-gray-500 hover:text-gray-700"
            aria-label="Close"
          >
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
              <path d="M4 4l8 8M12 4l-8 8" />
            </svg>
          </button>
        </div>

        {/* Toolbar */}
        <div className="flex items-center gap-2 px-4 py-2 border-b border-gray-100 shrink-0">
          <input
            ref={fileInputRef}
            type="file"
            accept=".csv"
            className="hidden"
            onChange={handleImportCSV}
          />
          <button
            onClick={() => fileInputRef.current?.click()}
            className="px-2.5 py-1 text-[10px] font-semibold uppercase tracking-wider bg-[#2A623D] text-white rounded hover:bg-[#245632] transition-colors"
          >
            Import CSV
          </button>
          <button
            onClick={handleExportCSV}
            disabled={!displayData}
            className="px-2.5 py-1 text-[10px] font-semibold uppercase tracking-wider border border-gray-300 rounded hover:bg-gray-100 transition-colors disabled:opacity-40"
          >
            Export CSV
          </button>
          <button
            onClick={handleReset}
            disabled={!isModified && !pauseEnabled}
            className="px-2.5 py-1 text-[10px] font-semibold uppercase tracking-wider text-amber-700 border border-amber-300 rounded hover:bg-amber-50 transition-colors disabled:opacity-40"
          >
            Reset
          </button>

          {/* Tab toggle */}
          <div className="ml-auto flex rounded border border-gray-300 overflow-hidden">
            <button
              onClick={() => setActiveTab('charts')}
              className={`px-2 py-0.5 text-[10px] font-semibold transition-colors ${
                activeTab === 'charts' ? 'bg-gray-800 text-white' : 'bg-white text-gray-600 hover:bg-gray-50'
              }`}
            >
              Charts
            </button>
            <button
              onClick={() => setActiveTab('table')}
              className={`px-2 py-0.5 text-[10px] font-semibold transition-colors ${
                activeTab === 'table' ? 'bg-gray-800 text-white' : 'bg-white text-gray-600 hover:bg-gray-50'
              }`}
            >
              Table
            </button>
          </div>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto px-4 py-3">
          {/* Pause controls — always visible at top */}
          <div className="mb-3">
            <PauseControls
              enabled={pauseEnabled}
              onToggle={handlePauseToggle}
              pauseMode={pauseMode}
              onPauseModeChange={onPauseModeChange}
              pauseYear={pauseYear}
              onPauseYearChange={onPauseYearChange}
              pauseMilestone={pauseMilestone}
              onPauseMilestoneChange={onPauseMilestoneChange}
              pauseMilestoneOptions={pauseMilestoneOptions}
              effectivePauseYear={effectivePauseYear}
              experimentReduction={experimentReduction}
              onExperimentReductionChange={onExperimentReductionChange}
              inferenceReduction={inferenceReduction}
              onInferenceReductionChange={onInferenceReductionChange}
              laborReduction={laborReduction}
              onLaborReductionChange={onLaborReductionChange}
              trainingReduction={trainingReduction}
              onTrainingReductionChange={onTrainingReductionChange}
            />
          </div>

          {!displayData ? (
            <div className="flex items-center justify-center h-40 text-sm text-gray-400">
              Loading data...
            </div>
          ) : activeTab === 'charts' ? (
            <div className="space-y-3">
              {SERIES_KEYS.map((key) => (
                <MiniChart
                  key={key}
                  time={displayData.time}
                  values={displayData[key]}
                  label={SERIES_META[key].label}
                  color={SERIES_META[key].color}
                  unit={SERIES_META[key].unit}
                />
              ))}
              <p className="text-[9px] text-gray-400 text-center pt-1">
                Scroll to zoom, drag to pan
              </p>
            </div>
          ) : (
            <div className="space-y-2">
              <RowControls data={baseData ?? displayData} onChange={handleBaseDataChange} />
              <EditableTable data={baseData ?? displayData} onChange={handleBaseDataChange} />
            </div>
          )}
        </div>
      </div>

      <style>{`
        @keyframes slideInRight {
          from { transform: translateX(100%); }
          to { transform: translateX(0); }
        }
      `}</style>
    </div>
  );
});

CsvEditorSidebar.displayName = 'CsvEditorSidebar';
