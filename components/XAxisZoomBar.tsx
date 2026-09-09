'use client';

import React, { useCallback, useRef, useState } from 'react';

/**
 * Brush-style range control for zooming a chart's x-axis.
 *
 * Shows the full data extent as a slim track with a highlighted window over
 * it. Drag a grip to resize the window, drag the window body to pan it,
 * click anywhere on the track to jump the nearest edge there, and
 * double-click (or press Reset) to zoom all the way back out.
 *
 * Reports `null` instead of a range when the window covers the full extent,
 * so callers can fall back to their own auto-computed domain.
 */
interface XAxisZoomBarProps {
  /** Full extent of the data along the x-axis. */
  min: number;
  max: number;
  /** Current zoom window, or null when showing the full extent. */
  value: [number, number] | null;
  onChange: (value: [number, number] | null) => void;
  /** Snap increment for the window edges (e.g. 1/12 for calendar months). */
  step?: number;
  /** Smallest allowed window span. Defaults to the step. */
  minSpan?: number;
  /** Formats the floating value bubble shown while dragging, and ARIA values. */
  formatValue?: (value: number) => string;
  /** Accessible name for the grips; never rendered visually. */
  label?: string;
  className?: string;
}

type DragState = {
  kind: 'lo' | 'hi' | 'pan';
  startClientX: number;
  startLo: number;
  startHi: number;
};

export function XAxisZoomBar({
  min,
  max,
  value,
  onChange,
  step = 1,
  minSpan,
  formatValue = (v) => v.toString(),
  label = 'Zoom x-axis',
  className = '',
}: XAxisZoomBarProps) {
  const trackRef = useRef<HTMLDivElement>(null);
  const [drag, setDrag] = useState<DragState | null>(null);

  const range = Math.max(max - min, 1e-9);
  const effMinSpan = Math.max(minSpan ?? step, step);

  const clampVal = useCallback(
    (v: number) => Math.min(max, Math.max(min, v)),
    [min, max]
  );
  // Snap relative to min so both endpoints are always exactly reachable
  const snap = useCallback(
    (v: number) => clampVal(min + Math.round((v - min) / step) * step),
    [min, step, clampVal]
  );

  const lo = value ? clampVal(value[0]) : min;
  const hi = value ? clampVal(value[1]) : max;
  const isZoomed = value !== null;

  const loPct = ((lo - min) / range) * 100;
  const hiPct = ((hi - min) / range) * 100;

  // Collapse a full-extent window back to null so the chart's auto domain
  // (with its own padding rules) takes over again.
  const commit = useCallback(
    (newLo: number, newHi: number) => {
      const eps = step / 2;
      if (newLo <= min + eps && newHi >= max - eps) {
        onChange(null);
      } else {
        onChange([newLo, newHi]);
      }
    },
    [min, max, step, onChange]
  );

  const valueFromClientX = useCallback(
    (clientX: number): number => {
      const rect = trackRef.current?.getBoundingClientRect();
      if (!rect || rect.width === 0) return min;
      const t = Math.min(1, Math.max(0, (clientX - rect.left) / rect.width));
      return min + t * range;
    },
    [min, range]
  );

  const beginDrag = useCallback(
    (kind: DragState['kind']) => (e: React.PointerEvent) => {
      e.preventDefault();
      e.stopPropagation();
      trackRef.current?.setPointerCapture(e.pointerId);
      setDrag({ kind, startClientX: e.clientX, startLo: lo, startHi: hi });
    },
    [lo, hi]
  );

  // Pointer capture is taken on the track element, so once a drag starts all
  // move/up events arrive here regardless of where the pointer wanders.
  const handlePointerMove = useCallback(
    (e: React.PointerEvent) => {
      if (!drag) return;
      if (drag.kind === 'lo') {
        const newLo = Math.min(snap(valueFromClientX(e.clientX)), hi - effMinSpan);
        commit(Math.max(min, newLo), hi);
      } else if (drag.kind === 'hi') {
        const newHi = Math.max(snap(valueFromClientX(e.clientX)), lo + effMinSpan);
        commit(lo, Math.min(max, newHi));
      } else {
        const rect = trackRef.current?.getBoundingClientRect();
        const perPx = rect && rect.width > 0 ? range / rect.width : 0;
        const span = drag.startHi - drag.startLo;
        const shifted = snap(drag.startLo + (e.clientX - drag.startClientX) * perPx);
        const newLo = Math.min(Math.max(shifted, min), max - span);
        commit(newLo, newLo + span);
      }
    },
    [drag, snap, valueFromClientX, commit, min, max, lo, hi, effMinSpan, range]
  );

  const endDrag = useCallback(() => setDrag(null), []);

  // Clicking the bare track jumps the nearest edge there and keeps dragging it
  const handleTrackPointerDown = useCallback(
    (e: React.PointerEvent) => {
      e.preventDefault();
      const v = snap(valueFromClientX(e.clientX));
      trackRef.current?.setPointerCapture(e.pointerId);
      if (Math.abs(v - lo) <= Math.abs(v - hi)) {
        const newLo = Math.max(min, Math.min(v, hi - effMinSpan));
        commit(newLo, hi);
        setDrag({ kind: 'lo', startClientX: e.clientX, startLo: newLo, startHi: hi });
      } else {
        const newHi = Math.min(max, Math.max(v, lo + effMinSpan));
        commit(lo, newHi);
        setDrag({ kind: 'hi', startClientX: e.clientX, startLo: lo, startHi: newHi });
      }
    },
    [snap, valueFromClientX, commit, min, max, lo, hi, effMinSpan]
  );

  const handleGripKeyDown = useCallback(
    (which: 'lo' | 'hi') => (e: React.KeyboardEvent) => {
      let delta: number | null = null;
      if (e.key === 'ArrowLeft' || e.key === 'ArrowDown') delta = -step;
      else if (e.key === 'ArrowRight' || e.key === 'ArrowUp') delta = step;
      else if (e.key === 'Home') delta = -Infinity;
      else if (e.key === 'End') delta = Infinity;
      if (delta === null) return;
      e.preventDefault();
      const scaled = Number.isFinite(delta) && e.shiftKey ? delta * 10 : delta;
      if (which === 'lo') {
        const newLo = Math.max(min, Math.min(snap(lo + scaled), hi - effMinSpan));
        commit(newLo, hi);
      } else {
        const newHi = Math.min(max, Math.max(snap(hi + scaled), lo + effMinSpan));
        commit(lo, newHi);
      }
    },
    [step, snap, commit, min, max, lo, hi, effMinSpan]
  );

  const idleTransition = drag ? '' : 'transition-all duration-150 ease-out';

  const gripClass = (active: boolean) =>
    `group absolute top-1/2 z-10 flex h-[18px] w-[11px] -translate-x-1/2 -translate-y-1/2 items-center justify-center rounded-[5px] border bg-white outline-none cursor-ew-resize ${
      active
        ? 'border-accent shadow ring-2 ring-accent/25'
        : 'border-gray-300 shadow-sm hover:border-accent/70 hover:shadow focus-visible:border-accent focus-visible:ring-2 focus-visible:ring-accent/30'
    } ${idleTransition}`;

  return (
    <div className={`select-none ${className}`}>
      {/* Interactive track */}
      <div
        ref={trackRef}
        className="relative h-7 cursor-pointer touch-none"
        onPointerDown={handleTrackPointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={endDrag}
        onPointerCancel={endDrag}
        onDoubleClick={() => onChange(null)}
      >
        {/* Full-extent rail */}
        <div className="absolute left-0 right-0 top-1/2 h-[6px] -translate-y-1/2 rounded-full bg-gray-200" />

        {/* Selected window */}
        <div
          className={`absolute top-1/2 h-[6px] -translate-y-1/2 rounded-full ${
            isZoomed ? 'bg-accent' : 'bg-accent/30'
          } ${idleTransition}`}
          style={{ left: `${loPct}%`, width: `${Math.max(hiPct - loPct, 0)}%` }}
        />

        {/* Pan hit area (only useful when zoomed; at full extent it would
            cover the whole track and swallow the jump-to-click gesture) */}
        {isZoomed && (
          <div
            className={`absolute top-1/2 h-5 -translate-y-1/2 ${
              drag?.kind === 'pan' ? 'cursor-grabbing' : 'cursor-grab'
            }`}
            style={{ left: `${loPct}%`, width: `${Math.max(hiPct - loPct, 0)}%` }}
            onPointerDown={beginDrag('pan')}
          />
        )}

        {/* Grips */}
        <div
          role="slider"
          tabIndex={0}
          aria-label={`${label}: start`}
          aria-valuemin={min}
          aria-valuemax={max}
          aria-valuenow={lo}
          aria-valuetext={formatValue(lo)}
          className={gripClass(drag?.kind === 'lo')}
          style={{ left: `${loPct}%` }}
          onPointerDown={beginDrag('lo')}
          onKeyDown={handleGripKeyDown('lo')}
        >
          <span className="h-2 w-[2px] rounded-full bg-gray-400 group-hover:bg-accent/70" />
        </div>
        <div
          role="slider"
          tabIndex={0}
          aria-label={`${label}: end`}
          aria-valuemin={min}
          aria-valuemax={max}
          aria-valuenow={hi}
          aria-valuetext={formatValue(hi)}
          className={gripClass(drag?.kind === 'hi')}
          style={{ left: `${hiPct}%` }}
          onPointerDown={beginDrag('hi')}
          onKeyDown={handleGripKeyDown('hi')}
        >
          <span className="h-2 w-[2px] rounded-full bg-gray-400 group-hover:bg-accent/70" />
        </div>

        {/* Floating value bubble while dragging */}
        {drag && (
          <div
            className="pointer-events-none absolute -top-5 -translate-x-1/2 whitespace-nowrap rounded bg-gray-900/90 px-1.5 py-0.5 text-[10px] font-medium tabular-nums text-white shadow"
            style={{
              left: `${drag.kind === 'lo' ? loPct : drag.kind === 'hi' ? hiPct : (loPct + hiPct) / 2}%`,
            }}
          >
            {drag.kind === 'lo'
              ? formatValue(lo)
              : drag.kind === 'hi'
                ? formatValue(hi)
                : `${formatValue(lo)} – ${formatValue(hi)}`}
          </div>
        )}
      </div>
    </div>
  );
}
