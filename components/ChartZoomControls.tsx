'use client';

import React, { useCallback, useRef, useState, useEffect } from 'react';

interface ChartZoomControlsProps {
  absoluteMin: number;
  absoluteMax: number;
  value: [number, number];
  onChange: (value: [number, number]) => void;
  onReset: () => void;
  label?: string;
  formatValue?: (value: number) => string;
}

export const ChartZoomControls: React.FC<ChartZoomControlsProps> = ({
  absoluteMin,
  absoluteMax,
  value,
  onChange,
  onReset,
  label = 'X-Axis Range',
  formatValue = (v) => v.toString(),
}) => {
  const [minVal, maxVal] = value;
  const trackRef = useRef<HTMLDivElement>(null);
  const [dragging, setDragging] = useState<'min' | 'max' | null>(null);

  const range = absoluteMax - absoluteMin;
  const minPercent = ((minVal - absoluteMin) / range) * 100;
  const maxPercent = ((maxVal - absoluteMin) / range) * 100;

  const getValueFromPosition = useCallback((clientX: number): number => {
    if (!trackRef.current) return absoluteMin;
    const rect = trackRef.current.getBoundingClientRect();
    const percent = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
    return Math.round(absoluteMin + percent * range);
  }, [absoluteMin, range]);

  const handleMouseDown = useCallback((handle: 'min' | 'max') => (e: React.MouseEvent) => {
    e.preventDefault();
    setDragging(handle);
  }, []);

  const handleMouseMove = useCallback((e: MouseEvent) => {
    if (!dragging) return;
    const newValue = getValueFromPosition(e.clientX);

    if (dragging === 'min') {
      const clampedMin = Math.max(absoluteMin, Math.min(newValue, maxVal - 1));
      onChange([clampedMin, maxVal]);
    } else {
      const clampedMax = Math.min(absoluteMax, Math.max(newValue, minVal + 1));
      onChange([minVal, clampedMax]);
    }
  }, [dragging, getValueFromPosition, absoluteMin, absoluteMax, minVal, maxVal, onChange]);

  const handleMouseUp = useCallback(() => {
    setDragging(null);
  }, []);

  useEffect(() => {
    if (dragging) {
      window.addEventListener('mousemove', handleMouseMove);
      window.addEventListener('mouseup', handleMouseUp);
      return () => {
        window.removeEventListener('mousemove', handleMouseMove);
        window.removeEventListener('mouseup', handleMouseUp);
      };
    }
  }, [dragging, handleMouseMove, handleMouseUp]);

  const handleTrackClick = useCallback((e: React.MouseEvent) => {
    if (dragging) return;
    const clickedValue = getValueFromPosition(e.clientX);
    const distToMin = Math.abs(clickedValue - minVal);
    const distToMax = Math.abs(clickedValue - maxVal);

    if (distToMin < distToMax) {
      const clampedMin = Math.max(absoluteMin, Math.min(clickedValue, maxVal - 1));
      onChange([clampedMin, maxVal]);
    } else {
      const clampedMax = Math.min(absoluteMax, Math.max(clickedValue, minVal + 1));
      onChange([minVal, clampedMax]);
    }
  }, [dragging, getValueFromPosition, absoluteMin, absoluteMax, minVal, maxVal, onChange]);

  return (
    <div className="flex flex-col gap-1">
      <div className="flex items-center justify-between text-xs text-gray-600">
        <span>{label}: {formatValue(minVal)} - {formatValue(maxVal)}</span>
        <button
          onClick={onReset}
          className="px-2 py-0.5 text-xs text-accent hover:opacity-80 hover:bg-accent/10 rounded transition-colors"
        >
          Reset
        </button>
      </div>
      <div
        ref={trackRef}
        className="relative h-6 cursor-pointer select-none"
        onClick={handleTrackClick}
      >
        {/* Track background */}
        <div className="absolute top-1/2 -translate-y-1/2 left-0 right-0 h-1.5 bg-gray-200 rounded-full" />

        {/* Active range */}
        <div
          className="absolute top-1/2 -translate-y-1/2 h-1.5 bg-accent rounded-full"
          style={{
            left: `${minPercent}%`,
            width: `${maxPercent - minPercent}%`,
          }}
        />

        {/* Min handle */}
        <div
          className={`absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-4 h-4 bg-white border-2 border-accent rounded-full shadow cursor-grab ${dragging === 'min' ? 'cursor-grabbing ring-2 ring-accent/30' : 'hover:ring-2 hover:ring-accent/20'}`}
          style={{ left: `${minPercent}%` }}
          onMouseDown={handleMouseDown('min')}
        />

        {/* Max handle */}
        <div
          className={`absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-4 h-4 bg-white border-2 border-accent rounded-full shadow cursor-grab ${dragging === 'max' ? 'cursor-grabbing ring-2 ring-accent/30' : 'hover:ring-2 hover:ring-accent/20'}`}
          style={{ left: `${maxPercent}%` }}
          onMouseDown={handleMouseDown('max')}
        />
      </div>

      {/* Min/Max labels */}
      <div className="flex justify-between text-[10px] text-gray-400">
        <span>{formatValue(absoluteMin)}</span>
        <span>{formatValue(absoluteMax)}</span>
      </div>
    </div>
  );
};
