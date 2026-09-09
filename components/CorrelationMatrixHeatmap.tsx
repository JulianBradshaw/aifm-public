'use client';

import { useState, useMemo } from 'react';
import type { CorrelationMatrix } from '@/types/samplingConfig';
import parameterDisplayNames from '@/config/parameter_display_names.json';

interface CorrelationMatrixHeatmapProps {
  correlationMatrix: CorrelationMatrix;
}

// Color interpolation from blue (-1) to white (0) to red (+1)
function getCorrelationColor(value: number): string {
  // Clamp value to [-1, 1]
  const v = Math.max(-1, Math.min(1, value));

  if (v < 0) {
    // Blue to white: interpolate from #3b82f6 (blue) to #ffffff (white)
    const t = 1 + v; // t goes from 0 to 1 as v goes from -1 to 0
    const r = Math.round(59 + (255 - 59) * t);
    const g = Math.round(130 + (255 - 130) * t);
    const b = Math.round(246 + (255 - 246) * t);
    return `rgb(${r}, ${g}, ${b})`;
  } else {
    // White to red: interpolate from #ffffff (white) to #ef4444 (red)
    const t = v; // t goes from 0 to 1 as v goes from 0 to 1
    const r = Math.round(255 - (255 - 239) * t);
    const g = Math.round(255 - (255 - 68) * t);
    const b = Math.round(255 - (255 - 68) * t);
    return `rgb(${r}, ${g}, ${b})`;
  }
}

// Get display name for parameter (full human-readable name)
function getDisplayName(name: string): string {
  return (parameterDisplayNames as Record<string, string>)[name] ?? name;
}

// Format parameter name for axis labels (shorter version for space constraints)
function formatParamName(name: string): string {
  const displayName = getDisplayName(name);
  // Truncate long names for axis labels
  if (displayName.length > 20) {
    return displayName.slice(0, 18) + '...';
  }
  return displayName;
}

export function CorrelationMatrixHeatmap({ correlationMatrix }: CorrelationMatrixHeatmapProps) {
  const [hoveredCell, setHoveredCell] = useState<{ row: number; col: number } | null>(null);

  const { parameters, correlation_matrix: matrix } = correlationMatrix;
  const size = parameters.length;

  // Calculate cell size based on number of parameters
  const cellSize = useMemo(() => {
    if (size <= 5) return 50;
    if (size <= 10) return 40;
    return 30;
  }, [size]);

  const labelWidth = 100;
  const labelHeight = 80;
  const svgWidth = labelWidth + size * cellSize + 20;
  const svgHeight = labelHeight + size * cellSize + 20;

  if (size === 0) {
    return (
      <div className="text-sm text-gray-500 italic">
        No correlation matrix available
      </div>
    );
  }

  return (
    <div className="relative">
      <svg width={svgWidth} height={svgHeight} className="font-mono">
        {/* Column labels (rotated) */}
        <g transform={`translate(${labelWidth}, ${labelHeight - 5})`}>
          {parameters.map((param, i) => (
            <text
              key={`col-${i}`}
              x={i * cellSize + cellSize / 2}
              y={0}
              fontSize={9}
              fill="#374151"
              textAnchor="start"
              transform={`rotate(-45, ${i * cellSize + cellSize / 2}, 0)`}
            >
              {formatParamName(param)}
            </text>
          ))}
        </g>

        {/* Row labels */}
        <g transform={`translate(${labelWidth - 5}, ${labelHeight})`}>
          {parameters.map((param, i) => (
            <text
              key={`row-${i}`}
              x={0}
              y={i * cellSize + cellSize / 2 + 3}
              fontSize={9}
              fill="#374151"
              textAnchor="end"
            >
              {formatParamName(param)}
            </text>
          ))}
        </g>

        {/* Heatmap cells */}
        <g transform={`translate(${labelWidth}, ${labelHeight})`}>
          {matrix.map((row: number[], i: number) =>
            row.map((value: number, j: number) => (
              <g key={`cell-${i}-${j}`}>
                <rect
                  x={j * cellSize}
                  y={i * cellSize}
                  width={cellSize - 1}
                  height={cellSize - 1}
                  fill={getCorrelationColor(value)}
                  stroke={
                    hoveredCell?.row === i && hoveredCell?.col === j
                      ? '#1f2937'
                      : '#e5e7eb'
                  }
                  strokeWidth={hoveredCell?.row === i && hoveredCell?.col === j ? 2 : 0.5}
                  onMouseEnter={() => setHoveredCell({ row: i, col: j })}
                  onMouseLeave={() => setHoveredCell(null)}
                  className="cursor-pointer"
                />
                {/* Show value for diagonal and hovered cells */}
                {(i === j || (hoveredCell?.row === i && hoveredCell?.col === j)) && (
                  <text
                    x={j * cellSize + cellSize / 2}
                    y={i * cellSize + cellSize / 2 + 3}
                    fontSize={10}
                    fill={Math.abs(value) > 0.5 ? '#fff' : '#374151'}
                    textAnchor="middle"
                    pointerEvents="none"
                  >
                    {value.toFixed(1)}
                  </text>
                )}
              </g>
            ))
          )}
        </g>
      </svg>

      {/* Tooltip */}
      {hoveredCell && (
        <div
          className="absolute bg-gray-900 text-white text-xs px-2 py-1 rounded shadow-lg pointer-events-none z-10"
          style={{
            left: labelWidth + hoveredCell.col * cellSize + cellSize,
            top: labelHeight + hoveredCell.row * cellSize,
          }}
        >
          <div className="font-medium">
            {getDisplayName(parameters[hoveredCell.row])} × {getDisplayName(parameters[hoveredCell.col])}
          </div>
          <div>
            Correlation: {matrix[hoveredCell.row][hoveredCell.col].toFixed(3)}
          </div>
        </div>
      )}

      {/* Legend */}
      <div className="flex items-center gap-2 mt-4 text-xs text-gray-600">
        <span>-1</span>
        <div
          className="h-3 w-24 rounded"
          style={{
            background: 'linear-gradient(to right, #3b82f6, #ffffff, #ef4444)',
          }}
        />
        <span>+1</span>
        <span className="ml-2 text-gray-500">({correlationMatrix.correlation_type})</span>
      </div>
    </div>
  );
}

export default CorrelationMatrixHeatmap;
