'use client';

import { useState, useRef } from 'react';
import { createPortal } from 'react-dom';

interface ProbabilityTableProps {
  title?: string;
  headers: string[];
  rows: { label: string; values: string[]; description?: string }[];
  className?: string;
}

// Tooltip component for row labels with descriptions
function LabelWithTooltip({ label, description }: { label: string; description: string }) {
  const [showTooltip, setShowTooltip] = useState(false);
  const [tooltipPosition, setTooltipPosition] = useState({ top: 0, left: 0 });
  const triggerRef = useRef<HTMLSpanElement>(null);

  const openTooltip = () => {
    const el = triggerRef.current;
    if (el) {
      const rect = el.getBoundingClientRect();
      const tooltipWidth = 280;
      const padding = 8;

      // Position to the right of the trigger
      let left = rect.right + padding;

      // Prevent tooltip from going off-screen horizontally
      if (left + tooltipWidth > window.innerWidth - padding) {
        left = rect.left - tooltipWidth - padding;
      }
      if (left < padding) {
        left = padding;
      }

      // Position below the trigger
      let top = rect.top;

      // Prevent tooltip from going off-screen vertically
      if (top + 100 > window.innerHeight) {
        top = rect.bottom - 100;
      }

      setTooltipPosition({ top, left });
    }
    setShowTooltip(true);
  };

  return (
    <span className="inline whitespace-nowrap">
      {label}
      <span
        ref={triggerRef}
        className="ml-1 inline-flex h-3.5 w-3.5 min-w-3.5 items-center justify-center rounded-full border border-gray-400 text-[9px] font-semibold leading-none text-gray-500 cursor-help transition-colors hover:border-gray-600 hover:text-gray-700 align-middle"
        role="button"
        tabIndex={0}
        aria-label="Show explanation"
        aria-expanded={showTooltip}
        onMouseEnter={openTooltip}
        onMouseLeave={() => setShowTooltip(false)}
        onFocus={openTooltip}
        onBlur={() => setShowTooltip(false)}
      >
        i
      </span>
      {showTooltip && typeof document !== 'undefined' && createPortal(
        <div
          className="pointer-events-none fixed z-[9999]"
          style={{ top: tooltipPosition.top, left: tooltipPosition.left }}
        >
          <div className="w-70 max-w-[280px] rounded-md bg-gray-900 px-3 py-2 text-xs text-white shadow-lg font-normal">
            {description}
          </div>
        </div>,
        document.body
      )}
    </span>
  );
}

export function ProbabilityTable({
  title,
  headers,
  rows,
  className = '',
}: ProbabilityTableProps) {
  return (
    <div className={`${className}`}>
      {title && <h3 className="text-lg font-medium text-gray-800 mb-3">{title}</h3>}
      <div className="overflow-x-auto">
        <table className="min-w-full border-collapse border border-gray-300">
          <thead>
            <tr className="bg-gray-50">
              <th className="border border-gray-300 px-4 py-2 text-left text-sm font-semibold text-gray-900">
                {headers[0]}
              </th>
              {headers.slice(1).map((header, i) => (
                <th key={i} className="border border-gray-300 px-4 py-2 text-right text-sm font-semibold text-gray-900">
                  {header}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, i) => (
              <tr key={i} className={i % 2 === 0 ? "bg-white" : "bg-gray-50"}>
                <td className="border border-gray-300 px-4 py-2 text-sm font-medium text-gray-900">
                  {row.description ? (
                    <LabelWithTooltip label={row.label} description={row.description} />
                  ) : (
                    row.label
                  )}
                </td>
                {row.values.map((value, j) => (
                  <td key={j} className="border border-gray-300 px-4 py-2 text-right text-sm text-gray-700">
                    {value}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
