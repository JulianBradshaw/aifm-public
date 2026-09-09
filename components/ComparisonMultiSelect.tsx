'use client';

import { useState, useRef, useEffect } from 'react';

interface AvailableSimulation {
  id: string;
  label: string;
}

interface ComparisonMultiSelectProps {
  availableSimulations: AvailableSimulation[];
  currentSimulationId: string;
  selectedIds: string[];
  onChange: (ids: string[]) => void;
  disabled?: boolean;
}

// Checkbox-dropdown for picking multiple forecasts to compare against.
// Selection order is preserved so chart styling stays stable as items are added.
export function ComparisonMultiSelect({
  availableSimulations,
  currentSimulationId,
  selectedIds,
  onChange,
  disabled = false,
}: ComparisonMultiSelectProps) {
  const [isOpen, setIsOpen] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  const options = availableSimulations.filter(sim => sim.id !== currentSimulationId);

  // Close on outside click
  useEffect(() => {
    if (!isOpen) return;
    const handleClick = (event: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    };
    document.addEventListener('mousedown', handleClick);
    return () => document.removeEventListener('mousedown', handleClick);
  }, [isOpen]);

  const toggleId = (id: string) => {
    if (selectedIds.includes(id)) {
      onChange(selectedIds.filter(selected => selected !== id));
    } else {
      onChange([...selectedIds, id]);
    }
  };

  const selectedLabels = selectedIds
    .map(id => options.find(sim => sim.id === id)?.label)
    .filter((label): label is string => label !== undefined);

  const summary = selectedLabels.length === 0
    ? 'None'
    : selectedLabels.length === 1
      ? selectedLabels[0]
      : `${selectedLabels.length} selected`;

  return (
    <div ref={containerRef} className="relative">
      <button
        type="button"
        onClick={() => setIsOpen(prev => !prev)}
        disabled={disabled}
        className="w-full flex items-center justify-between gap-1 text-xs px-2 py-1 rounded border border-gray-300 bg-white text-left focus:border-gray-500 focus:outline-none disabled:opacity-50"
        aria-haspopup="listbox"
        aria-expanded={isOpen}
      >
        <span className={`truncate ${selectedLabels.length === 0 ? 'text-gray-500' : 'text-gray-800'}`}>
          {summary}
        </span>
        <svg
          className={`h-3 w-3 shrink-0 text-gray-500 transition-transform ${isOpen ? 'rotate-180' : ''}`}
          xmlns="http://www.w3.org/2000/svg"
          viewBox="0 0 20 20"
          fill="currentColor"
        >
          <path fillRule="evenodd" d="M5.23 7.21a.75.75 0 011.06.02L10 11.168l3.71-3.938a.75.75 0 111.08 1.04l-4.25 4.5a.75.75 0 01-1.08 0l-4.25-4.5a.75.75 0 01.02-1.06z" clipRule="evenodd" />
        </svg>
      </button>
      {isOpen && (
        <div className="absolute z-20 mt-1 w-full max-h-56 overflow-y-auto rounded border border-gray-300 bg-white shadow-lg py-1">
          {options.length === 0 && (
            <p className="px-2 py-1 text-[10px] text-gray-500">No other forecasts available.</p>
          )}
          {options.map(sim => (
            <label
              key={sim.id}
              className="flex items-center gap-2 px-2 py-1 text-xs cursor-pointer hover:bg-gray-100"
            >
              <input
                type="checkbox"
                checked={selectedIds.includes(sim.id)}
                onChange={() => toggleId(sim.id)}
                className="rounded border-gray-300 text-gray-800 focus:ring-gray-800"
              />
              <span className="text-gray-700">{sim.label}</span>
            </label>
          ))}
          {selectedIds.length > 0 && (
            <button
              type="button"
              onClick={() => onChange([])}
              className="w-full text-left px-2 py-1 text-[10px] text-gray-500 hover:text-gray-700 hover:bg-gray-100 border-t border-gray-200 mt-1"
            >
              Clear selection
            </button>
          )}
        </div>
      )}
    </div>
  );
}
