'use client';

import { useRouter, useSearchParams } from 'next/navigation';
import { useCallback } from 'react';

export interface Simulation {
  id: string;
  label: string;
  forecaster: string;
  date: string;
  customIdentifier: string | null;
  dataPath: string;
  numSamples: number | null;
  modelVersion: string | null;
}

interface SimulationSelectorProps {
  simulations: Simulation[];
  currentSimulationId: string;
  className?: string;
}

// Parse date in MM-DD-YY format
const parseDate = (dateStr: string): number => {
  const [month, day, year] = dateStr.split('-').map(Number);
  // Convert YY to full year (assume 2000s)
  const fullYear = year < 100 ? 2000 + year : year;
  return new Date(fullYear, month - 1, day).getTime();
};

export function SimulationSelector({
  simulations,
  currentSimulationId,
  className = '',
}: SimulationSelectorProps) {
  const router = useRouter();
  const searchParams = useSearchParams();

  const handleChange = useCallback((event: React.ChangeEvent<HTMLSelectElement>) => {
    const newId = event.target.value;
    if (newId !== currentSimulationId) {
      // Preserve existing URL params (except old compare param)
      const params = new URLSearchParams(window.location.search);
      // Remove old compare param if present
      params.delete('compare');
      // Drop the new primary simulation from the comparison list so a
      // forecast is never compared against itself
      const csim = params.get('csim');
      if (csim) {
        const remaining = csim.split(',').map(s => s.trim()).filter(id => id.length > 0 && id !== newId);
        if (remaining.length > 0) {
          params.set('csim', remaining.join(','));
        } else {
          params.delete('csim');
        }
      }
      const queryString = params.toString();
      router.push(`/forecast/${encodeURIComponent(newId)}${queryString ? `?${queryString}` : ''}`);
    }
  }, [router, currentSimulationId, searchParams]);

  // Don't render if there's only one or no simulations
  if (simulations.length <= 1) {
    return null;
  }

  // Sort simulations by date (newest first), then by forecaster (Daniel first)
  // Date format is MM-DD-YY, so we need to parse and compare
  const sortedSimulations = [...simulations].sort((a, b) => {
    const dateA = parseDate(a.date);
    const dateB = parseDate(b.date);

    // Primary sort: descending by date (newest first)
    if (dateB !== dateA) {
      return dateB - dateA;
    }

    // Secondary sort: Daniel first, then alphabetically
    if (a.forecaster === 'Daniel' && b.forecaster !== 'Daniel') return -1;
    if (b.forecaster === 'Daniel' && a.forecaster !== 'Daniel') return 1;
    return a.forecaster.localeCompare(b.forecaster);
  });

  return (
    <div className={`flex flex-col gap-2 ${className}`}>
      <div className="flex items-center gap-2">
        <label htmlFor="simulation-selector" className="text-sm font-medium text-gray-700">
          Forecast:
        </label>
        <select
          id="simulation-selector"
          value={currentSimulationId}
          onChange={handleChange}
          className="block rounded-md border-gray-300 bg-white py-1.5 pl-3 pr-10 text-sm shadow-sm focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
        >
          {sortedSimulations.map((sim) => (
            <option key={sim.id} value={sim.id}>
              {sim.label}
            </option>
          ))}
        </select>
      </div>
    </div>
  );
}

interface SimulationMetadataProps {
  simulation: Simulation;
  className?: string;
}

export function SimulationMetadata({
  simulation,
  className = '',
}: SimulationMetadataProps) {
  return (
    <div className={`text-xs text-gray-500 ${className}`}>
      <span>
        {simulation.forecaster} forecast
        {simulation.numSamples && ` (${simulation.numSamples.toLocaleString()} samples)`}
        {simulation.modelVersion && ` - Model ${simulation.modelVersion}`}
      </span>
    </div>
  );
}
