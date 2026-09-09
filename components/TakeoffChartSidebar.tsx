'use client';

import { useState, useRef } from 'react';
import { createPortal } from 'react-dom';
import { MILESTONE_EXPLANATIONS } from '@/constants/chartExplanations';
import { ComparisonMultiSelect } from './ComparisonMultiSelect';

// Milestone display configuration
const MILESTONES = [
  { key: 'SAR-level-experiment-selection-skill', name: 'SAR', fullName: 'Superhuman AI Researcher', color: '#000090', explanationKey: 'SAR' },
  { key: 'SIAR-level-experiment-selection-skill', name: 'SIAR', fullName: 'Superintelligent AI Researcher', color: '#900000', explanationKey: 'SIAR' },
  { key: 'TED-AI', name: 'TED-AI', fullName: 'Top Expert Dominating AI', color: '#D62728', explanationKey: 'TED-AI' },
  { key: 'ASI', name: 'ASI', fullName: 'Artificial Superintelligence', color: '#8C564B', explanationKey: 'ASI' },
];

// Tooltip component for milestone labels
function MilestoneTooltip({ name, description }: { name: string; description: string }) {
  const [showTooltip, setShowTooltip] = useState(false);
  const [tooltipPosition, setTooltipPosition] = useState({ top: 0, left: 0 });
  const triggerRef = useRef<HTMLSpanElement>(null);

  const openTooltip = () => {
    const el = triggerRef.current;
    if (el) {
      const rect = el.getBoundingClientRect();
      const tooltipWidth = 240;
      const padding = 8;

      // Position to the left of the trigger (since sidebar is on the right)
      let left = rect.left - tooltipWidth - padding;

      // Fallback to right if not enough space on left
      if (left < padding) {
        left = rect.right + padding;
      }

      // Prevent tooltip from going off-screen horizontally
      if (left + tooltipWidth > window.innerWidth - padding) {
        left = window.innerWidth - tooltipWidth - padding;
      }

      let top = rect.top;

      // Prevent tooltip from going off-screen vertically
      if (top + 80 > window.innerHeight) {
        top = rect.bottom - 80;
      }

      setTooltipPosition({ top, left });
    }
    setShowTooltip(true);
  };

  return (
    <span className="inline-flex items-center gap-1">
      <span className="text-gray-700">{name}</span>
      <span
        ref={triggerRef}
        className="inline-flex h-3 w-3 min-w-3 items-center justify-center rounded-full border border-gray-400 text-[8px] font-semibold leading-none text-gray-500 cursor-help transition-colors hover:border-gray-600 hover:text-gray-700"
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
          <div className="w-60 max-w-[240px] rounded-md bg-gray-900 px-3 py-2 text-xs text-white shadow-lg font-normal">
            {description}
          </div>
        </div>,
        document.body
      )}
    </span>
  );
}

interface AvailableSimulation {
  id: string;
  label: string;
}

interface TakeoffChartSidebarProps {
  visibleMilestones: Set<string>;
  onToggleMilestone: (milestone: string) => void;
  showCdf: boolean;
  onToggleCdf: () => void;
  showGridlines: boolean;
  onToggleGridlines: () => void;
  // Compare section props
  compareMode: 'forecaster' | 'model-vs-atc';
  onCompareModeChange: (mode: 'forecaster' | 'model-vs-atc') => void;
  comparisonSimulationIds: string[];
  onComparisonSimulationsChange: (ids: string[]) => void;
  availableSimulations: AvailableSimulation[];
  currentSimulationId: string;
  // Forecaster mode: which type to show
  forecasterShowType: 'model' | 'atc';
  onForecasterShowTypeChange: (type: 'model' | 'atc') => void;
  // Model vs ATC mode: multi-select
  showModel: boolean;
  onToggleModel: () => void;
  showAtc: boolean;
  onToggleAtc: () => void;
  hasAtcData: boolean;
  // Loading state for comparison data fetch
  isLoadingComparison?: boolean;
}

export function TakeoffChartSidebar({
  visibleMilestones,
  onToggleMilestone,
  showCdf,
  onToggleCdf,
  showGridlines,
  onToggleGridlines,
  compareMode,
  onCompareModeChange,
  comparisonSimulationIds,
  onComparisonSimulationsChange,
  availableSimulations,
  currentSimulationId,
  forecasterShowType,
  onForecasterShowTypeChange,
  showModel,
  onToggleModel,
  showAtc,
  onToggleAtc,
  hasAtcData,
  isLoadingComparison = false,
}: TakeoffChartSidebarProps) {
  return (
    <div className="w-full md:w-48 shrink-0 bg-gray-50 rounded-lg p-4 border border-gray-200">
      <h4 className="text-xs font-semibold text-gray-700 mb-3">Chart Settings</h4>

      {/* Milestones Section */}
      <div className="mb-4">
        <div className="text-[10px] font-medium text-gray-500 uppercase tracking-wide mb-2">
          Milestones
        </div>
        <div className="space-y-2">
          {MILESTONES.map((milestone) => (
            <label
              key={milestone.key}
              className="flex items-center gap-2 cursor-pointer text-xs"
            >
              <input
                type="checkbox"
                checked={visibleMilestones.has(milestone.key)}
                onChange={() => onToggleMilestone(milestone.key)}
                className="rounded border-gray-300 text-gray-800 focus:ring-gray-800"
              />
              <span
                className="w-3 h-0.5 shrink-0"
                style={{ backgroundColor: milestone.color }}
              />
              <MilestoneTooltip
                name={milestone.name}
                description={MILESTONE_EXPLANATIONS[milestone.explanationKey] || milestone.fullName}
              />
            </label>
          ))}
        </div>
      </div>

      {/* Display Mode Section */}
      <div className="mb-4">
        <div className="text-[10px] font-medium text-gray-500 uppercase tracking-wide mb-2">
          Display
        </div>
        <div className="flex gap-1">
          <button
            onClick={onToggleCdf}
            className={`flex-1 px-2 py-1 text-xs rounded transition-colors ${
              !showCdf
                ? 'bg-gray-800 text-white'
                : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
            }`}
          >
            PDF
          </button>
          <button
            onClick={onToggleCdf}
            className={`flex-1 px-2 py-1 text-xs rounded transition-colors ${
              showCdf
                ? 'bg-gray-800 text-white'
                : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
            }`}
          >
            CDF
          </button>
        </div>
      </div>

      {/* Compare Section */}
      <div className="mb-4">
        <div className="text-[10px] font-medium text-gray-500 uppercase tracking-wide mb-2">
          Compare
        </div>
        {/* Mode toggler */}
        <div className="flex gap-1 mb-3">
          <button
            onClick={() => onCompareModeChange('forecaster')}
            className={`flex-1 px-1.5 py-1 text-[10px] rounded transition-colors ${
              compareMode === 'forecaster'
                ? 'bg-gray-800 text-white'
                : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
            }`}
          >
            Forecaster
          </button>
          <button
            onClick={() => onCompareModeChange('model-vs-atc')}
            className={`flex-1 px-1.5 py-1 text-[10px] rounded transition-colors ${
              compareMode === 'model-vs-atc'
                ? 'bg-gray-800 text-white'
                : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
            }`}
          >
            Model vs ATC
          </button>
        </div>

        {/* Forecaster mode UI */}
        {compareMode === 'forecaster' && (
          <div className="space-y-3">
            <div>
              <div className="text-[10px] text-gray-500 mb-1">Compare to:</div>
              <div className="relative">
                <ComparisonMultiSelect
                  availableSimulations={availableSimulations}
                  currentSimulationId={currentSimulationId}
                  selectedIds={comparisonSimulationIds}
                  onChange={onComparisonSimulationsChange}
                  disabled={isLoadingComparison}
                />
                {isLoadingComparison && (
                  <div className="flex items-center gap-1.5 mt-1.5">
                    <svg
                      className="animate-spin h-3 w-3 text-gray-500"
                      xmlns="http://www.w3.org/2000/svg"
                      fill="none"
                      viewBox="0 0 24 24"
                    >
                      <circle
                        className="opacity-25"
                        cx="12"
                        cy="12"
                        r="10"
                        stroke="currentColor"
                        strokeWidth="4"
                      />
                      <path
                        className="opacity-75"
                        fill="currentColor"
                        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                      />
                    </svg>
                    <span className="text-[10px] text-gray-500">Loading...</span>
                  </div>
                )}
              </div>
            </div>
            {hasAtcData && (
              <div>
                <div className="text-[10px] text-gray-500 mb-1">Show:</div>
                <div className="space-y-1.5">
                  <label className="flex items-center gap-2 cursor-pointer text-xs">
                    <input
                      type="radio"
                      name="takeoffForecasterShowType"
                      checked={forecasterShowType === 'model'}
                      onChange={() => onForecasterShowTypeChange('model')}
                      className="text-gray-800 focus:ring-gray-800"
                    />
                    <span className="text-gray-700">Model output</span>
                  </label>
                  <label className="flex items-center gap-2 cursor-pointer text-xs">
                    <input
                      type="radio"
                      name="takeoffForecasterShowType"
                      checked={forecasterShowType === 'atc'}
                      onChange={() => onForecasterShowTypeChange('atc')}
                      className="text-gray-800 focus:ring-gray-800"
                    />
                    <span className="text-gray-700">All-things-considered</span>
                  </label>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Model vs ATC mode UI */}
        {compareMode === 'model-vs-atc' && hasAtcData && (
          <div>
            <div className="text-[10px] text-gray-500 mb-1">Show:</div>
            <div className="space-y-1.5">
              <label className="flex items-center gap-2 cursor-pointer text-xs">
                <input
                  type="checkbox"
                  checked={showModel}
                  onChange={onToggleModel}
                  className="rounded border-gray-300 text-gray-800 focus:ring-gray-800"
                />
                <span className="text-gray-700">Model output</span>
              </label>
              <label className="flex items-center gap-2 cursor-pointer text-xs">
                <input
                  type="checkbox"
                  checked={showAtc}
                  onChange={onToggleAtc}
                  className="rounded border-gray-300 text-gray-800 focus:ring-gray-800"
                />
                <span className="text-gray-700">All-things-considered</span>
              </label>
            </div>
            <p className="text-[9px] text-gray-500 mt-1.5">
              ATC shown as dashed lines
            </p>
          </div>
        )}

        {/* Show simple message when no ATC data in model-vs-atc mode */}
        {compareMode === 'model-vs-atc' && !hasAtcData && (
          <p className="text-[10px] text-gray-500">
            No all-things-considered data available for this forecast.
          </p>
        )}
      </div>

      {/* Gridlines Section */}
      <div>
        <div className="text-[10px] font-medium text-gray-500 uppercase tracking-wide mb-2">
          Grid
        </div>
        <label className="flex items-center gap-2 cursor-pointer text-xs">
          <input
            type="checkbox"
            checked={showGridlines}
            onChange={onToggleGridlines}
            className="rounded border-gray-300 text-gray-800 focus:ring-gray-800"
          />
          <span className="text-gray-700">Show gridlines</span>
        </label>
      </div>
    </div>
  );
}
