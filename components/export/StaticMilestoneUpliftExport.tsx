'use client';

import { forwardRef, useMemo } from 'react';
import type { ChartDataPoint } from '@/app/types';
import type { MilestoneMap } from '@/types/milestones';
import { EXPORT_COLORS } from '@/utils/chartExport';
import { MILESTONE_EXPLANATIONS } from '@/constants/chartExplanations';
import { formatEffectiveComputeOOMsNode } from '@/utils/formatting';
import { StaticUpliftChart } from './StaticUpliftChart';

export interface StaticMilestoneUpliftExportProps {
  chartData: ChartDataPoint[];
  milestones: MilestoneMap | null;
  displayEndYear: number;
  simulationEndYear?: number;
  width?: number;
  height?: number;
  showMilestoneIndicators?: boolean;
  presentDay?: number;
}

// Key milestones to display
const KEY_MILESTONES = ['AC', 'TED-AI', 'ASI'] as const;

// Display names for milestone keys
const MILESTONE_DISPLAY_KEYS: Record<string, string> = {
  'AC': 'AC',
  'TED-AI': 'TED-AI',
  'ASI': 'ASI',
};

// Full names for milestones
const MILESTONE_FULL_NAMES: Record<string, string> = {
  'AC': 'Automated Coder',
  'TED-AI': 'Top-Expert-Dominating AI',
  'ASI': 'Artificial Superintelligence',
};

const FONTS = {
  title: 'et-book, Georgia, serif',
  label: 'Menlo, Consolas, monospace',
  date: 'et-book, Georgia, serif',
};

interface MilestoneDisplay {
  monthNumber: number;
  year: number;
}

function getMilestoneDisplay(decimalYear: number | null, simulationEndYear: number): MilestoneDisplay | null {
  if (decimalYear == null || !Number.isFinite(decimalYear)) {
    return null;
  }
  if (decimalYear > simulationEndYear) {
    return null;
  }

  const year = Math.floor(decimalYear);
  const fraction = decimalYear - year;
  const monthIndexRaw = Math.round(fraction * 12);
  const monthIndex = monthIndexRaw % 12;
  const adjustedYear = monthIndexRaw === 12 ? year + 1 : year;

  return {
    monthNumber: monthIndex + 1,
    year: adjustedYear,
  };
}

function formatDate(milestone: MilestoneDisplay | null, simulationEndYear: number): string {
  if (!milestone) {
    return `>${simulationEndYear}`;
  }
  const month = String(milestone.monthNumber).padStart(2, '0');
  return `${month}/${milestone.year}`;
}

/**
 * Export component showing milestone dates on the left and uplift chart on the right
 */
export const StaticMilestoneUpliftExport = forwardRef<HTMLDivElement, StaticMilestoneUpliftExportProps>(({
  chartData,
  milestones,
  displayEndYear,
  simulationEndYear = 2045,
  width = 900,
  height = 400,
  showMilestoneIndicators = true,
  presentDay,
}, ref) => {
  // Format milestone info
  const milestoneInfo = useMemo(() => {
    if (!milestones) return [];

    return KEY_MILESTONES.map(key => {
      const milestone = milestones[key];
      const time = milestone?.time as number | null | undefined;
      const displayKey = MILESTONE_DISPLAY_KEYS[key] || key;
      const fullName = MILESTONE_FULL_NAMES[key] || key;
      const explanation = MILESTONE_EXPLANATIONS[displayKey];
      const display = getMilestoneDisplay(time ?? null, simulationEndYear);
      const dateString = formatDate(display, simulationEndYear);
      // Only show a compute level for milestones that are actually reached.
      const reached = display != null;
      const effectiveComputeOOMs = reached
        ? (milestone?.effective_compute_ooms as number | null | undefined) ?? null
        : null;

      return {
        key: displayKey,
        fullName,
        date: dateString,
        explanation,
        effectiveComputeOOMs,
      };
    });
  }, [milestones, simulationEndYear]);

  // Layout calculations - 50/50 split with reduced padding
  const padding = 10;
  const gap = 24;
  const contentWidth = width - padding * 2;
  const milestoneWidth = Math.round((contentWidth - gap) * 0.5);
  const chartWidth = contentWidth - milestoneWidth - gap;
  const chartHeight = height - padding * 2;

  return (
    <div
      ref={ref}
      style={{
        width,
        height,
        backgroundColor: EXPORT_COLORS.background,
        color: EXPORT_COLORS.foreground,
        fontFamily: FONTS.title,
        display: 'flex',
        flexDirection: 'row',
        padding,
        boxSizing: 'border-box',
        gap,
      }}
    >
      {/* Left side: Milestone Dates */}
      <div
        style={{
          width: milestoneWidth,
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'center',
          gap: 28,
        }}
      >
        {milestoneInfo.map(({ key, fullName, date, explanation, effectiveComputeOOMs }) => (
          <div
            key={key}
            style={{
              display: 'flex',
              flexDirection: 'column',
              gap: 6,
            }}
          >
            {/* Label row with divider */}
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 10,
              }}
            >
              <span
                style={{
                  fontSize: 18,
                  fontWeight: 600,
                  fontFamily: FONTS.label,
                  color: EXPORT_COLORS.foreground,
                  whiteSpace: 'nowrap',
                }}
              >
                {key} ({fullName})
              </span>
              <div
                style={{
                  flex: 1,
                  height: 1,
                  backgroundColor: 'rgba(100, 116, 139, 0.3)',
                }}
              />
            </div>

            {/* Date row */}
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 16,
              }}
            >
              <div style={{ display: 'flex', flexDirection: 'column' }}>
                <span
                  style={{
                    fontSize: 58,
                    fontWeight: 'bold',
                    fontFamily: FONTS.date,
                    color: EXPORT_COLORS.foreground,
                    letterSpacing: 2,
                    lineHeight: 1,
                  }}
                >
                  {date}
                </span>
                {effectiveComputeOOMs != null && (
                  <span
                    style={{
                      fontSize: 15,
                      fontFamily: FONTS.label,
                      color: EXPORT_COLORS.foreground,
                      opacity: 0.7,
                      marginTop: 4,
                      whiteSpace: 'nowrap',
                    }}
                  >
                    {formatEffectiveComputeOOMsNode(effectiveComputeOOMs)} effective compute
                  </span>
                )}
              </div>
              {explanation && (
                <span
                  style={{
                    fontSize: 14,
                    fontFamily: FONTS.label,
                    color: EXPORT_COLORS.foreground,
                    opacity: 0.75,
                    lineHeight: 1.4,
                    flex: 1,
                  }}
                >
                  {explanation}
                </span>
              )}
            </div>
          </div>
        ))}
      </div>

      {/* Right side: Uplift Chart */}
      <div
        style={{
          width: chartWidth,
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'center',
        }}
      >
        <StaticUpliftChart
          chartData={chartData}
          milestones={milestones}
          displayEndYear={displayEndYear}
          width={chartWidth}
          height={chartHeight}
          title="AI Software R&D Uplift"
          showMilestoneIndicators={showMilestoneIndicators}
          presentDay={presentDay}
        />
      </div>
    </div>
  );
});

StaticMilestoneUpliftExport.displayName = 'StaticMilestoneUpliftExport';
