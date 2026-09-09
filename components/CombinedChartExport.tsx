'use client';

import { forwardRef, useMemo } from 'react';
import type { ChartDataPoint, BenchmarkPoint } from '@/app/types';
import type { MilestoneMap } from '@/types/milestones';
import { EXPORT_COLORS } from '@/utils/chartExport';
import { MILESTONE_EXPLANATIONS } from '@/constants/chartExplanations';
import { StaticHorizonChart } from './export/StaticHorizonChart';
import { StaticUpliftChart } from './export/StaticUpliftChart';

export interface CombinedChartExportProps {
  horizonChartData: ChartDataPoint[];
  upliftChartData: ChartDataPoint[];
  milestones: MilestoneMap | null;
  scHorizonMinutes: number;
  displayEndYear: number;
  simulationEndYear?: number;
  benchmarkData?: BenchmarkPoint[];
  showMilestoneIndicators?: boolean;
  trajectoryName?: string;
  presentDay?: number;
}

// Key milestones to display (subset of all milestones)
// Note: SAR uses full key name in the data
const KEY_MILESTONES = ['AC', 'SAR-level-experiment-selection-skill', 'ASI'] as const;

// Display names for milestone keys
const MILESTONE_DISPLAY_KEYS: Record<string, string> = {
  'AC': 'AC',
  'SAR-level-experiment-selection-skill': 'SAR',
  'ASI': 'ASI',
};

// Full names for milestones
const MILESTONE_FULL_NAMES: Record<string, string> = {
  'AC': 'Automated Coder',
  'SAR-level-experiment-selection-skill': 'Superhuman AI Researcher',
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
 * Combined chart export view showing both Horizon and Uplift charts
 * with milestone summary, optimized for social sharing
 */
export const CombinedChartExport = forwardRef<HTMLDivElement, CombinedChartExportProps>(({
  horizonChartData,
  upliftChartData,
  milestones,
  scHorizonMinutes,
  displayEndYear,
  simulationEndYear = 2045,
  benchmarkData,
  showMilestoneIndicators = true,
  trajectoryName = 'My Trajectory',
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
      // Use displayKey for explanation lookup since MILESTONE_EXPLANATIONS uses short keys
      const explanation = MILESTONE_EXPLANATIONS[displayKey];
      const display = getMilestoneDisplay(time ?? null, simulationEndYear);
      const dateString = formatDate(display, simulationEndYear);

      return {
        key: displayKey,
        fullName,
        date: dateString,
        explanation,
      };
    });
  }, [milestones, simulationEndYear]);

  const width = 1200;
  const height = 675;
  const chartHeight = 320;
  const chartWidth = 560;

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
        flexDirection: 'column',
        padding: '16px 24px 10px 24px',
        boxSizing: 'border-box',
        overflow: 'hidden',
      }}
    >
      {/* Header */}
      <div
        style={{
          textAlign: 'center',
          marginBottom: '8px',
          flexShrink: 0,
        }}
      >
        <h1
          style={{
            fontSize: '28px',
            fontWeight: 'bold',
            margin: '0',
            fontFamily: FONTS.title,
            color: EXPORT_COLORS.foreground,
          }}
        >
          AI Futures Model - {trajectoryName}
        </h1>
      </div>

      {/* Charts Row */}
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          gap: '16px',
          flex: 1,
        }}
      >
        {/* Horizon Chart */}
        <div style={{ display: 'flex', flexDirection: 'column', width: chartWidth }}>
          <h2
            style={{
              textAlign: 'center',
              margin: 0,
              padding: '4px 0',
              fontSize: '20px',
              fontWeight: 'bold',
              fontFamily: FONTS.title,
              color: EXPORT_COLORS.foreground,
            }}
          >
            Coding Time Horizon (METR 80%)
          </h2>
          <StaticHorizonChart
            chartData={horizonChartData}
            scHorizonMinutes={scHorizonMinutes}
            displayEndYear={displayEndYear}
            width={chartWidth}
            height={chartHeight - 48}
            title=""
            benchmarkData={benchmarkData}
            showMilestoneIndicators={showMilestoneIndicators}
          />
          <p
            style={{
              textAlign: 'center',
              margin: 0,
              padding: '0 4px',
              fontSize: '11px',
              fontFamily: FONTS.label,
              color: EXPORT_COLORS.foreground,
              lineHeight: 1.3,
            }}
          >
            The coding time horizon is the maximum length of coding tasks frontier AI systems can complete with a success rate of 80%, with the length defined as the time taken by typical AI company employees who do similar tasks.
          </p>
        </div>

        {/* Uplift Chart */}
        <div style={{ display: 'flex', flexDirection: 'column', width: chartWidth }}>
          <h2
            style={{
              textAlign: 'center',
              margin: 0,
              padding: '4px 0',
              fontSize: '20px',
              fontWeight: 'bold',
              fontFamily: FONTS.title,
              color: EXPORT_COLORS.foreground,
            }}
          >
            AI Software R&D Uplift
          </h2>
          <StaticUpliftChart
            chartData={upliftChartData}
            milestones={milestones}
            displayEndYear={displayEndYear}
            width={chartWidth}
            height={chartHeight - 48}
            title=""
            showMilestoneIndicators={showMilestoneIndicators}
            presentDay={presentDay}
          />
          <p
            style={{
              textAlign: 'center',
              margin: 0,
              padding: '0 4px',
              fontSize: '11px',
              fontFamily: FONTS.label,
              color: EXPORT_COLORS.foreground,
              lineHeight: 1.3,
            }}
          >
            The AI Software R&D Uplift is the speedup in software progress that would result if the frontier AI systems at a given time were deployed within today&apos;s leading AI company, relative to the company running with no AI assistance at all.
          </p>
        </div>
      </div>

      {/* Milestone Summary */}
      <div
        style={{
          marginTop: '0px',
          padding: '8px 16px',
          backgroundColor: 'rgba(0, 0, 0, 0.03)',
          borderRadius: '8px',
          flexShrink: 0,
        }}
      >
        <div
          style={{
            display: 'flex',
            flexDirection: 'column',
            gap: '6px',
          }}
        >
          {milestoneInfo.map(({ key, fullName, date, explanation }) => (
            <div
              key={key}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '20px',
              }}
            >
              <span
                style={{
                  fontSize: '20px',
                  fontWeight: 600,
                  fontFamily: FONTS.label,
                  color: EXPORT_COLORS.foreground,
                  width: '390px',
                  flexShrink: 0,
                }}
              >
                {key} ({fullName}):
              </span>
              <span
                style={{
                  fontSize: '32px',
                  fontWeight: 'bold',
                  fontFamily: FONTS.date,
                  color: EXPORT_COLORS.foreground,
                  letterSpacing: '3px',
                  width: '150px',
                  flexShrink: 0,
                }}
              >
                {date}
              </span>
              {explanation && (
                <span
                  style={{
                    fontSize: '15px',
                    fontFamily: FONTS.label,
                    color: EXPORT_COLORS.foreground,
                    flex: 1,
                  }}
                >
                  {explanation}
                </span>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Footer */}
      <div
        style={{
          marginTop: '4px',
          textAlign: 'center',
          flexShrink: 0,
        }}
      >
        <span
          style={{
            fontSize: '14px',
            fontFamily: FONTS.label,
            color: EXPORT_COLORS.foreground,
          }}
        >
          Made with aifuturesmodel.com
        </span>
      </div>
    </div>
  );
});

CombinedChartExport.displayName = 'CombinedChartExport';
