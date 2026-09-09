'use client';

import { useState, useCallback, useEffect, useRef, useMemo, ChangeEvent, Fragment } from 'react';
import { usePathname, useRouter, useSearchParams } from 'next/navigation';
import CustomHorizonChart from './CustomHorizonChart';
import CombinedComputeChart from './CombinedComputeChart';
import { tooltipBoxStyle, tooltipHeaderStyle, tooltipValueStyle } from './chartTooltipStyle';
import { ChartDataPoint, BenchmarkPoint } from '@/app/types';
import { CustomMetricChart } from './CustomMetricChart';
import type { DataPoint } from './CustomLineChart';

import { formatTo3SigFigs, formatTimeDuration, formatSCHorizon, yearsToMinutes, formatYearMonth, formatYearMonthShort, formatWorkTimeDuration, formatWorkTimeDurationDetailed, formatUplift, formatCompactNumberNode, formatAsPowerOfTenText, formatDollarsPerYear } from '@/utils/formatting';
import { DEFAULT_PARAMETERS, ParametersType, ParameterPrimitive, areParametersAtDefaults, PARAMETER_PRESETS, PRESET_BASELINES, CURRENT_DEFAULT_BASELINE_ID, resolveParamValue } from '@/constants/parameters';
import { CHART_LAYOUT } from '@/constants/chartLayout';
import { convertParametersToAPIFormat, convertSampledParametersToAPIFormat, ParameterRecord } from '@/utils/monteCarlo';
import { ParameterSlider } from './ParameterSlider';
import { AdvancedSections } from './AdvancedSections';
import { ChartType } from '@/types/chartConfig';
import { CHART_CONFIGS } from '@/utils/chartConfigs';
import { ChartSyncProvider } from './ChartSyncContext';
import { ChartLoadingOverlay } from './ChartLoadingOverlay';
import { encodeFullStateToParams, decodeFullStateFromParams, DEFAULT_CHECKBOX_STATES, DEFAULT_PAUSE_STATE, PauseState } from '@/utils/urlState';
import { CHECKBOX_PARAM_OVERRIDES, CHECKBOX_LOCKED_PARAMETERS } from '@/constants/simplificationOverrides';
import { useCheckboxParameterSync } from '@/utils/useCheckboxParameterSync';
import { AIRnDProgressMultiplierChart } from './AIRnDProgressMultiplierChart';
import { MilestoneTooltipSection } from './MilestoneTooltipSection';
import type { MilestoneMap } from '@/types/milestones';
import { SmallChartMetricTooltip } from './SmallChartMetricTooltip';
import { SMALL_CHART_EXPLANATIONS } from '@/constants/chartExplanations';
import { ParameterHoverProvider } from './ParameterHoverContext';
import { loadBenchmarkData } from '@/utils/benchmarkLoader';
import { loadObservedRevenueData } from '@/utils/revenueLoader';
import { loadEciFrontierData } from '@/utils/eciLoader';
import { HeaderContent } from './HeaderContent';
import { WithChartTooltip } from './ChartTitleTooltip';
import ModelArchitecture from './ModelArchitecture';
import { SharePredictionModal } from './SharePredictionModal';
import { SamplingConfig, generateParameterSample, generateParameterSampleWithFixedParams, generateParameterSampleWithUserValues, getDistributionMedian, initializeCorrelationSampling, extractSamplingConfigBounds } from '@/utils/sampling';
import { exportParametersToFile, importParametersFromFile } from '@/utils/parameterExportImport';
import type { ComputeApiResponse } from '@/lib/serverApi';
import { CsvEditorSidebar, InputTimeSeries, PauseMilestoneOption } from './CsvEditorSidebar';
import { PAUSE_MILESTONE_DEFS } from '@/constants/pauseMilestones';

// X-axis start for the revenue trend chart: one year before the earliest observed
// revenue datapoint, so the whole reported-revenue overlay is visible with some
// lead-in rather than starting exactly on (or after) the first marker.
const OBSERVED_REVENUE_X_START =
  Math.min(...loadObservedRevenueData().map(point => point.year)) - 1;

interface ModelDefaults {
  optimal_ces_eta_init?: number;
  automation_interp_type?: string;
  ai_research_taste_slope?: number;
  anchor_progress_at_strong_cognitive_horizon?: number;
  present_year?: number;
  present_progress?: number;
  progress_at_aa?: number;
  [key: string]: number | string | undefined;
}

interface OdeAttemptDiagnostics {
  method?: string;
  direction?: string;
  human_only?: boolean;
  t_span?: [number, number];
  tolerances?: Record<string, number>;
  success?: boolean;
  message?: string | null;
  exception?: string;
  wall_time_seconds?: number;
  cpu_time_seconds?: number;
  nfev?: number;
  njev?: number;
  nlu?: number;
  step_stats?: {
    num_steps?: number;
    min_step?: number;
    max_step?: number;
    mean_step?: number;
    step_variation_ratio?: number | null;
  };
}

interface ParameterConfig {
  defaults?: ModelDefaults;
  bounds?: Record<string, [number, number]>;
  metadata?: Record<string, unknown>;
}

interface SampleTrajectoryWithParams {
  trajectory: ChartDataPoint[];
  params: Record<string, number | string | boolean>;
}

interface TimeSeriesPoint {
  year: number;
  progress: number;
  effectiveCompute: number;
  horizonLength: number;
  researchStock: number;
  automationFraction?: number;
  trainingCompute?: number;
  trainingSystemPerformance?: number;
  trainingComputeH100e?: number;
  totalRdComputeH100e?: number;
  experimentCapacity?: number;
  aiResearchTaste?: number;
  aiSoftwareProgressMultiplier?: number;
  aiSwProgressMultRefPresentDay?: number;
  serialCodingLaborMultiplier?: number;
  // Newly exposed metrics from API
  humanLabor?: number;
  inferenceCompute?: number;
  experimentCompute?: number;
  researchEffort?: number;
  softwareProgressRate?: number;
  softwareEfficiency?: number;
  etsp?: number;
  trainingSystemPerformanceGrowthRate?: number;
  etspRate?: number;
  progressRate?: number;
  aiCodingLaborMultiplier?: number;
  aiCodingLaborMultRefPresentDay?: number;
  predictedRevenue?: number;
  capabilityEci?: number;
  capabilityEciRate?: number;
  excessSerialCodingLaborUplift?: number;
  // World compute / leading-AI-company compute decomposition (display only)
  worldComputeH100e?: number;
  totalAiCompanyComputeH100e?: number;
  commercialComputeH100e?: number;
  companyShareOfWorldPct?: number;
  companyShareOfWorldPctTotal?: number;
  spendPctInference?: number;
  spendPctTraining?: number;
  spendPctExperiment?: number;
  spendPctCommercial?: number;
  spendPctCumInference?: number;
  spendPctCumTraining?: number;
  spendPctCumExperiment?: number;
  spendPctCumCommercial?: number;
}

// Mode-specific "Key Parameters" slider definitions: when the AC anchor mode is
// not time-horizon, these replace the three horizon sliders (present doubling
// time / AC horizon / doubling difficulty) in the key-parameters area.
type AcAnchorKeyParamDef = {
  paramName: keyof ParametersType & string;
  label: string;
  explanation: string;
  min: number;
  max: number;
  step: number;
  decimalPlaces?: number;
  format: (value: number) => string;
  useLogScale?: boolean;
  // Hide when the underlying parameter is null (opt-in parameters)
  onlyWhenSet?: boolean;
};

const AC_ANCHOR_KEY_PARAM_DEFS: Record<string, AcAnchorKeyParamDef[]> = {
  'revenue': [
    {
      paramName: 'present_revenue',
      label: 'Leading AGI company annualized revenue as of {REF}',
      explanation: "The highest annualized revenue across all frontier model developers, as of {REF}. (Annualized revenue is the most recent month's revenue multiplied by 12.)",
      min: 1e8, max: 1e13, step: 0.1, format: formatDollarsPerYear, useLogScale: true,
    },
    {
      paramName: 'present_revenue_growth_rate',
      label: 'How fast annualized revenue grows as of {REF}',
      explanation: "Annual growth factor of annualized revenue (3 means revenue triples each year). This is used to set a long-term trend, so transient factors should be excluded; it shouldn't literally be the current growth rate.",
      min: 1.1, max: 10, step: 0.01, decimalPlaces: 2, format: (v) => `×${v.toFixed(2)}/yr`, useLogScale: true,
    },
    {
      paramName: 'revenue_required_for_ac',
      label: 'Annualized revenue corresponding to Automated Coder (AC)',
      explanation: 'The annualized revenue of the first frontier model developer that develops AC internally, when they do so.',
      min: 1e10, max: 1e15, step: 0.1, format: formatDollarsPerYear, useLogScale: true,
    },
  ],
  'eci': [
    {
      paramName: 'eci_at_present_day',
      label: 'ECI score as of {REF}',
      explanation: 'Epoch Capabilities Index score as of {REF}; the extrapolated ECI trend is anchored here.',
      min: 100, max: 220, step: 1, decimalPlaces: 0, format: (v) => v.toFixed(0),
    },
    {
      paramName: 'eci_growth_rate',
      label: 'How fast the ECI score grows as of {REF}',
      explanation: 'Instantaneous ECI growth as of {REF}, in points per year.',
      min: 1, max: 50, step: 0.5, decimalPlaces: 1, format: (v) => `${v.toFixed(1)} pts/yr`,
    },
    {
      paramName: 'eci_required_for_ac',
      label: 'ECI score required to achieve Automated Coder (AC)',
      explanation: 'AC is reached when the extrapolated ECI trend crosses this score.',
      min: 180, max: 500, step: 1, decimalPlaces: 0, format: (v) => v.toFixed(0),
    },
  ],
  'uplift-solved': [
    {
      paramName: 'serial_uplift_at_reference_date',
      label: 'Serial coding uplift as of {REF}',
      explanation: "The most capable model's coding uplift as of {REF}. Coding uplift is the speedup in coding work from AI assistance. An uplift of 2x would mean that AIs were increasing coding productivity by as much as if the AGI project sped up their employees by 2x.",
      min: 1.05, max: 100, step: 0.01, format: (v) => `${formatTo3SigFigs(v)}x`, useLogScale: true,
    },
    {
      paramName: 'excess_uplift_doubling_time',
      label: 'How long it takes serial coding uplift \u2212 1 to double, as of {REF}',
      explanation: 'How long it takes serial coding uplift \u2212 1 to double, as of {REF}. This is used to fit current behavior, the doubling time changes over time.',
      min: 0.6, max: 240, step: 0.1, decimalPlaces: 1, format: (v) => `${v.toFixed(1)} mo`, useLogScale: true,
    },
    {
      paramName: 'serial_uplift_at_ac',
      label: 'Serial coding uplift at AC',
      explanation: 'How much serial coding uplift exists when automation reaches 100%. When set, the logistic asymptote is solved to match it instead of being an input. Toggle this on in Advanced Settings. There is a floor on what is reachable, and it moves with the uplift level and doubling time.',
      min: 2, max: 1000, step: 0.1, format: (v) => `${formatTo3SigFigs(v)}x`, useLogScale: true,
      onlyWhenSet: true,
    },
  ],
};

// Each anchor mode's observables bind at their own reference date (null =>
// present day). {REF} in the defs above is replaced with the live date at
// render time, so the labels track the reference-date sliders.
const AC_ANCHOR_REF_DATE_PARAM: Record<string, keyof ParametersType & string> = {
  'revenue': 'revenue_reference_date',
  'eci': 'eci_reference_date',
  'uplift-solved': 'uplift_reference_date',
};

const withAnchorRefDate = (text: string, acAnchorMode: string, params: ParametersType): string => {
  const dateParam = AC_ANCHOR_REF_DATE_PARAM[acAnchorMode];
  if (!dateParam || !text.includes('{REF}')) return text;
  return text.replaceAll('{REF}', formatYearMonthShort(resolveParamValue(dateParam, params)));
};

const CHART_ANIMATION_DURATION_MS = 450;

const isFiniteNumber = (value: unknown): value is number =>
  typeof value === 'number' && Number.isFinite(value);

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

const interpolateNumericValue = (
  startValue: unknown,
  endValue: unknown,
  progress: number
): number | null => {
  const startNumeric = isFiniteNumber(startValue)
    ? startValue
    : isFiniteNumber(endValue)
      ? endValue
      : null;
  const endNumeric = isFiniteNumber(endValue) ? endValue : startNumeric;

  if (startNumeric == null || endNumeric == null) {
    return null;
  }

  if (progress <= 0) {
    return startNumeric;
  }

  if (progress >= 1) {
    return endNumeric;
  }

  return startNumeric + (endNumeric - startNumeric) * progress;
};

const interpolateChartDataPoints = (
  fromData: ChartDataPoint[],
  toData: ChartDataPoint[],
  progress: number
): ChartDataPoint[] => {
  if (progress >= 1) {
    return toData.map(point => ({ ...point }));
  }

  if (fromData.length === 0) {
    return toData.map(point => ({ ...point }));
  }

  const fromMap = new Map<number, ChartDataPoint>();
  const toMap = new Map<number, ChartDataPoint>();

  fromData.forEach(point => {
    fromMap.set(point.year, point);
  });
  toData.forEach(point => {
    toMap.set(point.year, point);
  });

  const years = Array.from(new Set([...fromMap.keys(), ...toMap.keys()])).sort(
    (a, b) => a - b
  );

  return years.map(year => {
    const startPoint = fromMap.get(year);
    const endPoint = toMap.get(year);

    if (!startPoint && !endPoint) {
      return {
        year,
        horizonLength: 0,
        horizonFormatted: '',
      };
    }

    const resolvedStart = startPoint ?? endPoint!;
    const resolvedEnd = endPoint ?? startPoint!;
    const mergedKeys = new Set([
      ...Object.keys(resolvedStart),
      ...Object.keys(resolvedEnd),
    ]);

    const interpolatedHorizon =
      interpolateNumericValue(
        resolvedStart.horizonLength,
        resolvedEnd.horizonLength,
        progress
      ) ??
      (typeof resolvedEnd.horizonLength === 'number'
        ? resolvedEnd.horizonLength
        : typeof resolvedStart.horizonLength === 'number'
          ? resolvedStart.horizonLength
          : 0);

    const interpolatedPoint: ChartDataPoint = {
      year,
      horizonLength: interpolatedHorizon,
      horizonFormatted:
        progress >= 1
          ? resolvedEnd.horizonFormatted ?? resolvedStart.horizonFormatted ?? ''
          : resolvedStart.horizonFormatted ?? resolvedEnd.horizonFormatted ?? '',
    };

    mergedKeys.forEach(key => {
      if (key === 'year' || key === 'horizonLength' || key === 'horizonFormatted') {
        return;
      }

      const startValue = resolvedStart[key];
      const endValue = resolvedEnd[key];

      if (isFiniteNumber(startValue) || isFiniteNumber(endValue)) {
        interpolatedPoint[key] = interpolateNumericValue(startValue, endValue, progress);
      } else {
        interpolatedPoint[key] = progress >= 1
          ? endValue ?? startValue ?? null
          : startValue ?? endValue ?? null;
      }
    });

    return interpolatedPoint;
  });
};

const VISIBLE_CHART_START_YEAR = 2019;

// The standing ECI panel is cropped to Epoch's observed frontier rather than sharing
// the other primary charts' 2019 left edge: the first frontier point is Feb 2023, so
// four years of axis carried the modelled trend with nothing to compare it against.
// A year of lead-in keeps the earliest marker off the axis edge. Static data, so this
// is computed once at module scope. Null only if the frontier dataset is ever empty,
// in which case the chart falls back to its dynamic domain.
const ECI_CHART_LEAD_IN_YEARS = 1;
const ECI_CHART_START_YEAR: number | null = (() => {
  // loadEciFrontierData returns points oldest release first.
  const [firstPoint] = loadEciFrontierData();
  return firstPoint ? firstPoint.year - ECI_CHART_LEAD_IN_YEARS : null;
})();
const SIMULATION_END_YEAR = 2200; // Integrate over the full input time series (to 2200)
const CODING_AUTOMATION_MARKER_COLOR = '#6b7280';
// Headroom above the AC tick on the uplift chart's log y-axis, as a factor:
// 100 = two orders of magnitude. The series runs to the 1e8 serial-multiplier
// ceiling shortly after AC, so the top is pinned to the tick rather than to the
// data; this is how much of the post-AC climb stays visible above it.
const UPLIFT_CHART_HEADROOM_ABOVE_AC = 100.0;
// Fixed floor for the uplift chart's log y-axis. Fixed rather than data-driven
// so the axis does not rescale as parameters move; the x-axis is trimmed to
// match so no empty space is spent on the stretch below it.
const UPLIFT_CHART_Y_FLOOR = 0.005;
// Benchmark data will be loaded dynamically from YAML file

type MilestoneDisplay = {
  month: string;
  monthNumber: number;
  year: string;
};

const getMilestoneDisplay = (decimalYear: number | null): MilestoneDisplay | null => {
  if (decimalYear == null || !Number.isFinite(decimalYear)) {
    return null;
  }

  const year = Math.floor(decimalYear);
  const fraction = decimalYear - year;
  const monthIndexRaw = Math.round(fraction * 12);

  const monthIndex = monthIndexRaw % 12;
  const adjustedYear = monthIndexRaw === 12 ? year + 1 : year;

  const months = [
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'
  ];

  return {
    month: months[monthIndex],
    monthNumber: monthIndex + 1,
    year: String(adjustedYear),
  };
};

interface KeyMilestoneValueProps {
  label: string;
  milestone: MilestoneDisplay | null;
  fallback: string;
  isLoading: boolean;
  explanation?: string;
}

const KeyMilestoneValue = ({
  label,
  milestone,
  fallback,
  isLoading,
  explanation,
}: KeyMilestoneValueProps) => (
  <div className="mt-3">
    <div className="flex items-center gap-2">
      <p className="leading-tight text-[12px] font-semibold text-primary text-left font-system-mono">{label}</p>
      <div className="flex-1 border-t border-gray-500/30" />
    </div>
    <div className="mt-1">
      {isLoading ? (
        <div className="h-12 w-2/3 animate-pulse rounded bg-slate-200/70" />
      ) : milestone ? (
        <div className="flex items-center gap-2">
          <div className="text-4xl font-et-book font-bold text-primary mb-0 leading-none flex justify-between flex-1">
            {`${String(milestone.monthNumber).padStart(2, '0')}/${milestone.year}`.split('').map((char, idx) => (
              <span key={idx}>{char}</span>
            ))}
          </div>
          {explanation && (
            <WithChartTooltip explanation={explanation} tooltipPlacement="left">
              <span className="sr-only">Info</span>
            </WithChartTooltip>
          )}
        </div>
      ) : (
        <div className="text-xl font-semibold text-slate-500">{fallback}</div>
      )}
    </div>
  </div>
);

interface KeyMilestonePanelProps {
  automationYear: number | null;
  asiYear: number | null;
  isLoading: boolean;
  onShareClick?: () => void;
  height?: number;
}

const KeyMilestonePanel = ({
  automationYear,
  asiYear,
  isLoading,
  onShareClick,
  height,
}: KeyMilestonePanelProps) => {
  const automationMilestone = getMilestoneDisplay(automationYear);
  const asiMilestone = getMilestoneDisplay(asiYear);

  return (
    <section
      className="flex h-full flex-col gap-1"
      style={{
        minWidth: CHART_LAYOUT.keyStats.width,
        maxWidth: 350,
        minHeight: height ?? CHART_LAYOUT.primary.height,
      }}
    >
      <div className="flex min-h-0 flex-col gap-0 justify-between h-full">
        <div>
          <div className="flex gap-2 items-center my-1">
            <span className="leading-tight text-[12px] font-semibold text-primary text-left font-system-mono">Explanation</span>
            <div className="flex-1 border-t border-gray-500/30" />
          </div>
          <p className="prose text-[16px] lg:text-[14px] mb-2">
            {`This website presents the AI Futures Model (Aug 2026 version), \
following up on the `}
            <a href="https://ai-2027.com/research/timelines-forecast" className="text-blue-600 underline" target="_blank" rel="noopener noreferrer">timelines</a>
            {` and `}
            <a href="https://ai-2027.com/research/takeoff-forecast" className="text-blue-600 underline" target="_blank" rel="noopener noreferrer">takeoff</a>
            {` models we published alongside `}
            <a href="https://ai-2027.com/" className="text-blue-600 underline" target="_blank" rel="noopener noreferrer">AI 2027</a>
            {`. Predicting superhuman AI capabilities inherently requires much intuition and guesswork, but we've nonetheless found quantitative modeling to be useful.`}
          </p>
          <p className="prose text-[16px] lg:text-[14px] mb-2">
            {`To the right you can see our model's output with each parameter set to \
co-author Daniel Kokotajlo's median estimate, last updated Aug 16, 2026. The Dec 2025 model version is accessible `}
            <a href="https://v1.aifuturesmodel.com" className="text-blue-600 underline" target="_blank" rel="noopener noreferrer">here</a>
            {`. Read about how our forecasts have changed since AI 2027 `}
            <a href="https://blog.ai-futures.org/p/clarifying-how-our-ai-timelines-forecasts" className="text-blue-600 underline" target="_blank" rel="noopener noreferrer">here</a>
            {` and `}
            <a href="https://blog.aifutures.org/p/q1-2026-timelines-update" className="text-blue-600 underline" target="_blank" rel="noopener noreferrer">here</a>
            {`. See Daniel's forecasted probability distributions on the `}
            <a href="https://aifuturesmodel.com/forecast" className="text-blue-600 underline" target="_blank" rel="noopener noreferrer">Forecasts page</a>
            {`.`}
          </p>
          {/* <p className="prose text-[13px]">
          {`Beginning at the bottom of the page is an explanation of our model, \
including motivation and limitations. If you scroll down to the core model description, \
you'll see a diagram of how our model works, and you can click on variables in the \
diagram to navigate to the corresponding part of the description.`}
        </p> */}
        </div>
        <div className="shrink-0">
          <KeyMilestoneValue
            label="Date of Automated Coder (AC)"
            milestone={automationMilestone}
            fallback="After 2200"
            isLoading={isLoading}
            explanation="An AC can fully automate an AGI project's coding work, autonomously replacing the AGI project’s entire software engineering staff."
          />
          <KeyMilestoneValue
            label="Date of Superintelligence (ASI)"
            milestone={asiMilestone}
            fallback="After 2200"
            isLoading={isLoading}
            explanation="The gap between an ASI and the best humans is 2× greater than the gap between the best humans and the median professional, at virtually all cognitive tasks."
          />
        </div>

      </div>
    </section>
  );
};

// Types
interface InitialSampleTrajectory {
  trajectory: Array<{
    year: number;
    horizonLength: number;
    effectiveCompute: number;
    automationFraction: number;
    trainingCompute?: number;
    trainingSystemPerformance?: number;
    aiSwProgressMultRefPresentDay?: number;
  }>;
  params: Record<string, number | string | boolean>;
}

interface ProgressChartProps {
  benchmarkData?: BenchmarkPoint[];
  modelDescriptionGDocPortionMarkdown?: React.ReactNode;
  initialComputeData?: ComputeApiResponse;
  initialParameters?: ParametersType;
  initialSampleTrajectories?: InitialSampleTrajectory[];
  initialSeed?: number;
}

// Tooltip for the AI-company compute spend decomposition small chart: one line per
// spend category (percent of the company's total compute), colored to match the bands.
const SPEND_SHARE_TOOLTIP_ROWS: Array<{ label: string; color: string; key: string }> = [
  { label: 'Automation inference', color: '#dc2626', key: 'spendPctInference' },
  { label: 'Training', color: '#d97706', key: 'spendPctTraining' },
  { label: 'Experiments', color: '#059669', key: 'spendPctExperiment' },
  { label: 'Commercial', color: '#64748b', key: 'spendPctCommercial' },
];

function SpendShareTooltip({ point }: { point: DataPoint }) {
  const maybeYear = point['year'];
  const year = typeof maybeYear === 'number' ? maybeYear : point.x;
  return (
    <div style={tooltipBoxStyle}>
      <span style={tooltipHeaderStyle}>{formatYearMonth(year)}</span>
      {SPEND_SHARE_TOOLTIP_ROWS.map(({ label, color, key }) => {
        const raw = point[key];
        if (typeof raw !== 'number' || !Number.isFinite(raw)) return null;
        return (
          <span key={key} style={{ ...tooltipValueStyle, color }}>
            {label}: {raw.toFixed(1)}%
          </span>
        );
      })}
      <span style={{ color: 'var(--vivid-foreground)' }}>{SMALL_CHART_EXPLANATIONS.spendPctCumCommercial}</span>
    </div>
  );
}

// Process initial compute data into ChartDataPoint format
function processInitialData(data: ComputeApiResponse | undefined): ChartDataPoint[] {
  if (!data?.success || !data.time_series?.length) return [];

  return data.time_series
    .filter((point) => point.year >= VISIBLE_CHART_START_YEAR)
    .sort((a, b) => a.year - b.year)
    .map((point) => ({
      year: point.year,
      horizonLength: point.horizonLength,
      horizonFormatted: formatWorkTimeDuration(point.horizonLength),
      effectiveCompute: point.effectiveCompute,
      automationFraction: point.automationFraction,
      trainingCompute: point.trainingCompute ?? null,
      trainingSystemPerformance: point.trainingSystemPerformance ?? null,
      trainingComputeH100e: point.trainingComputeH100e ?? null,
      totalRdComputeH100e: point.totalRdComputeH100e ?? null,
      experimentCapacity: point.experimentCapacity,
      aiResearchTaste: point.aiResearchTaste,
      aiSoftwareProgressMultiplier: point.aiSoftwareProgressMultiplier,
      aiSwProgressMultRefPresentDay: point.aiSwProgressMultRefPresentDay,
      serialCodingLaborMultiplier: point.serialCodingLaborMultiplier,
      humanLabor: point.humanLabor ?? null,
      inferenceCompute: point.inferenceCompute ?? null,
      experimentCompute: point.experimentCompute ?? null,
      researchEffort: point.researchEffort ?? null,
      researchStock: point.researchStock ?? null,
      softwareProgressRate: point.softwareProgressRate ?? null,
      softwareEfficiency: point.softwareEfficiency ?? null,
      etsp: point.etsp ?? null,
      trainingSystemPerformanceGrowthRate: point.trainingSystemPerformanceGrowthRate ?? null,
      etspRate: point.etspRate ?? null,
      progressRate: point.progressRate ?? null,
      aiCodingLaborMultiplier: point.aiCodingLaborMultiplier ?? null,
      aiCodingLaborMultRefPresentDay: point.aiCodingLaborMultRefPresentDay ?? null,
      predictedRevenue: point.predictedRevenue ?? null,
      capabilityEci: point.capabilityEci ?? null,
      capabilityEciRate: point.capabilityEciRate ?? null,
      excessSerialCodingLaborUplift: point.excessSerialCodingLaborUplift ?? null,
      worldComputeH100e: point.worldComputeH100e ?? null,
      totalAiCompanyComputeH100e: point.totalAiCompanyComputeH100e ?? null,
      commercialComputeH100e: point.commercialComputeH100e ?? null,
      companyShareOfWorldPct: point.companyShareOfWorldPct ?? null,
      companyShareOfWorldPctTotal: point.companyShareOfWorldPctTotal ?? null,
      spendPctInference: point.spendPctInference ?? null,
      spendPctTraining: point.spendPctTraining ?? null,
      spendPctExperiment: point.spendPctExperiment ?? null,
      spendPctCommercial: point.spendPctCommercial ?? null,
      spendPctCumInference: point.spendPctCumInference ?? null,
      spendPctCumTraining: point.spendPctCumTraining ?? null,
      spendPctCumExperiment: point.spendPctCumExperiment ?? null,
      spendPctCumCommercial: point.spendPctCumCommercial ?? null,
    }));
}

export default function ProgressChart({
  benchmarkData = [],
  modelDescriptionGDocPortionMarkdown,
  initialComputeData,
  initialParameters,
  initialSampleTrajectories = [],
  initialSeed = 12345
}: ProgressChartProps) {
  // Initialize state directly from server-provided data
  const initialChartData = useMemo(() => processInitialData(initialComputeData), [initialComputeData]);
  const resolvedInitialParameters = useMemo(() => initialParameters ?? { ...DEFAULT_PARAMETERS }, [initialParameters]);

  // Convert server sample trajectories to client format
  const convertedInitialSamples = useMemo(() => {
    return initialSampleTrajectories.map(sample => ({
      trajectory: sample.trajectory.map(point => ({
        year: point.year,
        horizonLength: point.horizonLength,
        effectiveCompute: point.effectiveCompute,
        automationFraction: point.automationFraction,
        trainingCompute: point.trainingCompute ?? null,
        trainingSystemPerformance: point.trainingSystemPerformance ?? null,
        aiSwProgressMultRefPresentDay: point.aiSwProgressMultRefPresentDay,
      })) as ChartDataPoint[],
      params: sample.params
    }));
  }, [initialSampleTrajectories]);

  const [chartData, setChartData] = useState<ChartDataPoint[]>(initialChartData);
  const [animatedChartData, setAnimatedChartData] = useState<ChartDataPoint[]>(initialChartData);
  const [scHorizonMinutes, setScHorizonMinutes] = useState<number>(
    Math.pow(10, DEFAULT_PARAMETERS.ac_time_horizon_minutes)
  );
  const [activePopover, setActivePopover] = useState<ChartType | null>(null);
  const [parameters, setParameters] = useState<ParametersType>(resolvedInitialParameters);
  // Separate UI parameters for immediate slider updates vs committed parameters for API calls
  const [uiParameters, setUiParameters] = useState<ParametersType>(resolvedInitialParameters);
  const [isDragging, setIsDragging] = useState(false);
  const [allParameters, setAllParameters] = useState<ParameterConfig | null>(null);
  const [trajectoryData, setTrajectoryData] = useState<ChartDataPoint[]>([]);
  const [milestones, setMilestones] = useState<MilestoneMap | null>(
    initialComputeData?.milestones && typeof initialComputeData.milestones === 'object'
      ? (initialComputeData.milestones as MilestoneMap)
      : null
  );
  const [mainLoading, setMainLoading] = useState(false);
  // The homepage passes no initialComputeData, so the chart is empty until the
  // first client-side compute returns. Until then an empty chart means "still
  // loading", not "no data" — gate the error state on this flag.
  const [hasCompletedInitialFetch, setHasCompletedInitialFetch] = useState(initialChartData.length > 0);
  const [isParameterConfigReady, setIsParameterConfigReady] = useState(false);
  const [isAdvancedPanelOpen, setIsAdvancedPanelOpen] = useState(false);
  const [selectedPresetId, setSelectedPresetId] = useState<string>('');
  const baselineParameters = useMemo(() => {
    const id = selectedPresetId || CURRENT_DEFAULT_BASELINE_ID;
    return PRESET_BASELINES[id] ?? DEFAULT_PARAMETERS;
  }, [selectedPresetId]);
  const [horizonParams, setHorizonParams] = useState<{ uses_shifted_form: boolean; anchor_progress: number | null } | null>(
    initialComputeData?.horizon_params ?? null
  );
  const [summary, setSummary] = useState<{ beta_software?: number; r_software?: number; slope_times_log_f?: number; [key: string]: unknown } | null>(
    initialComputeData?.summary as { beta_software?: number; r_software?: number; slope_times_log_f?: number; [key: string]: unknown } | null ?? null
  );
  const [hiddenCharts, setHiddenCharts] = useState<Set<string>>(new Set());
  // Third primary panel: ECI by default, effective/training compute on request.
  // See the ThirdPanelView type near primaryCharts for why this is view-only state.
  const [thirdPanelView, setThirdPanelView] = useState<'eci' | 'effective-compute'>('eci');

  // Sample trajectories state
  const [samplingConfig, setSamplingConfig] = useState<SamplingConfig | null>(null);
  const [sampleTrajectories, setSampleTrajectories] = useState<SampleTrajectoryWithParams[]>(convertedInitialSamples);
  const [isSamplingEnabled, setIsSamplingEnabled] = useState(true);
  const NUM_SAMPLES = 10; // Number of sample trajectories to show

  // Debug state for sample trajectories
  const [showSamplingDebug, setShowSamplingDebug] = useState(false);
  const [expandedTrajectoryIndex, setExpandedTrajectoryIndex] = useState<number | null>(null);
  const [showParamToggles, setShowParamToggles] = useState(false);
  const [enabledSamplingParams, setEnabledSamplingParams] = useState<Set<string>>(new Set());
  const [resampleTrigger, setResampleTrigger] = useState(0);
  // Trigger for sample trajectories - incremented when main request completes
  const [mainRequestCompleteTrigger, setMainRequestCompleteTrigger] = useState(0);
  const [samplingDebugLog, setSamplingDebugLog] = useState<Array<{
    id: number;
    timestamp: Date;
    type: 'request' | 'response' | 'error';
    params?: Record<string, unknown>;
    dataPoints?: number;
    error?: string;
  }>>([]);

  // Profiling state
  const [profilingEnabled, setProfilingEnabled] = useState(false);
  const [profilingData, setProfilingData] = useState<{
    totalTimeSeconds: number;
    totalCpuTimeSeconds?: number;
    timingBreakdown: Record<string, number>;
    cpuTimingBreakdown?: Record<string, number>;
    odeDiagnostics?: OdeAttemptDiagnostics[];
    modelInternal?: unknown;
    stats: string;
    timestamp: Date;
  } | null>(null);

  // Custom input time series state
  const [customTimeSeries, setCustomTimeSeries] = useState<InputTimeSeries | null>(null);
  const [defaultTimeSeries, setDefaultTimeSeries] = useState<InputTimeSeries | null>(null);
  const [isCsvEditorOpen, setIsCsvEditorOpen] = useState(false);

  // Pause mode state (lifted from CsvEditorSidebar for URL sync)
  const [pauseState, setPauseState] = useState<PauseState>(DEFAULT_PAUSE_STATE);

  // Fetch default input data on mount so we have it for the CSV editor
  useEffect(() => {
    fetch('/api/default-data')
      .then(res => res.json())
      .then(data => {
        if (data.time_series) {
          setDefaultTimeSeries(data.time_series as InputTimeSeries);
        }
      })
      .catch(() => { /* ignore - non-critical */ });
  }, []);

  const handleCsvEditorChange = useCallback((data: InputTimeSeries) => {
    setCustomTimeSeries(data);
  }, []);

  const handleCsvEditorReset = useCallback(() => {
    setCustomTimeSeries(null);
  }, []);

  // Initialize enabled params when sampling config loads
  // Extract bounds from sampling config for slider ranges
  // These ensure sliders stay within the valid sampling ranges
  const samplingConfigBounds = useMemo(() => {
    if (!samplingConfig) return {};
    return extractSamplingConfigBounds(samplingConfig);
  }, [samplingConfig]);

  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const [isUrlSyncReady, setIsUrlSyncReady] = useState(false);
  const [isShareModalOpen, setIsShareModalOpen] = useState(false);

  const MOBILE_PRIMARY_CHART_HEIGHT = 540;
  const [primaryChartHeight, setPrimaryChartHeight] = useState<number>(CHART_LAYOUT.primary.height);

  useEffect(() => {
    if (typeof window === 'undefined') {
      return;
    }
    const desktopQuery = window.matchMedia('(min-width: 1024px)'); // Tailwind `lg`
    const update = () => {
      setPrimaryChartHeight(desktopQuery.matches ? CHART_LAYOUT.primary.height : MOBILE_PRIMARY_CHART_HEIGHT);
    };
    update();

    desktopQuery.addEventListener('change', update);
    return () => desktopQuery.removeEventListener('change', update);
  }, []);

  const [enableExperimentAutomation, setEnableExperimentAutomation] = useState(DEFAULT_CHECKBOX_STATES.enableExperimentAutomation);
  const [useExperimentThroughputCES, setUseExperimentThroughputCES] = useState(DEFAULT_CHECKBOX_STATES.useExperimentThroughputCES);
  const [enableSoftwareProgress, setEnableSoftwareProgress] = useState(DEFAULT_CHECKBOX_STATES.enableSoftwareProgress);
  const [useComputeLaborGrowthSlowdown, setUseComputeLaborGrowthSlowdown] = useState(DEFAULT_CHECKBOX_STATES.useComputeLaborGrowthSlowdown);

  // Horizon chart benchmark percentile toggle (p50 vs p80)
  const [horizonPercentile, setHorizonPercentile] = useState<'p50' | 'p80'>('p80');
  const effectiveBenchmarkData = useMemo(
    () => horizonPercentile === 'p80' ? benchmarkData : loadBenchmarkData('p50'),
    [benchmarkData, horizonPercentile]
  );

  // Flag to prevent checkbox effects during initial URL hydration
  const isHydratingFromUrlRef = useRef(false);

  // Locked parameters based on checkbox state. Sourced from CHECKBOX_LOCKED_PARAMETERS
  // so a parameter cannot be overridden without also being locked.
  const lockedParameters = useMemo(() => {
    const locked = new Set<string>();
    if (!enableExperimentAutomation) {
      CHECKBOX_LOCKED_PARAMETERS.enableExperimentAutomation.forEach(k => locked.add(k));
    }
    if (!useExperimentThroughputCES) {
      CHECKBOX_LOCKED_PARAMETERS.useExperimentThroughputCES.forEach(k => locked.add(k));
    }
    if (!enableSoftwareProgress) {
      CHECKBOX_LOCKED_PARAMETERS.enableSoftwareProgress.forEach(k => locked.add(k));
    }
    return locked;
  }, [enableExperimentAutomation, useExperimentThroughputCES, enableSoftwareProgress]);

  const scHorizonLogBounds = useMemo(() => {
    // Priority: sampling config bounds > API bounds > fallback
    // Note: samplingConfigBounds already has ac_time_horizon_minutes converted to log10
    const samplingBounds = samplingConfigBounds?.ac_time_horizon_minutes;
    if (samplingBounds?.min !== undefined && samplingBounds?.max !== undefined) {
      return {
        min: samplingBounds.min,
        max: samplingBounds.max
      };
    }
    const rawBounds = allParameters?.bounds?.ac_time_horizon_minutes;
    if (rawBounds && rawBounds[0] > 0 && rawBounds[1] > 0) {
      return {
        min: Math.log10(rawBounds[0]),
        max: Math.log10(rawBounds[1])
      };
    }
    return { min: 3, max: 11 };
  }, [allParameters, samplingConfigBounds]);

  const parallelPenaltyBounds = useMemo(() => {
    // Priority: sampling config bounds > API bounds > fallback
    const samplingBounds = samplingConfigBounds?.parallel_penalty;
    if (samplingBounds?.min !== undefined && samplingBounds?.max !== undefined) {
      return {
        min: samplingBounds.min,
        max: samplingBounds.max
      };
    }
    const rawBounds = allParameters?.bounds?.parallel_penalty;
    if (rawBounds) {
      return {
        min: rawBounds[0],
        max: rawBounds[1]
      };
    }
    return { min: 0, max: 1 };
  }, [allParameters, samplingConfigBounds]);

  // Ref for debouncing Monte Carlo updates
  const monteCarloTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  // Ref for request deduplication
  const currentMonteCarloRequestRef = useRef<AbortController | null>(null);
  // Ref to cancel in-flight chart computations when parameters change quickly
  const chartRequestControllerRef = useRef<AbortController | null>(null);
  const latestChartRequestIdRef = useRef(0);
  const animationFrameRef = useRef<number | null>(null);
  const animationStartRef = useRef<number>(0);
  const animationFromRef = useRef<ChartDataPoint[]>([]);
  const animationToRef = useRef<ChartDataPoint[]>([]);
  const animatedDataRef = useRef<ChartDataPoint[]>([]);
  const importInputRef = useRef<HTMLInputElement | null>(null);
  const scrollContainerRef = useRef<HTMLDivElement | null>(null);

  const commitAnimatedChartData = useCallback((nextData: ChartDataPoint[]) => {
    animatedDataRef.current = nextData;
    setAnimatedChartData(nextData);
  }, []);

  const cancelChartAnimation = useCallback(() => {
    if (animationFrameRef.current !== null) {
      cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }
  }, []);

  const startChartAnimation = useCallback(
    (nextData: ChartDataPoint[], options: { immediate?: boolean } = {}) => {
      const shouldSkipAnimation =
        options.immediate ?? animatedDataRef.current.length === 0;

      if (shouldSkipAnimation) {
        cancelChartAnimation();
        animationFromRef.current = nextData;
        animationToRef.current = nextData;
        animationStartRef.current = 0;
        commitAnimatedChartData(nextData);
        return;
      }

      animationFromRef.current = animatedDataRef.current;
      animationToRef.current = nextData;
      animationStartRef.current = performance.now();
      cancelChartAnimation();

      const step = (timestamp: number) => {
        const elapsed = timestamp - animationStartRef.current;
        const progress = Math.min(1, elapsed / CHART_ANIMATION_DURATION_MS);
        const interpolated = interpolateChartDataPoints(
          animationFromRef.current,
          animationToRef.current,
          progress
        );
        commitAnimatedChartData(interpolated);

        if (progress < 1) {
          animationFrameRef.current = requestAnimationFrame(step);
        } else {
          animationFrameRef.current = null;
        }
      };

      animationFrameRef.current = requestAnimationFrame(step);
    },
    [cancelChartAnimation, commitAnimatedChartData]
  );

  useEffect(() => {
    return () => {
      cancelChartAnimation();
    };
  }, [cancelChartAnimation]);

  // Keyboard shortcut to toggle sampling debug panel (Alt+D)
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Use Alt+D to avoid browser shortcut conflicts
      if (e.altKey && (e.key === 'D' || e.key === 'd')) {
        e.preventDefault();
        setShowSamplingDebug(prev => !prev);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  // Handle hash scrolling after streamed content loads
  // Browser's default scroll-to-hash happens before Suspense content streams in,
  // so we manually scroll once the target element appears
  useEffect(() => {
    if (typeof window === 'undefined') return;
    
    const hash = window.location.hash;
    if (!hash) return;
    
    const targetId = hash.slice(1);
    if (!targetId) return;

    const scrollToTarget = () => {
      const element = document.getElementById(targetId);
      if (element) {
        element.scrollIntoView({ behavior: 'instant', block: 'start' });
        return true;
      }
      return false;
    };

    // If element already exists, scroll immediately
    if (scrollToTarget()) return;

    // Otherwise wait for streamed content
    const observer = new MutationObserver(() => {
      if (scrollToTarget()) {
        observer.disconnect();
      }
    });

    observer.observe(document.body, { childList: true, subtree: true });

    // Cleanup after 10 seconds
    const timeout = setTimeout(() => observer.disconnect(), 10000);

    return () => {
      observer.disconnect();
      clearTimeout(timeout);
    };
  }, []);

  // Memoized popover handlers to prevent chart re-renders
  const handlePopoverOpen = useCallback((type: ChartType) => {
    setActivePopover(type);
  }, []);

  const handlePopoverClose = useCallback(() => {
    setActivePopover(null);
  }, []);

  const handleExportParameters = useCallback(() => {
    exportParametersToFile(uiParameters);
  }, [uiParameters]);

  const handleExportCSV = useCallback(() => {
    if (!chartData || chartData.length === 0) {
      console.warn('No chart data available to export');
      return;
    }

    // Define CSV columns matching Flask app export format
    const headers = [
      'year',
      'horizon_length_minutes',
      'effective_compute',
      'coding_automation_fraction',
      'training_compute',
      'training_system_performance',
      'experiment_throughput',
      'ai_research_taste',
      'ai_software_rnd_uplift_ref_present_day',
      'serial_coding_labor_multiplier',
      'parallel_coding_labor_multiplier',
      'human_labor',
      'automation_compute',
      'experiment_H100e',
      'research_effort',
      'research_stock',
      'software_efficiency_growth_rate',
      'software_efficiency',
      'effective_training_system_performance',
      'training_H100e',
      'total_rnd_compute_h100e',
      'world_compute_h100e',
      'total_ai_company_compute_h100e',
      'commercial_compute_h100e',
      'company_share_of_world_pct',
    ];

    // Build CSV content
    const csvRows: string[] = [];
    csvRows.push(headers.join(','));

    for (const point of chartData) {
      const row = [
        point.year,
        point.horizonLength ?? '',
        point.effectiveCompute ?? '',
        point.automationFraction ?? '',
        point.trainingCompute ?? '',
        point.trainingSystemPerformance ?? '',
        point.experimentCapacity ?? '',
        point.aiResearchTaste ?? '',
        point.aiSwProgressMultRefPresentDay ?? '',
        point.serialCodingLaborMultiplier ?? '',
        point.aiCodingLaborMultiplier ?? '',
        point.humanLabor ?? '',
        point.inferenceCompute ?? '',
        point.experimentCompute ?? '',
        point.researchEffort ?? '',
        point.researchStock ?? '',
        point.softwareProgressRate ?? '',
        point.softwareEfficiency ?? '',
        point.etsp ?? '',
        point.trainingComputeH100e ?? '',
        point.totalRdComputeH100e ?? '',
        point.worldComputeH100e ?? '',
        point.totalAiCompanyComputeH100e ?? '',
        point.commercialComputeH100e ?? '',
        point.companyShareOfWorldPct ?? '',
      ];
      csvRows.push(row.join(','));
    }

    const csvContent = csvRows.join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
    link.download = `ai_futures_model_trajectory_${timestamp}.csv`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  }, [chartData]);

  const handleImportParameters = useCallback(async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const imported = await importParametersFromFile(file);
    if (imported) {
      setUiParameters(imported);
      setParameters(imported);
      setIsDragging(false);
    }
    event.target.value = '';
  }, [setUiParameters, setParameters, setIsDragging]);

  const handleImportButtonClick = useCallback(() => {
    importInputRef.current?.click();
  }, []);

  const handlePresetSelect = useCallback((event: ChangeEvent<HTMLSelectElement>) => {
    const presetId = event.target.value;
    setSelectedPresetId(presetId);

    // Apply preset parameters if a preset is selected
    if (presetId && PRESET_BASELINES[presetId]) {
      const newParams = { ...PRESET_BASELINES[presetId] };

      // If on homepage, do full navigation to /p with preset params
      if (pathname === '/' && !areParametersAtDefaults(newParams)) {
        const urlParams = encodeFullStateToParams({
          parameters: newParams,
          presetId,
          enableExperimentAutomation,
          useExperimentThroughputCES,
          enableSoftwareProgress,
          useComputeLaborGrowthSlowdown,
          horizonPercentile,
          pause: pauseState,
        });
        window.location.href = `/p?${urlParams.toString()}`;
        return;
      }

      setUiParameters(newParams);
      setParameters(newParams);
    }
  }, [setUiParameters, setParameters, pathname, enableExperimentAutomation, useExperimentThroughputCES, enableSoftwareProgress, useComputeLaborGrowthSlowdown, pauseState]);

  // Commit UI parameters to actual parameters when dragging ends
  const commitParameters = useCallback((nextParameters?: ParametersType) => {
    const newParams = nextParameters ?? uiParameters;
    setParameters(newParams);
    setIsDragging(false);

    // If on homepage and params are non-default, do full navigation to /p
    if (pathname === '/' && !areParametersAtDefaults(newParams)) {
      const urlParams = encodeFullStateToParams({
        parameters: newParams,
        presetId: selectedPresetId || CURRENT_DEFAULT_BASELINE_ID,
        enableExperimentAutomation,
        useExperimentThroughputCES,
        enableSoftwareProgress,
        useComputeLaborGrowthSlowdown,
        horizonPercentile,
        pause: pauseState,
      });
      window.location.href = `/p?${urlParams.toString()}`;
    }
  }, [uiParameters, selectedPresetId, pathname, enableExperimentAutomation, useExperimentThroughputCES, enableSoftwareProgress, useComputeLaborGrowthSlowdown, horizonPercentile, pauseState]);

  // Apply a sample trajectory's parameters to the main charts
  const applySampleTrajectory = useCallback((sampleWithParams: SampleTrajectoryWithParams) => {
    const { trajectory, params } = sampleWithParams;

    // Convert sampled parameters to UI format
    // Key conversion: ac_time_horizon_minutes needs to go from actual minutes to log10
    const uiFormatParams = { ...DEFAULT_PARAMETERS };

    for (const [key, value] of Object.entries(params)) {
      if (key === 'ac_time_horizon_minutes' && typeof value === 'number') {
        // Convert from actual minutes to log10 for UI slider
        uiFormatParams[key] = Math.log10(value);
      } else if (key === 'include_gap') {
        // Convert include_gap back to benchmarks_and_gaps_mode
        uiFormatParams.benchmarks_and_gaps_mode = value === 'gap';
      } else if (key === 'parallel_penalty' && typeof value === 'number') {
        // Convert parallel_penalty back to coding_labor_exponent
        uiFormatParams.coding_labor_exponent = value;
      } else if (key in uiFormatParams) {
        (uiFormatParams as Record<string, ParameterPrimitive>)[key] = value as ParameterPrimitive;
      }
    }

    // Set the parameters to update sliders
    setUiParameters(uiFormatParams);
    setParameters(uiFormatParams);
    setIsDragging(false);

    // Set the chart data to this trajectory
    setChartData(trajectory);
    startChartAnimation(trajectory);
  }, [startChartAnimation]);

  // Sync UI parameters with committed parameters when they change (unless we're dragging)
  useEffect(() => {
    if (!isDragging) {
      setUiParameters(parameters);
    }
  }, [parameters, isDragging]);

  // URL sync - sync checkbox states from URL on mount
  // The server has already handled parameters from URL, so we just need to sync checkbox states
  // which may not be included in initialParameters
  useEffect(() => {
    if (!searchParams || isUrlSyncReady) {
      return;
    }

    // If we're on / with query params, do full redirect to /p
    const currentSearch = searchParams.toString();
    if (pathname === '/' && currentSearch.length > 0) {
      window.location.href = `/p?${currentSearch}`;
      return;
    }

    const urlParams = new URLSearchParams(currentSearch);
    const state = decodeFullStateFromParams(urlParams);

    isHydratingFromUrlRef.current = true;

    // Sync preset and checkbox states from URL
    setSelectedPresetId(state.presetId);
    setEnableExperimentAutomation(state.enableExperimentAutomation);
    setUseExperimentThroughputCES(state.useExperimentThroughputCES);
    setEnableSoftwareProgress(state.enableSoftwareProgress);
    setUseComputeLaborGrowthSlowdown(state.useComputeLaborGrowthSlowdown);
    setHorizonPercentile(state.horizonPercentile);
    setPauseState(state.pause);
    if (state.pause.pauseEnabled) {
      setIsCsvEditorOpen(true);
    }

    // Allow checkbox effects to run after hydration
    setTimeout(() => {
      isHydratingFromUrlRef.current = false;
    }, 0);

    setIsUrlSyncReady(true);
  }, [searchParams, isUrlSyncReady, pathname]);

  useEffect(() => {
    if (!isUrlSyncReady) {
      return;
    }

    const currentPresetId = selectedPresetId || CURRENT_DEFAULT_BASELINE_ID;
    const nextParams = encodeFullStateToParams({
      parameters,
      presetId: currentPresetId,
      enableExperimentAutomation,
      useExperimentThroughputCES,
      enableSoftwareProgress,
      useComputeLaborGrowthSlowdown,
      horizonPercentile,
      pause: pauseState,
    });

    const nextSearch = nextParams.toString();
    const currentSearch = searchParams?.toString() ?? '';
    const isOnParamsPage = pathname === '/p';

    // Navigate back to / only when on baseline preset with no diffs (just base=baseline)
    const isDefaultState = currentPresetId === CURRENT_DEFAULT_BASELINE_ID && areParametersAtDefaults(parameters) && !pauseState.pauseEnabled;

    // Only update URL when on /p
    if (isOnParamsPage && nextSearch !== currentSearch) {
      if (isDefaultState) {
        // Full navigation back to / when resetting to defaults
        window.location.href = '/';
      } else {
        // Update query string on /p (soft navigation is fine here)
        router.replace(`/p?${nextSearch}`, { scroll: false });
      }
    }
  }, [parameters, selectedPresetId, enableExperimentAutomation, useExperimentThroughputCES, enableSoftwareProgress, useComputeLaborGrowthSlowdown, horizonPercentile, pauseState, isUrlSyncReady, router, pathname, searchParams]);

  // Load full parameter configuration on mount
  useEffect(() => {
    let isCancelled = false;

    const loadParameterConfig = async () => {
      try {
        const response = await fetch('/api/parameter-config');
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }
        const data = await response.json();
        if (!isCancelled) {
          // Unwrap the config from the {success, config} wrapper
          setAllParameters(data.config || data);
        }
      } catch (error) {
        if (!isCancelled) {
          console.error('Failed to load parameter configuration:', error);
        }
      } finally {
        if (!isCancelled) {
          setIsParameterConfigReady(true);
        }
      }
    };

    loadParameterConfig();

    return () => {
      isCancelled = true;
    };
  }, []);

  // Load sampling configuration on mount
  useEffect(() => {
    let isCancelled = false;

    const loadSamplingConfig = async () => {
      try {
        const response = await fetch('/api/sampling-config');
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }
        const data = await response.json();
        if (!isCancelled && data.success) {
          // Initialize correlation sampling with the loaded config
          initializeCorrelationSampling(data.config.correlation_matrix);
          // Enable every sampled parameter in the SAME state update as the config: a
          // separate effect would read a stale enabled-set and fire wasted samples.
          // TEMPORARY: Hidden for launch - time series parameters not exposed to users,
          // so time_series_parameters is deliberately not included here.
          setEnabledSamplingParams(new Set(Object.keys(data.config.parameters ?? {})));
          setSamplingConfig(data.config);
        }
      } catch (error) {
        if (!isCancelled) {
          console.error('Failed to load sampling configuration:', error);
        }
      }
    };

    loadSamplingConfig();

    return () => {
      isCancelled = true;
    };
  }, []);

  // API now returns values already in log form (OOMs), so no conversion needed
  const toEffectiveComputeOOM = useCallback((value?: number | null): number | null => {
    if (value === null || value === undefined || !Number.isFinite(value)) {
      return null;
    }
    // Values are already in OOM form from the API, just pass through with clamping
    return value
  }, []);

  const normalizeTrajectoryPoint = useCallback((point: ChartDataPoint): ChartDataPoint => {
    const normalized: ChartDataPoint = {
      ...point,
      // Values already in OOM form from API
      effectiveCompute: toEffectiveComputeOOM(point.effectiveCompute),
    };

    Object.keys(point).forEach(key => {
      if (key.startsWith('effective_compute_trajectory_')) {
        const original = point[key] as number | undefined | null;
        // Values already in OOM form from API
        normalized[key] = toEffectiveComputeOOM(original);
      }
    });

    return normalized;
  }, [toEffectiveComputeOOM]);

  const isVisibleYear = useCallback((year: number | null | undefined): boolean => {
    if (year === null || year === undefined) {
      return false;
    }
    return year >= VISIBLE_CHART_START_YEAR;
  }, []);

  const fetchChartData = useCallback(
    async (params: ParametersType, signal: AbortSignal, requestId: number) => {
      try {
        const apiParameters = convertParametersToAPIFormat(params as unknown as ParameterRecord);
        const requestBody: Record<string, unknown> = {
            parameters: apiParameters,
            time_range: [2017, SIMULATION_END_YEAR],
            initial_progress: 0.0,
            enable_profiling: profilingEnabled,
            // When compute/labor growth slowdown is disabled, use exponential growth inputs
            use_exponential_growth_inputs: !useComputeLaborGrowthSlowdown,
        };
        // If the user has customised the input time series, send it inline
        if (customTimeSeries) {
          requestBody.custom_time_series = customTimeSeries;
        }
        const response = await fetch('/api/compute', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(requestBody),
          signal,
          priority: 'high'
        });

        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }

        const data = await response.json();
        if (signal.aborted) {
          return false;
        }

        if (!data.success) {
          throw new Error(data.error || 'Failed to compute model');
        }

        if (requestId !== latestChartRequestIdRef.current) {
          return false;
        }

        // Store horizon params if available
        if (data.horizon_params) {
          setHorizonParams(data.horizon_params);
        }

        // Store summary data including beta_software
        if (data.summary) {
          setSummary(data.summary);
        }

        // Store profiling data if available
        if (data.profiling?.enabled) {
          const rawTimingBreakdown = isRecord(data.profiling.timing_breakdown)
            ? data.profiling.timing_breakdown
            : {};
          const modelInternal = rawTimingBreakdown.model_internal;
          const numericTimingBreakdown: Record<string, number> = {};
          for (const [key, value] of Object.entries(rawTimingBreakdown)) {
            if (key === 'model_internal') continue;
            if (isFiniteNumber(value)) {
              numericTimingBreakdown[key] = value;
            }
          }
          const rawCpuTimingBreakdown = isRecord(data.profiling.cpu_timing_breakdown)
            ? data.profiling.cpu_timing_breakdown
            : {};
          const numericCpuTimingBreakdown: Record<string, number> = {};
          for (const [key, value] of Object.entries(rawCpuTimingBreakdown)) {
            if (isFiniteNumber(value)) {
              numericCpuTimingBreakdown[key] = value;
            }
          }

          setProfilingData({
            totalTimeSeconds: data.profiling.total_time_seconds,
            totalCpuTimeSeconds: data.profiling.total_cpu_time_seconds,
            timingBreakdown: numericTimingBreakdown,
            cpuTimingBreakdown: numericCpuTimingBreakdown,
            odeDiagnostics: Array.isArray(data.profiling.ode_diagnostics) ? data.profiling.ode_diagnostics : [],
            modelInternal,
            stats: data.profiling.stats,
            timestamp: new Date()
          });
        }

        setMilestones(
          data.milestones && typeof data.milestones === 'object'
            ? (data.milestones as MilestoneMap)
            : null
        );

        if (data.time_series && data.time_series.length > 0) {
          const formattedData: ChartDataPoint[] = data.time_series
            .filter((point: TimeSeriesPoint) => isVisibleYear(point.year))
            .sort((a: TimeSeriesPoint, b: TimeSeriesPoint) => a.year - b.year)
            .map((point: TimeSeriesPoint) => ({
              year: point.year,
              horizonLength: point.horizonLength,
              horizonFormatted: formatWorkTimeDuration(point.horizonLength),
              effectiveCompute: toEffectiveComputeOOM(point.effectiveCompute),
              automationFraction: point.automationFraction,
              trainingCompute: point.trainingCompute ? toEffectiveComputeOOM(point.trainingCompute) : null,
              trainingSystemPerformance: point.trainingSystemPerformance ?? null,
              trainingComputeH100e: point.trainingComputeH100e ?? null,
              totalRdComputeH100e: point.totalRdComputeH100e ?? null,
              experimentCapacity: point.experimentCapacity,
              aiResearchTaste: point.aiResearchTaste,
              aiSoftwareProgressMultiplier: point.aiSoftwareProgressMultiplier,
              aiSwProgressMultRefPresentDay: point.aiSwProgressMultRefPresentDay,
              serialCodingLaborMultiplier: point.serialCodingLaborMultiplier,
              humanLabor: point.humanLabor ?? null,
              inferenceCompute: point.inferenceCompute ?? null,
              experimentCompute: point.experimentCompute ?? null,
              researchEffort: point.researchEffort ?? null,
              researchStock: point.researchStock ?? null,
              softwareProgressRate: point.softwareProgressRate ?? null,
              softwareEfficiency: point.softwareEfficiency ?? null,
              etsp: point.etsp ?? null,
              trainingSystemPerformanceGrowthRate: point.trainingSystemPerformanceGrowthRate ?? null,
              etspRate: point.etspRate ?? null,
              progressRate: point.progressRate ?? null,
              aiCodingLaborMultiplier: point.aiCodingLaborMultiplier ?? null,
              aiCodingLaborMultRefPresentDay: point.aiCodingLaborMultRefPresentDay ?? null,
              predictedRevenue: point.predictedRevenue ?? null,
              capabilityEci: point.capabilityEci ?? null,
              capabilityEciRate: point.capabilityEciRate ?? null,
              excessSerialCodingLaborUplift: point.excessSerialCodingLaborUplift ?? null,
              worldComputeH100e: point.worldComputeH100e ?? null,
              totalAiCompanyComputeH100e: point.totalAiCompanyComputeH100e ?? null,
              commercialComputeH100e: point.commercialComputeH100e ?? null,
              companyShareOfWorldPct: point.companyShareOfWorldPct ?? null,
              companyShareOfWorldPctTotal: point.companyShareOfWorldPctTotal ?? null,
              spendPctInference: point.spendPctInference ?? null,
              spendPctTraining: point.spendPctTraining ?? null,
              spendPctExperiment: point.spendPctExperiment ?? null,
              spendPctCommercial: point.spendPctCommercial ?? null,
              spendPctCumInference: point.spendPctCumInference ?? null,
              spendPctCumTraining: point.spendPctCumTraining ?? null,
              spendPctCumExperiment: point.spendPctCumExperiment ?? null,
              spendPctCumCommercial: point.spendPctCumCommercial ?? null,
            }));

          setChartData(formattedData);
          startChartAnimation(formattedData);
        }

        return true;
      } catch (error) {
        setMilestones(null);
        if (signal.aborted) {
          return false;
        }
        console.error('Error updating chart:', error);
        return false;
      }
    },
    [isVisibleYear, toEffectiveComputeOOM, startChartAnimation, profilingEnabled, useComputeLaborGrowthSlowdown, customTimeSeries]
  );

  // Fetch a single sample trajectory (doesn't set main chart state)
  const fetchSampleTrajectory = useCallback(
    async (sampleParams: Record<string, number | string | boolean>, signal: AbortSignal): Promise<ChartDataPoint[] | null> => {
      try {
        // Use convertSampledParametersToAPIFormat because sampled params already have
        // ac_time_horizon_minutes in actual minutes (not log scale like UI params)
        const apiParameters = convertSampledParametersToAPIFormat(sampleParams as unknown as ParameterRecord);
        const response = await fetch('/api/compute', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            parameters: apiParameters,
            time_range: [2017, SIMULATION_END_YEAR],
            initial_progress: 0.0,
            // When compute/labor growth slowdown is disabled, use exponential growth inputs
            use_exponential_growth_inputs: !useComputeLaborGrowthSlowdown,
          }),
          signal,
          priority: 'low'
        });

        if (!response.ok || signal.aborted) {
          return null;
        }

        const data = await response.json();
        if (!data.success || !data.time_series) {
          return null;
        }

        const formattedData: ChartDataPoint[] = data.time_series
          .filter((point: TimeSeriesPoint) => isVisibleYear(point.year))
          .sort((a: TimeSeriesPoint, b: TimeSeriesPoint) => a.year - b.year)
          .map((point: TimeSeriesPoint) => ({
            year: point.year,
            horizonLength: point.horizonLength,
            effectiveCompute: toEffectiveComputeOOM(point.effectiveCompute),
            trainingCompute: point.trainingCompute ? toEffectiveComputeOOM(point.trainingCompute) : null,
            trainingSystemPerformance: point.trainingSystemPerformance ?? null,
            trainingComputeH100e: point.trainingComputeH100e ?? null,
            aiSwProgressMultRefPresentDay: point.aiSwProgressMultRefPresentDay,
            // The three acAnchorMode trend dataKeys. A sample point that omits the
            // key its chart plots renders as nothing, so leaving these out made the
            // sample trajectories silently invisible on the trend chart in every
            // mode (and on the standalone ECI chart) while the main line drew fine.
            // Keep in step with the trendConfigs dataKeys.
            predictedRevenue: point.predictedRevenue ?? null,
            capabilityEci: point.capabilityEci ?? null,
            excessSerialCodingLaborUplift: point.excessSerialCodingLaborUplift ?? null,
          }));

        return formattedData;
      } catch {
        return null;
      }
    },
    [isVisibleYear, toEffectiveComputeOOM, useComputeLaborGrowthSlowdown]
  );

  // Fetch sample trajectories when sampling config is loaded
  const sampleTrajectoriesAbortRef = useRef<AbortController | null>(null);
  const sampleIdCounterRef = useRef(0);

  // Track if we've already used server-provided samples
  const hasUsedInitialSamplesRef = useRef(convertedInitialSamples.length > 0);

  useEffect(() => {
    if (!samplingConfig || !isSamplingEnabled) {
      return;
    }

    // Skip fetching if we already have server-provided samples (first render only)
    if (hasUsedInitialSamplesRef.current && sampleTrajectories.length > 0) {
      hasUsedInitialSamplesRef.current = false; // Allow future fetches
      return;
    }

    // Cancel any existing sample requests
    if (sampleTrajectoriesAbortRef.current) {
      sampleTrajectoriesAbortRef.current.abort();
    }

    const abortController = new AbortController();
    sampleTrajectoriesAbortRef.current = abortController;

    const fetchAllSamples = async () => {
      // Use current parameters as the base - distributions will be shifted to center on these values
      // Convert UI parameter format to sampling config format:
      // - ac_time_horizon_minutes: UI stores as log10(minutes), sampling expects actual minutes
      // - benchmarks_and_gaps_mode → include_gap
      // - coding_labor_exponent → parallel_penalty
      const rawParams = parameters as unknown as Record<string, number | string | boolean>;
      const userParams: Record<string, number | string | boolean> = { ...rawParams };

      // Convert ac_time_horizon_minutes from log10 to actual minutes
      if (typeof rawParams.ac_time_horizon_minutes === 'number') {
        userParams.ac_time_horizon_minutes = Math.pow(10, rawParams.ac_time_horizon_minutes);
      }

      // Map UI parameter names to sampling config names
      if ('benchmarks_and_gaps_mode' in rawParams) {
        const includeGap = rawParams.benchmarks_and_gaps_mode === true || rawParams.benchmarks_and_gaps_mode === 'gap';
        userParams.include_gap = includeGap ? 'gap' : 'no gap';
      }
      if ('coding_labor_exponent' in rawParams && typeof rawParams.coding_labor_exponent === 'number') {
        userParams.parallel_penalty = rawParams.coding_labor_exponent;
      }

      // Clear debug log when starting fresh
      setSamplingDebugLog([]);

      // Log sampling info for debugging
      if (enabledSamplingParams.size <= 3) {
        console.log('=== SAMPLING DEBUG ===');
        console.log('Enabled params (uncertainty will be sampled):', Array.from(enabledSamplingParams));
        console.log('Distributions are shifted to center on user values:');
        for (const [paramName, distConfig] of Object.entries(samplingConfig.parameters)) {
          if (enabledSamplingParams.has(paramName) && distConfig.dist !== 'fixed') {
            const defaultMedian = getDistributionMedian(distConfig);
            const userValue = userParams[paramName];
            console.log(`  ${paramName}: user=${typeof userValue === 'number' ? userValue.toExponential(4) : userValue}, default=${typeof defaultMedian === 'number' ? defaultMedian.toExponential(4) : defaultMedian} (dist: ${distConfig.dist})`);
          }
        }
        console.log('=== END SAMPLING DEBUG ===');
      }

      // Generate samples and fetch in parallel, but add them as they complete
      // Use generateParameterSampleWithUserValues to shift distributions to user's current values
      // This preserves uncertainty but centers it around the user's choices
      const currentEnabledParams = enabledSamplingParams;
      const samplePromises = Array.from({ length: NUM_SAMPLES }, async (_, index) => {
        const sampleId = ++sampleIdCounterRef.current;
        const sampleParams = generateParameterSampleWithUserValues(samplingConfig, userParams, currentEnabledParams);

        // Log the request with full sampled parameters
        setSamplingDebugLog(prev => [...prev, {
          id: sampleId,
          timestamp: new Date(),
          type: 'request',
          params: sampleParams as Record<string, unknown>,
        }]);

        try {
          const trajectory = await fetchSampleTrajectory(sampleParams, abortController.signal);

          if (abortController.signal.aborted) {
            return null;
          }

          if (trajectory) {
            // Log successful response
            setSamplingDebugLog(prev => [...prev, {
              id: sampleId,
              timestamp: new Date(),
              type: 'response',
              dataPoints: trajectory.length,
            }]);

            // Add trajectory with full params as it completes
            setSampleTrajectories(prev => [...prev, {
              trajectory,
              params: sampleParams as Record<string, number | string | boolean>
            }]);
          } else {
            // Log null response
            setSamplingDebugLog(prev => [...prev, {
              id: sampleId,
              timestamp: new Date(),
              type: 'error',
              error: 'Null trajectory returned',
            }]);
          }
          return trajectory;
        } catch (err) {
          // Log error
          setSamplingDebugLog(prev => [...prev, {
            id: sampleId,
            timestamp: new Date(),
            type: 'error',
            error: String(err),
          }]);
          return null;
        }
      });

      // Wait for all to complete (for cleanup purposes)
      await Promise.allSettled(samplePromises);
    };

    // Clear existing samples and start fresh
    setSampleTrajectories([]);
    fetchAllSamples();

    return () => {
      abortController.abort();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [samplingConfig, isSamplingEnabled, fetchSampleTrajectory, resampleTrigger, enabledSamplingParams, mainRequestCompleteTrigger]);

  // Update the SC horizon line immediately while the slider moves
  useEffect(() => {
    const instantHorizon = Math.pow(10, uiParameters.ac_time_horizon_minutes);
    setScHorizonMinutes(instantHorizon);
  }, [uiParameters.ac_time_horizon_minutes]);

  // Load dynamic Monte Carlo trajectories with streaming based on current slider values
  const loadDynamicMonteCarloTrajectories = useCallback(async (allParams: typeof parameters) => {
    // Cancel any existing request
    if (currentMonteCarloRequestRef.current) {
      currentMonteCarloRequestRef.current.abort();
      currentMonteCarloRequestRef.current = null;
    }

    // Create new AbortController for this request
    const abortController = new AbortController();
    currentMonteCarloRequestRef.current = abortController;

    const trajectoryMap: { [year: number]: ChartDataPoint } = {};

    try {
      const response = await fetch('/api/monte-carlo-dynamic', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sliderParams: allParams,
          timeRange: [2017, SIMULATION_END_YEAR]
        }),
        signal: abortController.signal
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      const reader = response.body?.getReader();
      const decoder = new TextDecoder();

      if (!reader) {
        throw new Error('Response body is not readable');
      }

      let buffer = '';
      while (true) {
        const { done, value } = await reader!.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');

        // Keep the last potentially incomplete line in the buffer
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (!line.trim()) continue;

          try {
            const message = JSON.parse(line);

            if (message.type === 'trajectory' && message.data && Array.isArray(message.data)) {
              // Dynamic endpoint streams individual trajectory arrays - merge by year
              message.data.forEach((point: ChartDataPoint) => {
                if (!isVisibleYear(point.year)) {
                  return;
                }

                const normalizedPoint = normalizeTrajectoryPoint(point);
                const year = normalizedPoint.year;
                if (!trajectoryMap[year]) {
                  trajectoryMap[year] = normalizedPoint;
                }

                Object.keys(normalizedPoint).forEach(key => {
                  if (key.startsWith('trajectory_') || key.startsWith('effective_compute_trajectory_')) {
                    trajectoryMap[year][key] = normalizedPoint[key];
                  }
                });
              });

              // Update trajectories progressively as each one completes
              const mergedTrajectories = Object.values(trajectoryMap)
                .filter((point) => isVisibleYear(point.year))
                .sort((a, b) => a.year - b.year);
              setTrajectoryData(mergedTrajectories);
            } else if (message.type === 'complete') {
              // Final update when all trajectories are complete
              const mergedTrajectories = Object.values(trajectoryMap)
                .filter((point) => isVisibleYear(point.year))
                .sort((a, b) => a.year - b.year);
              setTrajectoryData(mergedTrajectories);
            } else if (message.type === 'trajectory_error') {
              console.warn(`Failed to generate trajectory ${message.index}:`, message.error);
            } else if (message.type === 'error') {
              throw new Error(message.error);
            }
          } catch (parseError) {
            console.warn('Failed to parse streaming line:', line, parseError);
          }
        }
      }

    } catch (error: unknown) {
      // Ignore aborted requests - they were cancelled intentionally
      if (error instanceof Error && error.name === 'AbortError') {
        return;
      }
      console.error('Failed to load dynamic Monte Carlo trajectories:', error);
      setTrajectoryData([]);
    } finally {
      // Clear the request reference when done (whether success or error)
      if (currentMonteCarloRequestRef.current === abortController) {
        currentMonteCarloRequestRef.current = null;
      }
    }
  }, [normalizeTrajectoryPoint, isVisibleYear]);

  const scheduleMonteCarloUpdate = useCallback((allParams: typeof parameters, skipDelay: boolean = false) => {
    if (monteCarloTimeoutRef.current) {
      clearTimeout(monteCarloTimeoutRef.current);
    }

    setTrajectoryData([]);
    if (skipDelay) {
      loadDynamicMonteCarloTrajectories(allParams);
      return;
    }
    monteCarloTimeoutRef.current = setTimeout(() => {
      monteCarloTimeoutRef.current = null;
      loadDynamicMonteCarloTrajectories(allParams);
    }, 500);
  }, [loadDynamicMonteCarloTrajectories]);

  const cancelMonteCarloUpdate = useCallback(() => {
    if (monteCarloTimeoutRef.current) {
      clearTimeout(monteCarloTimeoutRef.current);
      monteCarloTimeoutRef.current = null;
    }

    if (currentMonteCarloRequestRef.current) {
      currentMonteCarloRequestRef.current.abort();
      currentMonteCarloRequestRef.current = null;
    }
  }, []);

  // Track if this is the first parameter change (to skip initial fetch if we have server data)
  const isFirstParameterRenderRef = useRef(true);

  useEffect(() => {
    if (!isParameterConfigReady) {
      return;
    }

    // The homepage no longer prerenders the chart server-side (ProgressChartServer
    // passes no initialComputeData), so we always compute the model client-side on
    // mount. The browser's relative /api/compute call is authenticated to THIS
    // deployment, so the page reflects this deployment's backend rather than the
    // production backend the server-side prerender used to resolve to on previews.
    isFirstParameterRenderRef.current = false;

    const paramsSnapshot = { ...parameters };
    const controller = new AbortController();

    if (chartRequestControllerRef.current) {
      chartRequestControllerRef.current.abort();
    }
    chartRequestControllerRef.current = controller;

    const requestId = latestChartRequestIdRef.current + 1;
    latestChartRequestIdRef.current = requestId;

    let isActive = true;
    setMainLoading(true);

    (async () => {
      const didUpdate = await fetchChartData(paramsSnapshot, controller.signal, requestId);
      const isCurrentRequest = latestChartRequestIdRef.current === requestId;

      if (!isActive || controller.signal.aborted || !isCurrentRequest) {
        return;
      }

      setMainLoading(false);
      setHasCompletedInitialFetch(true);
      if (didUpdate) {
        // Trigger sample trajectories after main request completes
        setMainRequestCompleteTrigger(prev => prev + 1);
      } else {
        cancelMonteCarloUpdate();
      }
    })();

    return () => {
      isActive = false;
      if (chartRequestControllerRef.current === controller) {
        chartRequestControllerRef.current = null;
      }
      controller.abort();
      cancelMonteCarloUpdate();
      if (latestChartRequestIdRef.current === requestId) {
        setMainLoading(false);
      }
    };
  }, [parameters, fetchChartData, scheduleMonteCarloUpdate, cancelMonteCarloUpdate, isParameterConfigReady]);

  useEffect(() => () => {
    cancelMonteCarloUpdate();
    if (chartRequestControllerRef.current) {
      chartRequestControllerRef.current.abort();
      chartRequestControllerRef.current = null;
    }
  }, [cancelMonteCarloUpdate]);

  // Sync checkbox states with parameter values
  // When checked (default): uses DEFAULT_PARAMETERS
  // When unchecked: applies nonDefaultParameterValues and locks parameters
  useCheckboxParameterSync({
    checkboxValue: enableExperimentAutomation,
    checkboxKey: 'enableExperimentAutomation',
    nonDefaultParameterValues: CHECKBOX_PARAM_OVERRIDES.enableExperimentAutomation,
    currentParameters: parameters,
    setParameters,
    setUiParameters,
    isHydratingFromUrlRef,
  });

  useCheckboxParameterSync({
    checkboxValue: useExperimentThroughputCES,
    checkboxKey: 'useExperimentThroughputCES',
    nonDefaultParameterValues: CHECKBOX_PARAM_OVERRIDES.useExperimentThroughputCES,
    currentParameters: parameters,
    setParameters,
    setUiParameters,
    isHydratingFromUrlRef,
  });

  useCheckboxParameterSync({
    checkboxValue: enableSoftwareProgress,
    checkboxKey: 'enableSoftwareProgress',
    nonDefaultParameterValues: CHECKBOX_PARAM_OVERRIDES.enableSoftwareProgress,
    currentParameters: parameters,
    setParameters,
    setUiParameters,
    isHydratingFromUrlRef,
  });



  const resolvedChartData = animatedChartData.length > 0 ? animatedChartData : chartData;

  // Merge chartData with trajectoryData for rendering (computed, not stored)
  const renderData = useMemo(() => {
    if (!Array.isArray(trajectoryData) || trajectoryData.length === 0) {
      return resolvedChartData;
    }

    const trajectoryLookup = new Map<string, ChartDataPoint>();
    trajectoryData.forEach(tp => {
      const key = Number(tp.year).toFixed(6);
      trajectoryLookup.set(key, tp);
    });

    const merged = resolvedChartData.map(point => {
      const lookupKey = Number(point.year).toFixed(6);
      const trajectoryPoint = trajectoryLookup.get(lookupKey);
      if (!trajectoryPoint) {
        return point;
      }

      const trajectoryFields: Record<string, string | number | null | undefined> = {};
      Object.keys(trajectoryPoint).forEach(key => {
        const rawValue = trajectoryPoint[key as keyof ChartDataPoint];

        if (key.startsWith('trajectory_')) {
          const numeric = typeof rawValue === 'number' ? rawValue : Number(rawValue);
          if (!Number.isFinite(numeric) || numeric <= 0) {
            trajectoryFields[key] = null;
          } else {
            trajectoryFields[key] = numeric
          }
        } else if (key.startsWith('effective_compute_trajectory_')) {
          const numeric = typeof rawValue === 'number' ? rawValue : Number(rawValue);
          if (!Number.isFinite(numeric)) {
            trajectoryFields[key] = null;
          } else {
            trajectoryFields[key] = numeric;
          }
        }
      });
      return { ...point, ...trajectoryFields };
    });

    return merged;
  }, [resolvedChartData, trajectoryData]);

  // Calculate the year when coding automation horizon is reached
  const codingAutomationHorizonYear = useMemo(() => {
    // Find the first year where horizonLength >= scHorizonMinutes
    const point = resolvedChartData.find(p =>
      p.horizonLength != null &&
      typeof p.horizonLength === 'number' &&
      p.horizonLength >= scHorizonMinutes
    );
    return point?.year ?? null;
  }, [resolvedChartData, scHorizonMinutes]);

  const asiMilestoneYear = useMemo(() => {
    if (!milestones) {
      return null;
    }
    const raw = milestones['ASI']?.time;
    return typeof raw === 'number' && Number.isFinite(raw) ? raw : null;
  }, [milestones]);

  // AC milestone time from backend (uses interpolation for precise timing)
  const acMilestoneYear = useMemo(() => {
    if (!milestones) {
      return null;
    }
    const raw = milestones['AC']?.time;
    return typeof raw === 'number' && Number.isFinite(raw) ? raw : null;
  }, [milestones]);

  // Milestone dots for the ECI chart. Times come from the backend milestone dict
  // (the same source the takeoff-period chart uses); the y position is read off the
  // plotted ECI curve, since the milestone dict carries no ECI value of its own.
  const eciMilestoneMarkers = useMemo(() => {
    const markers: Array<{ x: number; label: string }> = [];
    if (acMilestoneYear != null) {
      markers.push({ x: acMilestoneYear, label: 'AC' });
    }
    if (asiMilestoneYear != null) {
      markers.push({ x: asiMilestoneYear, label: 'ASI' });
    }
    return markers;
  }, [acMilestoneYear, asiMilestoneYear]);

  // Dynamic display end year: show at least to 2030, extend to show automation achievement, max 2045
  const displayEndYear = useMemo(() => {
    const MIN_END_YEAR = 2030;
    const MAX_END_YEAR = SIMULATION_END_YEAR; // 2200 (full input series)

    const targetYear = asiMilestoneYear ?? codingAutomationHorizonYear;
    if (targetYear == null) {
      return MAX_END_YEAR; // Show full simulation if no milestones
    }

    return Math.min(MAX_END_YEAR, Math.max(MIN_END_YEAR, targetYear) + 0.5);
  }, [asiMilestoneYear, codingAutomationHorizonYear]);

  // Sorted solver output times of the current trajectory, used by the CSV editor
  // to snap a milestone pause to the first datapoint after the milestone.
  const trajectoryTimes = useMemo<number[]>(() => {
    return chartData.map(p => p.year).filter(y => Number.isFinite(y)).sort((a, b) => a - b);
  }, [chartData]);

  // Curated pause-trigger milestones with their year from the current trajectory.
  // Year is null when the milestone is not reached within the simulation.
  const pauseMilestoneOptions = useMemo<PauseMilestoneOption[]>(() => {
    return PAUSE_MILESTONE_DEFS.map((m) => {
      const raw = milestones?.[m.key]?.time;
      const year = typeof raw === 'number' && Number.isFinite(raw) ? raw : null;
      return { key: m.key, name: m.name, fullName: m.fullName, year };
    });
  }, [milestones]);

  // Compute share URL from current state (not window.location.href which may be stale)
  const shareUrl = useMemo(() => {
    if (typeof window === 'undefined') return '';
    const params = encodeFullStateToParams({
      parameters,
      presetId: selectedPresetId || CURRENT_DEFAULT_BASELINE_ID,
      enableExperimentAutomation,
      useExperimentThroughputCES,
      enableSoftwareProgress,
      useComputeLaborGrowthSlowdown,
      horizonPercentile,
      pause: pauseState,
    });
    const queryString = params.toString();
    const base = `${window.location.origin}${pathname}`;
    return queryString ? `${base}?${queryString}` : base;
  }, [parameters, selectedPresetId, enableExperimentAutomation, useExperimentThroughputCES, enableSoftwareProgress, useComputeLaborGrowthSlowdown, horizonPercentile, pauseState, pathname]);

  // Excess uplift the series reaches at AC (the AC tick's y). Used to cap the
  // uplift chart's y-axis: past AC the series runs away to the
  // max_serial_coding_labor_multiplier ceiling (1e8), which would otherwise
  // compress the entire pre-AC story into the bottom sliver of the plot.
  const upliftAtAcYear = useMemo(() => {
    const acYear = acMilestoneYear ?? codingAutomationHorizonYear;
    if (acYear == null || !Array.isArray(renderData) || renderData.length < 2) return null;
    for (let i = 1; i < renderData.length; i += 1) {
      const a = renderData[i - 1] as { year?: number; excessSerialCodingLaborUplift?: number | null };
      const b = renderData[i] as { year?: number; excessSerialCodingLaborUplift?: number | null };
      if (a.year == null || b.year == null) continue;
      if (acYear >= a.year && acYear <= b.year) {
        const ya = a.excessSerialCodingLaborUplift;
        const yb = b.excessSerialCodingLaborUplift;
        if (typeof ya !== 'number' || typeof yb !== 'number') return null;
        const t = b.year === a.year ? 0 : (acYear - a.year) / (b.year - a.year);
        const v = ya + (yb - ya) * t;
        return Number.isFinite(v) && v > 0 ? v : null;
      }
    }
    return null;
  }, [acMilestoneYear, codingAutomationHorizonYear, renderData]);

  // First year the series reaches the y floor. The x-axis starts here so the
  // long, uninformative run-up below 0.005x is not drawn.
  const upliftXStart = useMemo(() => {
    if (!Array.isArray(renderData) || renderData.length < 2) return null;
    for (let i = 1; i < renderData.length; i += 1) {
      const a = renderData[i - 1] as { year?: number; excessSerialCodingLaborUplift?: number | null };
      const b = renderData[i] as { year?: number; excessSerialCodingLaborUplift?: number | null };
      const ya = a.excessSerialCodingLaborUplift;
      const yb = b.excessSerialCodingLaborUplift;
      if (a.year == null || b.year == null || typeof ya !== 'number' || typeof yb !== 'number') continue;
      if (ya < UPLIFT_CHART_Y_FLOOR && yb >= UPLIFT_CHART_Y_FLOOR) {
        // interpolate in log space, matching how the axis reads it
        const la = Math.log(Math.max(ya, Number.MIN_VALUE));
        const lb = Math.log(yb);
        const lf = Math.log(UPLIFT_CHART_Y_FLOOR);
        const t = lb === la ? 0 : (lf - la) / (lb - la);
        const yr = a.year + (b.year - a.year) * t;
        return Number.isFinite(yr) ? yr : a.year;
      }
    }
    return null;
  }, [renderData]);

  const automationReferenceLine = useMemo(() => {
    // Use the backend's AC milestone time, not the time-horizon crossing.
    // codingAutomationHorizonYear is "first grid year where horizonLength >=
    // scHorizonMinutes", which equals AC only in time-horizon mode -- and even
    // there only to the 0.1yr output grid. In the revenue / ECI / uplift-solved
    // modes AC is defined by something else entirely, so it would put the line in
    // the wrong place. Falls back to codingAutomationHorizonYear while milestones
    // are still loading.
    const acYear = acMilestoneYear ?? codingAutomationHorizonYear;
    if (acYear == null) {
      return undefined;
    }

    return {
      x: acYear,
      stroke: CODING_AUTOMATION_MARKER_COLOR,
      strokeWidth: 1,
      strokeOpacity: 0.8,
    };
  }, [acMilestoneYear, codingAutomationHorizonYear]);

  // Loading state for charts
  const isChartLoading = mainLoading || !hasCompletedInitialFetch;
  const isInitialMilestoneLoading = isChartLoading && resolvedChartData.length === 0;

  // Small chart configurations and paging (snap per page)
  type SmallChartDef = {
    key: string;
    title: string;
    tooltip: (p: DataPoint) => React.ReactNode;
    yScale?: 'linear' | 'log';
    domain?: [number, number];
    yFormatter?: (v: number) => string | React.ReactNode;
    // Explicit tag to control layout rows irrespective of yScale defaulting
    scaleType: 'linear' | 'log';
    logSuffix?: string;
    /** If true, the data values are already in log form (OOMs) */
    isDataInLogForm?: boolean;
    /** If set, render a stacked-area decomposition instead of a single line. */
    stackAreas?: Array<{ key: string; fill: string; stroke: string; name?: string }>;
  };

  // Tooltip for Horizon Length chart
  const HorizonTooltip = useCallback((point: ChartDataPoint | { x: number;[key: string]: number | null | undefined | string }) => {
    const year = 'year' in point ? point.year : (point.x as number);
    const horizonLength = 'horizonLength' in point ? point.horizonLength : null;

    return (
      <div style={tooltipBoxStyle}>
        <span style={tooltipHeaderStyle}>{formatYearMonth(year as number)}</span>
        {horizonLength != null && typeof horizonLength === 'number' && !isNaN(horizonLength) && (
          <span style={tooltipValueStyle}>{formatWorkTimeDurationDetailed(horizonLength as number)}</span>
        )}
      </div>
    );
  }, []);

  // Tooltip for AI R&D Progress Multiplier chart
  const AIProgressMultiplierTooltip = useCallback((point: ChartDataPoint | { x: number;[key: string]: number | null | undefined | string }) => {
    const year = 'year' in point ? point.year : (point.x as number);
    const value = 'aiSwProgressMultRefPresentDay' in point ? point.aiSwProgressMultRefPresentDay : null;

    return (
      <div style={tooltipBoxStyle}>
        <span style={tooltipHeaderStyle}>{formatYearMonth(year as number)}</span>
        {value != null && typeof value === 'number' && !isNaN(value) && value > 0 && (
          <span style={tooltipValueStyle}>
            {formatUplift(value as number)}x
          </span>
        )}
        <MilestoneTooltipSection point={point} />
      </div>
    );
  }, []);

  type PrimaryChartConfig = {
    id: string;
    title: string;
    render: () => React.ReactNode;
  };

  // Which chart the third primary panel shows. Unlike the AC-anchor dropdown on the
  // horizon panel -- which writes a model parameter and forces a recompute -- this is
  // pure view state: both charts read the trajectory the model already returned, so
  // switching is free and changes nothing about the run.
  type ThirdPanelView = 'eci' | 'effective-compute';

  const acAnchorMode = String(uiParameters.ac_anchor_mode ?? 'time-horizon');

  const revenueRequiredForAc = Number(uiParameters.revenue_required_for_ac);
  const eciRequiredForAc = Number(uiParameters.eci_required_for_ac);
  const serialUpliftAtReferenceDate = resolveParamValue(
    'serial_uplift_at_reference_date', uiParameters);
  const upliftReferenceDate = resolveParamValue(
    'uplift_reference_date', uiParameters);
  const excessUpliftDoublingTime = Number(uiParameters.excess_uplift_doubling_time);
  // The mode pins a doubling RATE at the reference date, so the meaningful
  // reference on the chart is a sloped tangent, not a horizontal threshold.
  // In calendar time the imposed rate is exactly excess_uplift_doubling_time
  // months (the anchor-month unit is defined at the reference date's own
  // progress rate, so D_progress / rate(t*) = D_months / 12 identically).
  // The real curve leaves this line as progress accelerates away from rate(t*)
  // -- that divergence is the point of drawing it.
  const upliftTangentAt = useCallback((year: number) => (
    (serialUpliftAtReferenceDate - 1) * Math.pow(2, ((year - upliftReferenceDate) * 12) / excessUpliftDoublingTime)
  ), [serialUpliftAtReferenceDate, upliftReferenceDate, excessUpliftDoublingTime]);

  // Same construction for the other anchored trends: each pins a (level, rate)
  // pair at its own reference date, so the calendar-time tangent there is the
  // constant-rate exponential. The plotted trend is exponential in PROGRESS,
  // which accelerates, so the curve peels away from the tangent -- the same
  // divergence the uplift chart draws.
  const presentRevenue = Number(uiParameters.present_revenue);
  const presentRevenueGrowthRate = Number(uiParameters.present_revenue_growth_rate);
  const revenueReferenceDate = resolveParamValue('revenue_reference_date', uiParameters);
  const revenueTangentAt = useCallback((year: number) => (
    presentRevenue * Math.pow(presentRevenueGrowthRate, year - revenueReferenceDate)
  ), [presentRevenue, presentRevenueGrowthRate, revenueReferenceDate]);

  const presentHorizon = Number(uiParameters.present_horizon);
  const presentDoublingTime = Number(uiParameters.present_doubling_time);
  const horizonReferenceDate = resolveParamValue('horizon_reference_date', uiParameters);
  const horizonRefDateLabel = formatYearMonthShort(horizonReferenceDate);
  const horizonTangentAt = useCallback((year: number) => (
    presentHorizon * Math.pow(2, (year - horizonReferenceDate) / presentDoublingTime)
  ), [presentHorizon, presentDoublingTime, horizonReferenceDate]);

  const handleAcAnchorModeSelect = useCallback((value: string) => {
    const next = { ...uiParameters, ac_anchor_mode: value };
    setUiParameters(next);
    commitParameters(next);
  }, [uiParameters, commitParameters]);

  const primaryCharts: PrimaryChartConfig[] = useMemo(
    () => [
      {
        id: 'primary-ai-software-rd',
        title: 'AI Software R&D Uplift',
        render: () => (
          <ChartLoadingOverlay isLoading={isChartLoading} className="flex flex-col">
            <AIRnDProgressMultiplierChart
              chartData={renderData}
              tooltip={AIProgressMultiplierTooltip}
              milestones={milestones}
              displayEndYear={displayEndYear}
              verticalReferenceLine={automationReferenceLine}
              height={primaryChartHeight}
              isLoading={isChartLoading}
            />
          </ChartLoadingOverlay>
        ),
      },
      {
        id: 'primary-coding-horizon',
        title: acAnchorMode === 'revenue'
          ? 'AGI Company Annualized Revenue'
          : acAnchorMode === 'eci'
            ? 'Epoch Capabilities Index'
            : acAnchorMode === 'uplift-solved'
              ? 'Serial Coding Uplift - 1'
              : 'Coding Time Horizon',
        render: () => {
          // The chart title doubles as the AC-anchor-mode switcher: picking a
          // trend here changes ac_anchor_mode (recomputing the model) and swaps
          // the displayed graph to that trend. 'uplift-solved' is the odd one
          // out: its curve is the model's OWN uplift rather than an exogenous
          // extrapolation, and its reference line marks the pinned point rather
          // than an AC threshold (AC is an output in that mode).
          const modeDropdown = (
            <select
              value={acAnchorMode}
              onChange={(e) => handleAcAnchorModeSelect(e.target.value)}
              className="text-[12px] font-semibold text-primary text-left font-system-mono bg-transparent border border-gray-400/40 rounded px-1 py-0.5 cursor-pointer hover:border-gray-400"
              title="AC anchor mode: which extrapolated trend defines the AC milestone"
            >
              <option value="uplift-solved">Serial Coding Uplift - 1</option>
              <option value="revenue">AGI Company Annualized Revenue</option>
              <option value="time-horizon">Coding Time Horizon</option>
              {/* ECI has its own standing chart in the primary row now, so it is no
                  longer offered here. It remains selectable in Advanced Settings and via
                  the URL; keep the option mounted while it IS the active mode so this
                  select never renders a value it has no option for. */}
              {acAnchorMode === 'eci' && <option value="eci">Capability (ECI)</option>}
            </select>
          );

          if (acAnchorMode !== 'time-horizon') {
            const trendConfigs = {
              'revenue': {
                dataKey: 'predictedRevenue' as const,
                chartTitle: 'AGI Company Annualized Revenue',
                yScale: 'log' as const,
                yFormatter: formatDollarsPerYear,
                tooltipFormatter: formatDollarsPerYear,
                requirePositive: true,
                explanation: SMALL_CHART_EXPLANATIONS.predictedRevenue,
                threshold: revenueRequiredForAc,
                thresholdLabel: (t: number) => `AC revenue: ${formatDollarsPerYear(t)}`,
                labelPosition: 'right' as const,
              },
              'eci': {
                dataKey: 'capabilityEci' as const,
                chartTitle: 'Epoch Capabilities Index',
                yScale: 'linear' as const,
                yFormatter: (v: number) => Number(v).toFixed(0),
                tooltipFormatter: (v: number) => v.toFixed(0),
                requirePositive: false,
                explanation: SMALL_CHART_EXPLANATIONS.capabilityEci,
                threshold: eciRequiredForAc,
                thresholdLabel: (t: number) => `AC ECI: ${t.toFixed(0)}`,
                labelPosition: 'right' as const,
              },
              'uplift-solved': {
                dataKey: 'excessSerialCodingLaborUplift' as const,
                chartTitle: 'Serial Coding Uplift - 1',
                yScale: 'log' as const,
                yFormatter: (v: number) => formatAsPowerOfTenText(v, { suffix: 'x' }),
                // The plotted quantity is the EXCESS (M - 1); show the multiplier
                // M alongside it so the hover reads without mental arithmetic.
                tooltipFormatter: (v: number) =>
                  `uplift − 1: ${formatTo3SigFigs(v)}x  (uplift: ${formatTo3SigFigs(v + 1)}x)`,
                requirePositive: true,
                explanation: SMALL_CHART_EXPLANATIONS.excessSerialCodingLaborUplift,
                // No threshold: AC is an output in this mode, and what is pinned
                // is a rate, not a level. Drawn as a tangent below instead.
                threshold: undefined as number | undefined,
                thresholdLabel: undefined as ((t: number) => string) | undefined,
                labelPosition: 'right' as const,
              },
            } as const;
            const trend = trendConfigs[acAnchorMode as keyof typeof trendConfigs] ?? trendConfigs['revenue'];
            const threshold = trend.threshold;
            const hasThreshold = threshold != null && Number.isFinite(threshold) && threshold > 0;
            return (
              <ChartLoadingOverlay isLoading={isChartLoading} className="flex flex-col">
                <div className="flex flex-col" style={{ position: 'relative' }}>
                  <div className="flex gap-2 items-center">
                    {modeDropdown}
                    <div className="flex-1 border-t border-gray-500/30" />
                    <WithChartTooltip
                      explanation={trend.explanation}
                      className="!gap-0"
                      isLoading={isChartLoading}
                    ><></></WithChartTooltip>
                  </div>
                  <CustomMetricChart
                    chartData={renderData}
                    tooltip={(p) => (
                      <SmallChartMetricTooltip
                        point={p}
                        fieldName={trend.dataKey}
                        formatter={trend.tooltipFormatter}
                        requirePositive={trend.requirePositive}
                      />
                    )}
                    title={trend.chartTitle}
                    showTitle={false}
                    dataKey={trend.dataKey}
                    yScale={trend.yScale}
                    yFormatter={trend.yFormatter}
                    displayEndYear={displayEndYear}
                    height={primaryChartHeight}
                    showXAxis
                    margin={{ top: 8, right: 10, left: 4, bottom: 35 }}
                    verticalReferenceLine={automationReferenceLine}
                    horizontalReferenceLine={hasThreshold && trend.thresholdLabel ? {
                      y: threshold,
                      stroke: '#6b7280',
                      label: trend.thresholdLabel(threshold),
                      labelPosition: trend.labelPosition,
                    } : undefined}
                    referenceCurve={acAnchorMode === 'uplift-solved' ? {
                      at: upliftTangentAt,
                      stroke: '#6b7280',
                      strokeDasharray: '6 3',
                      markerX: upliftReferenceDate,
                    } : acAnchorMode === 'revenue' ? {
                      at: revenueTangentAt,
                      stroke: '#6b7280',
                      strokeDasharray: '6 3',
                      markerX: revenueReferenceDate,
                    } : undefined}
                    xDomainOverride={acAnchorMode === 'uplift-solved' && upliftXStart != null
                      ? [upliftXStart, displayEndYear] as [number, number]
                      : acAnchorMode === 'revenue'
                        ? [OBSERVED_REVENUE_X_START, displayEndYear] as [number, number]
                        : undefined}
                    seriesMarkerAtX={acAnchorMode === 'uplift-solved' && (acMilestoneYear ?? codingAutomationHorizonYear) != null ? {
                      x: (acMilestoneYear ?? codingAutomationHorizonYear) as number,
                      stroke: CODING_AUTOMATION_MARKER_COLOR,
                    } : undefined}
                    domainOverride={acAnchorMode === 'uplift-solved' && upliftAtAcYear != null
                      ? () => [UPLIFT_CHART_Y_FLOOR, upliftAtAcYear * UPLIFT_CHART_HEADROOM_ABOVE_AC]
                      : hasThreshold ? (observed) => {
                      const maxObserved = Math.max(...observed);
                      const minObserved = Math.min(...observed);
                      const cap = Math.max(maxObserved, threshold);
                      if (trend.yScale === 'log') {
                        const positive = observed.filter(v => v > 0);
                        const lo = positive.length > 0 ? Math.min(...positive) * 0.9 : 1;
                        return [Math.max(lo, Number.MIN_VALUE), cap * 1.5];
                      }
                      return [minObserved - 5, cap + 10];
                    } : undefined}
                    sampleTrajectories={sampleTrajectories.map(s => s.trajectory)}
                  />
                </div>
              </ChartLoadingOverlay>
            );
          }

          return (
            <ChartLoadingOverlay isLoading={isChartLoading} className="flex flex-col">
              <CustomHorizonChart
                chartData={renderData}
                scHorizonMinutes={scHorizonMinutes}
                tooltip={HorizonTooltip}
                formatTimeDuration={formatWorkTimeDuration}
                benchmarkData={effectiveBenchmarkData}
                className="flex flex-col"
                displayEndYear={displayEndYear}
                height={primaryChartHeight}
                sampleTrajectories={sampleTrajectories.map(s => s.trajectory)}
                isLoading={isChartLoading}
                titleControl={modeDropdown}
                referenceCurve={Number.isFinite(presentHorizon) && presentHorizon > 0
                  && Number.isFinite(presentDoublingTime) && presentDoublingTime > 0 ? {
                    at: horizonTangentAt,
                    stroke: '#6b7280',
                    strokeDasharray: '6 3',
                    markerX: horizonReferenceDate,
                  } : undefined}
              />
            </ChartLoadingOverlay>
          );
        },
      },
      {
        // Standing Epoch Capabilities Index chart, with Epoch's observed ECI frontier
        // scattered over it (CustomMetricChart attaches that overlay by dataKey). This
        // took the slot the combined effective/training compute chart used to hold, and
        // is why 'eci' is no longer offered in the AC-anchor dropdown above: the trend
        // is on screen permanently rather than only when it defines AC.
        id: 'primary-capability-eci',
        title: 'Epoch Capabilities Index',
        render: () => {
          // The AC threshold line only means anything when ECI is what defines AC.
          const hasEciThreshold = acAnchorMode === 'eci'
            && eciRequiredForAc != null
            && Number.isFinite(eciRequiredForAc)
            && eciRequiredForAc > 0;

          const computeChartTitle = enableSoftwareProgress
            ? 'Effective & Training Compute'
            : 'Effective Compute';

          // Pure view switch -- no model parameter is written, nothing recomputes.
          const panelDropdown = (
            <select
              value={thirdPanelView}
              onChange={(e) => setThirdPanelView(e.target.value as ThirdPanelView)}
              className="text-[12px] font-semibold text-primary text-left font-system-mono bg-transparent border border-gray-400/40 rounded px-1 py-0.5 cursor-pointer hover:border-gray-400"
              title="Which chart this panel shows"
            >
              <option value="eci">Epoch Capabilities Index</option>
              <option value="effective-compute">{computeChartTitle}</option>
            </select>
          );

          if (thirdPanelView === 'effective-compute') {
            return (
              <ChartLoadingOverlay isLoading={isChartLoading} className="flex flex-col">
                <CombinedComputeChart
                  chartData={renderData}
                  title={computeChartTitle}
                  titleControl={panelDropdown}
                  verticalReferenceLine={automationReferenceLine}
                  className="flex flex-col"
                  displayEndYear={displayEndYear}
                  height={primaryChartHeight}
                  showTrainingSeries={enableSoftwareProgress}
                  sampleTrajectories={sampleTrajectories.map(s => s.trajectory)}
                  isLoading={isChartLoading}
                />
              </ChartLoadingOverlay>
            );
          }

          return (
            <ChartLoadingOverlay isLoading={isChartLoading} className="flex flex-col">
              <div className="flex flex-col" style={{ position: 'relative' }}>
                <div className="flex gap-2 items-center">
                  {panelDropdown}
                  <div className="flex-1 border-t border-gray-500/30" />
                  <WithChartTooltip
                    explanation={SMALL_CHART_EXPLANATIONS.capabilityEci}
                    className="!gap-0"
                    isLoading={isChartLoading}
                  ><></></WithChartTooltip>
                </div>
                <CustomMetricChart
                  chartData={renderData}
                  tooltip={(p) => (
                    <SmallChartMetricTooltip
                      point={p}
                      fieldName="capabilityEci"
                      formatter={(v) => v.toFixed(0)}
                    />
                  )}
                  title="Epoch Capabilities Index"
                  showTitle={false}
                  dataKey="capabilityEci"
                  milestoneMarkers={eciMilestoneMarkers}
                  yScale="linear"
                  yFormatter={(v) => Number(v).toFixed(0)}
                  displayEndYear={displayEndYear}
                  xDomainOverride={ECI_CHART_START_YEAR != null
                    ? [ECI_CHART_START_YEAR, displayEndYear] as [number, number]
                    : undefined}
                  height={primaryChartHeight}
                  showXAxis
                  margin={{ top: 8, right: 10, left: 4, bottom: 35 }}
                  verticalReferenceLine={automationReferenceLine}
                  horizontalReferenceLine={hasEciThreshold ? {
                    y: eciRequiredForAc,
                    stroke: '#6b7280',
                    label: `AC ECI: ${eciRequiredForAc.toFixed(0)}`,
                    labelPosition: 'right' as const,
                  } : undefined}
                  domainOverride={(observed) => {
                    const maxObserved = Math.max(...observed);
                    const minObserved = Math.min(...observed);
                    const cap = hasEciThreshold ? Math.max(maxObserved, eciRequiredForAc) : maxObserved;
                    return [minObserved - 5, cap + 10];
                  }}
                  sampleTrajectories={sampleTrajectories.map(s => s.trajectory)}
                />
              </div>
            </ChartLoadingOverlay>
          );
        },
      },
    ],
    [
      AIProgressMultiplierTooltip,
      HorizonTooltip,
      automationReferenceLine,
      effectiveBenchmarkData,
      displayEndYear,
      enableSoftwareProgress,
      thirdPanelView,
      formatWorkTimeDuration,
      isChartLoading,
      milestones,
      primaryChartHeight,
      renderData,
      sampleTrajectories,
      scHorizonMinutes,
      acAnchorMode,
      handleAcAnchorModeSelect,
      revenueRequiredForAc,
      eciRequiredForAc,
      serialUpliftAtReferenceDate,
      upliftReferenceDate,
      excessUpliftDoublingTime,
      upliftTangentAt,
      revenueTangentAt,
      revenueReferenceDate,
      horizonTangentAt,
      horizonReferenceDate,
      presentHorizon,
      presentDoublingTime,
      codingAutomationHorizonYear,
      acMilestoneYear,
      eciMilestoneMarkers,
      upliftAtAcYear,
      upliftXStart,
    ],
  );

  const visiblePrimaryCharts = useMemo(
    () => primaryCharts.filter((chart) => !hiddenCharts.has(chart.id)),
    [primaryCharts, hiddenCharts],
  );
  const visiblePrimaryCount = visiblePrimaryCharts.length;
  const primaryGridWidth = visiblePrimaryCount > 0
    ? visiblePrimaryCount * CHART_LAYOUT.primary.width
    : 0;


  const smallCharts: SmallChartDef[] = useMemo(() => ([
    {
      key: 'etspRate',
      title: 'ETSP Rate',
      tooltip: (p) => <SmallChartMetricTooltip point={p} fieldName="etspRate" formatter={formatTo3SigFigs} />,
      scaleType: 'linear'
    },
    {
      key: 'softwareProgressRate',
      title: 'Software Efficiency Growth Rate',
      tooltip: (p) => <SmallChartMetricTooltip point={p} fieldName="softwareProgressRate" formatter={formatTo3SigFigs} />,
      scaleType: 'linear'
    },
    {
      key: 'trainingSystemPerformanceGrowthRate',
      title: 'Training System Performance Growth Rate',
      tooltip: (p) => <SmallChartMetricTooltip point={p} fieldName="trainingSystemPerformanceGrowthRate" formatter={formatTo3SigFigs} />,
      scaleType: 'linear'
    },
    {
      key: 'automationFraction',
      title: 'Coding Automation Fraction',
      tooltip: (p) => <SmallChartMetricTooltip point={p} fieldName="automationFraction" formatter={(v) => `${(v * 100).toFixed(1)}%`} />,
      yFormatter: (v) => `${(v * 100).toFixed(1)}%`,
      scaleType: 'linear'
    },
    {
      // Anchored ECI trend (linear in progress); defines AC when ac_anchor_mode == 'eci'
      key: 'capabilityEci',
      title: 'Epoch Capabilities Index (ECI)',
      tooltip: (p) => <SmallChartMetricTooltip point={p} fieldName="capabilityEci" formatter={(v) => v.toFixed(0)} />,
      yFormatter: (v) => v.toFixed(0),
      scaleType: 'linear'
    },
    {
      // d(ECI)/dt in points per year: the slope of the ECI trend above. Analytic
      // (points_per_progress x d(progress)/dt) rather than differenced, so it is
      // exactly that curve's slope rather than an approximation of it.
      key: 'capabilityEciRate',
      title: 'ECI Growth Rate',
      tooltip: (p) => <SmallChartMetricTooltip point={p} fieldName="capabilityEciRate" formatter={(v) => `${formatTo3SigFigs(v)} pts/yr`} />,
      yFormatter: (v) => formatTo3SigFigs(v),
      scaleType: 'linear'
    },
    {
      key: 'softwareEfficiency',
      title: 'Software Efficiency',
      tooltip: (p) => <SmallChartMetricTooltip point={p} fieldName="softwareEfficiency" formatter={(v) => formatCompactNumberNode(Math.pow(10, v), { renderMode: 'html' })} />,
      yScale: 'log',
      scaleType: 'log',
      yFormatter: (v) => formatAsPowerOfTenText(Math.pow(10, v)),
      isDataInLogForm: true
    },
    {
      // ETSP = instantaneous effective-compute rate (OOMs), decomposed into
      // training system performance (hardware rate) + software efficiency. Values are
      // in OOMs, so the linear OOM axis is the right place to stack the two contributions.
      key: 'etsp',
      title: 'Effective Training System Performance (TSP + SW)',
      tooltip: (p) => <SmallChartMetricTooltip point={p} fieldName="etsp" formatter={(v) => formatCompactNumberNode(Math.pow(10, v), { renderMode: 'html' })} />,
      // OOM metric (like Software Efficiency / Effective Compute): grouped in the
      // "log" section. isDataInLogForm forces a linear-in-OOMs axis (so the stacked
      // bands add correctly) regardless of this tag.
      yScale: 'log',
      scaleType: 'log',
      yFormatter: (v) => formatAsPowerOfTenText(Math.pow(10, v)),
      isDataInLogForm: true,
      // No stackAreas override: CustomMetricChart's STACK_DECOMPOSITIONS already maps
      // this dataKey, and PlaygroundClient's copy of this chart relies on that. Stating
      // the bands twice is how they drifted off the app's series colours unnoticed.
    },
    {
      key: 'aiResearchTaste',
      title: 'AI Research Taste',
      tooltip: (p) => <SmallChartMetricTooltip point={p} fieldName="aiResearchTaste" formatter={(v) => formatCompactNumberNode(v, { renderMode: 'html' })} requirePositive />,
      yScale: 'log',
      scaleType: 'log',
      yFormatter: (v) => formatAsPowerOfTenText(v)
    },
    {
      key: 'aiSwProgressMultRefPresentDay',
      title: 'AI Software R&D Uplift',
      tooltip: (p) => <SmallChartMetricTooltip point={p} fieldName="aiSwProgressMultRefPresentDay" formatter={(v) => formatCompactNumberNode(v, { suffix: 'x', renderMode: 'html' })} requirePositive />,
      yScale: 'log',
      scaleType: 'log',
      logSuffix: 'x',
      yFormatter: (v) => formatAsPowerOfTenText(v, { suffix: 'x' })
    },
    {
      // Anchored revenue trend (exponential in progress); defines AC when ac_anchor_mode == 'revenue'
      key: 'predictedRevenue',
      title: 'AGI Company Annualized Revenue',
      tooltip: (p) => <SmallChartMetricTooltip point={p} fieldName="predictedRevenue" formatter={formatDollarsPerYear} requirePositive />,
      yScale: 'log',
      scaleType: 'log',
      yFormatter: formatDollarsPerYear
    },
    {
      // The model's OWN excess serial coding uplift (M - 1), ref-present-day.
      // This is what 'uplift-solved' mode pins; it is a real model output, not
      // an extrapolated trend.
      key: 'excessSerialCodingLaborUplift',
      title: 'Serial Coding Uplift - 1',
      tooltip: (p) => <SmallChartMetricTooltip point={p} fieldName="excessSerialCodingLaborUplift" formatter={(v) => (<>uplift − 1: {formatCompactNumberNode(v, { suffix: 'x', renderMode: 'html' })} (uplift: {formatCompactNumberNode(v + 1, { suffix: 'x', renderMode: 'html' })})</>)} requirePositive />,
      yScale: 'log',
      scaleType: 'log',
      logSuffix: 'x',
      yFormatter: (v) => formatAsPowerOfTenText(v, { suffix: 'x' })
    },
    {
      key: 'researchStock',
      title: 'Cumulative Research Effort',
      tooltip: (p) => <SmallChartMetricTooltip point={p} fieldName="researchStock" formatter={(v) => formatCompactNumberNode(v, { renderMode: 'html' })} requirePositive />,
      yScale: 'log',
      scaleType: 'log',
      yFormatter: (v) => formatAsPowerOfTenText(v)
    },
    {
      key: 'researchEffort',
      title: 'Software Research Effort',
      tooltip: (p) => <SmallChartMetricTooltip point={p} fieldName="researchEffort" formatter={(v) => formatCompactNumberNode(v, { renderMode: 'html' })} requirePositive />,
      yScale: 'log',
      scaleType: 'log',
      yFormatter: (v) => formatAsPowerOfTenText(v)
    },
    {
      key: 'experimentCapacity',
      title: 'Experiment Throughput',
      tooltip: (p) => <SmallChartMetricTooltip point={p} fieldName="experimentCapacity" formatter={(v) => formatCompactNumberNode(v, { renderMode: 'html' })} requirePositive />,
      yScale: 'log',
      scaleType: 'log',
      yFormatter: (v) => formatAsPowerOfTenText(v)
    },
    // The next three fill one grid column: the coding-labor group, with human labor
    // sitting directly under the AI coding-labor multipliers.
    {
      key: 'serialCodingLaborMultiplier',
      title: 'AI Serial Coding Labor Multiplier',
      tooltip: (p) => <SmallChartMetricTooltip point={p} fieldName="serialCodingLaborMultiplier" formatter={(v) => formatCompactNumberNode(v, { suffix: 'x', renderMode: 'html' })} requirePositive />,
      yScale: 'log',
      scaleType: 'log',
      logSuffix: 'x',
      yFormatter: (v) => formatAsPowerOfTenText(v, { suffix: 'x' })
    },
    {
      key: 'aiCodingLaborMultiplier',
      title: 'AI Parallel Coding Labor Multiplier',
      tooltip: (p) => <SmallChartMetricTooltip point={p} fieldName="aiCodingLaborMultiplier" formatter={(v) => formatCompactNumberNode(v, { suffix: 'x', renderMode: 'html' })} requirePositive />,
      yScale: 'log',
      scaleType: 'log',
      logSuffix: 'x',
      yFormatter: (v) => formatAsPowerOfTenText(v, { suffix: 'x' })
    },
    {
      key: 'humanLabor',
      title: 'Human Coding Labor',
      tooltip: (p) => <SmallChartMetricTooltip point={p} fieldName="humanLabor" formatter={(v) => formatCompactNumberNode(v, { renderMode: 'html' })} requirePositive />,
      yScale: 'log',
      scaleType: 'log',
      yFormatter: (v) => formatAsPowerOfTenText(v)
    },
    {
      key: 'totalRdComputeH100e',
      title: 'Total AI Company R&D Compute',
      tooltip: (p) => <SmallChartMetricTooltip point={p} fieldName="totalRdComputeH100e" formatter={(v) => formatCompactNumberNode(v, { suffix: ' H100e', renderMode: 'html' })} requirePositive />,
      yScale: 'log',
      scaleType: 'log',
      yFormatter: (v) => formatAsPowerOfTenText(v, { suffix: ' H100e' }),
      logSuffix: ' H100e'
    },
    {
      key: 'worldComputeH100e',
      title: 'World Compute',
      tooltip: (p) => <SmallChartMetricTooltip point={p} fieldName="worldComputeH100e" formatter={(v) => formatCompactNumberNode(v, { suffix: ' H100e', renderMode: 'html' })} requirePositive />,
      yScale: 'log',
      scaleType: 'log',
      yFormatter: (v) => formatAsPowerOfTenText(v, { suffix: ' H100e' }),
      logSuffix: ' H100e'
    },
    {
      // Stacked decomposition of the company's total compute into automation inference /
      // training / experiment / commercial spend shares. The dataKey is the top
      // cumulative band (~100%); CustomMetricChart's STACK_DECOMPOSITIONS maps it to
      // the full set of cumulative bands.
      key: 'spendPctCumCommercial',
      title: 'AI Company Compute Spend %',
      tooltip: (p) => <SpendShareTooltip point={p} />,
      yScale: 'linear',
      // scaleType is a LAYOUT tag: 'log' places the chart in the tall 3-row grid.
      // In the 'linear' strip these stacked percent charts each claim a whole extra
      // column (columns = max(ceil(logs/3), linears)), padding the grid with empty
      // log cells, and 100px of height is too cramped for a 4-band decomposition.
      scaleType: 'log',
      domain: [0, 100],
      yFormatter: (v) => `${Math.round(v)}%`
    },
    {
      key: 'companyShareOfWorldPct',
      title: 'AI Company Share of World Compute',
      tooltip: (p) => <SmallChartMetricTooltip point={p} fieldName="companyShareOfWorldPct" formatter={(v) => `${v.toFixed(1)}%`} requirePositive />,
      yScale: 'linear',
      scaleType: 'log', // layout tag: tall grid (see above); the y-axis stays linear
      domain: [0, 100],
      yFormatter: (v) => `${Math.round(v)}%`
    },
  ]), []);

  const visibleSmallCharts = useMemo(
    () => smallCharts.filter((chart) => !hiddenCharts.has(`metric-${chart.key}`)),
    [smallCharts, hiddenCharts],
  );

  type ColumnDef = { log1?: SmallChartDef; log2?: SmallChartDef; log3?: SmallChartDef; linear?: SmallChartDef };
  const columns: ColumnDef[] = useMemo(() => {
    const logs = visibleSmallCharts.filter(c => c.scaleType === 'log');
    const linears = visibleSmallCharts.filter(c => c.scaleType !== 'log');
    const numCols = Math.max(Math.ceil(logs.length / 3), linears.length);
    return Array.from({ length: numCols }, (_, i) => ({
      log1: logs[3 * i],
      log2: logs[3 * i + 1],
      log3: logs[3 * i + 2],
      linear: linears[i],
    })).filter(col => col.log1 || col.log2 || col.log3 || col.linear);
  }, [visibleSmallCharts]);
  const hasLogCharts = useMemo(() => columns.some(col => col.log1 || col.log2 || col.log3), [columns]);
  const hasLinearCharts = useMemo(() => columns.some(col => col.linear), [columns]);
  const hasMetricCharts = hasLogCharts || hasLinearCharts;

  // Mobile layout: separate log and linear charts for 2-screen layout
  const mobileLogCharts = useMemo(() => visibleSmallCharts.filter(c => c.scaleType === 'log'), [visibleSmallCharts]);
  const mobileLinearCharts = useMemo(() => visibleSmallCharts.filter(c => c.scaleType !== 'log'), [visibleSmallCharts]);
  // Each mobile small-charts screen holds a fixed 2x2 of log charts plus a row of
  // two linear charts. Paginate into as many screens as the chart lists need —
  // hard-coding the row counts silently stacked several charts per grid cell once
  // the lists outgrew them.
  const MOBILE_LOG_CHARTS_PER_SCREEN = 4;
  const MOBILE_LINEAR_CHARTS_PER_SCREEN = 2;
  const mobileChartScreens = useMemo(() => {
    const screenCount = Math.max(
      Math.ceil(mobileLogCharts.length / MOBILE_LOG_CHARTS_PER_SCREEN),
      Math.ceil(mobileLinearCharts.length / MOBILE_LINEAR_CHARTS_PER_SCREEN),
    );
    return Array.from({ length: screenCount }, (_, index) => ({
      logCharts: mobileLogCharts.slice(
        index * MOBILE_LOG_CHARTS_PER_SCREEN,
        (index + 1) * MOBILE_LOG_CHARTS_PER_SCREEN,
      ),
      linearCharts: mobileLinearCharts.slice(
        index * MOBILE_LINEAR_CHARTS_PER_SCREEN,
        (index + 1) * MOBILE_LINEAR_CHARTS_PER_SCREEN,
      ),
    }));
  }, [mobileLogCharts, mobileLinearCharts]);

  const SOFTWARE_PROGRESS_CHART_ID = 'primary-ai-software-rd';

  useEffect(() => {
    setHiddenCharts((prev) => {
      const next = new Set(prev);
      const shouldHide = !enableSoftwareProgress;
      const currentlyHidden = next.has(SOFTWARE_PROGRESS_CHART_ID);
      if (shouldHide && !currentlyHidden) {
        next.add(SOFTWARE_PROGRESS_CHART_ID);
        return next;
      }
      if (!shouldHide && currentlyHidden) {
        next.delete(SOFTWARE_PROGRESS_CHART_ID);
        return next;
      }
      return prev;
    });
  }, [enableSoftwareProgress]);

  return (
    <ParameterHoverProvider>
      {/* CSV Editor Sidebar */}
      <CsvEditorSidebar
        isOpen={isCsvEditorOpen}
        onClose={() => setIsCsvEditorOpen(false)}
        data={customTimeSeries ?? defaultTimeSeries}
        defaultData={defaultTimeSeries}
        onChange={handleCsvEditorChange}
        onReset={handleCsvEditorReset}
        pauseEnabled={pauseState.pauseEnabled}
        onPauseEnabledChange={(v) => setPauseState(prev => ({ ...prev, pauseEnabled: v }))}
        pauseMode={pauseState.pauseMode}
        onPauseModeChange={(v) => setPauseState(prev => ({ ...prev, pauseMode: v }))}
        pauseYear={pauseState.pauseYear}
        onPauseYearChange={(v) => setPauseState(prev => ({ ...prev, pauseYear: v }))}
        pauseMilestone={pauseState.pauseMilestone}
        onPauseMilestoneChange={(v) => setPauseState(prev => ({ ...prev, pauseMilestone: v }))}
        pauseMilestoneOptions={pauseMilestoneOptions}
        trajectoryTimes={trajectoryTimes}
        experimentReduction={pauseState.experimentReduction}
        onExperimentReductionChange={(v) => setPauseState(prev => ({ ...prev, experimentReduction: v }))}
        inferenceReduction={pauseState.inferenceReduction}
        onInferenceReductionChange={(v) => setPauseState(prev => ({ ...prev, inferenceReduction: v }))}
        laborReduction={pauseState.laborReduction}
        onLaborReductionChange={(v) => setPauseState(prev => ({ ...prev, laborReduction: v }))}
        trainingReduction={pauseState.trainingReduction}
        onTrainingReductionChange={(v) => setPauseState(prev => ({ ...prev, trainingReduction: v }))}
      />

      {/* Mobile/Tablet Advanced Parameters Drawer */}
      {isAdvancedPanelOpen && (
        <div className="fixed inset-0 z-50 lg:hidden">
          <div className="absolute inset-0 bg-black/50" onClick={() => setIsAdvancedPanelOpen(false)} />
          <div className="absolute inset-y-0 right-0 w-full max-w-sm bg-white shadow-xl flex flex-col">
            <div className="flex items-center justify-between px-4 py-3 border-b border-gray-100">
              <h2 className="text-sm font-semibold text-primary uppercase tracking-[0.2em] font-system-mono">
                Advanced Parameters
              </h2>
              <button
                type="button"
                onClick={() => setIsAdvancedPanelOpen(false)}
                className="p-2 -mr-2 text-gray-500 hover:text-gray-700"
              >
                <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
            <div className="flex-1 overflow-y-auto px-4 py-4">
              <div className="space-y-6">
                <div className="space-y-3">
                  <div className="flex flex-wrap gap-2">
                    <button
                      type="button"
                      onClick={handleExportParameters}
                      className="flex-1 min-w-[120px] px-3 py-2 text-xs font-semibold uppercase tracking-[0.1em] text-white bg-[#2A623D] border border-[#215337] rounded-sm shadow-sm hover:bg-[#245632]"
                    >
                      Export
                    </button>
                    <button
                      type="button"
                      onClick={handleImportButtonClick}
                      className="flex-1 min-w-[120px] px-3 py-2 text-xs font-semibold uppercase tracking-[0.1em] text-[#2A623D] bg-white border border-[#2A623D]/40 rounded-sm shadow-sm hover:bg-[#F4F8F5]"
                    >
                      Import
                    </button>
                  </div>
                  <div className="space-y-2">
                    <label className="block text-xs font-semibold uppercase tracking-[0.15em] text-primary font-system-mono">
                      Presets
                    </label>
                    <select
                      value={selectedPresetId}
                      onChange={handlePresetSelect}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md bg-white text-sm"
                    >
                      <option value="">Select preset</option>
                      {PARAMETER_PRESETS.map((preset) => (
                        <option key={preset.id} value={preset.id}>{preset.label}</option>
                      ))}
                    </select>
                  </div>
                </div>
                <AdvancedSections
                  uiParameters={uiParameters}
                  setUiParameters={setUiParameters}
                  allParameters={allParameters}
                  isDragging={isDragging}
                  setIsDragging={setIsDragging}
                  commitParameters={commitParameters}
                  scHorizonLogBounds={scHorizonLogBounds}
                  parallelPenaltyBounds={parallelPenaltyBounds}
                  summary={summary}
                  lockedParameters={lockedParameters}
                  onToggleTrajectoryDebugger={() => setShowSamplingDebug(prev => !prev)}
                  isTrajectoryDebuggerOpen={showSamplingDebug}
                  samplingConfigBounds={samplingConfigBounds}
                  simplificationCheckboxes={{
                              enableExperimentAutomation,
                    setEnableExperimentAutomation,
                    useExperimentThroughputCES,
                    setUseExperimentThroughputCES,
                    enableSoftwareProgress,
                    setEnableSoftwareProgress,
                    useComputeLaborGrowthSlowdown,
                    setUseComputeLaborGrowthSlowdown,
                            }}
                  onExportCSV={handleExportCSV}
                  onOpenCsvEditor={() => setIsCsvEditorOpen(true)}
                  hasCustomTimeSeries={customTimeSeries !== null}
                  baselineParameters={baselineParameters}
                  horizonPercentile={horizonPercentile}
                  onHorizonPercentileChange={setHorizonPercentile}
                />
              </div>
            </div>
          </div>
        </div>
      )}

      <div className="grid h-screen w-full grid-cols-[minmax(0,1fr)_auto] grid-rows-[minmax(0,1fr)] gap-0">
        <div className="relative col-start-1 row-start-1 flex min-h-0 flex-col">
          <div
            ref={scrollContainerRef}
            className="flex min-h-0 flex-col overflow-y-auto pb-10"
          >
            <HeaderContent variant="inline" className="pt-6 pb-4 pl-6 lg:pl-8 pr-6" onOpenAdvancedParams={() => setIsAdvancedPanelOpen(true)} />
            {resolvedChartData.length === 0 && !isChartLoading && (
              <div className="text-center py-8 text-red-500 text-lg">No chart data available</div>
            )}

            <main className="w-full px-0 lg:px-8">
              {/* Charts Container */}
              <div className="flex flex-col flex-1 min-h-0">
                <ChartSyncProvider>
                  {/* Mobile/Tablet Layout: Horizontal scroll-snap */}
                  <div className="lg:hidden relative w-full">
                    <div
                      className="flex gap-x-4 overflow-x-auto overflow-y-hidden snap-x snap-mandatory scrollbar-none pb-2"
                      style={{
                        scrollBehavior: 'smooth',
                        WebkitOverflowScrolling: 'touch',
                      }}
                    >
                      {/* Column 1: Explanation & Key Dates */}
                      <div
                        className="snap-start flex-shrink-0 pl-6 pr-2"
                        style={{ height: primaryChartHeight }}
                      >
                        <div className="h-full">
                          <KeyMilestonePanel
                            automationYear={acMilestoneYear}
                            asiYear={asiMilestoneYear}
                            isLoading={isInitialMilestoneLoading}
                            onShareClick={() => setIsShareModalOpen(true)}
                            height={primaryChartHeight}
                          />
                        </div>
                      </div>

                      {/* Columns 2-4: Big Charts (one per column) */}
                      {visiblePrimaryCharts.map((chart) => (
                        <div
                          key={chart.id}
                          className="snap-center flex-shrink-0"
                          style={{ width: 'calc(100vw - 48px)', height: primaryChartHeight + 4 }}
                        >
                          <div className="w-full h-full overflow-hidden">
                            {chart.render()}
                          </div>
                        </div>
                      ))}

                      {/* Columns 5..N: Small Charts, paginated so no cell holds two charts */}
                      {hasMetricCharts && mobileChartScreens.map((screen, screenIndex) => (
                        <div
                          key={`mobile-small-charts-${screenIndex}`}
                          className="snap-center flex-shrink-0"
                          style={{ width: 'calc(100vw - 48px)' }}
                        >
                          {/* Row template is the same on every screen so chart heights
                              stay consistent as you swipe, even when a screen has no
                              linear charts left to show. */}
                          <div className="grid grid-cols-2 grid-rows-[auto_1fr_1fr_auto_1fr] gap-x-2 gap-y-1 h-[420px]">
                            {screen.logCharts.length > 0 && (
                              <section className="col-span-2 row-span-3 grid grid-cols-subgrid grid-rows-subgrid">
                                <div className="col-span-2 flex items-center gap-2">
                                  <span className="font-mono small-caps text-[10px] font-semibold text-gray-500">Log Scale</span>
                                  <div className="border-t border-gray-300/50 flex-1" />
                                </div>
                                {screen.logCharts.map((chart) => (
                                  <ChartLoadingOverlay key={chart.key} isLoading={isChartLoading} className="h-full">
                                    <CustomMetricChart
                                      chartData={renderData}
                                      tooltip={chart.tooltip}
                                      title={chart.title}
                                      dataKey={chart.key}
                                      yFormatter={chart.yFormatter || ((v) => formatTo3SigFigs(v))}
                                      yScale={chart.yScale || 'linear'}
                                      domainOverride={chart.domain}
                                      verticalReferenceLine={automationReferenceLine}
                                      className="h-full"
                                      displayEndYear={displayEndYear}
                                      logSuffix={chart.logSuffix}
                                      isDataInLogForm={chart.isDataInLogForm}
                                    />
                                  </ChartLoadingOverlay>
                                ))}
                                {/* Keep the 2-up rhythm when a screen ends on an odd chart */}
                                {screen.logCharts.length % 2 === 1 && <div className="h-full" />}
                              </section>
                            )}
                            {screen.linearCharts.length > 0 && (
                              <section className="col-span-2 row-span-2 grid grid-cols-subgrid grid-rows-subgrid">
                                <div className="col-span-2 flex items-center gap-2">
                                  <span className="font-mono small-caps text-[10px] font-semibold text-gray-500">Linear Scale</span>
                                  <div className="border-t border-gray-300/50 flex-1" />
                                </div>
                                {screen.linearCharts.map((chart) => (
                                  <ChartLoadingOverlay key={chart.key} isLoading={isChartLoading} className="h-full">
                                    <CustomMetricChart
                                      chartData={renderData}
                                      tooltip={chart.tooltip}
                                      title={chart.title}
                                      dataKey={chart.key}
                                      yFormatter={chart.yFormatter || ((v) => formatTo3SigFigs(v))}
                                      yScale={chart.yScale || 'linear'}
                                      domainOverride={chart.domain}
                                      verticalReferenceLine={automationReferenceLine}
                                      className="h-full"
                                      displayEndYear={displayEndYear}
                                      logSuffix={chart.logSuffix}
                                      isDataInLogForm={chart.isDataInLogForm}
                                    />
                                  </ChartLoadingOverlay>
                                ))}
                                {screen.linearCharts.length % 2 === 1 && <div className="h-full" />}
                              </section>
                            )}
                          </div>
                        </div>
                      ))}

                      {/* Column 7: Parameters */}
                      <div className="snap-center flex-shrink-0 px-6" style={{ width: '100vw' }}>
                        <section className="w-full flex flex-col">
                          <div className="flex items-center gap-2 pb-4">
                            <span className="text-[12px] font-semibold text-primary text-left font-system-mono">Parameters</span>
                            <div className="border-t border-gray-500/30 flex-1" />
                            <button
                              onClick={() => {
                                setParameters({ ...DEFAULT_PARAMETERS });
                                setUiParameters({ ...DEFAULT_PARAMETERS });
                              }}
                              className="text-[10px] text-gray-400 hover:text-gray-600 font-system-mono font-bold uppercase tracking-wider"
                            >
                              Reset
                            </button>
                            <div className="w-3 border-t border-gray-500/30" />

                            <button
                              onClick={() => setIsAdvancedPanelOpen(true)}
                              className="text-[10px] text-gray-400 hover:text-gray-600 font-system-mono uppercase tracking-wider"
                            >
                              Advanced
                            </button>
                            {/* <div className="w-3 border-t border-gray-500/30" /> */}
                          </div>
                          <div className="flex flex-col gap-2">
                            {acAnchorMode === 'time-horizon' && (<>
                            {/* Present Doubling Time Slider */}
                            <div className="relative flex w-full flex-col">
                              <div className="flex items-center justify-between mb-2">
                                <label className="block w-full text-xs font-medium text-foreground">
                                  <WithChartTooltip
                                    explanation={`How long it takes the coding time horizon to double, as of ${horizonRefDateLabel} (considered as a long-term trend).`}
                                    fullWidth
                                  >
                                    {`How long does AIs' coding time horizon take to double, as of ${horizonRefDateLabel}?`}
                                  </WithChartTooltip>
                                </label>
                              </div>
                              <ParameterSlider
                                paramName="present_doubling_time"
                                label=""
                                customMin={samplingConfigBounds.present_doubling_time?.min}
                                fallbackMin={0.01}
                                fallbackMax={2.0}
                                step={0.01}
                                decimalPlaces={2}
                                customFormatValue={(years) => formatTimeDuration(yearsToMinutes(years))}
                                value={uiParameters.present_doubling_time}
                                uiParameters={uiParameters}
                                setUiParameters={setUiParameters}
                                allParameters={allParameters}
                                isDragging={isDragging}
                                setIsDragging={setIsDragging}
                                commitParameters={commitParameters}
                                baselineParameters={baselineParameters}
                                useLogScale={true}
                              />
                            </div>
                            {/* SC Time Horizon Slider */}
                            <div className="relative flex w-full flex-col">
                              <div className="flex items-center justify-between mb-2">
                                <label className="block w-full text-xs font-medium text-foreground">
                                  <WithChartTooltip
                                    explanation="If an AI can complete, with 80% reliability, tasks that take a skilled human this long, then an AI can do every type of coding work as well as the average company engineer."
                                    fullWidth
                                  >
                                    Automated Coder Time Horizon Requirement
                                  </WithChartTooltip>
                                </label>
                              </div>
                              <ParameterSlider
                                paramName="ac_time_horizon_minutes"
                                label=""
                                customMin={scHorizonLogBounds.min}
                                customMax={scHorizonLogBounds.max}
                                step={0.1}
                                customFormatValue={formatSCHorizon}
                                value={uiParameters.ac_time_horizon_minutes}
                                uiParameters={uiParameters}
                                setUiParameters={setUiParameters}
                                allParameters={allParameters}
                                isDragging={isDragging}
                                setIsDragging={setIsDragging}
                                commitParameters={commitParameters}
                                baselineParameters={baselineParameters}
                                useLogScale={true}
                              />
                            </div>
                            {/* Doubling Difficulty Growth Factor Slider */}
                            <div className="relative flex w-full flex-col">
                              <div className="flex items-center justify-between mb-2">
                                <label className="block w-full text-xs font-medium text-foreground">
                                  <WithChartTooltip
                                    explanation='If this parameter equals 0.9, this would mean each time horizon doubling requires an increase of effective compute orders of magnitude that is 0.9× what the previous doubling required. This parameter controls whether and how the time horizon trend is superexponential, with lower values meaning more superexponential. If you prefer an exponential extrapolation, set this parameter to 1.00x.'
                                    fullWidth
                                  >
                                    How much easier/harder each coding time horizon doubling gets
                                  </WithChartTooltip>
                                </label>
                              </div>
                              <ParameterSlider
                                paramName="doubling_difficulty_growth_factor"
                                label=""
                                customMin={samplingConfigBounds.doubling_difficulty_growth_factor?.min}
                                customMax={samplingConfigBounds.doubling_difficulty_growth_factor?.max}
                                fallbackMin={0.5}
                                fallbackMax={1.5}
                                step={0.01}
                                customFormatValue={(val) => `${val.toFixed(2)}x/doubling`}
                                value={uiParameters.doubling_difficulty_growth_factor}
                                uiParameters={uiParameters}
                                setUiParameters={setUiParameters}
                                allParameters={allParameters}
                                isDragging={isDragging}
                                setIsDragging={setIsDragging}
                                commitParameters={commitParameters}
                                baselineParameters={baselineParameters}
                                disabled={lockedParameters.has('doubling_difficulty_growth_factor')}
                              />
                            </div>
                            </>)}
                            {acAnchorMode !== 'time-horizon' && (AC_ANCHOR_KEY_PARAM_DEFS[acAnchorMode] ?? []).filter((def) => !def.onlyWhenSet || uiParameters[def.paramName] != null).map((def) => (
                              <div key={def.paramName} className="relative flex w-full flex-col">
                                <div className="flex items-center justify-between mb-2">
                                  <label className="block w-full text-xs font-medium text-foreground">
                                    <WithChartTooltip explanation={withAnchorRefDate(def.explanation, acAnchorMode, uiParameters)} fullWidth>
                                      {withAnchorRefDate(def.label, acAnchorMode, uiParameters)}
                                    </WithChartTooltip>
                                  </label>
                                </div>
                                <ParameterSlider
                                  paramName={def.paramName}
                                  label=""
                                  customMin={def.min}
                                  customMax={def.max}
                                  step={def.step}
                                  decimalPlaces={def.decimalPlaces}
                                  customFormatValue={def.format}
                                  value={resolveParamValue(def.paramName, uiParameters)}
                                  uiParameters={uiParameters}
                                  setUiParameters={setUiParameters}
                                  allParameters={allParameters}
                                  isDragging={isDragging}
                                  setIsDragging={setIsDragging}
                                  commitParameters={commitParameters}
                                  baselineParameters={baselineParameters}
                                  useLogScale={def.useLogScale}
                                />
                              </div>
                            ))}
                            {/* Median to Top Taste Multiplier Slider */}
                            <div className="relative flex w-full flex-col">
                              <div className="flex items-center justify-between mb-2">
                                <label className="block w-full text-xs font-medium text-foreground">
                                  <WithChartTooltip
                                    explanation="Ratio between the top human researcher's research taste and the median human researcher's research taste, i.e. how much more value per experiment the top researcher gets holding capacity to implement experiments fixed."
                                    fullWidth
                                  >
                                    Ratio of top to median researchers&apos; value per experiment
                                  </WithChartTooltip>
                                </label>
                              </div>
                              <ParameterSlider
                                paramName="median_to_top_taste_multiplier"
                                label=""
                                customMin={samplingConfigBounds.median_to_top_taste_multiplier?.min}
                                customMax={samplingConfigBounds.median_to_top_taste_multiplier?.max}
                                fallbackMin={1.1}
                                fallbackMax={20.0}
                                step={0.1}
                                customFormatValue={(val) => `${val.toFixed(2)}x`}
                                value={uiParameters.median_to_top_taste_multiplier}
                                uiParameters={uiParameters}
                                setUiParameters={setUiParameters}
                                allParameters={allParameters}
                                isDragging={isDragging}
                                setIsDragging={setIsDragging}
                                commitParameters={commitParameters}
                                baselineParameters={baselineParameters}
                                disabled={lockedParameters.has('median_to_top_taste_multiplier')}
                                useLogScale={true}
                              />
                            </div>
                            {/* Automated Research Taste Slope Slider */}
                            <div className="relative flex w-full flex-col">
                              <div className="flex items-start justify-between mb-2">
                                <label className="flex w-full text-xs font-medium text-foreground">
                                  <WithChartTooltip
                                    explanation="Each time effective compute increases by as many orders of magnitude as it did in 2025, by how many standard deviations (SDs) does the AI's research taste increase (median to best AGI company researcher requires 3 SDs)?"
                                    fullWidth
                                  >
                                    How quickly AIs improve at research taste
                                  </WithChartTooltip>
                                </label>
                              </div>
                              <ParameterSlider
                                paramName="ai_research_taste_slope"
                                label=""
                                step={0.1}
                                customMin={samplingConfigBounds.ai_research_taste_slope?.min}
                                customMax={samplingConfigBounds.ai_research_taste_slope?.max}
                                fallbackMin={0.1}
                                fallbackMax={10.0}
                                decimalPlaces={1}
                                value={uiParameters.ai_research_taste_slope}
                                uiParameters={uiParameters}
                                setUiParameters={setUiParameters}
                                allParameters={allParameters}
                                isDragging={isDragging}
                                setIsDragging={setIsDragging}
                                commitParameters={commitParameters}
                                baselineParameters={baselineParameters}
                                useLogScale={true}
                              />
                            </div>
                          </div>
                          <button
                            type="button"
                            onClick={() => setIsShareModalOpen(true)}
                            className="mt-4 w-full px-3 py-2 text-xs font-semibold uppercase tracking-[0.1em] text-white bg-[#2A623D] border border-[#215337] rounded-sm shadow-sm hover:bg-[#245632] active:bg-[#245632] flex items-center justify-center gap-2"
                          >
                            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                              <path strokeLinecap="round" strokeLinejoin="round" d="M8.684 13.342C8.886 12.938 9 12.482 9 12c0-.482-.114-.938-.316-1.342m0 2.684a3 3 0 110-2.684m0 2.684l6.632 3.316m-6.632-6l6.632-3.316m0 0a3 3 0 105.367-2.684 3 3 0 00-5.367 2.684zm0 9.316a3 3 0 105.368 2.684 3 3 0 00-5.368-2.684z" />
                            </svg>
                            Share Trajectory
                          </button>
                        </section>
                      </div>
                    </div>
                  </div>

                  {/* Desktop Layout: Original grid */}
                  <div className="hidden lg:block relative w-full">
                    <div className="overflow-x-auto overflow-y-hidden scrollbar-thin scrollbar-thumb-gray-300 scrollbar-track-transparent pb-3">
                      {/* Clean 4-row grid layout */}
                      <div
                        className="grid gap-4 min-w-max"
                        style={{
                          gridTemplateRows: 'repeat(3, 125px) minmax(125px, auto)',
                          gridTemplateColumns: `${CHART_LAYOUT.keyStats.width}px ${visiblePrimaryCount > 0 ? `${primaryGridWidth}px` : '0px'} ${hasMetricCharts ? `${columns.length * CHART_LAYOUT.metric.columnMinWidth + 100}px` : '0px'}`,
                        }}
                      >
                        {/* Column 1: Summary + KeyMilestone (spans all 4 rows) */}
                        <div className="row-span-4">
                          <KeyMilestonePanel
                            automationYear={acMilestoneYear}
                            asiYear={asiMilestoneYear}
                            isLoading={isInitialMilestoneLoading}
                            onShareClick={() => setIsShareModalOpen(true)}
                            height={primaryChartHeight}
                          />
                        </div>

                        {/* Column 2, Rows 1-3: Big Graphs */}
                        {visiblePrimaryCount > 0 && (
                          <div
                            className="col-start-2 row-span-3 grid gap-4"
                            style={{
                              gridTemplateColumns: `repeat(${visiblePrimaryCount}, minmax(0, 1fr))`,
                            }}
                          >
                            {visiblePrimaryCharts.map((chart) => (
                              <div key={chart.id} className="flex flex-col h-full">
                                {chart.render()}
                              </div>
                            ))}
                          </div>
                        )}

                        {/* Column 2, Row 4: Parameters */}
                        {visiblePrimaryCount > 0 && (
                          <section className="col-start-2 row-start-4 w-full px-2 flex flex-col justify-between">
                            <div className="flex items-center gap-2 pb-2" style={{ marginBottom: 10 }}>
                              <span className="text-[12px] font-semibold text-primary text-left font-system-mono">Key Parameters</span>
                              <div className="border-t border-gray-500/30 flex-1" />
                              <button
                                onClick={() => {
                                  setParameters({ ...DEFAULT_PARAMETERS });
                                  setUiParameters({ ...DEFAULT_PARAMETERS });
                                }}
                                className="text-[10px] text-gray-400 hover:text-gray-600 font-system-mono uppercase tracking-wider"
                              >
                                Reset
                              </button>

                              <div className="w-3 border-t border-gray-500/30" />
                              <button
                                onClick={() => setIsAdvancedPanelOpen(true)}
                                className="text-[10px] text-gray-400 hover:text-gray-600 font-system-mono font-bold uppercase tracking-wider"
                              >
                                Advanced Settings
                              </button>
                              <div className="w-3 border-t border-gray-500/30" />
                              <button
                                type="button"
                                onClick={() => setIsShareModalOpen(true)}
                                className="text-[10px] text-gray-600 font-system-mono font-bold uppercase tracking-wider border rounded px-1.5 py-[1px] flex items-center gap-1.5 transition-all duration-150 ease-out hover:bg-black hover:text-white hover:-translate-y-[1px] hover:shadow-sm active:translate-y-0 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-black/40 focus-visible:ring-offset-2 focus-visible:ring-offset-transparent"
                              >
                                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2} style={{ width: '1em', position: 'relative' }}>
                                  <path strokeLinecap="round" strokeLinejoin="round" d="M8.684 13.342C8.886 12.938 9 12.482 9 12c0-.482-.114-.938-.316-1.342m0 2.684a3 3 0 110-2.684m0 2.684l6.632 3.316m-6.632-6l6.632-3.316m0 0a3 3 0 105.367-2.684 3 3 0 00-5.367 2.684zm0 9.316a3 3 0 105.368 2.684 3 3 0 00-5.368-2.684z" />
                                </svg>
                                Share Trajectory
                              </button>
                            </div>
                            <div className="grid grid-cols-5 gap-10">
                              {acAnchorMode === 'time-horizon' && (<>
                              {/* Present Doubling Time Slider */}
                              <div className="relative flex w-full flex-col justify-between">
                                <div className="flex items-center justify-between mb-1">
                                  <label className="block w-full text-xs font-medium text-foreground">
                                    <WithChartTooltip
                                      explanation={`How long it takes the coding time horizon to double, as of ${horizonRefDateLabel} (considered as a long-term trend).`}
                                      fullWidth
                                    >
                                      {`How long it takes AIs' coding time horizon to double, as of ${horizonRefDateLabel}`}
                                    </WithChartTooltip>
                                  </label>
                                </div>
                                <ParameterSlider
                                  paramName="present_doubling_time"
                                  label=""
                                  customMin={samplingConfigBounds.present_doubling_time?.min}
                                  fallbackMin={0.01}
                                  fallbackMax={2.0}
                                  step={0.01}
                                  decimalPlaces={2}
                                  customFormatValue={(years) => formatTimeDuration(yearsToMinutes(years))}
                                  value={uiParameters.present_doubling_time}
                                  uiParameters={uiParameters}
                                  setUiParameters={setUiParameters}
                                  allParameters={allParameters}
                                  isDragging={isDragging}
                                  setIsDragging={setIsDragging}
                                  commitParameters={commitParameters}
                                  baselineParameters={baselineParameters}
                                  onMouseEnter={() => handlePopoverOpen('doubling_time')}
                                  onMouseLeave={handlePopoverClose}
                                  onFocus={() => handlePopoverOpen('doubling_time')}
                                  onBlur={handlePopoverClose}
                                  popoverConfig={CHART_CONFIGS.doubling_time}
                                  popoverVisible={activePopover === 'doubling_time'}
                                  onPopoverClose={handlePopoverClose}
                                  horizonParams={horizonParams}
                                  useLogScale={true}
                                />
                              </div>
                              {/* SC Time Horizon Slider */}
                              <div className="relative flex w-full flex-col justify-between">
                                <div className="flex items-center justify-between mb-1">
                                  <label className="block w-full text-xs font-medium text-foreground">
                                    <WithChartTooltip
                                      explanation="If an AI can complete, with 80% reliability, tasks that take a coder this long, then an AI can replace the coding staff of an AGI project."
                                      fullWidth
                                    >
                                      Coding time horizon required to achieve Automated Coder (AC)
                                    </WithChartTooltip>
                                  </label>
                                </div>
                                <ParameterSlider
                                  paramName="ac_time_horizon_minutes"
                                  label=""
                                  customMin={scHorizonLogBounds.min}
                                  customMax={scHorizonLogBounds.max}
                                  step={0.1}
                                  customFormatValue={formatSCHorizon}
                                  value={uiParameters.ac_time_horizon_minutes}
                                  uiParameters={uiParameters}
                                  setUiParameters={setUiParameters}
                                  allParameters={allParameters}
                                  isDragging={isDragging}
                                  setIsDragging={setIsDragging}
                                  commitParameters={commitParameters}
                                  baselineParameters={baselineParameters}
                                  onMouseEnter={() => handlePopoverOpen('sc_horizon')}
                                  onMouseLeave={handlePopoverClose}
                                  onFocus={() => handlePopoverOpen('sc_horizon')}
                                  onBlur={handlePopoverClose}
                                  popoverConfig={CHART_CONFIGS.sc_horizon}
                                  popoverVisible={activePopover === 'sc_horizon'}
                                  onPopoverClose={handlePopoverClose}
                                  horizonParams={horizonParams}
                                  useLogScale={true}
                                />
                              </div>
                              {/* Doubling Difficulty Growth Factor Slider */}
                              <div className="relative flex w-full flex-col justify-between">
                                <div className="flex items-center justify-between mb-1">
                                  <label className="block w-full text-xs font-medium text-foreground">
                                    <WithChartTooltip
                                      explanation={
                                        'If this equals 0.9, then each time horizon doubling requires an increase of log(effective compute) ' +
                                        'that is 0.9× what the previous doubling required. To make time horizon ' +
                                        'roughly exponential in time if compute and labor trends were to stay the same as in the recent past, set this ' +
                                        'to 1.00x. To make the trend superexponential, set its value to below 1, or set it to above 1 for subexponential.'
                                      }
                                      fullWidth
                                    >
                                      How much easier/harder each coding time horizon doubling gets
                                    </WithChartTooltip>
                                  </label>
                                </div>
                                <ParameterSlider
                                  paramName="doubling_difficulty_growth_factor"
                                  label=""
                                  fallbackMin={0.5}
                                  fallbackMax={1.5}
                                  step={0.01}
                                  customFormatValue={(val) => `${val.toFixed(2)}x/doubling`}
                                  value={uiParameters.doubling_difficulty_growth_factor}
                                  uiParameters={uiParameters}
                                  setUiParameters={setUiParameters}
                                  allParameters={allParameters}
                                  isDragging={isDragging}
                                  setIsDragging={setIsDragging}
                                  commitParameters={commitParameters}
                                  baselineParameters={baselineParameters}
                                  onMouseEnter={() => handlePopoverOpen('difficulty_growth_rate')}
                                  onMouseLeave={handlePopoverClose}
                                  onFocus={() => handlePopoverOpen('difficulty_growth_rate')}
                                  onBlur={handlePopoverClose}
                                  popoverConfig={CHART_CONFIGS.difficulty_growth_rate}
                                  popoverVisible={activePopover === 'difficulty_growth_rate'}
                                  onPopoverClose={handlePopoverClose}
                                  horizonParams={horizonParams}
                                  disabled={lockedParameters.has('doubling_difficulty_growth_factor')}
                                />
                              </div>
                              </>)}
                              {acAnchorMode !== 'time-horizon' && (AC_ANCHOR_KEY_PARAM_DEFS[acAnchorMode] ?? []).filter((def) => !def.onlyWhenSet || uiParameters[def.paramName] != null).map((def) => (
                                <div key={def.paramName} className="relative flex w-full flex-col justify-between">
                                  <div className="flex items-center justify-between mb-1">
                                    <label className="block w-full text-xs font-medium text-foreground">
                                      <WithChartTooltip explanation={withAnchorRefDate(def.explanation, acAnchorMode, uiParameters)} fullWidth>
                                        {withAnchorRefDate(def.label, acAnchorMode, uiParameters)}
                                      </WithChartTooltip>
                                    </label>
                                  </div>
                                  <ParameterSlider
                                    paramName={def.paramName}
                                    label=""
                                    customMin={def.min}
                                    customMax={def.max}
                                    step={def.step}
                                    decimalPlaces={def.decimalPlaces}
                                    customFormatValue={def.format}
                                    value={resolveParamValue(def.paramName, uiParameters)}
                                    uiParameters={uiParameters}
                                    setUiParameters={setUiParameters}
                                    allParameters={allParameters}
                                    isDragging={isDragging}
                                    setIsDragging={setIsDragging}
                                    commitParameters={commitParameters}
                                    baselineParameters={baselineParameters}
                                    useLogScale={def.useLogScale}
                                  />
                                </div>
                              ))}
                              {/* Median to Top Taste Multiplier Slider */}
                              <div className="relative flex w-full flex-col justify-between">
                                <div className="flex items-center justify-between mb-1">
                                  <label className="block w-full text-xs font-medium text-foreground">
                                    <WithChartTooltip
                                      explanation="Ratio of the top human researcher's research taste and the median human researcher's research taste, i.e. how much more value per experiment the top researcher gets holding capacity to implement experiments fixed."
                                      fullWidth
                                    >
                                      Ratio of top to median researchers&apos; value per experiment
                                    </WithChartTooltip>
                                  </label>
                                </div>
                                <ParameterSlider
                                  paramName="median_to_top_taste_multiplier"
                                  label=""
                                  customMin={samplingConfigBounds.median_to_top_taste_multiplier?.min}
                                  customMax={samplingConfigBounds.median_to_top_taste_multiplier?.max}
                                  fallbackMin={1.1}
                                  fallbackMax={20.0}
                                  step={0.1}
                                  customFormatValue={(val) => `${val.toFixed(2)}x`}
                                  value={uiParameters.median_to_top_taste_multiplier}
                                  uiParameters={uiParameters}
                                  setUiParameters={setUiParameters}
                                  allParameters={allParameters}
                                  isDragging={isDragging}
                                  setIsDragging={setIsDragging}
                                  commitParameters={commitParameters}
                                  baselineParameters={baselineParameters}
                                  disabled={lockedParameters.has('median_to_top_taste_multiplier')}
                                  useLogScale={true}
                                />
                              </div>
                              {/* Automated Research Taste Slope Slider */}
                              <div className="relative flex w-full flex-col justify-between">
                                <div className="flex items-start justify-between mb-1">
                                  <label className="flex w-full text-xs font-medium text-foreground">
                                    <WithChartTooltip
                                      explanation="Each time effective compute increases by as many orders of magnitude as it did in 2025, by how many standard deviations (SDs) does the AI's research taste increase (median to best AGI company researcher requires 3 SDs)?"
                                      fullWidth
                                    >
                                      How quickly AIs improve at research taste
                                    </WithChartTooltip>
                                  </label>
                                </div>
                                <ParameterSlider
                                  paramName="ai_research_taste_slope"
                                  label=""
                                  step={0.1}
                                  customMin={samplingConfigBounds.ai_research_taste_slope?.min}
                                  customMax={samplingConfigBounds.ai_research_taste_slope?.max}
                                  fallbackMin={0.1}
                                  fallbackMax={10.0}
                                  customFormatValue={(val) => `${val.toFixed(1)} SDs/2025-effective-FLOP-growth`}
                                  value={uiParameters.ai_research_taste_slope}
                                  uiParameters={uiParameters}
                                  setUiParameters={setUiParameters}
                                  allParameters={allParameters}
                                  isDragging={isDragging}
                                  setIsDragging={setIsDragging}
                                  commitParameters={commitParameters}
                                  baselineParameters={baselineParameters}
                                  useLogScale={true}
                                />
                              </div>
                            </div>
                          </section>
                        )}

                        {/* Column 3, Rows 1-3: Small Log Graphs */}
                        {hasMetricCharts && (
                          <div className="col-start-3 row-span-3 flex flex-col">
                            <div className="flex items-center gap-2 pb-2">
                              <span className="font-mono small-caps text-[12px] font-semibold">Log Scale</span>
                              <div className="border-t border-gray-500/30 flex-1" />
                            </div>
                            <div className="flex flex-1 flex-row gap-4 px-1 pr-10 min-h-0">
                              {hasLogCharts ? (
                                columns.map((col, columnIndex) => (
                                  <div
                                    key={`log-col-${columnIndex}`}
                                    className="h-full flex-shrink-0"
                                    style={{ width: CHART_LAYOUT.metric.columnMinWidth }}
                                  >
                                    <div className="grid h-full grid-rows-3 gap-4">
                                      {[col.log1, col.log2, col.log3].map((cfg, rowIndex) => (
                                        cfg ? (
                                          <ChartLoadingOverlay key={`${cfg.key}-${columnIndex}-${rowIndex}`} isLoading={isChartLoading} className="h-full">
                                            <CustomMetricChart
                                              chartData={renderData}
                                              tooltip={cfg.tooltip}
                                              title={cfg.title}
                                              dataKey={cfg.key}
                                              yFormatter={cfg.yFormatter || ((v) => formatTo3SigFigs(v))}
                                              yScale={cfg.yScale || 'linear'}
                                              domainOverride={cfg.domain}
                                              verticalReferenceLine={automationReferenceLine}
                                              className="h-full"
                                              displayEndYear={displayEndYear}
                                              logSuffix={cfg.logSuffix}
                                              isDataInLogForm={cfg.isDataInLogForm}
                                            />
                                          </ChartLoadingOverlay>
                                        ) : (
                                          <div
                                            key={`log-placeholder-${columnIndex}-${rowIndex}`}
                                            className="h-full rounded-lg border border-dashed border-slate-200/60"
                                          />
                                        )
                                      ))}
                                    </div>
                                  </div>
                                ))
                              ) : (
                                <div className="flex h-full flex-1 items-center justify-center rounded-lg border border-dashed border-slate-200/60 text-xs text-slate-500">
                                  All log-scale charts are hidden.
                                </div>
                              )}
                            </div>
                          </div>
                        )}

                        {/* Column 3, Row 4: Small Linear Graphs */}
                        {hasMetricCharts && (
                          <div className="col-start-3 row-start-4 flex flex-col">
                            <div className="flex items-center gap-2 pb-2">
                              <span className="font-mono small-caps text-[12px] font-semibold">Linear Scale</span>
                              <div className="border-t border-gray-500/30 flex-1" />
                            </div>
                            <div className="flex flex-1 flex-row gap-4 px-1 pr-10 min-h-0 h-[100px] max-h-[100px]">
                              {hasLinearCharts ? (
                                columns.map((col, columnIndex) => (
                                  <div
                                    key={`lin-col-${columnIndex}`}
                                    className="h-full flex-shrink-0"
                                    style={{ width: CHART_LAYOUT.metric.columnMinWidth }}
                                  >
                                    {col.linear ? (
                                      <ChartLoadingOverlay key={`${col.linear.key}-${columnIndex}-linear`} isLoading={isChartLoading} className="h-full">
                                        <CustomMetricChart
                                          chartData={renderData}
                                          tooltip={col.linear.tooltip}
                                          title={col.linear.title}
                                          dataKey={col.linear.key}
                                          yFormatter={col.linear.yFormatter || ((v) => formatTo3SigFigs(v))}
                                          yScale={col.linear.yScale || 'linear'}
                                          domainOverride={col.linear.domain}
                                          verticalReferenceLine={automationReferenceLine}
                                          className="h-full"
                                          displayEndYear={displayEndYear}
                                          logSuffix={col.linear.logSuffix}
                                          isDataInLogForm={col.linear.isDataInLogForm}
                                        />
                                      </ChartLoadingOverlay>
                                    ) : (
                                      <div className="h-full rounded-lg border border-dashed border-slate-200/60" />
                                    )}
                                  </div>
                                ))
                              ) : (
                                <div className="flex h-full flex-1 items-center justify-center rounded-lg border border-dashed border-slate-200/60 text-xs text-slate-500">
                                  All linear-scale charts are hidden.
                                </div>
                              )}
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                    <div
                      className="pointer-events-none absolute inset-y-0 right-0 w-32 bg-gradient-to-r from-transparent to-vivid-background"
                    />
                  </div>
                </ChartSyncProvider>
              </div>
            </main>

            {/* Model Architecture with Intro Content */}
            <ModelArchitecture
              modelDescriptionGDocPortionMarkdown={modelDescriptionGDocPortionMarkdown}
              scrollContainerRef={scrollContainerRef}
            />

          </div>
        </div>

        {/* Advanced Parameters Column - hidden on mobile/tablet, opens via Parameters section button */}
        {isAdvancedPanelOpen && (
          <div className="col-start-2 row-start-1 hidden lg:flex h-full overflow-visible shadow-[0_0_1em_0_rgba(0,0,0,0.5)] z-10">
            <aside
              id="advanced-panel"
              className="w-[22rem] bg-white border-l border-gray-200 flex flex-col"
            >
              <div className="flex items-center justify-between px-4 py-3 border-b border-gray-100">
                <h2 className="text-sm font-semibold text-primary uppercase tracking-[0.3em] font-system-mono">
                  Advanced Parameters
                </h2>
                <button
                  type="button"
                  onClick={() => setIsAdvancedPanelOpen(false)}
                  className="text-xs font-medium text-gray-500 hover:text-gray-700"
                >
                  Close
                </button>
              </div>
              <div className="flex-1 overflow-y-auto px-4 py-4">
                <div className="space-y-6">
                  <div className="space-y-3">
                    <div className="flex flex-wrap gap-2 sm:gap-3">
                      <button
                        type="button"
                        onClick={handleExportParameters}
                        className="flex-1 min-w-[140px] px-3 py-2 text-xs font-semibold uppercase tracking-[0.1em] text-white bg-[#2A623D] border border-[#215337] rounded-sm shadow-[0_1px_2px_rgba(32,83,53,0.25)] hover:bg-[#245632] focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-[#2A623D] font-system-mono"
                      >
                        Export Settings
                      </button>
                      <button
                        type="button"
                        onClick={handleImportButtonClick}
                        className="flex-1 min-w-[140px] px-3 py-2 text-xs font-semibold uppercase tracking-[0.1em] text-[#2A623D] bg-white border border-[#2A623D]/40 rounded-sm shadow-[0_1px_2px_rgba(32,83,53,0.1)] hover:bg-[#F4F8F5] focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-[#2A623D] font-system-mono"
                      >
                        Import Settings
                      </button>
                      <input
                        ref={importInputRef}
                        type="file"
                        accept="application/json"
                        className="hidden"
                        onChange={handleImportParameters}
                      />
                    </div>
                    <div className="space-y-2">
                      <label className="block text-xs font-semibold uppercase tracking-[0.18em] text-primary font-system-mono">
                        Parameter Presets
                      </label>
                      <div className="relative">
                        <span className="pointer-events-none absolute inset-y-0 left-3 flex items-center text-[11px] uppercase tracking-[0.2em] text-gray-400 font-system-mono">
                          Preset
                        </span>
                        <select
                          value={selectedPresetId}
                          onChange={handlePresetSelect}
                          className="w-full pl-20 pr-8 py-3 border border-gray-300 rounded-md shadow-sm bg-white text-sm font-system-mono focus:outline-none focus:ring-2 focus:ring-[#2A623D] appearance-none"
                        >
                          <option value="">Select</option>
                          {PARAMETER_PRESETS.map((preset) => (
                            <option key={preset.id} value={preset.id}>
                              {preset.label}
                            </option>
                          ))}
                        </select>
                        <span className="pointer-events-none absolute inset-y-0 right-3 flex items-center text-gray-400">
                          <svg
                            className="h-3 w-3"
                            viewBox="0 0 12 8"
                            fill="none"
                            xmlns="http://www.w3.org/2000/svg"
                            aria-hidden="true"
                          >
                            <path d="M1 1.5L6 6.5L11 1.5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                          </svg>
                        </span>
                      </div>
                    </div>
                  </div>
                  <AdvancedSections
                    uiParameters={uiParameters}
                    setUiParameters={setUiParameters}
                    allParameters={allParameters}
                    isDragging={isDragging}
                    setIsDragging={setIsDragging}
                    commitParameters={commitParameters}
                    scHorizonLogBounds={scHorizonLogBounds}
                    parallelPenaltyBounds={parallelPenaltyBounds}
                    summary={summary}
                    lockedParameters={lockedParameters}
                    onToggleTrajectoryDebugger={() => setShowSamplingDebug(prev => !prev)}
                    isTrajectoryDebuggerOpen={showSamplingDebug}
                    samplingConfigBounds={samplingConfigBounds}
                    simplificationCheckboxes={{
                                    enableExperimentAutomation,
                      setEnableExperimentAutomation,
                      useExperimentThroughputCES,
                      setUseExperimentThroughputCES,
                      enableSoftwareProgress,
                      setEnableSoftwareProgress,
                      useComputeLaborGrowthSlowdown,
                      setUseComputeLaborGrowthSlowdown,
                                  }}
                    onExportCSV={handleExportCSV}
                    onOpenCsvEditor={() => setIsCsvEditorOpen(true)}
                    hasCustomTimeSeries={customTimeSeries !== null}
                    baselineParameters={baselineParameters}
                    horizonPercentile={horizonPercentile}
                    onHorizonPercentileChange={setHorizonPercentile}
                  />
                </div>
              </div>
            </aside>
          </div>
        )}
      </div>

      {/* Sampling Debug Panel - Toggle with Alt+D */}
      {showSamplingDebug && (
        <div className="fixed bottom-4 right-4 z-[9999] w-[600px] max-h-[500px] bg-gray-900 text-white rounded-lg shadow-2xl overflow-hidden">
          <div className="flex items-center justify-between px-4 py-2 bg-gray-800 border-b border-gray-700">
            <h3 className="font-mono text-sm font-semibold">Sampling Debug</h3>
            <div className="flex items-center gap-4">
              <button
                onClick={() => setShowParamToggles(prev => !prev)}
                className={`px-2 py-0.5 text-[10px] font-semibold rounded uppercase tracking-wider ${showParamToggles ? 'bg-yellow-600 hover:bg-yellow-500' : 'bg-gray-600 hover:bg-gray-500'
                  }`}
              >
                {showParamToggles ? 'Hide' : 'Show'} Param Toggles
              </button>
              <button
                onClick={() => setProfilingEnabled(prev => !prev)}
                className={`px-2 py-0.5 text-[10px] font-semibold rounded uppercase tracking-wider ${profilingEnabled ? 'bg-purple-600 hover:bg-purple-500' : 'bg-gray-600 hover:bg-gray-500'
                  }`}
              >
                Profiling {profilingEnabled ? 'ON' : 'OFF'}
              </button>
              <span className="text-xs text-gray-400">
                {/* TEMPORARY: Hidden for launch - time_series_parameters not included in count */}
                {enabledSamplingParams.size}/{samplingConfig ? Object.keys(samplingConfig.parameters).length : 0} sampling
              </span>
              <button
                onClick={() => setShowSamplingDebug(false)}
                className="text-gray-400 hover:text-white"
              >
                ✕
              </button>
            </div>
          </div>

          {/* Parameter Toggles Section */}
          {showParamToggles && samplingConfig && (
            <div className="p-2 bg-gray-800/50 border-b border-gray-700 max-h-[200px] overflow-auto">
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs text-gray-400 font-semibold">Parameters to Sample (unchecked = fixed to median)</span>
                <div className="flex gap-2">
                  <button
                    onClick={() => {
                      const allParams = new Set<string>();
                      for (const paramName of Object.keys(samplingConfig.parameters)) {
                        allParams.add(paramName);
                      }
                      // TEMPORARY: Hidden for launch - time series parameters not exposed to users
                      // if (samplingConfig.time_series_parameters) {
                      //   for (const paramName of Object.keys(samplingConfig.time_series_parameters)) {
                      //     allParams.add(paramName);
                      //   }
                      // }
                      setEnabledSamplingParams(allParams);
                    }}
                    className="px-2 py-0.5 bg-green-600 hover:bg-green-500 text-[9px] font-semibold rounded uppercase"
                  >
                    All On
                  </button>
                  <button
                    onClick={() => setEnabledSamplingParams(new Set())}
                    className="px-2 py-0.5 bg-red-600 hover:bg-red-500 text-[9px] font-semibold rounded uppercase"
                  >
                    All Off
                  </button>
                  <button
                    onClick={() => {
                      // Clear samples and re-fetch with new settings
                      setSampleTrajectories([]);
                      setSamplingDebugLog([]);
                      setResampleTrigger(prev => prev + 1);
                    }}
                    className="px-2 py-0.5 bg-blue-600 hover:bg-blue-500 text-[9px] font-semibold rounded uppercase"
                  >
                    Resample
                  </button>
                </div>
              </div>
              <div className="grid grid-cols-2 gap-1 text-[10px] font-mono">
                {/* Model parameters */}
                {Object.entries(samplingConfig.parameters)
                  .filter(([, config]) => config.dist !== 'fixed')
                  .sort(([a], [b]) => a.localeCompare(b))
                  .map(([paramName, config]) => {
                    const isEnabled = enabledSamplingParams.has(paramName);
                    const medianValue = getDistributionMedian(config);
                    return (
                      <label
                        key={paramName}
                        className={`flex items-center gap-1.5 p-1 rounded cursor-pointer hover:bg-gray-700/50 ${isEnabled ? 'text-green-400' : 'text-gray-500'
                          }`}
                      >
                        <input
                          type="checkbox"
                          checked={isEnabled}
                          onChange={(e) => {
                            setEnabledSamplingParams(prev => {
                              const next = new Set(prev);
                              if (e.target.checked) {
                                next.add(paramName);
                              } else {
                                next.delete(paramName);
                              }
                              return next;
                            });
                          }}
                          className="w-3 h-3"
                        />
                        <span className="truncate flex-1" title={paramName}>{paramName}</span>
                        <span className="text-gray-600 text-[8px]">
                          {typeof medianValue === 'number'
                            ? (Math.abs(medianValue) > 1000 || (Math.abs(medianValue) < 0.01 && medianValue !== 0)
                              ? medianValue.toExponential(1)
                              : medianValue.toFixed(2))
                            : String(medianValue)}
                        </span>
                      </label>
                    );
                  })}
                {/* TEMPORARY: Hidden for launch - time series parameters not exposed to users
                {samplingConfig.time_series_parameters && Object.entries(samplingConfig.time_series_parameters)
                  .filter(([, config]) => config.dist !== 'fixed')
                  .sort(([a], [b]) => a.localeCompare(b))
                  .map(([paramName, config]) => {
                    const isEnabled = enabledSamplingParams.has(paramName);
                    const medianValue = getDistributionMedian(config);
                    return (
                      <label
                        key={paramName}
                        className={`flex items-center gap-1.5 p-1 rounded cursor-pointer hover:bg-gray-700/50 ${isEnabled ? 'text-yellow-400' : 'text-gray-500'
                          }`}
                        title="Time series parameter"
                      >
                        <input
                          type="checkbox"
                          checked={isEnabled}
                          onChange={(e) => {
                            setEnabledSamplingParams(prev => {
                              const next = new Set(prev);
                              if (e.target.checked) {
                                next.add(paramName);
                              } else {
                                next.delete(paramName);
                              }
                              return next;
                            });
                          }}
                          className="w-3 h-3"
                        />
                        <span className="truncate flex-1" title={`${paramName} (time series)`}>📊 {paramName}</span>
                        <span className="text-gray-600 text-[8px]">
                          {typeof medianValue === 'number'
                            ? (Math.abs(medianValue) > 1000 || (Math.abs(medianValue) < 0.01 && medianValue !== 0)
                              ? medianValue.toExponential(1)
                              : medianValue.toFixed(2))
                            : String(medianValue)}
                        </span>
                      </label>
                    );
                  })}
                */}
              </div>
            </div>
          )}

          {/* Profiling Results Section */}
          {profilingEnabled && (
            <div className="p-2 bg-purple-900/30 border-b border-gray-700">
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs text-purple-300 font-semibold">Server Profiling</span>
                {profilingData && (
                  <span className="text-xs text-purple-400">
                    Total: {profilingData.totalTimeSeconds.toFixed(3)}s
                    {typeof profilingData.totalCpuTimeSeconds === 'number'
                      ? ` (CPU ${profilingData.totalCpuTimeSeconds.toFixed(3)}s)`
                      : ''}
                    {' '}@ {profilingData.timestamp.toLocaleTimeString()}
                  </span>
                )}
              </div>
              {profilingData ? (
                <>
                  {/* Timing Breakdown */}
                  <div className="bg-gray-950 rounded p-2 mb-2">
                    <div className="text-[10px] text-purple-300 font-semibold mb-1">Timing Breakdown:</div>
                    <div className="grid grid-cols-2 gap-x-4 gap-y-0.5 text-[9px] font-mono">
                      {Object.entries(profilingData.timingBreakdown)
                        .sort(([, a], [, b]) => b - a)
                        .map(([key, value]) => (
                          <div key={key} className="flex justify-between">
                            <span className="text-gray-400">{key}:</span>
                            <span className="text-green-400">{value.toFixed(3)}s</span>
                          </div>
                        ))}
                    </div>
                  </div>

                  {/* CPU Timing Breakdown */}
                  {profilingData.cpuTimingBreakdown && Object.keys(profilingData.cpuTimingBreakdown).length > 0 && (
                    <div className="bg-gray-950 rounded p-2 mb-2">
                      <div className="text-[10px] text-purple-300 font-semibold mb-1">CPU Timing Breakdown:</div>
                      <div className="grid grid-cols-2 gap-x-4 gap-y-0.5 text-[9px] font-mono">
                        {Object.entries(profilingData.cpuTimingBreakdown)
                          .sort(([, a], [, b]) => (b as number) - (a as number))
                          .map(([key, value]) => (
                            <div key={key} className="flex justify-between">
                              <span className="text-gray-400">{key}:</span>
                              <span className="text-cyan-300">{(value as number).toFixed(3)}s</span>
                            </div>
                          ))}
                      </div>
                    </div>
                  )}

                  {/* ODE Diagnostics */}
                  {profilingData.odeDiagnostics && profilingData.odeDiagnostics.length > 0 && (
                    <details className="bg-gray-950 rounded mb-2">
                      <summary className="p-2 text-[10px] text-gray-400 cursor-pointer hover:text-gray-300">
                        ODE Diagnostics ({profilingData.odeDiagnostics.length} solve_ivp attempts)
                      </summary>
                      <div className="p-2 max-h-[200px] overflow-auto">
                        <div className="grid grid-cols-6 gap-x-3 gap-y-1 text-[9px] font-mono">
                          <div className="text-gray-500">method</div>
                          <div className="text-gray-500">success</div>
                          <div className="text-gray-500">wall</div>
                          <div className="text-gray-500">cpu</div>
                          <div className="text-gray-500">nfev</div>
                          <div className="text-gray-500">steps</div>
                          {profilingData.odeDiagnostics
                            .slice()
                            .sort((a, b) => (Number(b.wall_time_seconds) || 0) - (Number(a.wall_time_seconds) || 0))
                            .map((attempt, idx) => {
                              const method = typeof attempt.method === 'string' ? attempt.method : 'unknown';
                              const success = attempt.success === true;
                              const wall = typeof attempt.wall_time_seconds === 'number' ? attempt.wall_time_seconds : null;
                              const cpu = typeof attempt.cpu_time_seconds === 'number' ? attempt.cpu_time_seconds : null;
                              const nfev = typeof attempt.nfev === 'number' ? attempt.nfev : null;
                              const steps = typeof attempt.step_stats?.num_steps === 'number' ? attempt.step_stats.num_steps : null;

                              return (
                                <Fragment key={idx}>
                                  <div className="text-gray-300">{method}</div>
                                  <div className={success ? 'text-green-400' : 'text-red-400'}>
                                    {success ? 'yes' : 'no'}
                                  </div>
                                  <div className="text-purple-200">{wall === null ? '—' : `${wall.toFixed(3)}s`}</div>
                                  <div className="text-cyan-200">{cpu === null ? '—' : `${cpu.toFixed(3)}s`}</div>
                                  <div className="text-gray-300">{nfev === null ? '—' : nfev}</div>
                                  <div className="text-gray-300">{steps === null ? '—' : steps}</div>
                                </Fragment>
                              );
                            })}
                        </div>
                      </div>
                    </details>
                  )}

                  {/* Model Internal Timing */}
                  {isRecord(profilingData.modelInternal) && (
                    <details className="bg-gray-950 rounded mb-2">
                      <summary className="p-2 text-[10px] text-gray-400 cursor-pointer hover:text-gray-300">
                        Model Internal Timing (expand)
                      </summary>
                      <div className="p-2 max-h-[220px] overflow-auto">
                        {isFiniteNumber(profilingData.modelInternal.wall_time_seconds_total) && (
                          <div className="text-[9px] font-mono text-purple-200 mb-2">
                            total_wall={profilingData.modelInternal.wall_time_seconds_total.toFixed(3)}s
                            {isFiniteNumber(profilingData.modelInternal.cpu_time_seconds_total)
                              ? ` total_cpu=${profilingData.modelInternal.cpu_time_seconds_total.toFixed(3)}s`
                              : ''}
                          </div>
                        )}

                        {(() => {
                          const wall = profilingData.modelInternal.wall_time_breakdown_seconds;
                          if (!isRecord(wall)) return null;
                          const entries = Object.entries(wall)
                            .filter(([, value]) => isFiniteNumber(value))
                            .sort(([, a], [, b]) => (b as number) - (a as number));
                          if (entries.length === 0) return null;
                          return (
                            <div className="mb-3">
                              <div className="text-[10px] text-purple-300 font-semibold mb-1">Wall Breakdown:</div>
                              <div className="grid grid-cols-2 gap-x-4 gap-y-0.5 text-[9px] font-mono">
                                {entries.map(([key, value]) => (
                                  <div key={key} className="flex justify-between">
                                    <span className="text-gray-400">{key}:</span>
                                    <span className="text-purple-200">{(value as number).toFixed(3)}s</span>
                                  </div>
                                ))}
                              </div>
                            </div>
                          );
                        })()}

                        {(() => {
                          const cpu = profilingData.modelInternal.cpu_time_breakdown_seconds;
                          if (!isRecord(cpu)) return null;
                          const entries = Object.entries(cpu)
                            .filter(([, value]) => isFiniteNumber(value))
                            .sort(([, a], [, b]) => (b as number) - (a as number));
                          if (entries.length === 0) return null;
                          return (
                            <div className="mb-3">
                              <div className="text-[10px] text-purple-300 font-semibold mb-1">CPU Breakdown:</div>
                              <div className="grid grid-cols-2 gap-x-4 gap-y-0.5 text-[9px] font-mono">
                                {entries.map(([key, value]) => (
                                  <div key={key} className="flex justify-between">
                                    <span className="text-gray-400">{key}:</span>
                                    <span className="text-cyan-200">{(value as number).toFixed(3)}s</span>
                                  </div>
                                ))}
                              </div>
                            </div>
                          );
                        })()}

                        {(() => {
                          const meta = profilingData.modelInternal.meta;
                          if (!isRecord(meta)) return null;
                          const profile = meta.metrics_loop_profile_wall_seconds;
                          if (!isRecord(profile)) return null;
                          const entries = Object.entries(profile)
                            .filter(([, value]) => isFiniteNumber(value))
                            .sort(([, a], [, b]) => (b as number) - (a as number));
                          if (entries.length === 0) return null;
                          return (
                            <div>
                              <div className="text-[10px] text-purple-300 font-semibold mb-1">Metrics Loop Detail (wall):</div>
                              <div className="grid grid-cols-2 gap-x-4 gap-y-0.5 text-[9px] font-mono">
                                {entries.map(([key, value]) => (
                                  <div key={key} className="flex justify-between">
                                    <span className="text-gray-400">{key}:</span>
                                    <span className="text-gray-200">{(value as number).toFixed(3)}s</span>
                                  </div>
                                ))}
                              </div>
                            </div>
                          );
                        })()}
                      </div>
                    </details>
                  )}

                  {/* cProfile Stats */}
                  <details className="bg-gray-950 rounded">
                    <summary className="p-2 text-[10px] text-gray-400 cursor-pointer hover:text-gray-300">
                      Profile Details (click to expand)
                    </summary>
                    <div className="p-2 max-h-[150px] overflow-auto">
                      <pre className="text-[9px] text-gray-300 whitespace-pre font-mono leading-tight">
                        {profilingData.stats}
                      </pre>
                    </div>
                  </details>
                </>
              ) : (
                <p className="text-gray-500 text-xs">
                  Profiling enabled. Move a slider to trigger a request and see results.
                </p>
              )}
            </div>
          )}

          <div className="overflow-auto max-h-[280px] p-2">
            <div className="space-y-1 font-mono text-xs">
              {samplingDebugLog.length === 0 ? (
                <p className="text-gray-500 p-2">No sampling requests yet. Waiting for samplingConfig...</p>
              ) : (
                samplingDebugLog.map((log, idx) => (
                  <div
                    key={idx}
                    className={`p-2 rounded ${log.type === 'request' ? 'bg-blue-900/50' :
                      log.type === 'response' ? 'bg-green-900/50' :
                        'bg-red-900/50'
                      }`}
                  >
                    <div className="flex items-center gap-2">
                      <span className={`px-1 rounded text-[10px] uppercase ${log.type === 'request' ? 'bg-blue-600' :
                        log.type === 'response' ? 'bg-green-600' :
                          'bg-red-600'
                        }`}>
                        {log.type}
                      </span>
                      <span className="text-gray-400">#{log.id}</span>
                      <span className="text-gray-500">{log.timestamp.toISOString().split('T')[1].split('.')[0]}</span>
                    </div>
                    {log.params && (
                      <div className="mt-1 text-gray-300 text-[10px]">
                        {Object.entries(log.params).map(([k, v]) => (
                          <div key={k}>
                            <span className="text-gray-500">{k}:</span>{' '}
                            {typeof v === 'number' ? v.toExponential(2) : String(v)}
                          </div>
                        ))}
                      </div>
                    )}
                    {log.dataPoints !== undefined && (
                      <div className="mt-1 text-green-400">✓ {log.dataPoints} data points received</div>
                    )}
                    {log.error && (
                      <div className="mt-1 text-red-400">Error: {log.error}</div>
                    )}
                  </div>
                ))
              )}
            </div>
            {sampleTrajectories.length > 0 && (
              <div className="mt-3 p-2 bg-gray-800 rounded">
                <div className="text-xs text-gray-400 mb-2">Current Trajectories (click row to expand, Apply to use):</div>
                {sampleTrajectories.map((sample, idx) => {
                  const traj = sample.trajectory;
                  const horizonMinutes = sample.params.ac_time_horizon_minutes;
                  const isExpanded = expandedTrajectoryIndex === idx;
                  return (
                    <div key={idx} className="mb-1">
                      <div
                        className="flex items-center justify-between p-1.5 bg-gray-700/50 rounded hover:bg-gray-600/50 cursor-pointer transition-colors"
                        onClick={() => setExpandedTrajectoryIndex(isExpanded ? null : idx)}
                      >
                        <div className="text-[10px] text-gray-300 flex items-center gap-1">
                          <span className={`transition-transform ${isExpanded ? 'rotate-90' : ''}`}>▶</span>
                          <span className="font-semibold text-white">#{idx + 1}</span>
                          {' • '}{traj.length} pts
                          {' • '}SC: {typeof horizonMinutes === 'number' ? formatWorkTimeDuration(horizonMinutes) : 'N/A'}
                        </div>
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            applySampleTrajectory(sample);
                          }}
                          className="px-2 py-0.5 bg-green-600 hover:bg-green-500 text-white text-[9px] font-semibold rounded uppercase tracking-wider"
                        >
                          Apply
                        </button>
                      </div>
                      {isExpanded && (
                        <div className="mt-1 ml-4 p-2 bg-gray-900/50 rounded text-[9px] font-mono max-h-[200px] overflow-auto">
                          <div className="text-gray-400 mb-1 font-semibold">Full Parameters:</div>
                          {Object.entries(sample.params)
                            .sort(([a], [b]) => a.localeCompare(b))
                            .map(([key, value]) => (
                              <div key={key} className="flex gap-2">
                                <span className="text-gray-500 min-w-[200px]">{key}:</span>
                                <span className="text-gray-300">
                                  {typeof value === 'number'
                                    ? (Math.abs(value) > 1000 || Math.abs(value) < 0.01
                                      ? value.toExponential(3)
                                      : value.toFixed(4))
                                    : String(value)}
                                </span>
                              </div>
                            ))}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Share Prediction Modal */}
      <SharePredictionModal
        isOpen={isShareModalOpen}
        onClose={() => setIsShareModalOpen(false)}
        horizonChartData={renderData}
        upliftChartData={renderData}
        milestones={milestones}
        scHorizonMinutes={scHorizonMinutes}
        displayEndYear={displayEndYear}
        shareUrl={shareUrl}
        benchmarkData={effectiveBenchmarkData}
        presentDay={parameters.present_day}
      />
    </ParameterHoverProvider>
  );
}
