'use client';

import { useState, useCallback, useEffect, useRef, useMemo, ChangeEvent } from 'react';
import { usePathname, useRouter, useSearchParams } from 'next/navigation';
import CustomHorizonChart from './CustomHorizonChart';
import CombinedComputeChart from './CombinedComputeChart';
import { tooltipBoxStyle, tooltipHeaderStyle, tooltipValueStyle } from './chartTooltipStyle';
import { ChartDataPoint, BenchmarkPoint } from '@/app/types';
import { CustomMetricChart } from './CustomMetricChart';
import type { DataPoint } from './CustomLineChart';

import { formatTo3SigFigs, formatWorkTimeDuration, formatUplift, formatCompactNumberNode, formatAsPowerOfTenText } from '@/utils/formatting';
import { exportParametersToFile, importParametersFromFile } from '@/utils/parameterExportImport';
import { WithChartTooltip } from './ChartTitleTooltip';
import { DEFAULT_PARAMETERS, ParametersType, PARAMETER_PRESETS, PRESET_BASELINES, CURRENT_DEFAULT_BASELINE_ID } from '@/constants/parameters';
import { CHART_LAYOUT } from '@/constants/chartLayout';
import { convertParametersToAPIFormat, convertSampledParametersToAPIFormat, ParameterRecord } from '@/utils/monteCarlo';
import { ChartSyncProvider } from './ChartSyncContext';
import { ChartLoadingOverlay } from './ChartLoadingOverlay';
import { encodeFullStateToParams, decodeFullStateFromParams, DEFAULT_CHECKBOX_STATES, DEFAULT_PAUSE_STATE } from '@/utils/urlState';
import { CHECKBOX_PARAM_OVERRIDES, CHECKBOX_LOCKED_PARAMETERS } from '@/constants/simplificationOverrides';
import { useCheckboxParameterSync } from '@/utils/useCheckboxParameterSync';
import { AIRnDProgressMultiplierChart } from './AIRnDProgressMultiplierChart';
import type { MilestoneMap } from '@/types/milestones';
import { SmallChartMetricTooltip } from './SmallChartMetricTooltip';
import { ParameterHoverProvider } from './ParameterHoverContext';
import { loadBenchmarkData } from '@/utils/benchmarkLoader';
// import ModelDiagram from '@/svgs/model-diagram-semantic.svg';
import { SamplingConfig, generateParameterSampleWithUserValues, initializeCorrelationSampling, extractSamplingConfigBounds } from '@/utils/sampling';
import type { ComputeApiResponse } from '@/lib/serverApi';
import Link from 'next/link';
import { AdvancedSections } from './AdvancedSections';
import { ParameterSlider } from './ParameterSlider';
import { KEY_PARAMETER_CONFIGS } from '@/constants/keyParameterConfigs';

interface SampleTrajectoryWithParams {
  trajectory: ChartDataPoint[];
  params: Record<string, number | string | boolean>;
}

interface PlaygroundClientProps {
  benchmarkData: BenchmarkPoint[];
  initialComputeData: ComputeApiResponse;
  initialParameters: ParametersType;
  initialSampleTrajectories?: { trajectory: { year: number; horizonLength: number; effectiveCompute: number; automationFraction?: number; trainingCompute?: number; aiSwProgressMultRefPresentDay?: number }[]; params: Record<string, number | string | boolean> }[];
  initialSeed?: number;
}

interface SmallChartDef {
  key: string;
  title: string;
  tooltip: (p: DataPoint) => React.ReactNode;
  yFormatter?: (v: number) => string | React.ReactNode;
  yScale?: 'linear' | 'log';
  scaleType: 'log' | 'linear';
  logSuffix?: string;
  isDataInLogForm?: boolean;
}

const SIMULATION_END_YEAR = 2200; // Integrate over the full input time series (to 2200)
// Use relative URL to go through Next.js proxy (configured in next.config.ts)
// This avoids CORS issues and works in both dev and production
const API_BASE_URL = '';

function processInitialData(data: ComputeApiResponse): ChartDataPoint[] {
  if (!data?.time_series) return [];
  return data.time_series.map((point) => ({
    year: point.year,
    horizonLength: point.horizonLength,
    horizonFormatted: formatWorkTimeDuration(point.horizonLength),
    effectiveCompute: point.effectiveCompute,
    trainingCompute: point.trainingCompute ?? null,
    trainingSystemPerformance: point.trainingSystemPerformance ?? null,
    totalRdComputeH100e: point.totalRdComputeH100e ?? null,
    automationFraction: point.automationFraction ?? null,
    aiSoftwareProgressMultiplier: point.aiSoftwareProgressMultiplier,
    aiSwProgressMultRefPresentDay: point.aiSwProgressMultRefPresentDay,
    aiResearchTaste: point.aiResearchTaste ?? null,
    experimentCapacity: point.experimentCapacity ?? null,
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
    serialCodingLaborMultiplier: point.serialCodingLaborMultiplier ?? null,
    humanLabor: point.humanLabor ?? null,
  }));
}

// Simple slider component for playground
function SimpleSlider({
  label,
  value,
  onChange,
  onAfterChange,
  min,
  max,
  step,
  formatValue,
  disabled = false,
}: {
  label: string;
  value: number;
  onChange: (v: number) => void;
  onAfterChange: (v: number) => void;
  min: number;
  max: number;
  step: number;
  formatValue?: (v: number) => string;
  disabled?: boolean;
}) {
  return (
    <div className={`space-y-1 ${disabled ? 'opacity-50' : ''}`}>
      <div className="flex justify-between text-xs">
        <span className="text-gray-600">{label}</span>
        <span className="text-gray-800 font-mono">{formatValue ? formatValue(value) : value.toFixed(2)}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        onMouseUp={(e) => onAfterChange(parseFloat((e.target as HTMLInputElement).value))}
        onTouchEnd={(e) => onAfterChange(parseFloat((e.target as HTMLInputElement).value))}
        disabled={disabled}
        className="w-full h-1.5 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-emerald-600 disabled:cursor-not-allowed"
      />
    </div>
  );
}

// Simple checkbox component
function SimpleCheckbox({
  label,
  checked,
  onChange,
}: {
  label: string;
  checked: boolean;
  onChange: (checked: boolean) => void;
}) {
  return (
    <label className="flex items-center gap-2 text-xs cursor-pointer">
      <input
        type="checkbox"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
        className="w-3.5 h-3.5 rounded border-gray-300 text-emerald-600 focus:ring-emerald-500"
      />
      <span className="text-gray-700">{label}</span>
    </label>
  );
}

export default function PlaygroundClient({
  benchmarkData = [],
  initialComputeData,
  initialParameters,
  initialSampleTrajectories = [],
}: PlaygroundClientProps) {
  const initialChartData = useMemo(() => processInitialData(initialComputeData), [initialComputeData]);
  const resolvedInitialParameters = useMemo(() => initialParameters ?? { ...DEFAULT_PARAMETERS }, [initialParameters]);

  const convertedInitialSamples = useMemo(() => {
    return initialSampleTrajectories.map(sample => ({
      trajectory: sample.trajectory.map(point => ({
        year: point.year,
        horizonLength: point.horizonLength,
        horizonFormatted: formatWorkTimeDuration(point.horizonLength),
        effectiveCompute: point.effectiveCompute,
        automationFraction: point.automationFraction,
        trainingCompute: point.trainingCompute ?? null,
        aiSwProgressMultRefPresentDay: point.aiSwProgressMultRefPresentDay,
      })) as ChartDataPoint[],
      params: sample.params
    }));
  }, [initialSampleTrajectories]);

  const [chartData, setChartData] = useState<ChartDataPoint[]>(initialChartData);
  const [scHorizonMinutes, setScHorizonMinutes] = useState<number>(
    Math.pow(10, DEFAULT_PARAMETERS.ac_time_horizon_minutes)
  );
  const [parameters, setParameters] = useState<ParametersType>(resolvedInitialParameters);
  const [uiParameters, setUiParameters] = useState<ParametersType>(resolvedInitialParameters);
  const [milestones, setMilestones] = useState<MilestoneMap | null>(
    initialComputeData?.milestones && typeof initialComputeData.milestones === 'object'
      ? (initialComputeData.milestones as MilestoneMap)
      : null
  );
  const [mainLoading, setMainLoading] = useState(false);
  const [sampleTrajectories, setSampleTrajectories] = useState<SampleTrajectoryWithParams[]>(convertedInitialSamples);
  const [samplingConfig, setSamplingConfig] = useState<SamplingConfig | null>(null);
  const [enabledSamplingParams, setEnabledSamplingParams] = useState<Set<string>>(new Set());
  const [resampleTrigger, setResampleTrigger] = useState(0);
  const [numSamples, setNumSamples] = useState(10);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [selectedPresetId, setSelectedPresetId] = useState<string>('');
  const baselineParameters = useMemo(() => {
    const id = selectedPresetId || CURRENT_DEFAULT_BASELINE_ID;
    return PRESET_BASELINES[id] ?? DEFAULT_PARAMETERS;
  }, [selectedPresetId]);
  const [allParameters] = useState<{ defaults?: Record<string, unknown>; bounds?: Record<string, [number, number]>; metadata?: Record<string, unknown> } | null>(null);

  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();

  // Model simplification checkboxes
  const [enableExperimentAutomation, setEnableExperimentAutomation] = useState(DEFAULT_CHECKBOX_STATES.enableExperimentAutomation);
  const [enableSoftwareProgress, setEnableSoftwareProgress] = useState(DEFAULT_CHECKBOX_STATES.enableSoftwareProgress);
  const [useExperimentThroughputCES, setUseExperimentThroughputCES] = useState(DEFAULT_CHECKBOX_STATES.useExperimentThroughputCES);
  const [useComputeLaborGrowthSlowdown, setUseComputeLaborGrowthSlowdown] = useState(DEFAULT_CHECKBOX_STATES.useComputeLaborGrowthSlowdown);

  // Horizon chart benchmark percentile toggle (p50 vs p80)
  const [horizonPercentile, setHorizonPercentile] = useState<'p50' | 'p80'>('p80');
  const effectiveBenchmarkData = useMemo(
    () => horizonPercentile === 'p80' ? benchmarkData : loadBenchmarkData('p50'),
    [benchmarkData, horizonPercentile]
  );

  const currentRequestRef = useRef<AbortController | null>(null);
  const sampleTrajectoriesAbortRef = useRef<AbortController | null>(null);
  const hasUsedInitialSamplesRef = useRef(convertedInitialSamples.length > 0);
  const importInputRef = useRef<HTMLInputElement>(null);
  const isHydratingFromUrlRef = useRef(false);

  const displayEndYear = 2060;

  // Sampling config bounds for slider ranges
  const samplingConfigBounds = useMemo(() => {
    if (!samplingConfig) return {};
    return extractSamplingConfigBounds(samplingConfig);
  }, [samplingConfig]);

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

  // Bounds for specific parameter groups
  const scHorizonLogBounds = useMemo(() => {
    const samplingBounds = samplingConfigBounds?.ac_time_horizon_minutes;
    if (samplingBounds?.min !== undefined && samplingBounds?.max !== undefined) {
      return { min: samplingBounds.min, max: samplingBounds.max };
    }
    return { min: 3, max: 11 };
  }, [samplingConfigBounds]);

  const parallelPenaltyBounds = useMemo(() => {
    const samplingBounds = samplingConfigBounds?.parallel_penalty;
    if (samplingBounds?.min !== undefined && samplingBounds?.max !== undefined) {
      return { min: samplingBounds.min, max: samplingBounds.max };
    }
    return { min: 0, max: 1 };
  }, [samplingConfigBounds]);

  // Commit UI parameters to actual parameters when dragging ends
  const commitParameters = useCallback((nextParameters?: ParametersType) => {
    const newParams = nextParameters ?? uiParameters;
    setParameters(newParams);
    setIsDragging(false);
    fetchComputeData(newParams);
    setResampleTrigger(prev => prev + 1);
  }, [uiParameters]);

  // Load sampling config
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
          initializeCorrelationSampling(data.config.correlation_matrix);
          setSamplingConfig(data.config);

          const allParams = new Set<string>();
          for (const paramName of Object.keys(data.config.parameters)) {
            allParams.add(paramName);
          }
          if (data.config.time_series_parameters) {
            for (const paramName of Object.keys(data.config.time_series_parameters)) {
              allParams.add(paramName);
            }
          }
          setEnabledSamplingParams(allParams);
        }
      } catch (error) {
        console.error('Failed to load sampling configuration:', error);
      }
    };

    loadSamplingConfig();

    return () => {
      isCancelled = true;
    };
  }, []);

  // Update SC horizon line immediately while slider moves
  useEffect(() => {
    const instantHorizon = Math.pow(10, uiParameters.ac_time_horizon_minutes);
    setScHorizonMinutes(instantHorizon);
  }, [uiParameters.ac_time_horizon_minutes]);

  // Fetch compute data
  const fetchComputeData = useCallback(async (params: ParametersType) => {
    if (currentRequestRef.current) {
      currentRequestRef.current.abort();
    }

    const controller = new AbortController();
    currentRequestRef.current = controller;
    setMainLoading(true);

    try {
      const apiParameters = convertParametersToAPIFormat(params as unknown as ParameterRecord);
      const response = await fetch(`${API_BASE_URL}/api/compute`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          parameters: apiParameters,
          time_range: [2017, SIMULATION_END_YEAR],
          initial_progress: 0.0,
          // When compute/labor growth slowdown is disabled, use exponential growth inputs
          use_exponential_growth_inputs: !useComputeLaborGrowthSlowdown,
        }),
        signal: controller.signal,
      });

      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();

      if (data.success && data.time_series) {
        setChartData(processInitialData(data));
        if (data.milestones) {
          // Log raw milestone times for debugging
          console.log('[PlaygroundClient] Raw AC time:', data.milestones['AC']?.time);
          console.log('[PlaygroundClient] Raw SAR time:', data.milestones['SAR-level-experiment-selection-skill']?.time);
          setMilestones(data.milestones as MilestoneMap);
        }
      }
    } catch (e) {
      if (e instanceof Error && e.name === 'AbortError') return;
      console.error('Fetch error:', e);
    } finally {
      setMainLoading(false);
    }
  }, [useComputeLaborGrowthSlowdown]);

  // Fetch sample trajectory
  const fetchSampleTrajectory = useCallback(async (params: Record<string, number | string | boolean>, signal?: AbortSignal) => {
    try {
      const apiParams = convertSampledParametersToAPIFormat(params as ParameterRecord);
      const response = await fetch(`${API_BASE_URL}/api/compute`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          parameters: apiParams,
          time_range: [2017, SIMULATION_END_YEAR],
          initial_progress: 0.0,
          // When compute/labor growth slowdown is disabled, use exponential growth inputs
          use_exponential_growth_inputs: !useComputeLaborGrowthSlowdown,
        }),
        signal,
      });

      if (!response.ok) return null;
      const data = await response.json();
      return data.success ? data.time_series : null;
    } catch {
      return null;
    }
  }, [useComputeLaborGrowthSlowdown]);

  // Convert parameters for sampling (handles UI format differences)
  const convertParametersForSampling = useCallback((params: ParametersType): Record<string, number | string | boolean> => {
    const rawParams = params as unknown as Record<string, number | string | boolean>;
    const userParams: Record<string, number | string | boolean> = { ...rawParams };

    if (typeof rawParams.ac_time_horizon_minutes === 'number') {
      userParams.ac_time_horizon_minutes = Math.pow(10, rawParams.ac_time_horizon_minutes);
    }

    if ('benchmarks_and_gaps_mode' in rawParams) {
      const includeGap = rawParams.benchmarks_and_gaps_mode === true || rawParams.benchmarks_and_gaps_mode === 'gap';
      userParams.include_gap = includeGap ? 'gap' : 'no gap';
    }
    if ('coding_labor_exponent' in rawParams && typeof rawParams.coding_labor_exponent === 'number') {
      userParams.parallel_penalty = rawParams.coding_labor_exponent;
    }

    return userParams;
  }, []);

  // Fetch sample trajectories when parameters change
  useEffect(() => {
    if (!samplingConfig) return;

    if (hasUsedInitialSamplesRef.current && sampleTrajectories.length > 0) {
      hasUsedInitialSamplesRef.current = false;
      return;
    }

    if (sampleTrajectoriesAbortRef.current) {
      sampleTrajectoriesAbortRef.current.abort();
    }

    const abortController = new AbortController();
    sampleTrajectoriesAbortRef.current = abortController;

    const fetchAllSamples = async () => {
      const userParams = convertParametersForSampling(parameters);

      const samplePromises = Array.from({ length: numSamples }, async () => {
        const sampleParams = generateParameterSampleWithUserValues(samplingConfig, userParams, enabledSamplingParams);

        try {
          const trajectory = await fetchSampleTrajectory(sampleParams, abortController.signal);

          if (abortController.signal.aborted) return null;

          if (trajectory) {
            setSampleTrajectories(prev => [...prev, {
              trajectory: trajectory.map((p: { year: number; horizonLength: number; effectiveCompute: number; automationFraction?: number; trainingCompute?: number; aiSwProgressMultRefPresentDay?: number }) => ({
                ...p,
                horizonFormatted: formatWorkTimeDuration(p.horizonLength),
              })) as ChartDataPoint[],
              params: sampleParams as Record<string, number | string | boolean>
            }]);
          }
          return trajectory;
        } catch {
          return null;
        }
      });

      await Promise.allSettled(samplePromises);
    };

    setSampleTrajectories([]);
    fetchAllSamples();

    return () => {
      abortController.abort();
    };
  }, [samplingConfig, fetchSampleTrajectory, resampleTrigger, enabledSamplingParams, parameters, convertParametersForSampling, numSamples]);

  // Handle parameter change
  const handleParameterChange = useCallback((param: keyof ParametersType, value: number) => {
    setUiParameters(prev => ({ ...prev, [param]: value }));
  }, []);

  const handleParameterCommit = useCallback((param: keyof ParametersType, value: number) => {
    setParameters(prev => {
      const newParams = { ...prev, [param]: value };
      fetchComputeData(newParams);
      return newParams;
    });
    setResampleTrigger(prev => prev + 1);
  }, [fetchComputeData]);

  // Handle checkbox changes
  const handleCheckboxChange = useCallback((
    setter: React.Dispatch<React.SetStateAction<boolean>>,
    value: boolean
  ) => {
    setter(value);
    fetchComputeData(parameters);
    setResampleTrigger(prev => prev + 1);
  }, [fetchComputeData, parameters]);

  // Export/Import handlers
  const handleExportParameters = useCallback(() => {
    exportParametersToFile(uiParameters);
  }, [uiParameters]);

  const handleExportCSV = useCallback(() => {
    if (!chartData || chartData.length === 0) {
      console.warn('No chart data available to export');
      return;
    }

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
    ];

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
      fetchComputeData(imported);
      setResampleTrigger(prev => prev + 1);
    }
    event.target.value = '';
  }, [fetchComputeData]);

  const handleImportButtonClick = useCallback(() => {
    importInputRef.current?.click();
  }, []);

  const handlePresetSelect = useCallback((event: ChangeEvent<HTMLSelectElement>) => {
    const presetId = event.target.value;
    setSelectedPresetId(presetId);

    // Apply preset parameters if a preset is selected
    if (presetId && PRESET_BASELINES[presetId]) {
      const newParams = { ...PRESET_BASELINES[presetId] };
      setParameters(newParams);
      setUiParameters(newParams);
      fetchComputeData(newParams);
      setResampleTrigger(prev => prev + 1);
    }
  }, [fetchComputeData]);

  // URL sync - sync checkbox states from URL on mount
  // Note: The server already pre-computes with URL params, so we don't need to refetch
  useEffect(() => {
    const urlState = decodeFullStateFromParams(searchParams);
    if (urlState) {
      isHydratingFromUrlRef.current = true;
      // Sync preset and checkbox states from URL (parameters are already set from server via initialParameters)
      setSelectedPresetId(urlState.presetId);
      setEnableExperimentAutomation(urlState.enableExperimentAutomation);
      setEnableSoftwareProgress(urlState.enableSoftwareProgress);
      setHorizonPercentile(urlState.horizonPercentile);
      // Allow checkbox effects to run after hydration
      setTimeout(() => {
        isHydratingFromUrlRef.current = false;
      }, 0);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    const newParams = encodeFullStateToParams({
      parameters,
      presetId: selectedPresetId || CURRENT_DEFAULT_BASELINE_ID,
      enableExperimentAutomation,
      useExperimentThroughputCES: DEFAULT_CHECKBOX_STATES.useExperimentThroughputCES,
      enableSoftwareProgress,
      useComputeLaborGrowthSlowdown: DEFAULT_CHECKBOX_STATES.useComputeLaborGrowthSlowdown,
      horizonPercentile,
      pause: DEFAULT_PAUSE_STATE,
    });
    const newUrl = `${pathname}?${newParams.toString()}`;
    router.replace(newUrl, { scroll: false });
  }, [parameters, selectedPresetId, enableExperimentAutomation, enableSoftwareProgress, horizonPercentile, pathname, router]);

  // Sync checkbox states with parameter values.
  // Unchecked: applies the override values and locks the affected sliders.
  // Re-checked: restores whatever those parameters held before the override.
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

  // Refetch data when checkbox-controlled parameters change
  // This is needed because useCheckboxParameterSync updates parameters but doesn't trigger fetches
  const prevCheckboxStatesRef = useRef({
    enableExperimentAutomation,
    useExperimentThroughputCES,
    enableSoftwareProgress,
    useComputeLaborGrowthSlowdown,
  });

  useEffect(() => {
    const prev = prevCheckboxStatesRef.current;
    const changed =
      prev.enableExperimentAutomation !== enableExperimentAutomation ||
      prev.useExperimentThroughputCES !== useExperimentThroughputCES ||
      prev.enableSoftwareProgress !== enableSoftwareProgress ||
      prev.useComputeLaborGrowthSlowdown !== useComputeLaborGrowthSlowdown;

    if (changed && !isHydratingFromUrlRef.current) {
      prevCheckboxStatesRef.current = {
        enableExperimentAutomation,
        useExperimentThroughputCES,
        enableSoftwareProgress,
        useComputeLaborGrowthSlowdown,
      };
      // Use a microtask to ensure parameters have been updated by useCheckboxParameterSync
      queueMicrotask(() => {
        fetchComputeData(parameters);
        setResampleTrigger(prev => prev + 1);
      });
    }
  }, [enableExperimentAutomation, useExperimentThroughputCES, enableSoftwareProgress, useComputeLaborGrowthSlowdown, parameters, fetchComputeData]);

  // Render data
  const renderData = useMemo(() => chartData, [chartData]);
  const hasData = renderData.length > 0;

  // Tooltips
  const AIProgressMultiplierTooltip = useCallback((point: DataPoint) => {
    const y = point.aiSwProgressMultRefPresentDay;
    if (typeof y !== 'number' || !Number.isFinite(y)) return null;
    return (
      <div style={tooltipBoxStyle}>
        <div style={tooltipHeaderStyle}>{Math.round(point.x)}</div>
        <div style={tooltipValueStyle}>{formatUplift(y)}</div>
      </div>
    );
  }, []);

  const HorizonTooltip = useCallback((point: DataPoint) => {
    const y = point.horizonLength;
    if (typeof y !== 'number' || !Number.isFinite(y)) return null;
    return (
      <div style={tooltipBoxStyle}>
        <div style={tooltipHeaderStyle}>{Math.round(point.x)}</div>
        <div style={tooltipValueStyle}>{formatWorkTimeDuration(y)}</div>
      </div>
    );
  }, []);

  // Reference line for automation threshold
  const automationReferenceLineYear = milestones?.['100%-automation-fraction']?.year;
  const automationReferenceLine = typeof automationReferenceLineYear === 'number'
    ? { x: automationReferenceLineYear, stroke: '#888', strokeDasharray: '4 4', strokeWidth: 1 }
    : undefined;

  // All small chart definitions (matching ProgressChartClient)
  const smallCharts: SmallChartDef[] = useMemo(() => ([
    {
      key: 'automationFraction',
      title: 'Coding Automation Fraction',
      tooltip: (p) => <SmallChartMetricTooltip point={p} fieldName="automationFraction" formatter={(v) => `${(v * 100).toFixed(1)}%`} />,
      yFormatter: (v) => `${(v * 100).toFixed(1)}%`,
      scaleType: 'linear'
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
      key: 'serialCodingLaborMultiplier',
      title: 'AI Serial Coding Labor Multiplier',
      tooltip: (p) => <SmallChartMetricTooltip point={p} fieldName="serialCodingLaborMultiplier" formatter={(v) => formatCompactNumberNode(v, { suffix: 'x', renderMode: 'html' })} requirePositive />,
      yScale: 'log',
      scaleType: 'log',
      logSuffix: 'x',
      yFormatter: (v) => formatAsPowerOfTenText(v, { suffix: 'x' })
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
      key: 'researchEffort',
      title: 'Software Research Effort',
      tooltip: (p) => <SmallChartMetricTooltip point={p} fieldName="researchEffort" formatter={(v) => formatCompactNumberNode(v, { renderMode: 'html' })} requirePositive />,
      yScale: 'log',
      scaleType: 'log',
      yFormatter: (v) => formatAsPowerOfTenText(v)
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
      key: 'softwareEfficiency',
      title: 'Software Efficiency',
      tooltip: (p) => <SmallChartMetricTooltip point={p} fieldName="softwareEfficiency" formatter={(v) => formatCompactNumberNode(Math.pow(10, v), { renderMode: 'html' })} />,
      yScale: 'log',
      scaleType: 'log',
      yFormatter: (v) => formatAsPowerOfTenText(Math.pow(10, v)),
      isDataInLogForm: true
    },
    {
      // ETSP = training compute (hardware) + software efficiency, shown as a stacked decomposition.
      key: 'etsp',
      title: 'Effective Training System Performance (TC + SW)',
      tooltip: (p) => <SmallChartMetricTooltip point={p} fieldName="etsp" formatter={(v) => formatCompactNumberNode(Math.pow(10, v), { renderMode: 'html' })} />,
      // OOM metric: group with the other log-section charts; isDataInLogForm keeps
      // the axis linear-in-OOMs so the stacked bands add correctly.
      yScale: 'log',
      scaleType: 'log',
      yFormatter: (v) => formatAsPowerOfTenText(Math.pow(10, v)),
      isDataInLogForm: true
    },
    {
      key: 'experimentCapacity',
      title: 'Experiment Throughput',
      tooltip: (p) => <SmallChartMetricTooltip point={p} fieldName="experimentCapacity" formatter={(v) => formatCompactNumberNode(v, { renderMode: 'html' })} requirePositive />,
      yScale: 'log',
      scaleType: 'log',
      yFormatter: (v) => formatAsPowerOfTenText(v)
    },
    {
      key: 'inferenceCompute',
      title: 'Inference Compute for Coding',
      tooltip: (p) => <SmallChartMetricTooltip point={p} fieldName="inferenceCompute" formatter={(v) => formatCompactNumberNode(v, { suffix: ' H100e', renderMode: 'html' })} requirePositive />,
      yScale: 'log',
      scaleType: 'log',
      yFormatter: (v) => formatAsPowerOfTenText(v, { suffix: ' H100e' }),
      logSuffix: ' H100e'
    },
    {
      key: 'experimentCompute',
      title: 'Experiment Compute',
      tooltip: (p) => <SmallChartMetricTooltip point={p} fieldName="experimentCompute" formatter={(v) => formatCompactNumberNode(v, { suffix: ' H100e', renderMode: 'html' })} requirePositive />,
      yScale: 'log',
      scaleType: 'log',
      yFormatter: (v) => formatAsPowerOfTenText(v, { suffix: ' H100e' }),
      logSuffix: ' H100e'
    },
    {
      key: 'humanLabor',
      title: 'Human Coding Labor',
      tooltip: (p) => <SmallChartMetricTooltip point={p} fieldName="humanLabor" formatter={(v) => formatCompactNumberNode(v, { renderMode: 'html' })} requirePositive />,
      yScale: 'log',
      scaleType: 'log',
      yFormatter: (v) => formatAsPowerOfTenText(v)
    },
  ]), []);

  const mainChartHeight = 350;
  const smallChartHeight = 100;

  return (
    <ParameterHoverProvider>
      {/* Desktop-only warning */}
      <div className="lg:hidden h-screen flex items-center justify-center bg-vivid-background p-8">
        <div className="text-center">
          <h1 className="text-2xl font-bold mb-4">Playground is Desktop Only</h1>
          <p className="text-gray-600 mb-6">The playground requires a larger screen for the best experience.</p>
          <Link href="/" className="text-blue-600 hover:underline">← Back to main site</Link>
        </div>
      </div>

      {/* Desktop layout using CSS Grid */}
      {/* 
        Grid structure:
        +------------------+---------------------------+
        |   Sidebar with   | Chart | Chart | Chart     |
        |   Model Diagram  |   1   |   2   |   3       |
        |   & Parameters   +---------------------------+
        |   (row-span-2)   |    All Small Charts       |
        +------------------+---------------------------+
      */}
      <div
        className="hidden lg:grid h-screen bg-vivid-background overflow-hidden"
        style={{
          gridTemplateColumns: '260px 1fr',
          gridTemplateRows: 'minmax(300px, 380px) 1fr',
        }}
      >
        {/* Sidebar - spans all rows */}
        <aside className="row-span-2 border-r border-gray-200/60 bg-vivid-background flex flex-col overflow-hidden">
          <div className="px-3 py-2 border-b border-gray-200/60 shrink-0">
            <Link href="/" className="text-[10px] text-gray-500 hover:text-gray-700 mb-1 block font-system-mono">← Back</Link>
            <h1 className="text-sm font-bold text-primary font-et-book">Playground</h1>
          </div>

          {/* Model Diagram - commented out */}
          {/* <div className="px-2 py-2 border-b border-gray-200/60 shrink-0">
            <div className="flex gap-1 items-center mb-1">
              <span className="leading-tight text-[10px] font-semibold text-primary text-left font-system-mono">Model</span>
              <div className="flex-1 border-t border-gray-500/30" />
            </div>
            <div className="opacity-80 hover:opacity-100 transition-opacity">
              <ModelDiagram className="w-full h-auto" />
            </div>
          </div> */}

          <div className="flex-1 overflow-y-auto px-2 py-2">
            {/* Settings - Export/Import/Presets */}
            <div className="mb-2">
              <div className="flex gap-1 items-center mb-1.5">
                <span className="leading-tight text-[10px] font-semibold text-primary text-left font-system-mono">Settings</span>
                <div className="flex-1 border-t border-gray-500/30" />
              </div>
              <div className="space-y-2">
                <div className="flex flex-wrap gap-1.5">
                  <button
                    type="button"
                    onClick={handleExportParameters}
                    className="flex-1 min-w-[80px] px-2 py-1.5 text-[10px] font-semibold uppercase tracking-[0.1em] text-white bg-[#2A623D] border border-[#215337] rounded-sm shadow-sm hover:bg-[#245632]"
                  >
                    Export
                  </button>
                  <button
                    type="button"
                    onClick={handleImportButtonClick}
                    className="flex-1 min-w-[80px] px-2 py-1.5 text-[10px] font-semibold uppercase tracking-[0.1em] text-[#2A623D] bg-white border border-[#2A623D]/40 rounded-sm shadow-sm hover:bg-[#F4F8F5]"
                  >
                    Import
                  </button>
                  <input
                    ref={importInputRef}
                    type="file"
                    accept="application/json"
                    className="hidden"
                    onChange={handleImportParameters}
                  />
                </div>
                <div className="space-y-1">
                  <label className="block text-[10px] font-semibold uppercase tracking-[0.1em] text-primary font-system-mono">
                    Presets
                  </label>
                  <select
                    value={selectedPresetId}
                    onChange={handlePresetSelect}
                    className="w-full px-2 py-1.5 border border-gray-300 rounded-sm bg-white text-[11px] font-system-mono focus:outline-none focus:ring-1 focus:ring-[#2A623D]"
                  >
                    <option value="">Select preset</option>
                    {PARAMETER_PRESETS.map((preset) => (
                      <option key={preset.id} value={preset.id}>{preset.label}</option>
                    ))}
                  </select>
                </div>
              </div>
            </div>

            {/* Monte Carlo Samples */}
            <div className="mb-2">
              <div className="flex gap-1 items-center mb-1.5">
                <span className="leading-tight text-[10px] font-semibold text-primary text-left font-system-mono">Monte Carlo</span>
                <div className="flex-1 border-t border-gray-500/30" />
              </div>
              <div className="space-y-0.5">
                <label className="block text-[11px] font-medium text-foreground leading-tight">
                  Sample Trajectories: {numSamples}
                </label>
                <input
                  type="range"
                  min={0}
                  max={50}
                  step={1}
                  value={numSamples}
                  onChange={(e) => setNumSamples(parseInt(e.target.value))}
                  className="w-full rounded-lg appearance-none cursor-pointer slider"
                />
              </div>
            </div>

            {/* Key Parameters */}
            <div className="mb-2">
              <div className="flex gap-1 items-center mb-1.5">
                <span className="leading-tight text-[10px] font-semibold text-primary text-left font-system-mono">Key Parameters</span>
                <div className="flex-1 border-t border-gray-500/30" />
              </div>
              <div className="space-y-2">
                {/* Present Doubling Time */}
                <div className="relative flex w-full flex-col">
                  <div className="flex items-center justify-between mb-0.5">
                    <label className="block w-full text-[11px] font-medium text-foreground">
                      <WithChartTooltip
                        explanation={KEY_PARAMETER_CONFIGS.present_doubling_time.description}
                        docAnchor={KEY_PARAMETER_CONFIGS.present_doubling_time.docAnchor}
                        fullWidth
                      >
                        {KEY_PARAMETER_CONFIGS.present_doubling_time.label}
                      </WithChartTooltip>
                    </label>
                  </div>
                  <ParameterSlider
                    paramName="present_doubling_time"
                    label=""
                    customMin={samplingConfigBounds.present_doubling_time?.min}
                    fallbackMin={KEY_PARAMETER_CONFIGS.present_doubling_time.fallbackMin}
                    fallbackMax={KEY_PARAMETER_CONFIGS.present_doubling_time.fallbackMax}
                    step={KEY_PARAMETER_CONFIGS.present_doubling_time.step}
                    decimalPlaces={KEY_PARAMETER_CONFIGS.present_doubling_time.decimalPlaces}
                    customFormatValue={KEY_PARAMETER_CONFIGS.present_doubling_time.formatValue}
                    value={uiParameters.present_doubling_time}
                    uiParameters={uiParameters}
                    setUiParameters={setUiParameters}
                    allParameters={allParameters}
                    isDragging={isDragging}
                    setIsDragging={setIsDragging}
                    commitParameters={commitParameters}
                    baselineParameters={baselineParameters}
                    useLogScale={KEY_PARAMETER_CONFIGS.present_doubling_time.useLogScale}
                  />
                </div>
                {/* AC Time Horizon (Target) */}
                <div className="relative flex w-full flex-col">
                  <div className="flex items-center justify-between mb-0.5">
                    <label className="block w-full text-[11px] font-medium text-foreground">
                      <WithChartTooltip
                        explanation={KEY_PARAMETER_CONFIGS.ac_time_horizon_minutes.description}
                        docAnchor={KEY_PARAMETER_CONFIGS.ac_time_horizon_minutes.docAnchor}
                        fullWidth
                      >
                        {KEY_PARAMETER_CONFIGS.ac_time_horizon_minutes.label}
                      </WithChartTooltip>
                    </label>
                  </div>
                  <ParameterSlider
                    paramName="ac_time_horizon_minutes"
                    label=""
                    customMin={scHorizonLogBounds.min}
                    customMax={scHorizonLogBounds.max}
                    step={KEY_PARAMETER_CONFIGS.ac_time_horizon_minutes.step}
                    customFormatValue={KEY_PARAMETER_CONFIGS.ac_time_horizon_minutes.formatValue}
                    value={uiParameters.ac_time_horizon_minutes}
                    uiParameters={uiParameters}
                    setUiParameters={setUiParameters}
                    allParameters={allParameters}
                    isDragging={isDragging}
                    setIsDragging={setIsDragging}
                    commitParameters={commitParameters}
                    baselineParameters={baselineParameters}
                    useLogScale={KEY_PARAMETER_CONFIGS.ac_time_horizon_minutes.useLogScale}
                  />
                </div>
                {/* Doubling Difficulty Growth Factor */}
                <div className="relative flex w-full flex-col">
                  <div className="flex items-center justify-between mb-0.5">
                    <label className="block w-full text-[11px] font-medium text-foreground">
                      <WithChartTooltip
                        explanation={KEY_PARAMETER_CONFIGS.doubling_difficulty_growth_factor.description}
                        docAnchor={KEY_PARAMETER_CONFIGS.doubling_difficulty_growth_factor.docAnchor}
                        fullWidth
                      >
                        {KEY_PARAMETER_CONFIGS.doubling_difficulty_growth_factor.label}
                      </WithChartTooltip>
                    </label>
                  </div>
                  <ParameterSlider
                    paramName="doubling_difficulty_growth_factor"
                    label=""
                    customMin={KEY_PARAMETER_CONFIGS.doubling_difficulty_growth_factor.customMin}
                    customMax={KEY_PARAMETER_CONFIGS.doubling_difficulty_growth_factor.customMax}
                    step={KEY_PARAMETER_CONFIGS.doubling_difficulty_growth_factor.step}
                    customFormatValue={KEY_PARAMETER_CONFIGS.doubling_difficulty_growth_factor.formatValue}
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
                {/* Research Taste Slope */}
                <div className="relative flex w-full flex-col">
                  <div className="flex items-center justify-between mb-0.5">
                    <label className="block w-full text-[11px] font-medium text-foreground">
                      <WithChartTooltip
                        explanation={KEY_PARAMETER_CONFIGS.ai_research_taste_slope.description}
                        docAnchor={KEY_PARAMETER_CONFIGS.ai_research_taste_slope.docAnchor}
                        fullWidth
                      >
                        {KEY_PARAMETER_CONFIGS.ai_research_taste_slope.label}
                      </WithChartTooltip>
                    </label>
                  </div>
                  <ParameterSlider
                    paramName="ai_research_taste_slope"
                    label=""
                    step={KEY_PARAMETER_CONFIGS.ai_research_taste_slope.step}
                    customMin={samplingConfigBounds.ai_research_taste_slope?.min}
                    customMax={samplingConfigBounds.ai_research_taste_slope?.max}
                    fallbackMin={KEY_PARAMETER_CONFIGS.ai_research_taste_slope.fallbackMin}
                    fallbackMax={KEY_PARAMETER_CONFIGS.ai_research_taste_slope.fallbackMax}
                    decimalPlaces={KEY_PARAMETER_CONFIGS.ai_research_taste_slope.decimalPlaces}
                    customFormatValue={KEY_PARAMETER_CONFIGS.ai_research_taste_slope.formatValue}
                    value={uiParameters.ai_research_taste_slope}
                    uiParameters={uiParameters}
                    setUiParameters={setUiParameters}
                    allParameters={allParameters}
                    isDragging={isDragging}
                    setIsDragging={setIsDragging}
                    commitParameters={commitParameters}
                    baselineParameters={baselineParameters}
                    useLogScale={KEY_PARAMETER_CONFIGS.ai_research_taste_slope.useLogScale}
                  />
                </div>
                {/* Research Taste Median to Top Multiplier */}
                <div className="relative flex w-full flex-col">
                  <div className="flex items-center justify-between mb-0.5">
                    <label className="block w-full text-[11px] font-medium text-foreground">
                      <WithChartTooltip
                        explanation={KEY_PARAMETER_CONFIGS.median_to_top_taste_multiplier.description}
                        docAnchor={KEY_PARAMETER_CONFIGS.median_to_top_taste_multiplier.docAnchor}
                        fullWidth
                      >
                        {KEY_PARAMETER_CONFIGS.median_to_top_taste_multiplier.label}
                      </WithChartTooltip>
                    </label>
                  </div>
                  <ParameterSlider
                    paramName="median_to_top_taste_multiplier"
                    label=""
                    customMin={samplingConfigBounds.median_to_top_taste_multiplier?.min}
                    customMax={samplingConfigBounds.median_to_top_taste_multiplier?.max}
                    fallbackMin={KEY_PARAMETER_CONFIGS.median_to_top_taste_multiplier.fallbackMin}
                    fallbackMax={KEY_PARAMETER_CONFIGS.median_to_top_taste_multiplier.fallbackMax}
                    step={KEY_PARAMETER_CONFIGS.median_to_top_taste_multiplier.step}
                    decimalPlaces={KEY_PARAMETER_CONFIGS.median_to_top_taste_multiplier.decimalPlaces}
                    customFormatValue={KEY_PARAMETER_CONFIGS.median_to_top_taste_multiplier.formatValue}
                    value={uiParameters.median_to_top_taste_multiplier}
                    uiParameters={uiParameters}
                    setUiParameters={setUiParameters}
                    allParameters={allParameters}
                    isDragging={isDragging}
                    setIsDragging={setIsDragging}
                    commitParameters={commitParameters}
                    baselineParameters={baselineParameters}
                    disabled={lockedParameters.has('median_to_top_taste_multiplier')}
                    useLogScale={KEY_PARAMETER_CONFIGS.median_to_top_taste_multiplier.useLogScale}
                  />
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
              lockedParameters={lockedParameters}
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
              horizonPercentile={horizonPercentile}
              onHorizonPercentileChange={setHorizonPercentile}
            />

            <div className="mt-2 pt-2 border-t border-gray-200/60">
              <button
                onClick={() => {
                  setParameters({ ...DEFAULT_PARAMETERS });
                  setUiParameters({ ...DEFAULT_PARAMETERS });
                  setEnableExperimentAutomation(DEFAULT_CHECKBOX_STATES.enableExperimentAutomation);
                  setEnableSoftwareProgress(DEFAULT_CHECKBOX_STATES.enableSoftwareProgress);
                  setUseExperimentThroughputCES(DEFAULT_CHECKBOX_STATES.useExperimentThroughputCES);
                  setUseComputeLaborGrowthSlowdown(DEFAULT_CHECKBOX_STATES.useComputeLaborGrowthSlowdown);
                  fetchComputeData({ ...DEFAULT_PARAMETERS });
                  setResampleTrigger(prev => prev + 1);
                }}
                className="text-[10px] text-gray-500 hover:text-gray-700"
              >
                Reset to defaults
              </button>
            </div>
          </div>
        </aside>

        <ChartSyncProvider>
          {/* Row 1: Main Charts */}
          <div className="border-b border-gray-200/60 grid grid-cols-3 gap-0 min-h-0">
            {/* Main Chart 1 - AI R&D Progress */}
            <div className="p-3 min-w-0 overflow-hidden border-r border-gray-200/60">
              <ChartLoadingOverlay isLoading={mainLoading} blocking={!hasData} className="h-full">
                <AIRnDProgressMultiplierChart
                  chartData={renderData}
                  tooltip={AIProgressMultiplierTooltip}
                  className="h-full"
                  milestones={milestones}
                  displayEndYear={displayEndYear}
                  verticalReferenceLine={automationReferenceLine}
                  height={mainChartHeight}
                  sampleTrajectories={sampleTrajectories.map(s => s.trajectory)}
                />
              </ChartLoadingOverlay>
            </div>

            {/* Main Chart 2 - Coding Time Horizon */}
            <div className="p-3 min-w-0 overflow-hidden border-r border-gray-200/60">
              <ChartLoadingOverlay isLoading={mainLoading} blocking={!hasData} className="h-full">
                <CustomHorizonChart
                  chartData={renderData}
                  scHorizonMinutes={scHorizonMinutes}
                  tooltip={HorizonTooltip}
                  formatTimeDuration={formatWorkTimeDuration}
                  benchmarkData={effectiveBenchmarkData}
                  className="h-full"
                  displayEndYear={displayEndYear}
                  height={mainChartHeight}
                  sampleTrajectories={sampleTrajectories.map(s => s.trajectory)}
                />
              </ChartLoadingOverlay>
            </div>

            {/* Main Chart 3 - Effective Compute */}
            <div className="p-3 min-w-0 overflow-hidden">
              <ChartLoadingOverlay isLoading={mainLoading} blocking={!hasData} className="h-full">
                <CombinedComputeChart
                  chartData={renderData}
                  title={enableSoftwareProgress ? 'Effective & Training Compute' : 'Effective Compute'}
                  verticalReferenceLine={automationReferenceLine}
                  className="h-full"
                  displayEndYear={displayEndYear}
                  height={mainChartHeight}
                  showTrainingSeries={enableSoftwareProgress}
                  sampleTrajectories={sampleTrajectories.map(s => s.trajectory)}
                />
              </ChartLoadingOverlay>
            </div>
          </div>

          {/* Row 2: All Small Charts in Uniform Grid */}
          <div className="p-3 overflow-auto">
            <div
              style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(4, 1fr)',
                gridAutoRows: `${smallChartHeight}px`,
                gap: '12px',
              }}
            >
              {smallCharts.map((chart) => (
                <ChartLoadingOverlay key={chart.key} isLoading={mainLoading} blocking={!hasData} className="h-full w-full">
                  <CustomMetricChart
                    chartData={renderData}
                    dataKey={chart.key}
                    title={chart.title}
                    tooltip={chart.tooltip}
                    yScale={chart.yScale}
                    yFormatter={chart.yFormatter}
                    displayEndYear={displayEndYear}
                    height={smallChartHeight}
                    logSuffix={chart.logSuffix}
                    isDataInLogForm={chart.isDataInLogForm}
                    sampleTrajectories={sampleTrajectories.map(s => s.trajectory)}
                  />
                </ChartLoadingOverlay>
              ))}
            </div>
          </div>
        </ChartSyncProvider>
      </div>
    </ParameterHoverProvider>
  );
}
