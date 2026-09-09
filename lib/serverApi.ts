import { promises as fs } from 'fs';
import path from 'path';
import yaml from 'js-yaml';
import { DEFAULT_PARAMETERS, ParametersType } from '@/constants/parameters';
import { convertParametersToAPIFormat, convertSampledParametersToAPIFormat, ParameterRecord } from '@/utils/monteCarlo';
import {
  SamplingConfig,
  SeededRandom,
  setSamplingRng,
  generateParameterSample,
  initializeCorrelationSampling
} from '@/utils/sampling';
import { getApiBaseUrl, getApiBypassHeaders } from './apiBase';

const SIMULATION_END_YEAR = 2200; // Integrate over the full input time series (to 2200)
const VISIBLE_CHART_START_YEAR = 2018;
const NUM_SAMPLES = 10;

const API_BASE_URL = getApiBaseUrl();

export interface TimeSeriesPoint {
  year: number;
  horizonLength: number;
  effectiveCompute: number;
  automationFraction: number;
  trainingCompute?: number;
  trainingSystemPerformance?: number;
  trainingComputeH100e?: number;
  totalRdComputeH100e?: number;
  experimentCapacity?: number;
  aiResearchTaste?: number;
  aiSoftwareProgressMultiplier?: number;
  aiSwProgressMultRefPresentDay?: number;
  serialCodingLaborMultiplier?: number;
  humanLabor?: number;
  inferenceCompute?: number;
  experimentCompute?: number;
  researchEffort?: number;
  researchStock?: number;
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

export interface ComputeApiResponse {
  success: boolean;
  summary?: Record<string, unknown>;
  time_series?: TimeSeriesPoint[];
  milestones?: Record<string, unknown>;
  horizon_params?: {
    uses_shifted_form: boolean;
    anchor_progress: number | null;
  };
  exp_capacity_params?: {
    rho: number | null;
    alpha: number | null;
    experiment_compute_exponent: number | null;
  };
  error?: string;
}

export interface SampleTrajectory {
  trajectory: TimeSeriesPoint[];
  params: Record<string, number | string | boolean>;
}

// Load sampling config from YAML file
async function loadSamplingConfig(): Promise<SamplingConfig | null> {
  try {
    const configPath = path.join(process.cwd(), 'config/sampling_config.yaml');
    const fileContents = await fs.readFile(configPath, 'utf8');
    const config = yaml.load(fileContents) as SamplingConfig;
    return config;
  } catch (error) {
    console.error('Failed to load sampling config:', error);
    return null;
  }
}

// Fetch a single trajectory from the API
async function fetchSingleTrajectory(
  params: Record<string, number | string | boolean>
): Promise<TimeSeriesPoint[] | null> {
  try {
    const apiParameters = convertSampledParametersToAPIFormat(params as unknown as ParameterRecord);
    
    const response = await fetch(`${API_BASE_URL}/api/compute`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...getApiBypassHeaders() },
      body: JSON.stringify({
        parameters: apiParameters,
        time_range: [2017, SIMULATION_END_YEAR],
        initial_progress: 0.0
      }),
    });

    if (!response.ok) {
      return null;
    }

    const data = await response.json();
    if (!data.success || !data.time_series) {
      return null;
    }

    return data.time_series.filter((point: TimeSeriesPoint) => point.year >= VISIBLE_CHART_START_YEAR);
  } catch {
    return null;
  }
}

// Fetch compute data from API (caching handled at component level)
export async function fetchComputeData(parameters: ParametersType): Promise<ComputeApiResponse> {
  try {
    const apiParameters = convertParametersToAPIFormat(parameters as unknown as ParameterRecord);
    
    const response = await fetch(`${API_BASE_URL}/api/compute`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...getApiBypassHeaders() },
      body: JSON.stringify({
        parameters: apiParameters,
        time_range: [2017, SIMULATION_END_YEAR],
        initial_progress: 0.0
      }),
    });

    if (!response.ok) {
      console.error(`Failed to fetch compute data: HTTP ${response.status}`);
      return { success: false, error: `HTTP ${response.status}` };
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Failed to fetch compute data:', error);
    return { success: false, error: String(error) };
  }
}

// Fetch sample trajectories with a seed (no server-side caching)
export async function fetchSampleTrajectories(
  baseParameters: ParametersType,
  seed: number
): Promise<SampleTrajectory[]> {
  const samplingConfig = await loadSamplingConfig();
  if (!samplingConfig) {
    console.error('[fetchSampleTrajectories] Failed to load sampling config');
    return [];
  }

  // Initialize correlation sampling if configured
  if (samplingConfig.correlation_matrix) {
    initializeCorrelationSampling(samplingConfig.correlation_matrix);
  }

  // Set up seeded RNG
  const rng = new SeededRandom(seed);
  setSamplingRng(rng);

  try {
    const samples: SampleTrajectory[] = [];

    // Generate and fetch samples sequentially to ensure deterministic ordering
    for (let i = 0; i < NUM_SAMPLES; i++) {
      const sampleParams = generateParameterSample(samplingConfig, baseParameters as Record<string, number | string | boolean>);
      const trajectory = await fetchSingleTrajectory(sampleParams);
      
      if (trajectory) {
        samples.push({
          trajectory,
          params: sampleParams
        });
      }
    }

    return samples;
  } finally {
    // Reset RNG to not affect other code
    setSamplingRng(null);
  }
}

// Convenience function for default parameters
export async function fetchInitialComputeData(): Promise<ComputeApiResponse> {
  return fetchComputeData({ ...DEFAULT_PARAMETERS });
}

// ---------------------------------------------------------------------------
// Plan A explorer server-side data (defaults + default compute result)
// ---------------------------------------------------------------------------

export interface PlanAExplorerServerData {
  defaults: Record<string, unknown>;
  result: {
    result: Record<string, unknown>;
    series: Array<Record<string, number | null>>;
    milestones: Record<string, { time?: number; effective_compute_ooms?: number }>;
  };
}

/**
 * Fetch the Plan A explorer's seed defaults AND compute the default result server-side —
 * the /p pattern (cached in the page via 'use cache'), mirroring the client's first
 * compute exactly (same knob seeding + parameter conversion as PlanAExplorerClient).
 * Returns null on ANY failure (backend unreachable at build time, missing covert routes)
 * — the page then falls back to the baked constants/planADefault*.json snapshots.
 */
export async function fetchPlanAExplorerData(): Promise<PlanAExplorerServerData | null> {
  try {
    const dRes = await fetch(`${API_BASE_URL}/api/plan-a/defaults`, {
      headers: { ...getApiBypassHeaders() },
    });
    if (!dRes.ok) return null;
    const defaults = await dRes.json();
    if (!defaults?.success) return null;

    const { BREAKDOWN_DEFAULTS } = await import('@/constants/planASeeds');
    const knobs = { ...BREAKDOWN_DEFAULTS, ...(defaults.knobs ?? {}) };
    const parameters = convertParametersToAPIFormat({
      ...DEFAULT_PARAMETERS,
      ...(defaults.model_params ?? {}),
    } as unknown as ParameterRecord);

    const [covRes, refRes] = await Promise.all([
      fetch(`${API_BASE_URL}/api/plan-a`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...getApiBypassHeaders() },
        body: JSON.stringify({
          parameters,
          knobs,
          detection_grids: defaults.detection_grids,
          silver_bullet: defaults.silver_bullet ?? null,
        }),
      }),
      fetch(`${API_BASE_URL}/api/compute`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...getApiBypassHeaders() },
        body: JSON.stringify({ parameters, time_range: [2017, 2100], initial_progress: 0.0 }),
      }),
    ]);
    if (!covRes.ok || !refRes.ok) return null;
    const result = await covRes.json();
    const ref = await refRes.json();
    if (!result || result.success === false) return null;

    return {
      defaults,
      result: {
        result,
        series: ref?.time_series ?? [],
        milestones: ref?.milestones ?? {},
      },
    };
  } catch (e) {
    console.error('fetchPlanAExplorerData failed (falling back to baked snapshots):', e);
    return null;
  }
}
