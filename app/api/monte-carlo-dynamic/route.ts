import { generateDynamicMonteCarloBatch, convertSampledParametersToAPIFormat, convertParametersToAPIFormat } from '../../../utils/monteCarlo';
import { ParametersType } from '../../../constants/parameters';
import { getApiBaseUrl, getApiBypassHeaders } from '../../../lib/apiBase';

interface TimeSeriesPoint {
  year: number;
  progress: number;
  effectiveCompute: number;
  horizonLength: number;
  researchStock: number;
}


interface AllParameters {
  // Main parameters
  present_doubling_time: number;
  ac_time_horizon_minutes: number;
  doubling_difficulty_growth_factor: number;

  // CES Production Function Parameters
  rho_coding_labor: number;
  rho_experiment_capacity: number;
  alpha_experiment_capacity: number;

  // Software R&D Parameters
  r_software: number;
  coding_labor_normalization: number;
  experiment_compute_exponent: number;
  coding_labor_exponent: number;

  // Automation Parameters
  automation_fraction_at_coding_automation_anchor: number;
  swe_multiplier_at_present_day: number;

  // AI Research Taste Parameters
  ai_research_taste_at_coding_automation_anchor_sd: number;
  ai_research_taste_slope: number;
  median_to_top_taste_multiplier: number;
  top_percentile: number;

  // Progress Parameters
  progress_at_aa: number;

  // Horizon Parameters
  present_horizon: number;
  present_day: number;
  horizon_extrapolation_type: string;

  // Advanced CES Parameters
  inf_labor_asymptote: number;
  inf_compute_asymptote: number;
  labor_anchor_exp_cap: number;
  inv_compute_anchor_exp_cap: number;
}

const VISIBLE_CHART_START_YEAR = 2019;

async function generateTrajectory(params: ParametersType, timeRange: [number, number]) {
  // Talk to the right backend per environment (local dev / production / preview).
  const apiUrl = `${getApiBaseUrl()}/api/compute`;

  const response = await fetch(apiUrl, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...getApiBypassHeaders() },
    body: JSON.stringify({
      parameters: params,
      time_range: timeRange,
      initial_progress: 0.0
    })
  });

  if (!response.ok) {
    let errorBody = '';
    try {
      errorBody = await response.text();
    } catch {
      errorBody = '(failed to read error body)';
    }
    throw new Error(`API request failed: ${response.status} ${errorBody}`);
  }

  const data = await response.json();
  if (!data.success) {
    throw new Error(data.error || 'API request failed');
  }

  if (!Array.isArray(data.time_series)) {
    throw new Error('API response missing time_series');
  }

  return data.time_series.filter((point: TimeSeriesPoint) => point.year >= VISIBLE_CHART_START_YEAR);
}

export async function POST(request: Request) {
  const encoder = new TextEncoder();

  const stream = new ReadableStream({
    async start(controller) {
      try {
        const body = await request.json();
        const allParams: AllParameters = body.sliderParams;
        const timeRange: [number, number] = body.timeRange || [2012, 2035];
        
        if (!allParams || 
            typeof allParams.present_doubling_time !== 'number' ||
            typeof allParams.ac_time_horizon_minutes !== 'number' ||
            typeof allParams.doubling_difficulty_growth_factor !== 'number') {
          controller.enqueue(encoder.encode(
            JSON.stringify({
              type: 'error',
              error: 'Invalid parameters'
            }) + '\n'
          ));
          controller.close();
          return;
        }

        const startTime = Date.now();

        const numTrajectories = 10; // Reduced for faster updates
        const basePythonParams = convertParametersToAPIFormat({ ...(allParams as ParametersType) });
        const samples = generateDynamicMonteCarloBatch(allParams as ParametersType, numTrajectories, 42);
        
        // Create promises for individual trajectories and handle them as they complete
        const trajectoryPromises = samples.map(async (sample, index) => {
          const convertedSample = convertSampledParametersToAPIFormat({ ...sample });

          const params = {
            ...basePythonParams,
            ...convertedSample,
            horizon_extrapolation_type: convertedSample.horizon_extrapolation_type || "decaying doubling time",
            present_horizon: convertedSample.present_horizon || 15.0,
            present_day: convertedSample.present_day || 2025.25
          } as ParametersType;

          try {
            const timeSeries = await generateTrajectory(params, timeRange);
            
            const trajectoryData = timeSeries.map((point: TimeSeriesPoint) => ({
              year: point.year,
              horizonLength: NaN, // Will be filled by main trajectory
              horizonFormatted: '',
              effectiveCompute: NaN, // Will be filled by main trajectory
              [`trajectory_${index}`]: point.horizonLength,
              [`effective_compute_trajectory_${index}`]: point.effectiveCompute
            }));
            
            // Stream this trajectory immediately as it completes
            controller.enqueue(encoder.encode(
              JSON.stringify({
                type: 'trajectory',
                index,
                data: trajectoryData
              }) + '\n'
            ));
            
            return { index, success: true };
          } catch (error) {
            console.error(`Failed to generate dynamic trajectory ${index}:`, error);
            
            // Stream error for this specific trajectory
            controller.enqueue(encoder.encode(
              JSON.stringify({
                type: 'trajectory_error',
                index,
                error: error instanceof Error ? error.message : 'Unknown error'
              }) + '\n'
            ));
            
            return { index, success: false };
          }
        });

        // Wait for all trajectories to complete (or fail)
        const results = await Promise.allSettled(trajectoryPromises);
        const successCount = results.filter(result => 
          result.status === 'fulfilled' && result.value.success
        ).length;

        const generationTime = Date.now() - startTime;
        
        // Send completion signal
        controller.enqueue(encoder.encode(
          JSON.stringify({
            type: 'complete',
            count: successCount,
            total: numTrajectories,
            cached: false,
            generationTime
          }) + '\n'
        ));
        
        controller.close();
        
      } catch (error) {
        console.error('❌ Error streaming dynamic Monte Carlo trajectories:', error);
        controller.enqueue(encoder.encode(
          JSON.stringify({
            type: 'error',
            error: error instanceof Error ? error.message : 'Unknown error'
          }) + '\n'
        ));
        controller.close();
      }
    }
  });
  
  return new Response(stream, {
    headers: {
      'Content-Type': 'application/x-ndjson',
      'Cache-Control': 'no-cache',
      'X-Content-Type-Options': 'nosniff',
    }
  });
}

// GET method returns error since this endpoint requires POST with parameters
export async function GET() {
  return Response.json(
    { error: 'This endpoint requires POST with parameters' }, 
    { status: 405 }
  );
}
