import { NextRequest, NextResponse } from 'next/server';
import path from 'node:path';
import { readFile } from 'node:fs/promises';

interface Simulation {
  id: string;
  label: string;
  forecaster: string;
  date: string;
  customIdentifier: string | null;
  dataPath: string;
  numSamples: number | null;
  modelVersion: string | null;
}

interface SimulationRegistry {
  version: number;
  defaultSimulation: string | null;
  simulations: Simulation[];
}

export interface TakeoffCdfDataResponse {
  milestones: string[];
  timePoints: number[];
  data: { [milestone: string]: { timeFromAC: number; cdf: number }[] };
}

export async function GET(request: NextRequest): Promise<NextResponse> {
  const simulationId = request.nextUrl.searchParams.get('simulationId');

  if (!simulationId) {
    return NextResponse.json({ error: 'simulationId parameter is required' }, { status: 400 });
  }

  try {
    // Load simulation registry
    const registryPath = path.join(process.cwd(), 'app/forecast/data/simulations/simulations.json');
    const registryRaw = await readFile(registryPath, 'utf8');
    const registry = JSON.parse(registryRaw) as SimulationRegistry;

    // Find the simulation
    const simulation = registry.simulations.find(s => s.id === simulationId);
    if (!simulation) {
      return NextResponse.json({ error: `Simulation not found: ${simulationId}` }, { status: 404 });
    }

    // Load unconditional time-until CDF data
    const cdfPath = path.join(
      process.cwd(),
      'app/forecast/data',
      simulation.dataPath,
      'ac_unconditional_time_until_cdf.csv'
    );

    const raw = await readFile(cdfPath, 'utf8');
    const lines = raw
      .split(/\r?\n/)
      .map(line => line.trim())
      .filter(line => line.length > 0);

    const [header, ...dataLines] = lines;
    const headers = header.split(',');
    const milestoneNames = headers.slice(1); // Skip "time_decimal_year"

    // Parse the CDF data
    const data: { [milestone: string]: { timeFromAC: number; cdf: number }[] } = {};
    const timePoints: number[] = [];

    for (const milestoneName of milestoneNames) {
      data[milestoneName] = [];
    }

    for (const line of dataLines) {
      const values = line.split(',');
      const timeFromAC = Number.parseFloat(values[0]);

      if (!Number.isFinite(timeFromAC)) continue;

      timePoints.push(timeFromAC);

      for (let i = 0; i < milestoneNames.length; i++) {
        const cdfValue = Number.parseFloat(values[i + 1]);

        if (Number.isFinite(cdfValue)) {
          data[milestoneNames[i]].push({
            timeFromAC,
            cdf: cdfValue,
          });
        }
      }
    }

    const response: TakeoffCdfDataResponse = {
      milestones: milestoneNames,
      timePoints,
      data,
    };

    return NextResponse.json(response);
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unknown error';
    return NextResponse.json({ error: `Failed to load takeoff CDF data: ${message}` }, { status: 500 });
  }
}
