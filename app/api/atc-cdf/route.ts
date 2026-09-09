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

// Per-milestone summary statistics for one AC-anchor mode
export interface ModeBreakoutMilestoneStats {
  achievementRate: number; // fraction in [0, 1]
  p10: number;
  p50: number;
  p90: number;
}

// CDF series + legend metadata for one AC-anchor mode.
// Denominator semantics (matching /forecast's breakout): each mode's CDF
// plateaus at that mode's OWN achievement rate; failed rollouts are excluded.
export interface ModeBreakoutModeCdfData {
  data: { [milestone: string]: { year: number; cdf: number }[] };
  statistics: { [milestone: string]: ModeBreakoutMilestoneStats };
  // Successful rollouts sampled with this mode (the CDF denominator)
  numRollouts: number;
  // All rollouts sampled with this mode, including failed ones
  numSampled: number;
  numFailed: number;
  // Realized mixture weight: share of ALL sampled rollouts (incl. failed)
  weightPct: number;
}

export interface CdfModeBreakout {
  modes: string[];
  perMode: { [mode: string]: ModeBreakoutModeCdfData };
}

export interface CdfDataResponse {
  milestones: string[];
  timePoints: number[];
  data: { [milestone: string]: { year: number; cdf: number }[] };
  // Per-AC-anchor-mode breakout of the model CDF. Only present for mixture
  // simulations that shipped milestone_pdfs_mode_breakout_*.csv; null otherwise.
  modeBreakout: CdfModeBreakout | null;
}

// Minimal CSV field splitter (handles double-quoted fields; enough for our exports)
function parseCsvLine(line: string): string[] {
  const fields: string[] = [];
  let current = '';
  let inQuotes = false;
  for (let i = 0; i < line.length; i++) {
    const char = line[i];
    if (char === '"') {
      inQuotes = !inQuotes;
    } else if (char === ',' && !inQuotes) {
      fields.push(current);
      current = '';
    } else {
      current += char;
    }
  }
  fields.push(current);
  return fields;
}

/**
 * Load the per-AC-anchor-mode breakout CDFs for a simulation dir, if present.
 * Returns null when the simulation has no breakout CSVs (older sims) or fewer
 * than two modes appear in them.
 */
async function loadModeBreakout(simDir: string): Promise<CdfModeBreakout | null> {
  const cdfPath = path.join(simDir, 'milestone_pdfs_mode_breakout_cdf.csv');
  const statsPath = path.join(simDir, 'milestone_pdfs_mode_breakout_statistics.csv');

  let cdfRaw: string;
  let statsRaw: string;
  try {
    [cdfRaw, statsRaw] = await Promise.all([
      readFile(cdfPath, 'utf8'),
      readFile(statsPath, 'utf8'),
    ]);
  } catch {
    // Simulation without breakout data
    return null;
  }

  const parseLines = (raw: string): string[] =>
    raw.split(/\r?\n/).map(line => line.trim()).filter(line => line.length > 0);

  const perMode: { [mode: string]: ModeBreakoutModeCdfData } = {};
  const modeOrder: string[] = [];
  const ensureMode = (mode: string): ModeBreakoutModeCdfData => {
    if (!perMode[mode]) {
      perMode[mode] = {
        data: {},
        statistics: {},
        numRollouts: 0,
        numSampled: 0,
        numFailed: 0,
        weightPct: 0,
      };
      modeOrder.push(mode);
    }
    return perMode[mode];
  };

  // Statistics CSV carries the denominators and mixture weights for the legend
  const statsLines = parseLines(statsRaw);
  if (statsLines.length < 2) return null;
  const statsHeaders = parseCsvLine(statsLines[0]);
  const sIdx = (name: string) => statsHeaders.indexOf(name);
  const sModeIdx = sIdx('ac_anchor_mode');
  const sMilestoneIdx = sIdx('milestone_name');
  const sTotalIdx = sIdx('total_rollouts');
  const sFailedIdx = sIdx('num_failed_rollouts');
  const sSampledIdx = sIdx('num_sampled');
  const sWeightIdx = sIdx('sampled_weight_pct');
  const sRateIdx = sIdx('achievement_rate_pct');
  const sP10Idx = sIdx('p10');
  const sP50Idx = sIdx('p50');
  const sP90Idx = sIdx('p90');
  if (sModeIdx < 0 || sMilestoneIdx < 0) return null;

  for (const line of statsLines.slice(1)) {
    const fields = parseCsvLine(line);
    const mode = fields[sModeIdx];
    const milestoneName = fields[sMilestoneIdx];
    if (!mode || !milestoneName) continue;
    const modeData = ensureMode(mode);
    modeData.numRollouts = Number.parseInt(fields[sTotalIdx], 10) || 0;
    modeData.numFailed = Number.parseInt(fields[sFailedIdx], 10) || 0;
    modeData.numSampled = Number.parseInt(fields[sSampledIdx], 10) || 0;
    modeData.weightPct = Number.parseFloat(fields[sWeightIdx]) || 0;
    modeData.statistics[milestoneName] = {
      achievementRate: Number.parseFloat(fields[sRateIdx]) / 100,
      p10: Number.parseFloat(fields[sP10Idx]),
      p50: Number.parseFloat(fields[sP50Idx]),
      p90: Number.parseFloat(fields[sP90Idx]),
    };
  }

  // CDF CSV is long format: ac_anchor_mode, time_decimal_year, <milestones...>
  const cdfLines = parseLines(cdfRaw);
  if (cdfLines.length < 2) return null;
  const cdfHeaders = cdfLines[0].split(',');
  const milestoneNames = cdfHeaders.slice(2);

  for (const line of cdfLines.slice(1)) {
    const values = line.split(',');
    const mode = values[0];
    const year = Number.parseFloat(values[1]);
    if (!mode || !Number.isFinite(year)) continue;
    const modeData = ensureMode(mode);
    for (let i = 0; i < milestoneNames.length; i++) {
      const cdfValue = Number.parseFloat(values[i + 2]);
      if (Number.isFinite(cdfValue)) {
        (modeData.data[milestoneNames[i]] ??= []).push({ year, cdf: cdfValue });
      }
    }
  }

  for (const mode of modeOrder) {
    for (const milestoneName of Object.keys(perMode[mode].data)) {
      perMode[mode].data[milestoneName].sort((a, b) => a.year - b.year);
    }
  }

  // A breakout with a single mode adds no information over the combined CDF
  if (modeOrder.length < 2) return null;

  return { modes: modeOrder, perMode };
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

    // Load CDF data
    const simDir = path.join(process.cwd(), 'app/forecast/data', simulation.dataPath);
    const cdfPath = path.join(simDir, 'milestone_pdfs_overlay_cdf.csv');

    const raw = await readFile(cdfPath, 'utf8');
    const lines = raw
      .split(/\r?\n/)
      .map(line => line.trim())
      .filter(line => line.length > 0);

    const [header, ...dataLines] = lines;
    const headers = header.split(',');
    const milestoneNames = headers.slice(1); // Skip "time_decimal_year"

    // Parse the CDF data
    const data: { [milestone: string]: { year: number; cdf: number }[] } = {};
    const timePoints: number[] = [];

    for (const milestoneName of milestoneNames) {
      data[milestoneName] = [];
    }

    for (const line of dataLines) {
      const values = line.split(',');
      const year = Number.parseFloat(values[0]);

      if (!Number.isFinite(year)) continue;

      timePoints.push(year);

      for (let i = 0; i < milestoneNames.length; i++) {
        const cdfValue = Number.parseFloat(values[i + 1]);

        if (Number.isFinite(cdfValue)) {
          data[milestoneNames[i]].push({
            year,
            cdf: cdfValue,
          });
        }
      }
    }

    const response: CdfDataResponse = {
      milestones: milestoneNames,
      timePoints,
      data,
      modeBreakout: await loadModeBreakout(simDir),
    };

    return NextResponse.json(response);
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unknown error';
    return NextResponse.json({ error: `Failed to load CDF data: ${message}` }, { status: 500 });
  }
}
