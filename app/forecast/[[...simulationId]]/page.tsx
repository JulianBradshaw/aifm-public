import path from "node:path";
import { readFile } from "node:fs/promises";
import { Suspense } from "react";
import { redirect } from "next/navigation";
import Link from "next/link";
import { cacheLife, cacheTag } from "next/cache";
import {
  trimMilestoneDistributions,
  trimTimingDistributions,
  trimAtcCdf,
  trimAtcTakeoffCdf,
  trimModeBreakout,
} from "@/utils/forecastPayloadTrim";
import yaml from "js-yaml";
import ReactMarkdown from "react-markdown";
import { MilestoneDistributionPoint, MilestoneDistributionData, MilestoneCdfPoint, ModeBreakoutData, ModeBreakoutModeData } from "@/components/MilestoneDistributionChart";
import { ConditionalTimingPoint, ConditionalTimingData, ConditionalCdfPoint } from "@/components/ConditionalMilestoneTimingChart";
import { ForecastChartsSection } from "@/components/ForecastChartsSection";
import { HeaderContent } from "@/components/HeaderContent";
import { SimulationSelector, Simulation } from "@/components/SimulationSelector";
import { SamplingDistributionsButton } from "@/components/SamplingDistributionsButton";
import type { SamplingConfig } from "@/types/samplingConfig";
import TableOfContents from "@/components/TableOfContents";

function parseCsvLine(line: string): string[] {
  const fields: string[] = [];
  let current = "";
  let inQuotes = false;

  for (let i = 0; i < line.length; i += 1) {
    const char = line[i];

    if (char === "\"") {
      inQuotes = !inQuotes;
      continue;
    }

    if (char === "," && !inQuotes) {
      fields.push(current.trim());
      current = "";
      continue;
    }

    current += char;
  }

  fields.push(current.trim());

  return fields.map(value => value.replace(/^"|"$/g, ""));
}

interface MilestoneStatistics {
  [key: string]: {
    achievementRate: number;
    mode: number;
    p10: number;
    p50: number;
    p90: number;
  };
}

// Interpolate CDF value at a specific year from CDF data points
function interpolateCdfAtYear(cdfPoints: { year: number; cdf: number }[], targetYear: number): number {
  if (cdfPoints.length === 0) return 0;

  // Sort by year
  const sorted = [...cdfPoints].sort((a, b) => a.year - b.year);

  // If target is before first point, return 0
  if (targetYear <= sorted[0].year) return sorted[0].cdf;

  // If target is after last point, return last CDF value
  if (targetYear >= sorted[sorted.length - 1].year) return sorted[sorted.length - 1].cdf;

  // Find surrounding points and interpolate
  for (let i = 0; i < sorted.length - 1; i++) {
    if (sorted[i].year <= targetYear && sorted[i + 1].year >= targetYear) {
      const t = (targetYear - sorted[i].year) / (sorted[i + 1].year - sorted[i].year);
      return sorted[i].cdf + t * (sorted[i + 1].cdf - sorted[i].cdf);
    }
  }

  return 0;
}

// Interpolate CDF value at a specific time (years from AC) from conditional CDF data
function interpolateCdfAtTime(cdfPoints: { timeFromAC: number; cdf: number }[], targetTime: number): number {
  if (cdfPoints.length === 0) return 0;

  // Sort by time
  const sorted = [...cdfPoints].sort((a, b) => a.timeFromAC - b.timeFromAC);

  // If target is before first point, return 0
  if (targetTime <= sorted[0].timeFromAC) return sorted[0].cdf;

  // If target is after last point, return last CDF value
  if (targetTime >= sorted[sorted.length - 1].timeFromAC) return sorted[sorted.length - 1].cdf;

  // Find surrounding points and interpolate
  for (let i = 0; i < sorted.length - 1; i++) {
    if (sorted[i].timeFromAC <= targetTime && sorted[i + 1].timeFromAC >= targetTime) {
      const t = (targetTime - sorted[i].timeFromAC) / (sorted[i + 1].timeFromAC - sorted[i].timeFromAC);
      return sorted[i].cdf + t * (sorted[i + 1].cdf - sorted[i].cdf);
    }
  }

  return 0;
}

export interface ArrivalProbabilities {
  milestones: {
    name: string;
    displayName: string;
    probabilities: { [year: string]: number };
  }[];
  years: number[];
}

export interface TakeoffProbabilities {
  conditions: {
    label: string;
    probabilities: { [metric: string]: number };
  }[];
  metrics: { key: string; label: string }[];
}

// ATC (all-things-considered) CDF data types
export interface AtcCdfPoint {
  year: number;
  cdf: number;
}

export interface AtcCdfData {
  [milestoneName: string]: AtcCdfPoint[];
}

// ATC Takeoff CDF data types (time from AC to milestone)
export interface AtcTakeoffCdfPoint {
  timeFromAC: number;
  cdf: number;
}

export interface AtcTakeoffCdfData {
  [milestoneName: string]: AtcTakeoffCdfPoint[];
}

// ATC statistics derived from CDF data
export interface AtcStatistics {
  [milestoneName: string]: {
    p10: number;
    p50: number;
    p90: number;
  };
}

// Comparison dataset for comparing multiple forecasts
export interface ComparisonDataset {
  simulationId: string;
  forecaster: string;
  label: string;
  distributionData: MilestoneDistributionData;
  unconditionalTimingData: ConditionalTimingData;
  atcCdfData: AtcCdfData | null;
  atcTakeoffCdfData: AtcTakeoffCdfData | null;
  atcStatistics: AtcStatistics | null;
}

// Compute percentiles from CDF data by finding years where CDF reaches target probabilities
function computeAtcStatistics(atcCdfData: AtcCdfData): AtcStatistics {
  const statistics: AtcStatistics = {};

  for (const milestoneName of Object.keys(atcCdfData)) {
    const cdfPoints = atcCdfData[milestoneName];
    if (cdfPoints.length === 0) continue;

    // Sort by year
    const sorted = [...cdfPoints].sort((a, b) => a.year - b.year);

    // Helper to interpolate year at a given CDF value
    const findYearAtCdf = (targetCdf: number): number => {
      // If target is below first CDF value, extrapolate or return first year
      if (targetCdf <= sorted[0].cdf) return sorted[0].year;

      // If target is above last CDF value, return last year
      if (targetCdf > sorted[sorted.length - 1].cdf) return sorted[sorted.length - 1].year;

      // Find surrounding points and interpolate
      for (let i = 0; i < sorted.length - 1; i++) {
        if (sorted[i].cdf <= targetCdf && sorted[i + 1].cdf >= targetCdf) {
          const cdfDiff = sorted[i + 1].cdf - sorted[i].cdf;
          if (cdfDiff === 0) return sorted[i].year;
          const t = (targetCdf - sorted[i].cdf) / cdfDiff;
          return sorted[i].year + t * (sorted[i + 1].year - sorted[i].year);
        }
      }

      return sorted[sorted.length - 1].year;
    };

    statistics[milestoneName] = {
      p10: findYearAtCdf(0.10),
      p50: findYearAtCdf(0.50),
      p90: findYearAtCdf(0.90),
    };
  }

  return statistics;
}

interface SimulationRegistry {
  version: number;
  defaultSimulation: string | null;
  simulations: Simulation[];
}

async function loadSimulationRegistry(): Promise<SimulationRegistry> {
  const registryPath = path.join(process.cwd(), "app/forecast/data/simulations/simulations.json");
  try {
    const raw = await readFile(registryPath, "utf8");
    return JSON.parse(raw) as SimulationRegistry;
  } catch {
    // Return empty registry if file doesn't exist
    return {
      version: 1,
      defaultSimulation: null,
      simulations: [],
    };
  }
}

interface SamplingConfigWithRaw {
  config: SamplingConfig;
  rawYaml: string;
}

async function loadSamplingConfig(basePath: string): Promise<SamplingConfigWithRaw | null> {
  const configPath = path.join(process.cwd(), basePath, "sampling_config.yaml");
  try {
    const rawYaml = await readFile(configPath, "utf8");
    const config = yaml.load(rawYaml) as SamplingConfig;
    return { config, rawYaml };
  } catch {
    // Return null if file doesn't exist (for older simulations)
    return null;
  }
}

async function loadMilestoneDistributions(basePath: string): Promise<MilestoneDistributionData> {
  // Load the overlay distributions file
  const filePath = path.join(process.cwd(), basePath, "milestone_pdfs_overlay_distributions.csv");
  const raw = await readFile(filePath, "utf8");

  const lines = raw
    .split(/\r?\n/)
    .map(line => line.trim())
    .filter(line => line.length > 0);

  const [header, ...dataLines] = lines;
  const headers = header.split(",");
  const milestoneNames = headers.slice(1); // Skip "time_decimal_year"

  // Load statistics
  const statsPath = path.join(process.cwd(), basePath, "milestone_pdfs_overlay_statistics.csv");
  const statsRaw = await readFile(statsPath, "utf8");

  const statsLines = statsRaw
    .split(/\r?\n/)
    .map(line => line.trim())
    .filter(line => line.length > 0);

  const [statsHeader, ...statsDataLines] = statsLines;
  const statsHeaders = parseCsvLine(statsHeader);

  const milestoneIndex = statsHeaders.indexOf("milestone_name");
  const achievementRateIndex = statsHeaders.indexOf("achievement_rate_pct");
  const modeIndex = statsHeaders.indexOf("mode");
  const p10Index = statsHeaders.indexOf("p10");
  const p50Index = statsHeaders.indexOf("p50");
  const p90Index = statsHeaders.indexOf("p90");

  const statistics: MilestoneStatistics = {};

  for (const line of statsDataLines) {
    const fields = parseCsvLine(line);
    const milestoneName = fields[milestoneIndex];

    if (milestoneName) {
      statistics[milestoneName] = {
        achievementRate: Number.parseFloat(fields[achievementRateIndex]) / 100,
        mode: Number.parseFloat(fields[modeIndex]),
        p10: Number.parseFloat(fields[p10Index]),
        p50: Number.parseFloat(fields[p50Index]),
        p90: Number.parseFloat(fields[p90Index]),
      };
    }
  }

  // Parse distribution data for each milestone
  const milestones: { [key: string]: MilestoneDistributionPoint[] } = {};

  for (const milestoneName of milestoneNames) {
    milestones[milestoneName] = [];
  }

  for (const line of dataLines) {
    const values = line.split(",");
    const year = Number.parseFloat(values[0]);

    if (!Number.isFinite(year)) continue;

    for (let i = 0; i < milestoneNames.length; i += 1) {
      const density = Number.parseFloat(values[i + 1]);

      if (Number.isFinite(density)) {
        milestones[milestoneNames[i]].push({
          year,
          probabilityDensity: density,
        });
      }
    }
  }

  // Load empirical CDF data if available
  const cdf: { [key: string]: MilestoneCdfPoint[] } = {};
  const cdfPath = path.join(process.cwd(), basePath, "milestone_pdfs_overlay_cdf.csv");
  try {
    const cdfRaw = await readFile(cdfPath, "utf8");
    const cdfLines = cdfRaw
      .split(/\r?\n/)
      .map(line => line.trim())
      .filter(line => line.length > 0);

    const [cdfHeader, ...cdfDataLines] = cdfLines;
    const cdfHeaders = cdfHeader.split(",");
    const cdfMilestoneNames = cdfHeaders.slice(1);

    for (const milestoneName of cdfMilestoneNames) {
      cdf[milestoneName] = [];
    }

    for (const line of cdfDataLines) {
      const values = line.split(",");
      const year = Number.parseFloat(values[0]);

      if (!Number.isFinite(year)) continue;

      for (let i = 0; i < cdfMilestoneNames.length; i += 1) {
        const cdfValue = Number.parseFloat(values[i + 1]);

        if (Number.isFinite(cdfValue)) {
          cdf[cdfMilestoneNames[i]].push({
            year,
            cdf: cdfValue,
          });
        }
      }
    }
  } catch {
    // CDF file doesn't exist - fallback to PDF-derived CDF
  }

  // Normalize each milestone distribution by its achievement rate
  // Convert to proper density units (per year) by accounting for bin width
  for (const milestoneName of milestoneNames) {
    const points = milestones[milestoneName];
    const stats = statistics[milestoneName];

    if (!stats) continue;

    const sorted = points.sort((a, b) => a.year - b.year);
    const totalDensity = sorted.reduce((acc, point) => acc + point.probabilityDensity, 0);

    // Calculate bin width from data spacing (assumes uniform spacing)
    const binWidth = sorted.length > 1
      ? (sorted[sorted.length - 1].year - sorted[0].year) / (sorted.length - 1)
      : 1;

    if (totalDensity > 0) {
      // Normalize so that integral (sum * binWidth) = achievementRate
      // This gives density in "per year" units
      milestones[milestoneName] = sorted.map(point => ({
        ...point,
        probabilityDensity: (point.probabilityDensity / totalDensity) * stats.achievementRate / binWidth,
      }));
    }
  }

  // Filter to only show specific milestones
  // Remove milestones from this array to omit them from the chart
  const milestonesToShow = [
    'AC',
    'AI2027-SC',
    'SAR-level-experiment-selection-skill',
    'SIAR-level-experiment-selection-skill',
    'TED-AI',
    'ASI',
  ];

  const filteredMilestones: { [key: string]: MilestoneDistributionPoint[] } = {};
  const filteredStatistics: MilestoneStatistics = {};
  const filteredCdf: { [key: string]: MilestoneCdfPoint[] } = {};

  for (const milestoneName of milestonesToShow) {
    if (milestones[milestoneName]) {
      filteredMilestones[milestoneName] = milestones[milestoneName];
    }
    if (statistics[milestoneName]) {
      filteredStatistics[milestoneName] = statistics[milestoneName];
    }
    if (cdf[milestoneName]) {
      filteredCdf[milestoneName] = cdf[milestoneName];
    }
  }

  return {
    milestones: filteredMilestones,
    statistics: filteredStatistics,
    cdf: Object.keys(filteredCdf).length > 0 ? filteredCdf : undefined,
  };
}

interface ConditionalTimingStatistics {
  [key: string]: {
    achievementRate: number;
    mode: number;
    p10: number;
    p50: number;
    p90: number;
  };
}

async function loadUnconditionalTimingDistributions(basePath: string): Promise<ConditionalTimingData> {
  // Load the AC unconditional timing distributions
  const filePath = path.join(process.cwd(), basePath, "ac_unconditional_time_until_distributions.csv");

  let raw: string;
  try {
    raw = await readFile(filePath, "utf8");
  } catch {
    // File doesn't exist - return empty data
    return {
      milestones: {},
      statistics: {},
      conditionDescription: "Time until milestone achievement, unconditional (all trajectories achieving AC)",
    };
  }

  const lines = raw
    .split(/\r?\n/)
    .map(line => line.trim())
    .filter(line => line.length > 0);

  const [header, ...dataLines] = lines;
  const headers = header.split(",");
  const milestoneNames = headers.slice(1); // Skip "time_decimal_year"

  // Load statistics
  const statsPath = path.join(process.cwd(), basePath, "ac_unconditional_time_until_statistics.csv");
  let statsRaw: string;
  try {
    statsRaw = await readFile(statsPath, "utf8");
  } catch {
    // Stats file doesn't exist - return data without statistics
    statsRaw = "";
  }

  const statsLines = statsRaw
    .split(/\r?\n/)
    .map(line => line.trim())
    .filter(line => line.length > 0);

  const [statsHeader, ...statsDataLines] = statsLines;
  const statsHeaders = parseCsvLine(statsHeader);

  const milestoneIndex = statsHeaders.indexOf("milestone_name");
  const achievementRateIndex = statsHeaders.indexOf("achievement_rate_pct");
  const modeIndex = statsHeaders.indexOf("mode");
  const p10Index = statsHeaders.indexOf("p10");
  const p50Index = statsHeaders.indexOf("p50");
  const p90Index = statsHeaders.indexOf("p90");

  const statistics: ConditionalTimingStatistics = {};

  for (const line of statsDataLines) {
    const fields = parseCsvLine(line);
    const milestoneName = fields[milestoneIndex];

    if (milestoneName) {
      statistics[milestoneName] = {
        achievementRate: Number.parseFloat(fields[achievementRateIndex]) / 100,
        mode: Number.parseFloat(fields[modeIndex]),
        p10: Number.parseFloat(fields[p10Index]),
        p50: Number.parseFloat(fields[p50Index]),
        p90: Number.parseFloat(fields[p90Index]),
      };
    }
  }

  // Parse distribution data for each milestone
  const milestones: { [key: string]: ConditionalTimingPoint[] } = {};

  for (const milestoneName of milestoneNames) {
    milestones[milestoneName] = [];
  }

  for (const line of dataLines) {
    const values = line.split(",");
    const timeFromAC = Number.parseFloat(values[0]);

    if (!Number.isFinite(timeFromAC)) continue;

    for (let i = 0; i < milestoneNames.length; i += 1) {
      const density = Number.parseFloat(values[i + 1]);

      if (Number.isFinite(density) && density > 0) {
        milestones[milestoneNames[i]].push({
          timeFromAC,
          probabilityDensity: density,
        });
      }
    }
  }

  // Load empirical CDF data if available
  const cdf: { [key: string]: ConditionalCdfPoint[] } = {};
  const cdfPath = path.join(process.cwd(), basePath, "ac_unconditional_time_until_cdf.csv");
  try {
    const cdfRaw = await readFile(cdfPath, "utf8");
    const cdfLines = cdfRaw
      .split(/\r?\n/)
      .map(line => line.trim())
      .filter(line => line.length > 0);

    const [cdfHeader, ...cdfDataLines] = cdfLines;
    const cdfHeaders = cdfHeader.split(",");
    const cdfMilestoneNames = cdfHeaders.slice(1);

    for (const milestoneName of cdfMilestoneNames) {
      cdf[milestoneName] = [];
    }

    for (const line of cdfDataLines) {
      const values = line.split(",");
      const timeFromAC = Number.parseFloat(values[0]);

      if (!Number.isFinite(timeFromAC)) continue;

      for (let i = 0; i < cdfMilestoneNames.length; i += 1) {
        const cdfValue = Number.parseFloat(values[i + 1]);

        if (Number.isFinite(cdfValue)) {
          cdf[cdfMilestoneNames[i]].push({
            timeFromAC,
            cdf: cdfValue,
          });
        }
      }
    }
  } catch {
    // CDF file doesn't exist - fallback to PDF-derived CDF
  }

  // Filter to only show specific milestones
  const milestonesToShow = [
    'SAR-level-experiment-selection-skill',
    'SIAR-level-experiment-selection-skill',
    'TED-AI',
    'ASI',
  ];

  const filteredMilestones: { [key: string]: ConditionalTimingPoint[] } = {};
  const filteredStatistics: ConditionalTimingStatistics = {};
  const filteredCdf: { [key: string]: ConditionalCdfPoint[] } = {};

  for (const milestoneName of milestonesToShow) {
    if (milestones[milestoneName]) {
      filteredMilestones[milestoneName] = milestones[milestoneName];
    }
    if (statistics[milestoneName]) {
      filteredStatistics[milestoneName] = statistics[milestoneName];
    }
    if (cdf[milestoneName]) {
      filteredCdf[milestoneName] = cdf[milestoneName];
    }
  }

  return {
    milestones: filteredMilestones,
    statistics: filteredStatistics,
    conditionDescription: "Time until milestone achievement, unconditional (all trajectories achieving AC)",
    cdf: Object.keys(filteredCdf).length > 0 ? filteredCdf : undefined,
  };
}

// Milestones shown on the timeline chart (mirrors loadMilestoneDistributions)
const BREAKOUT_MILESTONES_TO_SHOW = [
  'AC',
  'AI2027-SC',
  'SAR-level-experiment-selection-skill',
  'SIAR-level-experiment-selection-skill',
  'TED-AI',
  'ASI',
];

/**
 * Load per-AC-anchor-mode breakout data, if this simulation has it.
 *
 * The CSVs are long-format with an `ac_anchor_mode` first column. Denominator
 * semantics: each mode's CDF plateaus at that mode's OWN achievement rate
 * (achieved / successful rollouts sampled with that mode; failed rollouts are
 * excluded, matching the combined data). PDFs are renormalized the same way
 * as the combined PDFs: integral = the mode's own achievement rate.
 */
async function loadModeBreakoutData(basePath: string): Promise<ModeBreakoutData | null> {
  const distPath = path.join(process.cwd(), basePath, "milestone_pdfs_mode_breakout_distributions.csv");
  const statsPath = path.join(process.cwd(), basePath, "milestone_pdfs_mode_breakout_statistics.csv");
  const cdfPath = path.join(process.cwd(), basePath, "milestone_pdfs_mode_breakout_cdf.csv");

  let distRaw: string;
  let statsRaw: string;
  let cdfRaw: string;
  try {
    [distRaw, statsRaw, cdfRaw] = await Promise.all([
      readFile(distPath, "utf8"),
      readFile(statsPath, "utf8"),
      readFile(cdfPath, "utf8"),
    ]);
  } catch {
    // Simulation without breakout data
    return null;
  }

  const parseLines = (raw: string): string[] =>
    raw.split(/\r?\n/).map(line => line.trim()).filter(line => line.length > 0);

  const perMode: { [mode: string]: ModeBreakoutModeData } = {};
  const modeOrder: string[] = [];

  const ensureMode = (mode: string): ModeBreakoutModeData => {
    if (!perMode[mode]) {
      perMode[mode] = {
        milestones: {},
        cdf: {},
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

  // Parse statistics first (carries the denominators and mixture weights)
  const statsLines = parseLines(statsRaw);
  if (statsLines.length < 2) return null;
  const statsHeaders = parseCsvLine(statsLines[0]);
  const sIdx = (name: string) => statsHeaders.indexOf(name);
  const sModeIdx = sIdx("ac_anchor_mode");
  const sMilestoneIdx = sIdx("milestone_name");
  const sTotalIdx = sIdx("total_rollouts");
  const sFailedIdx = sIdx("num_failed_rollouts");
  const sSampledIdx = sIdx("num_sampled");
  const sWeightIdx = sIdx("sampled_weight_pct");
  const sRateIdx = sIdx("achievement_rate_pct");
  const sModeYearIdx = sIdx("mode");
  const sP10Idx = sIdx("p10");
  const sP50Idx = sIdx("p50");
  const sP90Idx = sIdx("p90");

  for (const line of statsLines.slice(1)) {
    const fields = parseCsvLine(line);
    const mode = fields[sModeIdx];
    const milestoneName = fields[sMilestoneIdx];
    if (!mode || !milestoneName) continue;
    if (!BREAKOUT_MILESTONES_TO_SHOW.includes(milestoneName)) continue;

    const modeData = ensureMode(mode);
    modeData.numRollouts = Number.parseInt(fields[sTotalIdx], 10) || 0;
    modeData.numFailed = Number.parseInt(fields[sFailedIdx], 10) || 0;
    modeData.numSampled = Number.parseInt(fields[sSampledIdx], 10) || 0;
    modeData.weightPct = Number.parseFloat(fields[sWeightIdx]) || 0;
    modeData.statistics[milestoneName] = {
      achievementRate: Number.parseFloat(fields[sRateIdx]) / 100,
      mode: Number.parseFloat(fields[sModeYearIdx]),
      p10: Number.parseFloat(fields[sP10Idx]),
      p50: Number.parseFloat(fields[sP50Idx]),
      p90: Number.parseFloat(fields[sP90Idx]),
    };
  }

  // Parse a long-format series CSV (distributions or CDF) into per-mode maps
  const parseSeriesCsv = (
    raw: string,
    assign: (mode: string, milestoneName: string, year: number, value: number) => void,
  ) => {
    const lines = parseLines(raw);
    if (lines.length < 2) return;
    const headers = lines[0].split(",");
    // headers: ac_anchor_mode, time_decimal_year, <milestones...>
    const milestoneNames = headers.slice(2);
    for (const line of lines.slice(1)) {
      const values = line.split(",");
      const mode = values[0];
      const year = Number.parseFloat(values[1]);
      if (!mode || !Number.isFinite(year)) continue;
      for (let i = 0; i < milestoneNames.length; i += 1) {
        const milestoneName = milestoneNames[i];
        if (!BREAKOUT_MILESTONES_TO_SHOW.includes(milestoneName)) continue;
        const value = Number.parseFloat(values[i + 2]);
        if (Number.isFinite(value)) {
          assign(mode, milestoneName, year, value);
        }
      }
    }
  };

  parseSeriesCsv(distRaw, (mode, milestoneName, year, value) => {
    const modeData = ensureMode(mode);
    (modeData.milestones[milestoneName] ??= []).push({ year, probabilityDensity: value });
  });

  parseSeriesCsv(cdfRaw, (mode, milestoneName, year, value) => {
    const modeData = ensureMode(mode);
    (modeData.cdf[milestoneName] ??= []).push({ year, cdf: value });
  });

  // Normalize each mode's PDFs by that mode's own achievement rate,
  // exactly mirroring the combined loadMilestoneDistributions normalization.
  for (const mode of modeOrder) {
    const modeData = perMode[mode];
    for (const milestoneName of Object.keys(modeData.milestones)) {
      const stats = modeData.statistics[milestoneName];
      if (!stats) continue;

      const sorted = modeData.milestones[milestoneName].sort((a, b) => a.year - b.year);
      const totalDensity = sorted.reduce((acc, point) => acc + point.probabilityDensity, 0);
      const binWidth = sorted.length > 1
        ? (sorted[sorted.length - 1].year - sorted[0].year) / (sorted.length - 1)
        : 1;

      if (totalDensity > 0) {
        modeData.milestones[milestoneName] = sorted.map(point => ({
          ...point,
          probabilityDensity: (point.probabilityDensity / totalDensity) * stats.achievementRate / binWidth,
        }));
      }
    }
    for (const milestoneName of Object.keys(modeData.cdf)) {
      modeData.cdf[milestoneName].sort((a, b) => a.year - b.year);
    }
  }

  if (modeOrder.length < 2) return null;

  return { modes: modeOrder, perMode };
}

async function loadAtcCdfData(basePath: string): Promise<AtcCdfData | null> {
  const filePath = path.join(process.cwd(), basePath, "atc_cdf.csv");

  try {
    const raw = await readFile(filePath, "utf8");

    const lines = raw
      .split(/\r?\n/)
      .map(line => line.trim())
      .filter(line => line.length > 0);

    if (lines.length < 2) return null;

    const [header, ...dataLines] = lines;
    const headers = header.split(",");
    const milestoneNames = headers.slice(1); // Skip "time_decimal_year"

    const atcCdf: AtcCdfData = {};
    for (const milestoneName of milestoneNames) {
      atcCdf[milestoneName] = [];
    }

    for (const line of dataLines) {
      const values = line.split(",");
      const year = Number.parseFloat(values[0]);

      if (!Number.isFinite(year)) continue;

      for (let i = 0; i < milestoneNames.length; i += 1) {
        const cdfValue = Number.parseFloat(values[i + 1]);

        if (Number.isFinite(cdfValue)) {
          atcCdf[milestoneNames[i]].push({
            year,
            cdf: cdfValue,
          });
        }
      }
    }

    // Filter to only show milestones that are displayed in the chart
    const milestonesToShow = [
      'AC',
      'SAR-level-experiment-selection-skill',
      'AI2027-SC',
      'SIAR-level-experiment-selection-skill',
      'TED-AI',
      'ASI'
    ];

    const filteredAtcCdf: AtcCdfData = {};
    for (const milestoneName of milestonesToShow) {
      if (atcCdf[milestoneName] && atcCdf[milestoneName].length > 0) {
        filteredAtcCdf[milestoneName] = atcCdf[milestoneName];
      }
    }

    return Object.keys(filteredAtcCdf).length > 0 ? filteredAtcCdf : null;
  } catch {
    // File doesn't exist - return null
    return null;
  }
}

async function loadTakeoffNotes(basePath: string): Promise<string | null> {
  const notesPath = path.join(process.cwd(), basePath, "takeoff_notes.html");

  try {
    const raw = await readFile(notesPath, "utf8");
    return raw.trim() || null;
  } catch {
    // File doesn't exist - return null
    return null;
  }
}

async function loadTimelinesNotes(basePath: string): Promise<string | null> {
  const notesPath = path.join(process.cwd(), basePath, "timelines_notes.html");

  try {
    const raw = await readFile(notesPath, "utf8");
    return raw.trim() || null;
  } catch {
    // File doesn't exist - return null
    return null;
  }
}

// Helper function to load all data for a single simulation
async function loadSimulationData(simulation: Simulation): Promise<ComparisonDataset> {
  const dataBasePath = `app/forecast/data/${simulation.dataPath}`;

  const [distributionData, unconditionalTimingData, atcCdfData, atcTakeoffCdfData] = await Promise.all([
    loadMilestoneDistributions(dataBasePath),
    loadUnconditionalTimingDistributions(dataBasePath),
    loadAtcCdfData(dataBasePath),
    loadAtcTakeoffCdfData(dataBasePath),
  ]);

  // Compute ATC statistics for this simulation
  const atcStatistics = atcCdfData ? computeAtcStatistics(atcCdfData) : null;

  // Trim only what crosses to the client, and only once the statistics above
  // have been derived from the full-resolution series
  return {
    simulationId: simulation.id,
    forecaster: simulation.forecaster,
    label: simulation.label,
    distributionData: trimMilestoneDistributions(distributionData),
    unconditionalTimingData: trimTimingDistributions(unconditionalTimingData),
    atcCdfData: trimAtcCdf(atcCdfData),
    atcTakeoffCdfData: trimAtcTakeoffCdf(atcTakeoffCdfData),
    atcStatistics,
  };
}

async function loadAtcTakeoffCdfData(basePath: string): Promise<AtcTakeoffCdfData | null> {
  const filePath = path.join(process.cwd(), basePath, "atc_takeoff_cdf.csv");

  try {
    const raw = await readFile(filePath, "utf8");

    const lines = raw
      .split(/\r?\n/)
      .map(line => line.trim())
      .filter(line => line.length > 0);

    if (lines.length < 2) return null;

    const [header, ...dataLines] = lines;
    const headers = header.split(",");
    const milestoneNames = headers.slice(1); // Skip "time_decimal_year"

    const atcCdf: AtcTakeoffCdfData = {};
    for (const milestoneName of milestoneNames) {
      atcCdf[milestoneName] = [];
    }

    for (const line of dataLines) {
      const values = line.split(",");
      const timeFromAC = Number.parseFloat(values[0]);

      if (!Number.isFinite(timeFromAC)) continue;

      for (let i = 0; i < milestoneNames.length; i += 1) {
        const cdfValue = Number.parseFloat(values[i + 1]);

        if (Number.isFinite(cdfValue)) {
          atcCdf[milestoneNames[i]].push({
            timeFromAC,
            cdf: cdfValue,
          });
        }
      }
    }

    return Object.keys(atcCdf).length > 0 ? atcCdf : null;
  } catch {
    // File doesn't exist - return null
    return null;
  }
}

// Cached content component that loads and displays forecast data
async function CachedForecastContent({
  simulationId,
  comparisonIds,
  registry,
}: {
  simulationId: string | null;
  comparisonIds: string[];
  registry: SimulationRegistry;
}) {
  'use cache';
  cacheLife('hours');
  cacheTag(`forecast-${simulationId ?? 'default'}-compare-${comparisonIds.join(',')}`);

  // Determine which simulation to use
  let currentSimulation: Simulation | null = null;
  let dataBasePath: string;

  if (registry.simulations.length > 0) {
    // Find requested simulation or use default
    if (simulationId) {
      currentSimulation = registry.simulations.find(s => s.id === simulationId) ?? null;
    }
    if (!currentSimulation && registry.defaultSimulation) {
      currentSimulation = registry.simulations.find(s => s.id === registry.defaultSimulation) ?? null;
    }
    if (!currentSimulation) {
      currentSimulation = registry.simulations[0];
    }
  }

  // Determine data path
  if (currentSimulation) {
    dataBasePath = `app/forecast/data/${currentSimulation.dataPath}`;
  } else {
    // Fallback to legacy flat data structure for backward compatibility
    dataBasePath = "app/forecast/data";
  }

  const distributionData = await loadMilestoneDistributions(dataBasePath);
  const unconditionalTimingData = await loadUnconditionalTimingDistributions(dataBasePath);
  const samplingConfigData = await loadSamplingConfig(dataBasePath);
  const atcCdfData = await loadAtcCdfData(dataBasePath);
  const atcTakeoffCdfData = await loadAtcTakeoffCdfData(dataBasePath);
  const takeoffNotes = await loadTakeoffNotes(dataBasePath);
  const timelinesNotes = await loadTimelinesNotes(dataBasePath);
  const modeBreakoutData = await loadModeBreakoutData(dataBasePath);

  // Load comparison simulations in parallel
  const comparisonSimulations = comparisonIds
    .map(id => registry.simulations.find(s => s.id === id))
    .filter((s): s is Simulation => s !== undefined);

  const comparisonDatasets = await Promise.all(
    comparisonSimulations.map(sim => loadSimulationData(sim))
  );

  // Compute ATC statistics from CDF data
  const atcStatistics = atcCdfData ? computeAtcStatistics(atcCdfData) : null;

  // Calculate y-domain from unconditional timing data using scaled densities
  let maxDensity = 0;

  if (unconditionalTimingData) {
    for (const milestoneName of Object.keys(unconditionalTimingData.milestones)) {
      const stats = unconditionalTimingData.statistics[milestoneName];
      const points = [...unconditionalTimingData.milestones[milestoneName]].sort((a, b) => a.timeFromAC - b.timeFromAC);

      if (points.length === 0) continue;

      const totalDensity = points.reduce((sum, point) => sum + point.probabilityDensity, 0);
      if (!Number.isFinite(totalDensity) || totalDensity <= 0) continue;

      // Find the empirical median from the raw PDF
      let running = 0;
      let empiricalMedianTime = points[0].timeFromAC;
      for (const point of points) {
        running += point.probabilityDensity / totalDensity;
        if (running >= 0.5) {
          empiricalMedianTime = point.timeFromAC;
          break;
        }
      }

      // Calculate scaling factor to shift the median
      const targetMedian = stats?.p50;
      const hasMedian = typeof targetMedian === 'number' && Number.isFinite(targetMedian);

      if (!hasMedian || empiricalMedianTime === 0) {
        // No scaling, just check max normalized density
        for (const point of points) {
          const density = point.probabilityDensity / totalDensity;
          if (Number.isFinite(density) && density > maxDensity) {
            maxDensity = density;
          }
        }
        continue;
      }

      // Scale the time axis so empirical median becomes target median
      const timeScaleFactor = targetMedian / empiricalMedianTime;

      // Check max scaled density
      for (const point of points) {
        const scaledDensity = (point.probabilityDensity / totalDensity) / timeScaleFactor;
        if (Number.isFinite(scaledDensity) && scaledDensity > maxDensity) {
          maxDensity = scaledDensity;
        }
      }
    }
  }

  const headroom = maxDensity === 0 ? 0.1 : maxDensity * 0.1;
  const sharedYDomain: [number, number] = [0, maxDensity + headroom];

  // Compute arrival probabilities from CDF data
  const arrivalYears = [2027, 2030, 2035];
  const arrivalMilestones = [
    { name: 'AC', displayName: 'AC (Automated Coder)' },
    { name: 'SAR-level-experiment-selection-skill', displayName: 'SAR (Superhuman AI Researcher)' },
  ];

  const arrivalProbabilities: ArrivalProbabilities = {
    milestones: arrivalMilestones.map(milestone => ({
      name: milestone.name,
      displayName: milestone.displayName,
      probabilities: Object.fromEntries(
        arrivalYears.map(year => {
          const cdfPoints = distributionData.cdf?.[milestone.name] ?? [];
          // Use end of year (Dec 31) as target
          const prob = interpolateCdfAtYear(cdfPoints, year + 1);
          return [year.toString(), prob];
        })
      ),
    })),
    years: arrivalYears,
  };

  // Compute takeoff probabilities from unconditional timing CDF data
  const takeoffMetrics = [
    { key: 'TED-AI', label: 'AC\u2192TED-AI \u22649mo', threshold: 0.75 },
    { key: 'ASI', label: 'AC\u2192ASI \u22641yr', threshold: 1.0 },
  ];

  // Compute unconditional takeoff probabilities
  const unconditionalCondition = {
    label: 'Unconditional',
    probabilities: Object.fromEntries(
      takeoffMetrics.map(metric => {
        const cdfPoints = unconditionalTimingData.cdf?.[metric.key] ?? [];
        const prob = interpolateCdfAtTime(cdfPoints, metric.threshold);
        return [metric.key, prob];
      })
    ),
  };

  const takeoffProbabilities: TakeoffProbabilities = {
    conditions: [unconditionalCondition],
    metrics: takeoffMetrics.map(m => ({ key: m.key, label: m.label })),
  };

  return (
    <div className="grid h-screen w-full grid-cols-[minmax(0,1fr)_auto] grid-rows-[minmax(0,1fr)] gap-0">
      <div className="relative col-start-1 row-start-1 flex min-h-0 flex-col">
{/* Temporarily hidden
        <TableOfContents
          contentSelector="#forecast-content"
          headingSelector="h2, h3"
          minLevel={2}
          maxLevel={3}
          topOffsetPx={120}
        />
*/}
        <div className="flex min-h-0 flex-col overflow-y-auto px-4 sm:px-6 pb-10">
          <HeaderContent variant="inline" className="pt-6 pb-4" />
          <main id="forecast-content" className="mt-6 sm:mt-10 mx-auto max-w-6xl pb-16">
            {/* Page Title and Introduction */}
            <section className="space-y-6 mb-12">
              <h1 className="text-3xl font-et-book font-bold text-primary">Forecasts</h1>
              <p className="prose">
                This page shows our timelines and takeoff forecasts. We are highly uncertain about this, and have expressed our uncertainty as a probability distribution over the possible times when each milestone might be reached. We show the raw result of a Monte Carlo simulation of our model, as well as our subjective all-things-considered probability distributions. We plan to keep this page up to date as our predictions change.
              </p>
              <div>
                <details>
                  <summary>
                    What do we mean by &quot;all-things-considered&quot;?
                  </summary>
                  <p>
                    Though we view the model&apos;s outputs as an important source of evidence about what future AI progress might look like, we don&apos;t blindly trust it. Our all-things-considered views are informed by looking at the results of the model but then making adjustments based on intuition and which factors the model doesn&apos;t include.
                  </p>
                </details>
                <details>
                  <summary>
                    How have our forecasts changed since publishing the AI Futures Model?
                  </summary>
                  <p>
                    The forecast dropdown below shows the history of how our views have changed since publication. Here is a summary of the changes:
                  </p>
                  <ul className="list-disc pl-6 mt-4 space-y-2">
                    <li>
                      <strong>2025 Dec 31: Fix infrequent bug with determining when TED-AI and ASI are achieved.</strong> Fixed an issue with the thresholds for TED-AI and ASI in the case where research taste at AC is better than the best human, 
                      i.e. &gt;3.09 SDs. This affects a low percentage of our Monte Carlo simulations: 13% of Eli&apos;s, and 9% of Daniel&apos;s. 
                      This very slightly increases takeoff speeds: for example it increases P(AC-&gt;ASI &lt; 1 year) from 26.3 to 27.1% for Eli, and from 36.7% to 37.1% for Daniel. 
                      It increases P(AC-&gt;ASI &lt; 10 years) from 58.3% to 59.3% for Eli, and from 71.5% to 71.8% for Daniel.
                    </li>
                    <li>
                      <strong>2026 Jan 26: Fix minor bug in the model code, add all-things-considered forecasts for more quantities, minor updates to Eli&apos;s AC timelines and AC to ASI takeoff speeds.</strong> Fixed a numerical underflow causing incorrect coding labor calculation for certain values of the “Coding Labor Parallelization Penalty (λ)” and “Coding Automation Efficiency Improvement Factor (η)” parameters. This was formerly affecting approximately 3% of our Monte Carlo rollouts, and fixing the issue had a negligible effect on the outcome distributions. Daniel and Eli also added all-things-considered forecasts for more quantities: both added forecasts for SC, TED-AI, and ASI.
                      Daniel also added a forecast for SAR. Eli also added a forecast for the time from AC to TED-AI. Eli also updated his all-things-considered forecasts for AC timelines and AC to ASI takeoff.
                      He made his AC arrival median Mar 2032 instead of Jul 2032 and increased the probability of AC arrival in 2026 from 4% to 6%.
                      These changes were due to paying a bit more respect to an Anthropic-style worldview in which we&apos;re close to AC, in part informed by Claude Code&apos;s impressiveness.
                      He also increased the chance of fast takeoffs from AC to ASI, giving 25% to takeoff in &lt;0.5 years as opposed to 18% before. But he also increased uncertainty
                      at the upper end, decreasing the chance of takeoff in &lt;10 years from 85% to 83%. These changes were due to reflecting more that perhaps the distribution
                      should be more spread out than the model&apos;s due to outside-of-model factors.
                    </li>
                    <li>
                      <strong>2026 Apr 02: Incorporate Opus 4.6 and time horizon v1.1, update present doubling time to be faster, and a few other changes.</strong> 
                      Update present day and present horizon to match Opus 4.6&apos;s release date and time horizon according to METR&apos;s <a href="https://metr.org/blog/2026-1-29-time-horizon-1-1/">version 1.1</a> of their task suite.
                      Update the median estimate for the present day parallel coding labor multiplier from 1.8 to 2 due to the passage of time.
                      Lower Daniel&apos;s estimate from 5.5 to 4 months, and lower Eli&apos;s estimate from 5.5 to 4.5 months. These adjustments are due to the v1.1 data seeming to have a faster trend, Opus 4.6 continuing the recent fast trend, and further analysis of the time horizon data.
                      Lower Daniel&apos;s median estimate for the AC time horizon requirement from 3 years to 1 year due to the real-world impressiveness of recent coding agents.
                      Lower Daniel&apos;s estimate of the gap between SAR and TED-AI to a median of 1 median-to-top human jumps, deviating from Eli&apos;s 1.5.
                      Increase Daniel&apos;s maximum research taste median to 16 median-to-top human jumps above the best human. We previously unintentionally had it set to 8, the same as Eli&apos;s estimate.
                    </li>
                    <li>
                      <strong>2026 Aug 16: Add coding uplift and revenue as alternative methods for anchoring AC, explicitly simulate training runs, add Brendan&apos;s forecast, and update parameters.</strong>{' '}
                      In addition to METR time horizons, the effective compute required for AC can now be anchored by extrapolating coding uplift (how much of a speedup AIs provide to software engineers at AI companies) or AI company revenue.
                      Each forecaster assigns weights to the three methods and the model aggregates them into an overall forecast; you can explore the individual methods via the dropdown on the second graph.
                      Encouragingly, all three methods predict quite similar AC arrival dates.
                      The model also now explicitly simulates the training run of the leading AI model, accounting for the fact that training is needed to apply software improvements; this reduces the chance of very fast takeoffs.
                      We re-graded AI 2027&apos;s predictions: reality seems to be going roughly 70&ndash;90% as fast as it depicted.
                      Brendan added his first forecast, with an AC median of Jul 2028.
                      With these changes, Daniel&apos;s model median for AC is now Dec 2027 (down from Feb 2028) and Eli&apos;s is Mar 2029 (down from Jul 2029).
                      For details, see our <a href="https://blog.aifutures.org/i/211256047">blog post</a>.
                    </li>
                  </ul>
                </details>
              </div>
              {/* Simulation Selector and Controls */}
              <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
                <div className="flex flex-col sm:flex-row sm:items-center gap-3 sm:gap-4">
                  {registry.simulations.length > 1 && currentSimulation && (
                    <SimulationSelector
                      simulations={registry.simulations}
                      currentSimulationId={currentSimulation.id}
                    />
                  )}
                  {samplingConfigData && currentSimulation && (
                    <SamplingDistributionsButton
                      config={samplingConfigData.config}
                      rawYaml={samplingConfigData.rawYaml}
                      simulationId={currentSimulation.id}
                    />
                  )}
                </div>
              </div>
            </section>
            <ForecastChartsSection
              distributionData={trimMilestoneDistributions(distributionData)}
              unconditionalTimingData={trimTimingDistributions(unconditionalTimingData)}
              sharedYDomain={sharedYDomain}
              numSamples={currentSimulation?.numSamples ?? null}
              arrivalProbabilities={arrivalProbabilities}
              takeoffProbabilities={takeoffProbabilities}
              atcCdfData={trimAtcCdf(atcCdfData)}
              atcStatistics={atcStatistics}
              atcTakeoffCdfData={trimAtcTakeoffCdf(atcTakeoffCdfData)}
              forecasterName={currentSimulation?.forecaster}
              forecasterDate={currentSimulation?.date}
              timelinesNotes={timelinesNotes}
              comparisonDatasets={comparisonDatasets}
              simulations={registry.simulations}
              currentSimulationId={currentSimulation?.id ?? ''}
              modeBreakoutData={trimModeBreakout(modeBreakoutData)}
            />
            {takeoffNotes && currentSimulation && (
              <details className="mt-8">
                <summary>
                  {currentSimulation.forecaster}&apos;s notes on their all-things-considered forecast
                </summary>
                <div className="mt-4 space-y-4 [&_ul]:list-disc [&_ul]:pl-6 [&_ul]:space-y-4 [&_a]:text-blue-600 [&_a:hover]:text-blue-800 [&_a]:underline" dangerouslySetInnerHTML={{ __html: takeoffNotes }} />
              </details>
            )}
            {/* Link to results analysis */}
            <p className="prose mt-12">
              In our <Link href="/analysis" className="text-blue-600 hover:text-blue-800 underline">results analysis</Link>, we analyze which parameters are most important for the above forecasts. We also examine the correlation in our model between short timelines and fast takeoffs.
            </p>
          </main>
        </div>
      </div>
    </div>
  );
}

// Content component that resolves params and passes to cached component
async function ForecastPageContent({
  params,
  searchParams,
}: {
  params: Promise<{ simulationId?: string[] }>;
  searchParams: Promise<{ csim?: string; tkcsim?: string }>;
}) {
  const [resolvedParams, resolvedSearchParams] = await Promise.all([params, searchParams]);
  const simulationId = resolvedParams.simulationId?.[0] ?? null;

  // Get comparison simulation IDs from URL (comma-separated for forecaster
  // comparison mode). The timeline chart's list lives in `csim` and the takeoff
  // chart's in `tkcsim`; datasets are keyed by simulation id, so loading the
  // union serves both.
  const comparisonIds = Array.from(new Set(
    [resolvedSearchParams.csim ?? '', resolvedSearchParams.tkcsim ?? '']
      .flatMap(param => param.split(','))
      .map(id => id.trim())
      .filter(id => id.length > 0)
  ));

  // Load registry outside of cache to determine simulation
  const registry = await loadSimulationRegistry();

  // Redirect to the default simulation if no simulationId is provided
  // This ensures shared links point to a specific forecast version
  if (!simulationId && registry.defaultSimulation) {
    redirect(`/forecast/${encodeURIComponent(registry.defaultSimulation)}`);
  }

  return <CachedForecastContent simulationId={simulationId} comparisonIds={comparisonIds} registry={registry} />;
}

// Main page component that wraps content in Suspense
export default function ForecastPage({
  params,
  searchParams,
}: {
  params: Promise<{ simulationId?: string[] }>;
  searchParams: Promise<{ csim?: string; tkcsim?: string }>;
}) {
  return (
    <Suspense fallback={
      <div className="flex min-h-screen items-center justify-center text-sm text-slate-500">
        Loading forecast...
      </div>
    }>
      <ForecastPageContent params={params} searchParams={searchParams} />
    </Suspense>
  );
}
