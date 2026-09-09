import path from "node:path";
import { readFile, readdir } from "node:fs/promises";
import { Suspense } from "react";
import Link from "next/link";
import { cacheLife, cacheTag } from "next/cache";
import { HeaderContent } from "@/components/HeaderContent";
import Image from "next/image";
import TableOfContents from "@/components/TableOfContents";
import { RunSelectorClient } from "@/components/RunSelectorClient";
import { CorrelationMatrixHeatmap } from "@/components/CorrelationMatrixHeatmap";
import type { CorrelationMatrix } from "@/types/samplingConfig";

// Types
interface Run {
  id: string;
  label: string;
  forecaster: "eli" | "daniel";
  date: string;
  numSamples: number;
  dataPath: string;
}

interface RunRegistry {
  version: number;
  defaultRun: string | null;
  runs: Run[];
}

interface ArrivalProbabilities {
  milestone: string;
  byDec2027: string;
  byDec2030: string;
  byDec2035: string;
}

interface FastTakeoffProbabilities {
  condition: string;
  values: { [key: string]: string };
}

interface FigureManifest {
  shortTimelines: string[];
  fastTakeoff: string[];
  timelineCorrelation: string[];
  horizonTrajectories: string[];
  mOverBeta: string | null;
  sensitivityAnalysis: {
    tornadoAc: string | null;
    tornadoAcAsiTakeoff: string | null;
    spearmanTopAaTime: string | null;
  };
  continuousPlots: {
    doublingDifficultyGrowthFactor: string | null;
    presentDoublingTime: string | null;
    aiResearchTasteSlope: string | null;
    medianToTopTasteMultiplier: string | null;
  };
  noCorrelation: {
    probAi2027TakeoffVsSarArrival: string | null;
    prob1yrTakeoffVsAcArrival: string | null;
    parameterSensitivityScatterAc: string | null;
  };
}

// CSV parsing utility
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

// Data loading functions
async function loadRunRegistry(): Promise<RunRegistry> {
  const registryPath = path.join(process.cwd(), "app/results/data/runs/runs.json");
  try {
    const raw = await readFile(registryPath, "utf8");
    return JSON.parse(raw) as RunRegistry;
  } catch {
    return {
      version: 1,
      defaultRun: null,
      runs: [],
    };
  }
}

async function loadArrivalProbabilities(basePath: string): Promise<ArrivalProbabilities[]> {
  const filePath = path.join(process.cwd(), basePath, "short_timelines_outputs", "arrival_probabilities.csv");
  try {
    const raw = await readFile(filePath, "utf8");
    const lines = raw.split(/\r?\n/).map(line => line.trim()).filter(line => line.length > 0);
    const [_header, ...dataLines] = lines;

    return dataLines.map(line => {
      const fields = parseCsvLine(line);
      return {
        milestone: fields[0],
        byDec2027: fields[1],
        byDec2030: fields[2],
        byDec2035: fields[3],
      };
    });
  } catch {
    return [];
  }
}

async function loadFastTakeoffProbabilities(basePath: string, source: "sar" | "ac"): Promise<FastTakeoffProbabilities[]> {
  const filePath = path.join(process.cwd(), basePath, "fast_takeoff_outputs", `conditional_fast_takeoff_probs_${source}.csv`);
  try {
    const raw = await readFile(filePath, "utf8");
    const lines = raw.split(/\r?\n/).map(line => line.trim()).filter(line => line.length > 0);
    const [header, ...dataLines] = lines;
    const headers = parseCsvLine(header);

    return dataLines.map(line => {
      const fields = parseCsvLine(line);
      const values: { [key: string]: string } = {};
      for (let i = 1; i < headers.length; i++) {
        values[headers[i]] = fields[i];
      }
      return {
        condition: fields[0],
        values,
      };
    });
  } catch {
    return [];
  }
}

async function loadCorrelationMatrix(basePath: string): Promise<CorrelationMatrix | null> {
  const filePath = path.join(process.cwd(), basePath, "correlation_matrix.json");
  try {
    const raw = await readFile(filePath, "utf8");
    return JSON.parse(raw) as CorrelationMatrix;
  } catch {
    return null;
  }
}

async function loadFigureManifest(runId: string, forecaster: "eli" | "daniel"): Promise<FigureManifest> {
  const manifest: FigureManifest = {
    shortTimelines: [],
    fastTakeoff: [],
    timelineCorrelation: [],
    horizonTrajectories: [],
    mOverBeta: null,
    sensitivityAnalysis: {
      tornadoAc: null,
      tornadoAcAsiTakeoff: null,
      spearmanTopAaTime: null,
    },
    continuousPlots: {
      doublingDifficultyGrowthFactor: null,
      presentDoublingTime: null,
      aiResearchTasteSlope: null,
      medianToTopTasteMultiplier: null,
    },
    noCorrelation: {
      probAi2027TakeoffVsSarArrival: null,
      prob1yrTakeoffVsAcArrival: null,
      parameterSensitivityScatterAc: null,
    },
  };

  // Images are stored in public/results/<run-id>/
  const baseDir = path.join(process.cwd(), "public", "results", runId);

  // Short timelines figures
  try {
    const shortTimelineDir = path.join(baseDir, "short_timelines_outputs");
    const files = await readdir(shortTimelineDir);
    manifest.shortTimelines = files.filter(f => f.endsWith(".png"));
  } catch { /* directory doesn't exist */ }

  // Fast takeoff figures
  try {
    const fastTakeoffDir = path.join(baseDir, "fast_takeoff_outputs");
    const files = await readdir(fastTakeoffDir);
    manifest.fastTakeoff = files.filter(f => f.endsWith(".png"));
  } catch { /* directory doesn't exist */ }

  // Timeline correlation figures
  try {
    const timelineCorrelationDir = path.join(baseDir, "timeline_takeoff_correlation");
    const files = await readdir(timelineCorrelationDir);
    manifest.timelineCorrelation = files.filter(f => f.endsWith(".png"));
  } catch { /* directory doesn't exist */ }

  // Horizon trajectories
  try {
    const horizonDir = path.join(baseDir, "horizon_trajectories");
    const files = await readdir(horizonDir);
    manifest.horizonTrajectories = files.filter(f => f.endsWith(".png"));
  } catch { /* directory doesn't exist */ }

  // m/beta histogram
  try {
    const files = await readdir(baseDir);
    const mOverBetaFile = files.find(f => f === "m_over_beta_hist.png");
    if (mOverBetaFile) {
      manifest.mOverBeta = mOverBetaFile;
    }
  } catch { /* directory doesn't exist */ }

  // Sensitivity analysis figures
  try {
    const sensitivityDir = path.join(baseDir, "sensitivity_analysis");
    const files = await readdir(sensitivityDir);
    const tornadoFile = files.find(f => f === "tornado_ac.png");
    if (tornadoFile) {
      manifest.sensitivityAnalysis.tornadoAc = `sensitivity_analysis/${tornadoFile}`;
    }
    const tornadoTakeoffFile = files.find(f => f === "tornado_ac_asi_takeoff.png");
    if (tornadoTakeoffFile) {
      manifest.sensitivityAnalysis.tornadoAcAsiTakeoff = `sensitivity_analysis/${tornadoTakeoffFile}`;
    }
    const spearmanFile = files.find(f => f === "sensitivity_spearman_top_aa_time.png");
    if (spearmanFile) {
      manifest.sensitivityAnalysis.spearmanTopAaTime = `sensitivity_analysis/${spearmanFile}`;
    }
  } catch { /* directory doesn't exist */ }

  // Continuous plots
  const continuousPlotsConfig = [
    { key: "doublingDifficultyGrowthFactor" as const, dir: "doubling-difficulty-growth-factor", file: "aa-time_vs_doubling-difficulty-growth-factor.png" },
    { key: "presentDoublingTime" as const, dir: "present-doubling-time", file: "aa-time_vs_present-doubling-time.png" },
    { key: "aiResearchTasteSlope" as const, dir: "ai-research-taste-slope", file: "p-1yr-takeoff-AC_vs_ai-research-taste-slope.png" },
    { key: "medianToTopTasteMultiplier" as const, dir: "median-to-top-taste-multiplier", file: "p-1yr-takeoff-AC_vs_median-to-top-taste-multiplier.png" },
  ];

  for (const { key, dir, file } of continuousPlotsConfig) {
    try {
      const plotDir = path.join(baseDir, "continuous_plots", dir);
      const files = await readdir(plotDir);
      if (files.includes(file)) {
        manifest.continuousPlots[key] = `continuous_plots/${dir}/${file}`;
      }
    } catch { /* directory doesn't exist */ }
  }

  // No-correlation comparison figures - forecaster specific
  // Stored in public/results/<forecaster>_no_correlation/
  const noCorrelationDir = path.join(process.cwd(), "public", "results", `${forecaster}_no_correlation`);
  const noCorrelationConfig = [
    { key: "probAi2027TakeoffVsSarArrival" as const, path: "fast_takeoff_outputs/prob_ai2027_takeoff_vs_sar_arrival.png" },
    { key: "prob1yrTakeoffVsAcArrival" as const, path: "fast_takeoff_outputs/prob_1yr_takeoff_vs_ac_arrival.png" },
    { key: "parameterSensitivityScatterAc" as const, path: "timeline_takeoff_correlation/parameter_sensitivity_scatter_ac.png" },
  ];

  for (const { key, path: filePath } of noCorrelationConfig) {
    try {
      const fullPath = path.join(noCorrelationDir, filePath);
      await readFile(fullPath);
      manifest.noCorrelation[key] = `/results/${forecaster}_no_correlation/${filePath}`;
    } catch { /* file doesn't exist */ }
  }

  return manifest;
}

// Components
function ResultsSection({
  title,
  id,
  children,
}: {
  title: string;
  id?: string;
  children: React.ReactNode;
}) {
  return (
    <section id={id} className="space-y-6 mt-20 first:mt-0">
      <h2 className="font-mono font-semibold">
        {title}
      </h2>
      {children}
    </section>
  );
}

function ProbabilityTable({
  title,
  headers,
  rows,
}: {
  title?: string;
  headers: string[];
  rows: { label: string; values: string[] }[];
}) {
  return (
    <div className="my-6">
      {title && <h3 className="font-mono font-semibold mb-4">{title}</h3>}
      <div className="overflow-x-auto">
        <table className="min-w-full border-collapse border border-gray-300">
          <thead>
            <tr className="bg-gray-50">
              <th className="border border-gray-300 px-4 py-2 text-left text-sm font-semibold text-gray-900">
                {headers[0]}
              </th>
              {headers.slice(1).map((header, i) => (
                <th key={i} className="border border-gray-300 px-4 py-2 text-right text-sm font-semibold text-gray-900">
                  {header}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, i) => (
              <tr key={i} className={i % 2 === 0 ? "bg-white" : "bg-gray-50"}>
                <td className="border border-gray-300 px-4 py-2 text-sm font-medium text-gray-900">
                  {row.label}
                </td>
                {row.values.map((value, j) => (
                  <td key={j} className="border border-gray-300 px-4 py-2 text-right text-sm text-gray-700">
                    {value}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function FigurePlaceholder({
  alt,
  caption,
  className = "",
}: {
  alt: string;
  caption?: string;
  className?: string;
}) {
  return (
    <figure className={`my-8 ${className}`}>
      <div className="relative w-full bg-gray-100 border-2 border-dashed border-gray-300 rounded-lg flex items-center justify-center" style={{ minHeight: "400px" }}>
        <div className="text-center p-8">
          <svg className="mx-auto h-12 w-12 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
          </svg>
          <p className="mt-4 text-sm text-gray-500">Figure not yet available</p>
          <p className="mt-1 text-xs text-gray-400">{alt}</p>
        </div>
      </div>
      {caption && (
        <figcaption className="mt-2 text-center text-sm text-gray-600 italic">
          {caption}
        </figcaption>
      )}
    </figure>
  );
}

function FigureDisplay({
  src,
  alt,
  caption,
  className = "",
}: {
  src: string;
  alt: string;
  caption?: string;
  className?: string;
}) {
  return (
    <figure className={`my-8 ${className}`}>
      <div className="relative w-full">
        <Image
          src={src}
          alt={alt}
          width={1200}
          height={800}
          className="w-full h-auto rounded-lg"
          style={{ maxWidth: "100%", height: "auto" }}
        />
      </div>
      {caption && (
        <figcaption className="mt-2 text-center text-sm text-gray-600 italic">
          {caption}
        </figcaption>
      )}
    </figure>
  );
}

// Main cached content component
async function CachedResultsContent({
  runId,
  registry,
}: {
  runId: string | null;
  registry: RunRegistry;
}) {
  "use cache";
  cacheLife("hours");
  cacheTag(`results-${runId ?? "default"}`);

  // Determine which run to use
  let currentRun: Run | null = null;

  if (registry.runs.length > 0) {
    if (runId) {
      currentRun = registry.runs.find(r => r.id === runId) ?? null;
    }
    if (!currentRun && registry.defaultRun) {
      currentRun = registry.runs.find(r => r.id === registry.defaultRun) ?? null;
    }
    if (!currentRun) {
      currentRun = registry.runs[0];
    }
  }

  if (!currentRun) {
    return (
      <div className="flex min-h-screen items-center justify-center text-sm text-slate-500">
        No results data available. Please configure a results run.
      </div>
    );
  }

  const dataBasePath = `app/results/data/${currentRun.dataPath}`;
  const publicFiguresPath = `/results/${currentRun.id}`;

  // Load data
  const [arrivalProbs, fastTakeoffSAR, fastTakeoffAC, figures, correlationMatrix] = await Promise.all([
    loadArrivalProbabilities(dataBasePath),
    loadFastTakeoffProbabilities(dataBasePath, "sar"),
    loadFastTakeoffProbabilities(dataBasePath, "ac"),
    loadFigureManifest(currentRun.id, currentRun.forecaster),
    loadCorrelationMatrix(dataBasePath),
  ]);

  return (
    <div className="grid h-screen w-full grid-cols-[minmax(0,1fr)_auto] grid-rows-[minmax(0,1fr)] gap-0">
      <div className="relative col-start-1 row-start-1 flex min-h-0 flex-col">
        <TableOfContents
          contentSelector="#results-content"
          headingSelector="h2, h3, h4"
          minLevel={2}
          maxLevel={4}
          topOffsetPx={120}
        />
        <div className="flex min-h-0 flex-col overflow-y-auto px-6 pb-10" data-scroll-container="results">
          <HeaderContent variant="inline" className="pt-6 pb-4" />
          <main id="results-content" className="mt-10 mx-auto max-w-5xl pb-16 xl:pl-60">
            {/* Run Selector */}

            {/* Page Title */}
            <h1 className="text-3xl font-et-book font-bold text-primary">Analyzing Our Model&apos;s Behavior</h1>
            <div className="bg-amber-50 border border-amber-200 rounded-lg p-4 mt-6">
              <p className="text-sm text-amber-900">
                <strong>Note:</strong> this analysis has not been updated since the site&apos;s launch.
                The figures below reflect the model and forecasts as of launch, not our latest
                parameter updates — see the <Link href="/forecast" className="underline">forecast page</Link> for
                current distributions.
              </p>
            </div>
            <RunSelectorClient runs={registry.runs} currentRunId={currentRun.id} />

            {/* Introduction */}
            <section id="introduction" className="space-y-6 mt-10">
              {/* <p className="prose text-base leading-relaxed">
                We focus primarily on analyzing the following outcomes, chosen based on a combination of
                action relevance and which parts of our model behavior we are relatively more confident about:
              </p>
              <ol className="prose list-decimal list-inside space-y-2 text-base ml-4">
                <li>
                  <strong>Probability of short timelines to AI R&D automation:</strong> p(AC) and p(SAR) by 2027, 2030, and 2035.
                </li>
                <li>
                  <strong>Probability of fast takeoff post-AC+SAR:</strong> Probability that takeoff is at least as fast as AI 2027, and that it&apos;s &lt;= 1 year. Where takeoff ends at either TED-AI or ASI.
                </li>
              </ol>
              <p className="prose text-base leading-relaxed">
                We then analyze the correlation in our model between timelines and takeoff, followed by various outcomes.
              </p>
              <p className="prose text-base leading-relaxed">
                Next, we discuss the possibility of a software intelligence explosion, including conditions for it in our model and how it affects our model behavior.
              </p>
              <p className="text-sm text-gray-500">
                Results computed from {currentRun.numSamples.toLocaleString()} Monte Carlo samples using {currentRun.label}.
              </p> */}
            </section>

            {/* Probability of Short Timelines - TEMPORARILY HIDDEN */}
            {false && (<ResultsSection title="Probability of Short Timelines to AI R&D Automation" id="short-timelines">
              <p className="prose text-base leading-relaxed">
                Since we aren&apos;t modeling experiment selection labor, we technically cannot measure a SAR, we can only see when the experiment selection skill reaches top human level. But we think this is a good proxy for a SAR, so for the sake of space we will usually abbreviate SAR-level-experiment-selection as SAR.
              </p>

              {arrivalProbs.length > 0 && (
                <ProbabilityTable
                  title="Chance of AC and SAR by the end of 2027, 2030, and 2035"
                  headers={["Milestone", "By Dec 2027", "By Dec 2030", "By Dec 2035"]}
                  rows={arrivalProbs.map(p => ({
                    label: p.milestone,
                    values: [p.byDec2027, p.byDec2030, p.byDec2035],
                  }))}
                />
              )}

              <p className="prose text-base leading-relaxed">
                Essentially, there&apos;s a significant but clearly &lt;50% chance of timelines as short as AI 2027 (i.e. SAR by 2027), and a &gt;50% chance of full AI R&D automation (SAR) within a decade.
              </p>

              <div className="bg-amber-50 border border-amber-200 rounded-lg p-4 my-6">
                <h4 className="font-semibold text-amber-800 mb-2">Limitations</h4>
                <p className="text-sm text-amber-900 mb-2">The following limitations of our model may be substantially affecting these results:</p>
                <ol className="list-decimal list-inside space-y-2 text-sm text-amber-900">
                  <li>We aren&apos;t modeling data as an input to AI progress, which could either lengthen timelines if we get bottlenecked by data, or shorten timelines if data creation turns out to be particularly automatable. We&apos;ve attempted to account for the former possibility by putting some weight on exponential and subexponential-in-effective-compute time horizon progressions.</li>
                  <li>The shape of the super/sub-exponentials created by our methodology is different from what we expect. We think the true behavior will likely be more aggressive than the model predicts at later points in time, and less aggressive at earlier points in time. It&apos;s unclear how this would affect the models&apos; predictions.</li>
                  <li>We aren&apos;t modeling how software improvements change experiment compute requirements. This likely would push toward shorter timelines.</li>
                  <li>We aren&apos;t modeling the automation of hardware R&D or hardware production. This shouldn&apos;t affect AC timelines, but might affect SAR timelines a little.</li>
                </ol>
              </div>

              {figures.shortTimelines.includes("ac_sar_pdfs_overlay.png") && (
                <FigureDisplay
                  src={`${publicFiguresPath}/short_timelines_outputs/ac_sar_pdfs_overlay.png`}
                  alt="AC and SAR Arrival Time Distributions"
                  caption="AC and SAR arrival time probability distributions. The 90th percentiles show long tails, especially for SAR."
                />
              )}

              {figures.shortTimelines.includes("ac_to_sar_duration_pdf.png") && (
                <>
                  <h3 className="font-mono font-semibold mt-12">Transition Duration: AC → SAR</h3>
                  <p className="prose text-base leading-relaxed">
                    There is a substantial probability of a very fast AC→SAR transition, but also a long tail and a median of 1.2 years.
                  </p>
                  <FigureDisplay
                    src={`${publicFiguresPath}/short_timelines_outputs/ac_to_sar_duration_pdf.png`}
                    alt="Transition Duration from AC to SAR"
                    caption="Distribution of transition times from AC to SAR (including censored trajectories)."
                  />
                </>
              )}
            </ResultsSection>)}

            {/* Fast Takeoff - TEMPORARILY HIDDEN */}
            {false && (<ResultsSection title="Probability of Fast Takeoff" id="fast-takeoff">
              <p className="prose text-base leading-relaxed">
                We define 2 primary outcomes of interest:
              </p>
              <ol className="prose list-decimal list-inside space-y-3 text-base ml-4">
                <li>
                  <strong>Takeoff at least as fast as AI 2027:</strong> In accordance with AI 2027&apos;s capability progression, there are several possible indicators of this:
                  <ul className="list-disc list-inside ml-6 mt-1 space-y-1">
                    <li>p(SAR→TED-AI) ≤ 3 months</li>
                    <li>p(SAR→SIAR) ≤ 4 months</li>
                    <li>p(SAR→ASI) ≤ 5 months</li>
                    <li>p(AC→TED-AI) ≤ 9 months</li>
                    <li>p(AC→ASI) ≤ 1 year</li>
                  </ul>
                </li>
                <li>
                  <strong>One-year takeoff, beginning from SAR:</strong> Some possible indicators of this.
                  <ul className="list-disc list-inside ml-6 mt-1 space-y-1">
                    <li>p(SAR→TED-AI) ≤ 1 year</li>
                    <li>p(SAR→SIAR) ≤ 1 year</li>
                    <li>p(SAR→ASI) ≤ 1 year</li>
                  </ul>
                </li>
              </ol>

              {/* AI 2027 Takeoff Speed Section */}
              {fastTakeoffSAR.length > 0 && (() => {
                const ai2027Columns = Object.keys(fastTakeoffSAR[0]?.values || {}).filter(
                  col => col.includes("4mo") || col.includes("3mo") || col.includes("5mo")
                );
                return ai2027Columns.length > 0 && (
                  <ProbabilityTable
                    title="Takeoff as Fast as AI 2027 (from SAR)"
                    headers={["Condition", ...ai2027Columns]}
                    rows={fastTakeoffSAR.map(p => ({
                      label: p.condition,
                      values: ai2027Columns.map(col => p.values[col]),
                    }))}
                  />
                );
              })()}

              {/* P(AI 2027 Takeoff) vs SAR Arrival Year */}
              {figures.fastTakeoff.includes("prob_ai2027_takeoff_vs_sar_arrival.png") && (
                <FigureDisplay
                  src={`${publicFiguresPath}/fast_takeoff_outputs/prob_ai2027_takeoff_vs_sar_arrival.png`}
                  alt="P(AI 2027 Takeoff or Faster) vs SAR Arrival Year"
                  caption="Probability of AI-2027-speed takeoff vs SAR arrival year."
                />
              )}

              {/* 1 Year Takeoff Section */}
              {fastTakeoffSAR.length > 0 && (() => {
                const oneYearColumns = Object.keys(fastTakeoffSAR[0]?.values || {}).filter(
                  col => col.includes("1yr")
                );
                return oneYearColumns.length > 0 && (
                  <ProbabilityTable
                    title="One-Year Takeoff Probabilities (from SAR)"
                    headers={["Condition", ...oneYearColumns]}
                    rows={fastTakeoffSAR.map(p => ({
                      label: p.condition,
                      values: oneYearColumns.map(col => p.values[col]),
                    }))}
                  />
                );
              })()}

              {/* P(≤1 Year Takeoff) vs SAR Arrival Year */}
              {figures.fastTakeoff.includes("prob_1yr_takeoff_vs_sar_arrival.png") && (
                <FigureDisplay
                  src={`${publicFiguresPath}/fast_takeoff_outputs/prob_1yr_takeoff_vs_sar_arrival.png`}
                  alt="P(≤1 Year Takeoff) vs SAR Arrival Year"
                  caption="Probability of ≤1 year takeoff vs SAR arrival year."
                />
              )}

              <p className="prose text-base leading-relaxed">
                We see a strong correlation between timelines and takeoff, with takeoff as fast as AI 2027 being more likely than not if SAR arrives in 2027, but overall 30% likely. There&apos;s a very high chance of a one-year takeoff conditional on SAR in 2027, but unconditionally a one-year takeoff is 46%.
              </p>
            </ResultsSection>)}

            {/* Parameter Sensitivity: Spearman Correlations and Continuous Plots */}
            {(figures.sensitivityAnalysis.tornadoAc ||
              figures.sensitivityAnalysis.spearmanTopAaTime ||
              figures.continuousPlots.doublingDifficultyGrowthFactor ||
              figures.continuousPlots.presentDoublingTime ||
              figures.continuousPlots.aiResearchTasteSlope ||
              figures.continuousPlots.medianToTopTasteMultiplier) && (
                <ResultsSection title="Which parameters have the greatest impact?" id="parameter-sensitivity">
                  <p className="prose text-base leading-relaxed">
                    Our model has many parameters. Some of them are well-grounded in empirical data, but others are more uncertain and require &quot;judgmental forecasting&quot; (or, &quot;guessing&quot;) to estimate. In this section, we investigate how sensitive key model outcomes are to different parameters.
                  </p>
                  <h3 className="font-mono font-semibold mt-12">One-at-a-Time Sensitivity: Timelines to AC and Takeoff from AC to ASI</h3>
                  <p className="prose text-base leading-relaxed">
                    Beginning with all parameters set to our median estimates of their values, we individually change each one to our 10th percentile estimate and our 90th percentile estimate, then plot how the timeline and takeoff speed changes. The size of the &quot;spread&quot; resulting from these changes gives a simple measure of how much of our uncertainty in the outcome comes from our uncertainty about each parameter.
                  </p>
                  {figures.sensitivityAnalysis.tornadoAc && (
                    <>
                      {/* <h3 className="font-mono font-semibold mt-12">One-at-a-Time Sensitivity: AC Arrival Time</h3> */}
                      {/* <p className="prose text-base leading-relaxed">
                        This tornado plot shows how the AC arrival time changes when each parameter is individually varied from its 10th to 90th percentile while holding all other parameters at their median values. Parameters are ranked by their impact on the outcome.
                      </p> */}
                      <FigureDisplay
                        src={`${publicFiguresPath}/${figures.sensitivityAnalysis.tornadoAc}`}
                        alt="Tornado Plot: Parameter Sensitivity for AC Arrival Time"
                        caption="One-at-a-time sensitivity analysis showing how each parameter affects the AC arrival time when varied from 10th to 90th percentile. We plot only the top ten most significant parameters."
                      />
                    </>
                  )}

                  <p className="prose text-base leading-relaxed mt-6">
                    The most important parameters affecting the time to AC are:
                  </p>
                  <ol className="prose list-decimal list-inside space-y-2 text-base ml-4 mt-2">
                    <li>Doubling difficulty growth factor;</li>
                    <li>Present doubling time for time horizon;</li>
                    <li>Time horizon requirement for AC.</li>
                  </ol>
                  <p className="prose text-base leading-relaxed mt-4">
                    The first two parameters together determine the mapping from log(effective compute) to the AI&apos;s time horizon. The doubling difficulty parameter controls how steeply super or sub-exponential this mapping is, and the present doubling time gives the initial growth rate of time horizon. It&apos;s intuitive that these would be important: they tell us how difficult it is to double time horizon in the present and how that difficulty will change in the future. In particular, even a small change in the doubling difficulty growth factor <em>d</em> can make a big difference since we take powers of <em>d</em> to find doubling times after many doublings.
                  </p>
                  <p className="prose text-base leading-relaxed mt-4">
                    It&apos;s also intuitive that the AC time horizon requirement should have a significant effect: if you need a time horizon of 100 work years to automate all coding, that&apos;s going to take longer than if you need a time horizon of 6 months.
                  </p>
                  <p className="prose text-base leading-relaxed mt-4">
                    The other parameters shown in the graph influence how gains in effective compute are converted into AI software R&amp;D uplift. The sensitivity analysis reveals these are less important for AC timelines than the three parameters discussed above. This makes sense: before getting to AC, coding bottlenecks on the tasks the AIs can&apos;t complete yet. And the AIs&apos; research taste is not yet good enough to make a big difference in selecting good experiments.
                  </p>
                  <p className="prose text-base leading-relaxed mt-4">
                    Now let&apos;s look at what parameters takeoff speeds are most sensitive to.
                  </p>
                  {figures.sensitivityAnalysis.tornadoAcAsiTakeoff && (
                    <>
                      {/* <h3 className="font-mono font-semibold mt-12">One-at-a-Time Sensitivity: AC → ASI Takeoff Duration</h3>
                      <p className="prose text-base leading-relaxed">
                        This tornado plot shows how the takeoff duration (time from AC to ASI) changes when each parameter is individually varied from its 10th to 90th percentile while holding all other parameters at their median values.
                      </p> */}
                      <FigureDisplay
                        src={`${publicFiguresPath}/${figures.sensitivityAnalysis.tornadoAcAsiTakeoff}`}
                        alt="Tornado Plot: Parameter Sensitivity for AC to ASI Takeoff Duration"
                        caption="One-at-a-time sensitivity analysis showing how each parameter affects the takeoff duration from AC to ASI."
                      />
                    </>
                  )}

                  <p className="prose text-base leading-relaxed mt-6">
                    The most important parameters are:
                  </p>
                  <ol className="prose list-decimal list-inside space-y-3 text-base ml-4 mt-2">
                    <li>
                      <strong>The ones determining how quickly AIs&apos; research taste improves as effective compute increases: automated research taste slope and the median-to-top research taste multiplier.</strong> We call how quickly AIs&apos; research taste improves as effective compute increases <em>m</em> when discussing software intelligence explosions. <em>m</em> is primarily a function of these 2 parameters.
                    </li>
                    <li>
                      <strong>The software efficiency growth rate in 2024,</strong> which influences the rate at which (experiment) <a href="https://www.aeaweb.org/articles?id=10.1257/aer.20180338" className="text-accent hover:text-accent/80" target="_blank" rel="noopener noreferrer">ideas become harder to find</a>. Along with the <em>m</em> described above this parameter has lots of influence on the extent to which there is a software intelligence explosion: see <a href="https://www.aifuturesmodel.com/#S5" className="text-accent hover:text-accent/80" target="_blank" rel="noopener noreferrer">here</a> for discussion.
                    </li>
                    <li>
                      <strong>Median-to-top-human jumps above SAR needed to reach TED-AI:</strong> This determines at what research taste capabilities ASI is achieved, very directly affecting the time taken from AC to ASI.
                    </li>
                    <li>
                      <strong>Those controlling the level of compute bottlenecking in experiment throughput (infinite coding labor asymptote and slowdown from 10x less experiment compute).</strong> The most important parameters, i.e. the ones above, have to do with research taste rather than coding because coding gets bottlenecked on experimental compute. That said, coding automation still matters for takeoff: we see that parameters controlling the experiment throughput CES have a significant impact, particularly ones controlling the level of compute bottlenecking.
                    </li>
                  </ol>
                </ResultsSection>
              )}

                  {/* {figures.continuousPlots.doublingDifficultyGrowthFactor && (
                    <>
                      <h3 className="font-mono font-semibold mt-12">P(AC by Date) vs Doubling Difficulty Growth Factor</h3>
                      <FigureDisplay
                        src={`${publicFiguresPath}/${figures.continuousPlots.doublingDifficultyGrowthFactor}`}
                        alt="P(AC by Date) vs Doubling Difficulty Growth Factor"
                        caption="How the doubling difficulty growth factor affects probability of reaching AC by various dates."
                      />
                    </>
                  )}

                  {figures.continuousPlots.presentDoublingTime && (
                    <>
                      <h3 className="font-mono font-semibold mt-12">P(AC by Date) vs Present Doubling Time</h3>
                      <FigureDisplay
                        src={`${publicFiguresPath}/${figures.continuousPlots.presentDoublingTime}`}
                        alt="P(AC by Date) vs Present Doubling Time"
                        caption="How the present doubling time affects probability of reaching AC by various dates."
                      />
                    </>
                  )}

                  {figures.continuousPlots.aiResearchTasteSlope && (
                    <>
                      <h3 className="font-mono font-semibold mt-12">P(AI 2027 Speed Takeoff) vs Automated Research Taste Slope</h3>
                      <FigureDisplay
                        src={`${publicFiguresPath}/${figures.continuousPlots.aiResearchTasteSlope}`}
                        alt="P(AI 2027 Speed Takeoff) vs Automated Research Taste Slope"
                        caption="How AI research taste slope affects the probability of AI-2027-speed takeoff."
                      />
                    </>
                  )}

                  {figures.continuousPlots.medianToTopTasteMultiplier && (
                    <>
                      <h3 className="font-mono font-semibold mt-12">P(AI 2027 Speed Takeoff) vs Median To Top Taste Multiplier</h3>
                      <FigureDisplay
                        src={`${publicFiguresPath}/${figures.continuousPlots.medianToTopTasteMultiplier}`}
                        alt="P(AI 2027 Speed Takeoff) vs Median To Top Taste Multiplier"
                        caption="How the median-to-top taste multiplier affects the probability of AI-2027-speed takeoff."
                      />
                    </>
                  )}
                </ResultsSection>
              )} */}

            {/* Timeline-Takeoff Correlation */}
            <ResultsSection title="Correlation Between Timelines and Takeoff" id="correlation">
              <p className="prose text-base leading-relaxed">
                We observe that shorter timelines to the Automated Coder milestone (AC) are correlated with faster takeoffs. This plot shows how the empirical probability of &quot;fast&quot; takeoff (meaning less than 1 year) varies conditional on different AC arrival years.
              </p>
              {/* P(1 Year Takeoff or Faster) vs AC Arrival Year */}
              {figures.fastTakeoff.includes("prob_1yr_takeoff_vs_ac_arrival.png") && (
                <FigureDisplay
                  src={`${publicFiguresPath}/fast_takeoff_outputs/prob_1yr_takeoff_vs_ac_arrival.png`}
                  alt="P(1 Year Takeoff or Faster) vs AC Arrival Year"
                  caption="Probability of ≤1 year takeoff vs AC arrival year."
                />
              )}
              <p className="prose text-base leading-relaxed">
                For example, the probability of a one-year AC-to-ASI transition is 42% if AC comes in 2027, but only 16% if AC comes in 2035. Why are shorter timelines to AC correlated with faster takeoffs?
              </p>
              <p className="prose text-base leading-relaxed">
                There are two potential sources driving this pattern. For one, our Monte Carlo simulation draws correlated samples for some of the parameters.
              </p>
                            {/* Input Parameter Correlations */}
              {correlationMatrix && (
                <details className="mt-12">
                  <summary className="font-mono font-semibold cursor-pointer hover:text-accent">
                    Input Parameter Correlations
                  </summary>
                  <div className="mt-4">
                    <p className="prose text-base leading-relaxed mb-4">
                      The correlation matrix below shows the correlations between sampled parameters in our Monte Carlo simulation. These correlations are specified in the sampling configuration to capture dependencies between parameters.
                    </p>
                    <div className="overflow-x-auto">
                      <CorrelationMatrixHeatmap correlationMatrix={correlationMatrix} />
                    </div>
                  </div>
                </details>
              )}
              <p className="prose text-base leading-relaxed">
                The other potential driver would come from the intrinsic model dynamics. To isolate this source, we plot the probability of 1-year takeoff with various AC arrival years, but with no correlations between parameter values:
              </p>
              {figures.noCorrelation.prob1yrTakeoffVsAcArrival ? (
                <FigureDisplay
                  src={figures.noCorrelation.prob1yrTakeoffVsAcArrival}
                  alt="P(1 Year Takeoff) vs AC Arrival Year - No Correlation"
                  caption="P(≤1 year takeoff) vs AC arrival year with parameter correlations removed."
                />
              ) : (
                <FigurePlaceholder
                  alt="P(1 Year Takeoff) vs AC Arrival Year - No Correlation"
                  caption="No-correlation run not yet available for this forecaster."
                />
              )}

              <p className="prose text-base leading-relaxed">
                We see that the correlation is still significant, but lower than when we included correlations between parameter values. It appears that the change mostly derives from the shorter-timeline scenarios.
              </p>

              <p className="prose text-base leading-relaxed">
                Given that there is a significant correlation between timelines and takeoff speeds from the model dynamics alone, let&apos;s first explore the causes of that. In order to do this, we&apos;ll want to plot, for each parameter, how sensitive each of timelines and takeoff speeds are to it. The parameters that are correlated with both of them in the same direction are the ones driving the timelines/takeoff correlation. We&apos;ll also need a single summary statistic for each parameter&apos;s effect on an outcome variable: for this we&apos;ll use the Spearman correlation coefficient, which measures how well the rank ordering of parameter values correlates with the rank ordering of outcomes.
              </p>

              <p className="prose text-base leading-relaxed">
                Below is a scatter plot where each point is a parameter, the x-axis is the Spearman correlation between that parameter and takeoff length from AC to ASI, and the y-axis is the Spearman correlation with the AC arrival time.
              </p>

              {figures.noCorrelation.parameterSensitivityScatterAc ? (
                <FigureDisplay
                  src={figures.noCorrelation.parameterSensitivityScatterAc}
                  alt="Parameter Effects: AC Timeline vs Takeoff Speed - No Correlation"
                  caption="Same analysis but with parameter correlations removed. Note how some parameter-outcome relationships change."
                />
              ) : (
                <FigurePlaceholder
                  alt="Parameter Effects: AC Timeline vs Takeoff Speed - No Correlation"
                  caption="No-correlation run not yet available for this forecaster."
                />
              )}

              <p className="prose text-base leading-relaxed">
                It looks like the parameters that are most correlated in the same direction with timelines and takeoff speeds are:
              </p>

              <ol className="prose list-decimal list-inside space-y-3 text-base ml-4 mt-2">
                <li>
                  <strong>Software efficiency growth rate in 2024:</strong> The larger the share of recent progress that comes from software efficiency as opposed to training compute scaling, the larger an impact automation has (shortening timelines), and furthermore the slowdown in training compute growth has less of an effect. For example, if the current software progress share were only 1%, speeding up software progress by 10x would have a small impact compared to if it were 50%. This affects timelines via the level of pre-AC automation and takeoff speeds via the same mechanism but for post-AC automation.
                </li>
                <li>
                  <strong>Slowdown from 10x less compute:</strong> Higher values mean that experiment throughput and therefore research effort is more bottlenecked on experiment compute, leading to both longer timelines and slower takeoff. Although takeoff is primarily driven by research taste, coding can have a substantial effect, and if this parameter is large then it&apos;s less likely to play a big role.
                </li>
                <li>
                  <strong>Infinite coding labor asymptote:</strong> This parameter behaves similarly to slowdown from 10x compute but with the inverse correlations, and the correlations are due to the same factors. If the asymptote is higher, then there is more room for coding automation to contribute to both pre-AC and post-AC automation.
                </li>
                <li>
                  <strong>Automated research taste at AC:</strong> The higher this is, the higher an AC&apos;s AI software R&amp;D uplift is (due to AC having better research taste), meaning that pre-AC automation has a larger effect on timelines to AC. This also means that the post-AC takeoff starts at a higher uplift value, leading to a faster takeoff.
                </li>
              </ol>

              <p className="prose text-base leading-relaxed mt-4">
                We can also see that all of the above parameters affect automation both pre and post-AC, but they all affect takeoff at least a little more than timelines. This is because in most simulations there is not a huge effect from automation pre-AC.
              </p>

              <p className="prose text-base leading-relaxed">
                Now we&apos;ll explore how these effects change if we add back in our correlations between parameters.
              </p>

              {figures.timelineCorrelation.includes("parameter_sensitivity_scatter_ac.png") && (
                <FigureDisplay
                  src={`${publicFiguresPath}/timeline_takeoff_correlation/parameter_sensitivity_scatter_ac.png`}
                  alt="Parameter Effects: AC Timeline vs Takeoff Speed"
                  caption="Each point is a parameter. X-axis: correlation with AC arrival time. Y-axis: correlation with takeoff speed (AC→ASI)."
                />
              )}

              <p className="prose text-base leading-relaxed">
                There is now a cluster of 3 parameters that are most correlated in the same direction for each of timelines and takeoff speeds: automated research taste slope, doubling difficulty growth factor, and present doubling time. These 3 were already strongly associated with either short timelines or fast takeoff when we excluded correlations between parameters. In our forecast, we made the taste slope negatively correlated with both the doubling difficulty growth factor and present doubling time, because all 3 of them are measures of the rate at which increased effective compute translates into capabilities. These specific parameter correlations are a substantial driver in the correlations between timelines and takeoff.
              </p>

              <p className="prose text-base leading-relaxed">
                These are the biggest changes; some other parameters also now have larger correlations with each of timelines and takeoff, but the 3 discussed above are the largest by a substantial amount.
              </p>
            </ResultsSection>

            {/* Time Horizon Trajectories */}
            <ResultsSection title="Time Horizon Trajectories" id="trajectories">
              <p className="prose text-base leading-relaxed">
                Here are all the time horizon trajectories, shaded by how well they backcast the METR points according to mean squared error (MSE). We view this sort of backcasting as an important exercise but not the end-all be-all, given that the data we have are so limited. Thanks to titotal&apos;s review of our previous model for prompting us to do more of this.
              </p>

              {figures.horizonTrajectories.includes("horizon_trajectories_all_shaded.png") ? (
                <FigureDisplay
                  src={`${publicFiguresPath}/horizon_trajectories/horizon_trajectories_all_shaded.png`}
                  alt="Complete Time Horizon Trajectories (All)"
                  caption="All time horizon trajectories, colored by MSE: green (low) → yellow → red (high)."
                />
              ) : (
                <FigurePlaceholder
                  alt="Complete Time Horizon Trajectories (All)"
                  caption="All time horizon trajectories, colored by MSE: green (low) → yellow → red (high)."
                />
              )}

              <h3 className="font-et-book text-xl font-semibold leading-[1.2] mt-12">Filtered Trajectories (MSE ≤ 1.0)</h3>
              <p className="prose text-base leading-relaxed">
                Here are approximately the top 1/3 of trajectories in terms of how well they backcast. Trajectories that backcast better tend to reach higher time horizons earlier.
              </p>
              {figures.horizonTrajectories.includes("horizon_trajectories_mse_1.0.png") ? (
                <FigureDisplay
                  src={`${publicFiguresPath}/horizon_trajectories/horizon_trajectories_mse_1.0.png`}
                  alt="Complete Time Horizon Trajectories (MSE ≤ 1.0)"
                  caption="Time horizon trajectories filtered to MSE ≤ 1.0."
                />
              ) : (
                <FigurePlaceholder
                  alt="Complete Time Horizon Trajectories (MSE ≤ 1.0)"
                  caption="Time horizon trajectories filtered to MSE ≤ 1.0."
                />
              )}
            </ResultsSection>

            {/* m/β Distribution
            {figures.mOverBeta && (
              <ResultsSection title="Distribution of m/β" id="m-over-beta">
                <p className="prose text-base leading-relaxed">
                  The ratio m/β determines whether a taste-only singularity occurs. When m/β &gt; 1, the feedback loop from research taste results in explosive software progress before reaching algorithmic limits.
                </p>
                <FigureDisplay
                  src={`${publicFiguresPath}/${figures.mOverBeta}`}
                  alt="Distribution of m/β"
                  caption="Distribution of m/β across Monte Carlo samples. Values > 1 indicate conditions for a software intelligence explosion."
                />
              </ResultsSection>
            )} */}

          </main>
        </div>
      </div>
    </div>
  );
}

// Content component that resolves params
async function ResultsPageContent({
  params,
}: {
  params: Promise<{ runId?: string[] }>;
}) {
  const resolvedParams = await params;
  const runId = resolvedParams.runId?.[0] ?? null;

  const registry = await loadRunRegistry();

  return <CachedResultsContent runId={runId} registry={registry} />;
}

// Main page component
export default function ResultsPage({
  params,
}: {
  params: Promise<{ runId?: string[] }>;
}) {
  return (
    <Suspense
      fallback={
        <div className="flex min-h-screen items-center justify-center text-sm text-slate-500">
          Loading results...
        </div>
      }
    >
      <ResultsPageContent params={params} />
    </Suspense>
  );
}
