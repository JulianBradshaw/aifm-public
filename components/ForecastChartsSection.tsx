'use client';

import { useState, useCallback, useEffect, useRef, useMemo } from 'react';
import { useSearchParams, useRouter } from 'next/navigation';
import MilestoneDistributionChart, { MilestoneDistributionData, ModeBreakoutData } from './MilestoneDistributionChart';
import ConditionalMilestoneTimingChart, { ConditionalTimingData } from './ConditionalMilestoneTimingChart';
// import { ProbabilityTable } from './ProbabilityTable';  // TEMPORARILY HIDDEN
import { TakeoffChartSidebar } from './TakeoffChartSidebar';
import { TimelineChartSidebar } from './TimelineChartSidebar';
import { XAxisZoomBar } from './XAxisZoomBar';
import { formatYearMonthShort } from '@/utils/formatting';
import { ArrivalProbabilities, TakeoffProbabilities, AtcCdfData, AtcTakeoffCdfData, AtcStatistics, ComparisonDataset } from '@/app/forecast/[[...simulationId]]/page';
import { Simulation } from './SimulationSelector';
import { TIMELINE_DISPLAY_MAX_YEAR, TAKEOFF_DISPLAY_MAX_YEARS } from '@/constants/forecastDisplay';
// import { MILESTONE_EXPLANATIONS } from '@/constants/chartExplanations';  // TEMPORARILY HIDDEN

// Milestone alias mappings for shorter URLs
const MILESTONE_ALIASES: Record<string, string> = {
  'AC': 'AC',
  'SC': 'AI2027-SC',
  'SAR': 'SAR-level-experiment-selection-skill',
  'SIAR': 'SIAR-level-experiment-selection-skill',
  'TED-AI': 'TED-AI',
  'ASI': 'ASI',
};

const MILESTONE_ALIASES_REVERSE: Record<string, string> = Object.fromEntries(
  Object.entries(MILESTONE_ALIASES).map(([short, full]) => [full, short])
);

// Default milestone selections
const DEFAULT_TIMELINE_MILESTONES = new Set(['AC']);
const DEFAULT_TAKEOFF_MILESTONES = new Set(['ASI']);

// Parse milestone param from URL (comma-separated short names)
function parseMilestoneParam(param: string | null): Set<string> | null {
  if (!param) return null;
  const milestones = param.split(',')
    .map(s => s.trim())
    .filter(s => s.length > 0)
    .map(short => MILESTONE_ALIASES[short] ?? short)  // Convert short to full name
    .filter(full => Object.values(MILESTONE_ALIASES).includes(full));  // Validate
  return milestones.length > 0 ? new Set(milestones) : null;
}

// Encode milestone set to URL param (comma-separated short names)
function encodeMilestoneParam(milestones: Set<string>): string {
  return Array.from(milestones)
    .map(full => MILESTONE_ALIASES_REVERSE[full] ?? full)
    .join(',');
}

// Parse comparison simulation ids from URL (comma-separated, deduped, order preserved)
function parseCsimParam(param: string | null): string[] {
  if (!param) return [];
  return Array.from(new Set(
    param.split(',').map(s => s.trim()).filter(s => s.length > 0)
  ));
}

type CompareMode = 'forecaster' | 'model-vs-atc';

// Compare-section params: the timeline chart owns the original (unprefixed)
// names, the takeoff chart's live under a `tk` prefix. Links minted before the
// two charts got independent Compare settings carry only the unprefixed names,
// so the takeoff chart falls back to those when no `tk` param is present - see
// the writer below, which emits `tkcmode` as soon as either chart is
// non-default, marking a link as post-split.
function parseCompareModeParam(param: string | null): CompareMode {
  return param === 'forecaster' ? 'forecaster' : 'model-vs-atc';  // Default: model-vs-atc
}

function parseShowTypeParam(param: string | null): 'model' | 'atc' {
  return param === 'atc' ? 'atc' : 'model';  // Default: model
}

// Absent (or unrecognised) means show both
function parseShowParam(param: string | null): { model: boolean; atc: boolean } {
  return { model: param !== 'atc', atc: param !== 'model' };
}

// Check if two sets are equal
function setsEqual(a: Set<string>, b: Set<string>): boolean {
  if (a.size !== b.size) return false;
  for (const item of a) {
    if (!b.has(item)) return false;
  }
  return true;
}

// Format a "years after AC" value for the takeoff zoom bar, trimming ".0"
function formatTakeoffYears(value: number): string {
  const rounded = Math.round(value * 10) / 10;
  const text = rounded.toFixed(1);
  return `${text.endsWith('.0') ? text.slice(0, -2) : text} yr`;
}

// Parse date in MM-DD-YY format (same as SimulationSelector)
const parseDate = (dateStr: string): number => {
  const [month, day, year] = dateStr.split('-').map(Number);
  // Convert YY to full year (assume 2000s)
  const fullYear = year < 100 ? 2000 + year : year;
  return new Date(fullYear, month - 1, day).getTime();
};

interface ForecastChartsSectionProps {
  distributionData: MilestoneDistributionData;
  unconditionalTimingData?: ConditionalTimingData;
  sharedYDomain: [number, number];
  numSamples: number | null;
  arrivalProbabilities: ArrivalProbabilities;
  takeoffProbabilities: TakeoffProbabilities;
  atcCdfData: AtcCdfData | null;
  atcStatistics: AtcStatistics | null;
  atcTakeoffCdfData: AtcTakeoffCdfData | null;
  forecasterName?: string;
  forecasterDate?: string;
  timelinesNotes?: string | null;
  comparisonDatasets?: ComparisonDataset[];
  // New props for comparison UI
  simulations: Simulation[];
  currentSimulationId: string;
  // Per-AC-anchor-mode breakout data (null for simulations without it)
  modeBreakoutData?: ModeBreakoutData | null;
}

// TEMPORARILY HIDDEN - Format probability as percentage string with 2 significant figures
// function formatProbability(prob: number): string {
//   const percentage = prob * 100;
//   if (percentage === 0) return '0%';
//   if (percentage >= 100) return '100%';
//   // Use toPrecision for 2 significant figures, then convert to number to remove trailing zeros
//   const rounded = Number(percentage.toPrecision(2));
//   return `${rounded}%`;
// }

// TEMPORARILY HIDDEN - Map milestone names to their short codes for looking up explanations
// function getMilestoneDescription(milestoneName: string): string | undefined {
//   // Map full milestone names to their short codes used in MILESTONE_EXPLANATIONS
//   const nameToCode: Record<string, string> = {
//     'AC': 'AC',
//     'SAR-level-experiment-selection-skill': 'SAR',
//     'SIAR-level-experiment-selection-skill': 'SIAR',
//     'TED-AI': 'TED-AI',
//     'ASI': 'ASI',
//   };
//   const code = nameToCode[milestoneName];
//   return code ? MILESTONE_EXPLANATIONS[code] : undefined;
// }

export function ForecastChartsSection({
  distributionData,
  unconditionalTimingData,
  sharedYDomain,
  numSamples: _numSamples,
  arrivalProbabilities: _arrivalProbabilities,  // TEMPORARILY HIDDEN
  takeoffProbabilities: _takeoffProbabilities,  // TEMPORARILY HIDDEN
  atcCdfData,
  atcStatistics,
  atcTakeoffCdfData,
  forecasterName,
  forecasterDate,
  timelinesNotes,
  comparisonDatasets = [],
  simulations,
  currentSimulationId,
  modeBreakoutData = null,
}: ForecastChartsSectionProps) {
  const searchParams = useSearchParams();
  const router = useRouter();
  const isInitialized = useRef(false);

  // Timeline chart specific settings - initialize from URL params
  const [timelineVisibleMilestones, setTimelineVisibleMilestones] = useState<Set<string>>(() => {
    const fromUrl = parseMilestoneParam(searchParams.get('timeline'));
    return fromUrl ?? DEFAULT_TIMELINE_MILESTONES;
  });
  const [timelineShowCdf, setTimelineShowCdf] = useState(false);  // Default: PDF mode

  // Takeoff chart specific settings - initialize from URL params
  const [takeoffVisibleMilestones, setTakeoffVisibleMilestones] = useState<Set<string>>(() => {
    const fromUrl = parseMilestoneParam(searchParams.get('takeoff'));
    return fromUrl ?? DEFAULT_TAKEOFF_MILESTONES;
  });
  const [takeoffShowCdf, setTakeoffShowCdf] = useState(true);  // Default: CDF mode

  // Gridlines, per chart like every other setting in these sidebars. Shared
  // state here meant toggling the grid on one chart silently toggled the other.
  const [timelineShowGridlines, setTimelineShowGridlines] = useState(true);  // Default: gridlines on
  const [takeoffShowGridlines, setTakeoffShowGridlines] = useState(true);  // Default: gridlines on

  // X-axis zoom windows (null = full auto domain)
  const [timelineZoom, setTimelineZoom] = useState<[number, number] | null>(null);
  const [takeoffZoom, setTakeoffZoom] = useState<[number, number] | null>(null);

  // A different simulation means a different data extent; stale windows would
  // land who-knows-where on it. Adjusted during render (not in an effect) so
  // the charts never paint a frame with the old window.
  const [prevSimulationId, setPrevSimulationId] = useState(currentSimulationId);
  if (prevSimulationId !== currentSimulationId) {
    setPrevSimulationId(currentSimulationId);
    setTimelineZoom(null);
    setTakeoffZoom(null);
  }

  // Full x-extent of the timeline data, mirroring the chart's own auto-domain
  // rules (display capped at 2050) so the zoom bar spans what the chart can show
  const timelineExtent = useMemo<[number, number]>(() => {
    let min = Infinity;
    let max = -Infinity;
    const scan = (year: number) => {
      if (year < min) min = year;
      if (year > max) max = year;
    };
    for (const points of Object.values(distributionData.milestones)) {
      for (const point of points) scan(point.year);
    }
    if (atcCdfData) {
      for (const points of Object.values(atcCdfData)) {
        for (const point of points) scan(point.year);
      }
    }
    if (!Number.isFinite(min) || !Number.isFinite(max) || max <= min) {
      return [2025, TIMELINE_DISPLAY_MAX_YEAR];
    }
    return [Math.floor(min), Math.min(Math.ceil(max), TIMELINE_DISPLAY_MAX_YEAR)];
  }, [distributionData, atcCdfData]);

  // The takeoff chart caps its display at 10 years after AC
  const TAKEOFF_EXTENT: [number, number] = [0, TAKEOFF_DISPLAY_MAX_YEARS];

  // AC-anchor-mode breakout toggle (only offered when breakout data exists)
  const [showModeBreakout, setShowModeBreakout] = useState(() => {
    return searchParams.get('breakout') === '1';
  });

  // Compare-section state is per-chart: switching the timeline chart to a
  // forecaster comparison shouldn't drag the takeoff chart along with it.
  // A URL with no `tkcmode` predates the split (or was hand-written), so the
  // takeoff chart reads the unprefixed params instead and old links keep
  // showing what they always showed. Only the initializers below may read this.
  const takeoffCompareParam = (key: string, legacyKey: string): string | null =>
    searchParams.get('tkcmode') === null ? searchParams.get(legacyKey) : searchParams.get(key);

  const [timelineCompareMode, setTimelineCompareMode] = useState<CompareMode>(
    () => parseCompareModeParam(searchParams.get('cmode'))
  );
  const [takeoffCompareMode, setTakeoffCompareMode] = useState<CompareMode>(
    () => parseCompareModeParam(takeoffCompareParam('tkcmode', 'cmode'))
  );
  const [timelineComparisonIds, setTimelineComparisonIds] = useState<string[]>(
    () => parseCsimParam(searchParams.get('csim'))
  );
  const [takeoffComparisonIds, setTakeoffComparisonIds] = useState<string[]>(
    () => parseCsimParam(takeoffCompareParam('tkcsim', 'csim'))
  );
  const [timelineForecasterShowType, setTimelineForecasterShowType] = useState<'model' | 'atc'>(
    () => parseShowTypeParam(searchParams.get('ctype'))
  );
  const [takeoffForecasterShowType, setTakeoffForecasterShowType] = useState<'model' | 'atc'>(
    () => parseShowTypeParam(takeoffCompareParam('tkctype', 'ctype'))
  );
  const [timelineShow, setTimelineShow] = useState(
    () => parseShowParam(searchParams.get('show'))
  );
  const [takeoffShow, setTakeoffShow] = useState(
    () => parseShowParam(takeoffCompareParam('tkshow', 'show'))
  );

  // Client state persists across primary-forecast navigation, so the current
  // simulation can end up in a stale comparison list - filter it out here
  // rather than mutating state
  const effectiveTimelineComparisonIds = useMemo(
    () => timelineComparisonIds.filter(id => id !== currentSimulationId),
    [timelineComparisonIds, currentSimulationId]
  );
  const effectiveTakeoffComparisonIds = useMemo(
    () => takeoffComparisonIds.filter(id => id !== currentSimulationId),
    [takeoffComparisonIds, currentSimulationId]
  );
  // Comparison datasets are looked up by simulation id, so a single fetch of
  // the union serves whichever chart asked for them
  const requestedComparisonIds = useMemo(
    () => Array.from(new Set([...effectiveTimelineComparisonIds, ...effectiveTakeoffComparisonIds])),
    [effectiveTimelineComparisonIds, effectiveTakeoffComparisonIds]
  );
  // Track the previously requested set to detect when it changes (may need server fetch)
  const prevCsimRef = useRef<string>(requestedComparisonIds.join(','));
  // Derived, not stored: a chart is waiting on comparison data exactly when one
  // of the simulations it asked for has no dataset yet. Storing this as state
  // meant setting it synchronously inside the URL-sync effect and clearing it
  // from a second effect that watched for the data to land.
  const isMissingData = (ids: string[]) =>
    ids.some(id => !comparisonDatasets.some(d => d.simulationId === id));
  const isLoadingTimelineComparison = isMissingData(effectiveTimelineComparisonIds);
  const isLoadingTakeoffComparison = isMissingData(effectiveTakeoffComparisonIds);

  // Build available simulations list for the dropdown, sorted by date (newest first), then Daniel first
  const availableSimulations = useMemo(() => {
    return [...simulations]
      .sort((a, b) => {
        const dateA = parseDate(a.date);
        const dateB = parseDate(b.date);

        // Primary sort: descending by date (newest first)
        if (dateB !== dateA) {
          return dateB - dateA;
        }

        // Secondary sort: Daniel first, then alphabetically
        if (a.forecaster === 'Daniel' && b.forecaster !== 'Daniel') return -1;
        if (b.forecaster === 'Daniel' && a.forecaster !== 'Daniel') return 1;
        return a.forecaster.localeCompare(b.forecaster);
      })
      .map(sim => ({ id: sim.id, label: sim.label }));
  }, [simulations]);

  // Sync state changes to URL
  // Uses window.history.replaceState for UI-only params (no server roundtrip)
  // Uses router.push only when csim changes (needs server fetch for new comparison data)
  useEffect(() => {
    // Skip the initial render to avoid unnecessary URL updates
    if (!isInitialized.current) {
      isInitialized.current = true;
      return;
    }

    // Use window.location.search instead of searchParams to avoid infinite loop
    const params = new URLSearchParams(window.location.search);

    // Update timeline param
    if (setsEqual(timelineVisibleMilestones, DEFAULT_TIMELINE_MILESTONES)) {
      params.delete('timeline');
    } else {
      params.set('timeline', encodeMilestoneParam(timelineVisibleMilestones));
    }

    // Update takeoff param
    if (setsEqual(takeoffVisibleMilestones, DEFAULT_TAKEOFF_MILESTONES)) {
      params.delete('takeoff');
    } else {
      params.set('takeoff', encodeMilestoneParam(takeoffVisibleMilestones));
    }

    // Update the compare params, one family per chart. Returns whether this
    // chart's Compare settings are all at their defaults (nothing written).
    const writeCompareParams = (
      prefix: string,
      mode: CompareMode,
      ids: string[],
      showType: 'model' | 'atc',
      show: { model: boolean; atc: boolean },
    ): boolean => {
      const cmodeKey = `${prefix}cmode`;
      const csimKey = `${prefix}csim`;
      const ctypeKey = `${prefix}ctype`;
      const showKey = `${prefix}show`;

      if (mode === 'model-vs-atc') {
        params.delete(cmodeKey);  // Default, don't need in URL
        params.delete(csimKey);
        params.delete(ctypeKey);
        // Update show param for model-vs-atc mode
        if (show.model && !show.atc) {
          params.set(showKey, 'model');
          return false;
        }
        if (!show.model && show.atc) {
          params.set(showKey, 'atc');
          return false;
        }
        params.delete(showKey);  // Default: both (both off is weird but treat as default)
        return true;
      }

      // Forecaster mode
      params.set(cmodeKey, 'forecaster');
      params.delete(showKey);  // Not used in forecaster mode
      if (ids.length > 0) {
        params.set(csimKey, ids.join(','));
      } else {
        params.delete(csimKey);
      }
      if (showType === 'atc') {
        params.set(ctypeKey, 'atc');
      } else {
        params.delete(ctypeKey);  // Default is model
      }
      return false;
    };

    const timelineCompareIsDefault = writeCompareParams(
      '', timelineCompareMode, effectiveTimelineComparisonIds, timelineForecasterShowType, timelineShow
    );
    const takeoffCompareIsDefault = writeCompareParams(
      'tk', takeoffCompareMode, effectiveTakeoffComparisonIds, takeoffForecasterShowType, takeoffShow
    );
    if (!timelineCompareIsDefault || !takeoffCompareIsDefault) {
      // Marks the link as carrying per-chart Compare state even when the takeoff
      // chart itself is at its defaults, so a reader never falls back to
      // applying the timeline's params to both charts.
      params.set('tkcmode', takeoffCompareMode);
    }

    // Update AC-anchor breakout param
    if (showModeBreakout && modeBreakoutData) {
      params.set('breakout', '1');
    } else {
      params.delete('breakout');
    }

    const newUrl = params.toString() ? `?${params.toString()}` : window.location.pathname;

    // Check if the requested set changed - if so, we may need a server fetch
    const csimKey = requestedComparisonIds.join(',');
    const csimChanged = prevCsimRef.current !== csimKey;
    prevCsimRef.current = csimKey;

    // A server fetch is only needed when a newly selected simulation has no
    // data yet; deselections can be handled entirely client-side. This is the
    // same predicate as the per-chart loading flags above.
    const needsFetch = isLoadingTimelineComparison || isLoadingTakeoffComparison;

    if (csimChanged && needsFetch) {
      // csim gained a new simulation - need server fetch for its comparison data
      router.push(newUrl, { scroll: false });
    } else {
      // UI-only change - just update URL without server request
      window.history.replaceState(null, '', newUrl);
    }
  }, [timelineVisibleMilestones, takeoffVisibleMilestones, timelineCompareMode, takeoffCompareMode,
      effectiveTimelineComparisonIds, effectiveTakeoffComparisonIds, requestedComparisonIds,
      isLoadingTimelineComparison, isLoadingTakeoffComparison, timelineForecasterShowType,
      takeoffForecasterShowType, timelineShow, takeoffShow, showModeBreakout, modeBreakoutData, router]);

  const handleToggleTimelineMilestone = useCallback((milestone: string) => {
    setTimelineVisibleMilestones(prev => {
      const newSet = new Set(prev);
      if (newSet.has(milestone)) {
        newSet.delete(milestone);
      } else {
        newSet.add(milestone);
      }
      return newSet;
    });
  }, []);

  const handleToggleTakeoffMilestone = useCallback((milestone: string) => {
    setTakeoffVisibleMilestones(prev => {
      const newSet = new Set(prev);
      if (newSet.has(milestone)) {
        newSet.delete(milestone);
      } else {
        newSet.add(milestone);
      }
      return newSet;
    });
  }, []);

  // Build arrival probabilities table data - TEMPORARILY HIDDEN
  // const arrivalHeaders = ['Milestone', ...arrivalProbabilities.years.map(y => `By Dec ${y}`)];
  // const arrivalRows = arrivalProbabilities.milestones.map(m => ({
  //   label: m.displayName,
  //   values: arrivalProbabilities.years.map(y => formatProbability(m.probabilities[y.toString()])),
  //   description: getMilestoneDescription(m.name),
  // }));

  // Build takeoff probabilities table data - TEMPORARILY HIDDEN
  // const takeoffHeaders = ['Condition', ...takeoffProbabilities.metrics.map(m => m.label)];
  // const takeoffRows = takeoffProbabilities.conditions.map(c => ({
  //   label: c.label,
  //   values: takeoffProbabilities.metrics.map(m => formatProbability(c.probabilities[m.key])),
  // }));

  // Memoize comparison data to avoid creating new arrays on every render
  // This prevents unnecessary re-renders of the chart components
  const timelineComparisonData = useMemo(() => {
    if (timelineCompareMode !== 'forecaster' || effectiveTimelineComparisonIds.length === 0) {
      return [];
    }
    return effectiveTimelineComparisonIds
      .map(id => comparisonDatasets.find(d => d.simulationId === id))
      .filter((d): d is ComparisonDataset => d !== undefined)
      .map(d => ({
        forecaster: d.forecaster,
        label: d.label,
        data: d.distributionData,
        atcCdfData: d.atcCdfData,
        atcStatistics: d.atcStatistics,
      }));
  }, [timelineCompareMode, effectiveTimelineComparisonIds, comparisonDatasets]);

  const takeoffComparisonData = useMemo(() => {
    if (takeoffCompareMode !== 'forecaster' || effectiveTakeoffComparisonIds.length === 0) {
      return [];
    }
    return effectiveTakeoffComparisonIds
      .map(id => comparisonDatasets.find(d => d.simulationId === id))
      .filter((d): d is ComparisonDataset => d !== undefined)
      .map(d => ({
        forecaster: d.forecaster,
        label: d.label,
        data: d.unconditionalTimingData,
        atcTakeoffCdfData: d.atcTakeoffCdfData,
      }));
  }, [takeoffCompareMode, effectiveTakeoffComparisonIds, comparisonDatasets]);

  return (
    <>
      {/* Executive Summary Tables - TEMPORARILY HIDDEN
      <section className="space-y-8 mb-12 mt-12">
        <h2 className="text-xl font-semibold text-gray-900">Summary</h2>
        <p className="text-base leading-relaxed text-gray-600">
          We estimated each parameter in the model, quantified our uncertainty as a probability distribution over each one, then simulated thousands of trajectories. Here are the results.
        </p>
          <div className="space-y-4">
          <ProbabilityTable
            title="How likely are short timelines?"
            headers={arrivalHeaders}
            rows={arrivalRows}
          />
          </div>
          <div className="space-y-4">
          <ProbabilityTable
            title="How likely are fast takeoffs?"
            headers={takeoffHeaders}
            rows={takeoffRows}
          />
          </div>
        <p className="text-sm text-gray-600">
          Learn more about our parameter estimates and reasoning <a href="https://docs.google.com/document/d/1ru6Okbxb6XuH18Cz8439sdQJazMV39hNxsWDokh97r0/edit?tab=t.0#heading=h.pt3lm912lfru" className="text-blue-600 hover:text-blue-800 underline" target="_blank" rel="noopener noreferrer">here</a>.
        </p>
      </section>
      */}

      {/* Main Milestone Distribution Chart */}
      <section className="space-y-8">
        <h2 className="font-mono font-semibold">Milestone arrival dates</h2>
        <div className="space-y-4">
          <p className="prose text-[14px]">
          The chart below shows how long we project it will take to achieve various AI milestones (toggle them on in the sidebar).
          The x-axis is the year the milestone is achieved, and the y-axis is the probability density at a point in time, expressed in the % chance the milestone would happen within a year at that density level.
          </p>
          <div className="flex flex-col md:flex-row gap-4 items-stretch md:items-start">
            <div className="flex-1 min-w-0">
              <MilestoneDistributionChart
                data={distributionData}
                title="Probability of Reaching AI Milestones"
                showCdf={timelineShowCdf}
                atcCdfData={atcCdfData}
                atcStatistics={atcStatistics}
                showAtcOverlay={timelineCompareMode === 'model-vs-atc' ? timelineShow.atc : timelineForecasterShowType === 'atc'}
                showModelOutput={timelineCompareMode === 'model-vs-atc' ? timelineShow.model : timelineForecasterShowType === 'model'}
                visibleMilestones={timelineVisibleMilestones}
                showGridlines={timelineShowGridlines}
                forecasterName={forecasterName}
                forecasterDate={forecasterDate}
                compareMode={timelineCompareMode}
                forecasterShowType={timelineForecasterShowType}
                comparisonData={timelineComparisonData}
                modeBreakoutData={modeBreakoutData}
                showModeBreakout={showModeBreakout && timelineCompareMode === 'model-vs-atc'}
                simulationId={currentSimulationId}
                zoomDomain={timelineZoom}
              />
              {/* Padding matches the chart margins so the bar lines up with the plot area */}
              <XAxisZoomBar
                min={timelineExtent[0]}
                max={timelineExtent[1]}
                value={timelineZoom}
                onChange={setTimelineZoom}
                step={1 / 12}
                minSpan={1}
                formatValue={formatYearMonthShort}
                className="pl-[60px] pr-[40px]"
              />
            </div>
            <TimelineChartSidebar
              visibleMilestones={timelineVisibleMilestones}
              onToggleMilestone={handleToggleTimelineMilestone}
              showCdf={timelineShowCdf}
              onToggleCdf={() => setTimelineShowCdf(prev => !prev)}
              showGridlines={timelineShowGridlines}
              onToggleGridlines={() => setTimelineShowGridlines(prev => !prev)}
              compareMode={timelineCompareMode}
              onCompareModeChange={setTimelineCompareMode}
              comparisonSimulationIds={effectiveTimelineComparisonIds}
              onComparisonSimulationsChange={setTimelineComparisonIds}
              availableSimulations={availableSimulations}
              currentSimulationId={currentSimulationId}
              forecasterShowType={timelineForecasterShowType}
              onForecasterShowTypeChange={setTimelineForecasterShowType}
              showModel={timelineShow.model}
              onToggleModel={() => setTimelineShow(prev => ({ ...prev, model: !prev.model }))}
              showAtc={timelineShow.atc}
              onToggleAtc={() => setTimelineShow(prev => ({ ...prev, atc: !prev.atc }))}
              hasAtcData={!!atcCdfData}
              isLoadingComparison={isLoadingTimelineComparison}
              hasModeBreakout={!!modeBreakoutData && modeBreakoutData.modes.length > 0}
              showModeBreakout={showModeBreakout}
              onToggleModeBreakout={() => setShowModeBreakout(prev => !prev)}
            />
          </div>
        </div>
        {timelinesNotes && forecasterName && (
          <details className="mt-4">
            <summary>
              {forecasterName}&apos;s notes on their all-things-considered forecast
            </summary>
            <div className="mt-4 space-y-4 [&_ul]:list-disc [&_ul]:pl-6 [&_ul]:space-y-4 [&_a]:text-blue-600 [&_a:hover]:text-blue-800 [&_a]:underline" dangerouslySetInnerHTML={{ __html: timelinesNotes }} />
          </details>
        )}
      </section>

      {/* Takeoff Speeds Chart */}
      <section className="space-y-8 mt-16">
        <h2 className="font-mono font-semibold">Time from coding automation to future milestones</h2>
        <p className="prose text-[14px]">
          The chart below shows how long we project it will take to reach various milestones after achieving AC (Automated Coder).
          The x-axis represents years after AC achievement, and the curves show the cumulative probability for when each subsequent milestone might be reached.
        </p>

        {unconditionalTimingData && Object.keys(unconditionalTimingData.milestones).length > 0 && (
          <div className="space-y-4">
            <div className="flex flex-col md:flex-row gap-4 items-stretch md:items-start">
              <div className="flex-1 min-w-0">
                <ConditionalMilestoneTimingChart
                  data={unconditionalTimingData}
                  title="Takeoff Speeds: Time from Automated Coder to Advanced AI"
                  maxTimeYears={10}
                  sharedYDomain={sharedYDomain}
                  showLegend={true}
                  showCdf={takeoffShowCdf}
                  atcTakeoffCdfData={(takeoffCompareMode === 'model-vs-atc' ? takeoffShow.atc : takeoffForecasterShowType === 'atc') ? atcTakeoffCdfData : null}
                  showModelOutput={takeoffCompareMode === 'model-vs-atc' ? takeoffShow.model : takeoffForecasterShowType === 'model'}
                  visibleMilestones={takeoffVisibleMilestones}
                  showGridlines={takeoffShowGridlines}
                  forecasterName={forecasterName}
                  forecasterDate={forecasterDate}
                  compareMode={takeoffCompareMode}
                  forecasterShowType={takeoffForecasterShowType}
                  comparisonData={takeoffComparisonData}
                  simulationId={currentSimulationId}
                  zoomDomain={takeoffZoom}
                />
                {/* Padding matches the chart margins so the bar lines up with the plot area */}
                <XAxisZoomBar
                  min={TAKEOFF_EXTENT[0]}
                  max={TAKEOFF_EXTENT[1]}
                  value={takeoffZoom}
                  onChange={setTakeoffZoom}
                  step={0.1}
                  minSpan={0.5}
                  formatValue={formatTakeoffYears}
                  className="pl-[60px] pr-[40px]"
                />
              </div>
              <TakeoffChartSidebar
                visibleMilestones={takeoffVisibleMilestones}
                onToggleMilestone={handleToggleTakeoffMilestone}
                showCdf={takeoffShowCdf}
                onToggleCdf={() => setTakeoffShowCdf(prev => !prev)}
                showGridlines={takeoffShowGridlines}
                onToggleGridlines={() => setTakeoffShowGridlines(prev => !prev)}
                compareMode={takeoffCompareMode}
                onCompareModeChange={setTakeoffCompareMode}
                comparisonSimulationIds={effectiveTakeoffComparisonIds}
                onComparisonSimulationsChange={setTakeoffComparisonIds}
                availableSimulations={availableSimulations}
                currentSimulationId={currentSimulationId}
                forecasterShowType={takeoffForecasterShowType}
                onForecasterShowTypeChange={setTakeoffForecasterShowType}
                showModel={takeoffShow.model}
                onToggleModel={() => setTakeoffShow(prev => ({ ...prev, model: !prev.model }))}
                showAtc={takeoffShow.atc}
                onToggleAtc={() => setTakeoffShow(prev => ({ ...prev, atc: !prev.atc }))}
                hasAtcData={!!atcTakeoffCdfData}
                isLoadingComparison={isLoadingTakeoffComparison}
              />
            </div>
          </div>
        )}

      </section>
    </>
  );
}
