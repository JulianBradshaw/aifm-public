'use client';

import { useCallback, useEffect, useMemo, useRef, useState, ReactNode } from 'react';
import { createPortal } from 'react-dom';
import { usePathname, useRouter } from 'next/navigation';
import { CustomLineChart, DataPoint } from './CustomLineChart';
import { ChartSyncProvider } from './ChartSyncContext';
import { ParameterSlider } from './ParameterSlider';
import { SliderWithThumbValue } from './SliderWithThumbValue';
import { ParameterHoverProvider } from './ParameterHoverContext';
import { WithChartTooltip } from './ChartTitleTooltip';
import { tooltipBoxStyle, tooltipHeaderStyle, tooltipValueStyle } from './chartTooltipStyle';
import { calculateTicks } from '@/utils/chartUtils';
import { computePlanA, PlanAApiError } from '@/lib/planAApi';
import { DEFAULT_PARAMETERS, ParametersType } from '@/constants/parameters';
import { convertParametersToAPIFormat, ParameterRecord } from '@/utils/monteCarlo';
import { ParameterConfig, AdvancedSections } from './AdvancedSections';
import { formatYearMonth } from '@/utils/formatting';
import {
  PlanADefaultsResponse,
  PlanAKnobs,
  PlanAResponse,
  PlanASeriesRow,
  PlanASilverBullet,
} from '@/app/types';
import {
  COVERT_CHART_EXPLANATIONS,
  COVERT_KNOB_EXPLANATIONS,
  MILESTONE_FULL_NAMES,
  MILESTONE_EXPLANATIONS,
} from '@/constants/chartExplanations';
import {
  encodePlanAStateToParams,
  decodePlanAStateFromParams,
  PlanATrajectoryKey,
  COVERT_DEFAULT_HIDDEN,
} from '@/utils/urlState';
import { BREAKDOWN_DEFAULTS } from '@/constants/planASeeds';

// Embed mode (the ai-2030 Plan A site iframes /plan-a?embed=1). Read from the live URL
// inside effects only — never during render — so the page stays statically prerenderable
// without hydration mismatches. useSearchParams() would force the whole tree client-side.
const isEmbedNow = () =>
  typeof window !== 'undefined' && new URLSearchParams(window.location.search).get('embed') === '1';

// Pipeline / AI-2030 palette (scripts/plotting_utils/style.py): default = aifuturesmodel
// green, legal = pipeline navy/blue, covert = pipeline red. Consistent across all charts.
const COLOR = {
  default: '#2A623D',   // aifuturesmodel green
  legalDisc: '#1e3a5f', // AI2030_NAVY (legal discovered — dotted upper envelope)
  legalUsed: '#1e3a5f', // AI2030_NAVY (legal used/deployed — solid primary)
  covert: '#5c1a1a',    // AI2030_RED (covert)
  leading: '#3f6634',   // AI2030_OLIVE (leading company after the deal breakdown)
  acLine: '#B0B0B0',
};

const DEBOUNCE_MS = 350;

// ---------------------------------------------------------------------------
// The 5 headline MODEL parameters shown on the front page (reusing the same
// ParameterSlider component, so they look + behave identically here).
// ---------------------------------------------------------------------------
interface ModelParamDef {
  paramName: keyof ParametersType & string;
  label: string;
  explanation: string;
  fallbackMin?: number;
  fallbackMax?: number;
  customMin?: number;
  customMax?: number;
  step: number;
  decimalPlaces?: number;
  format?: (v: number) => string;
  useLogScale?: boolean;
}
// Same labels + explanations + bounds as the front-page "Key Parameters" panel.
const MODEL_PARAMS: ModelParamDef[] = [
  { paramName: 'inv_compute_anchor_exp_cap', label: 'Software progress slowdown from 10× less experiment compute', explanation: 'The factor by which research progress slows if experiment compute is cut by 10× — i.e. how much marginal experiment compute buys. Higher ⇒ compute matters more.', fallbackMin: 1, fallbackMax: 10, step: 0.1, decimalPlaces: 1, format: (v) => `${v.toFixed(1)}×`, useLogScale: true },
  { paramName: 'software_progress_rate_at_reference_year', label: 'Current rate of software progress (OOMs/yr)', explanation: 'Orders of magnitude of software-efficiency progress per year as of the 2024 reference year — the present-day pace of algorithmic improvement.', fallbackMin: 0.05, fallbackMax: 10, step: 0.05, decimalPlaces: 2, format: (v) => `${v.toFixed(2)} OOM/yr`, useLogScale: true },
  { paramName: 'ted_ai_m2b', label: 'Median→top-human jumps above SAR to reach TED-AI', explanation: 'How many median-to-top-human research-taste jumps above SAR are needed to reach TED-AI. Higher ⇒ TED-AI is farther beyond SAR.', fallbackMin: 0.5, fallbackMax: 5, step: 0.05, decimalPlaces: 2, format: (v) => `${v.toFixed(2)}`, useLogScale: true },
  { paramName: 'median_to_top_taste_multiplier', label: "Ratio of top to median researchers' value per experiment", explanation: "Ratio of the top human researcher's research taste to the median human researcher's research taste, i.e. how much more value per experiment the top researcher gets holding capacity to implement experiments fixed.", fallbackMin: 1.1, fallbackMax: 20.0, step: 0.1, format: (v) => `${v.toFixed(2)}x`, useLogScale: true },
  { paramName: 'ai_research_taste_slope', label: 'How quickly AIs improve at research taste', explanation: "Each time effective compute increases by as many orders of magnitude as it did in 2025, by how many standard deviations (SDs) does the AI's research taste increase (median to best AGI company researcher requires 3 SDs)?", fallbackMin: 0.1, fallbackMax: 10.0, step: 0.1, decimalPlaces: 1, format: (v) => `${v.toFixed(1)} SDs/2025-effective-FLOP-growth`, useLogScale: true },
];

// The curated headline knobs surfaced via the Key-parameters tabs (the full knob set
// still lives under Advanced), grouped by which project they parameterize. The
// covert-size param is injected at render time since it depends on compute_mode
// (frontier-relative fraction vs absolute H100e).
interface PlanAKeyParamDef {
  key: string; label: string; min: number; max: number; step: number;
  log?: boolean; format?: (v: number) => string; mode?: string;
}
const LEGAL_KEY_PARAMS: PlanAKeyParamDef[] = [
  // max 5 per row — setup_period_years intentionally lives only under Advanced settings
  // Negative = the deal lands AFTER AC (sim handles it; degenerates gracefully if the
  // un-paused world has already crossed TED-AI by then).
  { key: 'years_before_ac', label: 'How long before AC does the deal/pause land?', min: -2, max: 3, step: 0.05, format: (v) => `${v.toFixed(2)} yr` },
  { key: 'setup_sw_gain_progress_months', label: 'How large is the radical-transparency windfall?', min: 0, max: 24, step: 0.5, format: (v) => `${v.toFixed(1)} progress-mo` },
  { key: 'legal_scaling_duration', label: 'How soon does the consortium aim to reach TED-AI?', min: 0.5, max: 15, step: 0.25, format: (v) => `${v.toFixed(2)} yr` },
  { key: 'alignment_tax_progress_years', label: 'How large is the safety tax the consortium pays at TED-AI?', min: 0, max: 3, step: 0.05, format: (v) => `${v.toFixed(2)} yr` },
  { key: 'post_scaling_externalities_rate', label: 'How quickly do capabilities externalities accumulate after TED-AI?', min: 0, max: 2, step: 0.05, format: (v) => `${v.toFixed(2)} OOM/yr` },
];
const COVERT_KEY_PARAMS: PlanAKeyParamDef[] = [
  { key: 'frac_stolen', label: 'What % of consortium software improvements leak to covert projects?', min: 0, max: 1, step: 0.01, format: (v) => `${(v * 100).toFixed(0)}%` },
  { key: 'dormancy_years', label: 'How long does the covert project lie dormant before starting construction?', min: 0, max: 6, step: 0.1, format: (v) => `${v.toFixed(1)} yr` },
  { key: 'construction_years', label: 'How long does the covert datacenter take to build?', min: 0, max: 6, step: 0.1, format: (v) => `${v.toFixed(1)} yr` },
];
const BREAKDOWN_KEY_PARAMS: PlanAKeyParamDef[] = [
  { key: 'deal_breakdown_years_after_pause_end', label: 'How long after the pause ends does the deal dissolve?', min: 0, max: 12, step: 0.25, format: (v) => `${v.toFixed(2)} yr` },
  // Only the reduction variant matching breakdown_compute_mode is shown.
  { key: 'breakdown_compute_reduction', label: "Leading company's compute as a share of the pre-deal frontier?", mode: 'frontier_at_pause', min: 0.01, max: 1, step: 0.01, format: (v) => `${(v * 100).toFixed(0)}%` },
  { key: 'breakdown_compute_reduction', label: "What % of the world's compute does the leading company use after the dissolution?", mode: 'world_at_breakdown', min: 0.01, max: 1, step: 0.01, format: (v) => `${(v * 100).toFixed(0)}%` },
  { key: 'post_breakdown_compute_growth_rate', label: "How fast does the leading company's compute grow after the dissolution?", min: 0, max: 2, step: 0.05, format: (v) => `${v.toFixed(2)} OOM/yr` },
];
// Is the compute built during the deal destroyed at dissolution (frontier_at_pause: the
// leading company restarts from a share of the pre-deal frontier) or preserved
// (world_at_breakdown: it takes a share of the world's compute at the collapse)?
const BREAKDOWN_MODE_SELECT: SelectDef = {
  key: 'breakdown_compute_mode',
  label: 'Post-dissolution compute basis',
  options: ['frontier_at_pause', 'world_at_breakdown'],
};
// Chart-visibility toggles (Advanced settings): one per trajectory, display-only.
const TRAJECTORY_TOGGLES: Array<{ key: PlanATrajectoryKey; label: string; color: string }> = [
  { key: 'covert', label: 'Covert project', color: COLOR.covert },
  { key: 'consortium', label: 'Consortium', color: COLOR.legalUsed },
  { key: 'leading', label: 'Leading co. (post-dissolution)', color: COLOR.leading },
  { key: 'default', label: 'Default (un-paused)', color: COLOR.default },
];

// Reference-run milestones (from /api/compute), used for the green milestone dots.
type MilestoneMapLite = Record<string, { time?: number; effective_compute_ooms?: number }>;
// Which milestones to dot on the capability chart, and their display labels.
const MILESTONE_SEQUENCE: Array<{ key: string; label: string }> = [
  { key: 'AC', label: 'AC' },
  { key: 'SAR-level-experiment-selection-skill', label: 'SAR' },
  { key: 'SIAR-level-experiment-selection-skill', label: 'SIAR' },
  { key: 'TED-AI', label: 'TED-AI' },
  { key: 'ASI', label: 'ASI' },
];

// ---------------------------------------------------------------------------
// Covert knob control definitions (the curated subset, now under "Advanced").
// min/max/step are UI bounds only — the backend validates and returns a clear 400.
// ---------------------------------------------------------------------------
interface SliderDef { key: string; label: string; min: number; max: number; step: number; format?: (v: number) => string; mode?: string; }
interface SelectDef { key: string; label: string; options: string[]; }

// Consortium / deal-wide knobs: the pause, the legal project's scaling strategy and caps.
const CONSORTIUM_SELECTS: SelectDef[] = [
  { key: 'legal_strategy', label: 'Consortium strategy', options: ['target_year_actual_hw', 'target_year', 'target_year_capped_sw'] },
  { key: 'legal_mode', label: 'Consortium physics mode', options: ['capped', 'backsolve'] },
];
const CONSORTIUM_SLIDERS: SliderDef[] = [
  { key: 'years_before_ac', label: 'Pause lead (yr before AC)', min: -2, max: 3, step: 0.05 },
  { key: 'setup_period_years', label: 'Setup pause length (yr)', min: 0, max: 3, step: 0.05 },
  { key: 'setup_sw_gain_progress_months', label: 'Transparency windfall (progress-mo)', min: 0, max: 24, step: 0.5 },
  { key: 'alignment_tax_progress_years', label: 'Alignment tax (progress-yr)', min: 0, max: 3, step: 0.05 },
  { key: 'legal_scaling_duration', label: 'Consortium scaling duration (yr)', min: 0.5, max: 15, step: 0.25 },
  { key: 'post_scaling_externalities_rate', label: 'Post-scaling externalities (OOM/yr)', min: 0, max: 2, step: 0.05 },
  { key: 'min_sw_frac_of_tsp', label: 'Min SW growth (frac of TSP rate)', min: 0, max: 1, step: 0.05 },
  { key: 'max_hw_rate', label: 'Max HW rate (OOM/yr)', min: 0, max: 2, step: 0.05 },
  { key: 'max_sw_rate', label: 'Max SW rate (OOM/yr, capped_sw)', min: 0, max: 3, step: 0.05 },
];
// Covert-project knobs: sizing (the compute_mode-gated slider is rendered separately),
// software theft, and the build timeline.
const COVERT_SELECTS: SelectDef[] = [
  { key: 'compute_mode', label: 'Compute mode', options: ['frontier_relative', 'absolute'] },
];
const COVERT_SLIDERS: SliderDef[] = [
  { key: 'frac_stolen', label: 'Fraction stolen', min: 0, max: 1, step: 0.01 },
  { key: 'dormancy_years', label: 'Dormancy before construction (yr)', min: 0, max: 6, step: 0.1 },
  { key: 'construction_years', label: 'Datacenter construction (yr)', min: 0, max: 6, step: 0.1 },
];
// Detection knobs — grouped into their own "Detection" subsection (optimal_site_mw is rendered
// specially since its range comes from the detection_grids).
const DETECTION_SLIDERS: SliderDef[] = [
  { key: 'facility_w_per_h100e', label: 'Facility W per H100e', min: 100, max: 1500, step: 10, format: (v) => v.toFixed(0) },
];
// Deal-breakdown knobs (gated by enable_deal_breakdown). Absent from the covert yaml, so
// they're seeded here rather than by /api/plan-a/defaults.
const BREAKDOWN_SLIDERS: SliderDef[] = [
  { key: 'deal_breakdown_years_after_pause_end', label: 'Dissolution time (yr after pause end)', min: 0, max: 12, step: 0.25, format: (v) => `${v.toFixed(2)} yr` },
  // Only the reduction variant matching breakdown_compute_mode is shown.
  { key: 'breakdown_compute_reduction', label: 'Leading-co compute (frac of frontier at pause)', mode: 'frontier_at_pause', min: 0.01, max: 1, step: 0.01, format: (v) => `${(v * 100).toFixed(0)}%` },
  { key: 'breakdown_compute_reduction', label: 'Leading-co compute (frac of world at dissolution)', mode: 'world_at_breakdown', min: 0.01, max: 1, step: 0.01, format: (v) => `${(v * 100).toFixed(0)}%` },
  { key: 'post_breakdown_compute_growth_rate', label: 'Post-dissolution compute growth (OOM/yr)', min: 0, max: 2, step: 0.05, format: (v) => `${v.toFixed(2)} OOM/yr` },
];
const SB_SLIDERS: Array<{ key: keyof PlanASilverBullet; label: string; min: number; max: number; step: number }> = [
  { key: 'p_mid_pause_to_ted', label: 'P at midpoint', min: 0.01, max: 0.9, step: 0.01 },
  { key: 'p_used_ted_plus_d', label: 'P at used→TED + Δ', min: 0.02, max: 0.95, step: 0.01 },
  { key: 'delta_years', label: 'Δ years', min: 0, max: 6, step: 0.25 },
  { key: 'ceiling', label: 'Ceiling', min: 0.05, max: 1, step: 0.01 },
];

// ---------------------------------------------------------------------------
// Small UI atoms (mirror the SimpleSlider pattern in PlaygroundClient).
// ---------------------------------------------------------------------------
function Select({ label, value, options, onChange, title }: {
  label: string; value: string; options: string[]; onChange: (v: string) => void; title?: string;
}) {
  return (
    <div className="space-y-1" title={title}>
      <div className="text-xs text-gray-600">{label}</div>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="w-full text-xs font-mono border border-gray-300 rounded px-2 py-1 bg-white text-gray-900"
      >
        {options.map((o) => <option key={o} value={o}>{o}</option>)}
      </select>
    </div>
  );
}

// A covert-knob slider that renders identically to the takeoff ParameterSlider columns:
// the same question label on top and the same SliderWithThumbValue control beneath
// (thumb-value, default-value tick, optional log scale). `onChange` updates live while
// dragging; `onCommit` recomputes on release. `defaultValue` draws the baseline tick.
function PlanAParamSlider({ label, explanation, value, min, max, step, log, format, defaultValue, disabled, onChange, onCommit }: {
  label: string; explanation?: string; value: number; min: number; max: number; step: number;
  log?: boolean; format?: (v: number) => string; defaultValue?: number; disabled?: boolean;
  onChange: (v: number) => void; onCommit: (v: number) => void;
}) {
  const lastRef = useRef(value);
  return (
    <div className={`relative flex w-full flex-col justify-between ${disabled ? 'opacity-40 pointer-events-none' : ''}`}>
      <label className="block w-full text-xs font-medium text-foreground mb-2 leading-snug">
        {explanation ? (
          <WithChartTooltip explanation={explanation} fullWidth>{label}</WithChartTooltip>
        ) : label}
      </label>
      <div className="space-y-0.5 relative">
        <SliderWithThumbValue
          value={value}
          min={min}
          max={max}
          step={step}
          onChange={(v) => { lastRef.current = v; onChange(v); }}
          onMouseUp={() => onCommit(lastRef.current)}
          onTouchEnd={() => onCommit(lastRef.current)}
          displayValue=""
          formatValue={format}
          useLogScale={log}
          defaultValue={defaultValue}
        />
      </div>
    </div>
  );
}

function LegendSwatch({ color, label, dashed }: { color: string; label: string; dashed?: boolean }) {
  return (
    <span className="inline-flex items-center gap-1.5 text-xs text-gray-700">
      <svg width="20" height="8"><line x1="0" y1="4" x2="20" y2="4" stroke={color} strokeWidth="2.5" strokeDasharray={dashed ? '4 3' : undefined} /></svg>
      {label}
    </span>
  );
}

// ---------------------------------------------------------------------------
// Helpers: merge parallel-array series onto a union-of-x DataPoint[] (lines
// skip nulls, so differing grids compose without interpolation).
// ---------------------------------------------------------------------------
function buildChartData(series: Array<{ rows: PlanASeriesRow[]; map: Record<string, number> }>): DataPoint[] {
  const byX = new Map<number, DataPoint>();
  for (const s of series) {
    for (const row of s.rows) {
      const x = row[0];
      if (x == null) continue;
      let pt = byX.get(x);
      if (!pt) { pt = { x }; byX.set(x, pt); }
      for (const [key, col] of Object.entries(s.map)) {
        const v = row[col];
        if (v != null) pt[key] = v;
      }
    }
  }
  return Array.from(byX.values()).sort((a, b) => a.x - b.x);
}

function extent(data: DataPoint[], keys: string[]): [number, number] | null {
  let lo = Infinity, hi = -Infinity;
  for (const pt of data) {
    for (const k of keys) {
      const v = pt[k];
      if (typeof v === 'number' && isFinite(v)) { if (v < lo) lo = v; if (v > hi) hi = v; }
    }
  }
  return isFinite(lo) && isFinite(hi) ? [lo, hi] : null;
}

function fmtYear(v: number | null | undefined): string {
  return v == null ? '—' : v.toFixed(1);
}

// Fetch the reference ("default" project) time series from /api/compute — the same
// endpoint the front page uses — so the mini-metric charts can overlay it against the
// covert project. Best-effort: returns [] on failure (covert charts still render).
async function fetchDefaultSeries(
  parameters: Record<string, number | string | boolean>,
  signal: AbortSignal,
): Promise<{ series: Array<Record<string, number | null>>; milestones: MilestoneMapLite }> {
  try {
    const res = await fetch('/api/compute', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ parameters, time_range: [2017, 2100], initial_progress: 0.0 }),
      signal,
    });
    if (!res.ok) return { series: [], milestones: {} };
    const data = await res.json();
    return {
      series: (data?.time_series as Array<Record<string, number | null>>) ?? [],
      milestones: (data?.milestones as MilestoneMapLite) ?? {},
    };
  } catch {
    return { series: [], milestones: {} };
  }
}

// Mini-metric definitions: the metric field (shared key name across default + covert),
// a title, log/linear axis, and an optional value formatter. `defField` overrides which
// key the DEFAULT-project line reads (for covert-only fields whose natural default-project
// analog has a different name).
const MINI_METRICS: Array<{ field: string; defField?: string; title: string; log: boolean; format?: (v: number) => string }> = [
  { field: 'effectiveCompute', title: 'Effective compute (OOMs)', log: false, format: (v) => v.toFixed(1) },
  { field: 'aiSwProgressMultRefPresentDay', title: 'AI software R&D uplift', log: true, format: (v) => `${v.toFixed(1)}×` },
  { field: 'automationFraction', title: 'Coding automation fraction', log: false, format: (v) => `${(v * 100).toFixed(0)}%` },
  { field: 'etsp', title: 'ETSP (OOMs/yr)', log: false, format: (v) => v.toFixed(1) },
  { field: 'softwareEfficiency', title: 'Software efficiency (OOMs)', log: false, format: (v) => v.toFixed(1) },
  { field: 'trainingSystemPerformance', title: 'Training system perf (OOMs)', log: false, format: (v) => v.toFixed(1) },
  { field: 'researchEffort', title: 'Research effort (realized)', log: true, format: (v) => (v >= 1000 || v < 0.01 ? v.toExponential(0) : v.toPrecision(2)) },
  // Covert OWN-R&D quantities (theft + boost stripped). The covert line is the covert's
  // indigenous output; the default line is the default project's research effort / software
  // rate (itself theft-free), so the two are an apples-to-apples indigenous comparison.
  { field: 'researchEffortIndigenous', defField: 'researchEffort', title: 'Research effort (covert indigenous)', log: true, format: (v) => (v >= 1000 || v < 0.01 ? v.toExponential(0) : v.toPrecision(2)) },
  { field: 'swRateIndigenous', defField: 'softwareProgressRate', title: 'SW progress rate (covert indigenous, OOMs/yr)', log: true, format: (v) => v.toFixed(2) },
  // Capability-indexed (pure functions of E), shared field names with the default series.
  { field: 'aiCodingLaborMultRefPresentDay', title: 'AI coding labor mult (ref present-day)', log: true, format: (v) => (v >= 1000 || v < 0.01 ? v.toExponential(0) : v.toPrecision(2)) },
  { field: 'aiResearchTaste', title: 'AI research taste', log: true, format: (v) => v.toFixed(2) },
];

// Linear-interpolate a sorted [x,y] series at x; null outside its range (no extrapolation).
function interpAt(pts: Array<[number, number]>, x: number): number | null {
  if (pts.length === 0 || x < pts[0][0] || x > pts[pts.length - 1][0]) return null;
  for (let i = 1; i < pts.length; i++) {
    if (pts[i][0] >= x) {
      const [x0, y0] = pts[i - 1];
      const [x1, y1] = pts[i];
      return x1 === x0 ? y1 : y0 + (y1 - y0) * ((x - x0) / (x1 - x0));
    }
  }
  return pts[pts.length - 1][1];
}

// Resample a series' rows onto an exact 1/12-year grid via linear interpolation (for the
// CSV export's "monthly" option). The grid is computed in integer months so boundaries are
// float-exact. Each column interpolates independently over the rows where it has finite
// values and stays blank outside that coverage (no extrapolation) — so blocks whose
// columns span different windows keep their blanks.
function resampleMonthly(
  rows: Array<Record<string, number | null | undefined>>,
): Array<Record<string, number | null | undefined>> {
  const pts = rows
    .filter((r) => typeof r.year === 'number' && Number.isFinite(r.year))
    .sort((a, b) => (a.year as number) - (b.year as number));
  if (pts.length < 2) return pts;
  const cols = new Set<string>();
  for (const p of pts) for (const k of Object.keys(p)) if (k !== 'year') cols.add(k);
  const colPts: Record<string, Array<[number, number]>> = {};
  for (const c of cols) {
    colPts[c] = pts
      .filter((p) => typeof p[c] === 'number' && Number.isFinite(p[c] as number))
      .map((p) => [p.year as number, p[c] as number]);
  }
  const m0 = Math.ceil((pts[0].year as number) * 12 - 1e-9);
  const m1 = Math.floor((pts[pts.length - 1].year as number) * 12 + 1e-9);
  const out: Array<Record<string, number | null | undefined>> = [];
  for (let m = m0; m <= m1; m++) {
    const rec: Record<string, number | null | undefined> = { year: m / 12 };
    for (const c of cols) rec[c] = interpAt(colPts[c], m / 12);
    out.push(rec);
  }
  return out;
}

function MiniMetricChart({ def, cov, legal, leading, field, defField, legalField, title, logScale, format, displayEndYear,
                          covLabel = 'Covert', legalLabel = 'Consortium used', defLabel = 'Default', hidden, loading }: {
  def: Array<Record<string, number | null>>;
  cov?: Array<Record<string, number | null>>;
  legal?: Array<Record<string, number | null>>;
  // Leading company post-deal-breakdown — its own series, rendered olive.
  leading?: Array<Record<string, number | null>>;
  field: string;
  defField?: string;
  legalField?: string;
  title: string;
  logScale: boolean;
  format?: (v: number) => string;
  displayEndYear: number;
  covLabel?: string;
  legalLabel?: string;
  defLabel?: string;
  // Trajectory-visibility toggles; hidden series are treated as absent. Omit for charts
  // whose def/legal/cov slots aren't trajectories (e.g. the detection-survival breakout).
  hidden?: ReadonlySet<PlanATrajectoryKey>;
  // First-compute in flight: empty charts render a pulsing skeleton instead of "no data".
  loading?: boolean;
}) {
  const START_YEAR = 2026;
  const toPts = (src: Array<Record<string, number | null>>, key: string): Array<[number, number]> =>
    src
      .filter((p) => typeof p.year === 'number' && p.year >= START_YEAR && p.year <= displayEndYear && p[key] != null)
      .map((p) => [p.year as number, p[key] as number] as [number, number])
      .sort((a, b) => a[0] - b[0]);
  const defPts = hidden?.has('default') ? [] : toPts(def, defField ?? field);
  const covPts = cov && !hidden?.has('covert') ? toPts(cov, field) : [];
  const legalPts = legal && !hidden?.has('consortium') ? toPts(legal, legalField ?? field) : [];
  const leadPts = leading && !hidden?.has('leading') ? toPts(leading, field) : [];
  // Interpolate every present series onto the union of their x's, so each point carries
  // default, legal AND covert (where each exists) — the tooltip then shows all at once.
  // The legal/leading series are bounded to their own time ranges so interpolation
  // clamping can't extend either across the breakdown.
  const xsAll = Array.from(new Set([
    ...defPts.map((p) => p[0]), ...legalPts.map((p) => p[0]),
    ...leadPts.map((p) => p[0]), ...covPts.map((p) => p[0]),
  ])).sort((a, b) => a - b);
  const inRange = (pts: Array<[number, number]>, x: number) =>
    pts.length > 0 && x >= pts[0][0] && x <= pts[pts.length - 1][0];
  const data: DataPoint[] = xsAll.map((x) => ({
    x,
    def: defPts.length ? interpAt(defPts, x) : null,
    legal: inRange(legalPts, x) ? interpAt(legalPts, x) : null,
    lead: inRange(leadPts, x) ? interpAt(leadPts, x) : null,
    cov: covPts.length ? interpAt(covPts, x) : null,
  }));

  // The covert/legal series set the y-range; the default ('def') series is still plotted
  // but EXCLUDED from the domain — it often explodes (e.g. uplift / research effort) and
  // would otherwise blow up the axis and flatten the covert/legal lines. Fall back to the
  // default only when neither covert nor legal is present.
  const domainKeys = [
    ...(legalPts.length ? ['legal'] : []), ...(leadPts.length ? ['lead'] : []),
    ...(covPts.length ? ['cov'] : []),
  ];
  const ext = extent(data, domainKeys.length ? domainKeys : (defPts.length ? ['def'] : []));
  if (!ext || data.length === 0) {
    // Main-page skeleton treatment while the first compute is in flight.
    if (loading) {
      return (
        <div className="shrink-0 w-[250px] h-[150px] bg-gray-100 rounded animate-pulse flex items-center justify-center">
          <div className="w-6 h-6 border-2 border-gray-300 border-t-gray-500 rounded-full animate-spin" />
        </div>
      );
    }
    return (
      <div className="shrink-0 w-[250px] h-[150px] flex items-center justify-center text-[10px] text-gray-400 border border-gray-100 rounded">
        {title}: no data
      </div>
    );
  }
  let [lo, hi] = ext;
  if (logScale) { lo = Math.max(lo, 1e-6); if (hi <= lo) hi = lo * 10; }
  else { const pad = (hi - lo) * 0.08 || Math.abs(hi) * 0.08 || 1; lo -= pad; hi += pad; }
  const yDomain: [number, number] = [lo, hi];
  const yTicks = calculateTicks(yDomain, { targetCount: 4 }, logScale ? 'log' : 'linear');
  const yFmt = (v: number) => {
    if (logScale) return v >= 1000 || v < 0.01 ? v.toExponential(0) : String(+v.toPrecision(2));
    return format ? format(v) : v.toFixed(1);
  };

  return (
    <div className="shrink-0 w-[250px]">
      <div className="text-[10px] font-system-mono text-gray-600 mb-0.5 truncate" title={title}>{title}</div>
        <CustomLineChart
          data={data}
          width={250}
          height={150}
          margin={{ top: 10, right: 14, left: 8, bottom: 36 }}
          xDomain={[START_YEAR, displayEndYear]}
          xTickCount={3}
          xTickFormatter={(v) => v.toFixed(0)}
          yDomain={yDomain}
          yScale={logScale ? 'log' : 'linear'}
          yTicks={yTicks}
          yTickFormatter={(v) => yFmt(v as number)}
          lines={[
            // Match the main race chart: default = faint green, legal used = navy, covert =
            // red, post-breakdown leading company = olive.
            ...(defPts.length ? [{ dataKey: 'def', stroke: COLOR.default, strokeWidth: 1.5, strokeOpacity: 0.3 }] : []),
            ...(legalPts.length ? [{ dataKey: 'legal', stroke: COLOR.legalUsed, strokeWidth: 2.4 }] : []),
            ...(leadPts.length ? [{ dataKey: 'lead', stroke: COLOR.leading, strokeWidth: 2.4 }] : []),
            ...(covPts.length ? [{ dataKey: 'cov', stroke: COLOR.covert, strokeWidth: 2.4 }] : []),
          ]}
          tooltip={(p) => {
            // Use the chart's own value formatter (e.g. K/M/B/T for H100e) for numeric
            // values; fall back to the generic formatter (which also renders null as '—').
            const tfmt = (v: number | null | undefined | string) => (typeof v === 'number' && format ? format(v) : fmtNum(v));
            return (
              <ChartTip year={p.x as number} rows={[
                ...(covPts.length ? [[covLabel, tfmt(p.cov)] as [string, string]] : []),
                ...(legalPts.length ? [[legalLabel, tfmt(p.legal)] as [string, string]] : []),
                ...(leadPts.length ? [['Leading co.', tfmt(p.lead)] as [string, string]] : []),
                ...(defPts.length ? [[defLabel, tfmt(p.def)] as [string, string]] : []),
              ]} />
            );
          }}
        />
    </div>
  );
}

// Stacked-area decomposition mini (cumulative OOMs since pause), mirroring the pipeline's
// oom_cumulative figures. `rows` = [t, layer0, layer1, ...] (each cumulative-since-pause);
// `layers` are bottom→top. Rendered via CustomLineChart fills drawn back-to-front.
const STACK = {
  hw: '#4f6d8c',        // hardware (ΔTSP)
  windfall: '#e3b23c',  // transparency windfall
  mandatory: '#6a9b8f', // mandatory-minimum SW floor (frac of TSP growth)
  own: '#b5651d',       // own R&D / used (deployed) software
  stolen: '#7d97bf',    // stolen software
  tax: '#5c1a1a',       // alignment tax
};
// Monotone-x linear interpolator (clamps at the ends). Used to map a trajectory's effective
// compute → the present-day-referenced AI-software-R&D uplift via the default run's (E, uplift)
// curve — exactly how the backend derives the covert's uplift.
function makeInterp(xs: number[], ys: number[]): (x: number) => number {
  const n = xs.length;
  return (x: number) => {
    if (n === 0) return NaN;
    if (x <= xs[0]) return ys[0];
    if (x >= xs[n - 1]) return ys[n - 1];
    let lo = 0, hi = n - 1;
    while (hi - lo > 1) {
      const mid = (lo + hi) >> 1;
      if (xs[mid] <= x) lo = mid; else hi = mid;
    }
    const span = xs[hi] - xs[lo] || 1;
    return ys[lo] + ((x - xs[lo]) / span) * (ys[hi] - ys[lo]);
  };
}

// `layers` are bottom→top stacking order; each layer's `idx` is the column it reads from
// the row (default = position+1). `big` renders a headline-sized chart (responsive width,
// taller) instead of a 250px mini.
function StackedAreaChart({ rows, layers, title, displayEndYear, startYear = 2026, big }: {
  rows: PlanASeriesRow[];
  layers: Array<{ color: string; label: string; idx?: number }>;
  title: string;
  displayEndYear: number;
  startYear?: number;
  big?: boolean;
}) {
  const inRange = rows.filter((r) => typeof r[0] === 'number' && (r[0] as number) >= startYear && (r[0] as number) <= displayEndYear);
  const data: DataPoint[] = inRange.map((r) => {
    const o: DataPoint = { x: r[0] as number };
    let cum = 0;
    layers.forEach((l, i) => {
      const v = r[l.idx ?? i + 1];
      const val = typeof v === 'number' && isFinite(v) ? Math.max(v, 0) : 0;
      cum += val;
      o[`v${i}`] = val;
      o[`top${i}`] = cum;
    });
    return o;
  });
  const last = layers.length - 1;
  const maxTop = Math.max(1e-9, ...data.map((d) => (d[`top${last}`] as number) ?? 0));
  const yDomain: [number, number] = [0, maxTop * 1.05];
  // Areas: largest cumulative top first (back), smallest last (front) → visible bands.
  const lines = layers
    .map((l, i) => ({ dataKey: `top${i}`, stroke: l.color, strokeWidth: 0.8, fill: l.color, fillOpacity: 0.9 }))
    .reverse();
  return (
    <div className={big ? 'min-w-0 flex-1' : 'shrink-0 w-[250px]'}>
      <div className={`font-system-mono text-gray-600 mb-1 ${big ? 'text-[11px]' : 'text-[10px] truncate'}`} title={title}>{title}</div>
      <div className="flex flex-wrap gap-x-2 gap-y-0 mb-0.5">
        {layers.map((l) => <LegendSwatch key={l.label} color={l.color} label={l.label} />)}
      </div>
      <CustomLineChart
        data={data}
        width={big ? undefined : 250}
        height={big ? 340 : 138}
        margin={big ? { top: 16, right: 16, left: 8, bottom: 56 } : { top: 8, right: 14, left: 8, bottom: 30 }}
        xDomain={[startYear, displayEndYear]}
        xTickCount={big ? 5 : 3}
        xTickFormatter={(v) => v.toFixed(0)}
        xLabel={big ? 'Year' : undefined}
        yDomain={yDomain}
        yTicks={calculateTicks(yDomain, { targetCount: big ? 5 : 3 }, 'linear')}
        yTickFormatter={(v) => (v as number).toFixed(0)}
        lines={lines}
        tooltip={(p) => (
          <ChartTip year={p.x as number} rows={layers.map((l, i) => [l.label, fmtNum(p[`v${i}`])])} />
        )}
      />
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------
// Baked default compute result (scripts/export/bake_plan_a_defaults.ts): the /api/plan-a
// response + reference series for the exact seeded defaults, so charts prerender with data.
export interface PlanABakedResult {
  result: PlanAResponse;
  series: Array<Record<string, number | null>>;
  milestones: MilestoneMapLite;
}

export default function PlanAExplorerClient({ initialDefaults, initialResult }: {
  initialDefaults: PlanADefaultsResponse | null;
  initialResult?: PlanABakedResult | null;
}) {
  const defaultsOk = !!(initialDefaults && initialDefaults.success && initialDefaults.knobs);

  const router = useRouter();
  const pathname = usePathname();

  // Config-seeded baselines (URL values override these on load; write-back diffs against them).
  // Model params: seeded from the covert experiment's model config (model_scenario_branch.yaml
  // via /api/plan-a/defaults), falling back to the website defaults for any field the config
  // doesn't set — so the explorer's defaults reproduce the covert experiment, not the homepage.
  const seededParams = useMemo(
    () => ({ ...DEFAULT_PARAMETERS, ...(initialDefaults?.model_params ?? {}) }) as ParametersType,
    [initialDefaults],
  );
  // Breakdown knobs aren't in the covert yaml, so BREAKDOWN_DEFAULTS seeds them; config
  // knobs (spread second) win wherever both define a key.
  const seededKnobs = useMemo<PlanAKnobs>(() => ({ ...BREAKDOWN_DEFAULTS, ...(initialDefaults?.knobs ?? {}) }), [initialDefaults]);
  const seededSilverBullet = useMemo<PlanASilverBullet | null>(() => initialDefaults?.silver_bullet ?? null, [initialDefaults]);
  const seededSbEnabled = !!initialDefaults?.silver_bullet;

  // State initializes from the baked/seeded defaults ONLY — never from the URL — so the
  // server-prerendered HTML and the client's first render agree (no hydration mismatch).
  // URL overrides (precedence: URL value > config-seeded default > DEFAULT_PARAMETERS)
  // are applied in the mount effect below, before the first compute.
  const [knobs, setKnobs] = useState<PlanAKnobs>(() => ({ ...seededKnobs }));
  const [silverBullet, setSilverBullet] = useState<PlanASilverBullet | null>(seededSilverBullet);
  const [sbEnabled, setSbEnabled] = useState<boolean>(seededSbEnabled);
  const [advancedOpen, setAdvancedOpen] = useState<boolean>(false);
  // Trajectories hidden from the charts (display-only; recompute not needed).
  const [hiddenSeries, setHiddenSeries] = useState<Set<PlanATrajectoryKey>>(
    () => new Set(COVERT_DEFAULT_HIDDEN),
  );
  // Which set of 5 the Key-parameters panel shows: the takeoff model params or the covert knobs.
  const [keyParamTab, setKeyParamTab] = useState<'takeoff' | 'legal' | 'covert' | 'breakdown'>('takeoff');

  const [modelParams, setModelParams] = useState<ParametersType>(() => seededParams);
  const [uiModelParams, setUiModelParams] = useState<ParametersType>(() => seededParams);
  const [isDragging, setIsDragging] = useState(false);
  const [allParameters, setAllParameters] = useState<ParameterConfig | null>(null);

  // Charts seed from the baked default result so the static prerender ships real charts;
  // a silent revalidation on mount (below) replaces it with the live full-precision result.
  const [result, setResult] = useState<PlanAResponse | null>(initialResult?.result ?? null);
  // Reference ("default" project) time series from /api/compute — the front-page model
  // run, used to overlay each mini-metric chart against the covert project.
  const [defaultSeries, setDefaultSeries] = useState<Array<Record<string, number | null>>>(initialResult?.series ?? []);
  const [milestones, setMilestones] = useState<MilestoneMapLite>(initialResult?.milestones ?? {});
  const [loading, setLoading] = useState<boolean>(defaultsOk && !initialResult);
  const [error, setError] = useState<string | null>(null);

  const grids = initialDefaults?.detection_grids ?? [];
  const siteRange = initialDefaults?.site_mw_range ?? [10, 1000];

  // Refs so the debounced compute reads the latest state without stale closures.
  const stateRef = useRef({ knobs, silverBullet, sbEnabled, modelParams });
  stateRef.current = { knobs, silverBullet, sbEnabled, modelParams };
  const abortRef = useRef<AbortController | null>(null);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  // Debounce timer for URL write-back (kept separate from the compute debounce).
  const urlTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Parameter config (bounds/metadata) — same source the front page uses for its sliders.
  useEffect(() => {
    let cancelled = false;
    fetch('/api/parameter-config')
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => { if (!cancelled && d) setAllParameters(d.config || d); })
      .catch(() => { /* sliders fall back to their fallbackMin/Max */ });
    return () => { cancelled = true; };
  }, []);

  // `quiet` skips the loading state — used by the mount-time revalidation of the baked
  // default result, where the incoming data is (normally) identical to what's shown.
  const runCompute = useCallback(async (opts?: { quiet?: boolean }) => {
    if (!defaultsOk) return;
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    if (!opts?.quiet) setLoading(true);
    setError(null);
    const { knobs: k, silverBullet: sb, sbEnabled: on, modelParams: mp } = stateRef.current;
    const parameters = convertParametersToAPIFormat({ ...mp } as unknown as ParameterRecord) as Record<string, number | string | boolean>;
    try {
      // Covert race (slow, ~seconds) and the reference/default model run (/api/compute,
      // fast) in parallel. The default series feeds the mini-metric overlays; if it fails
      // we still show the covert charts.
      const [res, dflt] = await Promise.all([
        computePlanA({ parameters, knobs: k, detection_grids: grids, silver_bullet: on ? sb : null }, controller.signal),
        fetchDefaultSeries(parameters, controller.signal),
      ]);
      if (!controller.signal.aborted) { setResult(res); setDefaultSeries(dflt.series); setMilestones(dflt.milestones); setLoading(false); }
    } catch (e) {
      if (controller.signal.aborted || (e instanceof DOMException && e.name === 'AbortError')) return;
      setError(e instanceof PlanAApiError ? e.message : String(e));
      setLoading(false);
    }
  }, [defaultsOk, grids]);

  const scheduleCompute = useCallback(() => {
    if (timerRef.current) clearTimeout(timerRef.current);
    timerRef.current = setTimeout(() => { runCompute(); }, DEBOUNCE_MS);
  }, [runCompute]);

  // Mount: apply URL overrides (shared links) on top of the baked defaults, then run the
  // first compute with the merged values. Reading the URL here — after hydration — rather
  // than in the state initializers keeps the prerendered HTML and the first client render
  // identical.
  useEffect(() => {
    const overrides = decodePlanAStateFromParams(new URLSearchParams(window.location.search));
    const mergedKnobs = { ...seededKnobs, ...overrides.knobOverrides };
    const mergedSilverBullet = seededSilverBullet
      ? { ...seededSilverBullet, ...overrides.silverBulletOverrides }
      : seededSilverBullet;
    const mergedSbEnabled = overrides.sbEnabled ?? seededSbEnabled;
    const mergedParams = { ...seededParams, ...overrides.parameterOverrides } as ParametersType;
    setKnobs(mergedKnobs);
    setSilverBullet(mergedSilverBullet);
    setSbEnabled(mergedSbEnabled);
    setModelParams(mergedParams);
    setUiModelParams(mergedParams);
    setHiddenSeries(new Set(overrides.hiddenSeries));
    // The setStates above haven't flushed yet — hand the merged values to the compute
    // directly so the first run uses them.
    stateRef.current = { knobs: mergedKnobs, silverBullet: mergedSilverBullet, sbEnabled: mergedSbEnabled, modelParams: mergedParams };
    // No compute-affecting URL overrides ⇒ the baked default result already on screen is
    // (modulo staleness) exactly what this compute returns — revalidate silently. With
    // overrides, compute loudly so the charts show they're being replaced.
    const hasComputeOverrides =
      Object.keys(overrides.parameterOverrides).length > 0
      || Object.keys(overrides.knobOverrides).length > 0
      || Object.keys(overrides.silverBulletOverrides).length > 0
      || overrides.sbEnabled !== null;
    runCompute({ quiet: !hasComputeOverrides && !!initialResult });
    return () => {
      abortRef.current?.abort();
      if (timerRef.current) clearTimeout(timerRef.current);
      if (urlTimerRef.current) clearTimeout(urlTimerRef.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // URL state write-back. Reuses the homepage's ParametersType short-name scheme
  // (via encodePlanAStateToParams) and diffs against the config-seeded baselines so
  // the URL only carries real, user-changed values ('no silent defaults'). Debounced
  // and written via router.replace({ scroll: false }) — no navigation, no scroll jump.
  // Depends on committed state (modelParams, not uiModelParams), so it fires on a
  // committed change, matching the scheduleCompute cadence rather than every tick.
  useEffect(() => {
    if (!defaultsOk) return;
    if (urlTimerRef.current) clearTimeout(urlTimerRef.current);
    urlTimerRef.current = setTimeout(() => {
      const params = encodePlanAStateToParams({
        parameters: modelParams,
        baselineParameters: seededParams,
        knobs,
        baselineKnobs: seededKnobs,
        silverBullet: silverBullet as unknown as Record<string, number> | null,
        baselineSilverBullet: seededSilverBullet as unknown as Record<string, number> | null,
        sbEnabled,
        baselineSbEnabled: seededSbEnabled,
        hiddenSeries: Array.from(hiddenSeries),
      });
      const stateQuery = params.toString();
      // encodePlanAStateToParams builds the query from scratch; carry the embed
      // flag through so an iframed explorer stays in embed mode after edits.
      const isEmbed = isEmbedNow();
      if (isEmbed) params.set('embed', '1');
      const query = params.toString();
      router.replace(query ? `${pathname}?${query}` : pathname, { scroll: false });
      // Framed embed: mirror the state query (embed flag excluded) up to the
      // parent page — the ai-2030 takeoff supplement rewrites its own address
      // bar from it, so copying the supplement's URL captures the current
      // parameter state. targetOrigin is '*' because the parent origin varies
      // (prod site, previews, local dev) and the payload is non-sensitive
      // slider state; the receiver checks event.origin against this app's
      // origin instead (see TakeoffExplorerEmbed in the ai-2030 repo).
      if (isEmbed && window.parent !== window) {
        window.parent.postMessage({ type: 'plan-a-explorer-state', query: stateQuery }, '*');
      }
    }, DEBOUNCE_MS);
    return () => { if (urlTimerRef.current) clearTimeout(urlTimerRef.current); };
  }, [
    defaultsOk, modelParams, knobs, silverBullet, sbEnabled, hiddenSeries,
    seededParams, seededKnobs, seededSilverBullet, seededSbEnabled,
    pathname, router,
  ]);

  // Embed height reporting: the framing supplement page sizes the iframe to
  // the widget's full height, so the whole explorer shows without an internal
  // vertical scrollbar. Observe the widget's root element rather than
  // document.documentElement — the page wrapper is min-h-screen, so the
  // document never measures shorter than the iframe viewport and the frame
  // could only ever grow. Reports on every layout change (charts arriving,
  // Advanced settings toggling, tab switches).
  const embedRootRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!isEmbedNow() || window.parent === window) return;
    const el = embedRootRef.current;
    if (!el) return;
    // The frame is sized to the reported height, so the page must never grow
    // its own vertical scrollbar: rounding slop or a stray margin escaping
    // the measured root would otherwise paint a useless (near-full-height)
    // scrollbar inside the frame. Clip instead — anything past the report is
    // a pixel or two at most.
    const html = document.documentElement;
    const prevOverflowY = html.style.overflowY;
    html.style.overflowY = 'clip';
    let lastHeight = 0;
    const report = () => {
      const height = Math.ceil(el.getBoundingClientRect().height);
      if (height > 0 && height !== lastHeight) {
        lastHeight = height;
        window.parent.postMessage({ type: 'plan-a-explorer-height', height }, '*');
      }
    };
    report();
    const observer = new ResizeObserver(report);
    observer.observe(el);
    return () => {
      observer.disconnect();
      html.style.overflowY = prevOverflowY;
    };
  }, []);

  const setKnob = useCallback((key: string, value: number | string | boolean) => {
    setKnobs((prev) => ({ ...prev, [key]: value }));
  }, []);

  // Display-only: no recompute, just re-render (the URL effect picks it up).
  const toggleSeries = useCallback((key: PlanATrajectoryKey) => {
    setHiddenSeries((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key); else next.add(key);
      return next;
    });
  }, []);
  const showSeries = useCallback(
    (key: PlanATrajectoryKey) => !hiddenSeries.has(key),
    [hiddenSeries],
  );

  // ParameterSlider commit handler: receives the committed parameters, then recomputes.
  const commitModelParameters = useCallback((next?: ParametersType) => {
    const p = next ?? uiModelParams;
    setModelParams(p);
    setUiModelParams(p);
    stateRef.current.modelParams = p;
    scheduleCompute();
  }, [uiModelParams, scheduleCompute]);

  const computeMode = String(knobs.compute_mode ?? 'frontier_relative');
  const bdMode = String(knobs.breakdown_compute_mode ?? 'frontier_at_pause');

  // -------------------------------------------------------------------------
  // Chart data
  // -------------------------------------------------------------------------
  // Effective-compute → uplift map. Prefer the dense, uniform-E map the backend resamples from
  // the reference run (result.upliftMap) — the plotting series (defaultSeries) is too sparse in
  // E across the steep takeoff. Fall back to defaultSeries if the map is absent.
  const eToUplift = useMemo(() => {
    const map = result?.upliftMap;
    if (map && map.length > 1) {
      const pts = map
        .filter((r) => typeof r[0] === 'number' && typeof r[1] === 'number')
        .map((r) => [r[0] as number, r[1] as number]);
      return makeInterp(pts.map((p) => p[0]), pts.map((p) => p[1]));
    }
    const pts = defaultSeries
      .filter((p) => typeof p.effectiveCompute === 'number' && typeof p.aiSwProgressMultRefPresentDay === 'number')
      .map((p) => [p.effectiveCompute as number, p.aiSwProgressMultRefPresentDay as number])
      .sort((a, b) => a[0] - b[0]);
    return makeInterp(pts.map((p) => p[0]), pts.map((p) => p[1]));
  }, [result, defaultSeries]);

  // Headline chart now plots the AI-software-R&D uplift (not raw effective compute): map each
  // trajectory's E(t) through eToUplift. The legal "discovered" envelope is dropped.
  const capabilityData = useMemo(() => {
    if (!result) return [];
    const up = (e: number | null | undefined) => (typeof e === 'number' ? eToUplift(e) : null);
    // Clip the (full, to-2100) default series to the race window so it doesn't blow up the
    // chart's x/y range (covert/legal trajectories end at the capped horizon).
    const raceMaxYear = Math.max(
      ...result.capability.covert.map((r) => r[0] ?? -Infinity),
      ...result.capability.legalDiscovered.map((r) => r[0] ?? -Infinity),
    );
    const defRows = defaultSeries
      .filter((p) => typeof p.year === 'number' && p.effectiveCompute != null && (p.year as number) <= raceMaxYear)
      .map((p) => [p.year, up(p.effectiveCompute as number)] as PlanASeriesRow);
    return buildChartData([
      { rows: defRows, map: { defUp: 1 } },
      { rows: result.capability.covert.map((r) => [r[0], up(r[1])] as PlanASeriesRow), map: { covertUp: 1 } },
      { rows: result.capability.legalUsed.map((r) => [r[0], up(r[1])] as PlanASeriesRow), map: { legalUsedUp: 1 } },
      ...(result.capability.leading
        ? [{ rows: result.capability.leading.map((r) => [r[0], up(r[1])] as PlanASeriesRow), map: { leadingUp: 1 } }]
        : []),
    ]);
  }, [result, defaultSeries, eToUplift]);


  // Covert per-metric points (column order matches api/covert_formatters covert_metrics).
  const covertMetricPoints = useMemo<Array<Record<string, number | null>>>(() => {
    if (!result?.metrics?.covert) return [];
    return result.metrics.covert.map((r) => ({
      year: r[0], effectiveCompute: r[1], automationFraction: r[2], etsp: r[3],
      softwareEfficiency: r[4], trainingSystemPerformance: r[5], researchEffort: r[6],
      aiSwProgressMultRefPresentDay: r[7],
      researchEffortIndigenous: r[8], swRateIndigenous: r[9],
      aiCodingLaborMultRefPresentDay: r[10], aiResearchTaste: r[11],
    }));
  }, [result]);

  // Legal-used per-metric points (same column order as covert).
  const legalMetricPoints = useMemo<Array<Record<string, number | null>>>(() => {
    if (!result?.metrics?.legalUsed) return [];
    return result.metrics.legalUsed.map((r) => ({
      year: r[0], effectiveCompute: r[1], automationFraction: r[2], etsp: r[3],
      softwareEfficiency: r[4], trainingSystemPerformance: r[5], researchEffort: r[6],
      aiSwProgressMultRefPresentDay: r[7],
      researchEffortIndigenous: r[8], swRateIndigenous: r[9],
      aiCodingLaborMultRefPresentDay: r[10], aiResearchTaste: r[11],
    }));
  }, [result]);

  // Legal-discovered per-metric points (full research output; same column order).
  const legalDiscoveredMetricPoints = useMemo<Array<Record<string, number | null>>>(() => {
    if (!result?.metrics?.legalDiscovered) return [];
    return result.metrics.legalDiscovered.map((r) => ({
      year: r[0], effectiveCompute: r[1], automationFraction: r[2], etsp: r[3],
      softwareEfficiency: r[4], trainingSystemPerformance: r[5], researchEffort: r[6],
      aiSwProgressMultRefPresentDay: r[7],
      researchEffortIndigenous: r[8], swRateIndigenous: r[9],
      aiCodingLaborMultRefPresentDay: r[10], aiResearchTaste: r[11],
    }));
  }, [result]);

  // Leading-company per-metric points (post-deal-breakdown; same column order).
  const leadingMetricPoints = useMemo<Array<Record<string, number | null>>>(() => {
    if (!result?.metrics?.leading) return [];
    return result.metrics.leading.map((r) => ({
      year: r[0], effectiveCompute: r[1], automationFraction: r[2], etsp: r[3],
      softwareEfficiency: r[4], trainingSystemPerformance: r[5], researchEffort: r[6],
      aiSwProgressMultRefPresentDay: r[7],
      researchEffortIndigenous: r[8], swRateIndigenous: r[9],
      aiCodingLaborMultRefPresentDay: r[10], aiResearchTaste: r[11],
    }));
  }, [result]);

  // Safety tax paid = E discovered − E used (OOMs), over time. The legal DISCOVERED track
  // reaches TED-AI + tax while USED reaches bare TED-AI, so their effective-compute gap is
  // the alignment tax actually being spent. Interpolate used onto the discovered grid.
  const safetyTaxPoints = useMemo<Array<Record<string, number | null>>>(() => {
    const disc = result?.capability?.legalDiscovered ?? [];
    const used = result?.capability?.legalUsed ?? [];
    if (!disc.length || !used.length) return [];
    // Deal breakdown: the consortium dissolves and the leading company deploys freely — the
    // tax being paid drops to zero from the breakdown on. Insert explicit points at the
    // breakdown so the drop renders vertically (the frozen legal grid has no samples
    // between the breakdown and the horizon; without them the line interpolates a slope).
    const bdT = result?.scalars?.dealBreakdownT;
    const usedPts = used
      .filter((r) => typeof r[0] === 'number' && typeof r[1] === 'number')
      .map((r) => [r[0], r[1]] as [number, number]).sort((a, b) => a[0] - b[0]);
    const pts = disc.map((r) => {
      const t = r[0];
      if (typeof t !== 'number') return { year: null, tax: null };
      if (typeof bdT === 'number' && t >= bdT) return { year: t, tax: 0 };
      const u = interpAt(usedPts, t);
      const tax = (typeof r[1] === 'number' && u != null) ? Math.max(r[1] - u, 0) : null;
      return { year: t, tax };
    });
    if (typeof bdT === 'number') {
      // Vertical drop at the breakdown, then a DENSE zero tail — the frozen legal grid has
      // no samples between the breakdown and t_end, and the chart drops out-of-domain
      // endpoints, so a two-point tail would never draw.
      const before = pts.filter((p) => typeof p.year === 'number' && (p.year as number) < bdT);
      const years = pts.map((p) => p.year).filter((y): y is number => typeof y === 'number');
      const xmax = years.length ? Math.max(...years) : bdT;
      const zeros: Array<Record<string, number | null>> = [];
      for (let t = bdT; t <= xmax + 1e-9; t += 0.5) zeros.push({ year: t, tax: 0 });
      return [...before, ...zeros];
    }
    return pts;
  }, [result]);

  // Safety tax paid (dumb version): backend-computed [t, OOMs] = (SE_disc + cumulative
  // training compute) − TED-AI EC; uses the quasi-static SE×TC product instead of the
  // integrated effective compute (so it differs from the rigorous tax by the lag).
  const safetyTaxDumbPoints = useMemo<Array<Record<string, number | null>>>(
    () => (result?.safetyTaxDumb ?? []).map((r) => ({ year: r[0], taxDumb: r[1] })),
    [result]);

  // Detection survival over time: API returns [t (years after pause), pDetected, sbCdf,
  // sbSurvival, frailtySurvival, combinedSurvival]; convert t -> calendar year via pauseYear.
  const detectionPoints = useMemo<Array<Record<string, number | null>>>(() => {
    const pts = result?.detection?.points ?? [];
    const py = result?.scalars?.pauseYear;
    if (!pts.length || py == null) return [];
    const base: number = py;
    return pts.map((r) => ({
      year: r[0] == null ? null : base + r[0], pDetected: r[1], sbCdf: r[2],
      sbSurvival: r[3], frailtySurvival: r[4], combinedSurvival: r[5],
    }));
  }, [result]);

  // Input compute series (rows [year, training_H100e, total_rd_H100e]) → point objects.
  const inputDefaultPoints = useMemo<Array<Record<string, number | null>>>(
    () => (result?.inputs?.default ?? []).map((r) => ({ year: r[0], trainingH100e: r[1], totalRdH100e: r[2] })),
    [result]);
  const inputLegalPoints = useMemo<Array<Record<string, number | null>>>(
    () => (result?.inputs?.legal ?? []).map((r) => ({ year: r[0], trainingH100e: r[1], totalRdH100e: r[2] })),
    [result]);
  const inputPlanAPoints = useMemo<Array<Record<string, number | null>>>(
    () => (result?.inputs?.covert ?? []).map((r) => ({ year: r[0], trainingH100e: r[1], totalRdH100e: r[2] })),
    [result]);
  const inputLeadingPoints = useMemo<Array<Record<string, number | null>>>(
    () => (result?.inputs?.leading ?? []).map((r) => ({ year: r[0], trainingH100e: r[1], totalRdH100e: r[2] })),
    [result]);

  const scalars = result?.scalars;

  const displayEndYear = useMemo(() => {
    // Without the covert trajectory the late-race tail is uninteresting — cut at 2040.
    if (hiddenSeries.has('covert')) return 2040.0;
    const t = scalars?.covertTtt ?? scalars?.legalTtt ?? scalars?.tedAiTime ?? 2042;
    return Math.min(2060, Math.max(2038, Math.ceil(t) + 3));
  }, [scalars, hiddenSeries]);

  // Covert slowdown factor — computed per-rollout by the backend (simulate_covert:
  // time from the pause split to TED-AI, as a ratio to the un-paused default world's).
  // The year spans are re-derived here only for the tooltip's explanatory text.
  const covertSlowdown = useMemo(() => {
    if (scalars?.covertSlowdown == null || scalars.covertTtt == null
      || scalars.tedAiTime == null || scalars.pauseYear == null) return null;
    return {
      factor: scalars.covertSlowdown,
      covertYears: scalars.covertTtt - scalars.pauseYear,
      defaultYears: scalars.tedAiTime - scalars.pauseYear,
    };
  }, [scalars]);

  // CSV export (counterpart of the front page's Export Trajectory CSV). One long-format
  // file: every row is tagged by `series` (covert / legal / default / legal_discovered /
  // leading / input_default / input_legal / detection), with the columns that don't apply
  // to a given series left blank — filter on `series` to recover any one trajectory.
  // `csvMonthly` resamples every series onto an exact 1/12-year grid (linear interp)
  // instead of each series' native solver grid. For `detection`, whose `year` column is
  // years-after-pause, the monthly grid starts at 0 rather than a calendar month.
  const [csvMonthly, setCsvMonthly] = useState(false);
  const handleExportCSV = useCallback(() => {
    if (!result) return;
    const headers = [
      'series',
      'year',
      'effective_compute',
      'ai_software_rnd_uplift_ref_present_day',
      'coding_automation_fraction',
      'etsp',
      'software_efficiency',
      'training_system_performance',
      'research_effort_realized',
      'research_effort_indigenous',
      'sw_progress_rate_indigenous',
      'ai_coding_labor_mult_ref_present_day',
      'ai_research_taste',
      'training_h100e',
      'total_rd_h100e',
      'p_detected',
      'sb_cdf',
      'sb_survival',
      'frailty_survival',
      'combined_survival',
    ];
    const cell = (v: number | null | undefined) =>
      typeof v === 'number' && Number.isFinite(v) ? String(v) : '';
    const row = (series: string, vals: Record<string, number | null | undefined>) =>
      [series, ...headers.slice(1).map((h) => cell(vals[h]))].join(',');

    const metricVals = (p: Record<string, number | null>) => ({
      year: p.year,
      effective_compute: p.effectiveCompute,
      ai_software_rnd_uplift_ref_present_day: p.aiSwProgressMultRefPresentDay,
      coding_automation_fraction: p.automationFraction,
      etsp: p.etsp,
      software_efficiency: p.softwareEfficiency,
      training_system_performance: p.trainingSystemPerformance,
      research_effort_realized: p.researchEffort,
      research_effort_indigenous: p.researchEffortIndigenous,
      sw_progress_rate_indigenous: p.swRateIndigenous,
      ai_coding_labor_mult_ref_present_day: p.aiCodingLaborMultRefPresentDay,
      ai_research_taste: p.aiResearchTaste,
    });

    // The default project pays no theft/tax, so realized == indigenous (same mapping the
    // indigenous mini-charts use via defField).
    const blocks: Array<[string, Array<Record<string, number | null | undefined>>]> = [
      ['covert', covertMetricPoints.map(metricVals)],
      ['legal', legalMetricPoints.map(metricVals)],
      ['default', defaultSeries.map((p) => ({
        ...metricVals(p),
        research_effort_indigenous: p.researchEffort,
        sw_progress_rate_indigenous: p.softwareProgressRate,
      }))],
      // Full metric columns like covert/legal; falls back to capability-only rows if the
      // metrics block is somehow absent (e.g. a stale backend).
      ['legal_discovered', legalDiscoveredMetricPoints.length
        ? legalDiscoveredMetricPoints.map(metricVals)
        : result.capability.legalDiscovered.map((r) => ({ year: r[0], effective_compute: r[1] }))],
      // Leading company post-deal-breakdown (empty ⇒ no rows when the gate is off).
      ['leading', leadingMetricPoints.map(metricVals)],
      ['input_default', inputDefaultPoints.map((p) => ({ year: p.year, training_h100e: p.trainingH100e, total_rd_h100e: p.totalRdH100e }))],
      ['input_legal', inputLegalPoints.map((p) => ({ year: p.year, training_h100e: p.trainingH100e, total_rd_h100e: p.totalRdH100e }))],
      ['detection', result.detection.points.map((r) => ({
        year: r[0], p_detected: r[1], sb_cdf: r[2], sb_survival: r[3],
        frailty_survival: r[4], combined_survival: r[5],
      }))],
    ];

    const csvRows: string[] = [headers.join(',')];
    for (const [series, vals] of blocks) {
      for (const v of (csvMonthly ? resampleMonthly(vals) : vals)) csvRows.push(row(series, v));
    }

    const blob = new Blob([csvRows.join('\n')], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
    link.download = `ai_futures_plan_a_${timestamp}.csv`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  }, [result, covertMetricPoints, legalMetricPoints, legalDiscoveredMetricPoints, leadingMetricPoints,
      defaultSeries, inputDefaultPoints, inputLegalPoints, csvMonthly]);

  return (
    <div ref={embedRootRef}>
      <div className="max-w-[1500px] mx-auto px-5 lg:px-8 py-6">
      {!defaultsOk && (
        <div className="rounded border border-red-300 bg-red-50 text-red-800 text-sm px-4 py-3">
          Could not load model defaults from the backend. Is the Flask server running (npm run dev)?
        </div>
      )}

      {defaultsOk && (
        <>
          {error && (
            <div className="rounded border border-amber-300 bg-amber-50 text-amber-900 text-sm px-4 py-2 mb-4">
              {error}
            </div>
          )}

          {/* Front-page layout (KeyMilestonePanel): left column with Explanation pinned to the
              top and the key date readouts pushed to the bottom (justify-between), charts right. */}
          {/* Single horizontal-scroll canvas (front-page arrangement): ONE scrollbar moves the
              Explanation column, the big race charts, the Key Parameters sliders directly beneath
              them, and the per-metric mini charts — all together as one wide grid. */}
          <div className="relative">
          {/* One ChartSyncProvider for the whole canvas → hovering any chart shows a synced
              crosshair at the same year across every chart (front-page behaviour). */}
          <ChartSyncProvider>
          <div className="covert-scroll overflow-x-auto overflow-y-hidden pb-3">
          <div className="grid gap-x-6 gap-y-4 min-w-max" style={{ gridTemplateColumns: '290px 1320px max-content', gridTemplateRows: 'auto auto' }}>
          <aside className="row-start-1 row-span-2 flex flex-col gap-6 w-[290px]">
            <div>
              <SectionHeader title="Explanation" />
              <div className="mt-2 space-y-3 text-sm text-gray-700 leading-relaxed">
                <p>
                  This supplement presents an extension of the AI Futures Model, adapted to
                  simulate AI capabilities in Plan A. After the deal begins, the{' '}
                  <span className="font-medium">consortium</span> pauses briefly, then 
                  scales to TED-AI in a transparent, controlled manner. We also show what a{' '}
                  <span className="font-medium">covert project</span> defecting from Plan A
                  would look like. Optionally, the deal later{' '}
                  <span className="font-medium">dissolves</span>  and the leading 
                  company continues alone at full speed. The un-paused <span className="font-medium">default</span> world is shown for
                  reference.
                </p>
              </div>
            </div>

            <div>
              {showSeries('default') && (
                <BigReadout label="Default project reaches TED-AI" value={fmtMonthYear(scalars?.tedAiTime)} loading={loading} />
              )}
              {showSeries('consortium') && (
                <BigReadout label="Consortium reaches TED-AI" value={fmtMonthYear(scalars?.legalUsedTtt)} loading={loading} />
              )}
              {showSeries('covert') && (
                <BigReadout
                  label="Covert reaches TED-AI"
                  value={fmtMonthYear(scalars?.covertTtt)}
                  loading={loading}
                  hoverDetail={covertSlowdown
                    ? `Slowdown factor: ${covertSlowdown.factor.toFixed(2)}×. From the pause (where the covert project splits off from the default world), the covert project takes ${covertSlowdown.covertYears.toFixed(1)} yr to reach TED-AI vs ${covertSlowdown.defaultYears.toFixed(1)} yr for the un-paused default.`
                    : undefined}
                />
              )}
              {scalars?.dealBreakdownEnabled && showSeries('leading') && (
                <BigReadout label="Leading co. reaches TED-AI" value={fmtMonthYear(scalars?.leadingTtt)} loading={loading} />
              )}
            </div>
          </aside>

          {/* col 2, row 1: the "Big 3" — headline uplift race chart + the two ETSP-gains
              stacked-area decompositions, side by side. */}
          <div className="col-start-2 row-start-1 flex gap-4 items-start">
            <div className="min-w-0 flex-1">
              <div className="text-[11px] font-system-mono text-gray-600 mb-1 flex items-center gap-2 flex-wrap">
                <span title={COVERT_CHART_EXPLANATIONS.capability}>AI software R&amp;D uplift vs time</span>
                {showSeries('covert') && <LegendSwatch color={COLOR.covert} label="Covert" />}
                {showSeries('consortium') && <LegendSwatch color={COLOR.legalUsed} label="Consortium used" />}
                {scalars?.dealBreakdownEnabled && showSeries('leading') && (
                  <LegendSwatch color={COLOR.leading} label="Leading co. (post-dissolution)" />
                )}
                {showSeries('default') && <LegendSwatch color={COLOR.default} label="Default" />}
                <span className="inline-flex items-center gap-1.5 text-xs text-gray-700">
                  <svg width="10" height="10"><circle cx="5" cy="5" r="3.5" fill={COLOR.default} stroke="white" strokeWidth="1.5" /></svg>
                  milestones
                </span>
              </div>
              {capabilityData.length > 0
                ? <CapabilityChart data={capabilityData} scalars={scalars!} milestones={milestones} simMilestones={result!.milestones} eToUplift={eToUplift} capability={result!.capability} hidden={hiddenSeries} />
                : <NoData loading={loading} />}
            </div>
            {showSeries('consortium') && (result
              ? <StackedAreaChart
                  big
                  rows={result.breakdown.legal}
                  title="Consortium ETSP gains (OOMs since pause)"
                  layers={[
                    { color: STACK.hw, label: 'Hardware' },
                    { color: STACK.windfall, label: 'Windfall' },
                    { color: STACK.mandatory, label: 'Min SW (mandatory)' },
                    { color: STACK.own, label: 'Used SW' },
                    { color: STACK.tax, label: 'Forgone SW' },
                  ]}
                  displayEndYear={displayEndYear}
                />
              : <div className="min-w-0 flex-1"><NoData loading={loading} /></div>)}
            {showSeries('covert') && (result
              ? <StackedAreaChart
                  big
                  rows={result.breakdown.covert}
                  title="Covert ETSP gains (OOMs since pause)"
                  layers={[
                    { color: STACK.hw, label: 'Hardware', idx: 1 },
                    { color: STACK.stolen, label: 'Stolen SW', idx: 3 },
                    { color: STACK.own, label: 'Indigenous SW', idx: 2 },
                  ]}
                  displayEndYear={displayEndYear}
                />
              : <div className="min-w-0 flex-1"><NoData loading={loading} /></div>)}
          </div>

          {/* col 2, row 2: Key Parameters — directly beneath the big charts, scrolls with them */}
          <section className="col-start-2 row-start-2 pt-1">
            {/* Header: switch the headline sliders between takeoff / legal / covert /
                deal-collapse parameter groups. Constrained to the sliders' width so the
                divider + buttons don't strand across the full column. */}
            <div className="flex gap-2 items-center max-w-4xl">
              <div className="flex items-center gap-1 whitespace-nowrap">
                {([['takeoff', 'Takeoff Parameters'], ['legal', 'Consortium Scaling Parameters'],
                   ['covert', 'Covert Project Parameters'], ['breakdown', 'Deal Dissolution Parameters']] as const).map(([tab, txt]) => (
                  <button
                    key={tab}
                    type="button"
                    onClick={() => setKeyParamTab(tab)}
                    className={`text-[12px] font-semibold font-system-mono px-2 py-0.5 rounded transition-colors ${
                      keyParamTab === tab
                        ? 'text-primary bg-gray-500/10'
                        : 'text-gray-400 hover:text-gray-700'
                    }`}
                  >
                    {txt}
                  </button>
                ))}
              </div>
              <div className="flex-1 border-t border-gray-500/30" />
              <label
                className="inline-flex items-center gap-1 text-[11px] font-system-mono uppercase tracking-wider text-gray-500 hover:text-gray-900 whitespace-nowrap cursor-pointer select-none"
                title="Resample every exported series onto an exact 1/12-year grid (linear interpolation) instead of each series' native solver grid"
              >
                <input
                  type="checkbox"
                  checked={csvMonthly}
                  onChange={(e) => setCsvMonthly(e.target.checked)}
                  className="accent-gray-600"
                />
                monthly
              </label>
              <button
                type="button"
                onClick={handleExportCSV}
                disabled={!result}
                className="text-[11px] font-system-mono uppercase tracking-wider text-gray-500 hover:text-gray-900 whitespace-nowrap disabled:opacity-40 disabled:hover:text-gray-500"
              >
                Export CSV ↓
              </button>
              <button
                type="button"
                onClick={() => setAdvancedOpen(true)}
                className="text-[11px] font-system-mono uppercase tracking-wider text-gray-500 hover:text-gray-900 whitespace-nowrap"
              >
                Advanced settings ⚙
              </button>
            </div>
            {keyParamTab === 'takeoff' ? (
              <ParameterHoverProvider>
                <div className="grid grid-cols-5 gap-x-6 gap-y-5 mt-4 max-w-4xl">
                  {MODEL_PARAMS.map((m) => (
                    <div key={m.paramName} className="relative flex w-full flex-col justify-between">
                      <label className="block w-full text-xs font-medium text-foreground mb-2 leading-snug">
                        <WithChartTooltip explanation={m.explanation} fullWidth>
                          {m.label}
                        </WithChartTooltip>
                      </label>
                      <ParameterSlider
                        paramName={m.paramName}
                        label=""
                        step={m.step}
                        fallbackMin={m.fallbackMin}
                        fallbackMax={m.fallbackMax}
                        customMin={m.customMin}
                        customMax={m.customMax}
                        decimalPlaces={m.decimalPlaces}
                        customFormatValue={m.format}
                        useLogScale={m.useLogScale}
                        value={uiModelParams[m.paramName] as number}
                        uiParameters={uiModelParams}
                        setUiParameters={setUiModelParams}
                        allParameters={allParameters}
                        isDragging={isDragging}
                        setIsDragging={setIsDragging}
                        commitParameters={commitModelParameters}
                        baselineParameters={DEFAULT_PARAMETERS}
                      />
                    </div>
                  ))}
                </div>
              </ParameterHoverProvider>
            ) : keyParamTab === 'legal' ? (
              <ParameterHoverProvider>
                <div className="grid grid-cols-5 gap-x-6 gap-y-5 mt-4 max-w-4xl">
                  {LEGAL_KEY_PARAMS.map((p) => (
                    <PlanAParamSlider
                      key={p.key}
                      label={p.label}
                      explanation={COVERT_KNOB_EXPLANATIONS[p.key]}
                      value={Number(knobs[p.key] ?? p.min)}
                      defaultValue={Number(seededKnobs[p.key] ?? p.min)}
                      min={p.min} max={p.max} step={p.step} log={p.log}
                      format={p.format}
                      onChange={(v) => setKnob(p.key, v)}
                      onCommit={(v) => { setKnob(p.key, v); scheduleCompute(); }}
                    />
                  ))}
                </div>
              </ParameterHoverProvider>
            ) : keyParamTab === 'covert' ? (
              <ParameterHoverProvider>
                <div className="grid grid-cols-5 gap-x-6 gap-y-5 mt-4 max-w-4xl">
                  {/* Covert-size param: frontier-relative fraction (log) or absolute H100e. */}
                  {computeMode === 'frontier_relative' ? (
                    <PlanAParamSlider
                      label="How much less R&D compute does the covert project have than the leading AI company?"
                      explanation={COVERT_KNOB_EXPLANATIONS.total_h100e_frac}
                      value={Number(knobs.total_h100e_frac ?? 0.0126)}
                      defaultValue={Number(seededKnobs.total_h100e_frac ?? 0.0126)}
                      min={0.001} max={0.2} step={0.001} log
                      format={(v) => `${(v * 100).toFixed(2)}%`}
                      onChange={(v) => setKnob('total_h100e_frac', v)}
                      onCommit={(v) => { setKnob('total_h100e_frac', v); scheduleCompute(); }}
                    />
                  ) : (
                    <PlanAParamSlider
                      label="Covert compute (total H100e)"
                      explanation={COVERT_KNOB_EXPLANATIONS.total_h100e}
                      value={Number(knobs.total_h100e ?? 500000) / 1e6}
                      defaultValue={Number(seededKnobs.total_h100e ?? 500000) / 1e6}
                      min={0.1} max={5} step={0.1} log
                      format={(v) => `${v.toFixed(2)}M`}
                      onChange={(v) => setKnob('total_h100e', v * 1e6)}
                      onCommit={(v) => { setKnob('total_h100e', v * 1e6); scheduleCompute(); }}
                    />
                  )}
                  {COVERT_KEY_PARAMS.map((p) => (
                    <PlanAParamSlider
                      key={p.key}
                      label={p.label}
                      explanation={COVERT_KNOB_EXPLANATIONS[p.key]}
                      value={Number(knobs[p.key] ?? p.min)}
                      defaultValue={Number(seededKnobs[p.key] ?? p.min)}
                      min={p.min} max={p.max} step={p.step} log={p.log}
                      format={p.format}
                      onChange={(v) => setKnob(p.key, v)}
                      onCommit={(v) => { setKnob(p.key, v); scheduleCompute(); }}
                    />
                  ))}
                </div>
              </ParameterHoverProvider>
            ) : (
              <ParameterHoverProvider>
                <div className="grid grid-cols-5 gap-x-6 gap-y-5 mt-4 max-w-4xl">
                  {BREAKDOWN_KEY_PARAMS.filter((p) => !p.mode || p.mode === bdMode).map((p) => (
                    <PlanAParamSlider
                      key={p.key}
                      label={p.label}
                      explanation={COVERT_KNOB_EXPLANATIONS[p.key]}
                      value={Number(knobs[p.key] ?? p.min)}
                      defaultValue={Number(seededKnobs[p.key] ?? p.min)}
                      min={p.min} max={p.max} step={p.step} log={p.log}
                      format={p.format}
                      disabled={!knobs.enable_deal_breakdown}
                      onChange={(v) => setKnob(p.key, v)}
                      onCommit={(v) => { setKnob(p.key, v); scheduleCompute(); }}
                    />
                  ))}
                  {/* Toggle + compute-basis select fill the grid's spare right-hand columns. */}
                  <div className="col-span-2 flex flex-col gap-3">
                    <label className="flex items-center gap-2 w-fit cursor-pointer"
                           title={COVERT_KNOB_EXPLANATIONS.enable_deal_breakdown}>
                      <input
                        type="checkbox"
                        checked={Boolean(knobs.enable_deal_breakdown)}
                        onChange={(e) => { setKnob('enable_deal_breakdown', e.target.checked); scheduleCompute(); }}
                        className="w-3.5 h-3.5 accent-red-700"
                      />
                      <span className="text-xs font-medium text-foreground">Enable deal dissolution</span>
                    </label>
                    <div className="w-56">
                      <Select
                        label={BREAKDOWN_MODE_SELECT.label}
                        title={COVERT_KNOB_EXPLANATIONS.breakdown_compute_mode}
                        value={bdMode}
                        options={BREAKDOWN_MODE_SELECT.options}
                        onChange={(v) => { setKnob('breakdown_compute_mode', v); scheduleCompute(); }}
                      />
                    </div>
                  </div>
                </div>
              </ParameterHoverProvider>
            )}
          </section>

          {/* col 3: per-metric + input mini charts, stacked in a 4-row grid (fills the
              column down then wraps to the next column), spanning the chart + params rows. */}
          <div className="col-start-3 row-start-1 row-span-2 grid grid-flow-col grid-rows-3 gap-x-4 gap-y-2 content-start">
            {/* Input compute series each trajectory runs on: default (faint green), the
                Plan-A-relativized LEGAL input (navy), and the covert datacenter (red) — a flat
                slice held constant over [pause, t_end]. Tooltip values in K/M/B/T H100e. */}
            <MiniMetricChart def={inputDefaultPoints} legal={inputLegalPoints} cov={inputPlanAPoints}
              leading={inputLeadingPoints} hidden={hiddenSeries} loading={loading}
              field="trainingH100e" title="Input: training compute (H100e)" logScale format={fmtH100e} displayEndYear={displayEndYear} />
            <MiniMetricChart def={inputDefaultPoints} legal={inputLegalPoints} cov={inputPlanAPoints}
              leading={inputLeadingPoints} hidden={hiddenSeries} loading={loading}
              field="totalRdH100e" title="Input: total R&D compute (H100e)" logScale format={fmtH100e} displayEndYear={displayEndYear} />
            {/* Alignment tax actually spent over time = E discovered − E used (OOMs). */}
            <MiniMetricChart def={[]} legal={safetyTaxPoints} field="tax" legalLabel="Safety tax" hidden={hiddenSeries} loading={loading}
              title="Safety tax paid (OOMs)" logScale={false} format={(v) => v.toFixed(2)} displayEndYear={displayEndYear} />
            {/* Dumb version: (SE_disc × cumulative training compute) / TED-AI EC, in OOMs. */}
            <MiniMetricChart def={[]} legal={safetyTaxDumbPoints} field="taxDumb" legalLabel="Safety tax (dumb)" hidden={hiddenSeries} loading={loading}
              title="Safety tax paid (dumb version)" logScale={false} format={(v) => v.toFixed(2)} displayEndYear={displayEndYear} />
            {/* Detection survival = P(covert still undetected) over time, with the silver-bullet
                and frailty (re-inspection) components broken out. combined = sb × frailty. */}
            <MiniMetricChart def={detectionPoints} cov={detectionPoints} legal={detectionPoints} loading={loading}
              field="combinedSurvival" defField="sbSurvival" legalField="frailtySurvival"
              covLabel="undetected (combined)" legalLabel="frailty only" defLabel="silver-bullet only"
              title="Detection survival — P(covert undetected)" logScale={false}
              format={(v) => `${(v * 100).toFixed(0)}%`} displayEndYear={displayEndYear} />
            {MINI_METRICS.map((m) => (
              <MiniMetricChart key={m.field} def={defaultSeries} cov={covertMetricPoints} legal={legalMetricPoints}
                leading={leadingMetricPoints} hidden={hiddenSeries} loading={loading}
                field={m.field} defField={m.defField} title={m.title} logScale={m.log} format={m.format} displayEndYear={displayEndYear} />
            ))}
          </div>

          </div>
          </div>
          </ChartSyncProvider>
          {loading && (capabilityData.length > 0 || covertMetricPoints.length > 0) && (
            <div className="absolute inset-0 z-20 bg-vivid-background/70 backdrop-blur-[1px] flex items-center justify-center pointer-events-none">
              <span className="h-6 w-6 animate-spin rounded-full border-2 border-gray-400 border-t-transparent" aria-hidden="true" />
            </div>
          )}
          </div>
        </>
      )}
      </div>

      {/* Advanced parameters slide-out drawer (model advanced params + covert scenario) */}
      {defaultsOk && advancedOpen && (
        <div className="fixed inset-0 z-50">
          <div className="absolute inset-0 bg-black/40" onClick={() => setAdvancedOpen(false)} />
          <div className="absolute inset-y-0 right-0 w-full max-w-md bg-vivid-background shadow-xl flex flex-col">
            <div className="flex items-center justify-between px-4 py-3 border-b border-black/10">
              <h2 className="text-sm font-semibold text-primary uppercase tracking-[0.2em] font-system-mono">
                Advanced parameters
              </h2>
              <button
                type="button"
                onClick={() => setAdvancedOpen(false)}
                className="p-2 -mr-2 text-gray-500 hover:text-gray-700"
                aria-label="Close advanced parameters"
              >
                <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
            <div className="flex-1 overflow-y-auto px-4 py-4 space-y-6">
              {/* Which trajectories the charts draw — display-only, no recompute. */}
              <div className="space-y-3">
                <SectionHeader title="Visible trajectories" />
                {TRAJECTORY_TOGGLES.map((t) => (
                  <label key={t.key} className="flex items-center justify-between cursor-pointer">
                    <span className="flex items-center gap-2 text-xs font-medium text-foreground">
                      <span className="inline-block w-3 h-[3px] rounded-sm" style={{ backgroundColor: t.color }} />
                      {t.label}
                    </span>
                    <input
                      type="checkbox"
                      checked={!hiddenSeries.has(t.key)}
                      onChange={() => toggleSeries(t.key)}
                      className="w-3.5 h-3.5 accent-red-700"
                    />
                  </label>
                ))}
              </div>

              {/* Full model advanced parameters — the same component the front page uses */}
              <ParameterHoverProvider>
                <AdvancedSections
                  uiParameters={uiModelParams}
                  setUiParameters={setUiModelParams}
                  allParameters={allParameters}
                  isDragging={isDragging}
                  setIsDragging={setIsDragging}
                  commitParameters={commitModelParameters}
                  scHorizonLogBounds={{ min: 0, max: 8 }}
                  parallelPenaltyBounds={{ min: 0, max: 1 }}
                  baselineParameters={DEFAULT_PARAMETERS}
                  onExportCSV={handleExportCSV}
                />
              </ParameterHoverProvider>

              {/* Consortium / deal-wide knobs (pause, scaling strategy, caps) */}
              <div className="space-y-4 border-t border-black/10 pt-4">
                <SectionHeader title="Consortium scaling" />
                <div className="space-y-3">
                  {CONSORTIUM_SELECTS.map((s) => (
                    <Select
                      key={s.key}
                      label={s.label}
                      title={COVERT_KNOB_EXPLANATIONS[s.key]}
                      value={String(knobs[s.key] ?? s.options[0])}
                      options={s.options}
                      onChange={(v) => { setKnob(s.key, v); scheduleCompute(); }}
                    />
                  ))}
                  {CONSORTIUM_SLIDERS.map((s) => {
                    const disabled = s.key === 'max_sw_rate' && knobs.legal_strategy !== 'target_year_capped_sw';
                    return (
                      <PlanAParamSlider
                        key={s.key}
                        label={s.label}
                        explanation={COVERT_KNOB_EXPLANATIONS[s.key]}
                        value={Number(knobs[s.key] ?? s.min)}
                        defaultValue={seededKnobs[s.key] != null ? Number(seededKnobs[s.key]) : undefined}
                        min={s.min} max={s.max} step={s.step} format={s.format} disabled={disabled}
                        onChange={(v) => setKnob(s.key, v)}
                        onCommit={(v) => { setKnob(s.key, v); scheduleCompute(); }}
                      />
                    );
                  })}
                </div>
              </div>

              {/* Covert-project knobs (sizing, theft, build timeline) */}
              <div className="space-y-4 border-t border-black/10 pt-4">
                <SectionHeader title="Covert project" />
                <div className="space-y-3">
                  {COVERT_SELECTS.map((s) => (
                    <Select
                      key={s.key}
                      label={s.label}
                      title={COVERT_KNOB_EXPLANATIONS[s.key]}
                      value={String(knobs[s.key] ?? s.options[0])}
                      options={s.options}
                      onChange={(v) => { setKnob(s.key, v); scheduleCompute(); }}
                    />
                  ))}
                  {computeMode === 'frontier_relative' ? (
                    <PlanAParamSlider
                      label="Covert compute (frac of frontier)"
                      explanation={COVERT_KNOB_EXPLANATIONS.total_h100e_frac}
                      value={Number(knobs.total_h100e_frac ?? 0.0126)}
                      defaultValue={Number(seededKnobs.total_h100e_frac ?? 0.0126)}
                      min={0.001} max={0.2} step={0.001} format={(v) => v.toFixed(3)}
                      onChange={(v) => setKnob('total_h100e_frac', v)}
                      onCommit={(v) => { setKnob('total_h100e_frac', v); scheduleCompute(); }}
                    />
                  ) : (
                    <PlanAParamSlider
                      label="Covert compute (total H100e, millions)"
                      explanation={COVERT_KNOB_EXPLANATIONS.total_h100e}
                      value={Number(knobs.total_h100e ?? 500000) / 1e6}
                      defaultValue={Number(seededKnobs.total_h100e ?? 500000) / 1e6}
                      min={0.1} max={5} step={0.1} format={(v) => v.toFixed(1) + 'M'}
                      onChange={(v) => setKnob('total_h100e', v * 1e6)}
                      onCommit={(v) => { setKnob('total_h100e', v * 1e6); scheduleCompute(); }}
                    />
                  )}
                  {COVERT_SLIDERS.map((s) => (
                    <PlanAParamSlider
                      key={s.key}
                      label={s.label}
                      explanation={COVERT_KNOB_EXPLANATIONS[s.key]}
                      value={Number(knobs[s.key] ?? s.min)}
                      defaultValue={seededKnobs[s.key] != null ? Number(seededKnobs[s.key]) : undefined}
                      min={s.min} max={s.max} step={s.step} format={s.format}
                      onChange={(v) => setKnob(s.key, v)}
                      onCommit={(v) => { setKnob(s.key, v); scheduleCompute(); }}
                    />
                  ))}
                </div>
              </div>

              {/* Deal dissolution (gated): consortium dissolves; leading company continues alone */}
              <div className="space-y-4 border-t border-black/10 pt-4">
                <SectionHeader title="Deal dissolution" />
                <div className="space-y-3">
                  <label className="flex items-center justify-between" title={COVERT_KNOB_EXPLANATIONS.enable_deal_breakdown}>
                    <span className="text-xs font-medium text-foreground">Enable deal dissolution</span>
                    <input
                      type="checkbox"
                      checked={Boolean(knobs.enable_deal_breakdown)}
                      onChange={(e) => { setKnob('enable_deal_breakdown', e.target.checked); scheduleCompute(); }}
                      className="w-3.5 h-3.5 accent-red-700"
                    />
                  </label>
                  <Select
                    label={BREAKDOWN_MODE_SELECT.label}
                    title={COVERT_KNOB_EXPLANATIONS.breakdown_compute_mode}
                    value={bdMode}
                    options={BREAKDOWN_MODE_SELECT.options}
                    onChange={(v) => { setKnob('breakdown_compute_mode', v); scheduleCompute(); }}
                  />
                  {BREAKDOWN_SLIDERS.filter((s) => !s.mode || s.mode === bdMode).map((s) => (
                    <PlanAParamSlider
                      key={s.key}
                      label={s.label}
                      explanation={COVERT_KNOB_EXPLANATIONS[s.key]}
                      value={Number(knobs[s.key] ?? s.min)}
                      defaultValue={seededKnobs[s.key] != null ? Number(seededKnobs[s.key]) : undefined}
                      min={s.min} max={s.max} step={s.step} format={s.format}
                      disabled={!knobs.enable_deal_breakdown}
                      onChange={(v) => setKnob(s.key, v)}
                      onCommit={(v) => { setKnob(s.key, v); scheduleCompute(); }}
                    />
                  ))}
                  <p className="text-[11px] text-gray-400 leading-snug">
                    The consortium dissolves at pause-end + dissolution time (covert stealing stops).
                    The leading company continues alone — used capability, full discovered research
                    stock, and compute set by the basis above (a share of the pre-deal frontier if
                    deal-era compute is destroyed, or of the world&apos;s compute at the dissolution if
                    not), growing at the post-dissolution rate.
                  </p>
                </div>
              </div>

              {/* Detection knobs (drive the survival mini-chart) */}
              <div className="space-y-4 border-t border-black/10 pt-4">
                <SectionHeader title="Detection" />
                <div className="space-y-3">
                  {DETECTION_SLIDERS.map((s) => (
                    <PlanAParamSlider
                      key={s.key}
                      label={s.label}
                      explanation={COVERT_KNOB_EXPLANATIONS[s.key]}
                      value={Number(knobs[s.key] ?? s.min)}
                      defaultValue={seededKnobs[s.key] != null ? Number(seededKnobs[s.key]) : undefined}
                      min={s.min} max={s.max} step={s.step} format={s.format}
                      onChange={(v) => setKnob(s.key, v)}
                      onCommit={(v) => { setKnob(s.key, v); scheduleCompute(); }}
                    />
                  ))}
                  <PlanAParamSlider
                    label="Per-site power (MW)"
                    explanation={COVERT_KNOB_EXPLANATIONS.optimal_site_mw}
                    value={Number(knobs.optimal_site_mw ?? siteRange[0])}
                    defaultValue={seededKnobs.optimal_site_mw != null ? Number(seededKnobs.optimal_site_mw) : undefined}
                    min={siteRange[0]} max={siteRange[1]} step={5} format={(v) => v.toFixed(0)}
                    onChange={(v) => setKnob('optimal_site_mw', v)}
                    onCommit={(v) => { setKnob('optimal_site_mw', v); scheduleCompute(); }}
                  />
                </div>
                <div className="space-y-3 border-t border-black/10 pt-3">
                  <label className="flex items-center justify-between">
                    <span className="text-[11px] font-system-mono uppercase tracking-wider text-gray-400">Detection silver bullet</span>
                    <input
                      type="checkbox"
                      checked={sbEnabled}
                      onChange={(e) => { setSbEnabled(e.target.checked); scheduleCompute(); }}
                      className="w-3.5 h-3.5 accent-red-700"
                    />
                  </label>
                  {silverBullet && SB_SLIDERS.map((s) => (
                    <PlanAParamSlider
                      key={s.key}
                      label={s.label}
                      value={Number(silverBullet[s.key])}
                      min={s.min} max={s.max} step={s.step} disabled={!sbEnabled}
                      onChange={(v) => setSilverBullet((prev) => prev && { ...prev, [s.key]: v })}
                      onCommit={(v) => { setSilverBullet((prev) => prev && { ...prev, [s.key]: v }); scheduleCompute(); }}
                    />
                  ))}
                  <p className="text-[11px] text-gray-400 leading-snug">Requires 0 &lt; P(mid) &lt; P(used+Δ) &lt; ceiling ≤ 1.</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// Placeholder shown in a chart cell before the first compute completes.
function NoData({ loading }: { loading: boolean }) {
  if (!loading) {
    return (
      <div className="h-[340px] flex items-center justify-center text-sm text-gray-400">
        No data
      </div>
    );
  }
  // Main-page skeleton treatment (ChartsSkeleton): pulsing gray box + spinner ring.
  return (
    <div className="h-[340px] bg-gray-100 rounded animate-pulse flex items-center justify-center">
      <div className="w-8 h-8 border-2 border-gray-300 border-t-gray-500 rounded-full animate-spin" />
    </div>
  );
}

// Format a fractional year as MM / YYYY (front-page big-date style).
function fmtMonthYear(year: number | null | undefined): string {
  if (year == null || !isFinite(year)) return '—';
  const y = Math.floor(year);
  const m = Math.min(12, Math.max(1, Math.round((year - y) * 12) + 1));
  return `${String(m).padStart(2, '0')} / ${y}`;
}

// Big date readout — matches the front-page KeyMilestoneValue: monospace label + rule, then
// a large et-book value whose characters are justified across the column width.
// `hoverDetail` shows a tooltip while the big value is hovered; portalled to <body> because
// the readouts sit inside the covert-scroll overflow-hidden canvas, which would clip it.
function BigReadout({ label, value, hoverDetail, loading }: { label: string; value: string; hoverDetail?: string; loading?: boolean }) {
  const [tipPos, setTipPos] = useState<{ top: number; left: number } | null>(null);
  const valueRef = useRef<HTMLDivElement>(null);
  const showTip = useCallback(() => {
    const rect = valueRef.current?.getBoundingClientRect();
    if (rect) setTipPos({ top: rect.bottom + 4, left: rect.left });
  }, []);
  const hideTip = useCallback(() => setTipPos(null), []);
  // Main-page treatment: a pulsing bar where the big date will land, not a bare dash.
  if (loading && value === '—') {
    return (
      <div className="mt-3">
        <div className="flex items-center gap-2">
          <p className="leading-tight text-[12px] font-semibold text-primary text-left font-system-mono">{label}</p>
          <div className="flex-1 border-t border-gray-500/30" />
        </div>
        <div className="mt-1 h-9 w-2/3 animate-pulse rounded bg-slate-200/70" />
      </div>
    );
  }
  return (
    <div className="mt-3">
      <div className="flex items-center gap-2">
        <p className="leading-tight text-[12px] font-semibold text-primary text-left font-system-mono">{label}</p>
        <div className="flex-1 border-t border-gray-500/30" />
      </div>
      <div
        ref={valueRef}
        className={`mt-1 text-4xl font-et-book font-bold text-primary leading-none flex justify-between${hoverDetail ? ' cursor-help' : ''}`}
        onMouseEnter={hoverDetail ? showTip : undefined}
        onMouseLeave={hoverDetail ? hideTip : undefined}
      >
        {value.split('').map((char, idx) => <span key={idx}>{char}</span>)}
      </div>
      {hoverDetail && tipPos && typeof document !== 'undefined' && createPortal(
        <div className="fixed z-[9999] pointer-events-none" style={{ top: tipPos.top, left: tipPos.left }}>
          <div className="w-64 rounded-md bg-gray-900 px-3 py-2 text-xs text-white shadow-lg font-system-mono font-normal">
            {hoverDetail}
          </div>
        </div>,
        document.body
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Panel wrapper: title + explanation + legend + loading overlay
// ---------------------------------------------------------------------------
function Panel({ title, explanation, legend, loading, hasData, children }: {
  title: string; explanation: string; legend: ReactNode; loading: boolean; hasData: boolean;
  children: () => ReactNode;
}) {
  return (
    <section className="pt-1">
      <SectionHeader
        title={title}
        explanation={explanation}
        right={<div className="flex flex-wrap gap-x-3 gap-y-1 justify-end">{legend}</div>}
      />
      <div className="relative mt-2">
        {hasData ? children() : (
          <div className="h-[340px] flex items-center justify-center text-sm text-gray-400">
            {loading ? 'Computing…' : 'No data'}
          </div>
        )}
        {loading && hasData && (
          <div className="absolute inset-0 z-20 bg-vivid-background/70 backdrop-blur-[1px] flex items-center justify-center">
            <span className="h-6 w-6 animate-spin rounded-full border-2 border-gray-400 border-t-transparent" aria-hidden="true" />
          </div>
        )}
      </div>
    </section>
  );
}

// ---------------------------------------------------------------------------
// Panel A: AI-software-R&D uplift vs time (covert, legal-used, default), with milestone dots.
// y plots uplift (a pure function of effective compute, via eToUplift) on a log axis.
// ---------------------------------------------------------------------------
function fmtUplift(v: number): string {
  if (!isFinite(v)) return '';
  if (v >= 100) return `${Math.round(v)}×`;
  if (v >= 10) return `${v.toFixed(0)}×`;
  return `${v.toFixed(1)}×`;
}
// A milestone dot on one specific trajectory: crossing time x, uplift y, and which
// trajectory (display name) reached it — for the hover tooltip.
interface DotMarker { key: string; label: string; seriesName: string; x: number; y: number }

// First time a [t, E] series crosses an effective-compute threshold (linear interp).
// Returns null when the series starts at/above the threshold (the crossing predates the
// window — dotting the series start would be misleading) or never reaches it.
// E_EPS absorbs solver/threshold slop: the consortium's glidepath targets TED-AI exactly
// and can plateau a sub-milli-OOM short of the milestone's threshold (which comes from a
// separate /api/compute run) — within 1e-3 OOMs it has visually reached the milestone.
const E_EPS = 1e-3;
function firstCrossingTime(rows: PlanASeriesRow[], threshold: number): number | null {
  const pts = rows
    .filter((r): r is [number, number] => typeof r[0] === 'number' && typeof r[1] === 'number')
    .sort((a, b) => a[0] - b[0]);
  if (pts.length === 0 || pts[0][1] >= threshold - E_EPS) return null;
  for (let i = 1; i < pts.length; i++) {
    const [t0, e0] = pts[i - 1];
    const [t1, e1] = pts[i];
    if (e1 >= threshold - E_EPS) {
      // Clamp: when e1 lands in [threshold - E_EPS, threshold) the exact crossing would
      // extrapolate past t1 (badly so on a near-flat plateau), so cap at t1.
      const frac = e1 === e0 ? 1 : Math.min(Math.max((threshold - e0) / (e1 - e0), 0), 1);
      return t0 + frac * (t1 - t0);
    }
  }
  return null;
}

const CAPABILITY_CHART_MARGIN = { top: 16, right: 16, left: 8, bottom: 56 };

function CapabilityChart({ data, scalars, milestones, simMilestones, eToUplift, capability, hidden }: {
  data: DataPoint[]; scalars: PlanAResponse['scalars']; milestones: MilestoneMapLite;
  simMilestones: PlanAResponse['milestones']; eToUplift: (e: number) => number;
  capability: PlanAResponse['capability'];
  hidden: ReadonlySet<PlanATrajectoryKey>;
}) {
  // Milestone anchors: prefer the sim's OWN reference-run milestones — they live in the
  // same E space as the `capability` trajectories, so crossings are exact. /api/compute's
  // map (a separately-integrated run) can drift ~0.01 OOMs, which near the consortium's
  // asymptotic TED-AI plateau would turn into decades of crossing-time error.
  const msE = (key: string): number | undefined => {
    const e = simMilestones?.[key]?.e;
    if (typeof e === 'number') return e;
    const fallback = milestones[key]?.effective_compute_ooms;
    return typeof fallback === 'number' ? fallback : undefined;
  };
  const msT = (key: string): number | undefined => {
    const t = simMilestones?.[key]?.time;
    if (typeof t === 'number') return t;
    const fallback = milestones[key]?.time;
    return typeof fallback === 'number' ? fallback : undefined;
  };
  // Hovered milestone dot → front-page-style tooltip (name + explanation), pinned to the dot.
  const [hoverDot, setHoverDot] = useState<{ marker: DotMarker; cx: number; cy: number; chartWidth: number } | null>(null);
  const xs = data.map((d) => d.x);
  // Start at the present day (2026.0), not the 2017 start of the input series. Crop the
  // right edge to a few years after the covert reaches TED-AI (rather than the full series
  // tail) — but never past the available data.
  const dataMax = Math.max(...xs);
  // With the covert trajectory hidden the late-race tail is uninteresting — cut at 2040.
  const xEnd = hidden.has('covert')
    ? Math.min(dataMax, 2040.0)
    : scalars.covertTtt != null ? Math.min(dataMax, scalars.covertTtt + 3) : dataMax;
  const xDomain: [number, number] = [2026.0, xEnd];
  // Milestone dots (front-page style) on EVERY trajectory. Milestones are levels of
  // effective compute (shared E space), so each trajectory crosses a milestone when its
  // E(t) reaches the threshold; the uplift y = eToUplift(E) is the same for all of them.
  // `tedT` = the backend's authoritative TED-AI crossing time for the trajectory (the same
  // scalar the big date readouts show). The exported used-track E creeps asymptotically to
  // the TED-AI level (only numerically reaching it decades later), so an E-crossing scan
  // would badly overshoot for TED-AI specifically — trust the sim's scalar instead.
  const dotSeries: Array<{ name: string; rows: PlanASeriesRow[]; tedT: number | null | undefined }> = [
    ...(!hidden.has('covert') ? [{ name: 'Covert', rows: capability.covert, tedT: scalars.covertTtt }] : []),
    ...(!hidden.has('consortium') ? [{ name: 'Consortium used', rows: capability.legalUsed, tedT: scalars.legalUsedTtt }] : []),
    ...(scalars.dealBreakdownEnabled && capability.leading && !hidden.has('leading')
      ? [{ name: 'Leading co.', rows: capability.leading, tedT: scalars.leadingTtt }]
      : []),
  ];
  const markers: DotMarker[] = [];
  for (const { key, label } of MILESTONE_SEQUENCE) {
    const e = msE(key);
    if (typeof e !== 'number') continue;
    const y = eToUplift(e);
    if (!isFinite(y)) continue;
    // Default project: use the backend-interpolated milestone time directly.
    const defT = msT(key);
    if (typeof defT === 'number' && !hidden.has('default')) markers.push({ key, label, seriesName: 'Default', x: defT, y });
    for (const { name, rows, tedT } of dotSeries) {
      const t = key === 'TED-AI' && typeof tedT === 'number'
        ? tedT
        : firstCrossingTime(rows, e);
      if (t != null) markers.push({ key, label, seriesName: name, x: t, y });
    }
  }
  // Top the y-axis out a little above ASI's uplift so the full takeoff (AC → ASI) is visible
  // without the late covert/default blow-up dominating the (log) scale.
  const ext = extent(data, ['covertUp', 'legalUsedUp', 'leadingUp']) ?? [1, 100];
  const lo = Math.max(ext[0], 1e-3);
  const asiE = msE('ASI');
  const hi = typeof asiE === 'number' ? eToUplift(asiE) : ext[1];
  const yDomain: [number, number] = [lo / 1.3, hi * 1.3];
  const yTicks = calculateTicks(yDomain, { targetCount: 5 }, 'log');
  // Only dot milestones that fall inside the focused range (e.g. drop ASI if it's above).
  const visibleMarkers = markers.filter((m) =>
    m.x >= xDomain[0] && m.x <= xDomain[1] && m.y >= yDomain[0] && m.y <= yDomain[1]);
  // All of a milestone's dots share a y, so label only one per milestone — the leftmost —
  // and let hover identify the rest.
  const labeledMarker = new Map<string, DotMarker>();
  for (const m of visibleMarkers) {
    const cur = labeledMarker.get(m.key);
    if (!cur || m.x < cur.x) labeledMarker.set(m.key, m);
  }
  const tipFlip = hoverDot != null && hoverDot.cx > hoverDot.chartWidth * 0.55;

  return (
    <div className="relative">
      <CustomLineChart
        data={data}
        height={340}
        margin={CAPABILITY_CHART_MARGIN}
        xDomain={xDomain}
        xTickCount={5}
        xTickFormatter={(v) => v.toFixed(0)}
        xLabel="Year"
        yDomain={yDomain}
        yScale="log"
        yTicks={yTicks}
        yTickFormatter={(v) => fmtUplift(v as number)}
        suppressTooltip={hoverDot != null}
        lines={[
          ...(!hidden.has('default') ? [{ dataKey: 'defUp', stroke: COLOR.default, strokeWidth: 1.5, strokeOpacity: 0.3 }] : []),
          ...(!hidden.has('consortium') ? [{ dataKey: 'legalUsedUp', stroke: COLOR.legalUsed, strokeWidth: 2.4 }] : []),
          ...(!hidden.has('covert') ? [{ dataKey: 'covertUp', stroke: COLOR.covert, strokeWidth: 2.4 }] : []),
          ...(scalars.dealBreakdownEnabled && !hidden.has('leading')
            ? [{ dataKey: 'leadingUp', stroke: COLOR.leading, strokeWidth: 2.4 }]
            : []),
        ]}
        customElements={({ xScale, yScale, chartWidth }) => (
          <g>
            {/* Deal-breakdown moment: dashed vertical marker */}
            {typeof scalars.dealBreakdownT === 'number'
              && scalars.dealBreakdownT >= xDomain[0] && scalars.dealBreakdownT <= xDomain[1] && (
              <g>
                <line
                  x1={xScale(scalars.dealBreakdownT)} x2={xScale(scalars.dealBreakdownT)}
                  y1={yScale(yDomain[0])} y2={yScale(yDomain[1])}
                  stroke={COLOR.leading} strokeWidth={1} strokeDasharray="4 3" opacity={0.6}
                />
                <text
                  x={xScale(scalars.dealBreakdownT) + 4} y={yScale(yDomain[1]) + 12}
                  fontSize={10} fill={COLOR.leading}
                >
                  deal dissolves
                </text>
              </g>
            )}
            {/* Milestone dots on every trajectory (front-page style); hover for details */}
            {visibleMarkers.map((m) => (
              <MilestoneDot
                key={`${m.key}:${m.seriesName}`}
                x={xScale(m.x)} y={yScale(m.y)}
                label={labeledMarker.get(m.key) === m ? m.label : undefined}
                chartWidth={chartWidth}
                onEnter={() => setHoverDot({ marker: m, cx: xScale(m.x), cy: yScale(m.y), chartWidth })}
                onLeave={() => setHoverDot(null)}
              />
            ))}
          </g>
        )}
        tooltip={(p) => (
          <ChartTip year={p.x as number} rows={[
            ...(!hidden.has('covert') ? [['Covert', fmtUpliftOrDash(p.covertUp)] as [string, string]] : []),
            ...(!hidden.has('consortium') ? [['Consortium used', fmtUpliftOrDash(p.legalUsedUp)] as [string, string]] : []),
            ...(scalars.dealBreakdownEnabled && !hidden.has('leading')
              ? [['Leading co.', fmtUpliftOrDash(p.leadingUp)] as [string, string]]
              : []),
            ...(!hidden.has('default') ? [['Default', fmtUpliftOrDash(p.defUp)] as [string, string]] : []),
          ]} />
        )}
      />
      {/* Milestone-dot tooltip: same content as the front-page milestone hover — date,
          uplift, then the milestone's full name + explanation (plus which trajectory). */}
      {hoverDot && (
        <div
          style={{
            position: 'absolute',
            left: CAPABILITY_CHART_MARGIN.left + hoverDot.cx + (tipFlip ? -10 : 10),
            top: CAPABILITY_CHART_MARGIN.top + hoverDot.cy - 10,
            transform: tipFlip ? 'translateX(-100%)' : undefined,
            pointerEvents: 'none',
            zIndex: 20,
          }}
        >
          <div style={tooltipBoxStyle}>
            <span style={tooltipHeaderStyle}>{formatYearMonth(hoverDot.marker.x)}</span>
            <span>
              <span style={{ color: 'var(--dull-foreground)' }}>{hoverDot.marker.seriesName}: </span>
              <span style={tooltipValueStyle}>{fmtUplift(hoverDot.marker.y)}</span>
            </span>
            <div style={{ fontWeight: 600 }}>{MILESTONE_FULL_NAMES[hoverDot.marker.label] ?? hoverDot.marker.label}</div>
            {MILESTONE_EXPLANATIONS[hoverDot.marker.label] && (
              <span style={{ color: 'var(--vivid-foreground)' }}>{MILESTONE_EXPLANATIONS[hoverDot.marker.label]}</span>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
function fmtUpliftOrDash(v: unknown): string {
  return typeof v === 'number' && isFinite(v) ? fmtUplift(v) : '—';
}

// Front-page milestone marker: green dot (white halo) + optional label, nudged to stay
// in-bounds. A transparent r=9 circle widens the hover target for the dot tooltip.
function MilestoneDot({ x, y, label, chartWidth, onEnter, onLeave }: {
  x: number; y: number; label?: string; chartWidth: number;
  onEnter?: () => void; onLeave?: () => void;
}) {
  const nearRight = x >= chartWidth - 40;
  return (
    <g onMouseEnter={onEnter} onMouseLeave={onLeave}>
      <circle cx={x} cy={y} r={9} fill="transparent" />
      <circle cx={x} cy={y} r={4} fill="#2A623D" stroke="white" strokeWidth={1.5} />
      {label && (
        <text
          x={nearRight ? x - 8 : x + 6}
          y={y - 8}
          textAnchor={nearRight ? 'end' : 'start'}
          fontSize={11}
          fontWeight={600}
          fill="#2A623D"
        >
          {label}
        </text>
      )}
    </g>
  );
}

// ---------------------------------------------------------------------------
// Small chart helpers
// ---------------------------------------------------------------------------
function fmtNum(v: number | null | undefined | string): string {
  return typeof v === 'number' && isFinite(v) ? v.toFixed(2) : '—';
}

// Human-readable count (K/M/B/T) — for H100e values, which span orders of magnitude.
function fmtH100e(v: number): string {
  if (!isFinite(v)) return '—';
  const a = Math.abs(v);
  if (a >= 1e12) return `${(v / 1e12).toFixed(2)}T`;
  if (a >= 1e9) return `${(v / 1e9).toFixed(2)}B`;
  if (a >= 1e6) return `${(v / 1e6).toFixed(2)}M`;
  if (a >= 1e3) return `${(v / 1e3).toFixed(1)}K`;
  return v.toFixed(0);
}

// Front-page tooltip: cream box, monospace, bold header (first row), accent-green values.
// Matches the front-page tooltip (chartTooltipStyle): cream box, bold MMM-YYYY header,
// accent-green values. `year` is the hovered x (calendar year) → formatYearMonth.
function ChartTip({ year, rows }: { year: number; rows: Array<[string, string]> }) {
  const shown = rows.filter(([, v]) => v !== '—');
  return (
    <div style={tooltipBoxStyle}>
      <span style={tooltipHeaderStyle}>{formatYearMonth(year)}</span>
      {shown.map(([k, v]) => (
        <span key={k}>
          <span style={{ color: 'var(--dull-foreground)' }}>{k}: </span>
          <span style={tooltipValueStyle}>{v}</span>
        </span>
      ))}
    </div>
  );
}

// Front-page section header: monospace title + rule that fills the row + optional ⓘ tooltip.
function SectionHeader({ title, explanation, right }: { title: string; explanation?: string; right?: ReactNode }) {
  return (
    <div className="flex gap-2 items-center">
      <span className="text-[12px] font-semibold text-primary text-left font-system-mono whitespace-nowrap">{title}</span>
      <div className="flex-1 border-t border-gray-500/30" />
      {right}
      {explanation && (
        <WithChartTooltip explanation={explanation} className="!gap-0"><></></WithChartTooltip>
      )}
    </div>
  );
}
