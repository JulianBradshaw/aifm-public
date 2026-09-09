/**
 * Bake the Plan A explorer's seed defaults AND its default compute result into
 * constants/, so the page statically prerenders with real charts (no loading gate).
 *
 *   npx tsx scripts/export/bake_plan_a_defaults.ts [flaskBaseUrl]
 *
 * Requires the Flask backend (default http://localhost:5328). Run after any change to
 * the covert experiment config or the model physics. This replicates the client's
 * first-compute request EXACTLY (same knob seeding + parameter conversion as
 * PlanAExplorerClient), so the baked result matches what the client's silent
 * revalidation recomputes on mount.
 *
 * Writes:
 *   constants/planADefaults.json       — /api/plan-a/defaults payload (UI seeds)
 *   constants/planADefaultResult.json  — { result, series, milestones } for those seeds
 */
import { writeFileSync, statSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';
import { DEFAULT_PARAMETERS } from '../../constants/parameters';
import { BREAKDOWN_DEFAULTS } from '../../constants/planASeeds';
import { convertParametersToAPIFormat, ParameterRecord } from '../../utils/monteCarlo';

const base = process.argv[2] ?? 'http://localhost:5328';
const repoRoot = join(dirname(fileURLToPath(import.meta.url)), '..', '..');

async function getJson(url: string): Promise<Record<string, unknown>> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`GET ${url} -> HTTP ${res.status}`);
  return res.json();
}

async function postJson(url: string, body: unknown): Promise<Record<string, unknown>> {
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  const data = await res.json().catch(() => null);
  if (!res.ok || !data || data.success === false) {
    throw new Error(`POST ${url} -> HTTP ${res.status}: ${(data && data.error) || 'no body'}`);
  }
  return data;
}

async function main() {
  const defaults = await getJson(`${base}/api/plan-a/defaults`);
  if (!defaults.success) throw new Error('/api/plan-a/defaults returned success=false');

  // Mirror PlanAExplorerClient's seeding: knobs = BREAKDOWN_DEFAULTS overlaid by config
  // knobs; parameters = website defaults overlaid by the config's model params, converted
  // through the same API-format transform the client uses.
  const knobs = { ...BREAKDOWN_DEFAULTS, ...(defaults.knobs as Record<string, unknown>) };
  const parameters = convertParametersToAPIFormat({
    ...DEFAULT_PARAMETERS,
    ...(defaults.model_params as Record<string, unknown> ?? {}),
  } as unknown as ParameterRecord);

  console.log('computing /api/plan-a (takes a few seconds)...');
  const result = await postJson(`${base}/api/plan-a`, {
    parameters,
    knobs,
    detection_grids: defaults.detection_grids,
    silver_bullet: defaults.silver_bullet ?? null,
  });

  console.log('computing /api/compute reference series...');
  const dflt = await postJson(`${base}/api/compute`, {
    parameters,
    time_range: [2017, 2100],
    initial_progress: 0.0,
  });

  const defaultsPath = join(repoRoot, 'constants', 'planADefaults.json');
  const resultPath = join(repoRoot, 'constants', 'planADefaultResult.json');
  writeFileSync(defaultsPath, JSON.stringify(defaults, null, 1) + '\n');
  const baked = {
    result,
    series: dflt.time_series ?? [],
    milestones: dflt.milestones ?? {},
  };
  // Full-precision doubles double the payload for no visual gain; 6 significant digits
  // is far below chart-pixel resolution. The client's silent revalidation on mount
  // replaces this with the full-precision live result anyway.
  const rounded = JSON.stringify(baked, (_k, v) =>
    typeof v === 'number' && Number.isFinite(v) && !Number.isInteger(v)
      ? Number(v.toPrecision(6))
      : v);
  writeFileSync(resultPath, rounded + '\n');
  const mb = (p: string) => `${(statSync(p).size / 1024).toFixed(0)}KB`;
  console.log(`wrote ${defaultsPath} (${mb(defaultsPath)})`);
  console.log(`wrote ${resultPath} (${mb(resultPath)})`);
}

main().catch((e) => { console.error(e); process.exit(1); });
