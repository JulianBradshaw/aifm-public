/**
 * Derive copula-implied milestone-arrival ATC columns from a forecaster's
 * AC-arrival ATC plus their takeoff-duration ATCs.
 *
 * Method (exact-copula rank transform):
 *   arrival = AC + takeoff duration, but the sum needs a dependence between
 *   the two. We keep the model's own: each rollout's (AC arrival, duration)
 *   midranks are mapped through the forecaster's ATC inverse CDFs and summed.
 *   Both marginals match the ATCs exactly; the model's rank dependence is
 *   preserved exactly. Rollouts whose rank lands beyond an ATC curve's top
 *   (the curves stop at ~0.9) contribute censored mass that stays off the top
 *   of the implied CDF, matching the site convention.
 *
 * The implied empirical CDF is then represented the way /atc represents every
 * curve: control points evaluated with a monotone I-spline (evaluateSplineAt)
 * on the export grid. The printed control points can be pasted into /atc for
 * manual follow-up edits.
 *
 * Knot placement is --placement (see the Placement type); the default matches
 * /atc's initializeControlPointsFromTakeoff scheme. Smoothing weight is --reg,
 * defaulting to 0.2 rather than /atc's 0.05 for the reason given at that flag.
 *
 * Inputs, per simulation directory app/forecast/data/simulations/<sim>/:
 *   - atc_cdf.csv           AC column = the forecaster's AC-arrival ATC (kept)
 *   - atc_takeoff_cdf.csv   the forecaster's takeoff-duration ATCs
 *   - scripts/export/data/<sim>_ac_ted_asi.json
 *       per-rollout [AC, TED-AI, ASI] arrival triples (null = not achieved).
 *       Extracted from the run's rollouts.cache.json, e.g. on the runpod:
 *         cache = json.load(open(".../outputs/eli_repro/20260811_174746/rollouts.cache.json"))
 *         [[(r["results"]["milestones"].get(k) or {}).get("time")
 *           for k in ("AC","TED-AI","ASI")]
 *          for r in cache["rollouts"] if r.get("results")]
 *
 * Output: rewrites the derived columns of atc_cdf.csv in place.
 *
 * Usage:
 *   npx tsx scripts/export/derive_atc_arrivals.ts [--sim eli-08-10-26]
 *       [--columns TED-AI,ASI] [--points 15] [--reg 0.2]
 *       [--placement quantile|time|sqrt] [--place-bulk 0.95] [--dry-run]
 */
import * as fs from 'fs';
import * as path from 'path';
import { evaluateSplineAt, ControlPoint } from '../../utils/cdfSplineUtils';

interface Args {
  sim: string;
  columns: string[];
  points: number;
  placement: Placement;
  placeBulk: number;
  reg: number;
  dryRun: boolean;
}

function parseArgs(): Args {
  const argv = process.argv.slice(2);
  const get = (flag: string): string | undefined => {
    const i = argv.indexOf(flag);
    return i >= 0 ? argv[i + 1] : undefined;
  };
  return {
    sim: get('--sim') ?? 'eli-08-10-26',
    columns: (get('--columns') ?? 'TED-AI,ASI').split(','),
    points: Number(get('--points') ?? 15),
    // See the Placement doc above. 'quantile' is what the shipped columns were
    // derived with; the others exist to get knots off the mode, where the
    // quantile scheme's bunching lets them track sampling noise.
    placement: (get('--placement') ?? 'quantile') as Placement,
    placeBulk: Number(get('--place-bulk') ?? 0.95),
    // /atc's default is 0.05, but at that weight the spline preserves a
    // ~3.5%-deep density dip near 2028.5 that is Monte-Carlo noise (the dip
    // sits inside the +/-6% sampling band of its ~280-sample window); 0.2 is
    // the smallest sweep value that keeps both densities unimodal, at
    // sup-error 0.89/0.98pp -- still under the DKW noise floor.
    reg: Number(get('--reg') ?? 0.2),
    dryRun: argv.includes('--dry-run'),
  };
}

function readCsv(file: string): { header: string[]; rows: number[][] } {
  const lines = fs.readFileSync(file, 'utf8').replace(/\r/g, '').trim().split('\n');
  const header = lines[0].split(',');
  const rows = lines.slice(1).map(l => l.split(',').map(Number));
  return { header, rows };
}

function column(csv: { header: string[]; rows: number[][] }, name: string): number[] {
  const i = csv.header.indexOf(name);
  if (i < 0) throw new Error(`column ${name} not found (have: ${csv.header.join(', ')})`);
  return csv.rows.map(r => r[i]);
}

/** (rank - 0.5) / N midranks; null/NaN ranked at the top (censored). */
function midranks(values: (number | null)[]): number[] {
  const n = values.length;
  const order = values
    .map((v, i) => [v === null || Number.isNaN(v) ? Infinity : v, i] as [number, number])
    .sort((a, b) => a[0] - b[0]);
  const ranks = new Array<number>(n);
  order.forEach(([, idx], pos) => { ranks[idx] = (pos + 0.5) / n; });
  return ranks;
}

/** Piecewise-linear inverse CDF; ranks beyond the curve's top -> Infinity. */
function makeInverseCdf(F: number[], grid: number[]): (u: number) => number {
  const fMax = F[F.length - 1];
  return (u: number) => {
    if (u > fMax) return Infinity;
    if (u <= F[0]) return grid[0];
    let lo = 0, hi = F.length - 1;
    while (hi - lo > 1) {
      const mid = (lo + hi) >> 1;
      if (F[mid] < u) lo = mid; else hi = mid;
    }
    const span = F[hi] - F[lo];
    const frac = span > 0 ? (u - F[lo]) / span : 0;
    return grid[lo] + frac * (grid[hi] - grid[lo]);
  };
}

/** Empirical CDF of samples (Infinity = censored) over all n, on grid. */
function empiricalCdf(samples: number[], grid: number[]): number[] {
  const finite = samples.filter(Number.isFinite).sort((a, b) => a - b);
  const n = samples.length;
  return grid.map(t => {
    let lo = 0, hi = finite.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (finite[mid] <= t) lo = mid + 1; else hi = mid;
    }
    return lo / n;
  });
}

/**
 * Where to put the knots. All three snap each target to the closest grid point
 * and lock the first.
 *
 *   quantile - even steps in probability, the placement scheme of
 *     initializeControlPointsFromTakeoff. Even in probability means bunched in
 *     time wherever the density is high, i.e. densest exactly where Monte-Carlo
 *     wiggle lives, which is how knots end up tracking noise near the mode.
 *   time - even steps in time across the bulk of the distribution (to `timeMax`,
 *     the quantile given by --place-bulk), plus knots at the ends. Spreads the
 *     knots away from the mode.
 *   sqrt - even steps in sqrt(probability), which pushes targets to higher
 *     quantiles and so thins the knots over the early rise. Unimodal, but it
 *     fits the rise badly.
 *   hybrid - even in time across the bulk, then log-spaced through the tail.
 *     These distributions are a sharp rise plus a decades-long tail, and no
 *     single spacing serves both: even-in-time starves the rise once it has to
 *     stretch over the tail, and even-in-probability crowds the mode. Splitting
 *     the budget lets each regime get the spacing it needs.
 */
type Placement = 'quantile' | 'time' | 'sqrt' | 'hybrid';

function snapToGrid(grid: number[], cdf: number[], targetCdf: number): number {
  let best = 0, bestDiff = Infinity;
  for (let j = 0; j < cdf.length; j++) {
    const diff = Math.abs(cdf[j] - targetCdf);
    if (diff < bestDiff) { bestDiff = diff; best = j; }
  }
  return best;
}

function controlPointsFromCdf(
  grid: number[], cdf: number[], numPoints: number,
  placement: Placement = 'quantile', bulkQuantile = 0.95,
): ControlPoint[] {
  const minCdf = cdf[0];
  const maxCdf = cdf[cdf.length - 1];
  let indices: number[];

  if (placement === 'hybrid') {
    // Two thirds of the budget even in time over the bulk, the rest log-spaced
    // over the tail so a 35-year stretch is not left to a single knot.
    const bulkTarget = minCdf + bulkQuantile * (maxCdf - minCdf);
    const bulkIdx = snapToGrid(grid, cdf, bulkTarget);
    const nBulk = Math.max(2, Math.round(numPoints * (2 / 3)));
    const nTail = Math.max(1, numPoints - nBulk);
    const t0 = grid[0];
    const tBulk = grid[bulkIdx];
    const tEnd = grid[grid.length - 1];

    const nearest = (t: number) => {
      let best = 0, bestDiff = Infinity;
      for (let j = 0; j < grid.length; j++) {
        const diff = Math.abs(grid[j] - t);
        if (diff < bestDiff) { bestDiff = diff; best = j; }
      }
      return best;
    };

    indices = Array.from({ length: nBulk }, (_, i) =>
      nearest(t0 + (tBulk - t0) * (i / (nBulk - 1))));
    // Log-spaced in years past the bulk edge, so tail knots thin out gradually
    const tailSpan = Math.max(tEnd - tBulk, 1e-6);
    for (let i = 1; i <= nTail; i++) {
      const frac = (Math.pow(tailSpan + 1, i / nTail) - 1) / tailSpan;
      indices.push(nearest(tBulk + tailSpan * frac));
    }
  } else if (placement === 'time') {
    // Even in time out to the bulk quantile, then the final point at the end
    const bulkTarget = minCdf + bulkQuantile * (maxCdf - minCdf);
    const bulkIdx = snapToGrid(grid, cdf, bulkTarget);
    const t0 = grid[0];
    const t1 = grid[bulkIdx];
    indices = Array.from({ length: numPoints - 1 }, (_, i) => {
      const t = t0 + (t1 - t0) * (i / (numPoints - 2));
      let best = 0, bestDiff = Infinity;
      for (let j = 0; j < grid.length; j++) {
        const diff = Math.abs(grid[j] - t);
        if (diff < bestDiff) { bestDiff = diff; best = j; }
      }
      return best;
    });
    indices.push(cdf.length - 1);
  } else {
    const shape = placement === 'sqrt'
      ? (u: number) => Math.sqrt(u)
      : (u: number) => u;
    indices = Array.from({ length: numPoints }, (_, i) => {
      const q = minCdf + shape(i / (numPoints - 1)) * (maxCdf - minCdf);
      return snapToGrid(grid, cdf, q);
    });
  }

  // Snapping can collide when targets land close together; keep them distinct
  const seen = new Set<number>();
  const unique = indices.filter(idx => {
    if (seen.has(idx)) return false;
    seen.add(idx);
    return true;
  }).sort((a, b) => a - b);

  return unique.map((idx, i) => ({
    id: `cp-${i}`, x: grid[idx], y: cdf[idx], isLocked: i === 0,
  }));
}

function quantile(grid: number[], cdf: number[], q: number): number | null {
  if (q > cdf[cdf.length - 1]) return null;
  for (let i = 0; i < cdf.length; i++) {
    if (cdf[i] >= q) {
      if (i === 0) return grid[0];
      const span = cdf[i] - cdf[i - 1];
      const frac = span > 0 ? (q - cdf[i - 1]) / span : 0;
      return grid[i - 1] + frac * (grid[i] - grid[i - 1]);
    }
  }
  return null;
}

function main() {
  const args = parseArgs();
  const simDir = path.join('app/forecast/data/simulations', args.sim);
  const arrivalsCsv = readCsv(path.join(simDir, 'atc_cdf.csv'));
  const takeoffCsv = readCsv(path.join(simDir, 'atc_takeoff_cdf.csv'));
  const triples: (number | null)[][] = JSON.parse(
    fs.readFileSync(path.join('scripts/export/data', `${args.sim}_ac_ted_asi.json`), 'utf8'));
  const TRIPLE_COLS = ['AC', 'TED-AI', 'ASI'];

  const grid = column(arrivalsCsv, 'time_decimal_year');
  const durGrid = column(takeoffCsv, 'time_decimal_year');
  const invAc = makeInverseCdf(column(arrivalsCsv, 'AC'), grid);

  const acModel = triples.map(t => t[0]);
  const acRanks = midranks(acModel);
  const acImplied = acRanks.map(invAc);
  console.log(`${args.sim}: ${triples.length} rollouts, deriving ${args.columns.join(', ')} `
    + `(${args.points} control points, ${args.placement} placement, ispline reg ${args.reg})`);

  const derived: Record<string, number[]> = {};
  for (const col of args.columns) {
    const ti = TRIPLE_COLS.indexOf(col);
    if (ti < 0) throw new Error(`no triple column for ${col}`);
    const durModel = triples.map(t =>
      t[ti] === null || t[0] === null ? null : (t[ti] as number) - (t[0] as number));
    const invDur = makeInverseCdf(column(takeoffCsv, col), durGrid);
    const durRanks = midranks(durModel);
    const implied = acImplied.map((a, i) => a + invDur(durRanks[i]));

    const empirical = empiricalCdf(implied, grid);
    const cps = controlPointsFromCdf(grid, empirical, args.points, args.placement, args.placeBulk);
    const spline = grid.map(t => evaluateSplineAt(cps, t, args.reg));
    derived[col] = spline;

    console.log(`\n${col}: max CDF ${spline[spline.length - 1].toFixed(4)} `
      + `(empirical ${empirical[empirical.length - 1].toFixed(4)})`);
    console.log('  control points: '
      + cps.map(p => `(${p.x.toFixed(2)}, ${p.y.toFixed(3)})`).join(' '));
    // Error in probability at fixed year, vs the raw empirical implied CDF.
    // For context: the empirical CDF's own MC sampling band (DKW, 95%) is
    // +/- sqrt(ln(2/.05)/2n) -- representation error below that is free.
    let supErr = 0, supAt = grid[0];
    for (let i = 0; i < grid.length; i++) {
      const e = Math.abs(spline[i] - empirical[i]);
      if (e > supErr) { supErr = e; supAt = grid[i]; }
    }
    const dkw = Math.sqrt(Math.log(2 / 0.05) / (2 * triples.length));
    console.log(`  sup|spline - empirical| = ${(supErr * 100).toFixed(2)}pp at ${supAt.toFixed(1)}`
      + ` (MC noise floor, DKW 95%: +/-${(dkw * 100).toFixed(2)}pp)`);
    if (supErr > dkw) console.log('  WARNING: representation error exceeds MC noise floor');
    for (const q of [0.1, 0.25, 0.5, 0.75, 0.9]) {
      const e = quantile(grid, empirical, q);
      const s = quantile(grid, spline, q);
      if (e !== null && s !== null) {
        console.log(`  q${(q * 100).toFixed(0).padStart(2)}: empirical ${e.toFixed(3)}  `
          + `spline ${s.toFixed(3)}  (shift ${(s - e >= 0 ? '+' : '')}${(s - e).toFixed(3)} yr)`);
      }
    }
    // evaluateSplineAt returns the last control point's y beyond it, so the
    // least-squares fit can step down by a hair at that boundary (an /atc
    // artifact we inherit); tolerate that, fail on anything visible.
    let worstDip = 0;
    for (let i = 1; i < spline.length; i++) {
      worstDip = Math.max(worstDip, spline[i - 1] - spline[i]);
    }
    if (worstDip > 1e-4) throw new Error(`${col}: non-monotone, dip ${worstDip}`);
    if (worstDip > 0) console.log(`  boundary dip ${worstDip.toExponential(2)} (tolerated)`);
    // The implied density should be unimodal: multi-modality at this sample
    // size has always turned out to be control points snapping onto MC noise
    // (raise --reg if this fires).
    const modeAt: number[] = [];
    const modePdf: number[] = [];
    const pdfGrid = grid.filter(t => t <= 2060);
    const pdf = pdfGrid.slice(1).map((t, i) => (spline[i + 1] - spline[i]) / (t - pdfGrid[i]));
    for (let i = 1; i < pdf.length - 1; i++) {
      if (pdf[i] > pdf[i - 1] && pdf[i] >= pdf[i + 1]) {
        modeAt.push(pdfGrid[i + 1]);
        modePdf.push(pdf[i]);
      }
    }
    console.log(`  density modes (to 2060): ${modeAt.length}`
      + (modeAt.length > 1
        ? `  WARNING: not unimodal — at ${modeAt.map((t, i) =>
            `${t.toFixed(2)} (pdf ${modePdf[i].toExponential(2)})`).join(', ')}`
        : ''));
  }

  if (derived['TED-AI'] && derived['ASI']) {
    const viol = Math.max(...grid.map((_, i) => derived['ASI'][i] - derived['TED-AI'][i]));
    console.log(`\nordering: max(ASI cdf - TED cdf) = ${viol.toFixed(6)} (should be <= ~0)`);
  }

  if (args.dryRun) { console.log('\n--dry-run: not writing'); return; }

  const outCols = arrivalsCsv.header;
  const lines = [outCols.join(',')];
  for (let i = 0; i < grid.length; i++) {
    lines.push(outCols.map((c, j) => {
      if (c === 'time_decimal_year') return grid[i].toFixed(6);
      return (derived[c] ? derived[c][i] : arrivalsCsv.rows[i][j]).toFixed(10);
    }).join(','));
  }
  fs.writeFileSync(path.join(simDir, 'atc_cdf.csv'), lines.join('\n') + '\n');
  console.log(`\nwrote ${path.join(simDir, 'atc_cdf.csv')}`);
}

main();
