# MC rollout viewer — and how each artifact counts

Two things live here:

1. **The viewer** (`make_mc_viewer.py`, `summarize_rollouts.py`, `pull_mc_run.sh`) — a
   self-contained browser for batch-rollout runs pulled off the compute box.
2. **A consistency audit** of every place a run states a milestone-arrival statistic. They do
   not all mean the same thing, and two of them genuinely disagree. Read §3 before quoting a
   number out of a run directory.

---

## 1. Using it

```bash
# pull a finished run (summarizes remotely, skips the ~100MB rollouts.jsonl, rebuilds viewer)
scripts/pipelines/pull_mc_run.sh /root/<remote_run>/<YYYYMMDD_HHMMSS> <local_name> [config_name]

# rebuild the viewer from whatever is already in mc_outputs/
.venv/bin/python scripts/pipelines/make_mc_viewer.py

# serve it (file:// also works)
.venv/bin/python -m http.server -d mc_outputs 8010     # → http://127.0.0.1:8010/viewer.html
```

`mc_outputs/` is gitignored. `rollouts.jsonl` is never copied and never loaded whole —
`summarize_rollouts.py` streams it once and emits a ~9 KB `summary.json`, which is the only
per-run payload embedded in the page.

### The topline

The detail pane leads with the **production forecast content** for the run — the milestone
arrival distribution with the same **PDF / CDF toggle** the `/forecast` page has, plus the
statistics table — read verbatim from

```
<run>/forecast_data/milestone_pdfs_overlay_distributions.csv
<run>/forecast_data/milestone_pdfs_overlay_statistics.csv
<run>/forecast_data/milestone_pdfs_overlay_cdf.csv
```

All three are written by `scripts/export/generate_overlay_distributions.py` — the same
generator `forecast_export.py` drives when it populates `app/forecast/data/`.

**It reproduces production's transforms, not just its files.** Plotting the CSV columns
as-is would *not* match the page:

- **PDF** — `page.tsx` renormalizes each milestone so its discrete integral equals that
  milestone's achievement rate: `density = (d / Σd) · (achievement_rate_pct/100) / binWidth`.
  On uplift_eli's AC the raw CSV column integrates to 0.8975 but production shows 0.9028
  (the difference is grid-truncation loss from the shared-grid interpolation). The viewer
  applies the same transform. Axis is `value·100` `%/yr`.
- **CDF** — used **raw**, no renormalization anywhere in the chain. Axis is `value·100` `%`.
  See convention **B′** in §3: each curve plateaus at its achievement rate.

It also matches production's **display language**, taken from
`components/MilestoneDistributionChart.tsx` and confirmed against the live
aifuturesmodel.com/forecast page: `MILESTONE_COLORS` and `MILESTONE_DISPLAY_NAMES`, the
`computeNiceTicks` y-scale, **Month-Year** tick labels and quantiles (never decimal years),
`%/yr` and `%` tick units, dashed gridlines, 2.5px strokes, the `DISPLAY  PDF | CDF` toggle,
the per-series legend line carrying `(p10: …, p50: …, p90: …)`, and the
"estimated based on N simulated trajectories" footnote.

> **Accessibility note.** Production's six milestone colours FAIL the dataviz validator on the
> cream surface (`node scripts/validate_palette.js "#2A623D,#af1e86,#000090,#900000,#D62728,#8C564B"`):
> lightness band FAIL (`#000090`, `#900000`), chroma floor FAIL (`#2A623D`, `#8C564B` read gray),
> CVD separation FAIL (ASI `#8C564B` ↔ TED-AI `#D62728`, ΔE 4.6 protan), normal-vision floor FAIL
> (same pair, ΔE 14.6 — below the 15 hard floor, i.e. hard to tell apart even with full colour
> vision). They are used anyway **because matching production is the requirement**; the legend,
> tooltip and statistics table all name each series in text, so identity never rests on colour
> alone. Fixing this properly means re-stepping the palette in
> `MILESTONE_COLORS` itself, which would change the public site.

**What it covers:** the `/forecast` `MilestoneDistributionChart` (both modes) and its
statistics table. **What it omits:** the `ConditionalMilestoneTimingChart`
(`ac_{2027,2030,2035}_time_until_*.csv` — pulled and linked under data files, just not
charted) and the all-things-considered overlay, which has no per-run source in a rollout dir.

### Generating the topline CSVs (the minimal path)

Do **not** run `forecast_export.py`'s `main()` / `export_simulation()` — it also copies into
`app/forecast/data/simulations/` and rewrites `simulations.json`, which you do not want for a
scratch run. Call just the helper that makes the topline artifacts:

```python
import sys; sys.path.insert(0, "scripts/pipelines")
import forecast_export as fe
from pathlib import Path
d = Path("/root/<run>/<stamp>")
fe.ensure_overlay_distributions(d)   # ← the whole topline: distributions + statistics + cdf
fe.ensure_conditional_timing(d)      # optional: ac_{2027,2030,2035}_time_until_*.csv
fe.ensure_unconditional_timing(d)    # optional: ac_unconditional_time_until_*.csv
```

Then rsync the CSVs into `mc_outputs/<name>/<stamp>/forecast_data/`.

**Measured cost** on a 500-rollout run (uplift_eli, forcing regeneration): overlay **0.8 s**,
conditional timing **1.3 s**, unconditional **0.0 s**. Nothing here is slow enough to defer.
The reason is that `RolloutsReader` reads `rollouts.cache.json` (~12 MB) rather than
`rollouts.jsonl` (~130 MB); **a run with no cache file will be far slower**, because the
reader falls back to the full JSONL. If a run crawls, check for `rollouts.cache.json` first.

`milestone_statistics_summary.csv` exists in `app/forecast/data/` but **nothing in the repo
writes it and `page.tsx` never reads it** — `copy_data_files()` does not list it either. It
appears to be an orphan; the topline does not attempt to reproduce it.

---

## 2. The one distinction that matters

A rollout can end in three states:

| state | what happened | count for uplift_eli (n=500) |
|---|---|---|
| **failed** | `results` is null; `error` says why | 37 |
| **ran, no arrival** | ran fine, `aa_time` null/non-finite — never reached the milestone | 45 |
| **arrived** | finite arrival time | 418 |

Every statistic in a run directory picks a denominator and a censoring rule over those three
buckets, and **they do not all pick the same one**:

- **Arrival-conditioned** — drop failed *and* no-arrival, quantile the 418. Answers "when,
  given it happens". Cannot express "it might not happen". **Nothing in the repo uses this any
  more** (it was the summarizer's original convention; removed in favour of the one below).
- **Censored** — substitute the simulation end (~2050) for no-arrival rollouts, then
  quantile. Answers "when, unconditionally", with never-arrives folded in as "very late".
  A p90 at ~2050 means *"≥10% never get there"*, not *"arrives in 2050"*.
- **Failed rollouts are excluded from the denominator by every production artifact.**

---

## 3. Discrepancy table — uplift_eli / 20260807_172703

500 sampled · 37 failed · 463 non-failed · 418 reached AC · 45 ran without reaching it ·
`simulation_end` = 2050.0. Ground truth recomputed directly from `rollouts.jsonl`:

| convention | n | AC p10 | AC p50 | AC p90 |
|---|---|---|---|---|
| A — arrival-conditioned (drop failed + no-AC) | 418 | 2026.9371 | **2028.4237** | **2035.8916** |
| B — censor no-AC at sim_end, failed excluded | 463 | 2026.9662 | **2028.7253** | **2049.4103** |
| C — censor no-AC *and* failed at sim_end | 500 | 2027.0009 | 2029.0441 | 2050.0000 |

Where each representation lands:

| # | representation | n used | censoring | AC p50 | AC p90 | conv. |
|---|---|---|---|---|---|---|
| 1 | `forecast_data/…overlay_statistics.csv` (production `/forecast`) | 463 | no-AC → median `sim_end` | 2028.7253 | 2049.4103 | **B** |
| 2 | `milestone_summary.csv` — `Median Arrival` | 463 | no-AC → *per-rollout* `simulation_end` | 2028.7 | — | **B** |
| 3 | `milestone_summary.csv` — `Achieved/Total Runs/Rate` | **418** | n/a | — | — | **mismatched** |
| 4 | `milestone_time_hists/*.png` — quantile lines | 463 | no-AC → median `sim_end` | 2028.73 | 2049.41 | **B** |
| 5 | `milestone_pdfs_overlay.png` | 463 | no-AC → median `sim_end` | 2028.7253 | 2049.4103 | **B** |
| 6 | `summary.json` `ac_time` (this viewer's summarizer) | 463 | no-AC → own `simulation_end` | 2028.7253 | 2049.4103 | **B** |
| 7 | `rollout_summary.txt` | 463 | no-AC → own `simulation_end` | 2028.7253 | 2049.4103 | **B** |

Rows 6 and 7 used to be convention **A** (arrival-conditioned, n=418, p50 2028.4237 /
p90 2035.8916). That was the original summarizer spec — "quantiles of `aa_time` *where
finite*" — written before the audit established that production censors. It meant the viewer
displayed two numbers called "AC p90" that were 13.5 years apart. **Both now censor**, so
every artifact in the repo states one statistic. Cross-checked on all four pulled runs: the
stdlib summarizer and production's independent numpy path agree to 4 decimals on AC p10/p50/p90
*and* the achievement rate —

| run | p10 | p50 | p90 | rate | matches `overlay_statistics.csv` |
|---|---|---|---|---|---|
| uplift_eli | 2026.9662 | 2028.7253 | 2049.4103 | 90.28% | yes |
| uplift_daniel | 2026.8470 | 2027.9953 | 2033.3584 | 98.03% | yes |
| diag_5050_v2 | 2026.9712 | 2029.3855 | 2050.0000 | 89.23% | yes |
| diag_5050_v1 | 2027.1846 | 2029.5144 | 2044.6957 | 92.43% | yes |

(One residual difference is invisible here: the summarizer substitutes each rollout's *own*
`simulation_end`, production substitutes the *median* across rollouts — see F6. Every rollout
in these runs ends at 2050.0, so they coincide. They would diverge on a run with ragged ends.)

The arrival-conditioned numbers are no longer published anywhere. `n_with_ac`, `n_no_ac` and
`n_failed` are, which is the diagnostic that actually mattered — a censored p90 plus "45 of 463
never arrive" says everything the conditional p90 said, without a second number competing for
the name.
| 8 | `…overlay_cdf.csv` → `/forecast` CDF mode | 463 | plateaus at achievement rate | — | — | **B′** |

**B′ (the CDF)** is convention B expressed cumulatively: the curve is
`count(t_i <= x) / total_rollouts` with `total_rollouts = achieved + not_achieved = 463`
(`generate_overlay_distributions.compute_empirical_cdf`). Verified empirically — every
milestone's final CDF value equals its `achievement_rate_pct` to 5 decimals:

| milestone | CDF plateau | achieved/463 | saturates at 1.0? |
|---|---|---|---|
| AC | 0.90281 | 418/463 | no |
| AI2027-SC | 0.85529 | 396/463 | no |
| SAR-level-… | 0.85529 | 396/463 | no |
| SIAR-level-… | 0.76458 | 354/463 | no |
| TED-AI | 0.77322 | 358/463 | no |
| ASI | 0.65659 | 304/463 | no |

So the production CDF is **censoring-honest about no-arrival mass** (the gap to 100% is
visible as a plateau, not renormalized away) but **still blind to failures** (F1): AC's
plateau reads 90.28%, where the sampled-space figure is 83.6%. `page.tsx` uses the CSV values
raw (`cdf: cdfValue`, line 346) and `MilestoneDistributionChart.tsx` plots them unscaled
(`lookup.set(point.year, point.cdf)`, `yTickFormatter value*100 %`) — no renormalization
anywhere in the chain. The viewer reproduces this exactly and labels it.

Rows 6 and 7 share one accumulator (`summarize_rollouts.ModeAcc`), so they cannot disagree.
Rows 1/4/5 share `RolloutsReader.read_milestone_times_batch` and agree exactly.

### Findings

**F1 — Failed rollouts vanish from every production denominator. (loud)**
`RolloutsReader` skips any record whose `results` is not a dict, so failures are not counted
as "not achieved" — they are not counted at all. For uplift_eli the overlay reports AC
achievement **90.28% (418/463)**; over the parameter space actually sampled it is **83.6%
(418/500)**. This is worst on `diag_5050_v1`, where 130 of 500 failed: the overlay reports
**92.43%** achievement where the honest figure is **68.4%** — a 24-point overstatement.
The failures are *not* missing-at-random: every one is the `uplift-solved` "could not find a
logistic schedule" solve failure, i.e. a contiguous region of parameter space. Dropping them
is informative censoring and biases the published distribution by an unknown amount.
*Nothing in the pipeline currently reports the 500-relative rate.* The viewer now prints
"N failed rollout(s) are excluded from total_rollouts entirely (463 counted, 500 sampled)"
above the topline table.

**F2 — `milestone_summary.csv` is internally inconsistent. (loud)**
`Median Arrival` is computed over all 463 non-failed rollouts with censoring, but `Total Runs`
is hardcoded to the *AC-achieving* count (`scripts/utilities/milestone_summary_table.py:203`,
`total_runs = len(arrival_times["AC"])` → 418). So within one row the median and the rate use
different populations. Consequences:
- AC's `Achievement Rate` is **100.0% by construction** — it is 418/418.
- SAR's rate is 396/418 = **94.7%**, while `milestone_pdfs_overlay_statistics.csv` reports
  396/463 = **85.53%** for the same milestone in the same run. Two artifacts, one run, one
  milestone, two different achievement rates — and neither is the 500-relative 79.2%.
This is a genuine disagreement, not binning noise. It is a reporting bug in
`milestone_summary_table.py`, left unfixed here (out of scope) and flagged.

**F3 — Censoring choice moves AC p90 by 13.5 years.** 2035.89 (conditional) vs 2049.41
(censored) on identical data. p50 moves only 0.30 yr. Quote p90 only with its convention
attached. The viewer labels its own quantile table `arrival-conditioned` and spells out the
count in a note.

**F4 — The overlay's curve and its table are fit on slightly different sets.** The KDE drops
achieved times before `present_day` (`valid_mask = raw_data >= boundary`) but the quantiles
use unfiltered `raw_data`. For uplift_eli that is 4 of 418 samples (earliest AC 2024.69, i.e.
before the 2026.096 present day). Small, but the plotted mode and the tabulated p10 are not
from the same sample.

**F5 — Two independent implementations of the overlay.**
`scripts/plots/pdfs/milestone_overlay.py` draws the PNG; `scripts/export/generate_overlay_distributions.py`
writes the CSVs. Same conventions and same numbers, but duplicated code, so they can drift.
The CSV differs in output shape only: a monthly grid (with `--monthly`) vs 2048-point
linspace, and all series interpolated onto the *first* milestone's grid with `left=0, right=0`,
which truncates any milestone whose support runs past AC's grid.

**F6 — Two different sim-end substitutions.** `milestone_summary_table.py` substitutes each
rollout's own `simulation_end`; the overlay substitutes the *median* `simulation_end` for all
censored rollouts. Identical here (every sim ends at 2050) but they would diverge if runs
ended at different times.

**F7 — Sentinel values in the conditional-timing CSVs.** `ac_*_time_until_statistics.csv` uses
`1000000.0000` for a quantile that is never reached, and its `total_rollouts` is 418 —
conditioned on AC arrival, since it measures time *from* AC. Do not plot it raw.

### Not a discrepancy

`summary.json` / `rollout_summary.txt` reporting 2028.42 where production reports 2028.73 is
**correct behaviour of two different statistics**, not a bug. The summarizer is deliberately
arrival-conditioned: it is a run-health tool that also reports `n_failed` and `n_no_ac`
separately, so the censored mass is visible rather than baked into a quantile. The viewer
labels it as such.
