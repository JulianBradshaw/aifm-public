// Pure "pause" (freeze + reduce) transform for the input time series.
//
// Extracted from CsvEditorSidebar so the logic is (a) testable without React and
// (b) the single source of truth that the Python counterpart
// (progress_model/pause_transform.py) is golden-tested against. Behaviour is
// unchanged from the original in-component implementation.

export interface InputTimeSeries {
  time: number[];
  L_HUMAN: number[];
  inference_H100e: number[];
  experiment_H100e: number[];
  training_H100e: number[];
  // Optional display-only series (present when the source CSV carries them):
  // total AI-relevant compute in the world, and the leading AI company's total
  // compute (R&D + commercial serving). The core model never reads these.
  world_H100e?: number[];
  total_ai_company_H100e?: number[];
}

// Exponentially interpolate between two positive values at fractional position t in [0,1]
export function expInterp(a: number, b: number, t: number): number {
  if (a <= 0 || b <= 0) return a + (b - a) * t; // linear fallback for non-positive
  return a * Math.pow(b / a, t);
}

export function applyPause(
  base: InputTimeSeries,
  pauseYear: number,
  experimentReduction: number,
  inferenceReduction: number,
  laborReduction: number,
  trainingReduction: number,
): InputTimeSeries {
  // Find the index at or just before pauseYear
  let loIdx = 0;
  for (let i = 0; i < base.time.length; i++) {
    if (base.time[i] <= pauseYear) loIdx = i;
    else break;
  }

  const hiIdx = Math.min(loIdx + 1, base.time.length - 1);
  const tLo = base.time[loIdx];
  const tHi = base.time[hiIdx];
  const frac = tHi > tLo ? (pauseYear - tLo) / (tHi - tLo) : 0;

  // Interpolate the input values at the pause instant. We keep TWO sets:
  //  - *Full*: the original (un-reduced) value AT the pause instant.
  //  - *reduced*: the value held flat AFTER the pause.
  // The pause instant itself carries the FULL value (reduction starts strictly
  // after it). This matters for "pause at milestone": if pauseYear sits just past
  // the milestone's crossing, the inputs up to and including the pause instant are
  // identical to the unpaused run, so the milestone is reached under original
  // inputs and its date is preserved exactly (no feedback / oscillation).
  //
  // training_H100e is raw H100e (exponential), like experiment/inference, so it
  // interpolates in log space and a factor-R cut is a plain DIVIDE by R.
  const freezeTrainingFull = expInterp(base.training_H100e[loIdx], base.training_H100e[hiIdx], frac);
  const freezeExperimentFull = expInterp(base.experiment_H100e[loIdx], base.experiment_H100e[hiIdx], frac);
  const freezeInferenceFull = expInterp(base.inference_H100e[loIdx], base.inference_H100e[hiIdx], frac);
  const freezeLaborFull = expInterp(base.L_HUMAN[loIdx], base.L_HUMAN[hiIdx], frac);

  const freezeTraining = freezeTrainingFull / trainingReduction;
  const freezeExperiment = freezeExperimentFull / experimentReduction;
  const freezeInference = freezeInferenceFull / inferenceReduction;
  const freezeLabor = freezeLaborFull / laborReduction;

  // Sharp step: the pause instant carries the FULL value (so a pause just after
  // a milestone leaves its date invariant), then inputs DROP to the reduced
  // value immediately after via a point at pauseYear + EPS. The reduction does
  // NOT ramp in over the interval to the next CSV grid point. EPS matches the
  // backend (freeze_time_series_at_time) convention.
  const EPS = 1e-6;
  const doStep =
    experimentReduction !== 1 || inferenceReduction !== 1 ||
    laborReduction !== 1 || trainingReduction !== 1;

  const time: number[] = [];
  const training_H100e: number[] = [];
  const experiment_H100e: number[] = [];
  const inference_H100e: number[] = [];
  const L_HUMAN: number[] = [];

  const push = (t: number, tc: number, ec: number, ic: number, lh: number) => {
    time.push(t); training_H100e.push(tc); experiment_H100e.push(ec);
    inference_H100e.push(ic); L_HUMAN.push(lh);
  };

  // Base points strictly before the pause (unchanged).
  for (let i = 0; i < base.time.length; i++) {
    if (base.time[i] < pauseYear) {
      push(base.time[i], base.training_H100e[i], base.experiment_H100e[i],
           base.inference_H100e[i], base.L_HUMAN[i]);
    }
  }
  // The pause instant carries the FULL (un-reduced) value.
  push(pauseYear, freezeTrainingFull, freezeExperimentFull, freezeInferenceFull, freezeLaborFull);
  // Sharp drop to the reduced value immediately after the pause.
  if (doStep) {
    push(pauseYear + EPS, freezeTraining, freezeExperiment, freezeInference, freezeLabor);
  }
  // Base points strictly after the pause, held flat at the reduced value.
  for (let i = 0; i < base.time.length; i++) {
    if (base.time[i] > pauseYear) {
      push(base.time[i], freezeTraining, freezeExperiment, freezeInference, freezeLabor);
    }
  }

  const result: InputTimeSeries = { time, training_H100e, experiment_H100e, inference_H100e, L_HUMAN };

  // Optional display-only series pass through un-reduced and un-frozen (the pause
  // models the leading company cutting its R&D inputs, not world compute shrinking),
  // but must stay aligned with the new time grid, so re-sample them at each output
  // time by log-interpolating the base series.
  const resample = (src: number[]): number[] => {
    return time.map((t) => {
      let lo = 0;
      for (let i = 0; i < base.time.length; i++) {
        if (base.time[i] <= t) lo = i;
        else break;
      }
      const hi = Math.min(lo + 1, base.time.length - 1);
      const tl = base.time[lo];
      const th = base.time[hi];
      const f = th > tl ? Math.min(1, Math.max(0, (t - tl) / (th - tl))) : 0;
      return expInterp(src[lo], src[hi], f);
    });
  };
  if (base.world_H100e && base.world_H100e.length === base.time.length) {
    result.world_H100e = resample(base.world_H100e);
  }
  if (base.total_ai_company_H100e && base.total_ai_company_H100e.length === base.time.length) {
    result.total_ai_company_H100e = resample(base.total_ai_company_H100e);
  }

  return result;
}
