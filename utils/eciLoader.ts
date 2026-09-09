import {
  eciFrontierRows,
  EPOCH_ECI_ATTRIBUTION,
  EPOCH_ECI_LATEST_RELEASE_DATE,
  EPOCH_ECI_SOURCE_URL,
} from './eciData';

export { EPOCH_ECI_ATTRIBUTION, EPOCH_ECI_LATEST_RELEASE_DATE, EPOCH_ECI_SOURCE_URL };

/**
 * One frontier model, in the coordinates the Epoch Capabilities Index chart plots in:
 * decimal year on x, ECI score on y.
 *
 * Purely observational. The ECI trend is anchored on `eci_at_present_day` and
 * `eci_growth_rate` alone (see progress_model/ac_anchor.py fit_eci_trend); nothing
 * here reaches the model.
 *
 * "Frontier" means the running maximum of score over release date -- one point per
 * date, strictly increasing, all labs competing. The reduction happens at generation
 * time in scripts/bridge/epoch_eci_to_ts.py, which documents the rule.
 */
export interface EciFrontierPoint {
  year: number;
  eciScore: number;
  organization: string;
  displayName: string;
  /** ISO release date. */
  releaseDate: string;
}

// ISO yyyy-mm-dd -> decimal year. Parsed field-wise rather than via `new Date(...)`
// so the result does not shift with the viewer's timezone.
function isoDateToDecimalYear(isoDate: string): number {
  const [year, month, day] = isoDate.split('-').map(Number);
  const startOfYear = Date.UTC(year, 0, 1);
  const startOfNextYear = Date.UTC(year + 1, 0, 1);
  const elapsed = Date.UTC(year, month - 1, day) - startOfYear;
  return year + elapsed / (startOfNextYear - startOfYear);
}

/** ECI frontier datapoints for the capability chart overlay, oldest release first. */
export function loadEciFrontierData(): EciFrontierPoint[] {
  return eciFrontierRows
    .filter(row => Number.isFinite(row.eciScore))
    .map(row => ({
      year: isoDateToDecimalYear(row.releaseDate),
      eciScore: row.eciScore,
      organization: row.organization,
      displayName: row.displayName,
      releaseDate: row.releaseDate,
    }))
    .sort((a, b) => a.year - b.year);
}
