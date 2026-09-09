import {
  observedRevenueRows,
  EPOCH_REVENUE_ATTRIBUTION,
  EPOCH_REVENUE_SOURCE_URL,
} from './revenueData';
import { observedOpenAiRevenueRows } from './revenueDataOpenai';

export { EPOCH_REVENUE_ATTRIBUTION, EPOCH_REVENUE_SOURCE_URL };

const allObservedRevenueRows = [...observedRevenueRows, ...observedOpenAiRevenueRows];

/**
 * One reported revenue figure, in the coordinates the Predicted AI Revenue chart
 * plots in: decimal year on x, annual revenue in US DOLLARS on y.
 *
 * Purely observational. Nothing here reaches the model -- `present_revenue` and
 * `present_revenue_growth_rate` remain user-set parameters, and the anchored trend
 * is fit from those alone (see progress_model/ac_anchor.py fit_revenue_trend).
 */
export interface ObservedRevenuePoint {
  year: number;
  /** Annual revenue in US dollars, e.g. 4.7e10 for $47B. */
  revenueUsd: number;
  company: string;
  /** ISO date the figure describes. */
  date: string;
  revenueType: string;
  confidence: string;
  sourceType: string;
  sourceUrl: string;
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

/**
 * Observed annual-revenue datapoints for the revenue chart overlay.
 *
 * @param company Restrict to one company (e.g. 'Anthropic', the company the
 *                model's `present_revenue` default is anchored on). Omitted:
 *                every company's datapoints, for a multi-lab overlay.
 */
export function loadObservedRevenueData(company?: string): ObservedRevenuePoint[] {
  return allObservedRevenueRows
    .filter(row => company == null || row.company === company)
    .filter(row => Number.isFinite(row.annualizedRevenueUsd) && row.annualizedRevenueUsd > 0)
    .map(row => ({
      year: isoDateToDecimalYear(row.date),
      revenueUsd: row.annualizedRevenueUsd,
      company: row.company,
      date: row.date,
      revenueType: row.revenueType,
      confidence: row.confidence,
      sourceType: row.sourceType,
      sourceUrl: row.sourceUrl,
    }))
    .sort((a, b) => a.year - b.year);
}
