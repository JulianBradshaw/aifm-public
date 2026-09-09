/**
 * How far each forecast chart can actually show.
 *
 * These bound two things that must agree, or curves silently degrade:
 *   - the charts' auto x-domain (and the zoom bar's range, which is what stops
 *     a viewer reaching past them)
 *   - how much of each series is shipped at full resolution (forecastPayloadTrim)
 *
 * Raising a ceiling therefore means changing it here only.
 */

// The milestone-arrival chart caps its display at 2050, well short of the
// exported grid, which follows each distribution's full support to ~2182.
export const TIMELINE_DISPLAY_MAX_YEAR = 2050;

// The takeoff chart caps at 10 years after AC; its grid runs to ~188 years.
export const TAKEOFF_DISPLAY_MAX_YEARS = 10;
