// Routing predicate for legacy (Dec-2025-era) share links.
//
// Share links minted by the current model carry `v` (LINK_FORMAT_VERSION in
// utils/urlState.ts). At the moment the Aug-2026 model went live, zero
// `v`-bearing links existed in the wild — so any marker-less /p or
// /playground link that carries recognized share-link content is, by
// definition, a link minted against the Dec-2025 model. Those links are
// 302-redirected (by middleware.ts) to the frozen Dec-2025 deployment at
// LEGACY_HOST, which interprets them exactly as production did when they
// were minted.
//
// The predicate references ONLY the Dec-2025 codec's key set, which is frozen
// forever — nothing here needs updating when the current model adds
// parameters or presets.
//
// Known accepted trade-off: a hand-edited URL whose `v` was manually stripped
// (e.g. `?base=daniel-08-07-26&suard=2.5`) also redirects to the legacy
// deployment, which falls back to its default baseline for the unknown base
// and ignores unknown keys — the same behavior any garbled link gets today.
// Rescuing such URLs would require a keep-in-sync list of new-only keys;
// deliberately not worth it (decided 2026-08-12).

export const LEGACY_HOST = 'https://v1.aifuturesmodel.com';

// The COMPLETE query-key surface of main's (Dec-2025) share-link codec, taken
// verbatim from `git show main:utils/urlState.ts`:
// PARAM_ABBREVIATIONS (43) + CHECKBOX_ABBREVIATIONS (6) + 'hp'.
const OLD_ERA_KEYS = new Set([
  // PARAM_ABBREVIATIONS
  'tst', 'pdt', 'acth', 'ddgf', 'rcl', 'rec', 'aec', 'diec', 'rsw', 'spry',
  'cln', 'ece', 'cle', 'afca', 'ait', 'swm', 'aa', 'artc', 'arts', 'paa',
  'pd', 'ph', 'het', 'ila', 'ica', 'laec', 'caec', 'icaec', 'bgm', 'gy',
  'caes', 'mscl', 'mttm', 'tp', 'sam', 'tam', 'ocei', 'ctcgr', 'sdy',
  'pstcgr', 'ala', 'tl', 'tls',
  // CHECKBOX_ABBREVIATIONS
  'eca', 'eea', 'uetc', 'esp', 'uclgs', 'uvhd',
  // horizon percentile
  'hp',
]);

// Paths on which share links are minted (both use the same codec).
export const SHARE_LINK_PATHS = new Set(['/p', '/playground']);

/**
 * True iff this query string identifies a Dec-2025-era share link that must
 * be served by the frozen legacy deployment.
 *
 * Rules (exact for every link that existed when the marker was introduced):
 *   1. `v` present (any value)                          -> current site
 *   2. `base` present, or any Dec-2025 codec key present -> legacy
 *   3. anything else (empty, utm_* junk only)            -> current site
 */
export function isLegacyShareLink(searchParams: URLSearchParams): boolean {
  if (searchParams.has('v')) return false;
  if (searchParams.has('base')) return true;
  for (const key of searchParams.keys()) {
    if (OLD_ERA_KEYS.has(key)) return true;
  }
  return false;
}
