// Resolves which model backend the frontend talks to, per environment.
//
// - local dev (`next dev`): the local Flask server on 5328
// - production Vercel deploy: the canonical domain
// - preview Vercel deploy: the deployment's OWN backend, so branch changes to the
//   Python model are reflected instead of silently showing production's numbers.
//
// Preview deployments sit behind Vercel Deployment Protection, so a server-side
// fetch to the preview's own URL needs a protection-bypass secret (see
// getApiBypassHeaders). Configure it under
// Vercel > Project > Settings > Deployment Protection > Protection Bypass for Automation,
// which exposes the value as VERCEL_AUTOMATION_BYPASS_SECRET.
//
// Caveat: the static homepage prerenders this fetch at build time, when a preview
// deployment's own backend is not yet live. That build-time fetch will fail and the
// data fills in on the first runtime revalidation; the dynamic `/p` route and the
// interactive client paths fetch at request time and reflect the branch immediately.

export function getApiBaseUrl(): string {
  if (process.env.NODE_ENV === 'development') {
    // FLASK_DEV_PORT: see the rewrite in next.config.ts. Unset = 5328.
    return `http://127.0.0.1:${process.env.FLASK_DEV_PORT || '5328'}`;
  }
  if (process.env.VERCEL_ENV === 'production') {
    return 'https://www.aifuturesmodel.com';
  }
  // Non-production (previews AND custom environments like the Plan A embed): always this
  // deployment's OWN backend — its routes match this branch, whereas production's backend
  // may lack them entirely (e.g. /api/plan-a). If the deployment is protection-gated,
  // getApiBypassHeaders() authenticates when the bypass secret is configured; without it
  // the fetch fails and callers degrade gracefully (baked snapshots, client-side
  // recompute) — strictly better than silently rendering another branch's numbers.
  if (process.env.VERCEL_URL) {
    return `https://${process.env.VERCEL_URL}`;
  }
  // No Vercel env at all: fall back to production.
  return 'https://www.aifuturesmodel.com';
}

// Header that lets a server-side fetch reach a protection-gated preview deployment.
// Empty (no bypass) on production and local dev, or when the secret is unset.
export function getApiBypassHeaders(): Record<string, string> {
  const secret = process.env.VERCEL_AUTOMATION_BYPASS_SECRET;
  if (secret && process.env.VERCEL_ENV && process.env.VERCEL_ENV !== 'production') {
    return { 'x-vercel-protection-bypass': secret };
  }
  return {};
}
