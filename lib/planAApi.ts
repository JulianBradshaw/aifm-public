// Client-safe covert-race compute call. Kept OUT of lib/serverApi.ts (which imports
// `fs`/`path`/`yaml` and is server-only) so the interactive client component can import
// it without pulling Node built-ins into the browser bundle. Posts to /api/plan-a via a
// relative URL, which Next rewrites to the Flask backend (next.config.ts rewrites).

import { PlanAComputeRequest, PlanAResponse, PlanADefaultsResponse } from '@/app/types';

export class PlanAApiError extends Error {}

/**
 * GET the covert explorer's seed defaults (knob medians, detection grids, silver-bullet
 * block, central model params) from /api/plan-a/defaults via a RELATIVE url, so a browser
 * fetch resolves against the current page origin — exactly like the homepage's
 * /api/compute. This avoids a server-side fetch through getApiBaseUrl(), which on a
 * protection-gated preview deploy falls back to the production backend (which lacks the
 * covert routes). Returns null on any failure — no silent fallback to a different backend.
 */
export async function fetchPlanADefaults(
  signal?: AbortSignal,
): Promise<PlanADefaultsResponse | null> {
  try {
    const response = await fetch('/api/plan-a/defaults', { signal });
    if (!response.ok) return null;
    const data = (await response.json()) as PlanADefaultsResponse;
    return data && data.success ? data : null;
  } catch {
    return null;
  }
}

/**
 * POST one covert-vs-legal race to /api/plan-a. Throws PlanAApiError with the backend's
 * message on a non-2xx (the endpoint returns a clear 400 for invalid knobs / model errors).
 * Pass an AbortSignal to cancel a stale in-flight request when the user changes a knob.
 */
export async function computePlanA(
  request: PlanAComputeRequest,
  signal?: AbortSignal,
): Promise<PlanAResponse> {
  const response = await fetch('/api/plan-a', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
    signal,
  });
  const data = await response.json().catch(() => null);
  if (!response.ok || !data || data.success === false) {
    const message = (data && data.error) || `HTTP ${response.status}`;
    throw new PlanAApiError(message);
  }
  return data as PlanAResponse;
}
