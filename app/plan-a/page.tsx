import { cacheLife, cacheTag } from 'next/cache';
import PlanAExplorerClient, { PlanABakedResult } from '@/components/PlanAExplorerClient';
import { PlanADefaultsResponse } from '@/app/types';
import { fetchPlanAExplorerData } from '@/lib/serverApi';
import planADefaultsBaked from '@/constants/planADefaults.json';
import planADefaultResultBaked from '@/constants/planADefaultResult.json';

// The Plan A explorer, prerendered WITH data the same way /p does it: the seed defaults
// and the default compute result are fetched from the model backend inside a 'use cache'
// function (revalidated hourly, stale-while-revalidate), so the served page always ships
// rendered charts. When the backend is unreachable — next build before Flask is up, or a
// misconfigured deploy — we fall back to the baked snapshots in constants/ (regenerate
// with: npx tsx scripts/export/bake_plan_a_defaults.ts). The client silently revalidates
// against the live backend on mount either way, and URL param overrides apply client-side,
// keeping this route static. The ai-2030 Plan A site iframes /plan-a?embed=1; the Takeoff
// Supplement writeup is intentionally NOT public — the ai-2030 site hosts it.
async function getCachedExplorerData() {
  'use cache';
  cacheLife('hours');
  cacheTag('covert-explorer-data');
  return fetchPlanAExplorerData();
}

export default async function PlanAPage() {
  const live = await getCachedExplorerData();
  const defaults = (live?.defaults ?? planADefaultsBaked) as unknown as PlanADefaultsResponse;
  const initialResult = (live?.result ?? planADefaultResultBaked) as unknown as PlanABakedResult;
  return (
    <div className="bg-vivid-background text-vivid-foreground min-h-screen">
      <PlanAExplorerClient initialDefaults={defaults} initialResult={initialResult} />
    </div>
  );
}
