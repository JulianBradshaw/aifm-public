import { Suspense } from 'react';
import { cacheLife } from 'next/cache';
import { loadBenchmarkData } from '../utils/benchmarkLoader';
import ProgressChartClient from './ProgressChartClient';
import { fetchComputeData } from '@/lib/serverApi';
import GoogleDocSection from './GoogleDocSection';
import { DEFAULT_PARAMETERS } from '@/constants/parameters';

const DEFAULT_SEED = 12345;

// Loading skeleton for the Google Doc section (below the fold)
function GoogleDocSkeleton() {
  return (
    <div className="animate-pulse space-y-4 pt-6">
      <div className="h-16 bg-gray-200 rounded w-3/4" />
      <div className="space-y-2">
        <div className="h-4 bg-gray-200 rounded w-full" />
        <div className="h-4 bg-gray-200 rounded w-5/6" />
        <div className="h-4 bg-gray-200 rounded w-4/6" />
      </div>
      <div className="space-y-2 pt-4">
        <div className="h-4 bg-gray-200 rounded w-full" />
        <div className="h-4 bg-gray-200 rounded w-full" />
        <div className="h-4 bg-gray-200 rounded w-3/4" />
      </div>
    </div>
  );
}

// Cached chart data for the default parameters, so the homepage ships with the
// charts already populated instead of waiting on a client-side /api/compute.
//
// This must resolve to the CURRENT deployment's backend. On a protection-gated
// preview, a prerender that reaches PRODUCTION bakes production numbers into every
// branch's homepage — which is why getApiBaseUrl() points previews at their own
// backend (authenticated via getApiBypassHeaders) and a failed fetch returns
// { success: false } rather than throwing. See lib/apiBase.ts for the build-time
// caveat on previews.
//
// The client still recomputes on mount regardless (see ProgressChartClient), so
// even a stale or empty cached result is corrected per-visit.
async function getCachedDefaultChartData() {
  'use cache';
  cacheLife('hours');

  const [benchmarkData, initialComputeData] = await Promise.all([
    Promise.resolve(loadBenchmarkData()),
    fetchComputeData(DEFAULT_PARAMETERS),
  ]);

  return { benchmarkData, initialComputeData };
}

// Static server component - always renders with DEFAULT_PARAMETERS
// For custom parameters, use the /p route instead
export default async function ProgressChartServer() {
  const { benchmarkData, initialComputeData } = await getCachedDefaultChartData();

  // Google Doc is wrapped in Suspense so it streams in after the main chart
  const googleDocSlot = (
    <Suspense fallback={<GoogleDocSkeleton />}>
      <GoogleDocSection />
    </Suspense>
  );

  return (
    <ProgressChartClient
      benchmarkData={benchmarkData}
      modelDescriptionGDocPortionMarkdown={googleDocSlot}
      initialComputeData={initialComputeData.success ? initialComputeData : undefined}
      initialParameters={DEFAULT_PARAMETERS}
      initialSampleTrajectories={[]}
      initialSeed={DEFAULT_SEED}
    />
  );
}
