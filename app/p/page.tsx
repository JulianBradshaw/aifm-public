import { Suspense } from 'react';
import { Metadata } from 'next';
import { cacheLife, cacheTag } from 'next/cache';
import { loadBenchmarkData } from '@/utils/benchmarkLoader';
import ProgressChartClient from '@/components/ProgressChartClient';
import { fetchComputeData } from '@/lib/serverApi';
import GoogleDocSection from '@/components/GoogleDocSection';
import { DEFAULT_PARAMETERS, ParametersType } from '@/constants/parameters';
import { decodeFullStateFromParams } from '@/utils/urlState';
import { ChartsSkeleton } from '@/components/ChartsSkeleton';

const DEFAULT_SEED = 12345;

// Canonical URL points to homepage - this is just the dynamic parameter handler
export const metadata: Metadata = {
  alternates: {
    canonical: '/',
  },
};

function GoogleDocSkeleton() {
  return (
    <div className="animate-pulse space-y-4 pt-6">
      <div className="h-16 bg-gray-200 rounded w-3/4" />
      <div className="space-y-2">
        <div className="h-4 bg-gray-200 rounded w-full" />
        <div className="h-4 bg-gray-200 rounded w-5/6" />
        <div className="h-4 bg-gray-200 rounded w-4/6" />
      </div>
    </div>
  );
}

// Cached function for chart data - keyed by parameter combination
async function getCachedChartData(parameters: ParametersType, cacheKey: string) {
  'use cache';
  cacheLife('hours');
  cacheTag(`chart-data-${cacheKey}`);
  
  const [benchmarkData, initialComputeData] = await Promise.all([
    Promise.resolve(loadBenchmarkData()),
    fetchComputeData(parameters),
  ]);

  return { benchmarkData, initialComputeData };
}

// Generate a stable cache key from parameters
function getParametersCacheKey(params: Record<string, string | string[] | undefined>): string {
  const sortedEntries = Object.entries(params)
    .filter(([, v]) => v !== undefined)
    .sort(([a], [b]) => a.localeCompare(b));
  
  if (sortedEntries.length === 0) return 'default';
  
  const str = sortedEntries.map(([k, v]) => `${k}=${Array.isArray(v) ? v.join(',') : v}`).join('&');
  return str.slice(0, 100);
}

// Decode URL searchParams into ParametersType
function decodeParametersFromSearchParams(
  searchParams: Record<string, string | string[] | undefined>
): ParametersType {
  const urlParams = new URLSearchParams();
  for (const [key, value] of Object.entries(searchParams)) {
    if (value !== undefined) {
      if (Array.isArray(value)) {
        value.forEach(v => urlParams.append(key, v));
      } else {
        urlParams.set(key, value);
      }
    }
  }
  
  const decoded = decodeFullStateFromParams(urlParams);
  return decoded.parameters;
}

// Dynamic content component that handles searchParams
async function DynamicChartContent({
  searchParams,
}: {
  searchParams: Record<string, string | string[] | undefined>;
}) {
  const hasCustomParams = Object.keys(searchParams).length > 0;
  const parameters = hasCustomParams 
    ? decodeParametersFromSearchParams(searchParams)
    : DEFAULT_PARAMETERS;
  
  const cacheKey = hasCustomParams 
    ? getParametersCacheKey(searchParams) 
    : 'default';
  
  const { benchmarkData, initialComputeData } = await getCachedChartData(parameters, cacheKey);
  
  const googleDocSlot = (
    <Suspense fallback={<GoogleDocSkeleton />}>
      <GoogleDocSection />
    </Suspense>
  );
  
  return (
    <ProgressChartClient
      benchmarkData={benchmarkData} 
      modelDescriptionGDocPortionMarkdown={googleDocSlot}
      initialComputeData={initialComputeData}
      initialParameters={parameters}
      initialSampleTrajectories={[]}
      initialSeed={DEFAULT_SEED}
    />
  );
}

// Wrapper that awaits searchParams inside Suspense boundary
async function SearchParamsResolver({
  searchParams,
}: {
  searchParams: Promise<{ [key: string]: string | string[] | undefined }>;
}) {
  const resolvedParams = await searchParams;
  return <DynamicChartContent searchParams={resolvedParams} />;
}

// Dynamic route for custom parameters
// Users are redirected here when they change parameters from /
export default function ParametersPage({
  searchParams,
}: {
  searchParams: Promise<{ [key: string]: string | string[] | undefined }>;
}) {
  return (
    <div className="bg-vivid-background text-vivid-foreground min-h-screen">
      <Suspense fallback={<ChartsSkeleton />}>
        <SearchParamsResolver searchParams={searchParams} />
      </Suspense>
    </div>
  );
}

