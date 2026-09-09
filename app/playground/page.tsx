import { Suspense } from 'react';
import { cacheLife } from 'next/cache';
import { loadBenchmarkData } from '@/utils/benchmarkLoader';
import { fetchComputeData } from '@/lib/serverApi';
import PlaygroundClient from '@/components/PlaygroundClient';
import { decodeFullStateFromSearchParamsRecord } from '@/utils/urlState';
import { ParametersType } from '@/constants/parameters';

const DEFAULT_SEED = 12345;

interface PlaygroundPageProps {
  searchParams: Promise<Record<string, string | string[] | undefined>>;
}

// Cached component - cache key is automatically derived from parameters
async function CachedPlayground({ parameters }: { parameters: ParametersType }) {
  'use cache';
  cacheLife('hours');
  
  const benchmarkData = loadBenchmarkData();
  const initialComputeData = await fetchComputeData(parameters);

  return (
    <PlaygroundClient 
      benchmarkData={benchmarkData}
      initialComputeData={initialComputeData}
      initialParameters={parameters}
      initialSampleTrajectories={[]}
      initialSeed={DEFAULT_SEED}
    />
  );
}

// Outer component handles searchParams resolution
async function PlaygroundServer({ searchParams }: { searchParams: Promise<Record<string, string | string[] | undefined>> }) {
  const resolvedSearchParams = await searchParams;
  const { parameters } = decodeFullStateFromSearchParamsRecord(resolvedSearchParams);
  
  return <CachedPlayground parameters={parameters} />;
}

export default function PlaygroundPage({ searchParams }: PlaygroundPageProps) {
  return (
    <Suspense fallback={null}>
      <PlaygroundServer searchParams={searchParams} />
    </Suspense>
  );
}

