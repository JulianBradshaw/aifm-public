import path from 'node:path';
import { readFile } from 'node:fs/promises';
import { Suspense } from 'react';
import { Simulation } from '@/components/SimulationSelector';
import { AtcPageClient } from './AtcPageClient';

interface SimulationRegistry {
  version: number;
  defaultSimulation: string | null;
  simulations: Simulation[];
}

async function loadSimulationRegistry(): Promise<SimulationRegistry> {
  const registryPath = path.join(process.cwd(), 'app/forecast/data/simulations/simulations.json');
  try {
    const raw = await readFile(registryPath, 'utf8');
    return JSON.parse(raw) as SimulationRegistry;
  } catch {
    return {
      version: 1,
      defaultSimulation: null,
      simulations: [],
    };
  }
}

function LoadingFallback() {
  return (
    <div className="flex min-h-screen items-center justify-center text-sm text-slate-500">
      Loading ATC Editor...
    </div>
  );
}

async function AtcPageContent() {
  const registry = await loadSimulationRegistry();

  return (
    <AtcPageClient
      simulations={registry.simulations}
      defaultSimulationId={registry.defaultSimulation}
    />
  );
}

export default function AtcPage() {
  return (
    <Suspense fallback={<LoadingFallback />}>
      <AtcPageContent />
    </Suspense>
  );
}

export const metadata = {
  title: 'ATC Distribution Editor',
  description: 'Create all-things-considered distributions from empirical forecasts',
};
