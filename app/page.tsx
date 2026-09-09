import { Suspense } from 'react';
import ProgressChartServer from '@/components/ProgressChartServer';

// Fully static homepage - prerendered with DEFAULT_PARAMETERS
// When users change parameters, they navigate to /p?... for dynamic rendering
export default function Home() {
  return (
    <div className="bg-vivid-background text-vivid-foreground min-h-screen">
      <Suspense fallback={null}>
        <ProgressChartServer />
      </Suspense>
    </div>
  );
}
