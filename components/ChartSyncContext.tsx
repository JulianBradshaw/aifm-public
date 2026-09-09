'use client';

import { createContext, useContext, useState, useCallback, ReactNode } from 'react';

interface ChartSyncState {
  hoveredX: number | null;
  setHoveredX: (x: number | null) => void;
}

const ChartSyncContext = createContext<ChartSyncState | undefined>(undefined);

export function ChartSyncProvider({ children }: { children: ReactNode }) {
  const [hoveredX, setHoveredX] = useState<number | null>(null);

  const handleSetHoveredX = useCallback((x: number | null) => {
    setHoveredX(x);
  }, []);

  return (
    <ChartSyncContext.Provider value={{ hoveredX, setHoveredX: handleSetHoveredX }}>
      {children}
    </ChartSyncContext.Provider>
  );
}

export function useChartSync() {
  const context = useContext(ChartSyncContext);

  // Return safe defaults if used outside provider (chart will work without sync)
  if (context === undefined) {
    return {
      hoveredX: null,
      setHoveredX: () => {}, // no-op
    };
  }

  return context;
}
