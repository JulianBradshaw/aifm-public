"use client";

import Link from "next/link";
import { useEffect } from "react";

interface Run {
  id: string;
  label: string;
}

const SCROLL_STORAGE_KEY = "results-scroll-position";

export function RunSelectorClient({
  runs,
  currentRunId,
}: {
  runs: Run[];
  currentRunId: string;
}) {
  // Restore scroll position on mount
  useEffect(() => {
    const savedPosition = sessionStorage.getItem(SCROLL_STORAGE_KEY);
    if (savedPosition) {
      const scrollContainer = document.querySelector('[data-scroll-container="results"]');
      if (scrollContainer) {
        scrollContainer.scrollTop = parseInt(savedPosition, 10);
      }
      sessionStorage.removeItem(SCROLL_STORAGE_KEY);
    }
  }, []);

  const handleClick = () => {
    const scrollContainer = document.querySelector('[data-scroll-container="results"]');
    if (scrollContainer) {
      sessionStorage.setItem(SCROLL_STORAGE_KEY, String(scrollContainer.scrollTop));
    }
  };

  if (runs.length <= 1) return null;

  return (
    <div className="fixed bottom-4 right-4 z-50 flex items-center gap-3 bg-white/95 backdrop-blur-sm shadow-lg rounded-lg px-4 py-3 border border-gray-200">
      <label className="text-sm font-medium text-gray-700">Parameter Set:</label>
      <div className="flex gap-2">
        {runs.map(run => (
          <Link
            key={run.id}
            href={`/results/${run.id}`}
            scroll={false}
            onClick={handleClick}
            className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
              run.id === currentRunId
                ? "bg-[#2A623D] text-white"
                : "bg-gray-100 text-gray-700 hover:bg-gray-200"
            }`}
          >
            {run.label}
          </Link>
        ))}
      </div>
    </div>
  );
}
