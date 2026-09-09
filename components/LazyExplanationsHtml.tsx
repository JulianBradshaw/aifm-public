'use client';

import { useEffect, useState, forwardRef } from 'react';

interface LazyExplanationsHtmlProps {
  className?: string;
}

// Lazy-load the 250KB explanations HTML to avoid blocking initial render
const LazyExplanationsHtml = forwardRef<HTMLDivElement, LazyExplanationsHtmlProps>(
  function LazyExplanationsHtml({ className }, ref) {
    const [html, setHtml] = useState<string | null>(null);

    useEffect(() => {
      // Dynamic import - this creates a separate chunk that loads on demand
      import('@/public/explanations.html').then((module) => {
        setHtml(module.default);
      });
    }, []);

    if (!html) {
      // Loading skeleton - must have ref so MutationObserver can detect content changes
      return (
        <div ref={ref} className={`${className} animate-pulse space-y-4`}>
          <div className="h-6 bg-gray-200 rounded w-3/4" />
          <div className="space-y-2">
            <div className="h-4 bg-gray-200 rounded w-full" />
            <div className="h-4 bg-gray-200 rounded w-5/6" />
            <div className="h-4 bg-gray-200 rounded w-4/6" />
          </div>
          <div className="h-6 bg-gray-200 rounded w-2/3 mt-6" />
          <div className="space-y-2">
            <div className="h-4 bg-gray-200 rounded w-full" />
            <div className="h-4 bg-gray-200 rounded w-full" />
            <div className="h-4 bg-gray-200 rounded w-3/4" />
          </div>
        </div>
      );
    }

    return (
      <div
        ref={ref}
        className={className}
        dangerouslySetInnerHTML={{ __html: html }}
      />
    );
  }
);

export default LazyExplanationsHtml;

