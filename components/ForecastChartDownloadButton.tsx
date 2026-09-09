'use client';

import { memo, useState, useRef, useCallback } from 'react';
import { DownloadIcon } from './DownloadIcon';
import { exportForecastChartPng } from '@/utils/forecastChartExport';

interface ForecastChartDownloadButtonProps {
  /** Live plot SVG to capture. */
  chartSvgRef: React.RefObject<SVGSVGElement | null>;
  /** Live HTML legend container, redrawn into the PNG. */
  legendRef: React.RefObject<HTMLDivElement | null>;
  /** Headline shown at the top of the exported image. */
  title: string;
  /** Second line: which forecast, and which view is active. */
  subtitle?: string;
  /** Resolved lazily so the filename carries the current chart state. */
  getFilename: () => string;
  className?: string;
}

/**
 * Small download affordance sitting next to a forecast chart's title.
 * Exports the chart exactly as currently configured (simulation, PDF/CDF,
 * visible milestones) as a self-contained, presentation-grade PNG.
 */
export const ForecastChartDownloadButton = memo(({
  chartSvgRef,
  legendRef,
  title,
  subtitle,
  getFilename,
  className = '',
}: ForecastChartDownloadButtonProps) => {
  const [isExporting, setIsExporting] = useState(false);
  const [failed, setFailed] = useState(false);
  // A ref, not the state flag: both events of a double-click can arrive before
  // React re-renders, so a state-based guard would let the second one through.
  const inFlightRef = useRef(false);

  const handleDownload = useCallback(async () => {
    // Chrome grants roughly one download per user gesture. A double-click puts
    // both gestures ahead of both downloads, so the browser drops the extra —
    // and sometimes both, leaving the button looking dead. Collapsing a rapid
    // repeat into a single export is what makes the download land reliably.
    if (inFlightRef.current) return;

    const chartSvg = chartSvgRef.current;
    if (!chartSvg) {
      // Never fail silently — an inert button is indistinguishable from a bug.
      console.error('Failed to export chart as PNG: the chart is not mounted yet');
      setFailed(true);
      return;
    }

    inFlightRef.current = true;
    setIsExporting(true);
    setFailed(false);
    try {
      await exportForecastChartPng({
        chartSvg,
        legendEl: legendRef.current,
        title,
        subtitle,
        filename: getFilename(),
      });
    } catch (error) {
      console.error('Failed to export chart as PNG:', error);
      setFailed(true);
    } finally {
      inFlightRef.current = false;
      setIsExporting(false);
    }
  }, [chartSvgRef, legendRef, title, subtitle, getFilename]);

  const buttonClasses = [
    'relative inline-flex h-4 w-4 min-w-4 shrink-0 items-center justify-center',
    'rounded-full border text-gray-600 border-gray-400',
    'cursor-pointer transition-colors',
    'hover:border-gray-600 hover:text-gray-800',
    'focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-gray-500',
    isExporting ? 'cursor-wait' : '',
    failed ? 'border-red-500 text-red-600' : '',
    className,
  ].filter(Boolean).join(' ');

  return (
    <button
      type="button"
      className={buttonClasses}
      onClick={handleDownload}
      disabled={isExporting}
      aria-busy={isExporting}
      aria-label={failed ? 'Chart PNG export failed, click to retry' : 'Download this chart as a PNG'}
      title={failed ? 'Export failed — click to retry' : 'Download this chart as a PNG'}
    >
      <span
        className={`absolute inset-0 flex items-center justify-center transition-opacity duration-200 ${isExporting ? 'opacity-0' : 'opacity-100'}`}
      >
        <DownloadIcon size={10} />
      </span>
      <span
        className={`absolute inset-0 flex items-center justify-center transition-opacity duration-200 ${isExporting ? 'opacity-100' : 'opacity-0'}`}
        aria-hidden={!isExporting}
      >
        <span className="h-2.5 w-2.5 animate-spin rounded-full border-[1.5px] border-gray-400 border-t-transparent" />
      </span>
    </button>
  );
});

ForecastChartDownloadButton.displayName = 'ForecastChartDownloadButton';
