'use client';

import { memo, useState, useCallback } from 'react';
import { DownloadIcon } from './DownloadIcon';
import { exportChartAsPng, EXPORT_PRESETS } from '@/utils/chartExport';

interface ChartDownloadButtonProps {
  targetRef: React.RefObject<SVGSVGElement | null>;
  filename: string;
  className?: string;
  disabled?: boolean;
}

/**
 * Download button for individual chart export
 * Styled to match the existing info icon button
 */
export const ChartDownloadButton = memo(({
  targetRef,
  filename,
  className = '',
  disabled = false,
}: ChartDownloadButtonProps) => {
  const [isExporting, setIsExporting] = useState(false);

  const handleDownload = useCallback(async () => {
    if (!targetRef.current || isExporting || disabled) return;

    setIsExporting(true);
    try {
      await exportChartAsPng(
        targetRef.current,
        filename,
        EXPORT_PRESETS.individual
      );
    } catch (error) {
      console.error('Failed to export chart:', error);
    } finally {
      setIsExporting(false);
    }
  }, [targetRef, filename, isExporting, disabled]);

  const buttonClasses = [
    'mt-[0.5px] relative inline-flex h-4 w-4 min-w-4 items-center justify-center',
    'rounded-full border border-gray-400 text-gray-600',
    'cursor-pointer transition-colors',
    'hover:border-gray-600 hover:text-gray-800',
    'focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-gray-500',
    disabled || isExporting ? 'opacity-50 cursor-not-allowed' : '',
    className,
  ].filter(Boolean).join(' ');

  return (
    <button
      type="button"
      className={buttonClasses}
      onClick={handleDownload}
      disabled={disabled || isExporting}
      aria-label="Download chart as PNG"
      title="Download chart as PNG"
    >
      {/* Download icon - fades out when exporting */}
      <span
        className={`absolute inset-0 flex items-center justify-center transition-opacity duration-200 ${isExporting ? 'opacity-0' : 'opacity-100'}`}
      >
        <DownloadIcon size={10} />
      </span>
      {/* Loading spinner - fades in when exporting */}
      <span
        className={`absolute inset-0 flex items-center justify-center transition-opacity duration-200 ${isExporting ? 'opacity-100' : 'opacity-0'}`}
        aria-hidden={!isExporting}
      >
        <span className="h-2.5 w-2.5 animate-spin rounded-full border-[1.5px] border-gray-400 border-t-transparent" />
      </span>
    </button>
  );
});

ChartDownloadButton.displayName = 'ChartDownloadButton';
