import { useState, useRef, useEffect, useCallback, type ReactNode } from 'react';
import { createPortal } from 'react-dom';
import { buildDocLink } from '@/constants/keyParameterConfigs';

interface WithChartTooltipProps {
  explanation: string;
  children: ReactNode;
  className?: string;
  fullWidth?: boolean;
  tooltipPlacement?: 'left' | 'right';
  isLoading?: boolean;
  docAnchor?: string;
}

export const WithChartTooltip = ({
  explanation,
  children,
  className,
  fullWidth = false,
  tooltipPlacement = 'right',
  isLoading = false,
  docAnchor,
}: WithChartTooltipProps) => {
  const [showTooltip, setShowTooltip] = useState(false);
  const [tooltipPosition, setTooltipPosition] = useState({ top: 0, left: 0 });
  const triggerRef = useRef<HTMLSpanElement>(null);
  const tooltipRef = useRef<HTMLDivElement>(null);
  const hideTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const docLink = docAnchor ? buildDocLink(docAnchor) : null;
  const isInteractive = !!docLink;

  const clearHideTimeout = useCallback(() => {
    if (hideTimeoutRef.current) {
      clearTimeout(hideTimeoutRef.current);
      hideTimeoutRef.current = null;
    }
  }, []);

  const scheduleHide = useCallback(() => {
    clearHideTimeout();
    if (isInteractive) {
      hideTimeoutRef.current = setTimeout(() => {
        setShowTooltip(false);
      }, 150);
    } else {
      setShowTooltip(false);
    }
  }, [isInteractive, clearHideTimeout]);

  const updateTooltipPosition = useCallback(() => {
    if (triggerRef.current) {
      const rect = triggerRef.current.getBoundingClientRect();
      const tooltipWidth = 256; // w-64 = 16rem = 256px
      const padding = 8;

      let left: number;
      if (tooltipPlacement === 'left') {
        left = rect.left - tooltipWidth - padding;
      } else {
        left = rect.right + padding;
      }

      // Prevent tooltip from going off-screen horizontally
      if (left + tooltipWidth > window.innerWidth - padding) {
        left = rect.left - tooltipWidth - padding;
      }
      if (left < padding) {
        left = padding;
      }

      // Position below the trigger, with some offset
      let top = rect.bottom + 4;

      // Prevent tooltip from going off-screen vertically
      if (top + 100 > window.innerHeight) {
        top = rect.top - 100 - 4;
      }

      setTooltipPosition({ top, left });
    }
  }, [tooltipPlacement]);

  const handleShow = useCallback(() => {
    if (!isLoading) {
      clearHideTimeout();
      updateTooltipPosition();
      setShowTooltip(true);
    }
  }, [isLoading, clearHideTimeout, updateTooltipPosition]);

  useEffect(() => {
    return () => clearHideTimeout();
  }, [clearHideTimeout]);

  const containerClasses = [
    'relative items-start gap-2 align-middle',
    fullWidth ? 'flex w-full justify-between' : 'inline-flex',
    className ?? '',
  ]
    .filter(Boolean)
    .join(' ');

  const labelClasses = fullWidth ? 'flex-1 leading-tight' : 'inline-flex leading-tight';

  return (
    <div className={containerClasses}>
      <span className={labelClasses}>{children}</span>
      <span
        ref={triggerRef}
        className="mt-[0.5px] relative inline-flex h-4 w-4 min-w-4 items-center justify-center rounded-full border border-gray-400 text-[10px] font-semibold leading-none text-gray-600 cursor-help transition-colors hover:border-gray-600 hover:text-gray-800 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-gray-500"
        role="button"
        tabIndex={0}
        aria-label="Show explanation"
        aria-expanded={showTooltip}
        onMouseEnter={handleShow}
        onMouseLeave={scheduleHide}
        onFocus={handleShow}
        onBlur={scheduleHide}
      >
        {/* Info icon - fades out when loading */}
        <span 
          className={`absolute inset-0 flex items-center justify-center transition-opacity duration-200 ${isLoading ? 'opacity-0' : 'opacity-100'}`}
        >
          i
        </span>
        {/* Loading spinner - fades in when loading */}
        <span 
          className={`absolute inset-0 flex items-center justify-center transition-opacity duration-200 ${isLoading ? 'opacity-100' : 'opacity-0'}`}
          aria-hidden={!isLoading}
        >
          <span className="h-2.5 w-2.5 animate-spin rounded-full border-[1.5px] border-gray-400 border-t-transparent" />
        </span>
      </span>
      {showTooltip && !isLoading && typeof document !== 'undefined' && createPortal(
        <div
          ref={tooltipRef}
          className={`fixed z-[9999] ${isInteractive ? 'pointer-events-auto' : 'pointer-events-none'}`}
          style={{ top: tooltipPosition.top, left: tooltipPosition.left }}
          onMouseEnter={isInteractive ? handleShow : undefined}
          onMouseLeave={isInteractive ? scheduleHide : undefined}
        >
          <div className="w-64 rounded-md bg-gray-900 px-3 py-2 text-xs text-white shadow-lg font-system-mono font-normal">
            {explanation}
            {docLink && (
              <a
                href={docLink}
                target="_blank"
                rel="noopener noreferrer"
                className="mt-2 flex items-center gap-1.5 text-[11px] text-gray-400 hover:text-white transition-colors border-t border-gray-700 pt-2"
              >
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6" />
                  <polyline points="15 3 21 3 21 9" />
                  <line x1="10" y1="14" x2="21" y2="3" />
                </svg>
                View detailed rationale
              </a>
            )}
          </div>
        </div>,
        document.body
      )}
    </div>
  );
};
