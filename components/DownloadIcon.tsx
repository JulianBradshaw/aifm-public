'use client';

import { memo } from 'react';

interface DownloadIconProps {
  className?: string;
  size?: number;
}

/**
 * Simple download arrow icon matching the existing icon style
 */
export const DownloadIcon = memo(({ className = '', size = 14 }: DownloadIconProps) => {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={2}
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
    >
      {/* Arrow pointing down */}
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
      <polyline points="7 10 12 15 17 10" />
      <line x1="12" y1="15" x2="12" y2="3" />
    </svg>
  );
});

DownloadIcon.displayName = 'DownloadIcon';
