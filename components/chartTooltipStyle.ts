import type { CSSProperties } from 'react';

export const tooltipBoxStyle: CSSProperties = {
  backgroundColor: 'var(--vivid-background)',
  color: 'var(--vivid-foreground)',
  borderRadius: '6px',
  padding: '6px 8px',
  fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace',
  fontSize: '11px',
  lineHeight: 1.4,
  display: 'flex',
  flexDirection: 'column',
  gap: '2px',
  maxWidth: '300px'
  // whiteSpace: 'wrap',
};

export const tooltipHeaderStyle: CSSProperties = {
  fontWeight: 600,
};

export const tooltipValueStyle: CSSProperties = {
  color: 'var(--accent-color)',
};

