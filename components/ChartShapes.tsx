import { ShapeProps } from '@/app/types';

/**
 * The five marks below, but as bare geometry so a caller can size them.
 *
 * The components in this file are fixed at the METR horizon chart's radius. Observed
 * datapoints on the metric charts need the same vocabulary at a smaller size (the
 * small-chart grid is a fraction of the primary chart's height), so the point strings
 * are factored out here rather than re-derived at each call site.
 */
export type ObservedMarkShape = 'diamond' | 'square' | 'triangle' | 'star' | 'circle';

/** SVG polygon `points` for a mark, or null for 'circle' (draw a <circle> instead). */
export function observedMarkPoints(
  shape: ObservedMarkShape,
  cx: number,
  cy: number,
  r: number
): string | null {
  switch (shape) {
    case 'diamond':
      return `${cx},${cy - r} ${cx + r},${cy} ${cx},${cy + r} ${cx - r},${cy}`;
    case 'square': {
      const half = r * (5 / 6);
      return `${cx - half},${cy - half} ${cx + half},${cy - half} ${cx + half},${cy + half} ${cx - half},${cy + half}`;
    }
    case 'triangle':
      return `${cx},${cy - r} ${cx + r},${cy + r} ${cx - r},${cy + r}`;
    case 'star': {
      const inner = r / 2;
      const points: string[] = [];
      for (let i = 0; i < 10; i += 1) {
        const radius = i % 2 === 0 ? r : inner;
        const angle = (i * Math.PI) / 5 - Math.PI / 2;
        points.push(`${cx + radius * Math.cos(angle)},${cy + radius * Math.sin(angle)}`);
      }
      return points.join(' ');
    }
    case 'circle':
      return null;
  }
}

/**
 * Colour + shape per lab, mirroring the mapping CustomHorizonChart applies to METR
 * points. Keyed by the organization strings Epoch publishes, plus the short keys the
 * horizon chart uses, so an observed datapoint for a given lab reads identically
 * wherever it appears. Anything unlisted falls back to OBSERVED_MARK_FALLBACK.
 */
export const OBSERVED_MARK_BY_ORGANIZATION: Record<string, { fill: string; shape: ObservedMarkShape }> = {
  anthropic: { fill: '#dc2626', shape: 'diamond' },
  claude: { fill: '#dc2626', shape: 'diamond' },
  openai: { fill: '#059669', shape: 'square' },
  google: { fill: '#2563eb', shape: 'triangle' },
  'google deepmind': { fill: '#2563eb', shape: 'triangle' },
  'google deepmind,google': { fill: '#2563eb', shape: 'triangle' },
  deepseek: { fill: '#7c3aed', shape: 'star' },
};

export const OBSERVED_MARK_FALLBACK: { fill: string; shape: ObservedMarkShape } = {
  fill: '#6b7280',
  shape: 'circle',
};

export function observedMarkFor(organization: string): { fill: string; shape: ObservedMarkShape } {
  return OBSERVED_MARK_BY_ORGANIZATION[organization.trim().toLowerCase()] ?? OBSERVED_MARK_FALLBACK;
}

export const createDiamondShape = (fill: string, stroke: string = '#ffffff') => {
  const DiamondShape = (props: ShapeProps) => {
    const { cx, cy, onMouseEnter, onMouseLeave, style } = props;
    return (
      <polygon
        points={`${cx},${cy - 6} ${cx + 6},${cy} ${cx},${cy + 6} ${cx - 6},${cy}`}
        fill={fill}
        stroke={stroke}
        strokeWidth={2}
        onMouseEnter={onMouseEnter}
        onMouseLeave={onMouseLeave}
        style={onMouseEnter ? { cursor: 'pointer', ...style } : style}
      />
    );
  };
  DiamondShape.displayName = 'DiamondShape';
  return DiamondShape;
};

export const createSquareShape = (fill: string, stroke: string = '#ffffff') => {
  const SquareShape = (props: ShapeProps) => {
    const { cx, cy, onMouseEnter, onMouseLeave, style } = props;
    return (
      <rect
        x={cx - 5}
        y={cy - 5}
        width={10}
        height={10}
        fill={fill}
        stroke={stroke}
        strokeWidth={2}
        onMouseEnter={onMouseEnter}
        onMouseLeave={onMouseLeave}
        style={onMouseEnter ? { cursor: 'pointer', ...style } : style}
      />
    );
  };
  SquareShape.displayName = 'SquareShape';
  return SquareShape;
};

export const createTriangleShape = (fill: string, stroke: string = '#ffffff') => {
  const TriangleShape = (props: ShapeProps) => {
    const { cx, cy, onMouseEnter, onMouseLeave, style } = props;
    return (
      <polygon
        points={`${cx},${cy - 6} ${cx + 6},${cy + 6} ${cx - 6},${cy + 6}`}
        fill={fill}
        stroke={stroke}
        strokeWidth={2}
        onMouseEnter={onMouseEnter}
        onMouseLeave={onMouseLeave}
        style={onMouseEnter ? { cursor: 'pointer', ...style } : style}
      />
    );
  };
  TriangleShape.displayName = 'TriangleShape';
  return TriangleShape;
};

export const createStarShape = (fill: string, stroke: string = '#ffffff') => {
  const StarShape = (props: ShapeProps) => {
    const { cx, cy, onMouseEnter, onMouseLeave, style } = props;
    const r1 = 6, r2 = 3;
    const points = [];
    for (let i = 0; i < 10; i++) {
      const r = i % 2 === 0 ? r1 : r2;
      const angle = (i * Math.PI) / 5 - Math.PI / 2;
      points.push(`${cx + r * Math.cos(angle)},${cy + r * Math.sin(angle)}`);
    }
    return (
      <polygon
        points={points.join(' ')}
        fill={fill}
        stroke={stroke}
        strokeWidth={2}
        onMouseEnter={onMouseEnter}
        onMouseLeave={onMouseLeave}
        style={onMouseEnter ? { cursor: 'pointer', ...style } : style}
      />
    );
  };
  StarShape.displayName = 'StarShape';
  return StarShape;
};

export const createCircleShape = (fill: string, stroke: string = '#ffffff') => {
  const CircleShape = (props: ShapeProps) => {
    const { cx, cy, onMouseEnter, onMouseLeave, style } = props;
    return (
      <circle
        cx={cx}
        cy={cy}
        r={5}
        fill={fill}
        stroke={stroke}
        strokeWidth={2}
        onMouseEnter={onMouseEnter}
        onMouseLeave={onMouseLeave}
        style={onMouseEnter ? { cursor: 'pointer', ...style } : style}
      />
    );
  };
  CircleShape.displayName = 'CircleShape';
  return CircleShape;
};