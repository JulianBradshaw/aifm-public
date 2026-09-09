import type { ReactNode } from 'react';
import { clamp } from './chartUtils';

export type XYPoint = { x: number; y: number };
export type PixelPoint = { x: number; y: number };
export type PixelSegment = { a: PixelPoint; b: PixelPoint };
export type TextAnchor = 'start' | 'middle' | 'end';

export interface LabelPosition {
  textX: number;
  textY: number;
  anchor: TextAnchor;
}

export interface LabelBounds {
  x1: number;
  x2: number;
  y1: number;
  y2: number;
}

// Constants for label positioning
const LABEL_HEIGHT = 16;
const DEFAULT_LABEL_WIDTH = 60;
const LABEL_HORIZONTAL_GAP = 25;
const LABEL_VERTICAL_GAP = 16;
const LABEL_PADDING = 4;
const SEGMENTS_TO_CHECK = 20;
const SEGMENT_SAMPLE_STEPS = 6;
const OVERLAP_PADDING = 2;

/**
 * Interpolate y value at a given x position along a series of points
 */
export function interpolateAtX(points: XYPoint[], x: number): number | null {
  if (!points || points.length === 0) return null;
  if (x <= points[0].x) return points[0].y;
  if (x >= points[points.length - 1].x) return points[points.length - 1].y;

  for (let i = 0; i < points.length - 1; i++) {
    const p1 = points[i];
    const p2 = points[i + 1];
    if (x >= p1.x && x <= p2.x && p2.x !== p1.x) {
      const t = (x - p1.x) / (p2.x - p1.x);
      return p1.y + t * (p2.y - p1.y);
    }
  }

  return null;
}

/**
 * Get the visible left and right endpoints of a series within a domain
 */
export function getVisibleEndpoints(
  points: XYPoint[],
  xMin: number,
  xMax: number
): { left: XYPoint | null; right: XYPoint | null } {
  if (!points || points.length === 0) {
    return { left: null, right: null };
  }

  const firstX = points[0].x;
  const lastX = points[points.length - 1].x;

  if (lastX < xMin || firstX > xMax) {
    return { left: null, right: null };
  }

  const leftX = Math.max(xMin, Math.min(xMax, firstX));
  const rightX = Math.min(xMax, Math.max(xMin, lastX));

  const leftY = interpolateAtX(points, leftX);
  const rightY = interpolateAtX(points, rightX);

  return {
    left: leftY == null ? null : { x: leftX, y: leftY },
    right: rightY == null ? null : { x: rightX, y: rightY },
  };
}

/**
 * Estimate the width of a label based on its content
 */
export function estimateLabelWidth(label: ReactNode): number {
  if (typeof label === 'number') {
    return clamp(String(label).length * 6 + 12, 36, 140);
  }
  if (typeof label === 'string') {
    return clamp(label.length * 6 + 14, 36, 160);
  }
  if (Array.isArray(label)) {
    const joined = label
      .map((child) => (typeof child === 'string' || typeof child === 'number' ? String(child) : ''))
      .join('');
    if (joined) {
      return clamp(joined.length * 6 + 14, 36, 160);
    }
  }
  return DEFAULT_LABEL_WIDTH;
}

/**
 * Compute the bounding box for a label at a given position
 */
export function computeLabelBounds(
  textX: number,
  textY: number,
  anchor: TextAnchor,
  width: number,
  height: number
): LabelBounds {
  let x1: number;
  let x2: number;
  if (anchor === 'start') {
    x1 = textX;
    x2 = textX + width;
  } else if (anchor === 'end') {
    x1 = textX - width;
    x2 = textX;
  } else {
    x1 = textX - width / 2;
    x2 = textX + width / 2;
  }

  const y1 = textY - height / 2;
  const y2 = textY + height / 2;
  return { x1, x2, y1, y2 };
}

/**
 * Clamp a label position to stay within the chart bounds
 */
export function clampLabelWithinChart(
  candidate: LabelPosition,
  width: number,
  height: number,
  chartWidth: number,
  chartHeight: number
): LabelPosition {
  const { anchor } = candidate;
  let { textX, textY } = candidate;

  let bounds = computeLabelBounds(textX, textY, anchor, width, height);

  if (bounds.x1 < LABEL_PADDING) {
    const delta = LABEL_PADDING - bounds.x1;
    textX += delta;
    bounds = computeLabelBounds(textX, textY, anchor, width, height);
  } else if (bounds.x2 > chartWidth - LABEL_PADDING) {
    const delta = bounds.x2 - (chartWidth - LABEL_PADDING);
    textX -= delta;
    bounds = computeLabelBounds(textX, textY, anchor, width, height);
  }

  if (bounds.y1 < LABEL_PADDING) {
    const delta = LABEL_PADDING - bounds.y1;
    textY += delta;
    bounds = computeLabelBounds(textX, textY, anchor, width, height);
  } else if (bounds.y2 > chartHeight - LABEL_PADDING) {
    const delta = bounds.y2 - (chartHeight - LABEL_PADDING);
    textY -= delta;
  }

  return { textX, textY, anchor };
}

/**
 * Check if a point is inside a rectangle
 */
function pointInsideRect(x: number, y: number, rect: LabelBounds): boolean {
  return x >= rect.x1 && x <= rect.x2 && y >= rect.y1 && y <= rect.y2;
}

/**
 * Check if a line segment intersects a rectangle
 */
function segmentIntersectsRect(segment: PixelSegment, rect: LabelBounds): boolean {
  if (
    Math.max(segment.a.x, segment.b.x) < rect.x1 - OVERLAP_PADDING ||
    Math.min(segment.a.x, segment.b.x) > rect.x2 + OVERLAP_PADDING ||
    Math.max(segment.a.y, segment.b.y) < rect.y1 - OVERLAP_PADDING ||
    Math.min(segment.a.y, segment.b.y) > rect.y2 + OVERLAP_PADDING
  ) {
    return false;
  }

  for (let i = 0; i <= SEGMENT_SAMPLE_STEPS; i++) {
    const t = i / SEGMENT_SAMPLE_STEPS;
    const sampleX = segment.a.x + (segment.b.x - segment.a.x) * t;
    const sampleY = segment.a.y + (segment.b.y - segment.a.y) * t;
    if (pointInsideRect(sampleX, sampleY, rect)) {
      return true;
    }
  }

  return false;
}

/**
 * Check if a candidate label position overlaps with line segments
 */
export function doesCandidateOverlap(
  candidate: LabelPosition,
  width: number,
  height: number,
  segments: PixelSegment[]
): boolean {
  if (!segments.length) {
    return false;
  }
  const bounds = computeLabelBounds(candidate.textX, candidate.textY, candidate.anchor, width, height);
  const expandedBounds = {
    x1: bounds.x1 - OVERLAP_PADDING,
    x2: bounds.x2 + OVERLAP_PADDING,
    y1: bounds.y1 - OVERLAP_PADDING,
    y2: bounds.y2 + OVERLAP_PADDING,
  };

  return segments.some(segment => segmentIntersectsRect(segment, expandedBounds));
}

/**
 * Generate candidate positions for a label relative to a point
 */
export function createLabelCandidates(
  align: 'left' | 'right',
  baseX: number,
  baseY: number,
  labelWidth: number,
  labelHeight: number,
  chartWidth: number,
  chartHeight: number
): LabelPosition[] {
  const candidates: LabelPosition[] = [];
  
  const add = (anchor: TextAnchor, offsetX: number, offsetY: number) => {
    const textX = baseX + offsetX;
    const textY = clamp(baseY + offsetY, labelHeight / 2 + LABEL_PADDING, chartHeight - labelHeight / 2 - LABEL_PADDING);
    candidates.push(clampLabelWithinChart({ textX, textY, anchor }, labelWidth, labelHeight, chartWidth, chartHeight));
  };

  if (align === 'left') {
    add('middle', 0, -LABEL_VERTICAL_GAP);
    add('middle', 0, LABEL_VERTICAL_GAP);
    add('start', LABEL_HORIZONTAL_GAP, -LABEL_VERTICAL_GAP / 2);
    add('start', LABEL_HORIZONTAL_GAP, LABEL_VERTICAL_GAP / 2);
    add('end', -LABEL_HORIZONTAL_GAP, -LABEL_VERTICAL_GAP / 2);
    add('end', -LABEL_HORIZONTAL_GAP, LABEL_VERTICAL_GAP / 2);
  } else {
    add('start', LABEL_HORIZONTAL_GAP, 0);
    add('start', LABEL_HORIZONTAL_GAP, -LABEL_VERTICAL_GAP);
    add('start', LABEL_HORIZONTAL_GAP, LABEL_VERTICAL_GAP);
    add('end', -LABEL_HORIZONTAL_GAP, -LABEL_VERTICAL_GAP);
    add('end', -LABEL_HORIZONTAL_GAP, LABEL_VERTICAL_GAP);
    add('middle', 0, -LABEL_VERTICAL_GAP);
    add('middle', 0, LABEL_VERTICAL_GAP);
  }

  return candidates;
}

/**
 * Select the best label position that doesn't overlap with line segments
 */
export function selectLabelPosition(
  align: 'left' | 'right',
  baseX: number,
  baseY: number,
  labelWidth: number,
  labelHeight: number,
  chartWidth: number,
  chartHeight: number,
  segmentsToTest: PixelSegment[]
): LabelPosition {
  const candidates = createLabelCandidates(align, baseX, baseY, labelWidth, labelHeight, chartWidth, chartHeight);
  
  for (const candidate of candidates) {
    if (!doesCandidateOverlap(candidate, labelWidth, labelHeight, segmentsToTest)) {
      return candidate;
    }
  }
  
  return candidates[0];
}

/**
 * Convert data points to pixel segments for collision detection
 */
export function pointsToPixelSegments(
  points: XYPoint[],
  xScale: (value: number) => number,
  yScale: (value: number) => number,
  chartHeight: number,
  maxSegments?: number
): PixelSegment[] {
  const pixelPoints: PixelPoint[] = points
    .map(point => ({
      x: xScale(point.x),
      y: clamp(yScale(point.y), -LABEL_HEIGHT, chartHeight + LABEL_HEIGHT),
    }))
    .filter(point => Number.isFinite(point.x) && Number.isFinite(point.y));

  const segments: PixelSegment[] = [];
  for (let i = 0; i < pixelPoints.length - 1; i++) {
    segments.push({ a: pixelPoints[i], b: pixelPoints[i + 1] });
  }

  if (maxSegments !== undefined && segments.length > maxSegments) {
    return segments.slice(0, maxSegments);
  }

  return segments;
}

// Re-export constants for consumers that need them
export { LABEL_HEIGHT, SEGMENTS_TO_CHECK };

