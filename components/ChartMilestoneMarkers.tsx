'use client';

import { clamp } from '@/utils/chartUtils';
import { computeLabelBounds, type LabelBounds, type TextAnchor } from '@/utils/chartLabelUtils';

/** Colour of the milestone dot + label; matches the takeoff chart's series. */
export const MILESTONE_MARKER_COLOR = '#2A623D';

const LABEL_FONT_SIZE = 11;
/** Ascent above / descent below the text baseline at LABEL_FONT_SIZE. */
const LABEL_ASCENT = 9;
const LABEL_DESCENT = 3;
/** Generous per-character width: these labels are short and uppercase. */
const LABEL_CHAR_WIDTH = 7;

/** A milestone dot already resolved to pixel coordinates in the plot area. */
export interface PositionedMilestoneMarker {
  label: string;
  xPos: number;
  yPos: number;
}

interface ChartMilestoneMarkersProps {
  markers: PositionedMilestoneMarker[];
  chartWidth: number;
  chartHeight: number;
  /** Boxes the labels should dodge -- e.g. the series' endpoint value labels. A label
   *  whose default spot collides with one of these flips to the first free
   *  alternative (other side, below) instead of overprinting it. */
  avoidBounds?: LabelBounds[];
}

type LabelOffset = { anchor: TextAnchor; xOffset: number; yOffset: number };

/** Default: above-left of the dot. Used when nothing gets in the way. */
const ABOVE_LEFT: LabelOffset = { anchor: 'end', xOffset: 0, yOffset: -12 };
/** Used for a milestone stacked on top of an earlier one at the same x. */
const LEFT_OF_DOT: LabelOffset = { anchor: 'end', xOffset: -8, yOffset: 0 };

/**
 * Alternatives tried, in order, when the preferred spot collides with something:
 * mirrors of the two defaults -- the other side, then below, then beside the dot.
 */
const FALLBACK_OFFSETS: LabelOffset[] = [
  { anchor: 'start', xOffset: 4, yOffset: -12 },   // above-right
  { anchor: 'end', xOffset: 0, yOffset: 18 },      // below-left
  { anchor: 'start', xOffset: 4, yOffset: 18 },    // below-right
  { anchor: 'start', xOffset: 8, yOffset: 0 },     // right of the dot
  LEFT_OF_DOT,
];

/** Slack in the overlap test: the estimated boxes are approximate, and labels that
 *  merely touch still read as collided. */
const COLLISION_MARGIN = 2;

function boundsOverlap(a: LabelBounds, b: LabelBounds): boolean {
  return a.x1 - COLLISION_MARGIN < b.x2
    && a.x2 + COLLISION_MARGIN > b.x1
    && a.y1 - COLLISION_MARGIN < b.y2
    && a.y2 + COLLISION_MARGIN > b.y1;
}

/**
 * The milestone dots-with-labels overlay shared by the takeoff-period chart and the
 * ECI chart: a small filled circle sitting on the series with the milestone's short
 * label above-left of it. Markers that land on (nearly) the same x are stacked by
 * moving the later ones to the dot's left; labels near an edge are flipped inwards so
 * they stay inside the plot area; and a label that would land on top of an
 * `avoidBounds` box (or on an already-placed milestone label) flips to the first free
 * alternative position.
 *
 * Callers own the data->pixel mapping (each chart reads its own series), so this
 * component is purely presentational.
 */
export function ChartMilestoneMarkers({
  markers,
  chartWidth,
  chartHeight,
  avoidBounds,
}: ChartMilestoneMarkersProps) {
  const positionedMarkers = markers
    .map((marker, index) => ({ ...marker, index }))
    .filter(m => Number.isFinite(m.xPos) && Number.isFinite(m.yPos));

  // Detect overlapping milestones (same or very close x-position)
  const OVERLAP_THRESHOLD = 5; // pixels
  const horizontalPadding = 12;

  // Resolve one offset choice into a concrete, edge-clamped text placement.
  const place = (marker: { xPos: number; yPos: number }, offset: LabelOffset, labelWidth: number) => {
    const clampedY = clamp(marker.yPos, 0, chartHeight);

    const nearLeftEdge = marker.xPos <= horizontalPadding;
    const nearRightEdge = marker.xPos >= chartWidth - horizontalPadding;

    let anchor = offset.anchor;
    let textX = marker.xPos + offset.xOffset;
    const textY = clampedY + offset.yOffset;

    // Adjust for edge cases
    if (nearLeftEdge) {
      anchor = 'start';
      textX = Math.max(marker.xPos + horizontalPadding * 0.5, horizontalPadding);
    } else if (nearRightEdge) {
      anchor = 'end';
      textX = Math.min(marker.xPos - horizontalPadding * 0.5, chartWidth - horizontalPadding);
    } else {
      // Clamp to keep label within chart bounds
      textX = clamp(textX, horizontalPadding, chartWidth - horizontalPadding);
    }

    const clampedTextY = clamp(textY, 12, chartHeight - 4);

    // The text is drawn on its baseline, so its box sits mostly ABOVE clampedTextY;
    // computeLabelBounds centres on the y it is given, hence the half-shift.
    const bounds = computeLabelBounds(
      textX,
      clampedTextY - (LABEL_ASCENT - LABEL_DESCENT) / 2,
      anchor,
      labelWidth,
      LABEL_ASCENT + LABEL_DESCENT,
    );

    return { anchor, textX, textY: clampedTextY, clampedY, bounds };
  };

  // Labels already committed this pass; later markers dodge them the same way they
  // dodge the caller's avoidBounds.
  const placedBounds: LabelBounds[] = [...(avoidBounds ?? [])];

  const placements = positionedMarkers.map((marker, i) => {
    // Check if this milestone overlaps with any previous ones
    let overlapCount = 0;
    for (let j = 0; j < i; j++) {
      const other = positionedMarkers[j];
      if (Math.abs(marker.xPos - other.xPos) < OVERLAP_THRESHOLD) {
        overlapCount++;
      }
    }

    // Position overlapping milestones:
    // - First milestone: above-left (default position)
    // - Second milestone: left of the dot (same vertical level)
    const preferred = overlapCount === 0 ? ABOVE_LEFT : LEFT_OF_DOT;
    const labelWidth = marker.label.length * LABEL_CHAR_WIDTH;

    let chosen = place(marker, preferred, labelWidth);
    if (placedBounds.some(b => boundsOverlap(chosen.bounds, b))) {
      for (const fallback of FALLBACK_OFFSETS) {
        const candidate = place(marker, fallback, labelWidth);
        // A fallback that runs off the plot is worse than the collision it fixes;
        // only the preferred spot is allowed to sit against a clamped edge.
        const escapesPlot = candidate.bounds.x1 < 0 || candidate.bounds.x2 > chartWidth;
        if (!escapesPlot && !placedBounds.some(b => boundsOverlap(candidate.bounds, b))) {
          chosen = candidate;
          break;
        }
      }
    }

    placedBounds.push(chosen.bounds);
    return { marker, ...chosen };
  });

  return (
    <g>
      {placements.map(({ marker, anchor, textX, textY, clampedY }) => (
        <g key={`milestone-${marker.index}`}>
          <circle
            cx={marker.xPos}
            cy={clampedY}
            r={4}
            fill={MILESTONE_MARKER_COLOR}
            stroke="white"
            strokeWidth={1.5}
          />
          <text
            x={textX}
            y={textY}
            textAnchor={anchor}
            fontSize={LABEL_FONT_SIZE}
            fill={MILESTONE_MARKER_COLOR}
            fontWeight="600"
          >
            {marker.label}
          </text>
        </g>
      ))}
    </g>
  );
}
