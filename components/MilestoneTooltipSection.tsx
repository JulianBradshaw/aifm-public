import React from 'react';
import { MILESTONE_EXPLANATIONS, MILESTONE_FULL_NAMES } from '@/constants/chartExplanations';
import type { DataPoint } from './CustomLineChart';

/**
 * The milestone block a chart tooltip shows when the hovered point is the one a
 * milestone dot sits on: the milestone's full name plus its one-line explanation.
 *
 * Charts flag those points by writing `milestoneLabels` (an array of short labels
 * such as 'AC') onto the data point; `milestoneLabel` is the older single-string
 * form. Renders nothing when the hovered point carries neither.
 */
export function MilestoneTooltipSection({ point }: { point: DataPoint | Record<string, unknown> }) {
  const record = point as Record<string, unknown>;

  const labels: string[] = Array.isArray(record['milestoneLabels'])
    ? (record['milestoneLabels'] as string[])
    : typeof record['milestoneLabel'] === 'string'
      ? [record['milestoneLabel'] as string]
      : [];

  const withExplanations = labels.filter(label => MILESTONE_EXPLANATIONS[label]);
  if (withExplanations.length === 0) {
    return null;
  }

  return (
    <>
      {withExplanations.map((label, index) => (
        <div key={label} className={index > 0 ? 'mt-2' : ''}>
          <div className="font-semibold">{MILESTONE_FULL_NAMES[label] || label}</div>
          <span style={{ color: 'var(--vivid-foreground)' }}>{MILESTONE_EXPLANATIONS[label]}</span>
        </div>
      ))}
    </>
  );
}
