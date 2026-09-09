import React from 'react';
import { tooltipBoxStyle, tooltipHeaderStyle, tooltipValueStyle } from './chartTooltipStyle';
import { formatYearMonth } from '@/utils/formatting';
import type { DataPoint } from './CustomLineChart';
import { SMALL_CHART_EXPLANATIONS } from '@/constants/chartExplanations';
import { MilestoneTooltipSection } from './MilestoneTooltipSection';

interface SmallChartMetricTooltipProps {
  point: DataPoint;
  fieldName: keyof typeof SMALL_CHART_EXPLANATIONS;
  formatter: (value: number) => React.ReactNode;
  requirePositive?: boolean;
}

export function SmallChartMetricTooltip({
  point,
  fieldName,
  formatter,
  requirePositive = false
}: SmallChartMetricTooltipProps) {
  const getPointYear = (p: DataPoint): number => {
    const maybeYear = p['year'];
    return typeof maybeYear === 'number' ? maybeYear : p.x;
  };

  const getNumericField = (p: DataPoint, key: string): number | null => {
    const raw = p[key];
    return typeof raw === 'number' && Number.isFinite(raw) ? raw : null;
  };

  const year = getPointYear(point);
  const value = getNumericField(point, fieldName);
  const explanation = SMALL_CHART_EXPLANATIONS[fieldName];

  return (
    <div style={tooltipBoxStyle}>
      <span style={tooltipHeaderStyle}>{formatYearMonth(year)}</span>
      {value != null && (!requirePositive || value > 0) && (
        <span style={tooltipValueStyle}>{formatter(value)}</span>
      )}
      {explanation && <span style={{ color: 'var(--vivid-foreground)' }}>{explanation}</span>}
      {/* Only present on charts that draw milestone dots (see CustomMetricChart's
          milestoneMarkers prop); renders nothing otherwise. */}
      <MilestoneTooltipSection point={point} />
    </div>
  );
}

