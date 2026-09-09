'use client';

import React, { useMemo } from 'react';
import { CustomSliderChart } from './CustomSliderChart';
import {
  SliderPopoverProps,
  ChartDefinition,
  YDomainFunction
} from '@/types/chartConfig';
import { TRAINING_COMPUTE_REFERENCE_OOMS, ParametersType } from '@/constants/parameters';
import {
  generateHorizonData,
  generateSCHorizonData,
  createDefaultSCHorizonConfig
} from '@/utils/chartCalculations';

// Helper function to resolve y-domain values
function resolveYDomain(
  yDomain: ChartDefinition['yDomain'],
  data: Array<{ x: number; y: number }>,
  params: ParametersType
): [number, number] | undefined {
  if (!yDomain) return undefined;

  if (Array.isArray(yDomain)) {
    const minY = Math.min(...data.map(d => d.y));
    const maxY = Math.max(...data.map(d => d.y));

    const resolvedMin = typeof yDomain[0] === 'function'
      ? (yDomain[0] as YDomainFunction)(minY, params)
      : yDomain[0];

    const resolvedMax = typeof yDomain[1] === 'function'
      ? (yDomain[1] as YDomainFunction)(maxY, params)
      : yDomain[1];

    return [resolvedMin, resolvedMax];
  }

  return undefined;
}

function SliderPopoverComponent({ config, visible, onClose, uiParameters, horizonParams, anchorName }: SliderPopoverProps) {

  // Memoize the data generation
  // For horizon charts, we need to pass the anchor_progress if available
  const generatedData = useMemo(
    () => {
      if (!config) return [];

      const startOOM = TRAINING_COMPUTE_REFERENCE_OOMS - 3;
      const anchorProgress =
        horizonParams?.uses_shifted_form && typeof horizonParams.anchor_progress === 'number'
          ? horizonParams.anchor_progress
          : undefined;

      if (config.title === 'Automation & Research Taste vs Effective Compute') {
        const schConfig = createDefaultSCHorizonConfig(uiParameters);

        if (anchorProgress !== undefined) {
          schConfig.anchorProgress = anchorProgress;
        }

        return generateSCHorizonData(uiParameters, schConfig, startOOM);
      }

      // Check if this is a horizon-related chart that needs anchor_progress
      const isHorizonChart = config.title && config.title.includes('Horizon');

      if (isHorizonChart && anchorProgress !== undefined) {
        return generateHorizonData(uiParameters, startOOM, 20, anchorProgress);
      }

      // For other charts, use the default dataGenerator
      return config.dataGenerator(uiParameters, startOOM);
    },
    [
      config,
      horizonParams,
      uiParameters
    ]
  );

  // Get chart definitions - either static or generated from data
  const chartDefinitions = useMemo(() => {
    if (!config) return [];
    if (typeof config.charts === 'function') {
      return config.charts(generatedData, uiParameters);
    }
    return config.charts;
  }, [config, generatedData, uiParameters]);

  if (!visible) return null;
  if (!config) return null;

  return (
    <div
      popover="manual"
      className="bg-vivid-background border border-gray-300 rounded-lg shadow-lg p-3 min-w-[320px] max-w-[400px] anchor-positioned-tooltip"
      style={{
        positionAnchor: anchorName,
        insetArea: 'bottom span-right',
        margin: '4px',
        positionTryFallbacks: '--tooltip-top, --tooltip-left, --tooltip-right',
      }}
    >
      <div className="flex justify-between items-center mb-2">
        <h4 className="text-xs font-medium font-system-mono">{config.title}</h4>
        <button
          onClick={onClose}
          className="text-gray-500 hover:text-gray-700 text-xs"
        >
          ×
        </button>
      </div>
      <div className={config.containerClassName}>
        {chartDefinitions.map((chart, index) => {
          const chartData = typeof chart.data === 'function'
            ? chart.data(generatedData)
            : chart.data;

          return (
            <CustomSliderChart
              key={index}
              data={chartData}
              width={chart.width}
              height={chart.height}
              xLabel={chart.xLabel}
              xTickFormatter={chart.xTickFormatter}
              stroke={chart.stroke}
              yScale={chart.yScale}
              yDomain={resolveYDomain(chart.yDomain, chartData, uiParameters)}
              yTicks={chart.yTicks}
              yTickFormatter={chart.yTickFormatter}
              showYAxis={chart.showYAxis}
            />
          );
        })}
      </div>
    </div>
  );
}

// Export memoized version to prevent unnecessary re-renders
export const SliderPopover = React.memo(SliderPopoverComponent);