'use client';

import { CustomLineChart } from './CustomLineChart';

interface SliderChartDataPoint {
  x: number;
  y: number;
  [key: string]: number | null | undefined | string;
}

interface CustomSliderChartProps {
  data: SliderChartDataPoint[];
  width?: number;
  height?: number;
  xLabel?: string;
  xTickFormatter?: (value: number) => string;
  stroke?: string;
  yScale?: 'linear' | 'log';
  yDomain?: [number, number] | [(dataMin: number) => number, (dataMax: number) => number];
  yTicks?: number[];
  yTickFormatter?: (value: number) => string;
  showYAxis?: boolean;
}

export function CustomSliderChart({
  data,
  width = 280,
  height = 140,
  xLabel,
  xTickFormatter = (v) => v.toString(),
  stroke = '#2A623D',
  yScale = 'linear',
  yDomain,
  yTicks,
  yTickFormatter,
  showYAxis = false,
}: CustomSliderChartProps) {
  // Calculate domains
  const xValues = data.map(d => d.x);
  const yValues = data.map(d => d.y);

  const xDomainCalc: [number, number] = [
    Math.min(...xValues),
    Math.max(...xValues)
  ];

  let yDomainCalc: [number, number];
  if (yDomain) {
    if (typeof yDomain[0] === 'function' || typeof yDomain[1] === 'function') {
      const dataMin = Math.min(...yValues);
      const dataMax = Math.max(...yValues);
      const minVal = typeof yDomain[0] === 'function' ? yDomain[0](dataMin) : yDomain[0];
      const maxVal = typeof yDomain[1] === 'function' ? yDomain[1](dataMax) : yDomain[1];
      yDomainCalc = [minVal as number, maxVal as number];
    } else {
      yDomainCalc = yDomain as [number, number];
    }
  } else {
    yDomainCalc = [Math.min(...yValues), Math.max(...yValues)];
  }

  return (
    <CustomLineChart
      data={data}
      width={width}
      height={height}
      margin={{ top: 10, right: 16, left: 0, bottom: 30 }}
      xPadding={{ start: 0 }}
      xDomain={xDomainCalc}
      xTickCount={3}
      xTickFormatter={xTickFormatter}
      xLabel={xLabel}
      yDomain={yDomainCalc}
      yScale={yScale}
      yTicks={yTicks}
      yTickFormatter={yTickFormatter}
      showYAxis={showYAxis}
      lines={[
        {
          dataKey: 'y',
          stroke,
          strokeWidth: 2,
        }
      ]}
    />
  );
}
