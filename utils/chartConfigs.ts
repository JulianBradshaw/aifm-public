import { ChartConfig, ChartConfigMap, ChartDataPoint } from '@/types/chartConfig';
import {
  generateHorizonData,
  generateSCHorizonData
} from '@/utils/chartCalculations';
import { calculateLogTicksWithSubdivisions, calculateTimeLogTicks } from '@/utils/chartUtils';
import { formatTo3SigFigs, formatWorkTimeDuration } from '@/utils/formatting';

const tenMillenniaInMinutes = 5.256e9;

const horizonLengthConfig: ChartConfig = {
  title: 'Horizon Length vs Effective Compute',
  dataGenerator: (params, startOOM) => generateHorizonData(params, startOOM),
  charts: (data, params) => {
    const chartData = data as ChartDataPoint[];
    const yValues = chartData.map(d => d.y).filter(v => !isNaN(v));

    // Calculate y domain
    const dataMin = Math.min(...yValues);
    const dataMax = Math.max(...yValues);
    const scHorizonMinutes = Math.pow(10, params.ac_time_horizon_minutes);

    const yDomainMin = Math.max(0.1, Math.min(dataMin, scHorizonMinutes * 0.1));
    const yDomainMax = Math.min(tenMillenniaInMinutes, Math.max(scHorizonMinutes * 10, dataMax)) * 1e1;

    // Calculate y-axis ticks for this domain
    const yTicks = calculateTimeLogTicks([yDomainMin, yDomainMax], 5);

    return [{
      data: chartData,
      width: 280,
      height: 140,
      xLabel: "OOMs",
      xTickFormatter: (value) => `${value.toPrecision(2)}`,
      stroke: "#2A623D",
      yScale: "log" as const,
      yDomain: [yDomainMin, yDomainMax],
      yTicks,
      yTickFormatter: formatWorkTimeDuration,
      showYAxis: true
    }];
  },
  containerClassName: "font-system-mono"
}

// Configuration for all chart types - centralized for reuse
export const CHART_CONFIGS: ChartConfigMap = {
  doubling_time: horizonLengthConfig,
  sc_horizon: {
    title: 'Automation & Research Taste vs Effective Compute',
    dataGenerator: (params, startOOM) => generateSCHorizonData(params, startOOM),
    charts: (generatedData) => {
      const data = generatedData as ReturnType<typeof generateSCHorizonData>;
      const tasteSeries = data.effectiveComputes.map((oom, i) => ({
        x: oom,
        y: data.aiResearchTaste[i] || 0
      }));

      const tasteValues = tasteSeries
        .map(point => point.y)
        .filter(value => typeof value === 'number' && isFinite(value) && value > 0) as number[];

      const tasteDomainMin = tasteValues.length > 0
        ? Math.max(0.01, Math.min(...tasteValues) * 0.9)
        : 0.1;

      const tasteDomainMax = tasteValues.length > 0
        ? Math.max(tasteDomainMin * 1.001, Math.max(...tasteValues) * 1.1)
        : 10;

      let tasteTicks = calculateLogTicksWithSubdivisions([tasteDomainMin, tasteDomainMax], 4);

      if (tasteTicks.length === 0) {
        tasteTicks = [tasteDomainMin, tasteDomainMax];
      }

      const automationSeries = data.effectiveComputes.map((oom, i) => ({
        x: oom,
        y: (data.automationFraction[i] || 0) * 100
      }));

      const automationTicks = [0, 25, 50, 75, 100];

      return [
        {
          data: tasteSeries,
          width: 280,
          height: 100,
          xLabel: "OOMs",
          xTickFormatter: (value) => `${value.toPrecision(2)}`,
          stroke: "#ff7f0e",
          yScale: "log" as const,
          yDomain: [tasteDomainMin, tasteDomainMax],
          yTicks: tasteTicks,
          yTickFormatter: (value) => formatTo3SigFigs(value),
          showYAxis: true
        },
        {
          data: automationSeries,
          width: 280,
          height: 100,
          xLabel: "OOMs",
          xTickFormatter: (value) => `${value.toPrecision(2)}`,
          stroke: "#2ca02c",
          yDomain: [0, 100],
          yTicks: automationTicks,
          yTickFormatter: (value) => `${Math.round(value)}%`,
          showYAxis: true
        }
      ];
    }
  },

  difficulty_growth_rate: horizonLengthConfig,
};
