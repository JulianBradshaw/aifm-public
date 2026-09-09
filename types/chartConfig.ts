import { ParametersType } from '@/constants/parameters';

export type ChartType = 'doubling_time' | 'sc_horizon' | 'difficulty_growth_rate';

export interface ChartDataPoint {
  x: number;
  y: number;
  [key: string]: number | null | undefined | string;
}

export interface YDomainFunction {
  (value: number, params: ParametersType): number;
}

// Allow for flexible generated data type
export type GeneratedDataType = ChartDataPoint[] | ReturnType<typeof import('../utils/chartCalculations').generateSCHorizonData>;

export interface ChartDefinition {
  data: ChartDataPoint[] | ((generatedData: GeneratedDataType) => ChartDataPoint[]);
  width: number;
  height: number;
  xLabel: string;
  xTickFormatter: (value: number) => string;
  stroke: string;
  yScale?: 'linear' | 'log';
  yDomain?: [number | YDomainFunction, number | YDomainFunction] | [number, number];
  yTicks?: number[];
  yTickFormatter?: (value: number) => string;
  showYAxis?: boolean;
}

export interface ChartConfig {
  title: string;
  dataGenerator: (params: ParametersType, startOOM: number) => GeneratedDataType;
  charts: ChartDefinition[] | ((generatedData: GeneratedDataType, params: ParametersType) => ChartDefinition[]);
  containerClassName?: string;
}

export type ChartConfigMap = Record<ChartType, ChartConfig>;

export interface SliderPopoverProps {
  config: ChartConfig;
  visible: boolean;
  onClose: () => void;
  uiParameters: ParametersType;
  horizonParams?: {
    uses_shifted_form: boolean;
    anchor_progress: number | null;
  } | null;
  anchorName?: string;
}