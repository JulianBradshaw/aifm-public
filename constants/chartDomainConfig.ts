export const CHART_DOMAIN_CONFIG = {
  timeline: {
    absoluteMin: 2020,
    absoluteMax: 2200,
    defaultMin: 2025,
    defaultMax: 2050,
    padding: 0.02,
  },
  takeoff: {
    absoluteMin: 0,
    absoluteMax: 200,
    defaultMin: 0,
    defaultMax: 25,
    padding: 0.02,
  },
} as const;

export type ChartDomainConfig = typeof CHART_DOMAIN_CONFIG;
