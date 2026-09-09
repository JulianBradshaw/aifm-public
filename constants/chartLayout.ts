export const CHART_LAYOUT = {
  primary: {
    height: 400,
    width: 365,
    gridMinWidth: 1125,
  },
  metric: {
    width: 250,
    columnMinWidth: 250,
  },
  keyStats: {
    width: 300,
  },
} as const;

export type ChartLayout = typeof CHART_LAYOUT;

