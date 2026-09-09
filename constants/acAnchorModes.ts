// Shared display conventions for AC-anchor-mode breakout curves.
// Used by the /forecast breakout overlay (MilestoneDistributionChart) and the
// /atc breakout overlay (AtcPageClient) so both pages encode modes identically.

// Display names for AC-anchor modes
export const ANCHOR_MODE_DISPLAY_NAMES: { [key: string]: string } = {
  'uplift-solved': 'Uplift-solved anchor',
  'time-horizon': 'Time-horizon anchor',
  'revenue': 'Revenue anchor',
};

// Dash patterns distinguishing AC-anchor modes (assigned by index in modes[])
export const ANCHOR_MODE_DASH_PATTERNS = ['2 2', '9 3 2 3', '5 4'];
