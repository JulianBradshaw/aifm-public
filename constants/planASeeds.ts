import { PlanAKnobs } from '../app/types';

// Deal-breakdown knob seeds. Absent from the covert yaml, so they're seeded here rather
// than by /api/plan-a/defaults. Shared between the explorer client (PlanAExplorerClient)
// and the bake script (scripts/export/bake_plan_a_defaults.ts), which must replicate the
// client's first-compute request exactly so the baked result matches what a fresh client
// compute would return.
export const BREAKDOWN_DEFAULTS: PlanAKnobs = {
  enable_deal_breakdown: false,
  // Deliberately NOT matching legal_scaling_duration (5y), so the default breakdown lands
  // after the consortium reaches its target rather than exactly at it.
  deal_breakdown_years_after_pause_end: 6.5,
  breakdown_compute_mode: 'frontier_at_pause',
  breakdown_compute_reduction: 0.1,
  post_breakdown_compute_growth_rate: 1.0,
};
