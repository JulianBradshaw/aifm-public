export interface MilestoneInfo {
  time?: number | null;
  /** Effective compute level at the milestone, as log10 OOMs of FLOP (e.g. 30.2 => 10^30.2). */
  effective_compute_ooms?: number | null;
  [key: string]: number | string | boolean | null | undefined;
}

export type MilestoneMap = Record<string, MilestoneInfo>;


