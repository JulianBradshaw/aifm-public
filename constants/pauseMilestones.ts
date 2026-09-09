// Curated, ordered set of milestones offered as "pause at milestone" triggers.
// Keys match the milestone keys returned by the backend (/api/compute -> milestones).
// Mirrors the headline set shown in the timeline sidebar.
export interface PauseMilestoneDef {
  key: string;
  name: string;
  fullName: string;
}

export const PAUSE_MILESTONE_DEFS: PauseMilestoneDef[] = [
  { key: 'AC', name: 'AC', fullName: 'Automated Coder' },
  { key: 'AI2027-SC', name: 'SC', fullName: 'Superhuman Coder' },
  { key: 'SAR-level-experiment-selection-skill', name: 'SAR', fullName: 'Superhuman AI Researcher' },
  { key: 'SIAR-level-experiment-selection-skill', name: 'SIAR', fullName: 'Superintelligent AI Researcher' },
  { key: 'TED-AI', name: 'TED-AI', fullName: 'Top-Expert-Dominating AI' },
  { key: 'ASI', name: 'ASI', fullName: 'Artificial Superintelligence' },
];
