'use client';

import { ControlPoint } from '@/utils/cdfSplineUtils';

// Display names for milestones
const MILESTONE_DISPLAY_NAMES: { [key: string]: string } = {
  'AC': 'AC (Automated Coder)',
  'AI2027-SC': 'AI2027-SC',
  'SAR-level-experiment-selection-skill': 'SAR (Superhuman AI Researcher)',
  'SIAR-level-experiment-selection-skill': 'SIAR (Superintelligent AI Researcher)',
  'TED-AI': 'TED-AI (Top Expert Dominating AI)',
  'ASI': 'ASI (Artificial Superintelligence)',
};

interface AtcMilestoneSelectorProps {
  milestones: string[];
  selectedMilestone: string;
  editedMilestones: Map<string, ControlPoint[]>;
  onMilestoneChange: (milestone: string) => void;
}

export function AtcMilestoneSelector({
  milestones,
  selectedMilestone,
  editedMilestones,
  onMilestoneChange,
}: AtcMilestoneSelectorProps) {
  const getDisplayName = (milestone: string) => {
    return MILESTONE_DISPLAY_NAMES[milestone] ?? milestone;
  };

  const isEdited = (milestone: string) => {
    return editedMilestones.has(milestone);
  };

  return (
    <div className="flex items-center gap-2">
      <label htmlFor="milestone-selector" className="text-sm font-medium text-gray-700">
        Milestone:
      </label>
      <select
        id="milestone-selector"
        value={selectedMilestone}
        onChange={(e) => onMilestoneChange(e.target.value)}
        className="block rounded-md border-gray-300 bg-white py-1.5 pl-3 pr-10 text-sm shadow-sm focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
      >
        {milestones.map((milestone) => (
          <option key={milestone} value={milestone}>
            {getDisplayName(milestone)} {isEdited(milestone) ? '✓' : ''}
          </option>
        ))}
      </select>
    </div>
  );
}
