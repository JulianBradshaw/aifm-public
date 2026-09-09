'use client';

import { useState } from 'react';
import type { SamplingConfig } from '@/types/samplingConfig';
import { SamplingDistributionsModal } from './SamplingDistributionsModal';

interface SamplingDistributionsButtonProps {
  config: SamplingConfig;
  rawYaml: string;
  simulationId: string;
}

export function SamplingDistributionsButton({
  config,
  rawYaml,
  simulationId,
}: SamplingDistributionsButtonProps) {
  const [showModal, setShowModal] = useState(false);

  return (
    <>
      <button
        onClick={() => setShowModal(true)}
        className="px-3 py-1 rounded text-sm bg-gray-200 text-gray-700 hover:bg-gray-300 transition-colors"
      >
        View Sampling Distributions
      </button>
      <SamplingDistributionsModal
        isOpen={showModal}
        onClose={() => setShowModal(false)}
        config={config}
        rawYaml={rawYaml}
        simulationId={simulationId}
      />
    </>
  );
}

export default SamplingDistributionsButton;
