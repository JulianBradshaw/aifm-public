'use client';

import { useState, useCallback, useEffect, useRef } from 'react';
import type { ChartDataPoint, BenchmarkPoint } from '@/app/types';
import type { MilestoneMap } from '@/types/milestones';
import { copyToClipboard, exportElementAsPng, EXPORT_PRESETS, EXPORT_COLORS } from '@/utils/chartExport';
import { MILESTONE_EXPLANATIONS } from '@/constants/chartExplanations';
import { CombinedChartExport } from './CombinedChartExport';
import { StaticHorizonChart } from './export/StaticHorizonChart';
import { StaticUpliftChart } from './export/StaticUpliftChart';
import { StaticMilestoneDates } from './export/StaticMilestoneDates';
import { StaticMilestoneUpliftExport } from './export/StaticMilestoneUpliftExport';
import { DownloadIcon } from './DownloadIcon';

type ExportTab = 'combined' | 'horizon' | 'uplift' | 'dates' | 'milestones-uplift';

export interface SharePredictionModalProps {
  isOpen: boolean;
  onClose: () => void;
  horizonChartData: ChartDataPoint[];
  upliftChartData: ChartDataPoint[];
  milestones: MilestoneMap | null;
  scHorizonMinutes: number;
  displayEndYear: number;
  shareUrl: string;
  benchmarkData?: BenchmarkPoint[];
  presentDay?: number;
}

const FONTS = {
  title: 'et-book, Georgia, serif',
  footer: 'Menlo, Consolas, monospace',
};

/**
 * Modal for sharing predictions with preview tabs, download, and copy link
 */
export function SharePredictionModal({
  isOpen,
  onClose,
  horizonChartData,
  upliftChartData,
  milestones,
  scHorizonMinutes,
  displayEndYear,
  shareUrl,
  benchmarkData = [],
  presentDay,
}: SharePredictionModalProps) {
  const [activeTab, setActiveTab] = useState<ExportTab>('milestones-uplift');
  const [isExporting, setIsExporting] = useState(false);
  const [copySuccess, setCopySuccess] = useState(false);
  const [showMETRData, setShowMETRData] = useState(true);
  const [showMilestoneIndicators, setShowMilestoneIndicators] = useState(true);
  const [trajectoryName, setTrajectoryName] = useState('My Trajectory');
  const exportRef = useRef<HTMLDivElement>(null);

  // Close on escape key
  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isOpen) {
        onClose();
      }
    };
    window.addEventListener('keydown', handleEscape);
    return () => window.removeEventListener('keydown', handleEscape);
  }, [isOpen, onClose]);

  // Prevent body scroll when modal is open
  useEffect(() => {
    if (isOpen) {
      document.body.style.overflow = 'hidden';
    } else {
      document.body.style.overflow = '';
    }
    return () => {
      document.body.style.overflow = '';
    };
  }, [isOpen]);

  const handleBackdropClick = useCallback(
    (e: React.MouseEvent) => {
      if (e.target === e.currentTarget) {
        onClose();
      }
    },
    [onClose]
  );

  const handleDownload = useCallback(async () => {
    if (!exportRef.current || isExporting) return;

    setIsExporting(true);
    try {
      // Convert trajectory name to URL-friendly slug
      const nameSlug = trajectoryName
        .toLowerCase()
        .replace(/[^a-z0-9]+/g, '-')
        .replace(/^-+|-+$/g, '');

      const tabSuffix: Record<ExportTab, string> = {
        'combined': 'combined',
        'horizon': 'time-horizons',
        'uplift': 'uplift',
        'milestones-uplift': 'default',
        'dates': 'milestones',
      };

      const filename = `ai-futures-model-${nameSlug}-${tabSuffix[activeTab]}`;

      const preset = activeTab === 'combined'
        ? EXPORT_PRESETS.combined
        : EXPORT_PRESETS.individual;

      await exportElementAsPng(exportRef.current, filename, preset);
    } catch (error) {
      console.error('Failed to export:', error);
    } finally {
      setIsExporting(false);
    }
  }, [activeTab, isExporting, trajectoryName]);

  const handleCopyLink = useCallback(async () => {
    const success = await copyToClipboard(shareUrl);
    if (success) {
      setCopySuccess(true);
      setTimeout(() => setCopySuccess(false), 2000);
    }
  }, [shareUrl]);

  if (!isOpen) return null;

  const tabs: { key: ExportTab; label: string }[] = [
    { key: 'milestones-uplift', label: 'Default' },
    { key: 'horizon', label: 'Time Horizons' },
    { key: 'uplift', label: 'Uplift' },
    { key: 'dates', label: 'Milestones' },
  ];

  // All exports use Twitter-optimal dimensions (1200x675)
  const exportWidth = 1200;
  const exportHeight = 675;

  // Calculate preview scale to fit in modal
  const previewScale = 0.55;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/50"
      onClick={handleBackdropClick}
    >
      <div className="bg-white rounded-lg shadow-xl max-w-5xl w-full mx-4 max-h-[90vh] flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-gray-200">
          <div className="flex-1 mr-4">
            <div className="flex items-center gap-3">
              <label htmlFor="trajectory-name" className="text-sm font-medium text-gray-700 whitespace-nowrap">
                Trajectory name:
              </label>
              <input
                id="trajectory-name"
                type="text"
                value={trajectoryName}
                onChange={(e) => setTrajectoryName(e.target.value)}
                className="flex-1 max-w-xs px-3 py-1.5 text-sm border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-gray-500 focus:border-transparent"
                placeholder="My Trajectory"
              />
            </div>
            <p className="text-sm text-gray-500 mt-1">
              Download an image or copy a link to share
            </p>
          </div>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-600 transition-colors p-1"
            aria-label="Close"
          >
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Tabs and Options */}
        <div className="flex items-center justify-between border-b border-gray-200 px-6">
          <div className="flex">
            {tabs.map(({ key, label }) => (
              <button
                key={key}
                onClick={() => setActiveTab(key)}
                className={`px-4 py-3 text-sm font-medium border-b-2 transition-colors ${activeTab === key
                    ? 'text-gray-900 border-gray-900'
                    : 'text-gray-500 border-transparent hover:text-gray-700 hover:border-gray-300'
                  }`}
              >
                {label}
              </button>
            ))}
          </div>
          {/* Export options */}
          <div className="hidden sm:flex items-center gap-4">
            {(activeTab === 'combined' || activeTab === 'horizon') && benchmarkData.length > 0 && (
              <label className="flex items-center gap-2 text-sm text-gray-600 cursor-pointer select-none">
                <input
                  type="checkbox"
                  checked={showMETRData}
                  onChange={(e) => setShowMETRData(e.target.checked)}
                  className="w-4 h-4 rounded border-gray-300 text-gray-900 focus:ring-gray-500"
                />
                Show METR datapoints
              </label>
            )}
            {activeTab !== 'dates' && activeTab !== 'horizon' && (
              <label className="flex items-center gap-2 text-sm text-gray-600 cursor-pointer select-none">
                <input
                  type="checkbox"
                  checked={showMilestoneIndicators}
                  onChange={(e) => setShowMilestoneIndicators(e.target.checked)}
                  className="w-4 h-4 rounded border-gray-300 text-gray-900 focus:ring-gray-500"
                />
                Show milestone indicators
              </label>
            )}
          </div>
        </div>

        {/* Preview Area */}
        <div className="flex-1 overflow-auto p-3 sm:p-6 bg-gray-50">
          <div className="flex justify-start sm:justify-center">
            {/* Preview container with CSS scaling for display only */}
            <div
              className="bg-white shadow-lg rounded-lg flex-shrink-0"
              style={{
                width: exportWidth * previewScale,
                height: exportHeight * previewScale,
              }}
            >
              {/* Export content at full size, scaled down via CSS transform */}
              <div
                ref={exportRef}
                style={{
                  width: exportWidth,
                  height: exportHeight,
                  transform: `scale(${previewScale})`,
                  transformOrigin: 'top left',
                }}
              >
                {activeTab === 'combined' && (
                  <CombinedChartExport
                    horizonChartData={horizonChartData}
                    upliftChartData={upliftChartData}
                    milestones={milestones}
                    scHorizonMinutes={scHorizonMinutes}
                    displayEndYear={displayEndYear}
                    benchmarkData={showMETRData ? benchmarkData : undefined}
                    showMilestoneIndicators={showMilestoneIndicators}
                    trajectoryName={trajectoryName}
                    presentDay={presentDay}
                  />
                )}
                {activeTab === 'horizon' && (
                  <div
                    style={{
                      width: 1200,
                      height: 675,
                      backgroundColor: EXPORT_COLORS.background,
                      display: 'flex',
                      flexDirection: 'column',
                      padding: '20px 32px',
                      boxSizing: 'border-box',
                    }}
                  >
                    <h2
                      style={{
                        textAlign: 'center',
                        margin: 0,
                        marginBottom: '12px',
                        fontSize: 32,
                        fontWeight: 'bold',
                        fontFamily: FONTS.title,
                        color: EXPORT_COLORS.foreground,
                      }}
                    >
                      {trajectoryName} - Coding Time Horizon (METR 80%)
                    </h2>
                    <div style={{ flex: 1, minHeight: 0 }}>
                      <StaticHorizonChart
                        chartData={horizonChartData}
                        scHorizonMinutes={scHorizonMinutes}
                        displayEndYear={displayEndYear}
                        width={1136}
                        height={480}
                        title=""
                        benchmarkData={showMETRData ? benchmarkData : undefined}
                        showMilestoneIndicators={showMilestoneIndicators}
                      />
                    </div>
                    <p
                      style={{
                        textAlign: 'center',
                        fontSize: 15,
                        fontFamily: FONTS.footer,
                        color: EXPORT_COLORS.foreground,
                        margin: 0,
                        marginTop: '8px',
                        lineHeight: 1.4,
                      }}
                    >
                      The coding time horizon is the maximum length of coding tasks frontier AI systems can complete with a success rate of 80%, with the length defined as the time taken by typical AI company employees who do similar tasks.
                    </p>
                    <p
                      style={{
                        textAlign: 'center',
                        fontSize: 20,
                        fontWeight: 'bold',
                        fontFamily: FONTS.footer,
                        color: EXPORT_COLORS.foreground,
                        margin: 0,
                        marginTop: '6px',
                        opacity: 0.2,
                      }}
                    >
                      Made with aifuturesmodel.com
                    </p>
                  </div>
                )}
                {activeTab === 'uplift' && (
                  <div
                    style={{
                      width: 1200,
                      height: 675,
                      backgroundColor: EXPORT_COLORS.background,
                      display: 'flex',
                      flexDirection: 'column',
                      padding: '20px 32px',
                      boxSizing: 'border-box',
                    }}
                  >
                    <h2
                      style={{
                        textAlign: 'center',
                        margin: 0,
                        marginBottom: '12px',
                        fontSize: 32,
                        fontWeight: 'bold',
                        fontFamily: FONTS.title,
                        color: EXPORT_COLORS.foreground,
                      }}
                    >
                      {trajectoryName} - AI Software R&D Uplift
                    </h2>
                    <div style={{ flex: 1, minHeight: 0 }}>
                      <StaticUpliftChart
                        chartData={upliftChartData}
                        milestones={milestones}
                        displayEndYear={displayEndYear}
                        width={1136}
                        height={480}
                        title=""
                        showMilestoneIndicators={showMilestoneIndicators}
                        presentDay={presentDay}
                      />
                    </div>
                    <p
                      style={{
                        textAlign: 'center',
                        fontSize: 15,
                        fontFamily: FONTS.footer,
                        color: EXPORT_COLORS.foreground,
                        margin: 0,
                        marginTop: '8px',
                        lineHeight: 1.4,
                      }}
                    >
                      The AI Software R&D Uplift is the speedup in software progress that would result if the frontier AI systems at a given time were deployed within today&apos;s leading AI company.
                    </p>
                    <p
                      style={{
                        textAlign: 'center',
                        fontSize: 20,
                        fontWeight: 'bold',
                        fontFamily: FONTS.footer,
                        color: EXPORT_COLORS.foreground,
                        margin: 0,
                        marginTop: '6px',
                        opacity: 0.2,
                      }}
                    >
                      Made with aifuturesmodel.com
                    </p>
                  </div>
                )}
                {activeTab === 'dates' && (
                  <div
                    style={{
                      width: 1200,
                      height: 675,
                      backgroundColor: EXPORT_COLORS.background,
                      display: 'flex',
                      flexDirection: 'column',
                      alignItems: 'center',
                      justifyContent: 'flex-start',
                      padding: '24px 60px',
                      boxSizing: 'border-box',
                    }}
                  >
                    <h2
                      style={{
                        textAlign: 'center',
                        margin: 0,
                        marginBottom: '24px',
                        fontSize: 32,
                        fontWeight: 'bold',
                        fontFamily: FONTS.title,
                        color: EXPORT_COLORS.foreground,
                      }}
                    >
                      {trajectoryName} - AI Milestones
                    </h2>
                    <StaticMilestoneDates
                      milestones={[
                        {
                          label: 'Date of Automated Coder (AC)',
                          decimalYear: (milestones?.['AC']?.time as number | null) ?? null,
                          explanation: MILESTONE_EXPLANATIONS['AC'],
                          effectiveComputeOOMs: (milestones?.['AC']?.effective_compute_ooms as number | null) ?? null,
                        },
                        {
                          label: 'Date of Top-Expert-Dominating AI (TED-AI)',
                          decimalYear: (milestones?.['TED-AI']?.time as number | null) ?? null,
                          explanation: MILESTONE_EXPLANATIONS['TED-AI'],
                          effectiveComputeOOMs: (milestones?.['TED-AI']?.effective_compute_ooms as number | null) ?? null,
                        },
                        {
                          label: 'Date of Superintelligence (ASI)',
                          decimalYear: (milestones?.['ASI']?.time as number | null) ?? null,
                          explanation: MILESTONE_EXPLANATIONS['ASI'],
                          effectiveComputeOOMs: (milestones?.['ASI']?.effective_compute_ooms as number | null) ?? null,
                        },
                      ]}
                      simulationEndYear={2200}
                      width={850}
                    />
                    <p
                      style={{
                        textAlign: 'center',
                        fontSize: 20,
                        fontWeight: 'bold',
                        fontFamily: FONTS.footer,
                        color: EXPORT_COLORS.foreground,
                        margin: 0,
                        marginTop: 'auto',
                        paddingTop: '16px',
                        opacity: 0.2,
                      }}
                    >
                      Made with aifuturesmodel.com
                    </p>
                  </div>
                )}
                {activeTab === 'milestones-uplift' && (
                  <div
                    style={{
                      width: 1200,
                      height: 675,
                      backgroundColor: EXPORT_COLORS.background,
                      display: 'flex',
                      flexDirection: 'column',
                      padding: '20px 32px',
                      boxSizing: 'border-box',
                    }}
                  >
                    <h2
                      style={{
                        textAlign: 'center',
                        margin: 0,
                        marginTop: '12px',
                        marginBottom: '0px',
                        fontSize: 40,
                        fontWeight: 'bold',
                        fontFamily: FONTS.title,
                        color: EXPORT_COLORS.foreground,
                      }}
                    >
                      {trajectoryName}
                    </h2>
                    <div style={{ flex: 1, minHeight: 0 }}>
                      <StaticMilestoneUpliftExport
                        chartData={upliftChartData}
                        milestones={milestones}
                        displayEndYear={displayEndYear}
                        width={1136}
                        height={560}
                        showMilestoneIndicators={showMilestoneIndicators}
                        presentDay={presentDay}
                      />
                    </div>
                    <p
                      style={{
                        textAlign: 'center',
                        fontSize: 20,
                        fontWeight: 'bold',
                        fontFamily: FONTS.footer,
                        color: EXPORT_COLORS.foreground,
                        margin: 0,
                        marginTop: '8px',
                        opacity: 0.2,
                      }}
                    >
                      Made with aifuturesmodel.com
                    </p>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Mobile export options (moved below preview) */}
          <div className="sm:hidden mt-4 bg-white border border-gray-200 rounded-lg p-3">
            <div className="flex flex-col gap-3">
              {(activeTab === 'combined' || activeTab === 'horizon') && benchmarkData.length > 0 && (
                <label className="flex items-center gap-2 text-sm text-gray-700 cursor-pointer select-none">
                  <input
                    type="checkbox"
                    checked={showMETRData}
                    onChange={(e) => setShowMETRData(e.target.checked)}
                    className="w-4 h-4 rounded border-gray-300 text-gray-900 focus:ring-gray-500"
                  />
                  Show METR datapoints
                </label>
              )}
              {activeTab !== 'dates' && activeTab !== 'horizon' && (
                <label className="flex items-center gap-2 text-sm text-gray-700 cursor-pointer select-none">
                  <input
                    type="checkbox"
                    checked={showMilestoneIndicators}
                    onChange={(e) => setShowMilestoneIndicators(e.target.checked)}
                    className="w-4 h-4 rounded border-gray-300 text-gray-900 focus:ring-gray-500"
                  />
                  Show milestone indicators
                </label>
              )}
            </div>
          </div>
        </div>

        {/* Actions */}
        <div className="flex items-center justify-end px-6 py-4 border-t border-gray-200 bg-white">
          <div className="flex gap-3">
            <button
              onClick={handleCopyLink}
              className={`px-4 py-2 text-sm font-medium rounded-md border transition-colors ${copySuccess
                  ? 'bg-green-50 text-green-700 border-green-300'
                  : 'bg-white text-gray-700 border-gray-300 hover:bg-gray-50'
                }`}
            >
              {copySuccess ? 'Copied!' : 'Copy Link'}
            </button>
            <button
              onClick={handleDownload}
              disabled={isExporting}
              className="px-4 py-2 text-sm font-medium text-white bg-gray-900 rounded-md hover:bg-gray-800 transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
            >
              {isExporting ? (
                <>
                  <span className="h-4 w-4 animate-spin rounded-full border-2 border-white border-t-transparent" />
                  Exporting...
                </>
              ) : (
                <>
                  <DownloadIcon size={16} />
                  Download PNG
                </>
              )}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
