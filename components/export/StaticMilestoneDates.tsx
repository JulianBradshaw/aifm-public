'use client';

import { formatEffectiveComputeOOMsNode } from '@/utils/formatting';

export interface MilestoneConfig {
  label: string;
  decimalYear: number | null;
  explanation?: string;
  /** Effective compute level at the milestone, as log10 OOMs of FLOP. */
  effectiveComputeOOMs?: number | null;
}

export interface StaticMilestoneDatesProps {
  milestones: MilestoneConfig[];
  simulationEndYear?: number;
  width?: number;
}

const COLORS = {
  foreground: '#0D0D0D',
  divider: 'rgba(100, 116, 139, 0.3)',
};

const FONTS = {
  label: 'Menlo, Consolas, monospace',
  date: 'et-book, Georgia, serif',
};

interface MilestoneDisplay {
  monthNumber: number;
  year: number;
}

function getMilestoneDisplay(
  decimalYear: number | null,
  simulationEndYear: number
): MilestoneDisplay | null {
  if (decimalYear == null || !Number.isFinite(decimalYear)) {
    return null;
  }
  if (decimalYear > simulationEndYear) {
    return null;
  }

  const year = Math.floor(decimalYear);
  const fraction = decimalYear - year;
  const monthIndexRaw = Math.round(fraction * 12);
  const monthIndex = monthIndexRaw % 12;
  const adjustedYear = monthIndexRaw === 12 ? year + 1 : year;

  return {
    monthNumber: monthIndex + 1,
    year: adjustedYear,
  };
}

function formatDateChars(
  milestone: MilestoneDisplay | null,
  simulationEndYear: number
): string[] {
  if (!milestone) {
    return ['>', String(simulationEndYear)].join('').split('');
  }
  const month = String(milestone.monthNumber).padStart(2, '0');
  // Match front page format: "07/2030" (no spaces around slash)
  return `${month}/${milestone.year}`.split('');
}

/**
 * Static milestone dates display for PNG export
 * Renders milestone dates in large spaced-out digits with explanations to the right
 */
export function StaticMilestoneDates({
  milestones,
  simulationEndYear = 2045,
  width = 480,
}: StaticMilestoneDatesProps) {
  // Scale factors based on width
  const scale = width / 480;
  const labelFontSize = Math.round(13 * scale);
  const dateFontSize = Math.round(42 * scale); // Sized to fit 3 milestones with title and footer
  const explanationFontSize = Math.round(10 * scale);

  // Check if any milestone has an explanation
  const hasExplanations = milestones.some(m => m.explanation);

  // Width allocation when explanations are present - wider date area for better kerning like front page
  const dateWidth = hasExplanations ? Math.round(width * 0.48) : width;

  return (
    <div
      style={{
        width,
        display: 'flex',
        flexDirection: 'column',
        gap: 12 * scale,
      }}
    >
      {milestones.map((milestone, index) => {
        const display = getMilestoneDisplay(milestone.decimalYear, simulationEndYear);
        const dateChars = formatDateChars(display, simulationEndYear);

        return (
          <div key={index} style={{ display: 'flex', flexDirection: 'column' }}>
            {/* Label row with divider */}
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 8 * scale,
              }}
            >
              <span
                style={{
                  fontSize: labelFontSize,
                  fontWeight: 600,
                  fontFamily: FONTS.label,
                  color: COLORS.foreground,
                  whiteSpace: 'nowrap',
                }}
              >
                {milestone.label}
              </span>
              <div
                style={{
                  flex: 1,
                  height: 1,
                  backgroundColor: COLORS.divider,
                }}
              />
            </div>

            {/* Date + Explanation row - using table layout for html2canvas compatibility */}
            <div
              style={{
                display: 'table',
                width: '100%',
                marginTop: 4 * scale,
              }}
            >
              {/* Date characters cell */}
              <div
                style={{
                  display: 'table-cell',
                  verticalAlign: 'middle',
                  width: dateWidth,
                }}
              >
                <div
                  style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                  }}
                >
                  {dateChars.map((char, charIndex) => (
                    <span
                      key={charIndex}
                      style={{
                        fontSize: dateFontSize,
                        fontWeight: 'bold',
                        fontFamily: FONTS.date,
                        color: COLORS.foreground,
                      }}
                    >
                      {char}
                    </span>
                  ))}
                </div>
                {display && milestone.effectiveComputeOOMs != null && (
                  <div
                    style={{
                      fontSize: Math.round(13 * scale),
                      fontFamily: FONTS.label,
                      color: COLORS.foreground,
                      opacity: 0.7,
                      marginTop: 2 * scale,
                      whiteSpace: 'nowrap',
                    }}
                  >
                    {formatEffectiveComputeOOMsNode(milestone.effectiveComputeOOMs)} effective compute
                  </div>
                )}
              </div>

              {/* Spacer cell */}
              <div
                style={{
                  display: 'table-cell',
                  width: 20 * scale,
                }}
              />

              {/* Explanation text cell */}
              {milestone.explanation && (
                <div
                  style={{
                    display: 'table-cell',
                    verticalAlign: 'middle',
                  }}
                >
                  <p
                    style={{
                      margin: 0,
                      fontSize: explanationFontSize,
                      fontFamily: FONTS.label,
                      color: COLORS.foreground,
                      lineHeight: 1.35,
                      opacity: 0.85,
                    }}
                  >
                    {milestone.explanation}
                  </p>
                </div>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}
