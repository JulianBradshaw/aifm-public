'use client';

import { useState, useEffect, useCallback, useMemo } from 'react';
import { useSearchParams } from 'next/navigation';
import { Simulation } from '@/components/SimulationSelector';
import { AtcMilestoneSelector } from '@/components/AtcMilestoneSelector';
import { EditableCdfChart, CdfOverlayLine } from '@/components/EditableCdfChart';
import { PdfLineChart, DataPoint } from '@/components/PdfLineChart';
import { ChartSyncProvider } from '@/components/ChartSyncContext';
import { HeaderContent } from '@/components/HeaderContent';
import { ControlPointsTable } from '@/components/ControlPointsTable';
import {
  ControlPoint,
  CdfPoint,
  TakeoffCdfPoint,
  InterpolationMethod,
  initializeControlPoints,
  initializeControlPointsFromTakeoff,
  evaluateSplineAt,
  samplePdf,
  computePdfFromCdf,
} from '@/utils/cdfSplineUtils';
import {
  evaluateGammaMixtureAt,
  sampleGammaMixturePdf,
} from '@/utils/gammaMixtureUtils';
import { CdfDataResponse } from '@/app/api/atc-cdf/route';
import { TakeoffCdfDataResponse } from '@/app/api/atc-takeoff-cdf/route';
import { CHART_DOMAIN_CONFIG } from '@/constants/chartDomainConfig';
import { ANCHOR_MODE_DISPLAY_NAMES, ANCHOR_MODE_DASH_PATTERNS } from '@/constants/acAnchorModes';
import { ChartZoomControls } from '@/components/ChartZoomControls';

// Format a decimal year as "Mon YYYY" (e.g., 2032.5 -> "Jul 2032")
function formatDecimalYearAsDate(decimalYear: number): string {
  const year = Math.floor(decimalYear);
  const fraction = decimalYear - year;
  const monthIndex = Math.round(fraction * 12);
  const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
  return `${months[Math.min(monthIndex, 11)]} ${year}`;
}

// Color palette for milestones
const MILESTONE_COLORS: { [key: string]: string } = {
  'AC': '#2A623D',
  'AI2027-SC': '#af1e86ff',
  'SAR-level-experiment-selection-skill': '#000090',
  'SIAR-level-experiment-selection-skill': '#900000',
  'TED-AI': '#2A623D',
  'ASI': '#af1e86ff',
};

interface AtcPageClientProps {
  simulations: Simulation[];
  defaultSimulationId: string | null;
}

// Sort simulations by date (newest first)
function sortSimulationsByDate(simulations: Simulation[]): Simulation[] {
  return [...simulations].sort((a, b) => {
    const parseDate = (dateStr: string): number => {
      const [month, day, year] = dateStr.split('-').map(Number);
      const fullYear = year < 100 ? 2000 + year : year;
      return new Date(fullYear, month - 1, day).getTime();
    };
    return parseDate(b.date) - parseDate(a.date);
  });
}

// Per-milestone interpolation settings. Declared at module scope so the helpers
// below can use the type, and so defaultSettings has a stable identity — as a
// value recreated per render it was an unlisted dependency of eleven hooks.
interface MilestoneSettings {
  method: InterpolationMethod;
  gammaReg: number;
  isplineReg: number;
}
const defaultSettings: MilestoneSettings = { method: 'ispline', gammaReg: 0.01, isplineReg: 0.05 };

// URL (de)serialisation helpers. Pure and stateless, so they live at module
// scope: that gives them stable identities and makes them usable from the
// lazy useState initialisers below, which run before the component body.
function serializeControlPoints(points: ControlPoint[]): string {
    // Serialize to a compact format: x1,y1,l1;x2,y2,l2;...
    return points.map(p => `${p.x.toFixed(4)},${p.y.toFixed(4)},${p.isLocked ? '1' : '0'}`).join(';');
}

function deserializeControlPoints(str: string): ControlPoint[] {
    if (!str) return [];
    try {
      return str.split(';').map((part, i) => {
        const [x, y, locked] = part.split(',');
        return {
          id: `cp-url-${i}-${Date.now()}`,
          x: parseFloat(x),
          y: parseFloat(y),
          isLocked: locked === '1',
        };
      });
    } catch {
      return [];
    }
}

function serializeAllMilestones(milestones: Map<string, ControlPoint[]>): string {
    const parts: string[] = [];
    milestones.forEach((points, name) => {
      if (points.length >= 2) {
        parts.push(`${encodeURIComponent(name)}~${serializeControlPoints(points)}`);
      }
    });
    return parts.join('|');
}

function deserializeAllMilestones(str: string): Map<string, ControlPoint[]> {
    const result = new Map<string, ControlPoint[]>();
    if (!str) return result;
    try {
      const parts = str.split('|');
      for (const part of parts) {
        const [encodedName, pointsStr] = part.split('~');
        if (encodedName && pointsStr) {
          const name = decodeURIComponent(encodedName);
          const points = deserializeControlPoints(pointsStr);
          if (points.length >= 2) {
            result.set(name, points);
          }
        }
      }
    } catch {
      // Ignore parse errors
    }
    return result;
}

function serializeAllSettings(settings: Map<string, MilestoneSettings>): string {
    const parts: string[] = [];
    settings.forEach((s, name) => {
      parts.push(`${encodeURIComponent(name)}~${s.method},${s.gammaReg.toFixed(3)},${s.isplineReg.toFixed(3)}`);
    });
    return parts.join('|');
}

function deserializeAllSettings(str: string): Map<string, MilestoneSettings> {
    const result = new Map<string, MilestoneSettings>();
    if (!str) return result;
    try {
      const parts = str.split('|');
      for (const part of parts) {
        const [encodedName, settingsStr] = part.split('~');
        if (encodedName && settingsStr) {
          const name = decodeURIComponent(encodedName);
          const [method, gammaReg, isplineReg] = settingsStr.split(',');
          if (method === 'gamma' || method === 'ispline') {
            result.set(name, {
              method,
              gammaReg: parseFloat(gammaReg) || defaultSettings.gammaReg,
              isplineReg: parseFloat(isplineReg) || defaultSettings.isplineReg,
            });
          }
        }
      }
    } catch {
      // Ignore parse errors
    }
    return result;
}

export function AtcPageClient({ simulations, defaultSimulationId }: AtcPageClientProps) {
  const sortedSimulations = useMemo(() => sortSimulationsByDate(simulations), [simulations]);

  // Read the shareable URL state once, as the initial value of each piece of
  // state below, so the first render is already correct and nothing renders
  // defaults and then jumps. useSearchParams (rather than window.location)
  // keeps the initialisers safe to run during SSR.
  const searchParams = useSearchParams();

  // State - Timelines
  const [selectedSimulationId, setSelectedSimulationId] = useState<string>(() => {
    const sim = searchParams.get('sim');
    if (sim && simulations.some(s => s.id === sim)) return sim;
    return defaultSimulationId ?? sortSimulationsByDate(simulations)[0]?.id ?? '';
  });
  const [selectedMilestone, setSelectedMilestone] = useState<string>(
    () => searchParams.get('timelineMilestone') ?? ''
  );
  const [cdfData, setCdfData] = useState<CdfDataResponse | null>(null);
  const [editedMilestones, setEditedMilestones] = useState<Map<string, ControlPoint[]>>(() => {
    const raw = searchParams.get('timelines');
    return raw ? deserializeAllMilestones(raw) : new Map();
  });

  // Derived, not stored. Every edit writes straight into editedMilestones (see
  // handleControlPointsChange), so the selected milestone's points are exactly
  // its saved edits, or the empirical-CDF seed when it has none.
  const currentControlPoints = useMemo<ControlPoint[]>(() => {
    if (!cdfData || !selectedMilestone) return [];
    const saved = editedMilestones.get(selectedMilestone);
    if (saved) return saved;
    const milestoneData = cdfData.data[selectedMilestone];
    if (!milestoneData) return [];
    const cdfPoints: CdfPoint[] = milestoneData.map(p => ({ year: p.year, cdf: p.cdf }));
    return initializeControlPoints(cdfPoints);
  }, [cdfData, selectedMilestone, editedMilestones]);
  // Which simulation the held CDF payload belongs to. Loading and error are
  // derived from it rather than set synchronously at the top of the fetch
  // effect: we are loading when what we hold is not what is selected, and a
  // failure only counts while it still describes the current selection.
  const [cdfLoadedFor, setCdfLoadedFor] = useState<string | null>(null);
  const [cdfError, setCdfError] = useState<string | null>(null);
  const isLoading = !!selectedSimulationId && cdfLoadedFor !== selectedSimulationId;
  const error = cdfLoadedFor === selectedSimulationId ? cdfError : null;

  // State - Takeoff Speeds
  const [takeoffCdfData, setTakeoffCdfData] = useState<TakeoffCdfDataResponse | null>(null);
  const [selectedTakeoffMilestone, setSelectedTakeoffMilestone] = useState<string>(
    () => searchParams.get('takeoffMilestone') ?? ''
  );
  const [editedTakeoffMilestones, setEditedTakeoffMilestones] = useState<Map<string, ControlPoint[]>>(() => {
    const raw = searchParams.get('takeoffs');
    return raw ? deserializeAllMilestones(raw) : new Map();
  });

  // Same derivation as the timeline chart above.
  const currentTakeoffControlPoints = useMemo<ControlPoint[]>(() => {
    if (!takeoffCdfData || !selectedTakeoffMilestone) return [];
    const saved = editedTakeoffMilestones.get(selectedTakeoffMilestone);
    if (saved) return saved;
    const milestoneData = takeoffCdfData.data[selectedTakeoffMilestone];
    if (!milestoneData) return [];
    const cdfPoints: TakeoffCdfPoint[] = milestoneData.map(p => ({ timeFromAC: p.timeFromAC, cdf: p.cdf }));
    return initializeControlPointsFromTakeoff(cdfPoints);
  }, [takeoffCdfData, selectedTakeoffMilestone, editedTakeoffMilestones]);
  const [takeoffLoadedFor, setTakeoffLoadedFor] = useState<string | null>(null);
  const [takeoffCdfError, setTakeoffCdfError] = useState<string | null>(null);
  const isTakeoffLoading = !!selectedSimulationId && takeoffLoadedFor !== selectedSimulationId;
  const takeoffError = takeoffLoadedFor === selectedSimulationId ? takeoffCdfError : null;

  // State - Zoom/scale controls (null means auto-compute from data)
  const [timelineZoomDomain, setTimelineZoomDomain] = useState<[number, number] | null>(null);
  const [takeoffZoomDomain, setTakeoffZoomDomain] = useState<[number, number] | null>(null);

  // State - Overlay all milestones on PDF chart
  const [showAllTimelines, setShowAllTimelines] = useState(false);
  const [showAllTakeoffs, setShowAllTakeoffs] = useState(false);

  // State - Break out the base (model) forecast by AC-anchor method.
  // Arrival/timeline section only: the takeoff CSVs are not broken out by
  // anchor mode (arrival-only by design), so the takeoff section has no
  // breakout control.
  const [showModeBreakout, setShowModeBreakout] = useState(() => searchParams.get('breakout') === '1');

  // State - Per-milestone interpolation settings (maps milestone name -> settings)
  const [milestoneSettings, setMilestoneSettings] = useState<Map<string, MilestoneSettings>>(() => {
    const raw = searchParams.get('timelineSettings');
    return raw ? deserializeAllSettings(raw) : new Map();
  });
  const [takeoffSettings, setTakeoffSettings] = useState<Map<string, MilestoneSettings>>(() => {
    const raw = searchParams.get('takeoffSettings');
    return raw ? deserializeAllSettings(raw) : new Map();
  });

  // Get current milestone's settings (with defaults)
  const currentSettings = useMemo(() => {
    if (!selectedMilestone) return defaultSettings;
    return milestoneSettings.get(selectedMilestone) ?? defaultSettings;
  }, [selectedMilestone, milestoneSettings]);

  const currentTakeoffSettings = useMemo(() => {
    if (!selectedTakeoffMilestone) return defaultSettings;
    return takeoffSettings.get(selectedTakeoffMilestone) ?? defaultSettings;
  }, [selectedTakeoffMilestone, takeoffSettings]);

  // Helper to update settings for current milestone
  const updateCurrentSettings = useCallback((updates: Partial<MilestoneSettings>) => {
    if (!selectedMilestone) return;
    setMilestoneSettings(prev => {
      const next = new Map(prev);
      const current = prev.get(selectedMilestone) ?? defaultSettings;
      next.set(selectedMilestone, { ...current, ...updates });
      return next;
    });
  }, [selectedMilestone]);

  const updateCurrentTakeoffSettings = useCallback((updates: Partial<MilestoneSettings>) => {
    if (!selectedTakeoffMilestone) return;
    setTakeoffSettings(prev => {
      const next = new Map(prev);
      const current = prev.get(selectedTakeoffMilestone) ?? defaultSettings;
      next.set(selectedTakeoffMilestone, { ...current, ...updates });
      return next;
    });
  }, [selectedTakeoffMilestone]);

  // URL serialization helpers


  // Serialize all edited milestones to URL format: name1~points1|name2~points2

  // Deserialize all edited milestones from URL format

  // Serialize all milestone settings to URL format: name1~method,gammaReg,isplineReg|name2~...

  // Deserialize all milestone settings from URL format


  // Update URL when state changes
  const updateUrl = useCallback(() => {
    const params = new URLSearchParams();

    // Save selected simulation/forecast
    if (selectedSimulationId) {
      params.set('sim', selectedSimulationId);
    }

    // Build complete editedMilestones map (including current unsaved edits)
    const allEditedMilestones = new Map(editedMilestones);
    if (selectedMilestone && currentControlPoints.length >= 2) {
      allEditedMilestones.set(selectedMilestone, currentControlPoints);
    }

    // Build complete milestoneSettings map (including current settings)
    const allMilestoneSettings = new Map(milestoneSettings);
    if (selectedMilestone) {
      allMilestoneSettings.set(selectedMilestone, currentSettings);
    }

    // Save ALL timeline edited milestones and settings
    if (allEditedMilestones.size > 0) {
      params.set('timelines', serializeAllMilestones(allEditedMilestones));
    }
    if (allMilestoneSettings.size > 0) {
      params.set('timelineSettings', serializeAllSettings(allMilestoneSettings));
    }
    // Save selected milestone for UI state
    if (selectedMilestone) {
      params.set('timelineMilestone', selectedMilestone);
    }

    // Build complete editedTakeoffMilestones map (including current unsaved edits)
    const allEditedTakeoffMilestones = new Map(editedTakeoffMilestones);
    if (selectedTakeoffMilestone && currentTakeoffControlPoints.length >= 2) {
      allEditedTakeoffMilestones.set(selectedTakeoffMilestone, currentTakeoffControlPoints);
    }

    // Build complete takeoffSettings map (including current settings)
    const allTakeoffSettings = new Map(takeoffSettings);
    if (selectedTakeoffMilestone) {
      allTakeoffSettings.set(selectedTakeoffMilestone, currentTakeoffSettings);
    }

    // Save ALL takeoff edited milestones and settings
    if (allEditedTakeoffMilestones.size > 0) {
      params.set('takeoffs', serializeAllMilestones(allEditedTakeoffMilestones));
    }
    if (allTakeoffSettings.size > 0) {
      params.set('takeoffSettings', serializeAllSettings(allTakeoffSettings));
    }
    // Save selected takeoff milestone for UI state
    if (selectedTakeoffMilestone) {
      params.set('takeoffMilestone', selectedTakeoffMilestone);
    }

    // Save AC-anchor breakout toggle (only meaningful when the sim has breakout data)
    if (showModeBreakout) {
      params.set('breakout', '1');
    }

    // Update URL without reloading
    const newUrl = `${window.location.pathname}?${params.toString()}`;
    window.history.replaceState({}, '', newUrl);
  }, [
    showModeBreakout,
    selectedSimulationId,
    selectedMilestone,
    currentControlPoints,
    currentSettings,
    editedMilestones,
    milestoneSettings,
    selectedTakeoffMilestone,
    currentTakeoffControlPoints,
    currentTakeoffSettings,
    editedTakeoffMilestones,
    takeoffSettings,
  ]);

  // Debounced URL update
  useEffect(() => {
    const timer = setTimeout(updateUrl, 500);
    return () => clearTimeout(timer);
  }, [updateUrl]);

  // Handle forecast/simulation change - edits belong to the old forecast, so clear them
  const handleSimulationChange = useCallback((newSimulationId: string) => {
    if (newSimulationId !== selectedSimulationId) {
      setEditedMilestones(new Map());
      setEditedTakeoffMilestones(new Map());
    }
    setSelectedSimulationId(newSimulationId);
  }, [selectedSimulationId]);

  // Load CDF data when simulation changes
  useEffect(() => {
    if (!selectedSimulationId) return;
    let cancelled = false;

    const requestedSimId = selectedSimulationId;

    fetch(`/api/atc-cdf?simulationId=${encodeURIComponent(requestedSimId)}`)
      .then(res => {
        if (!res.ok) throw new Error(`Failed to load data: ${res.status}`);
        return res.json();
      })
      .then((data: CdfDataResponse) => {
        if (cancelled) return;
        setCdfData(data);
        setCdfError(null);
        setCdfLoadedFor(requestedSimId);
        // Select first milestone by default
        if (data.milestones.length > 0) {
          setSelectedMilestone(prev => prev || data.milestones[0]);
        }
      })
      .catch(err => {
        if (cancelled) return;
        setCdfError(err.message);
        setCdfData(null);
        setCdfLoadedFor(requestedSimId);
      });

    return () => {
      cancelled = true;
    };
  }, [selectedSimulationId]);

  // Load Takeoff CDF data when simulation changes
  useEffect(() => {
    if (!selectedSimulationId) return;
    let cancelled = false;

    const requestedSimId = selectedSimulationId;

    fetch(`/api/atc-takeoff-cdf?simulationId=${encodeURIComponent(requestedSimId)}`)
      .then(res => {
        if (!res.ok) throw new Error(`Failed to load takeoff data: ${res.status}`);
        return res.json();
      })
      .then((data: TakeoffCdfDataResponse) => {
        if (cancelled) return;
        setTakeoffCdfData(data);
        setTakeoffCdfError(null);
        setTakeoffLoadedFor(requestedSimId);
        // Select first milestone by default
        if (data.milestones.length > 0) {
          setSelectedTakeoffMilestone(prev => prev || data.milestones[0]);
        }
      })
      .catch(err => {
        if (cancelled) return;
        setTakeoffCdfError(err.message);
        setTakeoffCdfData(null);
        setTakeoffLoadedFor(requestedSimId);
      });

    return () => {
      cancelled = true;
    };
  }, [selectedSimulationId]);

  // Save control points when they change
  const handleControlPointsChange = useCallback((newPoints: ControlPoint[]) => {
    if (selectedMilestone) {
      setEditedMilestones(prev => {
        const next = new Map(prev);
        next.set(selectedMilestone, newPoints);
        return next;
      });
    }
  }, [selectedMilestone]);

  // Handle milestone change. No save-before-switch step is needed: edits go
  // straight into editedMilestones as they happen, and an untouched milestone
  // must stay absent from that map rather than being marked edited.
  const handleMilestoneChange = useCallback((newMilestone: string) => {
    setSelectedMilestone(newMilestone);
  }, []);

  // Save takeoff control points when they change
  const handleTakeoffControlPointsChange = useCallback((newPoints: ControlPoint[]) => {
    if (selectedTakeoffMilestone) {
      setEditedTakeoffMilestones(prev => {
        const next = new Map(prev);
        next.set(selectedTakeoffMilestone, newPoints);
        return next;
      });
    }
  }, [selectedTakeoffMilestone]);

  // As above — switching is just a selection change now.
  const handleTakeoffMilestoneChange = useCallback((newMilestone: string) => {
    setSelectedTakeoffMilestone(newMilestone);
  }, []);

  // Reset current milestone to empirical: drop the saved edit and the derived
  // control points fall back to the empirical-CDF seed on their own.
  const handleReset = useCallback(() => {
    if (!selectedMilestone) return;
    setEditedMilestones(prev => {
      const next = new Map(prev);
      next.delete(selectedMilestone);
      return next;
    });
  }, [selectedMilestone]);

  // As above.
  const handleTakeoffReset = useCallback(() => {
    if (!selectedTakeoffMilestone) return;
    setEditedTakeoffMilestones(prev => {
      const next = new Map(prev);
      next.delete(selectedTakeoffMilestone);
      return next;
    });
  }, [selectedTakeoffMilestone]);

  // Export all milestones to CSV
  const handleExport = useCallback(() => {
    if (!cdfData) return;

    // Use the original time points from the data
    const timePoints = cdfData.timePoints;

    // Build CSV rows
    const headers = ['time_decimal_year', ...cdfData.milestones];
    const rows: string[] = [headers.join(',')];

    for (const year of timePoints) {
      const values: string[] = [year.toFixed(6)];

      for (const milestone of cdfData.milestones) {
        const edited = editedMilestones.get(milestone);
        let cdfValue: number;

        if (edited && edited.length >= 2) {
          // Use per-milestone settings for evaluation
          const settings = milestoneSettings.get(milestone) ?? defaultSettings;
          if (settings.method === 'gamma') {
            cdfValue = evaluateGammaMixtureAt(edited, year, settings.gammaReg);
          } else {
            cdfValue = evaluateSplineAt(edited, year, settings.isplineReg);
          }
        } else {
          // Use the original empirical CDF
          const originalData = cdfData.data[milestone];
          const point = originalData.find(p => Math.abs(p.year - year) < 0.0001);
          cdfValue = point?.cdf ?? 0;
        }

        values.push(cdfValue.toFixed(10));
      }

      rows.push(values.join(','));
    }

    const csv = rows.join('\n');

    // Trigger download
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `atc_${selectedSimulationId}_cdf.csv`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }, [cdfData, editedMilestones, milestoneSettings, selectedSimulationId]);

  // Export all takeoff milestones to CSV
  const handleTakeoffExport = useCallback(() => {
    if (!takeoffCdfData) return;

    // Use the original time points from the data
    const timePoints = takeoffCdfData.timePoints;

    // Build CSV rows
    const headers = ['time_decimal_year', ...takeoffCdfData.milestones];
    const rows: string[] = [headers.join(',')];

    for (const time of timePoints) {
      const values: string[] = [time.toFixed(6)];

      for (const milestone of takeoffCdfData.milestones) {
        const edited = editedTakeoffMilestones.get(milestone);
        let cdfValue: number;

        if (edited && edited.length >= 2) {
          // Use per-milestone settings for evaluation
          const settings = takeoffSettings.get(milestone) ?? defaultSettings;
          if (settings.method === 'gamma') {
            cdfValue = evaluateGammaMixtureAt(edited, time, settings.gammaReg);
          } else {
            cdfValue = evaluateSplineAt(edited, time, settings.isplineReg);
          }
        } else {
          // Use the original empirical CDF
          const originalData = takeoffCdfData.data[milestone];
          const point = originalData.find(p => Math.abs(p.timeFromAC - time) < 0.0001);
          cdfValue = point?.cdf ?? 0;
        }

        values.push(cdfValue.toFixed(10));
      }

      rows.push(values.join(','));
    }

    const csv = rows.join('\n');

    // Trigger download
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `atc_takeoff_${selectedSimulationId}_cdf.csv`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }, [takeoffCdfData, editedTakeoffMilestones, takeoffSettings, selectedSimulationId]);

  // Get empirical CDF for the current milestone
  const empiricalCdf = useMemo(() => {
    if (!cdfData || !selectedMilestone) return [];
    const data = cdfData.data[selectedMilestone];
    return data?.map(p => ({ year: p.year, cdf: p.cdf })) ?? [];
  }, [cdfData, selectedMilestone]);

  // --- AC-anchor-mode breakout of the base (model) forecast ---
  // Only mixture simulations ship breakout CSVs; older sims get null and the
  // control is hidden. Denominator semantics match /forecast: each mode's CDF
  // plateaus at that mode's OWN achievement rate (failed rollouts excluded).
  const modeBreakout = cdfData?.modeBreakout ?? null;
  const hasModeBreakout = !!modeBreakout && modeBreakout.modes.length > 0;
  const breakoutActive = showModeBreakout && hasModeBreakout;

  // Per-mode CDF series for the selected milestone
  const breakoutCdfSeries = useMemo(() => {
    if (!breakoutActive || !modeBreakout || !selectedMilestone) return [];
    const result: { mode: string; modeIndex: number; points: { year: number; cdf: number }[] }[] = [];
    modeBreakout.modes.forEach((mode, modeIndex) => {
      const series = modeBreakout.perMode[mode]?.data[selectedMilestone];
      if (series && series.length >= 2) {
        result.push({ mode, modeIndex, points: series });
      }
    });
    return result;
  }, [breakoutActive, modeBreakout, selectedMilestone]);

  // Per-mode PDFs derived from the mode CDFs. Each integrates to that mode's
  // own achievement rate, mirroring how the "Original" PDF is derived from the
  // combined empirical CDF.
  const breakoutPdfSeries = useMemo(() => {
    return breakoutCdfSeries.map(({ mode, modeIndex, points }) => ({
      mode,
      modeIndex,
      samples: computePdfFromCdf(points.map(p => ({ year: p.year, cdf: p.cdf }))),
    }));
  }, [breakoutCdfSeries]);

  // Compute auto X domain for timeline charts (based on control points, not empirical CDF)
  // This ensures the visible x-axis matches the spline extent
  const timelineAutoDomain = useMemo<[number, number]>(() => {
    const config = CHART_DOMAIN_CONFIG.timeline;
    if (currentControlPoints.length < 2) return [config.defaultMin, config.defaultMax];
    const years = currentControlPoints.map(p => p.x);
    const min = Math.min(...years);
    const max = Math.max(...years);
    const padding = (max - min) * config.padding;
    return [min - padding, max + padding];
  }, [currentControlPoints]);

  // Effective X domain (zoom overrides auto)
  const xDomain = timelineZoomDomain ?? timelineAutoDomain;

  // Compute X-axis ticks (same logic as EditableCdfChart for consistency)
  const xTicks = useMemo(() => {
    const [min, max] = xDomain;
    const ticks: number[] = [];
    const startYear = Math.ceil(min);
    const step = Math.ceil((max - min) / 6);
    for (let year = startYear; year <= max; year += step) {
      ticks.push(year);
    }
    return ticks;
  }, [xDomain]);

  // Compute original PDF from empirical CDF
  const originalPdfSamples = useMemo(() => {
    if (empiricalCdf.length < 2) return [];
    return computePdfFromCdf(empiricalCdf);
  }, [empiricalCdf]);

  // Compute edited PDF from control points (using current milestone's settings)
  const editedPdfSamples = useMemo(() => {
    if (currentControlPoints.length < 2) return [];
    return currentSettings.method === 'gamma'
      ? sampleGammaMixturePdf(currentControlPoints, 300, currentSettings.gammaReg)
      : samplePdf(currentControlPoints, 300, currentSettings.isplineReg);
  }, [currentControlPoints, currentSettings]);

  // Merge both PDFs into a single data array for the chart
  const pdfData = useMemo<DataPoint[]>(() => {
    if (editedPdfSamples.length === 0 && originalPdfSamples.length === 0) return [];

    // Create a map of x values to data points
    const dataMap = new Map<number, DataPoint>();

    // Add original PDF data
    for (const sample of originalPdfSamples) {
      const rounded = Math.round(sample.x * 10000) / 10000; // Avoid floating point issues
      dataMap.set(rounded, { x: sample.x, original: sample.pdf });
    }

    // Add edited PDF data
    for (const sample of editedPdfSamples) {
      const rounded = Math.round(sample.x * 10000) / 10000;
      const existing = dataMap.get(rounded);
      if (existing) {
        existing.edited = sample.pdf;
      } else {
        dataMap.set(rounded, { x: sample.x, edited: sample.pdf });
      }
    }

    // Add per-AC-anchor-mode breakout PDF data (empty unless breakout is active)
    for (const { mode, samples } of breakoutPdfSeries) {
      const key = `mode_pdf_${mode}`;
      for (const sample of samples) {
        const rounded = Math.round(sample.x * 10000) / 10000;
        const existing = dataMap.get(rounded);
        if (existing) {
          existing[key] = sample.pdf;
        } else {
          dataMap.set(rounded, { x: sample.x, [key]: sample.pdf });
        }
      }
    }

    // Convert to array and sort by x
    return Array.from(dataMap.values()).sort((a, b) => a.x - b.x);
  }, [originalPdfSamples, editedPdfSamples, breakoutPdfSeries]);

  // Compute Y domain for PDF chart (auto-scale based on all rendered PDFs)
  const pdfYDomain = useMemo<[number, number]>(() => {
    if (pdfData.length === 0) return [0, 1];
    const maxOriginal = Math.max(...pdfData.map(d => (d.original as number) ?? 0));
    const maxEdited = Math.max(...pdfData.map(d => (d.edited as number) ?? 0));
    let maxPdf = Math.max(maxOriginal, maxEdited);
    for (const { mode } of breakoutPdfSeries) {
      const key = `mode_pdf_${mode}`;
      for (const d of pdfData) {
        const v = d[key] as number | undefined;
        if (v != null && v > maxPdf) maxPdf = v;
      }
    }
    return [0, maxPdf * 1.1 || 1];
  }, [pdfData, breakoutPdfSeries]);

  // Compute median for timeline CDF (x where CDF = 0.5)
  const timelineMedianYear = useMemo(() => {
    if (currentControlPoints.length < 2) return null;
    const sorted = [...currentControlPoints].sort((a, b) => a.x - b.x);
    const minX = sorted[0].x;
    const maxX = sorted[sorted.length - 1].x;
    const evaluate = currentSettings.method === 'gamma'
      ? (x: number) => evaluateGammaMixtureAt(sorted, x, currentSettings.gammaReg)
      : (x: number) => evaluateSplineAt(sorted, x, currentSettings.isplineReg);
    const minY = evaluate(minX);
    const maxY = evaluate(maxX);
    if (0.5 < minY || 0.5 > maxY) return null;
    let low = minX, high = maxX;
    for (let i = 0; i < 50; i++) {
      const mid = (low + high) / 2;
      if (Math.abs(evaluate(mid) - 0.5) < 0.001) return mid;
      if (evaluate(mid) < 0.5) low = mid; else high = mid;
    }
    return (low + high) / 2;
  }, [currentControlPoints, currentSettings]);

  // Get milestone color
  const milestoneColor = MILESTONE_COLORS[selectedMilestone] ?? '#2563eb';

  // Overlay all milestones PDF data (timeline)
  const timelineOverlay = useMemo(() => {
    if (!showAllTimelines || !cdfData) return null;

    const milestones = cdfData.milestones;
    // Compute PDF for each milestone
    const allPdfs: { milestone: string; samples: { x: number; pdf: number }[] }[] = [];

    for (const milestone of milestones) {
      const edited = milestone === selectedMilestone
        ? currentControlPoints
        : editedMilestones.get(milestone);
      const settings = milestone === selectedMilestone
        ? currentSettings
        : (milestoneSettings.get(milestone) ?? defaultSettings);

      if (edited && edited.length >= 2) {
        const samples = settings.method === 'gamma'
          ? sampleGammaMixturePdf(edited, 300, settings.gammaReg)
          : samplePdf(edited, 300, settings.isplineReg);
        allPdfs.push({ milestone, samples });
      } else {
        const empirical = cdfData.data[milestone];
        if (empirical && empirical.length >= 2) {
          const cdfPoints = empirical.map(p => ({ year: p.year, cdf: p.cdf }));
          const samples = computePdfFromCdf(cdfPoints);
          allPdfs.push({ milestone, samples });
        }
      }
    }

    // Merge into single DataPoint[] with per-milestone keys
    const dataMap = new Map<number, DataPoint>();
    for (const { milestone, samples } of allPdfs) {
      const key = `pdf_${milestone}`;
      for (const s of samples) {
        const rounded = Math.round(s.x * 10000) / 10000;
        const existing = dataMap.get(rounded);
        if (existing) {
          existing[key] = s.pdf;
        } else {
          dataMap.set(rounded, { x: s.x, [key]: s.pdf });
        }
      }
    }

    const data = Array.from(dataMap.values()).sort((a, b) => a.x - b.x);

    // Build lines config
    const lines = milestones.map(m => ({
      dataKey: `pdf_${m}`,
      stroke: MILESTONE_COLORS[m] ?? '#2563eb',
      strokeWidth: m === selectedMilestone ? 2.5 : 1.5,
      strokeOpacity: m === selectedMilestone ? 1 : 0.5,
      strokeDasharray: m === selectedMilestone ? undefined : '4 2',
      smooth: true,
      name: m,
    }));

    // Y domain
    let maxPdf = 0;
    for (const d of data) {
      for (const m of milestones) {
        const v = d[`pdf_${m}`] as number | undefined;
        if (v != null && v > maxPdf) maxPdf = v;
      }
    }
    const yDomain: [number, number] = [0, maxPdf * 1.1 || 1];

    return { data, lines, yDomain };
  }, [showAllTimelines, cdfData, selectedMilestone, currentControlPoints, editedMilestones, currentSettings, milestoneSettings]);

  // Overlay CDF lines for the editable CDF chart (timeline)
  const timelineCdfOverlayLines = useMemo<CdfOverlayLine[] | undefined>(() => {
    if (!showAllTimelines || !cdfData) return undefined;

    const result: CdfOverlayLine[] = [];
    for (const milestone of cdfData.milestones) {
      if (milestone === selectedMilestone) continue;

      const edited = editedMilestones.get(milestone);
      const settings = milestoneSettings.get(milestone) ?? defaultSettings;
      const color = MILESTONE_COLORS[milestone] ?? '#2563eb';

      if (edited && edited.length >= 2) {
        // Sample the edited CDF at many points
        const sorted = [...edited].sort((a, b) => a.x - b.x);
        const minX = sorted[0].x;
        const maxX = sorted[sorted.length - 1].x;
        const points: Array<{ x: number; y: number }> = [];
        const numSamples = 200;
        for (let i = 0; i <= numSamples; i++) {
          const x = minX + (maxX - minX) * (i / numSamples);
          const y = settings.method === 'gamma'
            ? evaluateGammaMixtureAt(sorted, x, settings.gammaReg)
            : evaluateSplineAt(sorted, x, settings.isplineReg);
          points.push({ x, y });
        }
        result.push({ points, color, name: milestone });
      } else {
        // Use empirical CDF directly
        const empirical = cdfData.data[milestone];
        if (empirical && empirical.length >= 2) {
          result.push({
            points: empirical.map(p => ({ x: p.year, y: p.cdf })),
            color,
            name: milestone,
          });
        }
      }
    }
    return result;
  }, [showAllTimelines, cdfData, selectedMilestone, editedMilestones, milestoneSettings]);

  // Per-AC-anchor-mode breakout overlay lines for the editable CDF chart.
  // Same encoding as /forecast: milestone color kept, dash pattern encodes mode.
  const breakoutCdfOverlayLines = useMemo<CdfOverlayLine[]>(() => {
    return breakoutCdfSeries.map(({ mode, modeIndex, points }) => ({
      points: points.map(p => ({ x: p.year, y: p.cdf })),
      color: milestoneColor,
      name: ANCHOR_MODE_DISPLAY_NAMES[mode] || mode,
      strokeDasharray: ANCHOR_MODE_DASH_PATTERNS[modeIndex % ANCHOR_MODE_DASH_PATTERNS.length],
      strokeOpacity: 0.85,
      strokeWidth: 1.5,
    }));
  }, [breakoutCdfSeries, milestoneColor]);

  // Combined overlay lines: "show all milestones" overlays plus breakout curves
  const combinedCdfOverlayLines = useMemo<CdfOverlayLine[] | undefined>(() => {
    if (breakoutCdfOverlayLines.length === 0) return timelineCdfOverlayLines;
    return [...(timelineCdfOverlayLines ?? []), ...breakoutCdfOverlayLines];
  }, [timelineCdfOverlayLines, breakoutCdfOverlayLines]);

  // Per-AC-anchor-mode breakout line configs for the PDF chart
  const breakoutPdfLines = useMemo(() => {
    return breakoutPdfSeries.map(({ mode, modeIndex }) => ({
      dataKey: `mode_pdf_${mode}`,
      stroke: milestoneColor,
      strokeWidth: 1.5,
      strokeOpacity: 0.85,
      strokeDasharray: ANCHOR_MODE_DASH_PATTERNS[modeIndex % ANCHOR_MODE_DASH_PATTERNS.length],
      smooth: true,
      name: ANCHOR_MODE_DISPLAY_NAMES[mode] || mode,
    }));
  }, [breakoutPdfSeries, milestoneColor]);

  // Get empirical CDF for the current takeoff milestone (converting to year-based format for the chart)
  const takeoffEmpiricalCdf = useMemo(() => {
    if (!takeoffCdfData || !selectedTakeoffMilestone) return [];
    const data = takeoffCdfData.data[selectedTakeoffMilestone];
    // Map timeFromAC to year field for compatibility with EditableCdfChart
    return data?.map(p => ({ year: p.timeFromAC, cdf: p.cdf })) ?? [];
  }, [takeoffCdfData, selectedTakeoffMilestone]);

  // Compute auto X domain for takeoff charts (based on control points, not empirical CDF)
  // This ensures the visible x-axis matches the spline extent
  const takeoffAutoDomain = useMemo<[number, number]>(() => {
    const config = CHART_DOMAIN_CONFIG.takeoff;
    if (currentTakeoffControlPoints.length < 2) return [config.defaultMin, config.defaultMax];
    const times = currentTakeoffControlPoints.map(p => p.x);
    const min = Math.min(...times);
    const max = Math.max(...times);
    const padding = (max - min) * config.padding;
    return [Math.max(config.absoluteMin, min - padding), max + padding];
  }, [currentTakeoffControlPoints]);

  // Effective X domain for takeoff (zoom overrides auto)
  const takeoffXDomain = takeoffZoomDomain ?? takeoffAutoDomain;

  // Compute X-axis ticks for takeoff (same logic as EditableCdfChart for consistency)
  const takeoffXTicks = useMemo(() => {
    const [min, max] = takeoffXDomain;
    const ticks: number[] = [];
    // For relative years, use nice round numbers starting from 0
    const step = Math.ceil((max - min) / 6);
    const niceStep = step <= 1 ? 1 : step <= 2 ? 2 : step <= 5 ? 5 : Math.ceil(step / 5) * 5;
    for (let t = 0; t <= max + 0.001; t += niceStep) {
      ticks.push(t);
    }
    return ticks;
  }, [takeoffXDomain]);

  // Compute original PDF from takeoff empirical CDF
  const takeoffOriginalPdfSamples = useMemo(() => {
    if (takeoffEmpiricalCdf.length < 2) return [];
    return computePdfFromCdf(takeoffEmpiricalCdf);
  }, [takeoffEmpiricalCdf]);

  // Compute edited PDF from takeoff control points (using current takeoff milestone's settings)
  const takeoffEditedPdfSamples = useMemo(() => {
    if (currentTakeoffControlPoints.length < 2) return [];
    return currentTakeoffSettings.method === 'gamma'
      ? sampleGammaMixturePdf(currentTakeoffControlPoints, 300, currentTakeoffSettings.gammaReg)
      : samplePdf(currentTakeoffControlPoints, 300, currentTakeoffSettings.isplineReg);
  }, [currentTakeoffControlPoints, currentTakeoffSettings]);

  // Merge both PDFs into a single data array for the takeoff chart
  const takeoffPdfData = useMemo<DataPoint[]>(() => {
    if (takeoffEditedPdfSamples.length === 0 && takeoffOriginalPdfSamples.length === 0) return [];

    // Create a map of x values to data points
    const dataMap = new Map<number, DataPoint>();

    // Add original PDF data
    for (const sample of takeoffOriginalPdfSamples) {
      const rounded = Math.round(sample.x * 10000) / 10000;
      dataMap.set(rounded, { x: sample.x, original: sample.pdf });
    }

    // Add edited PDF data
    for (const sample of takeoffEditedPdfSamples) {
      const rounded = Math.round(sample.x * 10000) / 10000;
      const existing = dataMap.get(rounded);
      if (existing) {
        existing.edited = sample.pdf;
      } else {
        dataMap.set(rounded, { x: sample.x, edited: sample.pdf });
      }
    }

    // Convert to array and sort by x
    return Array.from(dataMap.values()).sort((a, b) => a.x - b.x);
  }, [takeoffOriginalPdfSamples, takeoffEditedPdfSamples]);

  // Compute Y domain for takeoff PDF chart
  const takeoffPdfYDomain = useMemo<[number, number]>(() => {
    if (takeoffPdfData.length === 0) return [0, 1];
    const maxOriginal = Math.max(...takeoffPdfData.map(d => (d.original as number) ?? 0));
    const maxEdited = Math.max(...takeoffPdfData.map(d => (d.edited as number) ?? 0));
    const maxPdf = Math.max(maxOriginal, maxEdited);
    return [0, maxPdf * 1.1 || 1];
  }, [takeoffPdfData]);

  // Compute median for takeoff CDF (x where CDF = 0.5)
  const takeoffMedianYears = useMemo(() => {
    if (currentTakeoffControlPoints.length < 2) return null;
    const sorted = [...currentTakeoffControlPoints].sort((a, b) => a.x - b.x);
    const minX = sorted[0].x;
    const maxX = sorted[sorted.length - 1].x;
    const evaluate = currentTakeoffSettings.method === 'gamma'
      ? (x: number) => evaluateGammaMixtureAt(sorted, x, currentTakeoffSettings.gammaReg)
      : (x: number) => evaluateSplineAt(sorted, x, currentTakeoffSettings.isplineReg);
    const minY = evaluate(minX);
    const maxY = evaluate(maxX);
    if (0.5 < minY || 0.5 > maxY) return null;
    let low = minX, high = maxX;
    for (let i = 0; i < 50; i++) {
      const mid = (low + high) / 2;
      if (Math.abs(evaluate(mid) - 0.5) < 0.001) return mid;
      if (evaluate(mid) < 0.5) low = mid; else high = mid;
    }
    return (low + high) / 2;
  }, [currentTakeoffControlPoints, currentTakeoffSettings]);

  // Get takeoff milestone color
  const takeoffMilestoneColor = MILESTONE_COLORS[selectedTakeoffMilestone] ?? '#2563eb';

  // Overlay all milestones PDF data (takeoff)
  const takeoffOverlay = useMemo(() => {
    if (!showAllTakeoffs || !takeoffCdfData) return null;

    const milestones = takeoffCdfData.milestones;
    const allPdfs: { milestone: string; samples: { x: number; pdf: number }[] }[] = [];

    for (const milestone of milestones) {
      const edited = milestone === selectedTakeoffMilestone
        ? currentTakeoffControlPoints
        : editedTakeoffMilestones.get(milestone);
      const settings = milestone === selectedTakeoffMilestone
        ? currentTakeoffSettings
        : (takeoffSettings.get(milestone) ?? defaultSettings);

      if (edited && edited.length >= 2) {
        const samples = settings.method === 'gamma'
          ? sampleGammaMixturePdf(edited, 300, settings.gammaReg)
          : samplePdf(edited, 300, settings.isplineReg);
        allPdfs.push({ milestone, samples });
      } else {
        const empirical = takeoffCdfData.data[milestone];
        if (empirical && empirical.length >= 2) {
          const cdfPoints = empirical.map(p => ({ year: p.timeFromAC, cdf: p.cdf }));
          const samples = computePdfFromCdf(cdfPoints);
          allPdfs.push({ milestone, samples });
        }
      }
    }

    const dataMap = new Map<number, DataPoint>();
    for (const { milestone, samples } of allPdfs) {
      const key = `pdf_${milestone}`;
      for (const s of samples) {
        const rounded = Math.round(s.x * 10000) / 10000;
        const existing = dataMap.get(rounded);
        if (existing) {
          existing[key] = s.pdf;
        } else {
          dataMap.set(rounded, { x: s.x, [key]: s.pdf });
        }
      }
    }

    const data = Array.from(dataMap.values()).sort((a, b) => a.x - b.x);

    const lines = milestones.map(m => ({
      dataKey: `pdf_${m}`,
      stroke: MILESTONE_COLORS[m] ?? '#2563eb',
      strokeWidth: m === selectedTakeoffMilestone ? 2.5 : 1.5,
      strokeOpacity: m === selectedTakeoffMilestone ? 1 : 0.5,
      strokeDasharray: m === selectedTakeoffMilestone ? undefined : '4 2',
      smooth: true,
      name: m,
    }));

    let maxPdf = 0;
    for (const d of data) {
      for (const m of milestones) {
        const v = d[`pdf_${m}`] as number | undefined;
        if (v != null && v > maxPdf) maxPdf = v;
      }
    }
    const yDomain: [number, number] = [0, maxPdf * 1.1 || 1];

    return { data, lines, yDomain };
  }, [showAllTakeoffs, takeoffCdfData, selectedTakeoffMilestone, currentTakeoffControlPoints, editedTakeoffMilestones, currentTakeoffSettings, takeoffSettings]);

  // Overlay CDF lines for the editable CDF chart (takeoff)
  const takeoffCdfOverlayLines = useMemo<CdfOverlayLine[] | undefined>(() => {
    if (!showAllTakeoffs || !takeoffCdfData) return undefined;

    const result: CdfOverlayLine[] = [];
    for (const milestone of takeoffCdfData.milestones) {
      if (milestone === selectedTakeoffMilestone) continue;

      const edited = editedTakeoffMilestones.get(milestone);
      const settings = takeoffSettings.get(milestone) ?? defaultSettings;
      const color = MILESTONE_COLORS[milestone] ?? '#2563eb';

      if (edited && edited.length >= 2) {
        const sorted = [...edited].sort((a, b) => a.x - b.x);
        const minX = sorted[0].x;
        const maxX = sorted[sorted.length - 1].x;
        const points: Array<{ x: number; y: number }> = [];
        const numSamples = 200;
        for (let i = 0; i <= numSamples; i++) {
          const x = minX + (maxX - minX) * (i / numSamples);
          const y = settings.method === 'gamma'
            ? evaluateGammaMixtureAt(sorted, x, settings.gammaReg)
            : evaluateSplineAt(sorted, x, settings.isplineReg);
          points.push({ x, y });
        }
        result.push({ points, color, name: milestone });
      } else {
        const empirical = takeoffCdfData.data[milestone];
        if (empirical && empirical.length >= 2) {
          result.push({
            points: empirical.map(p => ({ x: p.timeFromAC, y: p.cdf })),
            color,
            name: milestone,
          });
        }
      }
    }
    return result;
  }, [showAllTakeoffs, takeoffCdfData, selectedTakeoffMilestone, editedTakeoffMilestones, takeoffSettings]);

  return (
    <div className="grid h-screen w-full grid-cols-[minmax(0,1fr)_auto] grid-rows-[minmax(0,1fr)] gap-0">
      <div className="relative col-start-1 row-start-1 flex min-h-0 flex-col">
        <div className="flex min-h-0 flex-col overflow-y-auto px-6 pb-10">
          <HeaderContent variant="inline" className="pt-6 pb-4" />
          <main className="mt-10 mx-auto max-w-7xl w-full px-6 pb-16">
            <h2 className="text-2xl font-semibold text-primary font-et-book mb-6">
              ATC Distribution Editor
            </h2>

            {/* Selectors */}
            <div className="flex flex-wrap gap-4 mb-6">
              {/* Forecast Selector */}
              <div className="flex items-center gap-2">
                <label htmlFor="forecast-selector" className="text-sm font-medium text-primary/70">
                  Forecast:
                </label>
                <select
                  id="forecast-selector"
                  value={selectedSimulationId}
                  onChange={(e) => handleSimulationChange(e.target.value)}
                  className="block rounded-md border-gray-300 bg-white py-1.5 pl-3 pr-10 text-sm shadow-sm focus:border-accent focus:outline-none focus:ring-1 focus:ring-accent"
                >
                  {sortedSimulations.map((sim) => (
                    <option key={sim.id} value={sim.id}>
                      {sim.label}
                    </option>
                  ))}
                </select>
              </div>

              {/* Milestone Selector */}
              {cdfData && (
                <AtcMilestoneSelector
                  milestones={cdfData.milestones}
                  selectedMilestone={selectedMilestone}
                  editedMilestones={editedMilestones}
                  onMilestoneChange={handleMilestoneChange}
                />
              )}
            </div>

            {/* Interpolation Method Toggle - per milestone */}
            <div className="flex flex-wrap items-center gap-4 mb-6 p-3 bg-gray-50 rounded-lg">
              <span className="text-sm font-medium text-primary/70">Interpolation method{selectedMilestone ? ` (${selectedMilestone})` : ''}:</span>
              <div className="flex gap-1 bg-white rounded-md shadow-sm border border-gray-200">
                <button
                  onClick={() => updateCurrentSettings({ method: 'ispline' })}
                  className={`px-3 py-1.5 text-sm font-medium rounded-l-md transition-colors ${
                    currentSettings.method === 'ispline'
                      ? 'bg-accent text-white'
                      : 'text-gray-600 hover:bg-gray-100'
                  }`}
                >
                  I-Spline
                </button>
                <button
                  onClick={() => updateCurrentSettings({ method: 'gamma' })}
                  className={`px-3 py-1.5 text-sm font-medium rounded-r-md transition-colors ${
                    currentSettings.method === 'gamma'
                      ? 'bg-accent text-white'
                      : 'text-gray-600 hover:bg-gray-100'
                  }`}
                >
                  Gamma Mixture
                </button>
              </div>
              <span className="text-xs text-primary/50">
                {currentSettings.method === 'ispline'
                  ? '(Smooth C² curve, approximately fits control points)'
                  : '(Weighted sum of gamma distributions, precisely fits control points)'}
              </span>
              {/* Regularization slider - shows different slider based on method */}
              {currentSettings.method === 'gamma' ? (
                <div className="flex items-center gap-3 mt-2">
                  <label className="text-xs text-primary/70 whitespace-nowrap">Gamma Reg:</label>
                  <input
                    type="range"
                    min="0"
                    max="0.5"
                    step="0.01"
                    value={currentSettings.gammaReg}
                    onChange={(e) => updateCurrentSettings({ gammaReg: parseFloat(e.target.value) })}
                    className="w-32 h-1.5 accent-accent"
                  />
                  <span className="text-xs text-primary/50 w-10">{currentSettings.gammaReg.toFixed(2)}</span>
                  <span className="text-xs text-primary/40">(0 = exact fit, higher = smoother PDF)</span>
                </div>
              ) : (
                <div className="flex items-center gap-3 mt-2">
                  <label className="text-xs text-primary/70 whitespace-nowrap">I-Spline Reg:</label>
                  <input
                    type="range"
                    min="0"
                    max="0.5"
                    step="0.01"
                    value={currentSettings.isplineReg}
                    onChange={(e) => updateCurrentSettings({ isplineReg: parseFloat(e.target.value) })}
                    className="w-32 h-1.5 accent-accent"
                  />
                  <span className="text-xs text-primary/50 w-10">{currentSettings.isplineReg.toFixed(2)}</span>
                  <span className="text-xs text-primary/40">(0 = exact fit, higher = smoother PDF)</span>
                </div>
              )}
            </div>

            {/* Loading/Error states */}
            {isLoading && (
              <div className="text-primary/50 py-8">Loading CDF data...</div>
            )}
            {error && (
              <div className="text-red-600 py-8">Error: {error}</div>
            )}

        {/* Chart and Control Points Table */}
        {cdfData && selectedMilestone && !isLoading && (
          <ChartSyncProvider>
            {/* Zoom Controls */}
            <div className="mb-4 max-w-md">
              <ChartZoomControls
                absoluteMin={CHART_DOMAIN_CONFIG.timeline.absoluteMin}
                absoluteMax={CHART_DOMAIN_CONFIG.timeline.absoluteMax}
                value={xDomain}
                onChange={setTimelineZoomDomain}
                onReset={() => setTimelineZoomDomain(null)}
                label="X-Axis Range"
                formatValue={(v) => v.toFixed(0)}
              />
            </div>
            {/* AC-anchor breakout toggle - only for mixture sims with breakout data.
                Arrival section only: takeoff data is not broken out by design. */}
            {hasModeBreakout && (
              <label className="flex items-center gap-2 mb-4 text-xs text-primary/60 cursor-pointer select-none">
                <input
                  type="checkbox"
                  checked={showModeBreakout}
                  onChange={(e) => setShowModeBreakout(e.target.checked)}
                  className="accent-accent"
                />
                <span>
                  Break out model forecast by AC-anchor method
                  <span className="text-primary/40"> (each curve is normalized within its own subset of rollouts)</span>
                </span>
              </label>
            )}
            <div className="flex gap-4">
              {/* Chart */}
              <div className="flex-1 min-w-0">
                <EditableCdfChart
                  empiricalCdf={empiricalCdf}
                  controlPoints={currentControlPoints}
                  onControlPointsChange={handleControlPointsChange}
                  milestoneName={selectedMilestone}
                  interpolationMethod={currentSettings.method}
                  gammaRegularization={currentSettings.gammaReg}
                  isplineRegularization={currentSettings.isplineReg}
                  xDomain={xDomain}
                  overlayLines={combinedCdfOverlayLines}
                />
              </div>
              {/* Control Points Table */}
              <div className="w-72 flex-shrink-0">
                <ControlPointsTable
                  controlPoints={currentControlPoints}
                  onControlPointsChange={handleControlPointsChange}
                  xAxisLabel="Year"
                  xMin={CHART_DOMAIN_CONFIG.timeline.absoluteMin}
                  xMax={CHART_DOMAIN_CONFIG.timeline.absoluteMax}
                  interpolationMethod={currentSettings.method}
                  gammaRegularization={currentSettings.gammaReg}
                  isplineRegularization={currentSettings.isplineReg}
                  empiricalCdf={empiricalCdf}
                />
              </div>
            </div>

            {/* PDF Chart - wrapped in flex to match CDF chart width */}
            {pdfData.length > 0 && (
              <div className="flex gap-4 mt-4">
                <div className="flex-1 min-w-0 bg-white rounded-lg shadow p-4">
                  <div className="flex items-center justify-between mb-2">
                    <h3 className="text-sm font-medium text-primary/70">Implied PDF (Probability Density)</h3>
                    <label className="flex items-center gap-1.5 text-xs text-primary/60 cursor-pointer select-none">
                      <input
                        type="checkbox"
                        checked={showAllTimelines}
                        onChange={(e) => setShowAllTimelines(e.target.checked)}
                        className="accent-accent"
                      />
                      Show all milestones
                    </label>
                  </div>
                  {showAllTimelines && timelineOverlay ? (
                    <>
                      <PdfLineChart
                        data={timelineOverlay.data}
                        height={200}
                        margin={{ top: 20, right: 40, bottom: 60, left: 60 }}
                        xDomain={xDomain}
                        xTicks={xTicks}
                        yDomain={timelineOverlay.yDomain}
                        xTickFormatter={(v) => v.toFixed(0)}
                        xLabel="Year"
                        lines={timelineOverlay.lines}
                      />
                      <div className="flex flex-wrap items-center gap-3 mt-2">
                        {timelineOverlay.lines.map(line => (
                          <div key={line.dataKey} className="flex items-center gap-1.5">
                            <div
                              className="w-4 h-0.5"
                              style={{
                                backgroundColor: line.stroke,
                                opacity: line.strokeOpacity,
                                borderTop: line.strokeDasharray ? `2px dashed ${line.stroke}` : undefined,
                                height: line.strokeDasharray ? 0 : 2,
                              }}
                            />
                            <span
                              className="text-xs"
                              style={{
                                color: line.stroke,
                                fontWeight: line.name === selectedMilestone ? 600 : 400,
                                opacity: line.name === selectedMilestone ? 1 : 0.7,
                              }}
                            >
                              {line.name}
                            </span>
                          </div>
                        ))}
                      </div>
                    </>
                  ) : (
                    <>
                      <PdfLineChart
                        data={pdfData}
                        height={200}
                        margin={{ top: 20, right: 40, bottom: 60, left: 60 }}
                        xDomain={xDomain}
                        xTicks={xTicks}
                        yDomain={pdfYDomain}
                        xTickFormatter={(v) => v.toFixed(0)}
                        xLabel="Year"
                        verticalReferenceLine={timelineMedianYear !== null ? {
                          x: timelineMedianYear,
                          stroke: '#f97316',
                          strokeDasharray: '6,4',
                          strokeWidth: 2,
                          strokeOpacity: 0.8,
                          label: `Median: ${formatDecimalYearAsDate(timelineMedianYear)}`,
                        } : undefined}
                        lines={[
                          {
                            dataKey: 'original',
                            stroke: '#9ca3af',
                            strokeWidth: 1.5,
                            strokeOpacity: 0.7,
                            smooth: true,
                            name: 'Original',
                          },
                          ...breakoutPdfLines,
                          {
                            dataKey: 'edited',
                            stroke: milestoneColor,
                            strokeWidth: 2,
                            smooth: true,
                            name: 'Edited',
                          },
                        ]}
                        tooltip={(point) => {
                          const getEmpiricalCdfAt = (x: number): number => {
                            if (empiricalCdf.length === 0) return 0;
                            const sorted = [...empiricalCdf].sort((a, b) => a.year - b.year);
                            if (x <= sorted[0].year) return sorted[0].cdf;
                            if (x >= sorted[sorted.length - 1].year) return sorted[sorted.length - 1].cdf;
                            for (let i = 0; i < sorted.length - 1; i++) {
                              if (x >= sorted[i].year && x <= sorted[i + 1].year) {
                                const t = (x - sorted[i].year) / (sorted[i + 1].year - sorted[i].year);
                                return sorted[i].cdf + t * (sorted[i + 1].cdf - sorted[i].cdf);
                              }
                            }
                            return 0;
                          };
                          const empiricalCdfValue = getEmpiricalCdfAt(point.x);
                          const editedCdfValue = currentSettings.method === 'gamma'
                            ? evaluateGammaMixtureAt(currentControlPoints, point.x, currentSettings.gammaReg)
                            : evaluateSplineAt(currentControlPoints, point.x, currentSettings.isplineReg);

                          return (
                            <div className="bg-white border border-gray-200 rounded px-2 py-1.5 shadow-md text-xs">
                              <div className="font-semibold text-primary">{point.x.toFixed(1)}</div>
                              <div className="text-primary/50">
                                Empirical: {(empiricalCdfValue * 100).toFixed(1)}%
                              </div>
                              <div style={{ color: milestoneColor }} className="font-medium">
                                ATC: {(editedCdfValue * 100).toFixed(1)}%
                              </div>
                            </div>
                          );
                        }}
                      />
                      <div className="flex items-center gap-4 mt-2">
                        <div className="flex items-center gap-1.5">
                          <div className="w-4 h-0.5 bg-gray-400 opacity-70"></div>
                          <span className="text-xs text-primary/50">Original (model forecast)</span>
                        </div>
                        <div className="flex items-center gap-1.5">
                          <div className="w-4 h-0.5" style={{ backgroundColor: milestoneColor }}></div>
                          <span className="text-xs text-primary/50">Edited</span>
                        </div>
                      </div>
                      {/* Per-AC-anchor-mode breakout legend (weight / n / achievement rate) */}
                      {breakoutActive && modeBreakout && (
                        <div className="flex flex-col gap-1 mt-2">
                          {breakoutCdfSeries.map(({ mode, modeIndex }) => {
                            const modeData = modeBreakout.perMode[mode];
                            if (!modeData) return null;
                            const modeStats = modeData.statistics[selectedMilestone];
                            const dash = ANCHOR_MODE_DASH_PATTERNS[modeIndex % ANCHOR_MODE_DASH_PATTERNS.length];
                            return (
                              <div key={`mode-${mode}`} className="flex items-center gap-1.5">
                                <svg width="32" height="4" className="shrink-0" aria-hidden="true">
                                  <line x1="0" y1="2" x2="32" y2="2" stroke={milestoneColor} strokeWidth="2" strokeDasharray={dash} opacity={0.85} />
                                </svg>
                                <span className="text-xs font-medium min-w-0" style={{ color: milestoneColor, opacity: 0.85 }}>
                                  {ANCHOR_MODE_DISPLAY_NAMES[mode] || mode}
                                  <span className="text-gray-500 font-normal">
                                    {` (mixture weight ${modeData.weightPct.toFixed(1)}%, n=${modeData.numRollouts.toLocaleString()}`}
                                    {modeData.numFailed > 0 ? ` of ${modeData.numSampled.toLocaleString()} sampled` : ''}
                                    {modeStats ? `; achieves in ${(modeStats.achievementRate * 100).toFixed(1)}% of runs` : ''}
                                    {modeStats ? `; p10: ${formatDecimalYearAsDate(modeStats.p10)}, p50: ${formatDecimalYearAsDate(modeStats.p50)}, p90: ${formatDecimalYearAsDate(modeStats.p90)}` : ''}
                                    {')'}
                                  </span>
                                </span>
                              </div>
                            );
                          })}
                        </div>
                      )}
                    </>
                  )}
                </div>
                {/* Spacer to match control points table width */}
                <div className="w-72 flex-shrink-0" />
              </div>
            )}

            {/* Median + Action buttons */}
            <div className="flex items-center gap-4 mt-6">
              {timelineMedianYear !== null && (
                <span className="text-sm font-medium text-orange-500">
                  Median: {formatDecimalYearAsDate(timelineMedianYear)}
                </span>
              )}
              <button
                onClick={handleReset}
                className="px-4 py-2 text-sm font-medium text-primary bg-white border border-gray-300 rounded-md hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-accent"
              >
                Reset to Empirical
              </button>
              <button
                onClick={handleExport}
                className="px-4 py-2 text-sm font-medium text-white bg-accent border border-transparent rounded-md hover:opacity-90 focus:outline-none focus:ring-2 focus:ring-accent"
              >
                Export ATC Forecast
              </button>
            </div>

            {/* Edited milestones indicator */}
            {editedMilestones.size > 0 && (
              <div className="mt-4 text-sm text-primary/50">
                Edited milestones: {Array.from(editedMilestones.keys()).join(', ')}
              </div>
            )}
          </ChartSyncProvider>
        )}

            {/* Takeoff Speeds Section */}
            <section className="mt-16 pt-8 border-t border-gray-200">
              <h3 className="text-xl font-semibold text-primary font-et-book mb-4">
                Takeoff Speeds (Time from AC)
              </h3>
              <p className="text-sm text-primary/70 mb-6">
                Edit distributions for how long it takes to reach milestones after achieving AC (Automated Coder).
              </p>

              {/* Takeoff Milestone Selector */}
              {takeoffCdfData && (
                <div className="mb-6">
                  <AtcMilestoneSelector
                    milestones={takeoffCdfData.milestones}
                    selectedMilestone={selectedTakeoffMilestone}
                    editedMilestones={editedTakeoffMilestones}
                    onMilestoneChange={handleTakeoffMilestoneChange}
                  />
                </div>
              )}

              {/* Takeoff Interpolation Method Toggle - per milestone */}
              <div className="flex flex-wrap items-center gap-4 mb-6 p-3 bg-gray-50 rounded-lg">
                <span className="text-sm font-medium text-primary/70">Interpolation method{selectedTakeoffMilestone ? ` (${selectedTakeoffMilestone})` : ''}:</span>
                <div className="flex gap-1 bg-white rounded-md shadow-sm border border-gray-200">
                  <button
                    onClick={() => updateCurrentTakeoffSettings({ method: 'ispline' })}
                    className={`px-3 py-1.5 text-sm font-medium rounded-l-md transition-colors ${
                      currentTakeoffSettings.method === 'ispline'
                        ? 'bg-accent text-white'
                        : 'text-gray-600 hover:bg-gray-100'
                    }`}
                  >
                    I-Spline
                  </button>
                  <button
                    onClick={() => updateCurrentTakeoffSettings({ method: 'gamma' })}
                    className={`px-3 py-1.5 text-sm font-medium rounded-r-md transition-colors ${
                      currentTakeoffSettings.method === 'gamma'
                        ? 'bg-accent text-white'
                        : 'text-gray-600 hover:bg-gray-100'
                    }`}
                  >
                    Gamma Mixture
                  </button>
                </div>
                {/* Regularization slider */}
                {currentTakeoffSettings.method === 'gamma' ? (
                  <div className="flex items-center gap-3 mt-2">
                    <label className="text-xs text-primary/70 whitespace-nowrap">Gamma Reg:</label>
                    <input
                      type="range"
                      min="0"
                      max="0.5"
                      step="0.01"
                      value={currentTakeoffSettings.gammaReg}
                      onChange={(e) => updateCurrentTakeoffSettings({ gammaReg: parseFloat(e.target.value) })}
                      className="w-32 h-1.5 accent-accent"
                    />
                    <span className="text-xs text-primary/50 w-10">{currentTakeoffSettings.gammaReg.toFixed(2)}</span>
                  </div>
                ) : (
                  <div className="flex items-center gap-3 mt-2">
                    <label className="text-xs text-primary/70 whitespace-nowrap">I-Spline Reg:</label>
                    <input
                      type="range"
                      min="0"
                      max="0.5"
                      step="0.01"
                      value={currentTakeoffSettings.isplineReg}
                      onChange={(e) => updateCurrentTakeoffSettings({ isplineReg: parseFloat(e.target.value) })}
                      className="w-32 h-1.5 accent-accent"
                    />
                    <span className="text-xs text-primary/50 w-10">{currentTakeoffSettings.isplineReg.toFixed(2)}</span>
                  </div>
                )}
              </div>

              {/* Loading/Error states */}
              {isTakeoffLoading && (
                <div className="text-primary/50 py-8">Loading takeoff CDF data...</div>
              )}
              {takeoffError && (
                <div className="text-red-600 py-8">Error: {takeoffError}</div>
              )}

              {/* Takeoff Chart and Control Points Table */}
              {takeoffCdfData && selectedTakeoffMilestone && !isTakeoffLoading && (
                <ChartSyncProvider>
                  {/* Zoom Controls */}
                  <div className="mb-4 max-w-md">
                    <ChartZoomControls
                      absoluteMin={CHART_DOMAIN_CONFIG.takeoff.absoluteMin}
                      absoluteMax={CHART_DOMAIN_CONFIG.takeoff.absoluteMax}
                      value={takeoffXDomain}
                      onChange={setTakeoffZoomDomain}
                      onReset={() => setTakeoffZoomDomain(null)}
                      label="X-Axis Range"
                      formatValue={(v) => v < 1 ? v.toFixed(1) : v.toFixed(0)}
                    />
                  </div>
                  <div className="flex gap-4">
                    {/* Chart */}
                    <div className="flex-1 min-w-0">
                      <EditableCdfChart
                        empiricalCdf={takeoffEmpiricalCdf}
                        controlPoints={currentTakeoffControlPoints}
                        onControlPointsChange={handleTakeoffControlPointsChange}
                        milestoneName={selectedTakeoffMilestone}
                        interpolationMethod={currentTakeoffSettings.method}
                        gammaRegularization={currentTakeoffSettings.gammaReg}
                        isplineRegularization={currentTakeoffSettings.isplineReg}
                        xAxisMode="relativeYears"
                        xAxisLabel="Years from AC"
                        xDomain={takeoffXDomain}
                        overlayLines={takeoffCdfOverlayLines}
                      />
                    </div>
                    {/* Control Points Table */}
                    <div className="w-72 flex-shrink-0">
                      <ControlPointsTable
                        controlPoints={currentTakeoffControlPoints}
                        onControlPointsChange={handleTakeoffControlPointsChange}
                        xAxisLabel="Years"
                        xMin={CHART_DOMAIN_CONFIG.takeoff.absoluteMin}
                        xMax={CHART_DOMAIN_CONFIG.takeoff.absoluteMax}
                        interpolationMethod={currentTakeoffSettings.method}
                        gammaRegularization={currentTakeoffSettings.gammaReg}
                        isplineRegularization={currentTakeoffSettings.isplineReg}
                        empiricalCdf={takeoffEmpiricalCdf}
                      />
                    </div>
                  </div>

                  {/* Takeoff PDF Chart - wrapped in flex to match CDF chart width */}
                  {takeoffPdfData.length > 0 && (
                    <div className="flex gap-4 mt-4">
                      <div className="flex-1 min-w-0 bg-white rounded-lg shadow p-4">
                        <div className="flex items-center justify-between mb-2">
                          <h3 className="text-sm font-medium text-primary/70">Implied PDF (Probability Density)</h3>
                          <label className="flex items-center gap-1.5 text-xs text-primary/60 cursor-pointer select-none">
                            <input
                              type="checkbox"
                              checked={showAllTakeoffs}
                              onChange={(e) => setShowAllTakeoffs(e.target.checked)}
                              className="accent-accent"
                            />
                            Show all milestones
                          </label>
                        </div>
                        {showAllTakeoffs && takeoffOverlay ? (
                          <>
                            <PdfLineChart
                              data={takeoffOverlay.data}
                              height={200}
                              margin={{ top: 20, right: 40, bottom: 60, left: 60 }}
                              xDomain={takeoffXDomain}
                              xTicks={takeoffXTicks}
                              yDomain={takeoffOverlay.yDomain}
                              xTickFormatter={(v) => v < 1 ? v.toFixed(1) : v.toFixed(0)}
                              xLabel="Years from AC"
                              lines={takeoffOverlay.lines}
                            />
                            <div className="flex flex-wrap items-center gap-3 mt-2">
                              {takeoffOverlay.lines.map(line => (
                                <div key={line.dataKey} className="flex items-center gap-1.5">
                                  <div
                                    className="w-4 h-0.5"
                                    style={{
                                      backgroundColor: line.stroke,
                                      opacity: line.strokeOpacity,
                                      borderTop: line.strokeDasharray ? `2px dashed ${line.stroke}` : undefined,
                                      height: line.strokeDasharray ? 0 : 2,
                                    }}
                                  />
                                  <span
                                    className="text-xs"
                                    style={{
                                      color: line.stroke,
                                      fontWeight: line.name === selectedTakeoffMilestone ? 600 : 400,
                                      opacity: line.name === selectedTakeoffMilestone ? 1 : 0.7,
                                    }}
                                  >
                                    {line.name}
                                  </span>
                                </div>
                              ))}
                            </div>
                          </>
                        ) : (
                          <>
                            <PdfLineChart
                              data={takeoffPdfData}
                              height={200}
                              margin={{ top: 20, right: 40, bottom: 60, left: 60 }}
                              xDomain={takeoffXDomain}
                              xTicks={takeoffXTicks}
                              yDomain={takeoffPdfYDomain}
                              xTickFormatter={(v) => v < 1 ? v.toFixed(1) : v.toFixed(0)}
                              xLabel="Years from AC"
                              verticalReferenceLine={takeoffMedianYears !== null ? {
                                x: takeoffMedianYears,
                                stroke: '#f97316',
                                strokeDasharray: '6,4',
                                strokeWidth: 2,
                                strokeOpacity: 0.8,
                                label: `Median: ${takeoffMedianYears.toFixed(2)} yr`,
                              } : undefined}
                              lines={[
                                {
                                  dataKey: 'original',
                                  stroke: '#9ca3af',
                                  strokeWidth: 1.5,
                                  strokeOpacity: 0.7,
                                  smooth: true,
                                  name: 'Original',
                                },
                                {
                                  dataKey: 'edited',
                                  stroke: takeoffMilestoneColor,
                                  strokeWidth: 2,
                                  smooth: true,
                                  name: 'Edited',
                                },
                              ]}
                              tooltip={(point) => {
                                const getEmpiricalCdfAt = (x: number): number => {
                                  if (takeoffEmpiricalCdf.length === 0) return 0;
                                  const sorted = [...takeoffEmpiricalCdf].sort((a, b) => a.year - b.year);
                                  if (x <= sorted[0].year) return sorted[0].cdf;
                                  if (x >= sorted[sorted.length - 1].year) return sorted[sorted.length - 1].cdf;
                                  for (let i = 0; i < sorted.length - 1; i++) {
                                    if (x >= sorted[i].year && x <= sorted[i + 1].year) {
                                      const t = (x - sorted[i].year) / (sorted[i + 1].year - sorted[i].year);
                                      return sorted[i].cdf + t * (sorted[i + 1].cdf - sorted[i].cdf);
                                    }
                                  }
                                  return 0;
                                };
                                const empiricalCdfValue = getEmpiricalCdfAt(point.x);
                                const editedCdfValue = currentTakeoffSettings.method === 'gamma'
                                  ? evaluateGammaMixtureAt(currentTakeoffControlPoints, point.x, currentTakeoffSettings.gammaReg)
                                  : evaluateSplineAt(currentTakeoffControlPoints, point.x, currentTakeoffSettings.isplineReg);

                                return (
                                  <div className="bg-white border border-gray-200 rounded px-2 py-1.5 shadow-md text-xs">
                                    <div className="font-semibold text-primary">{point.x.toFixed(2)} yr</div>
                                    <div className="text-primary/50">
                                      Empirical: {(empiricalCdfValue * 100).toFixed(1)}%
                                    </div>
                                    <div style={{ color: takeoffMilestoneColor }} className="font-medium">
                                      ATC: {(editedCdfValue * 100).toFixed(1)}%
                                    </div>
                                  </div>
                                );
                              }}
                            />
                            <div className="flex items-center gap-4 mt-2">
                              <div className="flex items-center gap-1.5">
                                <div className="w-4 h-0.5 bg-gray-400 opacity-70"></div>
                                <span className="text-xs text-primary/50">Original (model forecast)</span>
                              </div>
                              <div className="flex items-center gap-1.5">
                                <div className="w-4 h-0.5" style={{ backgroundColor: takeoffMilestoneColor }}></div>
                                <span className="text-xs text-primary/50">Edited</span>
                              </div>
                            </div>
                          </>
                        )}
                      </div>
                      {/* Spacer to match control points table width */}
                      <div className="w-72 flex-shrink-0" />
                    </div>
                  )}

                  {/* Takeoff Action buttons */}
                  <div className="flex gap-4 mt-6">
                    <button
                      onClick={handleTakeoffReset}
                      className="px-4 py-2 text-sm font-medium text-primary bg-white border border-gray-300 rounded-md hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-accent"
                    >
                      Reset to Empirical
                    </button>
                    <button
                      onClick={handleTakeoffExport}
                      className="px-4 py-2 text-sm font-medium text-white bg-accent border border-transparent rounded-md hover:opacity-90 focus:outline-none focus:ring-2 focus:ring-accent"
                    >
                      Export ATC Takeoff Speeds
                    </button>
                  </div>

                  {/* Edited takeoff milestones indicator */}
                  {editedTakeoffMilestones.size > 0 && (
                    <div className="mt-4 text-sm text-primary/50">
                      Edited milestones: {Array.from(editedTakeoffMilestones.keys()).join(', ')}
                    </div>
                  )}
                </ChartSyncProvider>
              )}
            </section>
          </main>
        </div>
      </div>
    </div>
  );
}
