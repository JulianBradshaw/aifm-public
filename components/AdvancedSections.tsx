import React, { createContext, useContext, useState, useRef } from 'react';
import { createPortal } from 'react-dom';
import { ParametersType, resolveParamValue } from '@/constants/parameters';
import { ParameterSlider, ParameterSliderProps } from './ParameterSlider';
import { formatAsPowerOfTenText, formatWorkTimeDuration, formatDollarsPerYear, formatYearMonthShort } from '@/utils/formatting';
import { formatTo3SigFigs } from '@/utils/formatting';
import { KEY_PARAMETER_CONFIGS, PARAMETER_DOC_ANCHORS, buildDocLink } from '@/constants/keyParameterConfigs';

interface ModelDefaults {
    optimal_ces_eta_init?: number;
    automation_interp_type?: string;
    ai_research_taste_slope?: number;
    anchor_progress_at_strong_cognitive_horizon?: number;
    present_year?: number;
    present_progress?: number;
    progress_at_aa?: number;
}

export interface ParameterConfig {
    defaults?: ModelDefaults;
    bounds?: Record<string, [number, number]>;
    metadata?: Record<string, unknown>;
}

export interface SimplificationCheckboxes {
    enableExperimentAutomation: boolean;
    setEnableExperimentAutomation: React.Dispatch<React.SetStateAction<boolean>>;
    useExperimentThroughputCES: boolean;
    setUseExperimentThroughputCES: React.Dispatch<React.SetStateAction<boolean>>;
    enableSoftwareProgress: boolean;
    setEnableSoftwareProgress: React.Dispatch<React.SetStateAction<boolean>>;
    useComputeLaborGrowthSlowdown: boolean;
    setUseComputeLaborGrowthSlowdown: React.Dispatch<React.SetStateAction<boolean>>;
}

export interface AdvancedSectionsProps {
    uiParameters: ParametersType;
    setUiParameters: React.Dispatch<React.SetStateAction<ParametersType>>;
    allParameters: ParameterConfig | null;
    isDragging: boolean;
    setIsDragging: React.Dispatch<React.SetStateAction<boolean>>;
    commitParameters: (nextParameters?: ParametersType) => void;
    scHorizonLogBounds: { min: number; max: number };
    parallelPenaltyBounds: { min: number; max: number };
    summary?: { beta_software?: number; r_software?: number; slope_times_log_f?: number; [key: string]: unknown } | null;
    lockedParameters?: Set<string>;
    onToggleTrajectoryDebugger?: () => void;
    isTrajectoryDebuggerOpen?: boolean;
    simplificationCheckboxes?: SimplificationCheckboxes;
    // Bounds from sampling config to ensure sliders stay within valid sampling ranges
    samplingConfigBounds?: Record<string, { min?: number; max?: number }>;
    onExportCSV?: () => void;
    onOpenCsvEditor?: () => void;
    hasCustomTimeSeries?: boolean;
    baselineParameters?: ParametersType;
    horizonPercentile?: 'p50' | 'p80';
    onHorizonPercentileChange?: (percentile: 'p50' | 'p80') => void;
}

interface SliderContextValue {
    uiParameters: ParametersType;
    setUiParameters: React.Dispatch<React.SetStateAction<ParametersType>>;
    allParameters: ParameterConfig | null;
    isDragging: boolean;
    setIsDragging: React.Dispatch<React.SetStateAction<boolean>>;
    commitParameters: (nextParameters?: ParametersType) => void;
    lockedParameters?: Set<string>;
    samplingConfigBounds?: Record<string, { min?: number; max?: number }>;
    baselineParameters?: ParametersType;
}

const SliderContext = createContext<SliderContextValue | null>(null);

type SliderProps = Omit<ParameterSliderProps,
    'value' | 'uiParameters' | 'setUiParameters' | 'allParameters' |
    'isDragging' | 'setIsDragging' | 'commitParameters'
> & {
    paramName: keyof ParametersType;
    lockedBy?: string | string[];
};

const Slider: React.FC<SliderProps> = ({ paramName, lockedBy, disabled, customMin, customMax, label, description, ...rest }) => {
    const ctx = useContext(SliderContext);
    if (!ctx) throw new Error('Slider must be used within SliderContext');

    const { uiParameters, setUiParameters, allParameters, isDragging, setIsDragging, commitParameters, lockedParameters, samplingConfigBounds, baselineParameters } = ctx;

    const isLocked = (() => {
        if (!lockedParameters) return false;
        const names = Array.isArray(lockedBy) ? lockedBy : lockedBy ? [lockedBy] : [paramName];
        return names.some(n => lockedParameters.has(n));
    })();

    // Get bounds from sampling config if available, using provided customMin/Max as override
    const samplingBounds = samplingConfigBounds?.[paramName];
    const effectiveMin = customMin ?? samplingBounds?.min;
    const effectiveMax = customMax ?? samplingBounds?.max;

    // Get doc anchor for this parameter
    const docAnchor = PARAMETER_DOC_ANCHORS[paramName];

    return (
        <div className="space-y-0.5">
            {label && (
                <div className="flex items-center">
                    <label className="block text-[11px] font-medium text-foreground leading-tight">
                        {label}
                    </label>
                    {description && <InfoTooltip description={description} docAnchor={docAnchor} />}
                </div>
            )}
            <ParameterSlider
                paramName={paramName}
                value={resolveParamValue(paramName, uiParameters)}
                uiParameters={uiParameters}
                setUiParameters={setUiParameters}
                allParameters={allParameters}
                isDragging={isDragging}
                setIsDragging={setIsDragging}
                commitParameters={commitParameters}
                disabled={disabled || isLocked}
                customMin={effectiveMin}
                customMax={effectiveMax}
                label=""
                interactiveRationale={false}
                baselineParameters={baselineParameters}
                {...rest}
            />
        </div>
    );
};

const Toggle: React.FC<{
    label: string;
    description: string;
    checked: boolean;
    onChange: (checked: boolean) => void;
    disabled?: boolean;
}> = ({ label, description, checked, onChange, disabled }) => (
    <div className="mb-2 rounded-lg">
        <div className="flex items-center justify-between gap-2">
            <div className="min-w-0">
                <label className="block text-[11px] font-medium text-foreground leading-tight">{label}</label>
                <div className="text-[10px] text-gray-500 leading-tight">{description}</div>
            </div>
            <label className={`relative inline-flex items-center shrink-0 ${disabled ? 'cursor-not-allowed opacity-50' : 'cursor-pointer'}`}>
                <input
                    type="checkbox"
                    checked={checked}
                    onChange={(e) => onChange(e.target.checked)}
                    disabled={disabled}
                    className="sr-only peer"
                />
                <div className="w-8 h-4 bg-gray-200 peer-focus:outline-none peer-focus:ring-2 peer-focus:ring-blue-300 rounded-full peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-3 after:w-3 after:transition-all peer-checked:bg-blue-600 peer-disabled:opacity-50 peer-disabled:cursor-not-allowed" />
            </label>
        </div>
    </div>
);

const Section: React.FC<{
    title: string;
    subtitle?: string;
    children: React.ReactNode;
    className?: string;
}> = ({ title, subtitle, children, className = "mb-3" }) => (
    <details className={`${className} pt-1 rounded-none !m-0 !px-0 border-t border-b-0 border-l-0 border-r-0 border-gray-300`}>
        <summary className="text-[11px] font-bold cursor-pointer py-0.5">
            {title}
            {subtitle && <span className="text-[10px] text-gray-500 font-normal ml-1">{subtitle}</span>}
        </summary>
        <div className="mt-2">{children}</div>
    </details>
);

const SliderGrid: React.FC<{ children: React.ReactNode }> = ({ children }) => (
    <div className="grid grid-cols-1 gap-2">{children}</div>
);

// Small info tooltip for advanced parameter descriptions
const InfoTooltip: React.FC<{ description: string; docAnchor?: string }> = ({ description, docAnchor }) => {
    const [showTooltip, setShowTooltip] = useState(false);
    const [tooltipPosition, setTooltipPosition] = useState({ top: 0, left: 0 });
    const triggerRef = useRef<HTMLSpanElement>(null);

    const openTooltip = () => {
        const el = triggerRef.current;
        if (el) {
            const rect = el.getBoundingClientRect();
            const tooltipWidth = 224; // w-56 = 14rem = 224px
            const padding = 8;

            let left = rect.right + padding;

            // Prevent tooltip from going off-screen horizontally
            if (left + tooltipWidth > window.innerWidth - padding) {
                left = rect.left - tooltipWidth - padding;
            }
            if (left < padding) {
                left = padding;
            }

            // Position below the trigger
            let top = rect.bottom + 4;

            // Prevent tooltip from going off-screen vertically
            if (top + 80 > window.innerHeight) {
                top = rect.top - 80 - 4;
            }

            setTooltipPosition({ top, left });
        }
        setShowTooltip(true);
    };

    const docLink = docAnchor ? buildDocLink(docAnchor) : null;

    return (
        <>
            <span
                ref={triggerRef}
                className="ml-auto relative inline-flex h-3 w-3 min-w-3 items-center justify-center rounded-full border border-gray-400 text-[8px] font-semibold leading-none text-gray-500 cursor-help transition-colors hover:border-gray-600 hover:text-gray-700"
                role="button"
                tabIndex={0}
                aria-label="Show description"
                aria-expanded={showTooltip}
                onMouseEnter={openTooltip}
                onMouseLeave={() => setShowTooltip(false)}
                onFocus={openTooltip}
                onBlur={() => setShowTooltip(false)}
            >
                i
            </span>
            {docLink && (
                <a
                    href={docLink}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="ml-0.5 inline-flex h-3 w-3 min-w-3 items-center justify-center text-gray-400 hover:text-[#2A623D] transition-colors"
                    title="View detailed definition"
                    onClick={(e) => e.stopPropagation()}
                >
                    <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6" />
                        <polyline points="15 3 21 3 21 9" />
                        <line x1="10" y1="14" x2="21" y2="3" />
                    </svg>
                </a>
            )}
            {showTooltip && typeof document !== 'undefined' && createPortal(
                <div
                    className="pointer-events-none fixed z-[9999]"
                    style={{ top: tooltipPosition.top, left: tooltipPosition.left }}
                >
                    <div className="w-56 rounded-md bg-gray-900 px-2.5 py-1.5 text-[10px] text-white shadow-lg font-system-mono font-normal">
                        {description}
                    </div>
                </div>,
                document.body
            )}
        </>
    );
};

const SimplificationCheckbox: React.FC<{
    id: string;
    label: string;
    tooltip: string;
    checked: boolean;
    onChange: (checked: boolean) => void;
}> = ({ id, label, tooltip, checked, onChange }) => (
    <div className="group relative flex items-center gap-1.5">
        <input
            type="checkbox"
            id={id}
            checked={checked}
            onChange={(e) => onChange(e.target.checked)}
            className="h-3 w-3 rounded border-gray-300 text-accent focus:ring-accent accent-accent cursor-pointer"
        />
        <label htmlFor={id} className="text-[10px] font-medium text-gray-900 cursor-pointer leading-tight">
            {label}
        </label>
        <div className="pointer-events-none absolute left-1/2 -translate-x-1/2 bottom-full mb-1 hidden w-48 rounded-md bg-gray-900 px-2 py-1.5 text-[10px] text-white shadow-lg group-hover:block z-50">
            <div className="relative">
                {tooltip}
                <div className="absolute left-1/2 -translate-x-1/2 top-full w-0 h-0 border-x-4 border-x-transparent border-t-4 border-t-gray-900"></div>
            </div>
        </div>
    </div>
);

export const AdvancedSections: React.FC<AdvancedSectionsProps> = ({
    uiParameters,
    setUiParameters,
    allParameters,
    isDragging,
    setIsDragging,
    commitParameters,
    scHorizonLogBounds,
    parallelPenaltyBounds,
    summary,
    lockedParameters,
    onToggleTrajectoryDebugger,
    isTrajectoryDebuggerOpen,
    simplificationCheckboxes,
    samplingConfigBounds,
    onExportCSV,
    onOpenCsvEditor,
    hasCustomTimeSeries,
    baselineParameters,
    horizonPercentile = 'p80',
    onHorizonPercentileChange,
}) => {
    const handleToggle = (paramName: keyof ParametersType) => (checked: boolean) => {
        const next = { ...uiParameters, [paramName]: checked };
        setUiParameters(next);
        commitParameters(next);
    };

    const handleTasteScheduleChange = (value: string) => {
        const next = { ...uiParameters, taste_schedule_type: value as ParametersType['taste_schedule_type'] };
        setUiParameters(next);
        commitParameters(next);
    };

    const handleAutomationInterpTypeChange = (value: string) => {
        const next = { ...uiParameters, automation_interp_type: value };
        setUiParameters(next);
        commitParameters(next);
    };

    // What the current schedule already implies at the A(p)=1 crossing, surfaced by
    // the backend inside `summary`. Used only to seed the toggle.
    const impliedSerialUpliftAtAc = (() => {
        const us = (summary as { uplift_solve?: { serial_uplift_at_ac_implied?: number } } | null | undefined)?.uplift_solve;
        const v = us?.serial_uplift_at_ac_implied;
        return typeof v === 'number' && Number.isFinite(v) ? v : null;
    })();

    // Seeding matters here: the reachable floor for uplift-at-AC moves with M* and
    // the doubling time (roughly 2.3x to 369x across plausible inputs), so there is
    // no constant that is safe to switch on with. impliedSerialUpliftAtAc is what the
    // current schedule already produces at the crossing, so turning the toggle on is
    // a no-op until the user actually moves the slider.
    const handleSolveAsymptoteToggle = (checked: boolean) => {
        const seed = impliedSerialUpliftAtAc;
        const next = {
            ...uiParameters,
            serial_uplift_at_ac: checked ? (seed != null && Number.isFinite(seed) ? seed : 20) : null,
        };
        setUiParameters(next);
        commitParameters(next);
    };

    const handleAcAnchorModeChange = (value: string) => {
        // The effective compute gap only exists in time-horizon mode; clear it on
        // the way out so a hidden toggle can't leave a gap silently applied.
        const next = {
            ...uiParameters,
            ac_anchor_mode: value,
            ...(value !== 'time-horizon' ? { benchmarks_and_gaps_mode: false } : {}),
        };
        setUiParameters(next);
        commitParameters(next);
    };

    // Each anchor observable binds at its own reference date (null => the present
    // day; presets pin each to its elicitation vintage). The "as of ..." labels
    // below quote the live value so moving a date slider can't leave a stale
    // date claim behind.
    const horizonRefLabel = formatYearMonthShort(resolveParamValue('horizon_reference_date', uiParameters));
    const revenueRefLabel = formatYearMonthShort(resolveParamValue('revenue_reference_date', uiParameters));
    const eciRefLabel = formatYearMonthShort(resolveParamValue('eci_reference_date', uiParameters));
    const upliftRefLabel = formatYearMonthShort(resolveParamValue('uplift_reference_date', uiParameters));

    const ctxValue: SliderContextValue = {
        uiParameters, setUiParameters, allParameters, isDragging, setIsDragging, commitParameters, lockedParameters, samplingConfigBounds, baselineParameters
    };

    return (
        <SliderContext.Provider value={ctxValue}>
            <div className="space-y-1 mt-4">
                {/* Input Time Series Editor */}
                {onOpenCsvEditor && (
                    <div className="pb-2 mb-1 border-b border-gray-200">
                        <button
                            onClick={onOpenCsvEditor}
                            className="w-full flex items-center justify-between px-3 py-2 text-xs font-semibold text-gray-700 bg-gray-50 border border-gray-200 rounded-sm hover:bg-gray-100 transition-colors group"
                        >
                            <span className="flex items-center gap-2">
                                <svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" className="text-gray-500 group-hover:text-gray-700">
                                    <path d="M2 2h8l4 4v8a1 1 0 0 1-1 1H3a1 1 0 0 1-1-1V2z" />
                                    <path d="M10 2v4h4" />
                                    <path d="M5 9h6M5 12h4" />
                                </svg>
                                Edit Input Time Series
                                {hasCustomTimeSeries && (
                                    <span className="inline-block w-1.5 h-1.5 rounded-full bg-amber-500" title="Custom data active" />
                                )}
                            </span>
                            <svg width="12" height="12" viewBox="0 0 12 12" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" className="text-gray-400">
                                <path d="M4.5 2.5l4 3.5-4 3.5" />
                            </svg>
                        </button>
                    </div>
                )}

                {/* Model Simplifications */}
                {simplificationCheckboxes && (
                    <Section title="Model Simplifications">
                        <p className="text-[10px] text-gray-500 mb-2">Uncheck to simplify the model</p>
                        <SliderGrid>
                            {/* "Coding Automation" checkbox removed: its unchecked path
                                is incompatible with the default ac_anchor_mode='uplift-solved',
                                which derives the AC date FROM the coding uplift -- so
                                "no coding automation" left the solver nothing to anchor on
                                and 400'd the API. */}
                            <SimplificationCheckbox
                                id="enable-experiment-automation"
                                label="Research Taste Automation"
                                tooltip="When disabled: Removes research taste automation from the model by setting the median-to-top taste multiplier to near-one."
                                checked={simplificationCheckboxes.enableExperimentAutomation}
                                onChange={simplificationCheckboxes.setEnableExperimentAutomation}
                            />
                            <SimplificationCheckbox
                                id="enable-software-progress"
                                label="Software Progress"
                                tooltip="When disabled: Only considers training compute scaling, assumes software efficiency does not improve. It sets the 2024 software efficiency growth rate (which we use to anchor software progress) to zero."
                                checked={simplificationCheckboxes.enableSoftwareProgress}
                                onChange={simplificationCheckboxes.setEnableSoftwareProgress}
                            />
                            <SimplificationCheckbox
                                id="use-experiment-throughput-ces"
                                label="Experiment Throughput Hard Bottlenecks"
                                tooltip="When disabled: Assumes that taking coding labor or experiment compute to infinity leads to unbounded experiment throughput. Uses a Cobb-Douglas production function instead of the CES function."
                                checked={simplificationCheckboxes.useExperimentThroughputCES}
                                onChange={simplificationCheckboxes.setUseExperimentThroughputCES}
                            />
                            {/* "Compute and Human Labor Growth Slowdown" checkbox removed for now:
                                its unchecked path loaded inputs/2200_exponential.csv, which was
                                dropped in the H100e input migration, so unchecking 500'd the API. */}
                            {/* "Variable Horizon Doubling Difficulty" checkbox removed: it was a
                                no-op under the default ac_anchor_mode='uplift-solved', where the
                                horizon params never feed the trajectory (ddgf 0.5/1.0/1.4 all give
                                byte-identical output). It only ever bit under 'time-horizon'. */}
                        </SliderGrid>
                    </Section>
                )}

                {/* Time Horizon & Progress */}
                <Section
                    title="Automated Coder Anchors"
                // subtitle={uiParameters.benchmarks_and_gaps_mode ? "(Benchmarks & Gaps Mode)" : "(Standard Mode)"}
                >

                    <div className="space-y-2 mb-4">
                        <label className="block text-xs font-medium text-foreground">
                            AC Anchor Mode
                        </label>
                        <select
                            value={uiParameters.ac_anchor_mode}
                            onChange={(e) => handleAcAnchorModeChange(e.target.value)}
                            className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm"
                        >
                            <option value="time-horizon">Time horizon</option>
                            <option value="revenue">Revenue</option>
                            <option value="eci">ECI</option>
                            <option value="uplift-solved">Coding uplift</option>
                        </select>
                    </div>

                    <div className={`mb-4 text-xs font-bold text-black/75 rounded-lg`}>
                        {uiParameters.ac_anchor_mode === 'revenue'
                            ? 'Revenue mode: AC is reached when the extrapolated revenue trend hits the required annual revenue.'
                            : uiParameters.ac_anchor_mode === 'eci'
                            ? 'ECI mode: AC is reached when the extrapolated ECI trend hits the required score.'
                            : uiParameters.ac_anchor_mode === 'uplift-solved'
                            ? 'Coding uplift mode: the automation schedule is solved so the model\u2019s own serial coding uplift matches the level and doubling rate you specify at the reference date. AC is then wherever automation reaches 100% \u2014 an output, not a threshold you set.'
                            : (uiParameters.benchmarks_and_gaps_mode
                                ? 'Gap mode: AC is reached when the time horizon requirement is met and after a further increase in effective compute equal to the specified gap.'
                                : 'Time horizon mode: AC is reached when the AC horizon threshold is met.')}
                    </div>

                    {uiParameters.ac_anchor_mode === 'time-horizon' && (
                        <Toggle
                            label="Include an Effective Compute Gap"
                            description="Require additional effective compute for AC to be achieved after the time horizon requirement is met."
                            checked={uiParameters.benchmarks_and_gaps_mode}
                            onChange={handleToggle('benchmarks_and_gaps_mode')}
                        />
                    )}

                    <SliderGrid>
                        {uiParameters.ac_anchor_mode === 'time-horizon' && (
                            <Slider
                                paramName="ac_time_horizon_minutes"
                                label={KEY_PARAMETER_CONFIGS.ac_time_horizon_minutes.label}
                                description={KEY_PARAMETER_CONFIGS.ac_time_horizon_minutes.description}
                                customMin={scHorizonLogBounds.min}
                                customMax={scHorizonLogBounds.max}
                                step={KEY_PARAMETER_CONFIGS.ac_time_horizon_minutes.step}
                                customFormatValue={KEY_PARAMETER_CONFIGS.ac_time_horizon_minutes.formatValue}
                                useLogScale={KEY_PARAMETER_CONFIGS.ac_time_horizon_minutes.useLogScale}
                            />
                        )}
                        {uiParameters.ac_anchor_mode === 'revenue' && (
                            <>
                                <Slider
                                    paramName="present_revenue"
                                    label={`Leading AGI company annualized revenue as of ${revenueRefLabel}`}
                                    description={`The highest annualized revenue across all frontier model developers, as of ${revenueRefLabel}. (Annualized revenue is the most recent month's revenue multiplied by 12.)`}
                                    customMin={1e8}
                                    customMax={1e13}
                                    step={0.1}
                                    customFormatValue={formatDollarsPerYear}
                                    useLogScale
                                />
                                <Slider
                                    paramName="present_revenue_growth_rate"
                                    label={`How fast annualized revenue grows as of ${revenueRefLabel}`}
                                    description="Annual growth factor of annualized revenue (3 means revenue triples each year). This is used to set a long-term trend, so transient factors should be excluded; it shouldn't literally be the current growth rate."
                                    customMin={1.1}
                                    customMax={10}
                                    step={0.01}
                                    customFormatValue={(v) => `×${v.toFixed(2)}/yr`}
                                    useLogScale
                                />
                                <Slider
                                    paramName="revenue_reference_date"
                                    label={`Revenue reference date (${revenueRefLabel})`}
                                    description="The date at which the revenue level and growth estimates hold. Defaults to the present day; presets pin it to the date the estimates were made."
                                    customMin={2025.0}
                                    customMax={2028.0}
                                    step={1 / 12}
                                    customFormatValue={formatYearMonthShort}
                                />
                                <Slider
                                    paramName="revenue_required_for_ac"
                                    label="Annualized revenue corresponding to Automated Coder (AC)"
                                    description="The annualized revenue of the first frontier model developer that develops AC internally, when they do so."
                                    customMin={1e10}
                                    customMax={1e15}
                                    step={0.1}
                                    customFormatValue={formatDollarsPerYear}
                                    useLogScale
                                />
                            </>
                        )}
                        {uiParameters.ac_anchor_mode === 'eci' && (
                            <>
                                <Slider
                                    paramName="eci_at_present_day"
                                    label={`ECI score as of ${eciRefLabel}`}
                                    description="Epoch Capabilities Index score at the reference date; the extrapolated ECI trend is anchored here."
                                    customMin={100}
                                    customMax={220}
                                    step={1}
                                    decimalPlaces={0}
                                />
                                <Slider
                                    paramName="eci_growth_rate"
                                    label={`How fast the ECI score grows as of ${eciRefLabel}`}
                                    description="Instantaneous ECI growth at the reference date, in points per year."
                                    customMin={1}
                                    customMax={50}
                                    step={0.5}
                                    decimalPlaces={1}
                                />
                                <Slider
                                    paramName="eci_reference_date"
                                    label={`ECI reference date (${eciRefLabel})`}
                                    description="The date at which the ECI score and growth estimates hold. Defaults to the present day; presets pin it to the date the estimates were made."
                                    customMin={2025.0}
                                    customMax={2028.0}
                                    step={1 / 12}
                                    customFormatValue={formatYearMonthShort}
                                />
                                <Slider
                                    paramName="eci_required_for_ac"
                                    label="ECI score required to achieve Automated Coder (AC)"
                                    description="AC is reached when the extrapolated ECI trend crosses this score."
                                    customMin={180}
                                    customMax={500}
                                    step={1}
                                    decimalPlaces={0}
                                />
                            </>
                        )}
                        {uiParameters.ac_anchor_mode === 'uplift-solved' && (
                            <>
                                {/* Also shown in the Coding Automation section; both stay in sync */}
                                <Slider
                                    paramName="serial_uplift_at_reference_date"
                                    label={`Serial coding uplift as of ${upliftRefLabel}`}
                                    description={`The most capable model's coding uplift as of ${upliftRefLabel}. Coding uplift is the speedup in coding work from AI assistance. An uplift of 2x would mean that AIs were increasing coding productivity by as much as if the AGI project sped up their employees by 2x.`}
                                    customMin={1.05}
                                    customMax={100}
                                    step={0.01}
                                    customFormatValue={(v) => `${formatTo3SigFigs(v)}x`}
                                    useLogScale
                                />
                                <Slider
                                    paramName="excess_uplift_doubling_time"
                                    label={`How long it takes serial coding uplift − 1 to double, as of ${upliftRefLabel}`}
                                    description={`How long it takes serial coding uplift − 1 to double, as of ${upliftRefLabel}. This is used to fit current behavior, the doubling time changes over time.`}
                                    customMin={0.6}
                                    customMax={240}
                                    step={0.1}
                                    decimalPlaces={1}
                                    customFormatValue={(v) => `${v.toFixed(1)} mo`}
                                    useLogScale
                                />
                                <Slider
                                    paramName="uplift_reference_date"
                                    label={`Uplift reference date (${upliftRefLabel})`}
                                    description="The date at which the uplift level and doubling-time estimates hold. Defaults to the present day; presets pin it to the date the estimates were made."
                                    customMin={2025.0}
                                    customMax={2028.0}
                                    step={1 / 12}
                                    customFormatValue={formatYearMonthShort}
                                />
                                <Toggle
                                    label="Solve the asymptote from uplift at AC"
                                    description="Off: you set the logistic asymptote directly. On: you state how much serial uplift exists when automation reaches 100%, and the asymptote is solved to match. AC is still the A(p)=1 crossing either way."
                                    checked={uiParameters.serial_uplift_at_ac != null}
                                    onChange={handleSolveAsymptoteToggle}
                                />
                                {uiParameters.serial_uplift_at_ac != null && (
                                    <Slider
                                        paramName="serial_uplift_at_ac"
                                        label="Serial coding uplift at AC"
                                        description="How much serial coding uplift exists when automation reaches 100%. When set, the logistic asymptote is solved to match it instead of being an input. There is a floor on what is reachable, and it moves with the uplift level and doubling time."
                                        customMin={2}
                                        customMax={1000}
                                        step={0.1}
                                        customFormatValue={(v) => `${formatTo3SigFigs(v)}x`}
                                        useLogScale
                                    />
                                )}
                            </>
                        )}
                        {uiParameters.ac_anchor_mode === 'time-horizon' && uiParameters.benchmarks_and_gaps_mode && (
                            <Slider
                                paramName="gap_years"
                                label="Size of Effective Compute Gap"
                                description="A value of 1 means the magnitude of the gap is the effective compute increase in the present year."
                                customMin={0.1}
                                fallbackMax={10.0}
                                step={0.1}
                                customFormatValue={(v) => `${v.toFixed(1)} × 2025-effective-FLOP-growth`}
                                useLogScale
                            />
                        )}
                        {uiParameters.ac_anchor_mode === 'time-horizon' && (
                            <>
                                <Slider
                                    paramName="present_horizon"
                                    label={`Time horizon as of ${horizonRefLabel}`}
                                    description="The coding time horizon at the horizon reference date; the extrapolated horizon trend is anchored here."
                                    step={0.1}
                                    fallbackMin={0.01}
                                    fallbackMax={100}
                                    customFormatValue={formatWorkTimeDuration}
                                    useLogScale
                                />
                                <Slider
                                    paramName="present_doubling_time"
                                    label={`Time horizon doubling time as of ${horizonRefLabel}`}
                                    description={`How long it takes the coding time horizon to double, as of ${horizonRefLabel} (considered as a long-term trend).`}
                                    fallbackMin={KEY_PARAMETER_CONFIGS.present_doubling_time.fallbackMin}
                                    fallbackMax={KEY_PARAMETER_CONFIGS.present_doubling_time.fallbackMax}
                                    step={KEY_PARAMETER_CONFIGS.present_doubling_time.step}
                                    decimalPlaces={KEY_PARAMETER_CONFIGS.present_doubling_time.decimalPlaces}
                                    customFormatValue={KEY_PARAMETER_CONFIGS.present_doubling_time.formatValue}
                                    useLogScale={KEY_PARAMETER_CONFIGS.present_doubling_time.useLogScale}
                                />
                                <Slider
                                    paramName="doubling_difficulty_growth_factor"
                                    label={KEY_PARAMETER_CONFIGS.doubling_difficulty_growth_factor.label}
                                    description={KEY_PARAMETER_CONFIGS.doubling_difficulty_growth_factor.description}
                                    customMin={KEY_PARAMETER_CONFIGS.doubling_difficulty_growth_factor.customMin}
                                    customMax={KEY_PARAMETER_CONFIGS.doubling_difficulty_growth_factor.customMax}
                                    step={KEY_PARAMETER_CONFIGS.doubling_difficulty_growth_factor.step}
                                    customFormatValue={KEY_PARAMETER_CONFIGS.doubling_difficulty_growth_factor.formatValue}
                                />
                                <Slider
                                    paramName="horizon_reference_date"
                                    label={`Horizon reference date (${horizonRefLabel})`}
                                    description="The date at which the time-horizon and doubling-time estimates hold. Defaults to the present day; presets pin it to the date the estimates were made."
                                    customMin={2025.0}
                                    customMax={2028.0}
                                    step={1 / 12}
                                    customFormatValue={formatYearMonthShort}
                                />
                            </>
                        )}
                    </SliderGrid>

                </Section>

                {/* Coding Automation */}
                <Section title="Coding Automation">
                    <SliderGrid>
                        <Slider
                            paramName="serial_uplift_at_reference_date"
                            label={`Serial coding uplift as of ${upliftRefLabel}`}
                            description={`The most capable model's coding uplift as of ${upliftRefLabel}. Coding uplift is the speedup in coding work from AI assistance. An uplift of 2x would mean that AIs were increasing coding productivity by as much as if the AGI project sped up their employees by 2x.`}
                            customMin={1.05}
                            customMax={100}
                            step={0.01}
                            customFormatValue={(v) => `${formatTo3SigFigs(v)}x`}
                            useLogScale
                        />
                        <Slider
                            paramName="excess_uplift_doubling_time"
                            label={`How long it takes serial coding uplift − 1 to double, as of ${upliftRefLabel}`}
                            description={`How long it takes serial coding uplift − 1 to double, as of ${upliftRefLabel}. This is used to fit current behavior, the doubling time changes over time.`}
                            customMin={0.6}
                            customMax={240}
                            step={0.1}
                            decimalPlaces={1}
                            customFormatValue={(v) => `${v.toFixed(1)} mo`}
                            useLogScale
                        />
                        <Slider
                            paramName="uplift_reference_date"
                            label={`Uplift reference date (${upliftRefLabel})`}
                            description="The date at which the uplift level and doubling-time estimates hold. Defaults to the present day; presets pin it to the date the estimates were made."
                            customMin={2025.0}
                            customMax={2028.0}
                            step={1 / 12}
                            customFormatValue={formatYearMonthShort}
                        />
                        <Slider
                            paramName="coding_automation_efficiency_slope"
                            label="Coding Automation Efficiency Improvement Factor (η)"
                            description="How many OOMs does the GPU→human 'conversion rate' improve per progress-year (i.e. per year's worth of effective compute growth at the present-day rate)? Internally converted to OOMs/OOM using the anchor progress rate."
                            step={0.1}
                            fallbackMin={0.01}
                            fallbackMax={10.0}
                            decimalPlaces={2}
                            customFormatValue={(v) => `${v.toFixed(2)} OOMs/progress-year`}
                            useLogScale
                        />
                        <Slider
                            paramName="rho_coding_labor"
                            label="Coding Labor Substitutability (ρ_c)"
                            description="This controls to what extent coding is a fixed series of tasks, vs. being able to substitute automated tasks for non-automated ones (lower values means more like the former)."
                            step={0.1}
                            fallbackMin={-10}
                            fallbackMax={0}
                            decimalPlaces={2}
                        />
                        <Slider
                            paramName="coding_labor_exponent"
                            label="Coding Labor Parallelization Penalty (λ)"
                            description="Multiplying the size of your coding labor force by N× is equivalent to speeding up coding labor by (N^λ)×. This is used to convert parallel to serial coding labor before entering the CES."
                            customMin={parallelPenaltyBounds.min}
                            customMax={parallelPenaltyBounds.max}
                            step={0.01}
                            decimalPlaces={3}
                            lockedBy={['parallel_penalty', 'coding_labor_exponent']}
                        />
                        {uiParameters.automation_interp_type === 'logistic' && (
                            <Slider
                                paramName="automation_logistic_asymptote"
                                label="Coding Automation Fraction Logistic Asymptote"
                                description={uiParameters.serial_uplift_at_ac != null
                                    ? 'Ignored while \u201cSolve the asymptote from uplift at AC\u201d is on \u2014 the asymptote is being solved to reproduce the uplift you specified at AC. Turn that toggle off to set it directly.'
                                    : 'Upper asymptote for the logistic automation schedule. Values above 1 allow automation fraction to overshoot before being clipped to 1.'}
                                disabled={uiParameters.serial_uplift_at_ac != null}
                                step={0.01}
                                fallbackMin={1.0001}
                                fallbackMax={2.0}
                                decimalPlaces={4}
                                useLogScale
                            />
                        )}
                        <Slider
                            paramName="max_serial_coding_labor_multiplier"
                            label="Maximum Serial Coding Labor Multiplier"
                            description="At the physical limits of coding capability, AI could provide productivity benefits equivalent to speeding up all human coders by this much."
                            step={1.0}
                            stepCount={30}
                            fallbackMin={1.0}
                            fallbackMax={1e12}
                            customFormatValue={formatAsPowerOfTenText}
                            useLogScale
                        />
                    </SliderGrid>
                </Section>

                {/* CES Production Functions */}
                <Section
                    title="Experiment Throughput Production"
                // subtitle={uiParameters.direct_input_exp_cap_ces_params ? "(Direct Input Mode)" : "(Computed from Constraints)"}
                >
                    <Toggle
                        label="Direct Input CES Params"
                        description="Use direct CES parameter input vs. computed from constraints"
                        checked={uiParameters.direct_input_exp_cap_ces_params}
                        onChange={handleToggle('direct_input_exp_cap_ces_params')}
                        disabled={lockedParameters?.has('direct_input_exp_cap_ces_params')}
                    />

                    {uiParameters.direct_input_exp_cap_ces_params ? (
                        <div className="space-y-4">
                            <div className="text-sm text-blue-600 bg-blue-50 p-3 rounded-lg">
                                Direct input mode: CES parameters are used as provided
                            </div>
                            <SliderGrid>
                                <Slider
                                    paramName="rho_experiment_capacity"
                                    label="Substitutability (ρ_x)"
                                    description="Controls the degree of substitutability between experiment compute and coding labor. Higher values mean more substitutable."
                                    step={0.01}
                                    fallbackMin={-1}
                                    fallbackMax={1}
                                    decimalPlaces={3}
                                />
                                <Slider
                                    paramName="alpha_experiment_capacity"
                                    label="Experiment Compute Weight (α)"
                                    description="Higher values mean experiment compute is more important relative to coding labor"
                                    step={0.01}
                                    fallbackMin={0.05}
                                    fallbackMax={0.95}
                                    decimalPlaces={3}
                                />
                                <Slider
                                    paramName="experiment_compute_exponent"
                                    label="Experiment Compute Discounting (ζ)"
                                    description="Experiment compute is taken to the exponent ζ before being combined with coding labor."
                                    step={0.01}
                                    fallbackMin={0.001}
                                    fallbackMax={10}
                                    decimalPlaces={3}
                                />
                            </SliderGrid>
                        </div>
                    ) : (
                        <div className="space-y-4">
                            <div className="text-sm text-blue-600 bg-blue-50 p-3 rounded-lg">
                                Constraints mode: CES parameters are computed from constraints
                            </div>
                            <SliderGrid>
                                <Slider
                                    paramName="inv_compute_anchor_exp_cap"
                                    label="Slowdown from 10× less Experiment Compute"
                                    description="By what factor slower AI software progress would go in present day if you immediately had 10× less experiment compute."
                                    step={0.1}
                                    fallbackMin={1}
                                    fallbackMax={10}
                                    decimalPlaces={1}
                                    customFormatValue={(v) => `${v.toFixed(1)}x slower`}
                                    useLogScale
                                />
                                <Slider
                                    paramName="inf_labor_asymptote"
                                    label="Infinite Coding Labor Asymptote"
                                    description="By what factor faster AI software progress would go in present day if you immediately got unlimited coding labor."
                                    step={0.1}
                                    fallbackMin={1}
                                    fallbackMax={100000}
                                    decimalPlaces={1}
                                    customFormatValue={(v) => `${v.toFixed(1)}x faster`}
                                    useLogScale
                                />
                                <Slider
                                    paramName="inf_compute_asymptote"
                                    label="Infinite Experiment Compute Asymptote"
                                    description="By what factor faster AI software progress would go in present day if you immediately got unlimited experiment compute."
                                    step={10}
                                    fallbackMin={1}
                                    fallbackMax={100000}
                                    decimalPlaces={0}
                                    customFormatValue={(v) => `${v.toFixed(0)}x faster`}
                                    useLogScale
                                />
                            </SliderGrid>
                        </div>
                    )}
                </Section>

                {/* Experiment Selection Automation */}
                <Section title="Research Taste Automation">
                    <SliderGrid>
                        <Slider
                            paramName="ai_research_taste_at_coding_automation_anchor_sd"
                            label="Automated Research Taste at AC"
                            description="When we reach AC, how good will AIs' research taste be relative to the median researcher in the AGI project (0 SDs=median, ~3 SDs=best)"
                            step={0.1}
                            fallbackMin={-10}
                            fallbackMax={23}
                            decimalPlaces={1}
                            customFormatValue={(v) => `${v.toFixed(1)} SDs`}
                        />
                        <Slider
                            paramName="median_to_top_taste_multiplier"
                            label={KEY_PARAMETER_CONFIGS.median_to_top_taste_multiplier.label}
                            description={KEY_PARAMETER_CONFIGS.median_to_top_taste_multiplier.description}
                            step={KEY_PARAMETER_CONFIGS.median_to_top_taste_multiplier.step}
                            fallbackMin={KEY_PARAMETER_CONFIGS.median_to_top_taste_multiplier.fallbackMin}
                            fallbackMax={KEY_PARAMETER_CONFIGS.median_to_top_taste_multiplier.fallbackMax}
                            decimalPlaces={KEY_PARAMETER_CONFIGS.median_to_top_taste_multiplier.decimalPlaces}
                            customFormatValue={KEY_PARAMETER_CONFIGS.median_to_top_taste_multiplier.formatValue}
                            useLogScale={KEY_PARAMETER_CONFIGS.median_to_top_taste_multiplier.useLogScale}
                        />
                        <Slider
                            paramName="ai_research_taste_slope"
                            label={KEY_PARAMETER_CONFIGS.ai_research_taste_slope.label}
                            description={KEY_PARAMETER_CONFIGS.ai_research_taste_slope.description}
                            step={KEY_PARAMETER_CONFIGS.ai_research_taste_slope.step}
                            fallbackMin={KEY_PARAMETER_CONFIGS.ai_research_taste_slope.fallbackMin}
                            fallbackMax={KEY_PARAMETER_CONFIGS.ai_research_taste_slope.fallbackMax}
                            decimalPlaces={KEY_PARAMETER_CONFIGS.ai_research_taste_slope.decimalPlaces}
                            customFormatValue={KEY_PARAMETER_CONFIGS.ai_research_taste_slope.formatValue}
                            useLogScale={KEY_PARAMETER_CONFIGS.ai_research_taste_slope.useLogScale}
                        />
                        <Slider
                            paramName="taste_limit"
                            label="Maximum Research Taste"
                            description="Number of multiplicative median-to-top research taste gaps between the best humans and maximally capable AIs."
                            step={0.1}
                            fallbackMin={0}
                            fallbackMax={100}
                            decimalPlaces={1}
                            customFormatValue={(v) => `${v.toFixed(1)} median-to-top gaps`}
                        />
                        <Slider
                            paramName="taste_limit_smoothing"
                            label="Research Taste Slowdown Factor Halfway to Algorithmic Limit"
                            description="The value L such that, halfway in log space to maximum research taste, each SD of research taste gives you Lx as many OOMs of research taste as an SD gave you near the human range."
                            step={0.001}
                            fallbackMin={0.001}
                            fallbackMax={0.999}
                            decimalPlaces={3}
                        />
                    </SliderGrid>
                </Section>

                {/* Software efficiency and intelligence explosions */}
                <Section title="Software efficiency and intelligence explosions">
                    <SliderGrid>
                        <Slider
                            paramName="software_progress_rate_at_reference_year"
                            label="Software Efficiency OOMs/year in 2024"
                            description="In OOMs/year, how quickly was software efficiency growing in 2024?"
                            step={0.1}
                            fallbackMin={0.00001}
                            fallbackMax={10}
                            decimalPlaces={2}
                        />
                    </SliderGrid>

                    {summary?.beta_software != null && (
                        <div className="mt-4 text-xs text-black/75">
                            <span className="font-semibold">Calculated from other parameters:</span><br />
                            <br />
                            β={formatTo3SigFigs(summary.beta_software)} OOMs of research stock required to get an OOM of software efficiency (used in software efficiency production)<br />
                            <br />
                            {summary?.slope_times_log_f != null && (
                                <>
                                    m={formatTo3SigFigs(summary.slope_times_log_f)} OOMs of automated research taste for each OOM of software efficiency<br />
                                    <br />
                                    Our model has a taste-only singularity, i.e. an intelligence explosion from research taste alone, if m&gt;β (more <a href="#S5" className="underline">here</a>)
                                </>
                            )}
                        </div>
                    )}
                </Section>

                {/* General Capabilities */}
                <Section title="General Capabilities">
                    <SliderGrid>
                        <Slider
                            paramName="ted_ai_m2b"
                            label="Median-to-top-human jumps above SAR needed to reach TED-AI"
                            description="This is the number of median-to-top research taste SD-gaps above SAR needed to reach TED-AI."
                            step={0.1}
                            fallbackMin={0.1}
                            fallbackMax={10}
                            decimalPlaces={1}
                            useLogScale
                        />
                    </SliderGrid>
                    <div className="mt-2 text-xs text-black/75 flex items-center">
                        <span>Median-to-top-human jumps above TED-AI corresponding to ASI: 2 (no uncertainty)</span>
                        <InfoTooltip description="Given that ASI must be 2 median-to-best jumps above TED-AI at its most lagging skill, it seems reasonable to assume that it will be about 2 median-to-best jumps above TED-AI at research taste in particular. For simplicity, we don't include uncertainty. Because an ASI needs to be both more capable than a TED-AI and more capable than a SIAR, it seems non-trivial to create a simple, smoothly behaving parameterization of our uncertainty." />
                    </div>
                </Section>

                {/* Extra Parameters */}
                <Section title="Extra (parameters we don't vary between simulations)">
                    <div className="">
                        <SliderGrid>
                            <Slider
                                paramName="present_day"
                                label="Present Day"
                                description="Used as a reference point for setting other parameters and for capability metrics."
                                step={0.1}
                                fallbackMin={2020.0}
                                fallbackMax={2030.0}
                                decimalPlaces={1}
                            />
                            <Slider
                                paramName="automation_fraction_at_coding_automation_anchor"
                                label="Coding Automation Fraction at AC"
                                description="AC is always achieved when a fraction of 1 is reached. If we implemented the ability to automate tasks even when the AI can’t do so as efficiently as humans, then the automation fraction needed to be an AC would decrease. We could to some extent simulate implementing that by decreasing this fraction requirement."
                                step={0.01}
                                fallbackMin={0.01}
                                fallbackMax={1.0}
                                decimalPlaces={2}
                            />
                            <div className="space-y-2">
                                <label className="block text-xs font-medium text-foreground">
                                    Coding Automation Fraction Schedule Type
                                </label>
                                <div className="text-xs text-gray-500 mb-2">
                                    How automation fraction interpolates between anchor points
                                </div>
                                <select
                                    value={uiParameters.automation_interp_type}
                                    onChange={(e) => handleAutomationInterpTypeChange(e.target.value)}
                                    className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm"
                                >
                                    <option value="linear">Linear</option>
                                    <option value="logistic">Logistic</option>
                                </select>
                            </div>
                            <Slider
                                paramName="optimal_ces_eta_init"
                                label="Initial Automation Efficiency (η_init)"
                                description="The automation efficiency of a task at the point it is considered efficiently automatable (via the coding automation fraction). Measured in human coder full-time-equivalents per H100be."
                                step={0.001}
                                fallbackMin={1e-12}
                                fallbackMax={1e12}
                                decimalPlaces={4}
                                useLogScale
                            />
                            <Slider
                                paramName="top_percentile"
                                label="Percentile of Best Human Researcher"
                                description="The percentile of the best researcher in the AGI project human range"
                                step={0.0001}
                                fallbackMin={0.5}
                                fallbackMax={0.999999}
                                decimalPlaces={6}
                                customFormatValue={(v) => {
                                    if (v >= 0.99999) return `${(v * 100).toFixed(4)}%`;
                                    if (v >= 0.9999) return `${(v * 100).toFixed(3)}%`;
                                    if (v >= 0.999) return `${(v * 100).toFixed(2)}%`;
                                    if (v >= 0.99) return `${(v * 100).toFixed(1)}%`;
                                    return `${(v * 100).toFixed(0)}%`;
                                }}
                                useLogOddsScale
                            />
                        </SliderGrid>
                        {onHorizonPercentileChange && (
                            <div className="mt-2">
                                <Toggle
                                    label="Show p50 on Time Horizon Plot"
                                    description="Show median (p50) instead of p80 benchmark data on the horizon chart"
                                    checked={horizonPercentile === 'p50'}
                                    onChange={(checked) => onHorizonPercentileChange(checked ? 'p50' : 'p80')}
                                />
                            </div>
                        )}
                    </div>
                </Section>

                {/* Configuration - Hidden */}
                <details className="mb-8 hidden">
                    <summary className="text-xs font-bold">Configuration</summary>
                    <div className="ml-4 mt-4">
                        <div className="grid grid-cols-1 gap-4">
                            <div className="space-y-2">
                                <label className="block text-sm font-medium text-foreground font-system-mono">
                                    Automation Interpolation Type
                                </label>
                                <div className="text-xs text-gray-500 mb-2">Type of automation interpolation</div>
                                <select
                                    value="linear"
                                    disabled
                                    className="w-full px-3 py-2 border border-gray-300 rounded-md bg-gray-100 text-gray-600 cursor-not-allowed"
                                >
                                    <option value="linear">Linear (model-supported)</option>
                                </select>
                                <p className="text-xs text-gray-500 mt-1">
                                    Note: only linear automation interpolation is implemented in the model today.
                                </p>
                            </div>

                            <div className="space-y-2">
                                <label className="block text-sm font-medium text-foreground font-system-mono">
                                    Taste Schedule Type
                                </label>
                                <div className="text-xs text-gray-500 mb-2">Type of research taste schedule</div>
                                <select
                                    value={uiParameters.taste_schedule_type}
                                    onChange={(e) => handleTasteScheduleChange(e.target.value)}
                                    className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                                >
                                    <option value="SDs per effective OOM">SDs per effective OOM</option>
                                    <option value="SDs per progress-year">SDs per progress-year</option>
                                </select>
                            </div>
                        </div>
                    </div>
                </details>

                {/* Trajectory Debugger Toggle - Hidden */}
                {onToggleTrajectoryDebugger && (
                    <div className="pt-4 mt-4 border-t border-gray-200 hidden">
                        <button
                            onClick={onToggleTrajectoryDebugger}
                            className={`text-xs font-mono ${isTrajectoryDebuggerOpen ? 'text-blue-600' : 'text-gray-500 hover:text-gray-700'}`}
                        >
                            {isTrajectoryDebuggerOpen ? '✓ Trajectory Debugger' : 'Trajectory Debugger'}
                        </button>
                        <span className="text-[10px] text-gray-400 ml-2">(Alt+D)</span>
                    </div>
                )}

                {/* Export CSV Button */}
                {onExportCSV && (
                    <div className="pt-4 mt-4 border-t border-gray-200">
                        <button
                            onClick={onExportCSV}
                            className="px-3 py-2 text-xs font-semibold uppercase tracking-[0.1em] text-white bg-[#2A623D] border border-[#215337] rounded-sm shadow-sm hover:bg-[#245632] focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-[#2A623D]"
                        >
                            Export Trajectory CSV
                        </button>
                    </div>
                )}
            </div>
        </SliderContext.Provider>
    );
};
