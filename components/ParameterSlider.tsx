import React, { useState, useId, useRef, useEffect, useCallback } from 'react';
import { SliderWithThumbValue } from './SliderWithThumbValue';
import { ParametersType, DEFAULT_PARAMETERS } from '@/constants/parameters';
import { SliderPopover } from './SliderPopover';
import { ChartConfig } from '@/types/chartConfig';
import { ParameterConfig } from './AdvancedSections';
import { useParameterHover, PARAMETER_RATIONALES } from './ParameterHoverContext';
import { PARAMETER_BOUNDS_OVERRIDES } from '@/constants/parameterBoundsOverrides';
import { PARAMETER_DOC_ANCHORS, buildDocLink } from '@/constants/keyParameterConfigs';

const SLIDER_KEYBOARD_KEYS = new Set([
    'ArrowLeft',
    'ArrowRight',
    'ArrowUp',
    'ArrowDown',
    'Home',
    'End',
    'PageUp',
    'PageDown',
]);


export interface ParameterSliderProps {
    // Core props
    paramName: keyof ParametersType & string;
    label: string;
    description?: string;

    // Slider configuration
    step?: number | string;
    fallbackMin?: number;
    fallbackMax?: number;

    // Override bounds/step (for dynamic values)
    customMin?: number;
    customMax?: number;
    customStep?: number | string;

    // Formatting
    decimalPlaces?: number;
    customFormatValue?: (value: number) => string;

    // State and handlers
    value: number;
    uiParameters: ParametersType;
    setUiParameters: React.Dispatch<React.SetStateAction<ParametersType>>;
    allParameters: ParameterConfig | null;
    isDragging: boolean;
    setIsDragging: React.Dispatch<React.SetStateAction<boolean>>;
    commitParameters: (nextParameters?: ParametersType) => void;

    // Optional event handlers for popovers
    onMouseEnter?: () => void;
    onMouseLeave?: () => void;
    onFocus?: () => void;
    onBlur?: () => void;

    // Popover support
    popoverConfig?: ChartConfig;
    popoverVisible?: boolean;
    onPopoverClose?: () => void;
    horizonParams?: {
        uses_shifted_form: boolean;
        anchor_progress: number | null;
    } | null;

    // Optional styling
    className?: string;
    
    // Disabled state
    disabled?: boolean;
    
    // Log scale
    useLogScale?: boolean;

    // Log-odds scale (for percentile values between 0 and 1)
    useLogOddsScale?: boolean;

    // Step count - divide range into exactly N steps
    stepCount?: number;

    // Whether the rationale tooltip should stay open when hovered (for clicking doc links)
    interactiveRationale?: boolean;

    // Baseline parameters for the tick mark (defaults to DEFAULT_PARAMETERS)
    baselineParameters?: ParametersType;
}

export const ParameterSlider: React.FC<ParameterSliderProps> = ({
    paramName,
    label,
    description,
    step = 0.1,
    fallbackMin = 0,
    fallbackMax = 100,
    customMin,
    customMax,
    customStep,
    decimalPlaces = 2,
    customFormatValue,
    value,
    uiParameters,
    setUiParameters,
    allParameters,
    setIsDragging,
    commitParameters,
    onMouseEnter,
    onMouseLeave,
    onFocus,
    onBlur,
    popoverConfig,
    popoverVisible,
    onPopoverClose,
    horizonParams,
    className,
    disabled = false,
    useLogScale = false,
    useLogOddsScale = false,
    stepCount,
    interactiveRationale = true,
    baselineParameters,
}) => {
    const baseline = baselineParameters ?? DEFAULT_PARAMETERS;
    const { setHoveredParameter } = useParameterHover();
    const [showRationale, setShowRationale] = useState(false);
    const [tooltipPosition, setTooltipPosition] = useState<{ top: number; left: number } | null>(null);
    const containerRef = useRef<HTMLDivElement>(null);
    const tooltipRef = useRef<HTMLDivElement>(null);
    const hideTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
    const anchorId = useId();
    const anchorName = `--param-slider-${anchorId.replace(/:/g, '')}`;

    // Get bounds override if available
    const boundsOverride = PARAMETER_BOUNDS_OVERRIDES[paramName];
    
    // Priority: override > customMin/Max > API bounds > fallback
    const min = boundsOverride?.min ?? customMin ?? (allParameters?.bounds?.[paramName as string]?.[0] ?? fallbackMin);
    const max = boundsOverride?.max ?? customMax ?? (allParameters?.bounds?.[paramName as string]?.[1] ?? fallbackMax);
    const actualStep = customStep ?? step;
    const formatValue = customFormatValue || ((val: number) => val.toFixed(decimalPlaces));
    const handleChange = (newValue: number) => {
        setUiParameters(prev => ({ ...prev, [paramName]: newValue }));
    };
    const displayMin = boundsOverride?.min ?? customMin ?? (allParameters?.bounds?.[paramName as string]?.[0] ?? fallbackMin);
    const displayMax = boundsOverride?.max ?? customMax ?? (allParameters?.bounds?.[paramName as string]?.[1] ?? fallbackMax);
    const fullDescription = description ? `${description} (${displayMin} to ${displayMax})` : undefined;
    const wrapperClass = popoverConfig ? "space-y-0.5 relative" : `${className || "space-y-0.5"} relative`;
    const containerClass = disabled ? `${wrapperClass} opacity-50 pointer-events-none` : wrapperClass;
    const rationale = PARAMETER_RATIONALES[paramName];
    const docAnchor = PARAMETER_DOC_ANCHORS[paramName];
    const docLink = docAnchor ? buildDocLink(docAnchor) : null;
    const isInteractive = interactiveRationale && !!docLink;

    const clearHideTimeout = useCallback(() => {
        if (hideTimeoutRef.current) {
            clearTimeout(hideTimeoutRef.current);
            hideTimeoutRef.current = null;
        }
    }, []);

    const scheduleHide = useCallback(() => {
        clearHideTimeout();
        if (isInteractive) {
            hideTimeoutRef.current = setTimeout(() => {
                setShowRationale(false);
                setHoveredParameter(null);
            }, 150);
        } else {
            setShowRationale(false);
            setHoveredParameter(null);
        }
    }, [isInteractive, clearHideTimeout, setHoveredParameter]);

    const handleShowRationale = useCallback(() => {
        clearHideTimeout();
        setShowRationale(true);
        setHoveredParameter(paramName);
    }, [clearHideTimeout, setHoveredParameter, paramName]);

    // Cleanup timeout on unmount
    useEffect(() => {
        return () => clearHideTimeout();
    }, [clearHideTimeout]);

    // Calculate tooltip position as a fallback for browsers that don't support CSS anchor positioning
    useEffect(() => {
        if (showRationale && containerRef.current) {
            const rect = containerRef.current.getBoundingClientRect();

            // Always position below the slider to avoid overlapping with other sliders above
            setTooltipPosition({
                top: rect.bottom + 4,
                left: rect.left,
            });
        }
    }, [showRationale]);

    const handleMouseEnter = () => {
        handleShowRationale();
        onMouseEnter?.();
    };
    
    const handleMouseLeave = () => {
        scheduleHide();
        onMouseLeave?.();
    };
    
    const handleFocus = () => {
        handleShowRationale();
        onFocus?.();
    };
    
    const handleBlur = () => {
        scheduleHide();
        onBlur?.();
    };

    const handleKeyDown = (event: React.KeyboardEvent<HTMLInputElement>) => {
        if (disabled) {
            return;
        }
        if (SLIDER_KEYBOARD_KEYS.has(event.key)) {
            setIsDragging(true);
        }
    };

    const handleKeyUp = (event: React.KeyboardEvent<HTMLInputElement>) => {
        if (disabled) {
            return;
        }
        if (SLIDER_KEYBOARD_KEYS.has(event.key)) {
            commitParameters();
        }
    };

    const showCustomChartPopover = popoverConfig && popoverVisible && onPopoverClose;
    
    return (
        <div
            ref={containerRef}
            className={containerClass}
            onMouseEnter={handleMouseEnter}
            onMouseLeave={handleMouseLeave}
            onFocus={handleFocus}
            onBlur={handleBlur}
            style={{ anchorName }}
        >
            {label && (
                <label className="block text-[11px] font-medium text-foreground leading-tight">
                    {label}
                    {disabled && <span className="ml-1 text-[10px] text-gray-400">(Locked)</span>}
                </label>
            )}
            <SliderWithThumbValue
                value={value}
                min={min}
                max={max}
                step={actualStep}
                onChange={handleChange}
                onMouseDown={() => !disabled && setIsDragging(true)}
                onMouseUp={() => !disabled && commitParameters()}
                onTouchStart={() => !disabled && setIsDragging(true)}
                onTouchEnd={() => !disabled && commitParameters()}
                onKeyDown={handleKeyDown}
                onKeyUp={handleKeyUp}
                displayValue=""
                formatValue={formatValue}
                useLogScale={useLogScale}
                useLogOddsScale={useLogOddsScale}
                stepCount={stepCount}
                defaultValue={typeof baseline[paramName] === 'number' ? baseline[paramName] : undefined}
            />
            {/* Render SliderPopover conditionally with config */}
            {showCustomChartPopover && (
                <SliderPopover
                    config={popoverConfig}
                    visible={popoverVisible}
                    onClose={onPopoverClose}
                    uiParameters={uiParameters}
                    horizonParams={horizonParams}
                    anchorName={anchorName}
                />
            )}
            {/* Render rationale tooltip on hover - uses CSS anchor positioning with JS fallback */}
            {showRationale && rationale && tooltipPosition && (
                <div
                    ref={tooltipRef}
                    className={`fixed z-[9999] p-3 bg-gray-900 text-white text-xs rounded-lg shadow-lg max-w-xs font-system-mono ${isInteractive ? 'pointer-events-auto' : 'pointer-events-none'}`}
                    style={{
                        top: `${tooltipPosition.top}px`,
                        left: `${tooltipPosition.left}px`,
                    }}
                    onMouseEnter={isInteractive ? handleShowRationale : undefined}
                    onMouseLeave={isInteractive ? scheduleHide : undefined}
                >
                    <div className="font-semibold mb-1">Default Value Rationale:</div>
                    <div className="text-gray-200">{rationale}</div>
                    {isInteractive && docLink && (
                        <a
                            href={docLink}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="mt-2 flex items-center gap-1.5 text-[11px] text-gray-400 hover:text-white transition-colors border-t border-gray-700 pt-2"
                        >
                            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6" />
                                <polyline points="15 3 21 3 21 9" />
                                <line x1="10" y1="14" x2="21" y2="3" />
                            </svg>
                            View detailed rationale
                        </a>
                    )}
                    {!isInteractive && docLink && (
                        <div className="mt-2 pt-2 border-t border-gray-700 text-[11px] text-gray-400">
                            For more on our default value rationale, click the{' '}
                            <svg className="inline-block align-text-bottom" width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6" />
                                <polyline points="15 3 21 3 21 9" />
                                <line x1="10" y1="14" x2="21" y2="3" />
                            </svg>{' '}
                            icon to the right of the parameter name.
                        </div>
                    )}
                </div>
            )}
        </div>
    );
};
