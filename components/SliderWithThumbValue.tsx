import React from 'react';

interface SliderWithThumbValueProps {
  value: number;
  min: number;
  max: number;
  step: string | number;
  onChange: (value: number) => void;
  onMouseDown?: () => void;
  onTouchStart?: () => void;
  onMouseUp?: () => void;
  onTouchEnd?: () => void;
  onKeyDown?: (event: React.KeyboardEvent<HTMLInputElement>) => void;
  onKeyUp?: (event: React.KeyboardEvent<HTMLInputElement>) => void;
  onMouseEnter?: () => void;
  onMouseLeave?: () => void;
  onFocus?: () => void;
  onBlur?: () => void;
  displayValue: string;
  className?: string;
  formatValue?: (value: number) => string;
  useLogScale?: boolean;
  useLogOddsScale?: boolean;
  stepCount?: number;
  defaultValue?: number;
}

export const SliderWithThumbValue: React.FC<SliderWithThumbValueProps> = ({
  value,
  min,
  max,
  step,
  onChange,
  onMouseDown,
  onTouchStart,
  onMouseUp,
  onTouchEnd,
  onKeyDown,
  onKeyUp,
  onMouseEnter,
  onMouseLeave,
  onFocus,
  onBlur,
  displayValue,
  className = "w-full rounded-lg appearance-none cursor-pointer slider",
  formatValue,
  useLogScale = false,
  useLogOddsScale = false,
  stepCount,
  defaultValue,
}) => {
  const canUseLogScale = useLogScale && min > 0 && max > min;
  const canUseLogOddsScale = useLogOddsScale && min > 0 && min < 1 && max > min && max < 1;

  if (useLogScale && !canUseLogScale) {
    console.warn(
      `Log scale requires min > 0 and max > min. Got min=${min}, max=${max}. Falling back to linear scale.`
    );
  }

  if (useLogOddsScale && !canUseLogOddsScale) {
    console.warn(
      `Log-odds scale requires 0 < min < max < 1. Got min=${min}, max=${max}. Falling back to linear scale.`
    );
  }

  const effectiveUseLogScale = canUseLogScale;
  const effectiveUseLogOddsScale = canUseLogOddsScale;
  const useStepCount = stepCount !== undefined && stepCount > 0;

  // Log-odds (logit) transformation: log(p / (1-p))
  const logit = (p: number): number => Math.log(p / (1 - p));
  // Inverse logit (sigmoid): exp(x) / (1 + exp(x))
  const invLogit = (x: number): number => Math.exp(x) / (1 + Math.exp(x));

  const logitMin = effectiveUseLogOddsScale ? logit(min) : 0;
  const logitMax = effectiveUseLogOddsScale ? logit(max) : 0;

  const valueToPosition = (val: number): number => {
    if (useStepCount) {
      if (effectiveUseLogScale) {
        return stepCount * Math.log(val / min) / Math.log(max / min);
      } else if (effectiveUseLogOddsScale) {
        return stepCount * (logit(val) - logitMin) / (logitMax - logitMin);
      } else {
        return stepCount * (val - min) / (max - min);
      }
    }
    if (effectiveUseLogOddsScale) {
      return 100 * (logit(val) - logitMin) / (logitMax - logitMin);
    }
    if (!effectiveUseLogScale) return val;
    return 100 * Math.log(val / min) / Math.log(max / min);
  };

  const positionToValue = (pos: number): number => {
    if (useStepCount) {
      if (effectiveUseLogScale) {
        return min * Math.pow(max / min, pos / stepCount);
      } else if (effectiveUseLogOddsScale) {
        return invLogit(logitMin + (pos / stepCount) * (logitMax - logitMin));
      } else {
        return min + pos * (max - min) / stepCount;
      }
    }
    if (effectiveUseLogOddsScale) {
      return invLogit(logitMin + (pos / 100) * (logitMax - logitMin));
    }
    if (!effectiveUseLogScale) return pos;
    return min * Math.pow(max / min, pos / 100);
  };

  // Guard against undefined/NaN values
  const safeValue = typeof value === 'number' && !isNaN(value) ? value : min;
  const safeMin = typeof min === 'number' && !isNaN(min) ? min : 0;
  const safeMax = typeof max === 'number' && !isNaN(max) ? max : 100;

  const sliderMin = useStepCount ? 0 : (effectiveUseLogScale || effectiveUseLogOddsScale ? 0 : safeMin);
  const sliderMax = useStepCount ? stepCount : (effectiveUseLogScale || effectiveUseLogOddsScale ? 100 : safeMax);
  const sliderValue = (useStepCount || effectiveUseLogScale || effectiveUseLogOddsScale) ? valueToPosition(safeValue) : safeValue;
  const sliderStep = useStepCount ? 1 : (effectiveUseLogScale || effectiveUseLogOddsScale ? 0.1 : step);

  const rawPercentage = useStepCount
    ? (sliderValue / stepCount) * 100
    : (effectiveUseLogScale || effectiveUseLogOddsScale
      ? sliderValue
      : ((safeValue - safeMin) / (safeMax - safeMin)) * 100);

  const percentage = typeof rawPercentage === 'number' && !isNaN(rawPercentage)
    ? Number(rawPercentage.toFixed(6))
    : 0;

  const currentDisplayValue = formatValue ? formatValue(safeValue) : displayValue;

  // Calculate default value tick position
  const defaultTickPercentage = (() => {
    if (defaultValue === undefined || defaultValue < min || defaultValue > max) return null;
    if (useStepCount) {
      if (effectiveUseLogScale) {
        return (Math.log(defaultValue / min) / Math.log(max / min)) * 100;
      } else if (effectiveUseLogOddsScale) {
        return ((logit(defaultValue) - logitMin) / (logitMax - logitMin)) * 100;
      } else {
        return ((defaultValue - min) / (max - min)) * 100;
      }
    }
    if (effectiveUseLogOddsScale) {
      return ((logit(defaultValue) - logitMin) / (logitMax - logitMin)) * 100;
    }
    if (effectiveUseLogScale) {
      return (Math.log(defaultValue / min) / Math.log(max / min)) * 100;
    }
    return ((defaultValue - safeMin) / (safeMax - safeMin)) * 100;
  })();

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const rawValue = parseFloat(e.target.value);
    const actualValue = (useStepCount || effectiveUseLogScale || effectiveUseLogOddsScale) ? positionToValue(rawValue) : rawValue;
    onChange(actualValue);
  };

  // Calculate label positioning to prevent overflow
  // At 0%, anchor to left edge; at 100%, anchor to right edge; otherwise center
  const getLabelStyle = (): React.CSSProperties => {
    if (percentage <= 15) {
      // Near left edge - anchor to left
      return { 
        left: `${percentage}%`, 
        transform: 'translateX(0)',
        whiteSpace: 'nowrap' 
      };
    } else if (percentage >= 85) {
      // Near right edge - anchor to right
      return { 
        left: `${percentage}%`, 
        transform: 'translateX(-100%)',
        whiteSpace: 'nowrap' 
      };
    } else {
      // Middle - center
      return { 
        left: `${percentage}%`, 
        transform: 'translateX(-50%)',
        whiteSpace: 'nowrap' 
      };
    }
  };

  return (
    <div className="flex flex-col">
      <div className="relative overflow-visible">
        {/* Default value tick mark - rendered before input so it's behind the track */}
        {/* Only show when the current value differs from the default */}
        {defaultTickPercentage !== null && value !== defaultValue && (
          <div
            className="absolute w-px bg-gray-400/70 pointer-events-none"
            style={{
              // Account for the thumb width (0.5rem = 8px) that browsers reserve on each side
              left: `calc(${defaultTickPercentage}% * (100% - 8px) / 100% + 4px)`,
              top: '9px',
              height: '14px',
            }}
          />
        )}
        <input
          type="range"
          min={sliderMin}
          max={sliderMax}
          step={sliderStep}
          value={sliderValue}
          onChange={handleChange}
          onMouseDown={onMouseDown}
          onTouchStart={onTouchStart}
          onMouseUp={onMouseUp}
          onTouchEnd={onTouchEnd}
          onKeyDown={onKeyDown}
          onKeyUp={onKeyUp}
          onMouseEnter={onMouseEnter}
          onMouseLeave={onMouseLeave}
          onFocus={onFocus}
          onBlur={onBlur}
          className={className}
        />
        <div
          className="absolute top-full mt-0.5 text-[9px] text-black/50 font-medium pointer-events-none"
          style={getLabelStyle()}
        >
          {currentDisplayValue}
        </div>
      </div>
      {/* Spacer to reserve space for the absolutely positioned value label */}
      <div className="h-3.5" />
    </div>
  );
};
