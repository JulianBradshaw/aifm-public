const SUPERSCRIPT_MAP: Record<string, string> = {
    '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴',
    '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹',
    '-': '⁻', '+': '⁺'
};

// Rounds an exponent to 2 significant figures for display purposes
// E.g., 1234 → 1200, 56 → 56, 7 → 7
function roundPowerOfTenExponent(value: number): number {
    if (!Number.isFinite(value) || value === 0) {
        return 0;
    }

    const absVal = Math.abs(value);

    if (absVal < 10) {
        return Math.round(value);
    }

    // Round to 2 significant figures
    // E.g., for 1234: digits=4, roundingPrecision=10^2=100, rounded=1200
    const digits = Math.floor(Math.log10(absVal)) + 1;
    const roundingPrecision = Math.pow(10, Math.max(digits - 2, 0));
    const rounded = Math.round(value / roundingPrecision) * roundingPrecision;

    return rounded;
}

export type FormatPowerOfTenNodeOptions = {
    suffix?: React.ReactNode;
    mantissaOpacity?: number;
    mantissaFontSize?: string | number;
    exponentFontSize?: string | number;
    renderMode?: 'html' | 'svg';
    baselineShift?: `0.${number}em`;
};

export function formatPowerOfTenNode(value: number, options: FormatPowerOfTenNodeOptions = {}): React.ReactNode {
    if (!Number.isFinite(value)) {
        return null;
    }

    const {
        suffix,
        mantissaOpacity = 0.7,
        mantissaFontSize = '0.92em',
        exponentFontSize,
        renderMode = 'html',
        baselineShift = '0.2em',
    } = options;

    const exponent = roundPowerOfTenExponent(value);

    if (exponent === 0) {
        if (renderMode === 'svg') {
            return (
                <>
                    <tspan>1</tspan>
                    {suffix != null ? <tspan>{suffix}</tspan> : null}
                </>
            );
        }

        return (
            <span>
                1
                {suffix}
            </span>
        );
    }

    if (renderMode === 'svg') {
        return (
            <>
                <tspan fillOpacity={mantissaOpacity} fontSize={mantissaFontSize}>10</tspan>
                <tspan baselineShift={baselineShift} fontSize={exponentFontSize}>{exponent}</tspan>
                {suffix != null ? <tspan>{suffix}</tspan> : null}
            </>
        );
    }

    return (
        <span>
            <span style={{ opacity: mantissaOpacity, fontSize: mantissaFontSize }}>10</span>
            <sup style={exponentFontSize ? { fontSize: exponentFontSize } : undefined}>{exponent}</sup>
            {suffix}
        </span>
    );
}

export function formatPowerOfTenText(value: number, options: { suffix?: string } = {}): string {
    if (!Number.isFinite(value)) {
        return '';
    }

    const exponent = roundPowerOfTenExponent(value);

    if (exponent === 0) {
        return `1${options.suffix ?? ''}`;
    }

    const exponentStr = exponent.toString();
    const superscript = exponentStr.split('').map(ch => SUPERSCRIPT_MAP[ch] ?? ch).join('');

    return `10${superscript}${options.suffix ?? ''}`;
}

export function formatAsPowerOfTenNode(value: number, options: FormatPowerOfTenNodeOptions = {}): React.ReactNode {
    if (!Number.isFinite(value) || value <= 0) {
        return null;
    }

    return formatPowerOfTenNode(Math.log10(value), options);
}

export function formatAsPowerOfTenText(value: number, options: { suffix?: string } = {}): string {
    if (!Number.isFinite(value) || value <= 0) {
        return '';
    }

    return formatPowerOfTenText(Math.log10(value), options);
}

export function formatTo3SigFigs(value: number): string {
    if (value === 0) return '0';

    // Use formatCompactNumber with 3 significant figures for consistency
    return formatCompactNumber(value, 3);
}

// Compact dollar amounts with K/M/B/T units (e.g. $10B/yr, $1.5T/yr, $100T/yr).
export function formatDollarsPerYear(value: number): string {
    if (!Number.isFinite(value)) return '';
    if (value < 0) return `-${formatDollarsPerYear(-value)}`;
    const units: Array<[number, string]> = [[1e12, 'T'], [1e9, 'B'], [1e6, 'M'], [1e3, 'K']];
    for (const [unit, suffix] of units) {
        if (value >= unit) {
            const scaled = value / unit;
            const str = scaled >= 10 ? scaled.toFixed(0) : String(parseFloat(scaled.toFixed(1)));
            return `$${str}${suffix}/yr`;
        }
    }
    return `$${value.toFixed(0)}/yr`;
}

export function yearsToMinutes(years: number): number {
    return years * 525600; // 365 days * 24 hours * 60 minutes
}

function formatUnitValue(value: number, unit: string, fractionDigits: number): string {
    const fixed = value.toFixed(fractionDigits);
    const trimmed = fixed.includes('.')
        ? fixed.replace(/(\.\d*?[1-9])0+$/, '$1').replace(/\.0+$/, '')
        : fixed;
    return `${trimmed} ${unit}`;
}

export function formatTimeDuration(minutes: number): string {
    const MINUTES_PER_MONTH = (60 * 24 * 365.25) / 12; // 43830
    if (minutes < 0.0167) {
        return formatUnitValue(minutes * 60, 'sec', 2);
    } else if (minutes < 1) {
        const seconds = minutes * 60;
        return Math.abs(seconds - 1) < 0.01 ? '1 sec' : formatUnitValue(seconds, 'sec', 1);
    } else if (minutes < 60) {
        return formatUnitValue(minutes, 'min', 1);
    } else if (minutes < 1440) {
        return formatUnitValue(minutes / 60, 'hr', 1);
    } else if (minutes < MINUTES_PER_MONTH) {
        return formatUnitValue(minutes / 1440, 'days', 1);
    } else if (minutes < 525600) {
        return formatUnitValue(minutes / MINUTES_PER_MONTH, 'months', 1);
    } else {
        return formatUnitValue(minutes / 525600, 'years', 1);
    }
}

export function formatWorkTimeDuration(minutes: number): string {
    const WORK_DAY_MINUTES = 8 * 60;              // 480 minutes
    const WORK_WEEK_MINUTES = 5 * WORK_DAY_MINUTES; // 2400 minutes (5 work days)
    const WORK_MONTH_MINUTES = (2000 / 12) * 60;  // ~10000 minutes
    const WORK_YEAR_MINUTES = 2000 * 60;          // 120000 minutes

    if (minutes < 0.0167) {
        return formatUnitValue(minutes * 60, 'sec', 2);
    } else if (minutes < 1) {
        const seconds = minutes * 60;
        return Math.abs(seconds - 1) < 0.01 ? '1 sec' : formatUnitValue(seconds, 'sec', 1);
    } else if (minutes < 60) {
        return formatUnitValue(minutes, 'min', 1);
    } else if (minutes < WORK_DAY_MINUTES) {
        return formatUnitValue(minutes / 60, 'hr', 1);
    } else if (minutes < WORK_WEEK_MINUTES) {
        return formatUnitValue(minutes / WORK_DAY_MINUTES, 'work days', 1);
    } else if (minutes < WORK_MONTH_MINUTES) {
        return formatUnitValue(minutes / WORK_WEEK_MINUTES, 'work weeks', 1);
    } else if (minutes < WORK_YEAR_MINUTES) {
        return formatUnitValue(minutes / WORK_MONTH_MINUTES, 'work months', 1);
    } else {
        const years = minutes / WORK_YEAR_MINUTES;
        // Use compact number formatting for years (handles K, M, B, scientific notation)
        return `${formatCompactNumber(years)} work years`;
    }
}

export function formatSCHorizon(logValue: number): string {
    const minutes = Math.pow(10, logValue);
    return formatWorkTimeDuration(minutes).replace(/\.0\s/, ' '); // Remove .0 for cleaner display
}

export function formatTimeDurationDetailed(minutes: number): string {
    const MINUTES_PER_YEAR = 525600;
    const MINUTES_PER_MONTH = 43800;
    const MINUTES_PER_DAY = 1440;
    const MINUTES_PER_HOUR = 60;

    const formatWithUnit = (value: number, unit: string): string => {
        const floored = Math.floor(value * 10) / 10;
        return `${floored.toFixed(1)} ${unit}`;
    };

    // Handle extremely large values (more than 10 billion years)
    if (minutes >= MINUTES_PER_YEAR * 1e10) {
        const years = minutes / MINUTES_PER_YEAR;
        // Use scientific notation for extremely large year values
        return `${formatTo3SigFigs(years)} years`;
    }

    if (minutes >= MINUTES_PER_YEAR) {
        const years = minutes / MINUTES_PER_YEAR;
        return formatWithUnit(years, 'years');
    }

    if (minutes >= MINUTES_PER_MONTH) {
        const months = minutes / MINUTES_PER_MONTH;
        return formatWithUnit(months, 'months');
    }

    if (minutes >= MINUTES_PER_DAY) {
        const days = minutes / MINUTES_PER_DAY;
        return formatWithUnit(days, 'days');
    }

    if (minutes >= MINUTES_PER_HOUR) {
        const hours = minutes / MINUTES_PER_HOUR;
        return formatWithUnit(hours, 'hours');
    }

    if (minutes >= 1) {
        return formatWithUnit(minutes, 'min');
    }

    // Handle sub-minute durations
    const seconds = minutes * 60;
    return formatWithUnit(seconds, 'sec');
}

export function formatWorkTimeDurationDetailed(minutes: number): string {
    const MINUTES_PER_WORK_YEAR = 2000 * 60;
    const MINUTES_PER_WORK_MONTH = (2000 / 12) * 60;
    const MINUTES_PER_WORK_DAY = 8 * 60;
    const MINUTES_PER_HOUR = 60;

    const formatWithUnit = (value: number, unit: string): string => {
        const floored = Math.floor(value * 10) / 10;
        return `${floored.toFixed(1)} ${unit}`;
    };

    if (minutes >= MINUTES_PER_WORK_YEAR) {
        const years = minutes / MINUTES_PER_WORK_YEAR;
        // Use compact number formatting for years (handles K, M, B, scientific notation)
        return `${formatCompactNumber(years)} years`;
    }

    if (minutes >= MINUTES_PER_WORK_MONTH) {
        const months = minutes / MINUTES_PER_WORK_MONTH;
        return formatWithUnit(months, 'months');
    }

    if (minutes >= MINUTES_PER_WORK_DAY) {
        const days = minutes / MINUTES_PER_WORK_DAY;
        return formatWithUnit(days, 'days');
    }

    if (minutes >= MINUTES_PER_HOUR) {
        const hours = minutes / MINUTES_PER_HOUR;
        return formatWithUnit(hours, 'hours');
    }

    if (minutes >= 1) {
        return formatWithUnit(minutes, 'min');
    }

    // Handle sub-minute durations
    const seconds = minutes * 60;
    return formatWithUnit(seconds, 'sec');
}


export function formatLogWorkTick(tickItem: number, workTimeLabel?: boolean): string {
    const WORK_DAY_MINUTES = 8 * 60;              // 480 minutes
    const WORK_WEEK_MINUTES = 5 * WORK_DAY_MINUTES; // 2400 minutes (5 work days)
    const WORK_MONTH_MINUTES = (2000 / 12) * 60;  // ~10000 minutes
    const WORK_YEAR_MINUTES = 2000 * 60;          // 120000 minutes
    const WORK_MILLENNIUM_MINUTES = WORK_YEAR_MINUTES * 1000;

    // Convert minutes to appropriate unit with 3 sig figs
    if (tickItem < 1) {
        const seconds = tickItem * 60;
        return `${formatTo3SigFigs(seconds)} sec`;
    } else if (tickItem < 60) {
        return `${formatTo3SigFigs(tickItem)} min`;
    } else if (tickItem < WORK_DAY_MINUTES) {
        const hours = tickItem / 60;
        return `${formatTo3SigFigs(hours)} hr`;
    } else if (tickItem < WORK_WEEK_MINUTES) {
        const days = tickItem / WORK_DAY_MINUTES;
        return `${formatTo3SigFigs(days)} ${workTimeLabel ? 'work days' : 'days'}`;
    } else if (tickItem < WORK_MONTH_MINUTES) {
        const weeks = tickItem / WORK_WEEK_MINUTES;
        return `${formatTo3SigFigs(weeks)} ${workTimeLabel ? 'work weeks' : 'weeks'}`;
    } else if (tickItem < WORK_YEAR_MINUTES) {
        const months = tickItem / WORK_MONTH_MINUTES;
        return `${formatTo3SigFigs(months)} ${workTimeLabel ? 'work months' : 'months'}`;
    } else if (tickItem < WORK_MILLENNIUM_MINUTES) {
        const years = tickItem / WORK_YEAR_MINUTES;
        return `${formatTo3SigFigs(years)} ${workTimeLabel ? 'work years' : 'years'}`;
    } else {
        const millennia = tickItem / WORK_MILLENNIUM_MINUTES;
        return `${formatTo3SigFigs(millennia)} ${workTimeLabel ? 'work millennia' : 'millennia'}`;
    }
}

export function formatEffectiveComputeValue(value: number): string {
    // value is a log10 exponent (e.g., 25 means 10^25)
    // Format as compact number with the actual value
    if (!Number.isFinite(value)) {
        return '';
    }
    return formatCompactNumber(Math.pow(10, value));
}

export function formatOOMSuperscript(value: number): React.ReactNode {
    return formatPowerOfTenNode(value, { suffix: ' FLOPs' });
}

export function formatOOMSuperscriptText(value: number): string {
    // value is a log10 exponent (e.g., 25 means 10^25)
    if (!Number.isFinite(value)) {
        return '';
    }
    return formatCompactNumber(Math.pow(10, value));
}

export function formatOOMSuperscriptNode(value: number): React.ReactNode {
    return formatPowerOfTenNode(value);
}

/**
 * Render an effective-compute level (given as log10 OOMs of FLOP) as "10^X.X FLOP"
 * with a superscript exponent. Unlike formatPowerOfTenNode, this keeps one decimal
 * place of the exponent — milestones can sit ~1 OOM (10x) apart, so rounding the
 * exponent to an integer would erase the distinction we are trying to show.
 */
export function formatEffectiveComputeOOMsNode(
    value: number | null | undefined,
    options: { suffix?: string; decimals?: number; mantissaOpacity?: number } = {}
): React.ReactNode {
    if (value == null || !Number.isFinite(value)) {
        return null;
    }
    const { suffix = ' FLOP', decimals = 1, mantissaOpacity = 0.6 } = options;
    return (
        <span style={{ whiteSpace: 'nowrap' }}>
            <span style={{ opacity: mantissaOpacity }}>10</span>
            <sup>{value.toFixed(decimals)}</sup>
            {suffix}
        </span>
    );
}

export function formatYearMonth(decimalYear: number): string {
    const year = Math.floor(decimalYear);
    const fraction = decimalYear - year;
    const monthIndexRaw = Math.round(fraction * 12);

    // Handle edge case where rounding gives us month 12 (should roll to next year)
    const monthIndex = monthIndexRaw % 12;
    const adjustedYear = monthIndexRaw === 12 ? year + 1 : year;

    const months = [
        'January', 'February', 'March', 'April', 'May', 'June',
        'July', 'August', 'September', 'October', 'November', 'December'
    ];

    return `${months[monthIndex]} ${adjustedYear}`;
}

export function formatYearMonthShort(decimalYear: number): string {
    if (!Number.isFinite(decimalYear)) {
        return '';
    }

    const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

    const baseYear = Math.floor(decimalYear);
    const fraction = decimalYear - baseYear;
    const rawMonths = Math.round(fraction * 12);
    const yearAdjustment = Math.floor(rawMonths / 12);
    const normalizedMonthIndex = ((rawMonths % 12) + 12) % 12;
    const year = baseYear + yearAdjustment;

    const monthLabel = months[normalizedMonthIndex];

    return `${monthLabel} ${year}`;
}

export function formatUplift(value: number): string {
    if (!Number.isFinite(value) || value <= 0) {
        return '';
    }

    // Round to 2 significant figures
    const orderOfMagnitude = Math.floor(Math.log10(value));
    const factor = Math.pow(10, orderOfMagnitude - 1);
    const rounded = Math.round(value / factor) * factor;

    // Format based on size
    if (rounded >= 1e9) {
        const billions = rounded / 1e9;
        const formatted = billions >= 10 ? Math.round(billions).toString() : billions.toFixed(1).replace(/\.0$/, '');
        return `${formatted} billion`;
    } else if (rounded >= 1e6) {
        const millions = rounded / 1e6;
        const formatted = millions >= 10 ? Math.round(millions).toString() : millions.toFixed(1).replace(/\.0$/, '');
        return `${formatted} million`;
    } else if (rounded >= 100) {
        return Math.round(rounded).toString();
    } else if (rounded >= 10) {
        return rounded.toFixed(1).replace(/\.0$/, '');
    } else {
        return rounded.toFixed(1);
    }
}

export type FormatCompactNumberNodeOptions = {
    suffix?: React.ReactNode;
    mantissaOpacity?: number;
    mantissaFontSize?: string | number;
    exponentFontSize?: string | number;
    renderMode?: 'html' | 'svg';
    baselineShift?: `0.${number}em`;
    sigFigs?: number;
};

export function formatCompactNumberNode(value: number, options: FormatCompactNumberNodeOptions = {}): React.ReactNode {
    const {
        suffix,
        mantissaOpacity = 0.5,
        mantissaFontSize,
        exponentFontSize,
        renderMode = 'svg',
        baselineShift = '0.3em',
        sigFigs = 2,
    } = options;

    if (!Number.isFinite(value)) {
        return null;
    }

    if (value === 0) {
        if (renderMode === 'svg') {
            return <><tspan>0</tspan>{suffix != null ? <tspan>{suffix}</tspan> : null}</>;
        }
        return <span>0{suffix}</span>;
    }

    const absValue = Math.abs(value);
    const sign = value < 0 ? '-' : '';

    // Round to specified significant figures
    // E.g., for 1234 with sigFigs=2: orderOfMagnitude=3, roundingPrecision=10^2=100, rounded=1200
    const orderOfMagnitude = Math.floor(Math.log10(absValue));
    const roundingPrecision = Math.pow(10, orderOfMagnitude - (sigFigs - 1));
    const rounded = Math.round(absValue / roundingPrecision) * roundingPrecision;

    // Helper to render simple text values
    const renderSimple = (text: string) => {
        if (renderMode === 'svg') {
            return <><tspan>{text}</tspan>{suffix != null ? <tspan>{suffix}</tspan> : null}</>;
        }
        return <span>{text}{suffix}</span>;
    };

    // Helper to render scientific notation with styled mantissa/exponent
    const renderScientific = (mantissaNum: number, exponent: number) => {
        if (renderMode === 'svg') {
            return (
                <>
                    {mantissaNum !== 1 && <tspan>{sign}{mantissaNum}×</tspan>}
                    {mantissaNum === 1 && sign && <tspan>{sign}</tspan>}
                    <tspan fillOpacity={mantissaOpacity} fontSize={mantissaFontSize}>10</tspan>
                    <tspan baselineShift={baselineShift} fontSize={exponentFontSize}>{exponent}</tspan>
                    {suffix != null ? <tspan>{suffix}</tspan> : null}
                </>
            );
        }
        return (
            <span>
                {mantissaNum !== 1 && <>{sign}{mantissaNum}×</>}
                {mantissaNum === 1 && sign}
                <span style={{ opacity: mantissaOpacity, fontSize: mantissaFontSize }}>10</span>
                <sup style={exponentFontSize ? { fontSize: exponentFontSize } : undefined}>{exponent}</sup>
                {suffix}
            </span>
        );
    };

    // For values >= 10^12, use scientific notation
    if (rounded >= 1e12) {
        const exponent = Math.floor(Math.log10(rounded));
        const mantissa = rounded / Math.pow(10, exponent);
        const mantissaRounded = Math.round(mantissa);
        return renderScientific(mantissaRounded, exponent);
    } else if (rounded >= 1e9) {
        const billions = rounded / 1e9;
        const formatted = billions >= 10 ? Math.round(billions).toString() : billions.toFixed(1).replace(/\.0$/, '');
        return renderSimple(`${sign}${formatted}B`);
    } else if (rounded >= 1e6) {
        const millions = rounded / 1e6;
        const formatted = millions >= 10 ? Math.round(millions).toString() : millions.toFixed(1).replace(/\.0$/, '');
        return renderSimple(`${sign}${formatted}M`);
    } else if (rounded >= 1e3) {
        const thousands = rounded / 1e3;
        const formatted = thousands >= 10 ? Math.round(thousands).toString() : thousands.toFixed(1).replace(/\.0$/, '');
        return renderSimple(`${sign}${formatted}K`);
    } else if (rounded >= 1) {
        let text: string;
        if (rounded >= 100) {
            text = sign + Math.round(rounded).toString();
        } else if (rounded >= 10) {
            text = sign + rounded.toFixed(1).replace(/\.0$/, '');
        } else {
            text = sign + rounded.toFixed(sigFigs - 1).replace(/\.?0+$/, '');
        }
        return renderSimple(text);
    } else if (rounded >= 0.001) {
        return renderSimple(sign + rounded.toPrecision(sigFigs).replace(/\.?0+$/, ''));
    } else {
        // Very small numbers: use scientific notation
        const exponent = Math.floor(Math.log10(rounded));
        const mantissa = rounded / Math.pow(10, exponent);
        const mantissaRounded = Math.round(mantissa);
        return renderScientific(mantissaRounded, exponent);
    }
}

export function formatCompactNumber(value: number, sigFigs: number = 2): string {
    if (!Number.isFinite(value)) {
        return '';
    }

    if (value === 0) {
        return '0';
    }

    const absValue = Math.abs(value);
    const sign = value < 0 ? '-' : '';

    // Round to specified significant figures
    // E.g., for 1234 with sigFigs=2: orderOfMagnitude=3, roundingPrecision=10^2=100, rounded=1200
    const orderOfMagnitude = Math.floor(Math.log10(absValue));
    const roundingPrecision = Math.pow(10, orderOfMagnitude - (sigFigs - 1));
    const rounded = Math.round(absValue / roundingPrecision) * roundingPrecision;

    // E.g., 5e12 → "5×10¹²", 1e15 → "10¹⁵"
    if (rounded >= 1e12) {
        const exponent = Math.floor(Math.log10(rounded));
        const mantissa = rounded / Math.pow(10, exponent);
        const mantissaRounded = Math.round(mantissa);
        const exponentStr = exponent.toString();
        const superscript = exponentStr.split('').map(ch => SUPERSCRIPT_MAP[ch] ?? ch).join('');

        if (mantissaRounded === 1) {
            return `${sign}10${superscript}`;
        }
        return `${sign}${mantissaRounded}×10${superscript}`;
    // E.g., 1.5e9 → "1.5B", 25e9 → "25B"
    } else if (rounded >= 1e9) {
        const billions = rounded / 1e9;
        const formatted = billions >= 10 ? Math.round(billions).toString() : billions.toFixed(1).replace(/\.0$/, '');
        return `${sign}${formatted}B`;
    // E.g., 3.2e6 → "3.2M", 50e6 → "50M"
    } else if (rounded >= 1e6) {
        const millions = rounded / 1e6;
        const formatted = millions >= 10 ? Math.round(millions).toString() : millions.toFixed(1).replace(/\.0$/, '');
        return `${sign}${formatted}M`;
    // E.g., 4.5e3 → "4.5K", 12e3 → "12K"
    } else if (rounded >= 1e3) {
        const thousands = rounded / 1e3;
        const formatted = thousands >= 10 ? Math.round(thousands).toString() : thousands.toFixed(1).replace(/\.0$/, '');
        return `${sign}${formatted}K`;
    // E.g., 456 → "456", 23 → "23", 5.6 → "5.6"
    } else if (rounded >= 1) {
        if (rounded >= 100) {
            return sign + Math.round(rounded).toString();
        } else if (rounded >= 10) {
            return sign + rounded.toFixed(1).replace(/\.0$/, '');
        } else {
            return sign + rounded.toFixed(sigFigs - 1).replace(/\.?0+$/, '');
        }
    // E.g., 0.056 → "0.056", 0.0012 → "0.0012"
    } else if (rounded >= 0.001) {
        return sign + rounded.toPrecision(sigFigs).replace(/\.?0+$/, '');
    // E.g., 2.5e-8 → "3×10⁻⁸", 1e-5 → "10⁻⁵"
    } else {
        const exponent = Math.floor(Math.log10(rounded));
        const mantissa = rounded / Math.pow(10, exponent);
        const mantissaRounded = Math.round(mantissa);
        const exponentStr = exponent.toString();
        const superscript = exponentStr.split('').map(ch => SUPERSCRIPT_MAP[ch] ?? ch).join('');

        if (mantissaRounded === 1) {
            return `${sign}10${superscript}`;
        }
        return `${sign}${mantissaRounded}×10${superscript}`;
    }
}
