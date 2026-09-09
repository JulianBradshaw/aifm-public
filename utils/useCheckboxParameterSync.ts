import { useEffect, useRef } from 'react';
import { DEFAULT_PARAMETERS, ParameterPrimitive, ParametersType } from '@/constants/parameters';
import { DEFAULT_CHECKBOX_STATES } from '@/utils/urlState';

export interface CheckboxParameterSyncConfig {
    checkboxValue: boolean;
    checkboxKey: keyof typeof DEFAULT_CHECKBOX_STATES;
    /** Values to apply while the checkbox is unchecked. Must be a stable object. */
    nonDefaultParameterValues: Partial<ParametersType>;
    /** The live parameters, so unchecking can remember what to put back. */
    currentParameters: ParametersType;
    setParameters: React.Dispatch<React.SetStateAction<ParametersType>>;
    setUiParameters: React.Dispatch<React.SetStateAction<ParametersType>>;
    isHydratingFromUrlRef: React.MutableRefObject<boolean>;
}

/**
 * Applies a "Model Simplifications" checkbox's parameter overrides while it is
 * unchecked, and puts the previous values back when it is re-checked.
 *
 * Restore must not read DEFAULT_PARAMETERS: that would discard the user's work
 * whenever a box is toggled off and on again on a non-default preset. Snapshot the
 * live values at the moment of unchecking and restore that snapshot, falling back to
 * DEFAULT_PARAMETERS only when there is none.
 */
export function useCheckboxParameterSync({
    checkboxValue,
    checkboxKey,
    nonDefaultParameterValues,
    currentParameters,
    setParameters,
    setUiParameters,
    isHydratingFromUrlRef,
}: CheckboxParameterSyncConfig) {
    // What these parameters held before this checkbox took them over. Null whenever
    // the checkbox is in its default state and owns nothing.
    const savedValuesRef = useRef<Partial<ParametersType> | null>(null);
    // The parameters as of the previous commit. On the commit where the checkbox
    // flips, `currentParameters` still holds the pre-override values -- which is
    // exactly what we want to snapshot.
    const preFlipParamsRef = useRef<ParametersType>(currentParameters);
    // Parameters already carry the right values on first render; syncing on mount
    // would write DEFAULT_PARAMETERS over whichever preset was just loaded.
    const didMount = useRef(false);

    useEffect(() => {
        if (!didMount.current) {
            didMount.current = true;
            return;
        }
        if (isHydratingFromUrlRef.current) {
            return;
        }

        const parameterKeys = Object.keys(nonDefaultParameterValues);
        const isUnchecked = checkboxValue !== DEFAULT_CHECKBOX_STATES[checkboxKey];

        if (isUnchecked) {
            if (savedValuesRef.current === null) {
                const before = preFlipParamsRef.current;
                const snapshot: Record<string, ParameterPrimitive> = {};
                parameterKeys.forEach((key) => {
                    const value = before?.[key] ?? DEFAULT_PARAMETERS[key];
                    if (value !== undefined) {
                        snapshot[key] = value;
                    }
                });
                savedValuesRef.current = snapshot as Partial<ParametersType>;
            }
            const apply = (prev: ParametersType) => ({ ...prev, ...nonDefaultParameterValues } as ParametersType);
            setUiParameters(apply);
            setParameters(apply);
            return;
        }

        const saved = savedValuesRef.current;
        savedValuesRef.current = null;
        const restore = (prev: ParametersType) => {
            const patch: Record<string, ParameterPrimitive> = {};
            parameterKeys.forEach((key) => {
                // Only undo values this checkbox is still responsible for. If a preset
                // load or a slider drag has moved one since, that newer value wins.
                if (prev[key] !== nonDefaultParameterValues[key]) {
                    return;
                }
                const value = saved?.[key] ?? DEFAULT_PARAMETERS[key];
                // Skip keys that had no value to begin with rather than writing an
                // explicit `undefined` into the parameter object.
                if (value !== undefined) {
                    patch[key] = value;
                }
            });
            return { ...prev, ...patch } as ParametersType;
        };
        setUiParameters(restore);
        setParameters(restore);
    }, [checkboxValue, checkboxKey, nonDefaultParameterValues, setParameters, setUiParameters, isHydratingFromUrlRef]);

    // Declared after the effect above on purpose: effects run in declaration order,
    // so on the commit where the checkbox flips that effect still reads the
    // pre-flip parameters out of the ref.
    useEffect(() => {
        preFlipParamsRef.current = currentParameters;
    }, [currentParameters]);
}
