import { ParametersType } from '@/constants/parameters';

/**
 * Parameter values applied when a "Model Simplifications" checkbox is UNCHECKED.
 *
 * Single source of truth: both /playground and the main progress chart import this,
 * so the two explorers cannot drift apart. The objects are module-level constants
 * so their identity is stable across renders -- `useCheckboxParameterSync` keeps
 * them in its effect dependencies, and a fresh object literal each render would
 * re-fire the effect continuously.
 *
 * Every value here must stay inside the parameter's range in
 * `progress_model.config.PARAMETER_BOUNDS`. Those bounds are not enforced on the model path
 * (they drive sliders and Monte Carlo sampling only), so an out-of-range value does
 * not error -- it quietly runs the model in a regime nothing calibrated or tested,
 * which surfaces as empty or 400 responses rather than as a validation failure.
 */
export const CHECKBOX_PARAM_OVERRIDES = {
    enableExperimentAutomation: {
        // 1.01 is the lower bound of median_to_top_taste_multiplier, and going below it
        // is not safe: the model enters a degenerate regime where SAR, SIAR, ASI, TED-AI
        // and STRAT-AI never resolve and the chart silently loses those milestones. At
        // the bound the median-to-top taste gap is a negligible 1% (so taste automation
        // is still effectively switched off) while every milestone continues to resolve.
        median_to_top_taste_multiplier: 1.01,
    },
    useExperimentThroughputCES: {
        // rho_experiment_capacity: 0 is what actually selects the Cobb-Douglas branch
        // (|rho| < RHO_COBB_DOUGLAS_THRESHOLD in progress_model/ces_functions.py).
        //
        // direct_input_exp_cap_ces_params: true makes the model read rho/alpha/exponent
        // straight off the parameters instead of deriving them from the asymptotes
        // (progress_model/_impl.py). That is also why inf_labor_asymptote,
        // inf_compute_asymptote and inv_compute_anchor_exp_cap are NOT listed here:
        // this branch never reads them, so the huge sentinel values this override used
        // to carry were provably inert -- swapping them for the defaults produced
        // byte-identical trajectories.
        //
        // parallel_penalty is likewise absent: convertParametersToAPIFormat overwrites
        // it from coding_labor_exponent on every request, so setting it here never
        // reached the model.
        direct_input_exp_cap_ces_params: true,
        rho_experiment_capacity: 0,
    },
    enableSoftwareProgress: {
        software_progress_rate_at_reference_year: 0,
    },
} as const satisfies Record<string, Partial<ParametersType>>;

/**
 * Parameters each checkbox takes control of while unchecked, and which the UI
 * therefore locks. Kept next to the overrides so a parameter cannot be silently
 * overridden without also being locked -- the mismatch the old
 * `useExperimentThroughputCES` entry had, where it moved alpha_experiment_capacity
 * and experiment_compute_exponent but left both sliders live and editable.
 */
export const CHECKBOX_LOCKED_PARAMETERS: Record<keyof typeof CHECKBOX_PARAM_OVERRIDES, readonly string[]> = {
    enableExperimentAutomation: ['median_to_top_taste_multiplier'],
    useExperimentThroughputCES: ['direct_input_exp_cap_ces_params', 'rho_experiment_capacity'],
    enableSoftwareProgress: ['software_progress_rate_at_reference_year'],
};
