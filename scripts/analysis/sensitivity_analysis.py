#!/usr/bin/env python3
"""
Sensitivity analysis for batch rollout outputs.

Reads a run directory produced by scripts/batch_rollout.py (or a direct path to
rollouts.jsonl), extracts a target (default `aa_time` or a milestone transition
duration if `--transition-pair FROM:TO` is provided) and input parameters, and
reports how each parameter relates to the target via:

- Pearson and Spearman correlations for numeric parameters
- One-way ANOVA and eta-squared effect size for categorical parameters
- Model-based permutation importance using a linear regression with one-hot
  encoding for categorical variables (no external ML dependencies required)

Usage examples:
  python scripts/sensitivity_analysis.py --run-dir outputs/20250813_020347
  python scripts/sensitivity_analysis.py --rollouts outputs/20250813_020347/rollouts.jsonl

Optional:
  --out-json outputs/20250813_020347/sensitivity_summary.json
  --plot  # save bar plots into the run directory
"""

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml
from scipy import stats

# Use a non-interactive backend in case this runs on a headless server
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Import plotting utilities and RolloutsReader
sys.path.insert(0, str(Path(__file__).parent.parent))

# Apply website-consistent styling (off-white background, monospace fonts)
from plotting_utils.style import apply_style
apply_style()

from plotting.scatter import plot_correlation_scatter
from plotting_utils.rollouts_reader import RolloutsReader


def _is_numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and np.isfinite(value)


def _map_include_gap_to_binary(values: List[Any]) -> np.ndarray:
    """
    Map include_gap values to binary for correlation purposes:
    - "gap" (and common truthy variants) -> 1.0
    - "no gap" (and common falsy variants) -> 0.0
    Anything else -> NaN
    """
    mapped = np.full(len(values), np.nan, dtype=float)
    for i, v in enumerate(values):
        if v is None:
            continue
        if isinstance(v, str):
            s = v.strip().lower()
            if s in ("gap", "yes", "true", "1"):
                mapped[i] = 1.0
            elif s in ("no gap", "no", "false", "0"):
                mapped[i] = 0.0
        elif isinstance(v, bool):
            mapped[i] = 1.0 if v else 0.0
        elif isinstance(v, (int, float)):
            mapped[i] = 1.0 if float(v) != 0.0 else 0.0
    return mapped


def _collect_dataset(
    rollouts_path: Path,
    transition_pair: Optional[Tuple[str, str]] = None,
    target_statistic: str = "aa_time",
) -> Tuple[List[str], List[str], np.ndarray, Dict[str, List[Any]], int]:
    """
    From rollout records, build:
    - numeric_param_names: list of parameter names treated as numeric
    - categorical_param_names: list of parameter names treated as categorical
    - y: np.ndarray of target values (target_statistic or transition duration)
    - param_values_raw: dict param_name -> list of raw values (for correlation/ANOVA)
    - total_records: total number of records processed
    """
    reader = RolloutsReader(rollouts_path)

    # Filter to usable records
    usable: List[Tuple[Dict[str, Any], float]] = []
    total_records = 0

    if transition_pair is not None:
        # Use iter_transition_with_params for efficient extraction with proper censoring
        from_name, to_name = transition_pair
        for from_time, duration, params in reader.iter_transition_with_params(
            from_name, to_name,
            include_censored=True,
            censored_value=1e10  # Spearman ranks censored cases at the end
        ):
            total_records += 1
            if duration > 0.0 and np.isfinite(duration):
                usable.append((params, duration))
    else:
        # Use iter_rollouts for aa_time / other statistics
        for rec in reader.iter_rollouts():
            total_records += 1
            params = rec.get("parameters", {})
            ts_params = rec.get("time_series_parameters", {})
            results = rec.get("results")
            if not results:
                continue
            # Merge time series parameters into params for analysis
            all_params = {**params, **ts_params} if params or ts_params else None
            if not all_params:
                continue
            if rec.get("error") is not None:
                continue

            assert isinstance(results, dict), f"results must be a dict, instead got {type(results)}"

            # Determine target y from target_statistic
            assert target_statistic in results, f"results does not have key {target_statistic}"
            target_statistic_val = results[target_statistic]
            # Skip records where target statistic is None (e.g., milestone not reached)
            if target_statistic_val is not None and isinstance(target_statistic_val, (int, float)) and np.isfinite(target_statistic_val):
                usable.append((all_params, float(target_statistic_val)))

    if not usable:
        raise RuntimeError("No usable records with finite aa_time and parameters")

    # Determine param names and types by inspecting values across usable records
    # Exclude r_software since it's computed from other parameters, not sampled
    all_param_names: List[str] = sorted({k for params, _ in usable for k in params.keys() if k != 'r_software'})
    numeric_param_names: List[str] = []
    categorical_param_names: List[str] = []

    for name in all_param_names:
        # Identify if any non-numeric values present
        saw_string = False
        saw_numeric = False
        for params, _ in usable:
            v = params.get(name)
            if isinstance(v, str):
                saw_string = True
                break
            if _is_numeric(v):
                saw_numeric = True
        if saw_string:
            categorical_param_names.append(name)
        elif saw_numeric:
            numeric_param_names.append(name)
        # else: missing entirely or null everywhere — ignore

    # Collect raw values
    y = np.array([sc for _, sc in usable], dtype=float)
    param_values_raw: Dict[str, List[Any]] = {name: [] for name in (numeric_param_names + categorical_param_names)}
    for params, _sc in usable:
        for name in numeric_param_names:
            v = params.get(name)
            param_values_raw[name].append(float(v) if _is_numeric(v) else np.nan)
        for name in categorical_param_names:
            param_values_raw[name].append(params.get(name))

    return numeric_param_names, categorical_param_names, y, param_values_raw, total_records


def _compute_numeric_correlations(values: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    mask = np.isfinite(values) & np.isfinite(y)
    if mask.sum() < 3:
        return {"pearson_r": float("nan"), "pearson_p": float("nan"), "spearman_rho": float("nan"), "spearman_p": float("nan")}
    r, p = stats.pearsonr(values[mask], y[mask])
    rho, p_s = stats.spearmanr(values[mask], y[mask])
    return {"pearson_r": float(r), "pearson_p": float(p), "spearman_rho": float(rho), "spearman_p": float(p_s)}


def _compute_categorical_anova(values: List[Any], y: np.ndarray) -> Dict[str, float]:
    # Group y by category; ignore None categories
    groups: Dict[Any, List[float]] = defaultdict(list)
    for v, target in zip(values, y.tolist()):
        if v is None:
            continue
        groups[v].append(float(target))
    if len(groups) <= 1:
        return {"anova_F": float("nan"), "anova_p": float("nan"), "eta_squared": float("nan"), "num_categories": float(len(groups))}
    group_arrays = [np.array(vals, dtype=float) for vals in groups.values() if len(vals) > 0]
    # One-way ANOVA
    F, p = stats.f_oneway(*group_arrays)
    # Eta-squared effect size
    y_all = np.concatenate(group_arrays)
    grand_mean = float(np.mean(y_all))
    ss_between = float(sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in group_arrays))
    ss_total = float(np.sum((y_all - grand_mean) ** 2))
    eta2 = ss_between / ss_total if ss_total > 0 else float("nan")
    return {"anova_F": float(F), "anova_p": float(p), "eta_squared": float(eta2), "num_categories": float(len(group_arrays))}


def _one_hot_encode_categoricals(categorical_param_names: List[str], param_values_raw: Dict[str, List[Any]]) -> Tuple[List[str], np.ndarray, Dict[str, List[int]]]:
    feature_names: List[str] = []
    X_cat_list: List[np.ndarray] = []
    param_to_feature_indices: Dict[str, List[int]] = {}
    feature_start_idx = 0

    for name in categorical_param_names:
        values = param_values_raw[name]
        categories = sorted({v for v in values if v is not None})
        if len(categories) == 0:
            continue
        # Full one-hot; we will handle collinearity via pseudo-inverse in regression
        cat_to_idx = {c: i for i, c in enumerate(categories)}
        X = np.zeros((len(values), len(categories)), dtype=float)
        for row_idx, v in enumerate(values):
            if v is None:
                continue
            j = cat_to_idx.get(v)
            if j is not None:
                X[row_idx, j] = 1.0
        feature_names.extend([f"{name}=={c}" for c in categories])
        param_to_feature_indices[name] = list(range(feature_start_idx, feature_start_idx + len(categories)))
        feature_start_idx += len(categories)
        X_cat_list.append(X)

    if not X_cat_list:
        return feature_names, np.zeros((len(next(iter(param_values_raw.values()), [])), 0), dtype=float), param_to_feature_indices

    X_cat = np.concatenate(X_cat_list, axis=1)
    return feature_names, X_cat, param_to_feature_indices


def _standardize_train_test(X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    std_adj = np.where(std == 0, 1.0, std)
    return (X_train - mean) / std_adj, (X_test - mean) / std_adj


def _fit_linear_regression_pinv(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    # Add intercept
    X_aug = np.c_[np.ones((X.shape[0], 1), dtype=float), X]
    # Moore-Penrose pseudo-inverse for stability
    w = np.linalg.pinv(X_aug) @ y
    return w  # intercept first


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def _predict_linear(w: np.ndarray, X: np.ndarray) -> np.ndarray:
    X_aug = np.c_[np.ones((X.shape[0], 1), dtype=float), X]
    return X_aug @ w


def _permutation_importance(
    X: np.ndarray,
    y: np.ndarray,
    feature_groupings: Dict[str, List[int]],
    num_folds: int = 5,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Tuple[float, float]]:
    if rng is None:
        rng = np.random.default_rng(42)
    n = X.shape[0]
    indices = np.arange(n)
    rng.shuffle(indices)
    folds = np.array_split(indices, num_folds)

    importances: Dict[str, List[float]] = {k: [] for k in feature_groupings.keys()}

    for fold_idx in range(num_folds):
        val_idx = folds[fold_idx]
        train_idx = np.concatenate([folds[i] for i in range(num_folds) if i != fold_idx])
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Standardize based on train
        X_train_s, X_val_s = _standardize_train_test(X_train, X_val)

        w = _fit_linear_regression_pinv(X_train_s, y_train)
        y_pred = _predict_linear(w, X_val_s)
        baseline_r2 = _r2_score(y_val, y_pred)

        for param_name, cols in feature_groupings.items():
            X_val_perm = X_val_s.copy()
            if len(cols) == 1:
                col = cols[0]
                shuffled = X_val_perm[:, col].copy()
                rng.shuffle(shuffled)
                X_val_perm[:, col] = shuffled
            else:
                # Shuffle columns as a block by permuting rows
                perm = rng.permutation(X_val_perm.shape[0])
                X_val_perm[:, cols] = X_val_perm[perm][:, cols]
            y_perm = _predict_linear(w, X_val_perm)
            r2_perm = _r2_score(y_val, y_perm)
            importances[param_name].append(baseline_r2 - r2_perm)

    # Aggregate
    summary: Dict[str, Tuple[float, float]] = {}
    for k, vals in importances.items():
        if len(vals) == 0:
            summary[k] = (float("nan"), float("nan"))
        else:
            summary[k] = (float(np.mean(vals)), float(np.std(vals)))
    return summary


def analyze_milestone_transitions(
    rollouts_path: Path,
    output_dir: Path,
    transition_pair: Tuple[str, str],
    include_censored: bool = False,
) -> None:
    """Run sensitivity analysis for milestone transitions.

    Args:
        rollouts_path: Path to rollouts.jsonl
        output_dir: Directory to save outputs
        transition_pair: (from_milestone, to_milestone)
        include_censored: If True, include rollouts where upper milestone not achieved
                         (censored transitions get duration=1e10 for Spearman ranking);
                         if False, only include rollouts where both milestones achieved
    """
    from_name, to_name = transition_pair
    suffix = "censored" if include_censored else "both_achieved"
    safe_from = "".join([c if c.isalnum() or c in ("_", "-") else "-" for c in from_name])
    safe_to = "".join([c if c.isalnum() or c in ("_", "-") else "-" for c in to_name])

    # Create sensitivity_analysis subfolder
    sensitivity_dir = output_dir / 'sensitivity_analysis'
    sensitivity_dir.mkdir(parents=True, exist_ok=True)
    out_json = sensitivity_dir / f"sensitivity_{safe_from}_to_{safe_to}_{suffix}.json"

    result = analyze(
        rollouts_path,
        out_json=out_json,
        make_plots=True,
        transition_pair=transition_pair,
        include_censored=include_censored,
        target_statistic="transition_duration",
        censored_suffix=suffix
    )

    print(f"Saved sensitivity analysis: {from_name} -> {to_name} ({suffix})")


def analyze(rollouts_path: Path, out_json: Optional[Path] = None, make_plots: bool = False, make_scatter_plots: bool = False, transition_pair: Optional[Tuple[str, str]] = None, include_censored: bool = True, target_statistic: str = "aa_time", censored_suffix: Optional[str] = None) -> Dict[str, Any]:
    numeric_names, categorical_names, y, param_values_raw, total_records = _collect_dataset(rollouts_path, transition_pair=transition_pair, target_statistic=target_statistic)

    used_n = y.shape[0]
    dropped = total_records - used_n

    # Correlations for numeric
    correlations: Dict[str, Dict[str, float]] = {}
    for name in numeric_names:
        vals = np.array(param_values_raw[name], dtype=float)
        correlations[name] = _compute_numeric_correlations(vals, y)

    # ANOVA for categorical
    for name in categorical_names:
        vals = param_values_raw[name]
        correlations[name] = _compute_categorical_anova(vals, y)

    # Special-case: treat include_gap as binary for Spearman and Pearson
    if "include_gap" in categorical_names:
        mapped = _map_include_gap_to_binary(param_values_raw["include_gap"])  # type: ignore[index]
        sp = _compute_numeric_correlations(mapped, y)
        correlations.setdefault("include_gap", {})
        # Record Spearman and Pearson so it can be plotted/printed alongside numeric params
        correlations["include_gap"]["spearman_rho"] = float(sp.get("spearman_rho", float("nan")))
        correlations["include_gap"]["spearman_p"] = float(sp.get("spearman_p", float("nan")))
        correlations["include_gap"]["pearson_r"] = float(sp.get("pearson_r", float("nan")))
        correlations["include_gap"]["pearson_p"] = float(sp.get("pearson_p", float("nan")))

    # Build feature matrix for permutation importance
    # Numeric features
    feature_names: List[str] = []
    X_numeric_list: List[np.ndarray] = []
    feature_groupings: Dict[str, List[int]] = {}
    feature_idx = 0

    for name in numeric_names:
        v = np.array(param_values_raw[name], dtype=float)
        v[np.logical_not(np.isfinite(v))] = np.nan
        # Impute NaN to column mean (computed over finite entries)
        if np.isnan(v).any():
            finite_mask = np.isfinite(v)
            col_mean = float(np.mean(v[finite_mask])) if finite_mask.any() else 0.0
            v[~finite_mask] = col_mean
        X_numeric_list.append(v.reshape(-1, 1))
        feature_names.append(name)
        feature_groupings[name] = [feature_idx]
        feature_idx += 1

    X_num = np.concatenate(X_numeric_list, axis=1) if X_numeric_list else np.zeros((used_n, 0), dtype=float)

    # Categorical one-hot
    cat_feature_names, X_cat, param_to_cat_cols = _one_hot_encode_categoricals(categorical_names, param_values_raw)
    # Offset group column indices by current feature_idx
    for pname, cols in param_to_cat_cols.items():
        offset_cols = [feature_idx + c for c in cols]
        feature_groupings[pname] = offset_cols
    feature_names.extend(cat_feature_names)

    X_all = np.concatenate([X_num, X_cat], axis=1) if X_cat.shape[1] > 0 else X_num

    permutation_summary = _permutation_importance(X_all, y, feature_groupings, num_folds=5)

    target_label = "aa_time" if transition_pair is None else f"duration:{transition_pair[0]}->{transition_pair[1]}"

    result: Dict[str, Any] = {
        "num_records_total": total_records,
        "num_samples_used": used_n,
        "dropped_due_to_missing_or_invalid": dropped,
        "target": target_label,
        "numeric_parameters": numeric_names,
        "categorical_parameters": categorical_names,
        "associations": correlations,
        "permutation_importance": {
            k: {"mean_drop_in_r2": float(mu), "std": float(sd)} for k, (mu, sd) in permutation_summary.items()
        },
    }

    if out_json is not None:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(result, indent=2))

    if make_plots:
        try:
            def _safe_name(s: str) -> str:
                return "".join([c if c.isalnum() or c in ("_", "-") else "-" for c in s])

            if transition_pair is None:
                suffix = "_aa_time"
            else:
                base_suffix = f"_{_safe_name(transition_pair[0])}-to-{_safe_name(transition_pair[1])}"
                if censored_suffix:
                    suffix = f"{base_suffix}_{censored_suffix}"
                else:
                    suffix = base_suffix

            # Spearman correlation for numeric parameters (+ special-case include_gap)
            spearman_names = list(numeric_names)
            if "include_gap" not in spearman_names and "include_gap" in correlations:
                spearman_names.append("include_gap")
            spearman_items = [
                (name, correlations[name]["spearman_rho"]) for name in spearman_names if np.isfinite(correlations[name].get("spearman_rho", float("nan")))  # type: ignore[index]
            ]
            spearman_items.sort(key=lambda x: abs(x[1]), reverse=True)

            if spearman_items:
                names, vals = zip(*spearman_items[:30])
                plt.figure(figsize=(13, max(4, len(names) * 0.35)))
                y_pos = np.arange(len(names))
                plt.barh(y_pos, vals, align='center')
                plt.yticks(y_pos, names, fontsize=9)
                plt.xlabel(f'Spearman rho with {target_label}')
                # Create a shorter title - extract censored/both achieved info from suffix if present
                if transition_pair is not None:
                    title_suffix_text = ""
                    if "both-achieved" in suffix or "both_achieved" in suffix:
                        title_suffix_text = " (both achieved)"
                    elif "censored" in suffix:
                        title_suffix_text = " (assuming achieved at simulation cutoff)"
                    short_title = f"Parameter sensitivity: {transition_pair[0]} → {transition_pair[1]}{title_suffix_text}"
                else:
                    short_title = "Parameter sensitivity for AC time"
                plt.title(short_title, pad=15, fontsize=10)
                plt.gca().invert_yaxis()
                # Save main sensitivity plot in sensitivity_analysis folder
                sensitivity_dir = rollouts_path.parent / 'sensitivity_analysis'
                sensitivity_dir.mkdir(parents=True, exist_ok=True)
                plot_path = sensitivity_dir / f'sensitivity_spearman_top{suffix}.png'
                plt.tight_layout()
                plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                plt.close()

            # Scatter plots: each numeric parameter vs target_statistic
            # Only generate if explicitly requested (many plots, large output)
            if make_scatter_plots:
                sensitivity_dir = rollouts_path.parent / 'sensitivity_analysis'
                sensitivity_dir.mkdir(parents=True, exist_ok=True)
                scatter_dir = sensitivity_dir / 'scatter_plots'
                scatter_dir.mkdir(parents=True, exist_ok=True)

                for name in numeric_names:
                    x_vals = np.array(param_values_raw[name], dtype=float)
                    mask = np.isfinite(x_vals) & np.isfinite(y)
                    if mask.sum() < 2:
                        continue
                    plt.figure(figsize=(6.5, 4.5))
                    plt.scatter(x_vals[mask], y[mask], s=12, alpha=0.6, edgecolors='none')
                    plt.xlabel(name)
                    plt.ylabel(target_label)
                    plt.title(f"{name} vs {target_label}", fontsize=11, pad=12)
                    plt.tight_layout()
                    scatter_path = scatter_dir / f"scatter_{_safe_name(name)}{suffix}.png"
                    plt.savefig(scatter_path, dpi=150, bbox_inches='tight')
                    plt.close()

                    # Rank-based scatter (Spearman visualization): average ranks for ties
                    x_rank = stats.rankdata(x_vals[mask], method='average')
                    y_rank = stats.rankdata(y[mask], method='average')
                    plt.figure(figsize=(6.5, 4.5))
                    plt.scatter(x_rank, y_rank, s=12, alpha=0.6, edgecolors='none')
                    plt.xlabel(f"rank({name})")
                    plt.ylabel(f"rank({target_label})")
                    plt.title(f"Ranks: {name} vs {target_label}", fontsize=11, pad=12)
                    plt.tight_layout()
                    scatter_rank_path = scatter_dir / f"scatter_rank_{_safe_name(name)}{suffix}.png"
                    plt.savefig(scatter_rank_path, dpi=150, bbox_inches='tight')
                    plt.close()
        except Exception as e:
            print(f"Error making plots: {e}")

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sensitivity analysis for batch rollout outputs")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run-dir", type=str, help="Path to a run directory under outputs/ containing rollouts.jsonl")
    group.add_argument("--rollouts", type=str, help="Direct path to a rollouts.jsonl file")
    parser.add_argument("--out-json", type=str, default=None, help="Path to write JSON summary (default: run_dir/sensitivity_summary.json)")
    parser.add_argument("--plot", action="store_true", help="Save summary bar chart alongside outputs")
    parser.add_argument("--scatter-plots", action="store_true", help="Also generate individual scatter plots for each parameter (many files)")
    parser.add_argument("--transition-pair", type=str, default=None, help="Analyze duration for milestone pair FROM:TO instead of aa_time")
    parser.add_argument("--target-statistic", type=str, default="aa_time", help="What are we analyzing the sensitivity of? (default: aa_time)")
    return parser.parse_args()


def _load_correlation_config(
    run_dir: Path,
) -> Tuple[Optional[np.ndarray], Optional[List[str]], Optional[str], Optional[Dict[str, str]]]:
    """Load correlation matrix configuration from run directory.

    Returns:
        corr_matrix: correlation matrix as numpy array, or None if not found
        param_names: list of parameter names, or None if not found
        corr_type: correlation type (e.g., 'spearman'), or None if not found
        param_dists: dict mapping parameter name to distribution type, or None
    """
    config_path = run_dir / "input_distributions.yaml"
    if not config_path.exists():
        return None, None, None, None

    try:
        with config_path.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"Warning: Could not load {config_path}: {e}")
        return None, None, None, None

    corr_config = config.get("correlation_matrix")
    if not isinstance(corr_config, dict):
        return None, None, None, None

    corr_matrix_list = corr_config.get("correlation_matrix")
    param_names = corr_config.get("parameters")
    corr_type = corr_config.get("correlation_type", "pearson")

    if not corr_matrix_list or not param_names:
        return None, None, None, None

    try:
        corr_matrix = np.array(corr_matrix_list, dtype=float)
    except Exception as e:
        print(f"Warning: Could not parse correlation matrix: {e}")
        return None, None, None, None

    # Extract distribution types for parameters
    param_dists: Dict[str, str] = {}

    # Check both 'parameters' and 'time_series_parameters' sections
    params_config = config.get("parameters", {})
    ts_params_config = config.get("time_series_parameters", {})

    for param_name in param_names:
        # Look in both sections
        param_info = params_config.get(param_name) or ts_params_config.get(param_name)
        if isinstance(param_info, dict) and "dist" in param_info:
            param_dists[param_name] = param_info["dist"]

    return corr_matrix, param_names, corr_type, param_dists


def _load_all_samples(samples_file: Path) -> Dict[str, np.ndarray]:
    """Load all sampled parameter values from samples.jsonl.

    Returns:
        param_values: dictionary mapping parameter name to array of sampled values
    """
    param_values: Dict[str, List[float]] = {}

    with samples_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Merge parameters and time_series_parameters
            all_params = {}
            if "parameters" in rec and isinstance(rec["parameters"], dict):
                all_params.update(rec["parameters"])
            if "time_series_parameters" in rec and isinstance(rec["time_series_parameters"], dict):
                all_params.update(rec["time_series_parameters"])

            # Store numeric values only
            for key, value in all_params.items():
                if isinstance(value, (int, float)) and np.isfinite(value):
                    if key not in param_values:
                        param_values[key] = []
                    param_values[key].append(float(value))

    # Convert lists to numpy arrays
    param_arrays = {k: np.array(v, dtype=float) for k, v in param_values.items()}
    return param_arrays


def plot_all_correlations(run_dir: Path) -> None:
    """Generate scatter plots for all correlated parameter pairs.

    Reads the correlation matrix from input_distributions.yaml and creates
    scatter plots for all parameter pairs with non-zero correlation.
    Creates both value-based and rank-based plots.

    Args:
        run_dir: path to run directory containing input_distributions.yaml and samples.jsonl
    """
    print(f"Loading correlation configuration from {run_dir}")

    # Load correlation matrix
    corr_matrix, param_names, corr_type, param_dists = _load_correlation_config(run_dir)
    if corr_matrix is None or param_names is None:
        print("Warning: Could not load correlation configuration")
        return

    print(f"Found {len(param_names)} parameters with {corr_type} correlations")
    if param_dists:
        print(f"Loaded distribution types for {len(param_dists)} parameters")

    # Load samples
    samples_file = run_dir / "samples.jsonl"
    if not samples_file.exists():
        print(f"Warning: samples.jsonl not found in {run_dir}")
        return

    print("Loading samples...")
    param_values = _load_all_samples(samples_file)
    print(f"Loaded samples for {len(param_values)} parameters")

    # Create output directory
    output_dir = run_dir / "correlation_plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving plots to {output_dir}")

    # Find all non-zero correlations
    num_plots = 0
    for i in range(len(param_names)):
        for j in range(i + 1, len(param_names)):
            corr = corr_matrix[i, j]

            # Skip zero or near-zero correlations
            if abs(corr) < 0.01:
                continue

            param_i = param_names[i]
            param_j = param_names[j]

            # Check if we have data for both parameters
            if param_i not in param_values or param_j not in param_values:
                print(f"Warning: Missing data for {param_i} or {param_j}, skipping")
                continue

            x_vals = param_values[param_i]
            y_vals = param_values[param_j]

            # Check if arrays have same length
            if len(x_vals) != len(y_vals):
                print(
                    f"Warning: Mismatched lengths for {param_i} ({len(x_vals)}) "
                    f"and {param_j} ({len(y_vals)}), skipping"
                )
                continue

            print(f"  Creating plots for {param_i} vs {param_j} (target {corr_type} corr: {corr:.2f})")

            # Get distribution types
            x_dist = param_dists.get(param_i) if param_dists else None
            y_dist = param_dists.get(param_j) if param_dists else None

            # Create safe filenames
            safe_i = param_i.replace("/", "_").replace(" ", "_")
            safe_j = param_j.replace("/", "_").replace(" ", "_")

            # Plot actual values (without actual correlation)
            try:
                out_path_values = output_dir / f"corr_{safe_i}_vs_{safe_j}_values.png"
                plot_correlation_scatter(
                    x_vals,
                    y_vals,
                    out_path_values,
                    param_i,
                    param_j,
                    corr,
                    use_ranks=False,
                    corr_type=corr_type,
                    show_actual_corr=False,
                    x_dist=x_dist,
                    y_dist=y_dist,
                )
                num_plots += 1
            except Exception as e:
                print(f"    Warning: Could not create values plot: {e}")

            # Plot ranks (with actual correlation)
            try:
                out_path_ranks = output_dir / f"corr_{safe_i}_vs_{safe_j}_ranks.png"
                plot_correlation_scatter(
                    x_vals,
                    y_vals,
                    out_path_ranks,
                    param_i,
                    param_j,
                    corr,
                    use_ranks=True,
                    corr_type=corr_type,
                    show_actual_corr=True,
                )
                num_plots += 1
            except Exception as e:
                print(f"    Warning: Could not create ranks plot: {e}")

    print(f"\nGenerated {num_plots} correlation plots in {output_dir}")


def main() -> None:
    args = parse_args()
    if args.run_dir:
        run_dir = Path(args.run_dir)
        rollouts_path = run_dir / "rollouts.jsonl"
        if not rollouts_path.exists():
            raise FileNotFoundError(f"rollouts.jsonl not found in run dir: {run_dir}")
        # Create sensitivity_analysis subfolder
        sensitivity_dir = run_dir / 'sensitivity_analysis'
        sensitivity_dir.mkdir(parents=True, exist_ok=True)
        out_json = Path(args.out_json) if args.out_json else (sensitivity_dir / "sensitivity_summary.json")
    else:
        rollouts_path = Path(args.rollouts)
        if not rollouts_path.exists():
            raise FileNotFoundError(f"rollouts file not found: {rollouts_path}")
        # Create sensitivity_analysis subfolder
        sensitivity_dir = rollouts_path.parent / 'sensitivity_analysis'
        sensitivity_dir.mkdir(parents=True, exist_ok=True)
        out_json = Path(args.out_json) if args.out_json else (sensitivity_dir / "sensitivity_summary.json")

    tp: Optional[Tuple[str, str]] = None
    if args.transition_pair:
        if ":" not in args.transition_pair:
            raise ValueError("--transition-pair must be formatted as FROM:TO")
        left, right = [s.strip() for s in args.transition_pair.split(":", 1)]
        if not left or not right:
            raise ValueError("--transition-pair requires both FROM and TO names")
        tp = (left, right)

    result = analyze(rollouts_path, out_json=out_json, make_plots=args.plot, make_scatter_plots=args.scatter_plots, transition_pair=tp, target_statistic=args.target_statistic)

    # Print concise top-10 summary to stdout
    print(f"Target: {result.get('target', 'aa_time')}")
    print(f"Loaded {result['num_samples_used']} usable samples (of {result['num_records_total']}).")

    # Numeric parameters - Pearson correlation (plus include_gap special-case)
    numeric_names = result.get("numeric_parameters", [])
    categorical_names = result.get("categorical_parameters", [])
    assoc = result.get("associations", {})

    pearson_list = []
    pearson_names = list(numeric_names)
    if "include_gap" not in pearson_names and "include_gap" in assoc and np.isfinite(assoc.get("include_gap", {}).get("pearson_r", float("nan"))):
        pearson_names.append("include_gap")
    for name in pearson_names:
        entry = assoc.get(name, {})
        r = entry.get("pearson_r")
        if r is not None and np.isfinite(r):
            pearson_list.append((name, float(r)))
    pearson_list.sort(key=lambda x: abs(x[1]), reverse=True)
    print("Top numeric by Pearson correlation (abs):")
    for name, r in pearson_list[:10]:
        print(f"  {name:30s}  r={r:+.3f}")

    # Numeric parameters - Spearman correlation (plus include_gap special-case)
    spearman_list = []
    spearman_names = list(numeric_names)
    if "include_gap" not in spearman_names and "include_gap" in assoc and np.isfinite(assoc.get("include_gap", {}).get("spearman_rho", float("nan"))):
        spearman_names.append("include_gap")
    for name in spearman_names:
        entry = assoc.get(name, {})
        rho = entry.get("spearman_rho")
        if rho is not None and np.isfinite(rho):
            spearman_list.append((name, float(rho)))
    spearman_list.sort(key=lambda x: abs(x[1]), reverse=True)
    print("Top numeric by Spearman correlation (abs):")
    for name, rho in spearman_list[:10]:
        print(f"  {name:30s}  rho={rho:+.3f}")

    # Categorical parameters - ANOVA F-statistic
    anova_list = []
    for name in categorical_names:
        entry = assoc.get(name, {})
        f_stat = entry.get("anova_F")
        if f_stat is not None and np.isfinite(f_stat):
            anova_list.append((name, float(f_stat)))
    anova_list.sort(key=lambda x: x[1], reverse=True)
    print("Top categorical by ANOVA F-statistic:")
    for name, f_stat in anova_list[:10]:
        print(f"  {name:30s}  F={f_stat:.2f}")

    # Categorical parameters - Eta-squared effect size
    eta_list = []
    for name in categorical_names:
        entry = assoc.get(name, {})
        eta_sq = entry.get("eta_squared")
        if eta_sq is not None and np.isfinite(eta_sq):
            eta_list.append((name, float(eta_sq)))
    eta_list.sort(key=lambda x: x[1], reverse=True)
    print("Top categorical by eta-squared effect size:")
    for name, eta_sq in eta_list[:10]:
        print(f"  {name:30s}  η²={eta_sq:.4f}")

    # Permutation importance
    perm = result.get("permutation_importance", {})
    perm_items = [(k, v.get("mean_drop_in_r2", float("nan"))) for k, v in perm.items()]
    perm_items = [(k, v) for k, v in perm_items if np.isfinite(v)]
    perm_items.sort(key=lambda x: x[1], reverse=True)
    print("Top by permutation importance (mean ΔR^2):")
    for name, imp in perm_items[:10]:
        print(f"  {name:30s}  ΔR^2={imp:.4f}")


if __name__ == "__main__":
    main()

