#!/usr/bin/env python3
"""
Convert samples.jsonl to samples_quantile.jsonl by mapping parameter values to quantiles.

This script inverts the sampling process: for each parameter value in samples.jsonl,
it computes the quantile (0-1) that would have generated that value from its distribution.

Usage:
  python scripts/samples_to_quantiles.py --run-dir outputs/20251023_184450/
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import yaml
import scipy.special
import scipy.stats

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert samples to quantiles")
    parser.add_argument("--run-dir", type=str, required=True, help="Path to run directory containing samples.jsonl")
    return parser.parse_args()


def _value_to_quantile(dist_spec: Dict[str, Any], value: Any, param_name: Optional[str] = None) -> float:
    """Invert the sampling process: compute the quantile that would produce the given value.
    
    Args:
        dist_spec: Distribution specification
        value: The sampled value
        param_name: Parameter name for error messages
        
    Returns:
        Quantile in [0, 1]
    """
    kind = dist_spec.get("dist", "fixed")
    
    if value is None:
        return None
    
    if kind == "fixed":
        # Fixed values don't have a meaningful quantile; use 0.5 as convention
        return 0.5
    
    if kind == "uniform":
        a = float(dist_spec["min"])
        b = float(dist_spec["max"])
        if b <= a:
            return 0.5
        quantile = (float(value) - a) / (b - a)
        return float(np.clip(quantile, 0.0, 1.0))
    
    if kind == "normal":
        # Support parameterization by mean/sd (or sigma) OR by 80% CI (q10,q90)
        has_pair = "ci80_low" in dist_spec and "ci80_high" in dist_spec
        has_array = "ci80" in dist_spec
        if has_array or has_pair:
            if has_array:
                ci = dist_spec["ci80"]
                q10 = float(ci[0])
                q90 = float(ci[1])
            else:
                q10 = float(dist_spec["ci80_low"])
                q90 = float(dist_spec["ci80_high"])
            if q10 > q90:
                q10, q90 = q90, q10
            z = 1.2815515655446004
            mu = 0.5 * (q10 + q90)
            sigma = (q90 - q10) / (2.0 * z)
        else:
            mu = float(dist_spec["mean"])
            sigma = float(dist_spec["sd"]) if "sd" in dist_spec else float(dist_spec.get("sigma", 1.0))
        
        if sigma <= 0:
            return 0.5
        
        # Use normal CDF
        z_score = (float(value) - mu) / sigma
        quantile = 0.5 * (1.0 + scipy.special.erf(z_score / np.sqrt(2)))
        return float(np.clip(quantile, 0.0, 1.0))
    
    if kind == "lognormal":
        # Support parameterization by mu/sigma in log-space OR by 80% CI in original space
        has_pair = "ci80_low" in dist_spec and "ci80_high" in dist_spec
        has_array = "ci80" in dist_spec
        if has_array or has_pair:
            if has_array:
                ci = dist_spec["ci80"]
                q10 = float(ci[0])
                q90 = float(ci[1])
            else:
                q10 = float(dist_spec["ci80_low"])
                q90 = float(dist_spec["ci80_high"])
            if q10 > q90:
                q10, q90 = q90, q10
            z = 1.2815515655446004
            ln_q10 = float(np.log(q10))
            ln_q90 = float(np.log(q90))
            mu = 0.5 * (ln_q10 + ln_q90)
            sigma = (ln_q90 - ln_q10) / (2.0 * z)
        else:
            mu = float(dist_spec["mu"])
            sigma = float(dist_spec["sigma"])
        
        if sigma <= 0 or value <= 0:
            return 0.5
        
        # Use lognormal CDF (which is normal CDF of log(value))
        z_score = (np.log(float(value)) - mu) / sigma
        quantile = 0.5 * (1.0 + scipy.special.erf(z_score / np.sqrt(2)))
        return float(np.clip(quantile, 0.0, 1.0))
    
    if kind == "shifted_lognormal":
        # Shifted lognormal: x = shift + LogNormal(mu, sigma)
        has_pair = "ci80_low" in dist_spec and "ci80_high" in dist_spec
        has_array = "ci80" in dist_spec
        if has_array or has_pair:
            if has_array:
                ci = dist_spec["ci80"]
                q10 = float(ci[0])
                q90 = float(ci[1])
            else:
                q10 = float(dist_spec["ci80_low"])
                q90 = float(dist_spec["ci80_high"])
            if q10 > q90:
                q10, q90 = q90, q10
            z = 1.2815515655446004
            ln_q10 = float(np.log(q10))
            ln_q90 = float(np.log(q90))
            mu = 0.5 * (ln_q10 + ln_q90)
            sigma = (ln_q90 - ln_q10) / (2.0 * z)
        else:
            mu = float(dist_spec["mu"])
            sigma = float(dist_spec["sigma"])
        
        shift = dist_spec.get("shift")
        if shift is None:
            raise ValueError(f"shift for shifted_lognormal is not set for parameter {param_name}")
        shift = float(shift)
        
        # Invert: value = shift + exp(mu + sigma * sqrt(2) * erfinv(2*q - 1))
        # value - shift = exp(mu + sigma * sqrt(2) * erfinv(2*q - 1))
        x_core = float(value) - shift
        
        if sigma <= 0 or x_core <= 0:
            return 0.5
        
        z_score = (np.log(x_core) - mu) / sigma
        quantile = 0.5 * (1.0 + scipy.special.erf(z_score / np.sqrt(2)))
        return float(np.clip(quantile, 0.0, 1.0))
    
    if kind == "beta":
        a = float(dist_spec["alpha"])
        b = float(dist_spec["beta"])
        lo = float(dist_spec.get("min", 0.0))
        hi = float(dist_spec.get("max", 1.0))
        
        if hi <= lo:
            return 0.5
        
        # Invert: value = lo + (hi - lo) * beta_cdf^{-1}(quantile)
        # (value - lo) / (hi - lo) = beta_cdf^{-1}(quantile)
        x01 = (float(value) - lo) / (hi - lo)
        x01 = float(np.clip(x01, 0.0, 1.0))
        
        # Use beta CDF
        quantile = scipy.stats.beta.cdf(x01, a, b)
        return float(np.clip(quantile, 0.0, 1.0))
    
    if kind == "choice":
        values = dist_spec["values"]
        p = dist_spec.get("p")
        
        # Find which value was chosen
        try:
            idx = values.index(value)
        except (ValueError, AttributeError):
            # Value not in list or values not subscriptable; return middle quantile
            return 0.5
        
        if p is None:
            # Uniform choice: map to middle of the corresponding quantile bin
            n = len(values)
            return (idx + 0.5) / n
        else:
            # Weighted choice: map to middle of the cumulative probability bin
            cumsum = np.cumsum([0.0] + list(p))
            return float((cumsum[idx] + cumsum[idx + 1]) / 2.0)
    
    raise ValueError(f"Unknown distribution kind: {kind}")


def _convert_sample_to_quantiles(
    sample: Dict[str, Any],
    param_dists: Dict[str, Any],
    ts_param_dists: Dict[str, Any],
) -> Dict[str, Any]:
    """Convert a sample record to quantiles.
    
    Args:
        sample: Sample record with parameters and time_series_parameters
        param_dists: Model parameter distributions
        ts_param_dists: Time series parameter distributions
        
    Returns:
        Sample record with quantiles instead of values
    """
    result = {
        "sample_id": sample.get("sample_id"),
    }
    
    # Handle error cases
    if "error" in sample:
        result["error"] = sample["error"]
        if "traceback" in sample:
            result["traceback"] = sample["traceback"]
    
    # Convert model parameters to quantiles
    if sample.get("parameters") is not None:
        params = sample["parameters"]
        quantile_params = {}
        for param_name, param_value in params.items():
            if param_name in param_dists:
                try:
                    quantile_params[param_name] = _value_to_quantile(
                        param_dists[param_name],
                        param_value,
                        param_name
                    )
                except Exception as e:
                    print(f"Warning: Failed to convert {param_name}={param_value} to quantile: {e}")
                    quantile_params[param_name] = None
            else:
                # Parameter not in distribution (e.g., computed values like r_software)
                # Keep as-is or mark as None
                quantile_params[param_name] = None
        result["parameters"] = quantile_params
    else:
        result["parameters"] = None
    
    # Convert time series parameters to quantiles
    if sample.get("time_series_parameters") is not None:
        ts_params = sample["time_series_parameters"]
        quantile_ts_params = {}
        for param_name, param_value in ts_params.items():
            if param_name in ts_param_dists:
                try:
                    quantile_ts_params[param_name] = _value_to_quantile(
                        ts_param_dists[param_name],
                        param_value,
                        param_name
                    )
                except Exception as e:
                    print(f"Warning: Failed to convert {param_name}={param_value} to quantile: {e}")
                    quantile_ts_params[param_name] = None
            else:
                quantile_ts_params[param_name] = None
        result["time_series_parameters"] = quantile_ts_params
    else:
        result["time_series_parameters"] = None
    
    return result


def main() -> None:
    args = parse_args()
    
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"Error: Run directory not found: {run_dir}")
        sys.exit(1)
    
    # Load input distributions
    input_dist_path = run_dir / "input_distributions.yaml"
    if not input_dist_path.exists():
        print(f"Error: input_distributions.yaml not found in {run_dir}")
        sys.exit(1)
    
    with input_dist_path.open("r") as f:
        config = yaml.safe_load(f)
    
    param_dists = config.get("parameters", {})
    ts_param_dists = config.get("time_series_parameters", {})
    
    # Load samples
    samples_path = run_dir / "samples.jsonl"
    if not samples_path.exists():
        print(f"Error: samples.jsonl not found in {run_dir}")
        sys.exit(1)
    
    # Process samples and write quantiles
    output_path = run_dir / "samples_quantile.jsonl"
    
    print(f"Converting samples from {samples_path} to quantiles...")
    print(f"Output: {output_path}")
    
    sample_count = 0
    error_count = 0
    
    with samples_path.open("r") as f_in, output_path.open("w") as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            
            try:
                sample = json.loads(line)
                quantile_sample = _convert_sample_to_quantiles(sample, param_dists, ts_param_dists)
                f_out.write(json.dumps(quantile_sample) + "\n")
                sample_count += 1
                
                if "error" in sample:
                    error_count += 1
            except Exception as e:
                print(f"Warning: Failed to process sample: {e}")
                continue
    
    print(f"\nProcessed {sample_count} samples ({error_count} with errors)")
    print(f"Wrote quantiles to: {output_path}")


if __name__ == "__main__":
    main()




