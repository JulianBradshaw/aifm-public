#!/usr/bin/env python3
"""
Export a batch_rollout run to the Forecast page data format.

This script orchestrates the generation of all required CSV files and copies them
to the app/forecast/data/simulations/ directory structure. It also updates the
simulations.json registry.

Usage:
  # Basic export with auto-generated ID
  python scripts/export_to_forecast.py --run-dir outputs/20241210_143022 --forecaster "Eli"

  # Full control over ID and metadata
  python scripts/export_to_forecast.py \\
    --run-dir outputs/20241210_143022 \\
    --forecaster "Eli" \\
    --date "12-10-24" \\
    --custom-id "baseline" \\
    --model-version "v1.2.3" \\
    --set-default

  # Force overwrite existing simulation
  python scripts/export_to_forecast.py --run-dir outputs/20241210_143022 --forecaster "Eli" --force
"""

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


# Paths
REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
FORECAST_DATA_DIR = REPO_ROOT / "app" / "forecast" / "data"
SIMULATIONS_DIR = FORECAST_DATA_DIR / "simulations"
REGISTRY_PATH = SIMULATIONS_DIR / "simulations.json"


def validate_run_dir(run_dir: Path) -> None:
    """Validate that the run directory contains required files."""
    required_files = ["rollouts.jsonl"]

    for filename in required_files:
        if not (run_dir / filename).exists():
            raise FileNotFoundError(f"Required file not found: {run_dir / filename}")


def load_run_metadata(run_dir: Path) -> Dict[str, Any]:
    """Load metadata from the run directory."""
    metadata = {}

    # Load from metadata.json if exists
    metadata_path = run_dir / "metadata.json"
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)

    # Load from input_distributions.yaml if exists
    input_dist_path = run_dir / "input_distributions.yaml"
    if input_dist_path.exists():
        with input_dist_path.open("r", encoding="utf-8") as f:
            input_dist = yaml.safe_load(f)
            if input_dist:
                metadata["input_distributions"] = input_dist
                # Extract num_samples
                if "num_samples" in input_dist:
                    metadata["num_samples"] = input_dist["num_samples"]
                elif "num_rollouts" in input_dist:
                    metadata["num_samples"] = input_dist["num_rollouts"]
                # Extract time_range
                if "time_range" in input_dist:
                    metadata["time_range"] = input_dist["time_range"]
                # Extract present_day from parameters if available
                params = input_dist.get("parameters", {})
                present_day_spec = params.get("present_day", {})
                if isinstance(present_day_spec, dict) and "value" in present_day_spec:
                    metadata["present_day"] = present_day_spec["value"]

    return metadata


def infer_forecaster_from_config(run_dir: Path) -> Optional[str]:
    """
    Try to infer forecaster name from the config file.

    Looks at input_distributions.yaml for the original config path or naming hints.
    """
    input_dist_path = run_dir / "input_distributions.yaml"
    if not input_dist_path.exists():
        return None

    with input_dist_path.open("r", encoding="utf-8") as f:
        content = f.read()

    # Look for common forecaster names in the config
    forecaster_names = {
        "daniel": "Daniel",
        "eli": "Eli",
        "brendan": "Brendan",
    }

    content_lower = content.lower()
    for key, name in forecaster_names.items():
        if key in content_lower:
            return name

    return None


def generate_simulation_id(
    forecaster: str,
    date: str,
    custom_id: Optional[str] = None
) -> str:
    """
    Generate simulation ID from components.

    Format: {forecaster_lowercase}-{MM-DD-YY}[-{custom_id}]
    e.g., "eli-12-10-24" or "eli-12-10-24-baseline"
    """
    # Normalize forecaster name
    forecaster_slug = forecaster.lower().replace(" ", "-")

    # Build ID
    sim_id = f"{forecaster_slug}-{date}"

    if custom_id:
        custom_slug = custom_id.lower().replace(" ", "-")
        sim_id = f"{sim_id}-{custom_slug}"

    return sim_id


def ensure_overlay_distributions(run_dir: Path) -> bool:
    """
    Check if milestone_pdfs_overlay_*.csv exist, generate if missing.

    Returns True if files exist or were generated successfully.
    """
    dist_path = run_dir / "milestone_pdfs_overlay_distributions.csv"
    stats_path = run_dir / "milestone_pdfs_overlay_statistics.csv"

    if dist_path.exists() and stats_path.exists():
        print("  Overlay distributions already exist")
        return True

    print("  Generating overlay distributions...")
    generate_script = SCRIPTS_DIR / "export" / "generate_overlay_distributions.py"

    try:
        result = subprocess.run(
            [
                sys.executable,
                str(generate_script),
                "--rollouts", str(run_dir / "rollouts.jsonl"),
                "--out-dir", str(run_dir),
                "--overlay",
                "--monthly",
            ],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"    Error running generate_overlay_distributions.py: {result.stderr}")
            return False
        print("    Generated successfully")
        return True
    except Exception as e:
        print(f"    Error: {e}")
        return False


def ensure_conditional_timing(run_dir: Path, years: list = None) -> bool:
    """
    Generate conditional timing distributions if missing.

    Returns True if files exist or were generated successfully.
    """
    if years is None:
        years = [2027, 2030, 2035]

    # Check if all files exist
    all_exist = True
    for year in years:
        dist_path = run_dir / f"ac_{year}_time_until_distributions.csv"
        stats_path = run_dir / f"ac_{year}_time_until_statistics.csv"
        if not dist_path.exists() or not stats_path.exists():
            all_exist = False
            break

    if all_exist:
        print("  Conditional timing distributions already exist")
        return True

    print("  Generating conditional timing distributions...")
    generate_script = SCRIPTS_DIR / "export" / "generate_conditional_timing.py"

    try:
        result = subprocess.run(
            [
                sys.executable,
                str(generate_script),
                "--rollouts", str(run_dir / "rollouts.jsonl"),
                "--out-dir", str(run_dir),
                "--years", *[str(y) for y in years],
            ],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"    Error running generate_conditional_timing.py: {result.stderr}")
            return False
        print("    Generated successfully")
        return True
    except Exception as e:
        print(f"    Error: {e}")
        return False


MODE_BREAKOUT_FILES = [
    "milestone_pdfs_mode_breakout_distributions.csv",
    "milestone_pdfs_mode_breakout_statistics.csv",
    "milestone_pdfs_mode_breakout_cdf.csv",
]


def ensure_mode_breakout_distributions(run_dir: Path) -> bool:
    """
    Generate per-ac_anchor_mode breakout distributions if missing.

    Only meaningful for mixture runs (ac_anchor_mode sampled from a choice
    distribution). For non-mixture runs the generator writes nothing and this
    still returns True.
    """
    if all((run_dir / name).exists() for name in MODE_BREAKOUT_FILES):
        print("  Mode-breakout distributions already exist")
        return True

    print("  Generating mode-breakout distributions...")
    generate_script = SCRIPTS_DIR / "export" / "generate_mode_breakout_distributions.py"

    try:
        result = subprocess.run(
            [
                sys.executable,
                str(generate_script),
                "--rollouts", str(run_dir / "rollouts.jsonl"),
                "--out-dir", str(run_dir),
                "--monthly",
            ],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"    Error running generate_mode_breakout_distributions.py: {result.stderr}")
            return False
        print("    Generated successfully")
        return True
    except Exception as e:
        print(f"    Error: {e}")
        return False


def ensure_unconditional_timing(run_dir: Path) -> bool:
    """
    Generate unconditional timing distributions if missing.

    Returns True if files exist or were generated successfully.
    """
    # Check if unconditional files exist
    dist_path = run_dir / "ac_unconditional_time_until_distributions.csv"
    stats_path = run_dir / "ac_unconditional_time_until_statistics.csv"
    cdf_path = run_dir / "ac_unconditional_time_until_cdf.csv"

    if dist_path.exists() and stats_path.exists() and cdf_path.exists():
        print("  Unconditional timing distributions already exist")
        return True

    print("  Generating unconditional timing distributions...")
    generate_script = SCRIPTS_DIR / "export" / "generate_conditional_timing.py"

    try:
        result = subprocess.run(
            [
                sys.executable,
                str(generate_script),
                "--rollouts", str(run_dir / "rollouts.jsonl"),
                "--out-dir", str(run_dir),
                "--years", "2027",  # Need at least one year to run
                "--include-unconditional",
            ],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"    Error running generate_conditional_timing.py: {result.stderr}")
            return False
        print("    Generated successfully")
        return True
    except Exception as e:
        print(f"    Error: {e}")
        return False


def copy_data_files(
    run_dir: Path,
    target_dir: Path,
    years: list = None,
    mode_breakout: bool = False,
) -> None:
    """Copy all required CSV files to target simulation directory."""
    if years is None:
        years = [2027, 2030, 2035]

    target_dir.mkdir(parents=True, exist_ok=True)

    # Files to copy
    files_to_copy = [
        "milestone_pdfs_overlay_distributions.csv",
        "milestone_pdfs_overlay_statistics.csv",
        "milestone_pdfs_overlay_cdf.csv",
    ]

    # Optional per-ac_anchor_mode breakout files (mixture runs only)
    if mode_breakout:
        files_to_copy.extend(MODE_BREAKOUT_FILES)

    # Add conditional timing files for each year
    for year in years:
        files_to_copy.append(f"ac_{year}_time_until_distributions.csv")
        files_to_copy.append(f"ac_{year}_time_until_statistics.csv")
        files_to_copy.append(f"ac_{year}_time_until_cdf.csv")

    # Add unconditional timing files
    files_to_copy.append("ac_unconditional_time_until_distributions.csv")
    files_to_copy.append("ac_unconditional_time_until_statistics.csv")
    files_to_copy.append("ac_unconditional_time_until_cdf.csv")

    for filename in files_to_copy:
        src = run_dir / filename
        dst = target_dir / filename
        if src.exists():
            shutil.copy2(src, dst)
            print(f"  Copied {filename}")
        else:
            print(f"  Warning: {filename} not found in run directory")

    # Copy sampling configuration for reproducibility
    sampling_config_src = run_dir / "input_distributions.yaml"
    sampling_config_dst = target_dir / "sampling_config.yaml"
    if sampling_config_src.exists():
        shutil.copy2(sampling_config_src, sampling_config_dst)
        print(f"  Copied sampling_config.yaml")
    else:
        print(f"  Warning: input_distributions.yaml not found in run directory")


def write_simulation_metadata(
    target_dir: Path,
    simulation_id: str,
    forecaster: str,
    date: str,
    custom_id: Optional[str],
    model_version: Optional[str],
    run_metadata: Dict[str, Any],
    source_run_dir: Path,
) -> None:
    """Write metadata.json in the simulation directory."""
    metadata = {
        "id": simulation_id,
        "forecaster": forecaster,
        "date": date,
        "customIdentifier": custom_id,
        "sourceRunDir": str(source_run_dir),
        "modelVersion": model_version,
        "exportedAt": datetime.now().isoformat(),
    }

    # Add fields from run metadata
    if "num_samples" in run_metadata:
        metadata["numSamples"] = run_metadata["num_samples"]
    if "present_day" in run_metadata:
        metadata["presentDay"] = run_metadata["present_day"]
    if "time_range" in run_metadata:
        metadata["timeRange"] = run_metadata["time_range"]
    if "seed" in run_metadata:
        metadata["seed"] = run_metadata["seed"]

    metadata_path = target_dir / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Wrote metadata.json")


def load_registry() -> Dict[str, Any]:
    """Load the simulations registry, creating it if it doesn't exist."""
    if not REGISTRY_PATH.exists():
        return {
            "version": 1,
            "defaultSimulation": None,
            "simulations": [],
        }

    with REGISTRY_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_registry(registry: Dict[str, Any]) -> None:
    """Save the simulations registry."""
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REGISTRY_PATH.open("w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)


def update_simulations_registry(
    simulation_id: str,
    label: str,
    forecaster: str,
    date: str,
    custom_id: Optional[str],
    num_samples: Optional[int],
    model_version: Optional[str],
    set_default: bool = False,
    force: bool = False,
) -> None:
    """Add or update simulation entry in simulations.json."""
    registry = load_registry()

    # Check if simulation already exists
    existing_idx = None
    for i, sim in enumerate(registry["simulations"]):
        if sim["id"] == simulation_id:
            existing_idx = i
            break

    if existing_idx is not None and not force:
        raise ValueError(
            f"Simulation '{simulation_id}' already exists. Use --force to overwrite."
        )

    # Build simulation entry
    entry = {
        "id": simulation_id,
        "label": label,
        "forecaster": forecaster,
        "date": date,
        "customIdentifier": custom_id,
        "dataPath": f"simulations/{simulation_id}",
        "numSamples": num_samples,
        "modelVersion": model_version,
    }

    if existing_idx is not None:
        registry["simulations"][existing_idx] = entry
        print(f"  Updated existing simulation entry")
    else:
        registry["simulations"].append(entry)
        print(f"  Added new simulation entry")

    # Set as default if requested or if it's the first simulation
    if set_default or registry["defaultSimulation"] is None:
        registry["defaultSimulation"] = simulation_id
        print(f"  Set as default simulation")

    save_registry(registry)
    print(f"  Updated simulations.json")


def export_simulation(
    run_dir: Path,
    forecaster: str,
    date: str,
    custom_id: Optional[str] = None,
    model_version: Optional[str] = None,
    set_default: bool = False,
    force: bool = False,
    mode_breakout: bool = False,
) -> Path:
    """
    Export a batch_rollout run to the forecast page data format.

    Returns path to the created simulation directory.
    """
    print(f"\nExporting simulation from: {run_dir}")

    # Validate run directory
    validate_run_dir(run_dir)

    # Load run metadata
    run_metadata = load_run_metadata(run_dir)

    # Generate simulation ID
    simulation_id = generate_simulation_id(forecaster, date, custom_id)
    print(f"  Simulation ID: {simulation_id}")

    # Build label
    label = f"{forecaster} - {date}"
    if custom_id:
        label = f"{label} - {custom_id}"
    print(f"  Label: {label}")

    # Target directory
    target_dir = SIMULATIONS_DIR / simulation_id

    # Check if target exists and force is not set
    if target_dir.exists() and not force:
        raise ValueError(
            f"Target directory already exists: {target_dir}\nUse --force to overwrite."
        )

    # Ensure all required files are generated
    print("\nGenerating required files...")

    if not ensure_overlay_distributions(run_dir):
        raise RuntimeError("Failed to generate overlay distributions")

    if not ensure_conditional_timing(run_dir):
        raise RuntimeError("Failed to generate conditional timing distributions")

    if not ensure_unconditional_timing(run_dir):
        raise RuntimeError("Failed to generate unconditional timing distributions")

    if mode_breakout:
        if not ensure_mode_breakout_distributions(run_dir):
            raise RuntimeError("Failed to generate mode-breakout distributions")

    # Copy files to target directory
    print("\nCopying files to target directory...")
    copy_data_files(run_dir, target_dir, mode_breakout=mode_breakout)

    # Write simulation metadata
    print("\nWriting metadata...")
    write_simulation_metadata(
        target_dir,
        simulation_id,
        forecaster,
        date,
        custom_id,
        model_version,
        run_metadata,
        run_dir,
    )

    # Update registry
    print("\nUpdating registry...")
    update_simulations_registry(
        simulation_id,
        label,
        forecaster,
        date,
        custom_id,
        run_metadata.get("num_samples"),
        model_version,
        set_default=set_default,
        force=force,
    )

    print(f"\nExport complete!")
    print(f"  Simulation directory: {target_dir}")
    print(f"  Registry: {REGISTRY_PATH}")

    return target_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a batch_rollout run to the Forecast page data format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Path to batch_rollout output directory"
    )
    parser.add_argument(
        "--forecaster",
        type=str,
        default=None,
        help="Forecaster name (e.g., 'Eli', 'Daniel'). Auto-detected if not provided."
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Date string in MM-DD-YY format (default: today)"
    )
    parser.add_argument(
        "--custom-id",
        type=str,
        default=None,
        help="Optional custom identifier suffix for the simulation ID"
    )
    parser.add_argument(
        "--model-version",
        type=str,
        default=None,
        help="Model version string (e.g., 'v1.2.3')"
    )
    parser.add_argument(
        "--set-default",
        action="store_true",
        help="Set this simulation as the default"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing simulation with same ID"
    )
    parser.add_argument(
        "--mode-breakout",
        action="store_true",
        help=(
            "Also export per-ac_anchor_mode breakout distributions "
            "(for runs where ac_anchor_mode is sampled from a mixture)"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    # Determine forecaster
    forecaster = args.forecaster
    if not forecaster:
        forecaster = infer_forecaster_from_config(run_dir)
        if not forecaster:
            raise ValueError(
                "Could not infer forecaster name. Please provide --forecaster argument."
            )
        print(f"Inferred forecaster: {forecaster}")

    # Determine date
    date = args.date
    if not date:
        date = datetime.now().strftime("%m-%d-%y")
        print(f"Using today's date: {date}")

    # Export
    export_simulation(
        run_dir,
        forecaster,
        date,
        custom_id=args.custom_id,
        model_version=args.model_version,
        set_default=args.set_default,
        force=args.force,
        mode_breakout=args.mode_breakout,
    )


if __name__ == "__main__":
    main()
