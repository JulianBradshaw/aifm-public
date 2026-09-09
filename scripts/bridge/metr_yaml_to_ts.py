#!/usr/bin/env python3
"""
Generate utils/benchmarkData.ts from benchmark_results_1_1.yaml.

This script eliminates duplication by making the YAML file the single source of truth.

  --refresh-yaml [SRC]   Re-download benchmark_results_1_1.yaml from METR (or copy it
                         from a local path) before regenerating. The repo file is a
                         verbatim copy of METR's published asset, same filename.

NOT display-only, unlike the Epoch overlays: ac_anchor.py's _extract_metr_pairs autofits
the horizon trend from this data whenever present_horizon / present_doubling_time /
doubling_difficulty_growth_factor are left None, and compute_metr_mse always reads it.
At the current defaults all three are stated, so autofit is dormant and a refresh moves
no trajectory -- but verify with the battery rather than assuming, because the day a
default goes None this file starts steering the model.

Source:  METR, "Task-Completion Time Horizons of Frontier AI Models"
         https://metr.org/time-horizons/
         Asset: https://metr.org/assets/benchmark_results_1_1.yaml
         Analysis code: https://github.com/METR/eval-analysis-public
Cite as: @misc{metr-2026-time-horizons,
           title = {Task-Completion Time Horizons of Frontier AI Models},
           author = {METR}, url = {https://metr.org/time-horizons/}}

Caveat METR states on that page: measurements above 16 hrs are unreliable with their
current task suite, so the top p50 points are lower bounds rather than estimates.
"""

import urllib.request
import yaml
import re
import shutil
from pathlib import Path

METR_YAML_URL = "https://metr.org/assets/benchmark_results_1_1.yaml"


def is_valid_ts_identifier(key: str) -> bool:
    """Check if a key is a valid unquoted TypeScript identifier."""
    return bool(re.match(r'^[a-zA-Z_$][a-zA-Z0-9_$]*$', key))


def generate_typescript(data: dict) -> str:
    """Generate TypeScript code from benchmark data."""
    lines = []

    # Add interface definitions
    lines.append("""export interface BenchmarkResult {
  release_date: string;
  agents: {
    [key: string]: {
      p50_horizon_length?: {
        estimate: number;
        ci_low: number;
        ci_high: number;
      };
      p80_horizon_length?: {
        estimate: number;
        ci_low: number;
        ci_high: number;
      };
      average_score?: {
        estimate: number;
      };
      usage?: {
        working_time: number;
        usd: number;
      };
    };
  };
  always_show?: boolean;
}

export interface BenchmarkData {
  benchmark_name: string;
  long_tasks_version: string;
  swaa_version: string;
  results: {
    [key: string]: BenchmarkResult;
  };
}

export const benchmarkData: BenchmarkData = {""")

    # Add top-level fields
    lines.append(f'  benchmark_name: "{data["benchmark_name"]}",')
    lines.append(f'  long_tasks_version: "{data.get("long_tasks_version", "")}",')
    lines.append(f'  swaa_version: "{data.get("swaa_version", "")}",')
    lines.append("  results: {")

    # Process each model
    model_items = list(data["results"].items())
    for model_idx, (model_key, model_data) in enumerate(model_items):
        model_key_str = model_key if is_valid_ts_identifier(model_key) else f'"{model_key}"'
        lines.append(f"    {model_key_str}: {{")
        lines.append("      agents: {")

        # Determine the structure - new format uses "metrics", old uses "agents"
        if "metrics" in model_data:
            # New format: metrics + scaffolds
            metrics = model_data["metrics"]
            scaffolds = model_data.get("scaffolds", ["unknown"])
            # Use first non-null scaffold as agent key
            agent_key = next((s for s in scaffolds if s and s != "null"), scaffolds[0] if scaffolds else "unknown")
            agent_key_str = f'"{agent_key}"'

            lines.append(f"        {agent_key_str}: {{")

            # Collect agent fields to handle trailing commas
            agent_fields = []

            # Handle p50_horizon_length
            if "p50_horizon_length" in metrics:
                p50 = metrics["p50_horizon_length"]
                agent_fields.append([
                    "          p50_horizon_length: {",
                    f"            estimate: {p50['estimate']},",
                    f"            ci_low: {p50['ci_low']},",
                    f"            ci_high: {p50['ci_high']}",
                    "          }"
                ])

            # Handle p80_horizon_length
            if "p80_horizon_length" in metrics:
                p80 = metrics["p80_horizon_length"]
                agent_fields.append([
                    "          p80_horizon_length: {",
                    f"            estimate: {p80['estimate']},",
                    f"            ci_low: {p80['ci_low']},",
                    f"            ci_high: {p80['ci_high']}",
                    "          }"
                ])

            # Handle average_score
            if "average_score" in metrics:
                avg = metrics["average_score"]
                agent_fields.append([
                    "          average_score: {",
                    f"            estimate: {avg['estimate']}",
                    "          }"
                ])

            # Handle usage
            if "usage" in metrics:
                usage = metrics["usage"]
                agent_fields.append([
                    "          usage: {",
                    f"            working_time: {usage['working_time']},",
                    f"            usd: {usage['usd']}",
                    "          }"
                ])

            # Add fields with trailing commas except for the last one
            for i, field_lines in enumerate(agent_fields):
                for line in field_lines[:-1]:
                    lines.append(line)
                if i < len(agent_fields) - 1:
                    lines.append(field_lines[-1] + ",")
                else:
                    lines.append(field_lines[-1])

            lines.append("        }")

        else:
            for agent_key, agent_data in model_data["agents"].items():
                agent_key_str = f'"{agent_key}"'
                lines.append(f"        {agent_key_str}: {{")

                agent_fields = []

                if "p50_horizon_length" in agent_data:
                    p50 = agent_data["p50_horizon_length"]
                    agent_fields.append([
                        "          p50_horizon_length: {",
                        f"            estimate: {p50['estimate']},",
                        f"            ci_low: {p50['ci_low']},",
                        f"            ci_high: {p50['ci_high']}",
                        "          }"
                    ])

                if "p80_horizon_length" in agent_data:
                    p80 = agent_data["p80_horizon_length"]
                    agent_fields.append([
                        "          p80_horizon_length: {",
                        f"            estimate: {p80['estimate']},",
                        f"            ci_low: {p80['ci_low']},",
                        f"            ci_high: {p80['ci_high']}",
                        "          }"
                    ])

                if "average_score" in agent_data:
                    avg = agent_data["average_score"]
                    agent_fields.append([
                        "          average_score: {",
                        f"            estimate: {avg['estimate']}",
                        "          }"
                    ])

                if "usage" in agent_data:
                    usage = agent_data["usage"]
                    agent_fields.append([
                        "          usage: {",
                        f"            working_time: {usage['working_time']},",
                        f"            usd: {usage['usd']}",
                        "          }"
                    ])

                for i, field_lines in enumerate(agent_fields):
                    for line in field_lines[:-1]:
                        lines.append(line)
                    if i < len(agent_fields) - 1:
                        lines.append(field_lines[-1] + ",")
                    else:
                        lines.append(field_lines[-1])

                lines.append("        }")

        lines.append("      },")

        # Add release_date (convert date object to string if needed)
        release_date = model_data["release_date"]
        if hasattr(release_date, "isoformat"):
            release_date = release_date.isoformat()
        lines.append(f'      release_date: "{release_date}",')

        # Add trailing comma except for last model
        if model_idx < len(model_items) - 1:
            lines.append("    },")
        else:
            lines.append("    }")

    lines.append("  }")
    lines.append("};")

    return "\n".join(lines)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate benchmarkData.ts from benchmark_results_1_1.yaml")
    parser.add_argument("--input", "-i", help="Input YAML file (default: benchmark_results_1_1.yaml)")
    parser.add_argument("--output", "-o", help="Output file path (default: utils/benchmarkData.generated.ts)")
    parser.add_argument("--in-place", action="store_true", help="Overwrite utils/benchmarkData.ts directly")
    parser.add_argument(
        "--refresh-yaml",
        nargs="?",
        const=METR_YAML_URL,
        metavar="SRC",
        help=f"Re-fetch benchmark_results_1_1.yaml first (path or URL; default {METR_YAML_URL})",
    )
    args = parser.parse_args()

    # Determine paths relative to this script
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent  # scripts/bridge -> scripts -> project_root

    if args.input:
        yaml_path = Path(args.input)
        if not yaml_path.is_absolute():
            yaml_path = project_root / yaml_path
    else:
        yaml_path = project_root / "inputs" / "metr" / "benchmark_results_1_1.yaml"

    if args.refresh_yaml:
        src = args.refresh_yaml
        if src.startswith("http://") or src.startswith("https://"):
            with urllib.request.urlopen(src) as response:  # noqa: S310 - fixed, documented host
                yaml_path.write_bytes(response.read())
        else:
            shutil.copyfile(src, yaml_path)
        print(f"Refreshed {yaml_path} from {src}")

    if args.output:
        ts_path = Path(args.output)
    elif args.in_place:
        ts_path = project_root / "utils" / "benchmarkData.ts"
    else:
        ts_path = project_root / "utils" / "benchmarkData.generated.ts"

    # Read YAML
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    # Generate TypeScript
    ts_content = generate_typescript(data)

    # Write TypeScript file
    with open(ts_path, "w") as f:
        f.write(ts_content)

    print(f"Generated {ts_path} from {yaml_path}")


if __name__ == "__main__":
    main()
