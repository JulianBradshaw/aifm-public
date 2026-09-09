#!/usr/bin/env python3
"""
Bridge Epoch AI's Capabilities Index (ECI) scores into the frontier overlay.

Two directions, mirroring scripts/bridge/epoch_revenue_to_ts.py:

  --refresh-csv [SRC]   Re-extract epoch_capabilities_index.csv from Epoch's
                        benchmark_data.zip. SRC may be a local path or a URL; it
                        defaults to Epoch's download URL. The repo CSV is a verbatim
                        copy of the zip member of the same name -- no reshaping, so
                        the ECI consumers keep reading it unchanged.

  (default)             Regenerate utils/eciData.ts from the repo CSV, reduced to the
                        ECI FRONTIER (see below).

WHY THE FRONTIER IS COMPUTED HERE, NOT IN TYPESCRIPT
The CSV carries ~800 rows, most of them non-frontier or unscored. Shipping all of
them to the browser to throw ~96% away would bloat the bundle for nothing, so the
reduction happens at generation time and only the frontier reaches utils/eciData.ts.
(utils/benchmarkLoader.ts filters METR to the frontier in TS instead; that dataset is
an order of magnitude smaller.) The frontier rule is pinned by tests/test_epoch_eci_data.py,
which recomputes it from the CSV and compares.

FRONTIER DEFINITION
The frontier is the running maximum of ECI score over release date. Concretely:

  * Rows without an ECI score or without a release date are dropped -- they cannot
    be placed on the chart.
  * Models are grouped by release date and each date is represented by its single
    best-scoring model, so the frontier has at most one point per date and never
    stacks markers on one x position.
  * That date's best score must STRICTLY exceed the best score of every model
    released on any strictly earlier date. Equal scores therefore do not produce a
    new point: the frontier is a strictly increasing step function.
  * Ties for "best on this date" are broken by model version string, purely so
    regeneration is deterministic. A tie only matters when the tied models would
    also beat the prior maximum, and then the two are interchangeable by definition.

All organizations compete for the frontier; nothing is filtered by lab.

Source:  Epoch AI, "AI Benchmarking Hub", https://epoch.ai/benchmarks
         CSV ships inside https://epoch.ai/data/benchmark_data.zip
License: CC-BY 4.0 -- free to use/redistribute with attribution.
Cite as: Epoch AI, 'AI Benchmarking Hub'. Published online at epoch.ai. Retrieved
         from 'https://epoch.ai/benchmarks' [online resource].
"""

import argparse
import csv
import io
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

EPOCH_BENCHMARK_ZIP_URL = "https://epoch.ai/data/benchmark_data.zip"
EPOCH_BENCHMARKS_PAGE = "https://epoch.ai/benchmarks"
ZIP_MEMBER = "epoch_capabilities_index.csv"

ECI_CSV = PROJECT_ROOT / "inputs" / "epoch" / ZIP_MEMBER
OUTPUT_TS = PROJECT_ROOT / "utils" / "eciData.ts"


def _read_bytes(src: str) -> bytes:
    if src.startswith("http://") or src.startswith("https://"):
        with urllib.request.urlopen(src) as response:  # noqa: S310 - fixed, documented host
            return response.read()
    return Path(src).read_bytes()


def refresh_csv(src: str) -> int:
    """Extract the ECI CSV verbatim from Epoch's benchmark_data.zip."""
    archive = zipfile.ZipFile(io.BytesIO(_read_bytes(src)))
    payload = archive.read(ZIP_MEMBER)
    ECI_CSV.write_bytes(payload)
    return len(payload.decode("utf-8").splitlines()) - 1


def read_csv_rows() -> list[dict]:
    with ECI_CSV.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def plottable_rows(rows: list[dict]) -> list[dict]:
    """Rows that can be placed on the chart: a numeric score and a release date."""
    usable = []
    for row in rows:
        raw_score = (row.get("ECI Score") or "").strip()
        release_date = (row.get("Release date") or "").strip()
        if not raw_score or not release_date:
            continue
        try:
            float(raw_score)
        except ValueError:
            continue
        usable.append(row)
    return usable


def compute_frontier(rows: list[dict]) -> list[dict]:
    """Running maximum of ECI score over release date. See module docstring."""
    by_date: dict[str, list[dict]] = defaultdict(list)
    for row in plottable_rows(rows):
        by_date[row["Release date"].strip()].append(row)

    frontier: list[dict] = []
    running_max = float("-inf")
    for release_date in sorted(by_date):
        best = max(
            by_date[release_date],
            key=lambda row: (float(row["ECI Score"]), row["Model version"]),
        )
        score = float(best["ECI Score"])
        if score > running_max:
            running_max = score
            frontier.append(best)
    return frontier


def display_name_for(row: dict) -> str:
    """Epoch leaves 'Display name' blank on older entries; fall back sensibly."""
    for column in ("Display name", "Model name", "Model version"):
        value = (row.get(column) or "").strip()
        if value:
            return value
    return "Unknown model"


def _ts_string(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def generate_typescript(frontier: list[dict], total_rows: int, plottable: int) -> str:
    latest_release = max(row["Release date"].strip() for row in frontier)
    lines = [
        "// GENERATED FILE -- do not edit by hand.",
        "// Regenerate with: python scripts/bridge/epoch_eci_to_ts.py",
        "// Source of truth: epoch_capabilities_index.csv",
        "//",
        f"// Data: Epoch AI, \"AI Benchmarking Hub\" ({EPOCH_BENCHMARKS_PAGE}), CC-BY 4.0.",
        "// Cite as: Epoch AI, 'AI Benchmarking Hub'. Published online at epoch.ai.",
        f"// CSV ships inside {EPOCH_BENCHMARK_ZIP_URL}",
        "//",
        "// Reduced to the ECI FRONTIER: the running maximum of score over release date,",
        "// one point per date, strictly increasing. All organizations compete for it.",
        f"// {len(frontier)} frontier models out of {plottable} scored-and-dated rows"
        f" ({total_rows} rows total).",
        "//",
        "// DISPLAY ONLY: overlaid on the Epoch Capabilities Index chart for comparison.",
        "// Never fed to the model or to any fit.",
        "",
        "export interface EciFrontierRow {",
        "  /** ISO yyyy-mm-dd release date of the model. */",
        "  releaseDate: string;",
        "  /** Epoch Capabilities Index score (dimensionless, ~100-200 over the covered era). */",
        "  eciScore: number;",
        "  /** Lab that released the model, verbatim from Epoch. */",
        "  organization: string;",
        "  /** Human-readable model name for the tooltip. */",
        "  displayName: string;",
        "  /** Epoch's model version identifier. */",
        "  modelVersion: string;",
        "}",
        "",
        f"export const EPOCH_ECI_SOURCE_URL = {_ts_string(EPOCH_BENCHMARKS_PAGE)};",
        'export const EPOCH_ECI_ATTRIBUTION = "Epoch AI, AI Benchmarking Hub (CC-BY 4.0)";',
        f"/** Newest frontier release in this snapshot; the CSV's vintage in one value. */",
        f"export const EPOCH_ECI_LATEST_RELEASE_DATE = {_ts_string(latest_release)};",
        "",
        "export const eciFrontierRows: EciFrontierRow[] = [",
    ]

    for row in frontier:
        lines.append("  {")
        lines.append(f"    releaseDate: {_ts_string(row['Release date'].strip())},")
        lines.append(f"    eciScore: {float(row['ECI Score'])},")
        lines.append(f"    organization: {_ts_string((row.get('Organization') or '').strip())},")
        lines.append(f"    displayName: {_ts_string(display_name_for(row))},")
        lines.append(f"    modelVersion: {_ts_string((row.get('Model version') or '').strip())},")
        lines.append("  },")

    lines.append("];")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--refresh-csv",
        nargs="?",
        const=EPOCH_BENCHMARK_ZIP_URL,
        metavar="SRC",
        help=(
            f"Re-extract {ZIP_MEMBER} from Epoch's benchmark_data.zip "
            f"(path or URL; default {EPOCH_BENCHMARK_ZIP_URL})"
        ),
    )
    parser.add_argument("--output", "-o", help=f"Output .ts path (default {OUTPUT_TS})")
    args = parser.parse_args()

    if args.refresh_csv:
        count = refresh_csv(args.refresh_csv)
        print(f"Wrote {ECI_CSV} ({count} rows) from {args.refresh_csv}")

    rows = read_csv_rows()
    usable = plottable_rows(rows)
    frontier = compute_frontier(rows)

    ts_path = Path(args.output) if args.output else OUTPUT_TS
    ts_path.write_text(generate_typescript(frontier, len(rows), len(usable)), encoding="utf-8")
    print(
        f"Wrote {ts_path}: {len(frontier)} frontier models "
        f"out of {len(usable)} scored-and-dated rows ({len(rows)} rows total)"
    )


if __name__ == "__main__":
    main()
