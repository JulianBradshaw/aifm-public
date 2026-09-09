#!/usr/bin/env python3
"""
Bridge Epoch AI's company-revenue data into the frontend.

Two directions, mirroring scripts/bridge/metr_yaml_to_ts.py:

  --refresh-csv [SRC]   Re-derive the curated repo CSVs (epoch_anthropic_revenue.csv,
                        epoch_openai_revenue.csv) from Epoch's raw
                        `ai_companies_revenue_reports.csv`. SRC may be a local path
                        or a URL; it defaults to Epoch's download URL.

  (default)             Regenerate utils/revenueData.ts and utils/revenueDataOpenai.ts
                        from the curated repo CSVs. Each CSV is the single source of
                        truth for its company; the .ts files exist only because Next.js
                        cannot import a CSV from a client bundle (exactly why
                        utils/benchmarkData.ts is generated from YAML).

These datapoints are DISPLAY-ONLY. Nothing in progress_model/ or api/ reads them; they
are overlaid on the Predicted AI Revenue chart so the extrapolated trend can be eyeballed
against what Anthropic has actually reported. Changing them cannot move the model.

Source:  Epoch AI, "Data on AI Companies", https://epoch.ai/data/ai-companies
License: CC-BY 4.0 -- free to use/redistribute with attribution.
Cite as: Josh You, John Croxton, Venkat Somala, Yafah Edelman, 'Data on AI Companies'.
         Published online at epoch.ai. Retrieved from
         'https://epoch.ai/data/ai-companies' [online resource].
"""

import argparse
import csv
import io
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

EPOCH_RAW_URL = "https://epoch.ai/data/ai_companies_revenue_reports.csv"
EPOCH_DATA_PAGE = "https://epoch.ai/data/ai-companies"

# Only whole-company figures belong on the chart: the model's `present_revenue` is a
# whole-company annual revenue level, so a product-line run rate (e.g. "Claude Code $1B")
# would be an apples-to-oranges point sitting well below the trend.
SCOPE = "Full company"

# One curated CSV + generated .ts module per company. Anthropic is the primary
# series (the company `present_revenue` is anchored on): its module declares the
# ObservedRevenueRow interface and the Epoch attribution constants, and the other
# modules import the type from it.
COMPANIES = [
    {
        "company": "Anthropic",
        "csv": PROJECT_ROOT / "inputs" / "epoch" / "epoch_anthropic_revenue.csv",
        "ts": PROJECT_ROOT / "utils" / "revenueData.ts",
        "export_name": "observedRevenueRows",
        "primary": True,
    },
    {
        "company": "OpenAI",
        "csv": PROJECT_ROOT / "inputs" / "epoch" / "epoch_openai_revenue.csv",
        "ts": PROJECT_ROOT / "utils" / "revenueDataOpenai.ts",
        "export_name": "observedOpenAiRevenueRows",
        "primary": False,
    },
]

CURATED_COLUMNS = [
    "company",
    "date",
    "annualized_revenue_usd",
    "revenue_type",
    "confidence",
    "source_type",
    "source_url",
    "report_date",
]


def _read_source(src: str) -> str:
    if src.startswith("http://") or src.startswith("https://"):
        with urllib.request.urlopen(src) as response:  # noqa: S310 - fixed, documented host
            return response.read().decode("utf-8")
    return Path(src).read_text(encoding="utf-8")


def refresh_csv(src: str, spec: dict) -> int:
    """Filter Epoch's raw revenue-reports CSV down to one company's curated repo CSV."""
    rows = list(csv.DictReader(io.StringIO(_read_source(src))))

    curated = []
    for row in rows:
        if row.get("Company") != spec["company"]:
            continue
        if row.get("Scope") != SCOPE:
            continue
        raw_value = (row.get("Annualized revenue (USD)") or "").strip()
        if not raw_value:
            # Epoch carries period-revenue-only rows (e.g. "2025 full year") with no
            # annualized figure; those have nothing to plot against an annual-rate axis.
            continue
        curated.append(
            {
                "company": spec["company"],
                "date": row["Date"].strip(),
                "annualized_revenue_usd": f"{float(raw_value):.0f}",
                "revenue_type": (row.get("Annualized revenue type") or "").strip(),
                "confidence": (row.get("Confidence") or "").strip(),
                "source_type": (row.get("Source type") or "").strip(),
                "source_url": (row.get("Source 1") or "").strip(),
                "report_date": (row.get("Report date") or "").strip(),
            }
        )

    curated.sort(key=lambda r: r["date"])

    with spec["csv"].open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CURATED_COLUMNS)
        writer.writeheader()
        writer.writerows(curated)

    return len(curated)


def read_curated(spec: dict) -> list[dict]:
    with spec["csv"].open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _ts_string(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def generate_typescript(rows: list[dict], spec: dict) -> str:
    lines = [
        "// GENERATED FILE -- do not edit by hand.",
        "// Regenerate with: python scripts/bridge/epoch_revenue_to_ts.py",
        f"// Source of truth: {spec['csv'].name}",
        "//",
        f"// Data: Epoch AI, \"Data on AI Companies\" ({EPOCH_DATA_PAGE}), CC-BY 4.0.",
        "// Cite as: Josh You, John Croxton, Venkat Somala, Yafah Edelman,",
        "// 'Data on AI Companies'. Published online at epoch.ai.",
        "//",
        "// DISPLAY ONLY: these are observed, reported revenue figures overlaid on the",
        "// Predicted AI Revenue chart. They are never fed to the model or to any fit.",
        "",
    ]

    if spec["primary"]:
        lines += [
            "export interface ObservedRevenueRow {",
            "  /** Company the figure refers to. */",
            "  company: string;",
            "  /** Date the figure describes (ISO yyyy-mm-dd), not the date it was reported. */",
            "  date: string;",
            "  /** Annualized revenue in US DOLLARS (not millions, not billions, not trillions). */",
            "  annualizedRevenueUsd: number;",
            "  /** e.g. 'Annualized run rate' or 'Annual recurring revenue (ARR)'. */",
            "  revenueType: string;",
            "  /** Epoch's confidence in the figure, e.g. 'Confident' / 'Likely'. */",
            "  confidence: string;",
            "  /** e.g. 'Company disclosure' / 'Media report'. */",
            "  sourceType: string;",
            "  sourceUrl: string;",
            "  /** Date the figure was reported. */",
            "  reportDate: string;",
            "}",
            "",
            f"export const EPOCH_REVENUE_SOURCE_URL = {_ts_string(EPOCH_DATA_PAGE)};",
            'export const EPOCH_REVENUE_ATTRIBUTION = "Epoch AI, Data on AI Companies (CC-BY 4.0)";',
            "",
        ]
    else:
        lines += [
            'import type { ObservedRevenueRow } from "./revenueData";',
            "",
        ]

    lines.append(f"export const {spec['export_name']}: ObservedRevenueRow[] = [")

    for row in rows:
        lines.append("  {")
        lines.append(f"    company: {_ts_string(row['company'])},")
        lines.append(f"    date: {_ts_string(row['date'])},")
        lines.append(f"    annualizedRevenueUsd: {float(row['annualized_revenue_usd']):.0f},")
        lines.append(f"    revenueType: {_ts_string(row['revenue_type'])},")
        lines.append(f"    confidence: {_ts_string(row['confidence'])},")
        lines.append(f"    sourceType: {_ts_string(row['source_type'])},")
        lines.append(f"    sourceUrl: {_ts_string(row['source_url'])},")
        lines.append(f"    reportDate: {_ts_string(row['report_date'])},")
        lines.append("  },")

    lines.append("];")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    curated_names = ", ".join(spec["csv"].name for spec in COMPANIES)
    parser.add_argument(
        "--refresh-csv",
        nargs="?",
        const=EPOCH_RAW_URL,
        metavar="SRC",
        help=f"Re-derive {curated_names} from Epoch's raw CSV (path or URL; default {EPOCH_RAW_URL})",
    )
    args = parser.parse_args()

    for spec in COMPANIES:
        if args.refresh_csv:
            count = refresh_csv(args.refresh_csv, spec)
            print(f"Wrote {spec['csv']} ({count} datapoints) from {args.refresh_csv}")

        rows = read_curated(spec)
        spec["ts"].write_text(generate_typescript(rows, spec), encoding="utf-8")
        print(f"Wrote {spec['ts']} ({len(rows)} datapoints)")


if __name__ == "__main__":
    main()
