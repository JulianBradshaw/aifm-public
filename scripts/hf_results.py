#!/usr/bin/env python3
"""Upload/download Monte Carlo rollout results to/from HuggingFace Hub.

Usage:
    # Upload a local results folder
    python scripts/hf_results.py upload outputs/results_20260117_205600

    # Download a specific results folder
    python scripts/hf_results.py download results_20260117_205600

    # Download all results
    python scripts/hf_results.py download --all

    # List what's on HF
    python scripts/hf_results.py list

    # Verify uploads match local files, then delete local source files
    python scripts/hf_results.py clean outputs/results_20260117_205600
"""

import argparse
import sys
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download, list_repo_tree

DEFAULT_REPO = "bhalstead/ai-futures-calculator-results"

# Source files that are irreplaceable (everything else is derived).
SOURCE_FILES = {
    "rollouts.jsonl",
    "samples.jsonl",
    "input_distributions.yaml",
    "metadata.json",
    "correlation_matrix.json",
    "model_config_snapshot.json",
    "model_config_snapshot.py",
}


def find_source_files(results_dir: Path) -> list[tuple[Path, str]]:
    """Walk a results folder and return (local_path, hf_path) pairs for source files.

    Expected structure: results_dir / scenario / timestamp / {source files}
    HF path mirrors local: results_XXXX / scenario / timestamp / file
    """
    results_name = results_dir.name  # e.g. results_20260117_205600
    pairs = []
    for scenario_dir in sorted(results_dir.iterdir()):
        if not scenario_dir.is_dir():
            continue
        for run_dir in sorted(scenario_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            for fname in sorted(run_dir.iterdir()):
                if fname.name in SOURCE_FILES and fname.is_file():
                    hf_path = f"{results_name}/{scenario_dir.name}/{run_dir.name}/{fname.name}"
                    pairs.append((fname, hf_path))
    return pairs


def cmd_upload(args):
    """Upload a local results folder to HuggingFace."""
    results_dir = Path(args.path).resolve()
    if not results_dir.is_dir():
        print(f"Error: {results_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    pairs = find_source_files(results_dir)
    if not pairs:
        print(f"No source files found in {results_dir}", file=sys.stderr)
        sys.exit(1)

    api = HfApi()

    print(f"Uploading {len(pairs)} files to {args.repo}...")
    for local_path, hf_path in pairs:
        size_mb = local_path.stat().st_size / (1024 * 1024)
        print(f"  {hf_path} ({size_mb:.1f} MB)")

    if args.dry_run:
        print("\n(dry run — no files uploaded)")
        return

    # Upload files one at a time so we get per-file progress for large files.
    for local_path, hf_path in pairs:
        size_mb = local_path.stat().st_size / (1024 * 1024)
        print(f"Uploading {hf_path} ({size_mb:.1f} MB)...")
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=hf_path,
            repo_id=args.repo,
            repo_type="dataset",
        )

    print(f"\nDone. Uploaded {len(pairs)} files.")


def cmd_download(args):
    """Download results from HuggingFace to local outputs/."""
    api = HfApi()
    output_root = Path("outputs")

    # List all files in the repo.
    all_files = [
        f.rfilename
        for f in list_repo_tree(args.repo, repo_type="dataset", recursive=True)
        if hasattr(f, "rfilename")
    ]

    # Filter to the requested results folder(s).
    if args.all:
        targets = all_files
    else:
        prefix = args.name
        # Allow with or without trailing slash.
        if not prefix.endswith("/"):
            prefix_match = prefix + "/"
        else:
            prefix_match = prefix
        targets = [f for f in all_files if f.startswith(prefix_match) or f == prefix]

    if not targets:
        print(f"No files found matching '{args.name}' in {args.repo}", file=sys.stderr)
        sys.exit(1)

    # Only download source files.
    targets = [f for f in targets if Path(f).name in SOURCE_FILES]

    print(f"Downloading {len(targets)} source files from {args.repo}...")

    if args.dry_run:
        for f in targets:
            local = output_root / f
            status = "EXISTS" if local.exists() else "NEW"
            print(f"  [{status}] {f}")
        print("\n(dry run — no files downloaded)")
        return

    downloaded = 0
    skipped = 0
    for hf_path in targets:
        local_path = output_root / hf_path
        if local_path.exists() and not args.force:
            skipped += 1
            continue
        local_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"  Downloading {hf_path}...")
        hf_hub_download(
            repo_id=args.repo,
            filename=hf_path,
            repo_type="dataset",
            local_dir=str(output_root),
        )
        downloaded += 1

    print(f"\nDone. Downloaded {downloaded}, skipped {skipped} (already exist).")


def cmd_list(args):
    """List contents of the HF repo."""
    all_files = [
        f
        for f in list_repo_tree(args.repo, repo_type="dataset", recursive=True)
        if hasattr(f, "rfilename")
    ]

    if not all_files:
        print(f"No files found in {args.repo}")
        return

    for f in sorted(all_files, key=lambda x: x.rfilename):
        size_mb = f.size / (1024 * 1024) if hasattr(f, "size") and f.size else 0
        marker = " *" if Path(f.rfilename).name in SOURCE_FILES else ""
        print(f"  {f.rfilename} ({size_mb:.1f} MB){marker}")

    print(f"\n{len(all_files)} files total. * = source file")


def _verify_results(results_dir: Path, repo: str) -> tuple[list[str], list[str]]:
    """Compare local source files against HF. Returns (ok_files, mismatches)."""
    pairs = find_source_files(results_dir)
    if not pairs:
        return [], ["No source files found locally"]

    # Build a map of HF path -> size from the repo.
    hf_files = {
        f.rfilename: f.size
        for f in list_repo_tree(repo, repo_type="dataset", recursive=True)
        if hasattr(f, "rfilename") and hasattr(f, "size")
    }

    ok = []
    mismatches = []
    for local_path, hf_path in pairs:
        local_size = local_path.stat().st_size
        if hf_path not in hf_files:
            mismatches.append(f"MISSING on HF: {hf_path}")
        elif hf_files[hf_path] != local_size:
            mismatches.append(
                f"SIZE MISMATCH: {hf_path} "
                f"(local={local_size:,}, HF={hf_files[hf_path]:,})"
            )
        else:
            ok.append(hf_path)

    return ok, mismatches


def cmd_verify(args):
    """Verify local source files match what's on HF."""
    results_dir = Path(args.path).resolve()
    if not results_dir.is_dir():
        print(f"Error: {results_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    print(f"Verifying {results_dir.name} against {args.repo}...")
    ok, mismatches = _verify_results(results_dir, args.repo)

    for f in ok:
        print(f"  OK  {f}")
    for m in mismatches:
        print(f"  FAIL  {m}")

    print(f"\n{len(ok)} OK, {len(mismatches)} failed.")
    if mismatches:
        sys.exit(1)


def cmd_clean(args):
    """Delete local source files after verifying they exist on HF."""
    results_dir = Path(args.path).resolve()
    if not results_dir.is_dir():
        print(f"Error: {results_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    print(f"Verifying {results_dir.name} against {args.repo}...")
    ok, mismatches = _verify_results(results_dir, args.repo)

    if mismatches:
        for m in mismatches:
            print(f"  FAIL  {m}")
        print("\nVerification failed — not deleting anything.", file=sys.stderr)
        sys.exit(1)

    pairs = find_source_files(results_dir)
    total_bytes = sum(p.stat().st_size for p, _ in pairs)
    total_gb = total_bytes / (1024**3)

    print(f"All {len(ok)} files verified. Will delete {total_gb:.1f} GB of local source files:")
    for local_path, hf_path in pairs:
        size_mb = local_path.stat().st_size / (1024 * 1024)
        print(f"  {hf_path} ({size_mb:.1f} MB)")

    if args.dry_run:
        print("\n(dry run — no files deleted)")
        return

    if not args.yes:
        answer = input(f"\nDelete {len(pairs)} local files ({total_gb:.1f} GB)? [y/N] ")
        if answer.lower() not in ("y", "yes"):
            print("Aborted.")
            return

    deleted = 0
    for local_path, hf_path in pairs:
        local_path.unlink()
        deleted += 1

    print(f"\nDeleted {deleted} local files ({total_gb:.1f} GB freed).")


def cmd_delete(args):
    """Delete files from the HF repo (for reorganization)."""
    api = HfApi()

    all_files = [
        f.rfilename
        for f in list_repo_tree(args.repo, repo_type="dataset", recursive=True)
        if hasattr(f, "rfilename")
    ]

    if args.prefix:
        targets = [f for f in all_files if f.startswith(args.prefix)]
    else:
        print("Error: --prefix is required for delete", file=sys.stderr)
        sys.exit(1)

    if not targets:
        print(f"No files matching prefix '{args.prefix}'")
        return

    print(f"Will delete {len(targets)} files from {args.repo}:")
    for f in targets:
        print(f"  {f}")

    if args.dry_run:
        print("\n(dry run — no files deleted)")
        return

    if not args.yes:
        answer = input(f"\nDelete {len(targets)} files from {args.repo}? [y/N] ")
        if answer.lower() not in ("y", "yes"):
            print("Aborted.")
            return

    for f in targets:
        print(f"  Deleting {f}...")
        api.delete_file(path_in_repo=f, repo_id=args.repo, repo_type="dataset")

    print(f"\nDone. Deleted {len(targets)} files.")


def main():
    parser = argparse.ArgumentParser(
        description="Manage Monte Carlo results on HuggingFace Hub"
    )
    parser.add_argument(
        "--repo", default=DEFAULT_REPO, help=f"HF repo ID (default: {DEFAULT_REPO})"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # upload
    p_upload = sub.add_parser("upload", help="Upload a local results folder to HF")
    p_upload.add_argument("path", help="Path to local results folder (e.g. outputs/results_20260117_205600)")
    p_upload.add_argument("--dry-run", action="store_true", help="Show what would be uploaded")
    p_upload.set_defaults(func=cmd_upload)

    # download
    p_download = sub.add_parser("download", help="Download results from HF")
    p_download.add_argument("name", nargs="?", help="Results folder name (e.g. results_20260117_205600)")
    p_download.add_argument("--all", action="store_true", help="Download all results")
    p_download.add_argument("--force", action="store_true", help="Overwrite existing files")
    p_download.add_argument("--dry-run", action="store_true", help="Show what would be downloaded")
    p_download.set_defaults(func=cmd_download)

    # list
    p_list = sub.add_parser("list", help="List files in the HF repo")
    p_list.set_defaults(func=cmd_list)

    # verify
    p_verify = sub.add_parser("verify", help="Verify local files match HF")
    p_verify.add_argument("path", help="Path to local results folder")
    p_verify.set_defaults(func=cmd_verify)

    # clean
    p_clean = sub.add_parser("clean", help="Delete local source files after verifying they're on HF")
    p_clean.add_argument("path", help="Path to local results folder")
    p_clean.add_argument("--dry-run", action="store_true", help="Show what would be deleted")
    p_clean.add_argument("-y", "--yes", action="store_true", help="Skip confirmation prompt")
    p_clean.set_defaults(func=cmd_clean)

    # delete
    p_delete = sub.add_parser("delete", help="Delete files from the HF repo")
    p_delete.add_argument("--prefix", required=True, help="Path prefix to delete")
    p_delete.add_argument("--dry-run", action="store_true", help="Show what would be deleted")
    p_delete.add_argument("-y", "--yes", action="store_true", help="Skip confirmation prompt")
    p_delete.set_defaults(func=cmd_delete)

    args = parser.parse_args()

    if args.command == "download" and not args.all and not args.name:
        parser.error("download requires a results folder name or --all")

    args.func(args)


if __name__ == "__main__":
    main()
