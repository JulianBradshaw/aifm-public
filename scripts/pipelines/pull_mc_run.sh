#!/usr/bin/env bash
# Pull a Monte Carlo batch-rollout run from the remote compute box into mc_outputs/.
#
# rollouts.jsonl is ~100MB+ and is NEVER copied. Instead the run is summarized REMOTELY
# (scripts/pipelines/summarize_rollouts.py, stdlib-only, streams the file) so the compact
# summary.json comes across with everything else (figures, csvs, samples.jsonl, configs).
#
# Usage:
#   scripts/pipelines/pull_mc_run.sh <remote_run_dir> <local_run_name> [config_name]
#
# Examples:
#   scripts/pipelines/pull_mc_run.sh /root/diag_5050_v2/20260807_032429 diag_5050_v2 \
#       sampling_config_revenue_uplift_5050
#
# The local layout mirrors the remote one: mc_outputs/<local_run_name>/<stamp>/...
# After pulling, the viewer is regenerated automatically.
set -euo pipefail

REMOTE_HOST="${AIFM_REMOTE:-aifm-runpod}"
REMOTE_PY="${AIFM_REMOTE_PY:-python3.13}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MC_ROOT="${MC_OUTPUTS_ROOT:-$REPO_ROOT/mc_outputs}"

if [[ $# -lt 2 ]]; then
  sed -n '2,20p' "${BASH_SOURCE[0]}" >&2
  exit 2
fi

REMOTE_DIR="${1%/}"
RUN_NAME="$2"
CONFIG_NAME="${3:-}"
STAMP="$(basename "$REMOTE_DIR")"
DEST="$MC_ROOT/$RUN_NAME/$STAMP"

# ssh on this box prints post-quantum key-exchange warnings on stderr; drop that noise but
# keep real failures (set -e still catches non-zero exits).
quiet_ssh() { ssh -o BatchMode=yes "$@" 2> >(grep -v -e 'post-quantum' -e 'Warning: ' >&2); }

echo "==> uploading summarizer to $REMOTE_HOST"
scp -q "$SCRIPT_DIR/summarize_rollouts.py" "$REMOTE_HOST:/tmp/summarize_rollouts.py" \
  2> >(grep -v -e 'post-quantum' -e 'Warning: ' >&2)

echo "==> summarizing $REMOTE_DIR remotely (streams rollouts.jsonl, writes summary.json)"
CFG_ARG=""
[[ -n "$CONFIG_NAME" ]] && CFG_ARG="--config-name $CONFIG_NAME"
quiet_ssh "$REMOTE_HOST" \
  "$REMOTE_PY /tmp/summarize_rollouts.py '$REMOTE_DIR' --force $CFG_ARG"

echo "==> generating forecast topline remotely (ensure_overlay_distributions, ~1s with a cache)"
REMOTE_REPO="${AIFM_REMOTE_REPO:-/root/uplift-clean}"
if ! quiet_ssh "$REMOTE_HOST" "cd '$REMOTE_REPO' && $REMOTE_PY -c \"
import sys; sys.path.insert(0, 'scripts/pipelines')
from pathlib import Path
import forecast_export as fe
run = Path('$REMOTE_DIR')
ok = fe.ensure_overlay_distributions(run)
for fn in ('ensure_conditional_timing', 'ensure_unconditional_timing'):
    f = getattr(fe, fn, None)
    f and f(run)
raise SystemExit(0 if ok else 1)
\""; then
  echo "WARNING: forecast topline generation FAILED — the viewer will show its no-forecast_data placeholder for this run" >&2
fi

echo "==> rsyncing to $DEST (excluding rollouts.jsonl; forecast CSVs go to forecast_data/)"
mkdir -p "$DEST"
rsync -az --info=stats1 --partial \
  --exclude 'rollouts.jsonl' \
  --exclude 'rollouts.cache.json' \
  --exclude 'median_trajectory_rollout.jsonl' \
  --exclude 'milestone_pdfs_overlay_*.csv' \
  --exclude 'ac_*_time_until_*.csv' \
  -e "ssh -o BatchMode=yes" \
  "$REMOTE_HOST:$REMOTE_DIR/" "$DEST/" \
  2> >(grep -v -e 'post-quantum' -e '^\*\*' -e '^$' >&2)

echo "==> pulling forecast topline CSVs into forecast_data/"
mkdir -p "$DEST/forecast_data"
rsync -az \
  --include 'milestone_pdfs_overlay_*.csv' \
  --include 'ac_*_time_until_*.csv' \
  --exclude '*' \
  -e "ssh -o BatchMode=yes" \
  "$REMOTE_HOST:$REMOTE_DIR/" "$DEST/forecast_data/" \
  2> >(grep -v -e 'post-quantum' -e '^\*\*' -e '^$' >&2)

echo "==> regenerating viewer"
PY="$REPO_ROOT/.venv/bin/python"
[[ -x "$PY" ]] || PY="$(command -v python3)"
"$PY" "$SCRIPT_DIR/make_mc_viewer.py" --root "$MC_ROOT"
