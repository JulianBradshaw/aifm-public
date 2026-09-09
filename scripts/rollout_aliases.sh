#!/bin/bash
# Convenience aliases and functions for rollout analysis
#
# Usage: Source this file in your shell to get convenient aliases
#   source scripts/rollout_aliases.sh
#
# Or add to your ~/.bashrc or ~/.zshrc:
#   source /path/to/ai-futures-calculator/scripts/rollout_aliases.sh

# Get the directory where this script is located
if [ -n "${BASH_SOURCE[0]}" ]; then
    SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
    REPO_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
else
    # Fallback for when not sourced from a file
    REPO_ROOT="$(pwd)"
    while [ ! -f "$REPO_ROOT/CLAUDE.md" ] && [ "$REPO_ROOT" != "/" ]; do
        REPO_ROOT="$(dirname "$REPO_ROOT")"
    done
fi

# Quick scan of all rollouts
alias rollout-stats='python3 "$REPO_ROOT/scripts/scan_all_rollouts.py"'

# View summary for a specific run
# Usage: rollout-summary 1105_daniel
rollout-summary() {
    if [ -z "$1" ]; then
        echo "Usage: rollout-summary <run_dir_name>"
        echo "Example: rollout-summary 1105_daniel"
        return 1
    fi

    local run_dir="$REPO_ROOT/outputs/$1"
    if [ ! -d "$run_dir" ]; then
        echo "Error: Run directory not found: $run_dir"
        return 1
    fi

    if [ -f "$run_dir/rollout_summary.txt" ]; then
        cat "$run_dir/rollout_summary.txt"
    else
        echo "Generating summary for $1..."
        python3 "$REPO_ROOT/scripts/rollout_summary.py" "$run_dir"
    fi
}

# View summary for the most recent run
rollout-summary-latest() {
    local latest_dir=$(ls -t "$REPO_ROOT/outputs" | head -1)
    if [ -z "$latest_dir" ]; then
        echo "Error: No output directories found"
        return 1
    fi
    echo "Latest run: $latest_dir"
    rollout-summary "$latest_dir"
}

# List all output directories with rollouts
rollout-list() {
    echo "Available rollout runs:"
    ls -t "$REPO_ROOT/outputs" | while read dir; do
        if [ -f "$REPO_ROOT/outputs/$dir/rollouts.jsonl" ]; then
            local count=$(wc -l < "$REPO_ROOT/outputs/$dir/rollouts.jsonl" 2>/dev/null || echo "?")
            echo "  $dir ($count rollouts)"
        fi
    done
}

echo "Rollout analysis commands loaded:"
echo "  rollout-stats            - View statistics for all runs"
echo "  rollout-summary <dir>    - View summary for a specific run"
echo "  rollout-summary-latest   - View summary for the most recent run"
echo "  rollout-list             - List all available rollout runs"
