#!/bin/bash
# Run RolloutsReader test suite with coverage

set -e

# Get the directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$DIR/../.."

# Activate virtual environment
source "$PROJECT_ROOT/.venv/bin/activate"

# Change to project root
cd "$PROJECT_ROOT"

# Run tests with coverage
echo "Running RolloutsReader test suite..."
python -m pytest scripts/plotting_utils/test_rollouts_reader.py \
    --cov=plotting_utils.rollouts_reader \
    --cov-report=term-missing \
    --cov-report=html \
    "$@"

echo ""
echo "Coverage report saved to htmlcov/index.html"
