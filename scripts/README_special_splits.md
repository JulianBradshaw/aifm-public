# Special Parameter Splits

This document describes the special threshold-based parameter splits that have been created for meaningful interpretation of certain parameters.

## Overview

In addition to the standard median-based splits, certain parameters have natural threshold values where behavior changes qualitatively. The `create_special_parameter_splits.py` script creates these threshold-based splits and generates corresponding visualizations.

## Parameters with Special Splits

### 1. Doubling Difficulty Growth Factor
- **Threshold**: 1.0
- **Split labels**: `below_1.0` (easier doubling) vs `above_1.0` (harder doubling)
- **Interpretation**:
  - `< 1.0`: Progress gets easier over time (each doubling takes less time than the previous)
  - `≥ 1.0`: Progress gets harder over time (each doubling takes more time than the previous)
- **Distribution**: ~85% below 1.0, ~15% above 1.0

### 2. Gap Years
- **Threshold**: 0.01 (effectively zero)
- **Split labels**: `no_gap` vs `has_gap`
- **Interpretation**:
  - `< 0.01`: No significant gap between AC and subsequent progress
  - `≥ 0.01`: Has a gap period where progress slows or stalls
- **Note**: In current dataset, all rollouts have gap_years ≥ 0.01

### 3. R Software (m/β)
- **Threshold**: 1.0
- **Split labels**: `below_1.0` (sublinear) vs `above_1.0` (superlinear)
- **Interpretation**:
  - `< 1.0`: Sublinear returns to scale in software progress
  - `≥ 1.0`: Linear or superlinear returns to scale in software progress
- **Distribution**: ~10% below 1.0, ~90% above 1.0

## Usage

### Create all special splits for a run:
```bash
python scripts/create_special_parameter_splits.py --run-dir outputs/251110_eli_2200
```

### Create split for a specific parameter:
```bash
python scripts/create_special_parameter_splits.py \
  --run-dir outputs/251110_eli_2200 \
  --parameter doubling_difficulty_growth_factor
```

### Generate detailed analysis plots (outcome vs parameter value):
```bash
python scripts/parameter_split_analysis.py \
  --split-dir outputs/251110_eli_2200/parameter_splits/doubling-difficulty-growth-factor_split
```

## Output Files

For each parameter split, the following files are created in `parameter_splits/<parameter-name>_split/`:

### Threshold-based split files:
- `rollouts_below_1.0.jsonl` / `rollouts_above_1.0.jsonl` (or appropriate labels)
- `milestone_pdfs_below_1.0.png` / `milestone_pdfs_above_1.0.png`
- Distribution CSV files with KDE data

### Median-based split files (pre-existing):
- `rollouts_below_median.jsonl` / `rollouts_above_median.jsonl`
- `milestone_pdfs_below_median.png` / `milestone_pdfs_above_median.png`

### Combined analysis files:
- `rollouts_combined.jsonl` - All rollouts for continuous analysis

### Analysis plots (parameter value on x-axis, outcomes on y-axis):
- `ac_timeline_probabilities.png` - P(AC by date) vs parameter value
- `sar_timeline_probabilities.png` - P(SAR by date) vs parameter value
- `ac_pdf_comparison.png` - AC arrival time histogram: above vs below median
- `sar_pdf_comparison.png` - SAR arrival time histogram: above vs below median
- `sar_one_year_takeoff_probabilities.png` - P(≤1 year takeoff from SAR) vs parameter
- `ac_one_year_takeoff_probabilities.png` - P(≤1 year takeoff from AC) vs parameter
- `sar_ai2027_speed_takeoff_probabilities.png` - P(AI 2027 speed takeoff from SAR) vs parameter
- `ac_ai2027_speed_takeoff_probabilities.png` - P(AI 2027 speed takeoff from AC) vs parameter

## Adding New Special Parameters

To add a new parameter with special threshold behavior:

1. Edit `scripts/create_special_parameter_splits.py`
2. Add entry to `SPECIAL_PARAMETERS` dict:
   ```python
   'your_parameter_name': {
       'threshold': 1.0,  # Your threshold value
       'below_label': 'descriptive_label',
       'above_label': 'descriptive_label',
       'display_name': 'Human Readable Name',
       'description': 'What this split means'
   }
   ```
3. Run the script on your data

## Example Workflow

```bash
# 1. Run batch rollouts (creates parameter_splits/ with median splits)
python scripts/batch_rollout.py --config config/sampling_config.yaml

# 2. Create special threshold-based splits
python scripts/create_special_parameter_splits.py --run-dir outputs/your_run/

# 3. Generate detailed analysis plots for each parameter
python scripts/parameter_split_analysis.py --run-dir outputs/your_run/
```
