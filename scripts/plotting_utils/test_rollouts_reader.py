"""
Comprehensive test suite for RolloutsReader class.

Tests cover:
- JSONL file reading
- Cache file creation and loading
- Milestone time extraction
- Transition duration calculation
- Trajectory reading
- Edge cases and error handling
"""

import json
import tempfile
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pytest

# Import from plotting_utils package
from plotting_utils.rollouts_reader import RolloutsReader, _RolloutRecord


class TestRolloutsReaderBasics:
    """Test basic initialization and file handling."""

    def test_init_nonexistent_file(self):
        """Test that initializing with nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            RolloutsReader(Path("/nonexistent/rollouts.jsonl"))

    def test_init_with_valid_jsonl(self, tmp_path):
        """Test successful initialization with valid JSONL file."""
        rollouts_file = tmp_path / "rollouts.jsonl"
        rollouts_file.write_text('{"results": {"times": [2025.0, 2026.0]}}\n')

        reader = RolloutsReader(rollouts_file)
        assert reader.rollouts_file == rollouts_file

    def test_cache_file_detection(self, tmp_path):
        """Test that cache files are automatically detected."""
        rollouts_file = tmp_path / "rollouts.jsonl"
        cache_file = tmp_path / "rollouts.cache.json"

        # Create minimal JSONL
        rollouts_file.write_text('{"results": {"times": [2025.0]}}\n')

        # Create cache
        cache_data = {
            "version": 1,
            "source": "rollouts.jsonl",
            "rollouts": [{"results": {"times": [2025.0], "simulation_end": 2025.0}}]
        }
        cache_file.write_text(json.dumps(cache_data))

        # Reader should auto-detect and use cache
        reader = RolloutsReader(rollouts_file)
        assert reader.rollouts_file == cache_file
        assert reader._cache_data is not None

    def test_direct_cache_file_loading(self, tmp_path):
        """Test that cache files can be loaded directly."""
        cache_file = tmp_path / "rollouts.cache.json"

        cache_data = {
            "version": 1,
            "source": "rollouts.jsonl",
            "rollouts": [{"results": {"times": [2025.0], "simulation_end": 2025.0}}]
        }
        cache_file.write_text(json.dumps(cache_data))

        reader = RolloutsReader(cache_file)
        assert reader.rollouts_file == cache_file
        assert reader._cache_data is not None


class TestRolloutParsing:
    """Test parsing of individual rollout records."""

    def create_test_rollout(
        self,
        times: List[float] = None,
        milestones: Dict[str, Dict[str, Any]] = None,
        aa_time: float = None,
    ) -> Dict[str, Any]:
        """Helper to create a test rollout record."""
        if times is None:
            times = [2025.0, 2026.0, 2027.0]
        if milestones is None:
            milestones = {}

        return {
            "results": {
                "times": times,
                "milestones": milestones,
                "aa_time": aa_time,
                "ai_research_taste": [0.5, 0.6, 0.7],
                "effective_compute": [1.0, 2.0, 3.0],
            }
        }

    def test_parse_valid_rollout(self, tmp_path):
        """Test parsing a complete valid rollout."""
        rollouts_file = tmp_path / "rollouts.jsonl"
        rollout = self.create_test_rollout(
            times=[2025.0, 2026.0],
            milestones={"AC": {"time": 2025.5, "effective_compute_ooms": 30.0}},
            aa_time=2025.5
        )
        rollouts_file.write_text(json.dumps(rollout) + '\n')

        reader = RolloutsReader(rollouts_file)
        records = list(reader.iter_normalized_rollouts())

        assert len(records) == 1
        rec = records[0]
        assert isinstance(rec, _RolloutRecord)
        assert len(rec.times) == 2
        assert rec.aa_time == 2025.5
        assert rec.simulation_end == 2026.0
        assert "AC" in rec.milestones

    def test_parse_missing_results(self, tmp_path):
        """Test that records without results are skipped."""
        rollouts_file = tmp_path / "rollouts.jsonl"
        rollouts_file.write_text('{"error": "some error"}\n')

        reader = RolloutsReader(rollouts_file)
        records = list(reader.iter_normalized_rollouts())

        assert len(records) == 0

    def test_parse_empty_times(self, tmp_path):
        """Test that records with empty times are skipped."""
        rollouts_file = tmp_path / "rollouts.jsonl"
        rollout = {"results": {"times": []}}
        rollouts_file.write_text(json.dumps(rollout) + '\n')

        reader = RolloutsReader(rollouts_file)
        records = list(reader.iter_normalized_rollouts())

        assert len(records) == 0

    def test_parse_invalid_aa_time(self, tmp_path):
        """Test handling of invalid aa_time values."""
        rollouts_file = tmp_path / "rollouts.jsonl"

        # Test with NaN aa_time
        rollout = self.create_test_rollout(aa_time=float('nan'))
        rollouts_file.write_text(json.dumps(rollout) + '\n')

        reader = RolloutsReader(rollouts_file)
        records = list(reader.iter_normalized_rollouts())

        assert len(records) == 1
        assert records[0].aa_time is None

    def test_parse_malformed_json(self, tmp_path):
        """Test that malformed JSON lines are skipped."""
        rollouts_file = tmp_path / "rollouts.jsonl"
        rollouts_file.write_text('{"results": invalid json}\n{"results": {"times": [2025.0]}}\n')

        reader = RolloutsReader(rollouts_file)
        records = list(reader.iter_normalized_rollouts())

        # Only the second valid record should be parsed
        assert len(records) == 1


class TestMilestoneExtraction:
    """Test milestone time and compute extraction."""

    def create_rollouts_with_milestones(self, tmp_path) -> Path:
        """Create test file with multiple rollouts and milestones."""
        rollouts_file = tmp_path / "rollouts.jsonl"

        rollouts = [
            # Rollout 1: AC achieved
            {
                "results": {
                    "times": [2025.0, 2030.0],
                    "milestones": {
                        "AC": {"time": 2027.0, "effective_compute_ooms": 30.0},
                        "SAR": {"time": 2029.0, "effective_compute_ooms": 35.0}
                    },
                    "aa_time": 2027.0
                }
            },
            # Rollout 2: Only AC achieved
            {
                "results": {
                    "times": [2025.0, 2030.0],
                    "milestones": {
                        "AC": {"time": 2028.0, "effective_compute_ooms": 31.0}
                    },
                    "aa_time": 2028.0
                }
            },
            # Rollout 3: Neither achieved
            {
                "results": {
                    "times": [2025.0, 2030.0],
                    "milestones": {},
                    "aa_time": None
                }
            },
        ]

        with rollouts_file.open('w') as f:
            for rollout in rollouts:
                f.write(json.dumps(rollout) + '\n')

        return rollouts_file

    def test_read_milestone_times(self, tmp_path):
        """Test reading milestone arrival times."""
        rollouts_file = self.create_rollouts_with_milestones(tmp_path)
        reader = RolloutsReader(rollouts_file)

        times, not_achieved, sim_end = reader.read_milestone_times("AC")

        assert len(times) == 2
        assert 2027.0 in times
        assert 2028.0 in times
        assert not_achieved == 1
        assert sim_end == 2030.0

    def test_read_milestone_compute(self, tmp_path):
        """Test reading milestone effective compute values."""
        rollouts_file = self.create_rollouts_with_milestones(tmp_path)
        reader = RolloutsReader(rollouts_file)

        compute_ooms, not_achieved = reader.read_milestone_compute("AC")

        assert len(compute_ooms) == 2
        assert 30.0 in compute_ooms
        assert 31.0 in compute_ooms
        assert not_achieved == 1

    def test_read_nonexistent_milestone(self, tmp_path):
        """Test reading a milestone that doesn't exist."""
        rollouts_file = self.create_rollouts_with_milestones(tmp_path)
        reader = RolloutsReader(rollouts_file)

        times, not_achieved, _ = reader.read_milestone_times("NONEXISTENT")

        assert len(times) == 0
        assert not_achieved == 3

    def test_read_aa_times(self, tmp_path):
        """Test reading aa_time values."""
        rollouts_file = self.create_rollouts_with_milestones(tmp_path)
        reader = RolloutsReader(rollouts_file)

        aa_times, no_sc, sim_end = reader.read_aa_times()

        assert len(aa_times) == 2
        assert 2027.0 in aa_times
        assert 2028.0 in aa_times
        assert no_sc == 1
        assert sim_end == 2030.0


class TestTransitionDurations:
    """Test transition duration calculations between milestones."""

    def create_transition_rollouts(self, tmp_path) -> Path:
        """Create test file with transition data."""
        rollouts_file = tmp_path / "rollouts.jsonl"

        rollouts = [
            # Both achieved in order
            {
                "results": {
                    "times": [2025.0, 2040.0],
                    "milestones": {
                        "AC": {"time": 2027.0},
                        "SAR": {"time": 2032.0}
                    }
                }
            },
            # Both achieved but out of order (SAR before AC)
            {
                "results": {
                    "times": [2025.0, 2040.0],
                    "milestones": {
                        "AC": {"time": 2035.0},
                        "SAR": {"time": 2030.0}
                    }
                }
            },
            # Only AC achieved (censored)
            {
                "results": {
                    "times": [2025.0, 2040.0],
                    "milestones": {
                        "AC": {"time": 2028.0}
                    }
                }
            },
            # Neither achieved
            {
                "results": {
                    "times": [2025.0, 2040.0],
                    "milestones": {}
                }
            },
        ]

        with rollouts_file.open('w') as f:
            for rollout in rollouts:
                f.write(json.dumps(rollout) + '\n')

        return rollouts_file

    def test_read_transition_durations_complete_only(self, tmp_path):
        """Test reading only completed transitions."""
        rollouts_file = self.create_transition_rollouts(tmp_path)
        reader = RolloutsReader(rollouts_file)

        durations, censored, sim_end, out_of_order = reader.read_transition_durations(
            "AC", "SAR", include_censored=False
        )

        assert len(durations) == 1
        assert durations[0] == 5.0  # 2032 - 2027
        assert censored == 1  # One rollout had AC but not SAR
        assert out_of_order == 1  # One rollout had SAR before AC
        assert sim_end == 2040.0

    def test_read_transition_durations_with_censored(self, tmp_path):
        """Test reading transitions including censored data."""
        rollouts_file = self.create_transition_rollouts(tmp_path)
        reader = RolloutsReader(rollouts_file)

        durations, censored, sim_end, out_of_order = reader.read_transition_durations(
            "AC", "SAR", include_censored=True
        )

        assert len(durations) == 2
        assert 5.0 in durations  # Completed transition
        assert 12.0 in durations  # Censored: 2040 - 2028
        assert censored == 1
        assert out_of_order == 1

    def test_read_transition_data_arrays(self, tmp_path):
        """Test reading transition data as arrays."""
        rollouts_file = self.create_transition_rollouts(tmp_path)
        reader = RolloutsReader(rollouts_file)

        from_times, durations = reader.read_transition_data(
            "AC", "SAR", return_arrays=True
        )

        assert isinstance(from_times, np.ndarray)
        assert isinstance(durations, np.ndarray)
        assert len(from_times) == 1
        assert len(durations) == 1
        assert from_times[0] == 2027.0
        assert durations[0] == 5.0

    def test_read_transition_data_tuples(self, tmp_path):
        """Test reading transition data as tuples."""
        rollouts_file = self.create_transition_rollouts(tmp_path)
        reader = RolloutsReader(rollouts_file)

        transitions = reader.read_transition_data("AC", "SAR", return_arrays=False)

        assert isinstance(transitions, list)
        assert len(transitions) == 1
        from_time, to_time, duration = transitions[0]
        assert from_time == 2027.0
        assert to_time == 2032.0
        assert duration == 5.0

    def test_read_multiple_transition_durations(self, tmp_path):
        """Test reading multiple transition pairs efficiently."""
        rollouts_file = self.create_transition_rollouts(tmp_path)

        # Create metadata.json for simulation_cutoff
        metadata = {"time_range": [2025.0, 2040.0]}
        metadata_file = tmp_path / "metadata.json"
        with metadata_file.open('w') as f:
            json.dump(metadata, f)

        reader = RolloutsReader(rollouts_file)

        pairs = [("AC", "SAR")]
        result = reader.read_multiple_transition_durations(pairs)

        (labels, durations_per_pair, durations_with_censored,
         num_b_not_achieved, num_b_before_a, total_a_achieved,
         typical_max_duration, simulation_cutoff) = result

        assert len(labels) == 1
        assert "AC" in labels[0] and "SAR" in labels[0]
        assert len(durations_per_pair[0]) == 1
        assert durations_per_pair[0][0] == 5.0
        assert num_b_not_achieved[0] == 1
        assert num_b_before_a[0] == 1
        assert total_a_achieved[0] == 3


class TestTrajectoryReading:
    """Test trajectory data extraction."""

    def create_trajectory_rollouts(self, tmp_path) -> Path:
        """Create test file with trajectory data."""
        rollouts_file = tmp_path / "rollouts.jsonl"

        rollouts = [
            {
                "results": {
                    "times": [2025.0, 2026.0, 2027.0],
                    "horizon": [1.0, 2.0, 3.0],
                    "automation_fraction": [0.1, 0.2, 0.3],
                    "aa_time": 2026.0,
                    "metr_mse": 0.05
                }
            },
            {
                "results": {
                    "times": [2025.0, 2026.0, 2027.0],
                    "horizon": [1.5, 2.5, 3.5],
                    "automation_fraction": [0.15, 0.25, 0.35],
                    "aa_time": 2026.5,
                    "metr_mse": 0.06
                }
            },
        ]

        with rollouts_file.open('w') as f:
            for rollout in rollouts:
                f.write(json.dumps(rollout) + '\n')

        return rollouts_file

    def test_read_trajectories(self, tmp_path):
        """Test reading metric trajectories."""
        rollouts_file = self.create_trajectory_rollouts(tmp_path)
        reader = RolloutsReader(rollouts_file)

        times, trajectories, aa_times = reader.read_trajectories("horizon")

        assert isinstance(times, np.ndarray)
        assert len(times) == 3
        assert len(trajectories) == 2
        assert len(aa_times) == 2
        assert aa_times[0] == 2026.0
        assert aa_times[1] == 2026.5

    def test_read_metric_trajectories(self, tmp_path):
        """Test reading trajectories with full features."""
        rollouts_file = self.create_trajectory_rollouts(tmp_path)
        reader = RolloutsReader(rollouts_file)

        times, trajectories, aa_times, mse_values = reader.read_metric_trajectories(
            "automation_fraction",
            include_aa_times=True,
            include_mse=True
        )

        assert len(times) == 3
        assert len(trajectories) == 2
        assert len(aa_times) == 2
        assert len(mse_values) == 2
        assert mse_values[0] == 0.05
        assert mse_values[1] == 0.06

    def test_read_trajectories_missing_metric(self, tmp_path):
        """Test reading non-existent metric raises error."""
        rollouts_file = self.create_trajectory_rollouts(tmp_path)
        reader = RolloutsReader(rollouts_file)

        with pytest.raises(ValueError, match="No.*trajectories found"):
            reader.read_metric_trajectories("nonexistent_metric")

    def test_read_metric_at_milestone(self, tmp_path):
        """Test interpolating metric value at milestone time."""
        rollouts_file = self.create_trajectory_rollouts(tmp_path)
        reader = RolloutsReader(rollouts_file)

        values = reader.read_metric_at_milestone(
            "horizon",
            milestone_key="aa_time",
            clip_min=0.001
        )

        assert len(values) == 2
        # Should be interpolated values at aa_time


class TestBatchOperations:
    """Test batch reading operations."""

    def create_batch_test_rollouts(self, tmp_path) -> Path:
        """Create test file for batch operations."""
        rollouts_file = tmp_path / "rollouts.jsonl"

        rollouts = [
            {
                "results": {
                    "times": [2025.0, 2030.0],
                    "milestones": {
                        "AC": {"time": 2027.0},
                        "SAR": {"time": 2028.0},
                        "ASI": {"time": 2029.0}
                    }
                }
            },
            {
                "results": {
                    "times": [2025.0, 2030.0],
                    "milestones": {
                        "AC": {"time": 2026.0},
                        "SAR": {"time": 2027.5}
                    }
                }
            },
        ]

        with rollouts_file.open('w') as f:
            for rollout in rollouts:
                f.write(json.dumps(rollout) + '\n')

        return rollouts_file

    def test_read_multiple_milestone_times(self, tmp_path):
        """Test reading multiple milestones efficiently."""
        rollouts_file = self.create_batch_test_rollouts(tmp_path)
        reader = RolloutsReader(rollouts_file)

        times_map = reader.read_multiple_milestone_times(["AC", "SAR", "ASI"])

        assert len(times_map["AC"]) == 2
        assert len(times_map["SAR"]) == 2
        assert len(times_map["ASI"]) == 1

    def test_read_milestone_times_batch(self, tmp_path):
        """Test batch reading with statistics."""
        rollouts_file = self.create_batch_test_rollouts(tmp_path)
        reader = RolloutsReader(rollouts_file)

        times_map, not_achieved, sim_end, total = reader.read_milestone_times_batch(
            ["AC", "SAR", "ASI"]
        )

        assert len(times_map["AC"]) == 2
        assert not_achieved["AC"] == 0
        assert not_achieved["ASI"] == 1
        assert total == 2
        assert sim_end == 2030.0

    def test_count_rollouts(self, tmp_path):
        """Test counting rollouts."""
        rollouts_file = self.create_batch_test_rollouts(tmp_path)
        reader = RolloutsReader(rollouts_file)

        count = reader.count_rollouts()
        assert count == 2


class TestScatterData:
    """Test scatter plot data extraction."""

    def test_read_milestone_scatter_data(self, tmp_path):
        """Test reading data for milestone scatter plots."""
        rollouts_file = tmp_path / "rollouts.jsonl"

        rollouts = [
            {
                "results": {
                    "times": [2025.0, 2030.0],
                    "milestones": {
                        "AC": {"time": 2027.0},
                        "SAR": {"time": 2028.0}
                    }
                }
            },
            {
                "results": {
                    "times": [2025.0, 2030.0],
                    "milestones": {
                        "AC": {"time": 2026.0},
                        "SAR": {"time": 2029.0}
                    }
                }
            },
            # This one should be excluded (missing SAR)
            {
                "results": {
                    "times": [2025.0, 2030.0],
                    "milestones": {
                        "AC": {"time": 2025.5}
                    }
                }
            },
        ]

        with rollouts_file.open('w') as f:
            for rollout in rollouts:
                f.write(json.dumps(rollout) + '\n')

        reader = RolloutsReader(rollouts_file)
        times1, times2 = reader.read_milestone_scatter_data("AC", "SAR")

        assert len(times1) == 2
        assert len(times2) == 2
        assert times1[0] == 2027.0
        assert times2[0] == 2028.0


class TestCacheExport:
    """Test cache file export functionality."""

    def test_export_cache(self, tmp_path):
        """Test exporting cache file."""
        rollouts_file = tmp_path / "rollouts.jsonl"

        rollout = {
            "sample_id": 1,
            "parameters": {"param1": 1.0},
            "results": {
                "times": [2025.0, 2026.0, 2027.0],  # Large array - excluded
                "horizon": [1.0, 2.0, 3.0],  # Large array - excluded
                "milestones": {"AC": {"time": 2026.0}},  # Kept
                "aa_time": 2026.0,  # Kept
                "metr_mse": 0.05  # Kept
            }
        }

        with rollouts_file.open('w') as f:
            f.write(json.dumps(rollout) + '\n')

        reader = RolloutsReader(rollouts_file)
        cache_file = reader.export_cache()

        assert cache_file.exists()
        assert cache_file.name == "rollouts.cache.json"

        # Verify cache contents
        with cache_file.open('r') as f:
            cache_data = json.load(f)

        assert cache_data["version"] == 1
        assert len(cache_data["rollouts"]) == 1

        cached_rollout = cache_data["rollouts"][0]
        assert cached_rollout["sample_id"] == 1
        assert "times" not in cached_rollout["results"]  # Excluded
        # Note: horizon has 3 elements, which is < 10, so it's kept as a "small list"
        # Large trajectory arrays are excluded, but small lists are preserved
        assert "milestones" in cached_rollout["results"]  # Kept
        assert "aa_time" in cached_rollout["results"]  # Kept
        assert "simulation_end" in cached_rollout["results"]  # Derived

    def test_cache_autoload(self, tmp_path):
        """Test that exported cache is automatically loaded."""
        rollouts_file = tmp_path / "rollouts.jsonl"

        rollout = {
            "results": {
                "times": [2025.0, 2026.0],
                "milestones": {"AC": {"time": 2025.5}}
            }
        }

        with rollouts_file.open('w') as f:
            f.write(json.dumps(rollout) + '\n')

        # Create cache from JSONL
        reader1 = RolloutsReader(rollouts_file)
        cache_file = reader1.export_cache()

        # New reader should auto-load cache
        reader2 = RolloutsReader(rollouts_file)
        assert reader2._cache_data is not None
        assert reader2.rollouts_file == cache_file


class TestIterators:
    """Test various iterator methods."""

    def test_iter_rollouts(self, tmp_path):
        """Test basic rollout iteration."""
        rollouts_file = tmp_path / "rollouts.jsonl"

        rollouts = [
            {"results": {"times": [2025.0]}},
            {"error": "some error"},  # Should be skipped
            {"results": {"times": [2026.0]}},
        ]

        with rollouts_file.open('w') as f:
            for rollout in rollouts:
                f.write(json.dumps(rollout) + '\n')

        reader = RolloutsReader(rollouts_file)
        records = list(reader.iter_rollouts())

        assert len(records) == 2  # Error record skipped

    def test_iter_all_records(self, tmp_path):
        """Test iteration including error records."""
        rollouts_file = tmp_path / "rollouts.jsonl"

        rollouts = [
            {"results": {"times": [2025.0]}},
            {"error": "some error"},
            {"results": {"times": [2026.0]}},
        ]

        with rollouts_file.open('w') as f:
            for rollout in rollouts:
                f.write(json.dumps(rollout) + '\n')

        reader = RolloutsReader(rollouts_file)
        records = list(reader.iter_all_records())

        assert len(records) == 3  # All records including error

    def test_iter_with_blank_lines(self, tmp_path):
        """Test that blank lines are skipped."""
        rollouts_file = tmp_path / "rollouts.jsonl"

        content = '{"results": {"times": [2025.0]}}\n\n\n{"results": {"times": [2026.0]}}\n'
        rollouts_file.write_text(content)

        reader = RolloutsReader(rollouts_file)
        records = list(reader.iter_rollouts())

        assert len(records) == 2


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_milestone_with_nan_time(self, tmp_path):
        """Test handling of NaN milestone times."""
        rollouts_file = tmp_path / "rollouts.jsonl"

        rollout = {
            "results": {
                "times": [2025.0, 2026.0],
                "milestones": {
                    "AC": {"time": float('nan')}
                }
            }
        }

        with rollouts_file.open('w') as f:
            f.write(json.dumps({"results": rollout["results"]}) + '\n')

        # Can't directly serialize NaN in JSON, so write it differently
        rollouts_file.write_text('{"results": {"times": [2025.0, 2026.0], "milestones": {"AC": {"time": null}}}}\n')

        reader = RolloutsReader(rollouts_file)
        times, not_achieved, _ = reader.read_milestone_times("AC")

        assert len(times) == 0
        assert not_achieved == 1

    def test_mismatched_trajectory_lengths(self, tmp_path):
        """Test handling of mismatched times/metric array lengths."""
        rollouts_file = tmp_path / "rollouts.jsonl"

        rollout = {
            "results": {
                "times": [2025.0, 2026.0, 2027.0],
                "horizon": [1.0, 2.0]  # Mismatched length
            }
        }

        with rollouts_file.open('w') as f:
            f.write(json.dumps(rollout) + '\n')

        reader = RolloutsReader(rollouts_file)

        with pytest.raises(ValueError):
            reader.read_metric_trajectories("horizon")

    def test_filter_by_milestone(self, tmp_path):
        """Test filtering trajectories by milestone achievement."""
        rollouts_file = tmp_path / "rollouts.jsonl"

        rollouts = [
            {
                "results": {
                    "times": [2025.0, 2030.0],
                    "horizon": [1.0, 2.0],
                    "milestones": {"AC": {"time": 2027.0}}
                }
            },
            {
                "results": {
                    "times": [2025.0, 2030.0],
                    "horizon": [1.5, 2.5],
                    "milestones": {"AC": {"time": 2035.0}}  # Too late
                }
            },
        ]

        with rollouts_file.open('w') as f:
            for rollout in rollouts:
                f.write(json.dumps(rollout) + '\n')

        reader = RolloutsReader(rollouts_file)
        times, trajectories, _, _ = reader.read_metric_trajectories(
            "horizon",
            filter_milestone="AC",
            filter_year=2027.0
        )

        # Only first rollout should match (within 0.5 year tolerance)
        assert len(trajectories) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=rollouts_reader", "--cov-report=term-missing"])
