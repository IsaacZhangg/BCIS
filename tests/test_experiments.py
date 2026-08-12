"""Tests for experiment tracking."""

import pytest

from src.experiments import (
    compare_experiments,
    list_experiments,
    load_experiment,
    save_experiment,
)


@pytest.fixture
def exp_dir(tmp_path):
    return tmp_path / "experiments"


class TestSaveLoad:
    def test_save_creates_json_file(self, exp_dir):
        save_experiment(
            name="test_baseline",
            results={"nested_mean": 0.589, "augmented_mean": 0.646},
            config_diff={},
            verdict="baseline",
            seed=42,
            experiments_dir=exp_dir,
        )
        files = list(exp_dir.glob("*.json"))
        assert len(files) == 1
        assert "test_baseline" in files[0].name

    def test_load_roundtrip(self, exp_dir):
        save_experiment(
            name="test_roundtrip",
            results={"nested_mean": 0.60},
            config_diff={"k_best": [10, 15]},
            verdict="neutral",
            seed=42,
            experiments_dir=exp_dir,
        )
        loaded = load_experiment("test_roundtrip", experiments_dir=exp_dir)
        assert loaded["name"] == "test_roundtrip"
        assert loaded["results"]["nested_mean"] == 0.60
        assert loaded["config_diff"]["k_best"] == [10, 15]
        assert loaded["verdict"] == "neutral"

    def test_load_nonexistent_raises(self, exp_dir):
        with pytest.raises(FileNotFoundError):
            load_experiment("nonexistent", experiments_dir=exp_dir)


class TestListExperiments:
    def test_list_empty_dir(self, exp_dir):
        result = list_experiments(experiments_dir=exp_dir)
        assert result == []

    def test_list_returns_summaries(self, exp_dir):
        save_experiment("exp_a", {"nested_mean": 0.55}, {}, "negative", 42, exp_dir)
        save_experiment("exp_b", {"nested_mean": 0.62}, {}, "positive", 42, exp_dir)
        result = list_experiments(experiments_dir=exp_dir)
        assert len(result) == 2
        names = {e["name"] for e in result}
        assert names == {"exp_a", "exp_b"}


class TestCompareExperiments:
    def test_compare_two_experiments(self, exp_dir):
        save_experiment("baseline", {"nested_mean": 0.589}, {}, "baseline", 42, exp_dir)
        save_experiment(
            "new_idea",
            {"nested_mean": 0.610},
            {"k_best": [10, 15]},
            "positive",
            42,
            exp_dir,
        )
        comparison = compare_experiments(
            "baseline", "new_idea", experiments_dir=exp_dir
        )
        assert "nested_mean" in comparison
        assert comparison["nested_mean"]["delta"] == pytest.approx(0.021, abs=1e-6)
