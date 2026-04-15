"""Tests for CLI argument parsing and RunConfig."""

from src.cli import RunConfig, parse_args
from src.config import TrainingConfig


class TestRunConfig:
    def test_defaults(self):
        rc = RunConfig()
        assert rc.stages == ["all"]
        assert rc.subjects is None
        assert rc.classifiers is None
        assert rc.experiment_name is None
        assert rc.export_models is True

    def test_custom_stages(self):
        rc = RunConfig(stages=["cv"])
        assert rc.stages == ["cv"]

    def test_custom_subjects(self):
        rc = RunConfig(subjects=["subject0001", "subject0003"])
        assert len(rc.subjects) == 2


class TestParseArgs:
    def test_no_args_gives_defaults(self):
        rc = parse_args([])
        assert rc.stages == ["all"]

    def test_stages_flag(self):
        rc = parse_args(["--stages", "cv,features"])
        assert rc.stages == ["cv", "features"]

    def test_subjects_flag(self):
        rc = parse_args(["--subjects", "0001,0003"])
        assert rc.subjects == ["0001", "0003"]

    def test_classifier_flag(self):
        rc = parse_args(["--classifier", "lda,svm"])
        assert rc.classifiers == ["lda", "svm"]

    def test_experiment_name_flag(self):
        rc = parse_args(["--experiment-name", "test_run"])
        assert rc.experiment_name == "test_run"

    def test_no_export_flag(self):
        rc = parse_args(["--no-export"])
        assert rc.export_models is False


class TestConfigHash:
    def test_same_config_same_hash(self):
        a = TrainingConfig()
        b = TrainingConfig()
        assert a.config_hash() == b.config_hash()

    def test_different_config_different_hash(self):
        a = TrainingConfig(k_best=10)
        b = TrainingConfig(k_best=15)
        assert a.config_hash() != b.config_hash()

    def test_hash_is_deterministic(self):
        cfg = TrainingConfig()
        assert cfg.config_hash() == cfg.config_hash()

    def test_hash_is_string(self):
        cfg = TrainingConfig()
        h = cfg.config_hash()
        assert isinstance(h, str)
        assert len(h) == 16
