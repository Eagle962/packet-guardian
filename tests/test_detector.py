"""Tests for ml_engine.detector: load_model and predict."""

import joblib
import numpy as np
import pytest
from sklearn.ensemble import IsolationForest

from core.extractor import FEATURE_NAMES
from ml_engine.detector import EXPECTED_SCHEMA_VERSION, load_model, predict
from ml_engine.train import MODEL_PATH, build_pipeline


def _make_artifact(estimator=None, feature_names=None, feature_count=None, schema_version=None, sklearn_version=None):
    n_features = len(FEATURE_NAMES)
    if estimator is None:
        estimator = IsolationForest(n_estimators=10, random_state=0)
    pipeline = build_pipeline(estimator)
    rng = np.random.default_rng(0)
    pipeline.fit(rng.normal(size=(50, n_features)))

    import sklearn as sklearn_module

    return {
        "model": pipeline,
        "feature_names": list(FEATURE_NAMES) if feature_names is None else feature_names,
        "feature_count": n_features if feature_count is None else feature_count,
        "schema_version": EXPECTED_SCHEMA_VERSION if schema_version is None else schema_version,
        "sklearn_version": sklearn_module.__version__ if sklearn_version is None else sklearn_version,
    }


class TestLoadModel:
    def test_load_real_shipped_model(self) -> None:
        """The checked-in model.pkl (trained via `python -m ml_engine.train`)
        must load cleanly against the current schema -- if this fails, the
        model needs retraining, which main.py already handles gracefully
        (falls back to rule-only detection with a warning), but it should
        not happen for a freshly-committed model."""
        artifact = load_model(MODEL_PATH)
        assert artifact["feature_names"] == list(FEATURE_NAMES)
        assert artifact["feature_count"] == len(FEATURE_NAMES)
        assert artifact["schema_version"] == EXPECTED_SCHEMA_VERSION
        assert "sklearn_version" in artifact
        assert "model_family" in artifact
        assert hasattr(artifact["model"], "predict")
        assert hasattr(artifact["model"], "decision_function")

    def test_missing_file_raises(self, tmp_path) -> None:
        missing_path = tmp_path / "does_not_exist.pkl"
        with pytest.raises(FileNotFoundError):
            load_model(missing_path)

    def test_mismatched_feature_names_raises(self, tmp_path) -> None:
        bad_path = tmp_path / "bad_model.pkl"
        joblib.dump(_make_artifact(feature_names=["wrong", "feature", "list"], feature_count=3), bad_path)
        with pytest.raises(ValueError):
            load_model(bad_path)

    def test_missing_metadata_key_raises(self, tmp_path) -> None:
        bad_path = tmp_path / "incomplete_model.pkl"
        joblib.dump({"model": _make_artifact()["model"]}, bad_path)
        with pytest.raises(ValueError):
            load_model(bad_path)

    def test_schema_version_mismatch_raises(self, tmp_path) -> None:
        bad_path = tmp_path / "old_schema_model.pkl"
        joblib.dump(_make_artifact(schema_version=EXPECTED_SCHEMA_VERSION - 1), bad_path)
        with pytest.raises(ValueError):
            load_model(bad_path)

    def test_sklearn_version_mismatch_warns_but_still_loads(self, tmp_path) -> None:
        warn_path = tmp_path / "old_sklearn_model.pkl"
        joblib.dump(_make_artifact(sklearn_version="0.0.0-not-a-real-version"), warn_path)
        with pytest.warns(UserWarning, match="scikit-learn"):
            artifact = load_model(warn_path)
        assert artifact["feature_names"] == list(FEATURE_NAMES)


class TestPredict:
    """predict()'s own mechanics (shape handling, key names) -- not the
    shipped model's quality, which is tests/test_ml_quality.py's job."""

    def test_predict_returns_expected_keys(self) -> None:
        artifact = _make_artifact()
        vector = np.zeros(len(FEATURE_NAMES))
        result = predict(vector, artifact)
        assert "anomaly_score" in result
        assert "status" in result
        assert isinstance(result["anomaly_score"], float)
        assert result["status"] in ("NORMAL", "ANOMALY")

    def test_predict_accepts_zero_vector(self) -> None:
        artifact = _make_artifact()
        vector = np.zeros(len(FEATURE_NAMES))
        result = predict(vector, artifact)
        assert "anomaly_score" in result
        assert result["status"] in ("NORMAL", "ANOMALY")

    def test_predict_works_against_real_shipped_model(self) -> None:
        artifact = load_model(MODEL_PATH)
        vector = np.zeros(len(FEATURE_NAMES))
        result = predict(vector, artifact)
        assert result["status"] in ("NORMAL", "ANOMALY")
