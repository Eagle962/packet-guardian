"""Tests for ml_engine.detector: load_model and predict."""

import pickle

import numpy as np
import pytest
from sklearn.ensemble import IsolationForest

from core.extractor import FEATURE_NAMES
from ml_engine.detector import load_model, predict
from ml_engine.train import MODEL_PATH, generate_normal_traffic_samples, train_model


class TestLoadModel:
    def test_load_real_trained_model(self) -> None:
        artifact = load_model(MODEL_PATH)
        assert artifact["feature_names"] == list(FEATURE_NAMES)
        assert artifact["feature_count"] == len(FEATURE_NAMES)
        assert isinstance(artifact["model"], IsolationForest)

    def test_missing_file_raises(self, tmp_path) -> None:
        missing_path = tmp_path / "does_not_exist.pkl"
        with pytest.raises(FileNotFoundError):
            load_model(missing_path)

    def test_mismatched_feature_names_raises(self, tmp_path) -> None:
        bad_path = tmp_path / "bad_model.pkl"
        model = train_model(generate_normal_traffic_samples())
        with open(bad_path, "wb") as handle:
            pickle.dump(
                {
                    "model": model,
                    "feature_names": ["wrong", "feature", "list"],
                    "feature_count": 3,
                },
                handle,
            )
        with pytest.raises(ValueError):
            load_model(bad_path)

    def test_missing_metadata_key_raises(self, tmp_path) -> None:
        bad_path = tmp_path / "incomplete_model.pkl"
        model = train_model(generate_normal_traffic_samples())
        with open(bad_path, "wb") as handle:
            pickle.dump({"model": model}, handle)
        with pytest.raises(ValueError):
            load_model(bad_path)


class TestPredict:
    def test_predict_returns_expected_keys(self) -> None:
        artifact = load_model(MODEL_PATH)
        vector = np.array([500, 20, 10000, 0.7, 0.2, 0.02, 4, 3], dtype=np.float64)
        result = predict(vector, artifact)
        assert "anomaly_score" in result
        assert "status" in result
        assert isinstance(result["anomaly_score"], float)
        assert result["status"] in ("NORMAL", "ANOMALY")

    def test_normal_looking_traffic_is_classified_normal(self) -> None:
        artifact = load_model(MODEL_PATH)
        vector = np.array([500, 20, 10000, 0.7, 0.2, 0.02, 4, 3], dtype=np.float64)
        result = predict(vector, artifact)
        assert result["status"] == "NORMAL"

    def test_extreme_outlier_is_classified_anomaly(self) -> None:
        artifact = load_model(MODEL_PATH)
        # Wildly atypical traffic: huge packet rate, all-ICMP, many unique
        # destination ports/ips -- looks like a scan/flood, not normal use.
        vector = np.array(
            [1500, 5000, 7_500_000, 0.0, 0.0, 1.0, 500, 400], dtype=np.float64
        )
        result = predict(vector, artifact)
        assert result["status"] == "ANOMALY"
        assert result["anomaly_score"] < 0

    def test_predict_accepts_zero_vector(self) -> None:
        artifact = load_model(MODEL_PATH)
        vector = np.zeros(len(FEATURE_NAMES))
        result = predict(vector, artifact)
        assert "anomaly_score" in result
        assert result["status"] in ("NORMAL", "ANOMALY")
