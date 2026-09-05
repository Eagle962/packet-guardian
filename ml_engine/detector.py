"""Loads the trained IsolationForest model and runs inference on feature
vectors produced by ``core.extractor``.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict

import numpy as np

from core.extractor import FEATURE_NAMES

MODEL_PATH = Path(__file__).parent / "model.pkl"

ModelArtifact = Dict[str, Any]


def load_model(path: Path = MODEL_PATH) -> ModelArtifact:
    """Load a trained model artifact from disk and validate its metadata.

    Args:
        path: Location of the pickled model artifact produced by
            ``ml_engine.train.save_model``.

    Returns:
        The artifact dict with keys ``model``, ``feature_names``, and
        ``feature_count``.

    Raises:
        FileNotFoundError: If no model file exists at ``path``.
        ValueError: If the artifact's feature metadata does not match the
            current ``core.extractor.FEATURE_NAMES`` (i.e. the model was
            trained against a different feature schema).
    """
    if not path.exists():
        raise FileNotFoundError(
            f"No trained model found at {path}. Run `python -m ml_engine.train` first."
        )

    with open(path, "rb") as handle:
        artifact = pickle.load(handle)

    for required_key in ("model", "feature_names", "feature_count"):
        if required_key not in artifact:
            raise ValueError(f"Model artifact at {path} is missing required key '{required_key}'.")

    if artifact["feature_names"] != list(FEATURE_NAMES):
        raise ValueError(
            "Model artifact feature_names do not match core.extractor.FEATURE_NAMES. "
            "Retrain the model with `python -m ml_engine.train`."
        )

    if artifact["feature_count"] != len(FEATURE_NAMES):
        raise ValueError(
            "Model artifact feature_count does not match core.extractor.FEATURE_NAMES length. "
            "Retrain the model with `python -m ml_engine.train`."
        )

    return artifact


def predict(feature_vector: np.ndarray, model_artifact: ModelArtifact) -> Dict[str, Any]:
    """Run inference on a single feature vector.

    Args:
        feature_vector: A 1-D array of length ``len(FEATURE_NAMES)``.
        model_artifact: The dict returned by :func:`load_model`.

    Returns:
        A dict with ``anomaly_score`` (float; lower means more anomalous)
        and ``status`` (``"NORMAL"`` or ``"ANOMALY"``).
    """
    model = model_artifact["model"]
    vector = np.asarray(feature_vector, dtype=np.float64).reshape(1, -1)

    anomaly_score = float(model.decision_function(vector)[0])
    prediction = int(model.predict(vector)[0])
    status = "NORMAL" if prediction == 1 else "ANOMALY"

    return {"anomaly_score": anomaly_score, "status": status}
