"""Loads the trained anomaly-detection pipeline and runs inference on
feature vectors produced by core.extractor.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import sklearn

from core.extractor import FEATURE_NAMES

MODEL_PATH = Path(__file__).parent / "model.pkl"

#: Bumped whenever the artifact's stored structure changes (new metadata
#: keys, preprocessing pipeline shape, etc.), independent of FEATURE_NAMES
#: itself -- feature_names/feature_count already catch a schema drift in
#: the feature vector; schema_version catches drift in the artifact
#: format itself (e.g. an older artifact missing keys this version reads).
EXPECTED_SCHEMA_VERSION = 2

ModelArtifact = Dict[str, Any]


def load_model(path: Path = MODEL_PATH) -> ModelArtifact:
    """Load a trained model artifact from disk and validate its metadata.

    Args:
        path: Location of the joblib-serialized model artifact produced by
            ``ml_engine.train.save_model``.

    Returns:
        The artifact dict (keys: model, feature_names, feature_count,
        schema_version, sklearn_version, model_family, hyperparameters,
        validation_metrics, training_provenance).

    Raises:
        FileNotFoundError: If no model file exists at ``path``.
        ValueError: If the artifact's feature or schema metadata does not
            match what this codebase currently expects.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"No trained model found at {path}. Run `python -m ml_engine.train` first."
        )

    # SECURITY: joblib.load (like pickle, which it uses internally for
    # non-array objects) executes arbitrary code embedded in the file
    # during deserialization. Only ever load a model file you trained
    # yourself or obtained from a source you trust as much as you'd trust
    # running its code directly -- see README's Security & Privacy section.
    artifact = joblib.load(path)

    for required_key in ("model", "feature_names", "feature_count", "schema_version"):
        if required_key not in artifact:
            raise ValueError(f"Model artifact at {path} is missing required key '{required_key}'.")

    if artifact["schema_version"] != EXPECTED_SCHEMA_VERSION:
        raise ValueError(
            f"Model artifact at {path} has schema_version={artifact['schema_version']}, "
            f"expected {EXPECTED_SCHEMA_VERSION}. Retrain with `python -m ml_engine.train`."
        )

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

    trained_sklearn_version = artifact.get("sklearn_version")
    if trained_sklearn_version and trained_sklearn_version != sklearn.__version__:
        warnings.warn(
            f"Model at {path} was trained with scikit-learn {trained_sklearn_version}, "
            f"but the running environment has {sklearn.__version__}. Fitted estimator "
            f"internals are not guaranteed compatible across versions -- predictions may "
            f"be silently wrong. Retrain with `python -m ml_engine.train` to be safe.",
            stacklevel=2,
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
    pipeline = model_artifact["model"]
    vector = np.asarray(feature_vector, dtype=np.float64).reshape(1, -1)

    anomaly_score = float(pipeline.decision_function(vector)[0])
    prediction = int(pipeline.predict(vector)[0])
    status = "NORMAL" if prediction == 1 else "ANOMALY"

    return {"anomaly_score": anomaly_score, "status": status}
