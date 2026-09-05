"""Trains the ML anomaly detector against labelled synthetic traffic
(ml_engine.traffic_profiles), tuning hyperparameters and comparing model
families honestly on a validation split, then saves the winning pipeline.

Run directly to (re)train and save the model:

    python -m ml_engine.train

Three disjoint seeds keep the process honest:
- SEED_TRAIN: normal-only windows the model is actually fit on.
- SEED_VAL:   normal + attack windows used ONLY to tune hyperparameters
              and pick between candidate model families.
- SEED_EVAL:  used exclusively by ml_engine/evaluate.py for the final,
              hard-gated evaluation. Never touched here.
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import OneClassSVM

from core.extractor import FEATURE_NAMES
from ml_engine.preprocessing import Log1pColumns
from ml_engine.traffic_profiles import ALL_PROFILES, NORMAL_PROFILES, build_feature_dataset

MODEL_PATH = Path(__file__).parent / "model.pkl"
SCHEMA_VERSION = 2

SEED_TRAIN = 1001
SEED_VAL = 2002
SEED_EVAL = 3003

N_TRAIN_PER_PROFILE = 80
N_VAL_PER_PROFILE = 30
N_EVAL_PER_PROFILE = 50

WINDOW_SECONDS = 5.0
RANDOM_STATE = 42

#: False-positive-rate budget used to pick among candidate hyperparameters/
#: model families on the validation set: a candidate within budget is
#: ranked by recall; one outside budget is penalized, not disqualified
#: outright, so a narrow miss still outranks a much worse candidate.
VALIDATION_FPR_BUDGET = 0.05


def build_pipeline(estimator: Any) -> Pipeline:
    return Pipeline(
        [
            ("log1p", Log1pColumns()),
            ("scaler", RobustScaler()),
            ("model", estimator),
        ]
    )


def load_train_val_data():
    """Returns (X_train_all, X_train_fit, X_val, y_val, scenarios_val).

    X_train_all: every normal training window generated.
    X_train_fit: 80% of X_train_all -- what candidates are fit on during
        hyperparameter/model-family search.
    X_val / y_val / scenarios_val: the remaining 20% of normal training
        windows (never fit on) folded together with a separately-seeded
        normal+attack validation set. y_val is 1 for normal, -1 for
        attack (matching sklearn's outlier-detector predict() convention).
    """
    normal_names = list(NORMAL_PROFILES.keys())
    X_train_all, _, _ = build_feature_dataset(normal_names, N_TRAIN_PER_PROFILE, SEED_TRAIN, WINDOW_SECONDS)

    rng = np.random.default_rng(RANDOM_STATE)
    n = X_train_all.shape[0]
    idx = rng.permutation(n)
    split = int(n * 0.8)
    fit_idx, held_out_idx = idx[:split], idx[split:]
    X_train_fit = X_train_all[fit_idx]
    X_val_normal = X_train_all[held_out_idx]

    all_names = list(ALL_PROFILES.keys())
    X_val_labeled, labels_val, scenarios_val = build_feature_dataset(
        all_names, N_VAL_PER_PROFILE, SEED_VAL, WINDOW_SECONDS
    )
    y_val_labeled = np.array([1 if label == "normal" else -1 for label in labels_val])

    X_val = np.vstack([X_val_labeled, X_val_normal])
    y_val = np.concatenate([y_val_labeled, np.ones(X_val_normal.shape[0], dtype=int)])
    scenarios_val_combined = scenarios_val + ["held_out_train_normal"] * X_val_normal.shape[0]

    return X_train_all, X_train_fit, X_val, y_val, scenarios_val_combined


def _validation_metrics(
    pipeline: Pipeline, X_val: np.ndarray, y_val: np.ndarray, scenarios_val: List[str]
) -> Dict[str, Any]:
    preds = pipeline.predict(X_val)  # 1 = normal, -1 = anomaly
    is_normal = y_val == 1
    is_attack = y_val == -1

    fpr = float(np.mean(preds[is_normal] == -1)) if is_normal.any() else float("nan")
    recall = float(np.mean(preds[is_attack] == -1)) if is_attack.any() else float("nan")

    scenarios_arr = np.array(scenarios_val)
    per_scenario_recall = {}
    for scenario in sorted({str(s) for s in scenarios_arr[is_attack]}):
        mask = is_attack & (scenarios_arr == scenario)
        per_scenario_recall[scenario] = float(np.mean(preds[mask] == -1))

    return {"fpr": fpr, "recall": recall, "per_scenario_recall": per_scenario_recall}


def _selection_score(metrics: Dict[str, Any], fpr_budget: float = VALIDATION_FPR_BUDGET) -> float:
    """Higher is better. A candidate within the FPR budget is ranked by
    recall; one outside budget is penalized proportionally to the
    overshoot rather than disqualified outright, so a narrow miss still
    outranks a much worse candidate during search."""
    fpr = metrics["fpr"]
    recall = metrics["recall"]
    if fpr <= fpr_budget:
        return recall
    return recall - 10.0 * (fpr - fpr_budget)


def _search_isolation_forest(
    X_fit: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, scenarios_val: List[str]
) -> List[Dict[str, Any]]:
    candidates = []
    grid = itertools.product([0.01, 0.05, 0.1], [100, 200], ["auto", 0.5])
    for contamination, n_estimators, max_samples in grid:
        params = {"contamination": contamination, "n_estimators": n_estimators, "max_samples": max_samples}
        estimator = IsolationForest(random_state=RANDOM_STATE, **params)
        pipeline = build_pipeline(estimator)
        pipeline.fit(X_fit)
        metrics = _validation_metrics(pipeline, X_val, y_val, scenarios_val)
        candidates.append(
            {
                "family": "IsolationForest",
                "params": params,
                "pipeline": pipeline,
                "metrics": metrics,
                "score": _selection_score(metrics),
            }
        )
    return candidates


def _search_one_class_svm(
    X_fit: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, scenarios_val: List[str]
) -> List[Dict[str, Any]]:
    candidates = []
    grid = itertools.product([0.01, 0.05, 0.1], ["scale", "auto"])
    for nu, gamma in grid:
        params = {"nu": nu, "gamma": gamma, "kernel": "rbf"}
        estimator = OneClassSVM(**params)
        pipeline = build_pipeline(estimator)
        pipeline.fit(X_fit)
        metrics = _validation_metrics(pipeline, X_val, y_val, scenarios_val)
        candidates.append(
            {
                "family": "OneClassSVM",
                "params": params,
                "pipeline": pipeline,
                "metrics": metrics,
                "score": _selection_score(metrics),
            }
        )
    return candidates


def _search_lof(
    X_fit: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, scenarios_val: List[str]
) -> List[Dict[str, Any]]:
    candidates = []
    grid = itertools.product([10, 20, 35], [0.05, 0.1])
    for n_neighbors, contamination in grid:
        params = {"n_neighbors": n_neighbors, "contamination": contamination, "novelty": True}
        estimator = LocalOutlierFactor(**params)
        pipeline = build_pipeline(estimator)
        pipeline.fit(X_fit)
        metrics = _validation_metrics(pipeline, X_val, y_val, scenarios_val)
        candidates.append(
            {
                "family": "LocalOutlierFactor",
                "params": params,
                "pipeline": pipeline,
                "metrics": metrics,
                "score": _selection_score(metrics),
            }
        )
    return candidates


def select_best_model(
    X_fit: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, scenarios_val: List[str]
) -> List[Dict[str, Any]]:
    """Search all three candidate families and return every candidate,
    sorted best-first by validation score. candidates[0] is the winner."""
    candidates = (
        _search_isolation_forest(X_fit, X_val, y_val, scenarios_val)
        + _search_one_class_svm(X_fit, X_val, y_val, scenarios_val)
        + _search_lof(X_fit, X_val, y_val, scenarios_val)
    )
    candidates.sort(key=lambda c: c["score"], reverse=True)
    return candidates


def _rebuild_estimator(family: str, params: Dict[str, Any]) -> Any:
    if family == "IsolationForest":
        return IsolationForest(random_state=RANDOM_STATE, **params)
    if family == "OneClassSVM":
        return OneClassSVM(**params)
    if family == "LocalOutlierFactor":
        return LocalOutlierFactor(**params)
    raise ValueError(f"Unknown model family: {family}")


def save_model(
    pipeline: Pipeline,
    path: Path = MODEL_PATH,
    model_family: str = "",
    hyperparameters: Optional[Dict[str, Any]] = None,
    validation_metrics: Optional[Dict[str, Any]] = None,
) -> None:
    """Persist the trained pipeline alongside feature/version/provenance
    metadata, using joblib (not bare pickle) per the project's model
    artifact hygiene policy -- see README's Security & Privacy section."""
    payload = {
        "model": pipeline,
        "feature_names": list(FEATURE_NAMES),
        "feature_count": len(FEATURE_NAMES),
        "schema_version": SCHEMA_VERSION,
        "sklearn_version": sklearn.__version__,
        "model_family": model_family,
        "hyperparameters": hyperparameters or {},
        "validation_metrics": validation_metrics or {},
        "training_provenance": {
            "generator": "ml_engine.traffic_profiles",
            "data_source": "synthetic (see EVALUATION.md for why real datasets were not used)",
            "seed_train": SEED_TRAIN,
            "seed_val": SEED_VAL,
            "n_train_per_profile": N_TRAIN_PER_PROFILE,
            "n_val_per_profile": N_VAL_PER_PROFILE,
            "window_seconds": WINDOW_SECONDS,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, path)


def main() -> None:
    X_train_all, X_train_fit, X_val, y_val, scenarios_val = load_train_val_data()

    print(f"Training pool: {X_train_all.shape[0]} normal windows "
          f"({X_train_fit.shape[0]} for search-fitting, "
          f"{X_train_all.shape[0] - X_train_fit.shape[0]} held out for validation FPR)")
    print(f"Validation set: {X_val.shape[0]} windows "
          f"({int((y_val == 1).sum())} normal, {int((y_val == -1).sum())} attack)")

    candidates = select_best_model(X_train_fit, X_val, y_val, scenarios_val)

    print("\nModel family / hyperparameter search results (validation set), best first:")
    for c in candidates[:8]:
        print(f"  score={c['score']:.3f} fpr={c['metrics']['fpr']:.3f} "
              f"recall={c['metrics']['recall']:.3f} family={c['family']} params={c['params']}")

    best = candidates[0]
    print(f"\nSelected: {best['family']} {best['params']} "
          f"(validation fpr={best['metrics']['fpr']:.3f}, recall={best['metrics']['recall']:.3f})")

    # Refit the winning configuration on *all* available normal training
    # data (search-fit + the slice held out for validation FPR) for the
    # final shipped model -- validation only ever influenced which
    # hyperparameters/family were chosen, not what data the final artifact
    # is fit on. The held-out evaluation set (SEED_EVAL) is untouched here.
    final_estimator = _rebuild_estimator(best["family"], best["params"])
    final_pipeline = build_pipeline(final_estimator)
    final_pipeline.fit(X_train_all)

    save_model(
        final_pipeline,
        model_family=best["family"],
        hyperparameters=best["params"],
        validation_metrics=best["metrics"],
    )
    print(f"\nSaved {best['family']} pipeline to {MODEL_PATH}")


if __name__ == "__main__":
    main()
