"""Shared feature preprocessing for the ML pipeline.

Must stay importable from this exact module path: joblib/pickle serialize
a custom transformer by its class's module + qualname, so the class used
when a model is trained (ml_engine/train.py) must resolve to the same
place when the model is later loaded (ml_engine/detector.py). Keeping it
in one shared module -- rather than defining it twice -- is also what
keeps train/serve parity for the preprocessing step itself.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from core.extractor import FEATURE_NAMES

#: Non-negative, unbounded counts/rates/timing stats (including raw
#: Shannon entropy in bits, which grows with the number of distinct
#: destination ports rather than being bounded) with heavy right tails --
#: a single extreme flood or scan window can otherwise dominate an
#: isolation-tree split, or a distance/kernel-based model's decision
#: boundary, purely by raw magnitude. log1p compresses those tails.
#: The *_ratio features are already bounded in [0, 1] and are left
#: untransformed -- log-compressing an already-bounded quantity buys
#: nothing and just makes it harder to reason about.
LOG1P_FEATURES: List[str] = [
    "average_packet_size",
    "packets_per_second",
    "unique_destination_port_count",
    "unique_destination_ip_count",
    "unique_source_ip_count",
    "dest_port_entropy",
    "packet_size_std",
    "mean_inter_arrival",
    "std_inter_arrival",
]
LOG1P_COLUMN_INDICES: List[int] = [FEATURE_NAMES.index(name) for name in LOG1P_FEATURES]


class Log1pColumns(BaseEstimator, TransformerMixin):
    """Applies log1p to a fixed subset of feature columns, leaving the
    rest -- and the overall column order (FEATURE_NAMES layout) -- exactly
    as they were.

    A plain per-column transform was chosen over sklearn's
    ColumnTransformer specifically so the output columns stay in
    FEATURE_NAMES order throughout the pipeline: ColumnTransformer
    concatenates its sub-transformers' outputs in the order they're
    declared, which would otherwise silently reorder the vector and make
    the pipeline harder to reason about/debug against FEATURE_NAMES.
    """

    def __init__(self, column_indices: Optional[Sequence[int]] = None) -> None:
        self.column_indices = list(LOG1P_COLUMN_INDICES) if column_indices is None else list(column_indices)

    def fit(self, X: np.ndarray, y: object = None) -> "Log1pColumns":
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.array(X, dtype=np.float64, copy=True)
        # Clip defensively: every current feature is non-negative by
        # construction, but log1p on a negative value would raise/NaN, and
        # this transformer has no way to know if a future feature breaks
        # that invariant.
        X[:, self.column_indices] = np.log1p(np.clip(X[:, self.column_indices], 0.0, None))
        return X
