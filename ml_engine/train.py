"""Trains the IsolationForest anomaly detection model on synthetic
"normal" traffic feature vectors and exports it to ``ml_engine/model.pkl``.

Run directly to (re)generate the model:

    python -m ml_engine.train
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.ensemble import IsolationForest

from core.extractor import FEATURE_NAMES

MODEL_PATH = Path(__file__).parent / "model.pkl"

#: Number of synthetic "normal traffic" samples to generate for training.
SAMPLE_COUNT = 500


def generate_normal_traffic_samples(
    sample_count: int = SAMPLE_COUNT, random_state: int = 42
) -> np.ndarray:
    """Generate synthetic feature vectors representative of normal traffic.

    Each feature is sampled from a distribution with realistic center and
    spread, then clipped to a sane non-negative range so the synthetic
    data resembles plausible traffic statistics rather than pure noise.

    Feature order matches ``core.extractor.FEATURE_NAMES``:
    [average_packet_size, packets_per_second, bytes_per_second, tcp_ratio,
    udp_ratio, icmp_ratio, unique_destination_port_count,
    unique_destination_ip_count].
    """
    rng = np.random.default_rng(random_state)

    average_packet_size = rng.normal(loc=500, scale=150, size=sample_count)
    packets_per_second = rng.normal(loc=20, scale=8, size=sample_count)
    bytes_per_second = average_packet_size * packets_per_second

    tcp_ratio = rng.normal(loc=0.7, scale=0.1, size=sample_count)
    udp_ratio = rng.normal(loc=0.2, scale=0.08, size=sample_count)
    icmp_ratio = rng.normal(loc=0.02, scale=0.02, size=sample_count)

    unique_destination_port_count = rng.normal(loc=4, scale=2, size=sample_count)
    unique_destination_ip_count = rng.normal(loc=3, scale=1.5, size=sample_count)

    features = np.column_stack(
        [
            average_packet_size,
            packets_per_second,
            bytes_per_second,
            tcp_ratio,
            udp_ratio,
            icmp_ratio,
            unique_destination_port_count,
            unique_destination_ip_count,
        ]
    )

    # Clip to physically plausible non-negative ranges; ratios in [0, 1].
    features[:, 0] = np.clip(features[:, 0], 40, None)
    features[:, 1] = np.clip(features[:, 1], 0.1, None)
    features[:, 2] = np.clip(features[:, 2], 0, None)
    features[:, 3] = np.clip(features[:, 3], 0, 1)
    features[:, 4] = np.clip(features[:, 4], 0, 1)
    features[:, 5] = np.clip(features[:, 5], 0, 1)
    features[:, 6] = np.clip(np.round(features[:, 6]), 1, None)
    features[:, 7] = np.clip(np.round(features[:, 7]), 1, None)

    return features


def train_model(
    training_data: Optional[np.ndarray] = None, random_state: int = 42
) -> IsolationForest:
    """Train an IsolationForest on the given (or freshly generated) data."""
    if training_data is None:
        training_data = generate_normal_traffic_samples(random_state=random_state)

    model = IsolationForest(random_state=random_state)
    model.fit(training_data)
    return model


def save_model(model: IsolationForest, path: Path = MODEL_PATH) -> None:
    """Persist the trained model alongside feature metadata."""
    payload = {
        "model": model,
        "feature_names": list(FEATURE_NAMES),
        "feature_count": len(FEATURE_NAMES),
    }
    with open(path, "wb") as handle:
        pickle.dump(payload, handle)


def main() -> None:
    training_data = generate_normal_traffic_samples()
    model = train_model(training_data)
    save_model(model)
    print(f"Model trained on {training_data.shape[0]} samples and saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
