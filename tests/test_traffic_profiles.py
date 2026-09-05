"""Tests for ml_engine.traffic_profiles: the labelled synthetic traffic
generator used to train and evaluate the ML detector."""

import random

import numpy as np

from core.extractor import FEATURE_NAMES
from ml_engine.traffic_profiles import (
    ALL_PROFILES,
    ATTACK_PROFILES,
    NORMAL_PROFILES,
    build_feature_dataset,
    generate_dataset,
    generate_window,
)


class TestProfilesProduceValidPackets:
    def test_every_profile_runs_without_crashing(self) -> None:
        rng = random.Random(0)
        for name in ALL_PROFILES:
            packets = generate_window(name, rng, window_seconds=5.0)
            assert isinstance(packets, list)

    def test_every_profile_packets_have_timestamps_within_window(self) -> None:
        rng = random.Random(0)
        window_seconds = 5.0
        for name in ALL_PROFILES:
            packets = generate_window(name, rng, window_seconds)
            for pkt in packets:
                assert 0.0 <= float(pkt.time) <= window_seconds

    def test_every_profile_features_extractable(self) -> None:
        rng = random.Random(0)
        for name in ALL_PROFILES:
            packets = generate_window(name, rng, window_seconds=5.0)
            from core.extractor import extract_features

            features = extract_features(packets, window_seconds=5.0)
            assert features.shape == (len(FEATURE_NAMES),)
            assert np.all(np.isfinite(features))


class TestLabels:
    def test_normal_and_attack_profiles_disjoint(self) -> None:
        assert set(NORMAL_PROFILES) & set(ATTACK_PROFILES) == set()

    def test_minimum_required_profiles_present(self) -> None:
        required_normal = {
            "idle",
            "web_browsing",
            "video_streaming",
            "bulk_file_transfer",
            "dns_heavy",
            "voip_udp",
            "mixed_office",
            "normal_with_bursts",
        }
        required_attack = {
            "fast_syn_scan",
            "slow_syn_scan",
            "port_sweep",
            "icmp_flood",
            "udp_flood",
            "syn_flood_single_port",
        }
        assert required_normal <= set(NORMAL_PROFILES)
        assert required_attack <= set(ATTACK_PROFILES)


class TestGenerateDataset:
    def test_reproducible_with_same_seed(self) -> None:
        names = list(ALL_PROFILES.keys())
        a = generate_dataset(names, n_per_profile=3, seed=42)
        b = generate_dataset(names, n_per_profile=3, seed=42)
        assert len(a) == len(b)
        for sa, sb in zip(a, b):
            assert sa.scenario == sb.scenario
            assert sa.label == sb.label
            assert len(sa.packets) == len(sb.packets)
            for pa, pb in zip(sa.packets, sb.packets):
                assert float(pa.time) == float(pb.time)

    def test_different_seeds_produce_different_instances(self) -> None:
        names = ["web_browsing"]
        a = generate_dataset(names, n_per_profile=5, seed=1)
        b = generate_dataset(names, n_per_profile=5, seed=2)
        counts_a = [len(s.packets) for s in a]
        counts_b = [len(s.packets) for s in b]
        assert counts_a != counts_b

    def test_labels_assigned_correctly(self) -> None:
        names = ["idle", "fast_syn_scan"]
        samples = generate_dataset(names, n_per_profile=2, seed=1)
        for s in samples:
            if s.scenario == "idle":
                assert s.label == "normal"
            else:
                assert s.label == "attack"


class TestBuildFeatureDataset:
    def test_shapes(self) -> None:
        names = list(ALL_PROFILES.keys())
        X, labels, scenarios = build_feature_dataset(names, n_per_profile=2, seed=1)
        n_expected = len(names) * 2
        assert X.shape == (n_expected, len(FEATURE_NAMES))
        assert len(labels) == n_expected
        assert len(scenarios) == n_expected
        assert set(labels) <= {"normal", "attack"}

    def test_fast_scan_has_high_syn_ratio_and_port_entropy(self) -> None:
        X, labels, scenarios = build_feature_dataset(["fast_syn_scan"], n_per_profile=10, seed=1)
        syn_ratio_idx = FEATURE_NAMES.index("syn_ratio")
        entropy_idx = FEATURE_NAMES.index("dest_port_entropy")
        assert np.mean(X[:, syn_ratio_idx]) > 0.9
        assert np.mean(X[:, entropy_idx]) > 0.7

    def test_icmp_flood_has_high_icmp_ratio(self) -> None:
        X, labels, scenarios = build_feature_dataset(["icmp_flood"], n_per_profile=10, seed=1)
        icmp_idx = FEATURE_NAMES.index("icmp_ratio")
        assert np.all(X[:, icmp_idx] > 0.95)
