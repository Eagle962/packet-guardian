"""Tests for core.extractor.extract_features."""

import numpy as np
import pytest
from scapy.layers.inet import ICMP, IP, TCP, UDP
from scapy.layers.inet6 import ICMPv6EchoRequest, IPv6
from scapy.layers.l2 import ARP, Ether

from core.extractor import FEATURE_NAMES, _shannon_entropy_normalized, extract_features


def make_tcp_packet(src: str = "10.0.0.1", dst: str = "10.0.0.2", dport: int = 80, flags: str = "S", timestamp=None):
    pkt = IP(src=src, dst=dst) / TCP(sport=12345, dport=dport, flags=flags)
    if timestamp is not None:
        pkt.time = timestamp
    return pkt


def make_udp_packet(src: str = "10.0.0.1", dst: str = "10.0.0.2", dport: int = 53, timestamp=None):
    pkt = IP(src=src, dst=dst) / UDP(sport=12345, dport=dport)
    if timestamp is not None:
        pkt.time = timestamp
    return pkt


def make_icmp_packet(src: str = "10.0.0.1", dst: str = "10.0.0.2", timestamp=None):
    pkt = IP(src=src, dst=dst) / ICMP(type=8)
    if timestamp is not None:
        pkt.time = timestamp
    return pkt


class TestFeatureNames:
    def test_feature_names_exact_order(self) -> None:
        assert FEATURE_NAMES == [
            "average_packet_size",
            "packets_per_second",
            "tcp_ratio",
            "udp_ratio",
            "icmp_ratio",
            "other_ratio",
            "syn_ratio",
            "unique_destination_port_count",
            "unique_destination_ip_count",
            "unique_source_ip_count",
            "dest_port_entropy",
            "packet_size_std",
            "mean_inter_arrival",
            "std_inter_arrival",
        ]

    def test_no_collinear_bytes_per_second(self) -> None:
        # bytes_per_second was exactly average_packet_size * packets_per_second
        # -- a perfectly collinear feature -- and has been removed.
        assert "bytes_per_second" not in FEATURE_NAMES


class TestShannonEntropyNormalized:
    def test_single_value_is_zero_entropy(self) -> None:
        assert _shannon_entropy_normalized([10]) == 0.0

    def test_uniform_distribution_is_max_entropy(self) -> None:
        assert _shannon_entropy_normalized([5, 5, 5, 5]) == pytest.approx(1.0)

    def test_skewed_distribution_is_between(self) -> None:
        entropy = _shannon_entropy_normalized([100, 1, 1, 1])
        assert 0.0 < entropy < 1.0

    def test_empty_is_zero(self) -> None:
        assert _shannon_entropy_normalized([]) == 0.0


class TestExtractFeatures:
    def test_empty_packet_list_returns_zero_vector(self) -> None:
        features = extract_features([], window_seconds=5.0)
        assert features.shape == (len(FEATURE_NAMES),)
        assert np.all(features == 0)

    def test_zero_window_seconds_does_not_crash(self) -> None:
        packets = [make_tcp_packet()]
        features = extract_features(packets, window_seconds=0)
        assert features.shape == (len(FEATURE_NAMES),)
        assert np.all(features == 0)

    def test_dimensions_correct(self) -> None:
        packets = [make_tcp_packet(), make_udp_packet(), make_icmp_packet()]
        features = extract_features(packets, window_seconds=5.0)
        assert features.shape == (len(FEATURE_NAMES),)

    def test_all_tcp_traffic_ratios(self) -> None:
        packets = [make_tcp_packet(dport=p) for p in (80, 443, 8080)]
        features = extract_features(packets, window_seconds=5.0)
        as_dict = dict(zip(FEATURE_NAMES, features))
        assert as_dict["tcp_ratio"] == 1.0
        assert as_dict["udp_ratio"] == 0.0
        assert as_dict["icmp_ratio"] == 0.0
        assert as_dict["other_ratio"] == 0.0
        assert as_dict["unique_destination_port_count"] == 3
        assert as_dict["unique_destination_ip_count"] == 1

    def test_mixed_protocol_traffic(self) -> None:
        packets = [
            make_tcp_packet(dport=80),
            make_tcp_packet(dport=443),
            make_udp_packet(dport=53),
            make_icmp_packet(),
        ]
        features = extract_features(packets, window_seconds=2.0)
        as_dict = dict(zip(FEATURE_NAMES, features))
        assert as_dict["tcp_ratio"] == 0.5
        assert as_dict["udp_ratio"] == 0.25
        assert as_dict["icmp_ratio"] == 0.25
        assert as_dict["other_ratio"] == 0.0
        assert as_dict["packets_per_second"] == 4 / 2.0
        assert as_dict["unique_destination_port_count"] == 3  # 80, 443, 53

    def test_packets_missing_ip_layer_do_not_crash(self) -> None:
        packets = [Ether(), Ether()]
        features = extract_features(packets, window_seconds=5.0)
        assert features.shape == (len(FEATURE_NAMES),)
        as_dict = dict(zip(FEATURE_NAMES, features))
        assert as_dict["unique_destination_ip_count"] == 0
        assert as_dict["unique_destination_port_count"] == 0
        assert as_dict["other_ratio"] == 1.0

    def test_non_tcp_udp_icmp_traffic_counted_as_other_not_dropped(self) -> None:
        packets = [make_tcp_packet() for _ in range(3)] + [Ether() / ARP() for _ in range(3)]
        features = extract_features(packets, window_seconds=1.0)
        as_dict = dict(zip(FEATURE_NAMES, features))
        assert as_dict["tcp_ratio"] == 0.5
        assert as_dict["other_ratio"] == 0.5
        assert (
            as_dict["tcp_ratio"] + as_dict["udp_ratio"] + as_dict["icmp_ratio"] + as_dict["other_ratio"]
            == 1.0
        )

    def test_unique_destination_counts(self) -> None:
        packets = [
            make_tcp_packet(dst="10.0.0.2", dport=80),
            make_tcp_packet(dst="10.0.0.3", dport=80),
            make_tcp_packet(dst="10.0.0.2", dport=443),
        ]
        features = extract_features(packets, window_seconds=5.0)
        as_dict = dict(zip(FEATURE_NAMES, features))
        assert as_dict["unique_destination_ip_count"] == 2
        assert as_dict["unique_destination_port_count"] == 2

    def test_average_packet_size(self) -> None:
        packets = [make_tcp_packet(), make_tcp_packet()]
        features = extract_features(packets, window_seconds=1.0)
        as_dict = dict(zip(FEATURE_NAMES, features))
        total_bytes = sum(len(p) for p in packets)
        assert as_dict["average_packet_size"] == total_bytes / 2

    def test_syn_ratio_counts_syn_without_ack_over_tcp_total(self) -> None:
        packets = [
            make_tcp_packet(dport=1, flags="S"),
            make_tcp_packet(dport=2, flags="S"),
            make_tcp_packet(dport=80, flags="SA"),
            make_tcp_packet(dport=80, flags="A"),
        ]
        features = extract_features(packets, window_seconds=1.0)
        as_dict = dict(zip(FEATURE_NAMES, features))
        assert as_dict["syn_ratio"] == 0.5  # 2 bare SYNs / 4 TCP packets

    def test_syn_ratio_zero_when_no_tcp(self) -> None:
        packets = [make_udp_packet(), make_icmp_packet()]
        features = extract_features(packets, window_seconds=1.0)
        as_dict = dict(zip(FEATURE_NAMES, features))
        assert as_dict["syn_ratio"] == 0.0

    def test_unique_source_ip_count(self) -> None:
        packets = [
            make_tcp_packet(src="10.0.0.1"),
            make_tcp_packet(src="10.0.0.2"),
            make_tcp_packet(src="10.0.0.1"),
        ]
        features = extract_features(packets, window_seconds=1.0)
        as_dict = dict(zip(FEATURE_NAMES, features))
        assert as_dict["unique_source_ip_count"] == 2

    def test_dest_port_entropy_low_for_single_service(self) -> None:
        packets = [make_tcp_packet(dport=443) for _ in range(20)]
        features = extract_features(packets, window_seconds=5.0)
        as_dict = dict(zip(FEATURE_NAMES, features))
        assert as_dict["dest_port_entropy"] == 0.0

    def test_dest_port_entropy_high_for_scan(self) -> None:
        packets = [make_tcp_packet(dport=p) for p in range(1, 21)]
        features = extract_features(packets, window_seconds=5.0)
        as_dict = dict(zip(FEATURE_NAMES, features))
        assert as_dict["dest_port_entropy"] == pytest.approx(1.0)

    def test_packet_size_std_zero_for_uniform_sizes(self) -> None:
        packets = [make_tcp_packet(dport=80) for _ in range(5)]
        features = extract_features(packets, window_seconds=1.0)
        as_dict = dict(zip(FEATURE_NAMES, features))
        assert as_dict["packet_size_std"] == 0.0

    def test_packet_size_std_single_packet_is_zero(self) -> None:
        features = extract_features([make_tcp_packet()], window_seconds=1.0)
        as_dict = dict(zip(FEATURE_NAMES, features))
        assert as_dict["packet_size_std"] == 0.0

    def test_inter_arrival_stats_computed_from_timestamps(self) -> None:
        packets = [
            make_tcp_packet(dport=80, timestamp=0.0),
            make_tcp_packet(dport=80, timestamp=1.0),
            make_tcp_packet(dport=80, timestamp=2.0),
        ]
        features = extract_features(packets, window_seconds=5.0)
        as_dict = dict(zip(FEATURE_NAMES, features))
        assert as_dict["mean_inter_arrival"] == pytest.approx(1.0)
        assert as_dict["std_inter_arrival"] == pytest.approx(0.0)

    def test_inter_arrival_stats_sorted_regardless_of_input_order(self) -> None:
        packets = [
            make_tcp_packet(dport=80, timestamp=2.0),
            make_tcp_packet(dport=80, timestamp=0.0),
            make_tcp_packet(dport=80, timestamp=1.0),
        ]
        features = extract_features(packets, window_seconds=5.0)
        as_dict = dict(zip(FEATURE_NAMES, features))
        assert as_dict["mean_inter_arrival"] == pytest.approx(1.0)

    def test_inter_arrival_stats_zero_for_single_packet(self) -> None:
        features = extract_features([make_tcp_packet(timestamp=5.0)], window_seconds=5.0)
        as_dict = dict(zip(FEATURE_NAMES, features))
        assert as_dict["mean_inter_arrival"] == 0.0
        assert as_dict["std_inter_arrival"] == 0.0


class TestExtractFeaturesIPv6:
    def test_ipv6_destination_and_source_counted(self) -> None:
        packets = [
            IPv6(src="2001:db8::1", dst="2001:db8::2") / TCP(sport=1, dport=80, flags="S")
            for _ in range(5)
        ]
        features = extract_features(packets, window_seconds=1.0)
        as_dict = dict(zip(FEATURE_NAMES, features))
        assert as_dict["unique_destination_ip_count"] == 1
        assert as_dict["unique_source_ip_count"] == 1
        assert as_dict["tcp_ratio"] == 1.0

    def test_icmpv6_counts_toward_icmp_ratio(self) -> None:
        packets = [IPv6(src="2001:db8::1", dst="2001:db8::2") / ICMPv6EchoRequest() for _ in range(4)]
        features = extract_features(packets, window_seconds=1.0)
        as_dict = dict(zip(FEATURE_NAMES, features))
        assert as_dict["icmp_ratio"] == 1.0
