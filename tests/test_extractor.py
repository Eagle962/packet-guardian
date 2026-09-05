"""Tests for core.extractor.extract_features."""

import numpy as np
from scapy.layers.inet import ICMP, IP, TCP, UDP
from scapy.layers.l2 import ARP, Ether

from core.extractor import FEATURE_NAMES, extract_features


def make_tcp_packet(src: str = "10.0.0.1", dst: str = "10.0.0.2", dport: int = 80):
    return IP(src=src, dst=dst) / TCP(sport=12345, dport=dport, flags="S")


def make_udp_packet(src: str = "10.0.0.1", dst: str = "10.0.0.2", dport: int = 53):
    return IP(src=src, dst=dst) / UDP(sport=12345, dport=dport)


def make_icmp_packet(src: str = "10.0.0.1", dst: str = "10.0.0.2"):
    return IP(src=src, dst=dst) / ICMP(type=8)


class TestFeatureNames:
    def test_feature_names_exact_order(self) -> None:
        assert FEATURE_NAMES == [
            "average_packet_size",
            "packets_per_second",
            "bytes_per_second",
            "tcp_ratio",
            "udp_ratio",
            "icmp_ratio",
            "other_ratio",
            "unique_destination_port_count",
            "unique_destination_ip_count",
        ]


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
        """Regression test: ARP (and any other non-TCP/UDP/ICMP protocol)
        used to vanish from the ratio features entirely -- tcp_ratio +
        udp_ratio + icmp_ratio summed to less than 1 with no accounting
        for where the rest of the traffic went."""
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

    def test_average_packet_size_and_bytes_per_second(self) -> None:
        packets = [make_tcp_packet(), make_tcp_packet()]
        features = extract_features(packets, window_seconds=1.0)
        as_dict = dict(zip(FEATURE_NAMES, features))
        total_bytes = sum(len(p) for p in packets)
        assert as_dict["average_packet_size"] == total_bytes / 2
        assert as_dict["bytes_per_second"] == total_bytes / 1.0
