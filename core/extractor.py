"""Converts a window of raw packets into a fixed-length numerical feature
vector suitable for the ML anomaly detector.
"""

from __future__ import annotations

from typing import Any, List, Sequence

import numpy as np
from scapy.layers.inet import ICMP, IP, TCP, UDP

#: Canonical ordering of feature vector components. The ML training script
#: and the detector must both honor this exact order.
FEATURE_NAMES: List[str] = [
    "average_packet_size",
    "packets_per_second",
    "bytes_per_second",
    "tcp_ratio",
    "udp_ratio",
    "icmp_ratio",
    "unique_destination_port_count",
    "unique_destination_ip_count",
]


def extract_features(packets: Sequence[Any], window_seconds: float = 5.0) -> np.ndarray:
    """Compute a feature vector summarizing a window of packets.

    Args:
        packets: Sequence of Scapy packets captured within the window.
            May be empty.
        window_seconds: Duration of the window in seconds, used to derive
            per-second rates. Must be positive.

    Returns:
        A 1-D numpy array of length ``len(FEATURE_NAMES)``, ordered
        according to ``FEATURE_NAMES``.
    """
    packet_count = len(packets)

    if packet_count == 0 or window_seconds <= 0:
        return np.zeros(len(FEATURE_NAMES), dtype=np.float64)

    total_bytes = 0
    tcp_count = 0
    udp_count = 0
    icmp_count = 0
    dest_ports = set()
    dest_ips = set()

    for packet in packets:
        try:
            total_bytes += len(packet)
        except Exception:
            pass

        has_ip = packet.haslayer(IP)
        if has_ip:
            dest_ips.add(packet[IP].dst)

        if packet.haslayer(TCP):
            tcp_count += 1
            dest_ports.add(int(packet[TCP].dport))
        elif packet.haslayer(UDP):
            udp_count += 1
            dest_ports.add(int(packet[UDP].dport))
        elif packet.haslayer(ICMP):
            icmp_count += 1

    average_packet_size = total_bytes / packet_count
    packets_per_second = packet_count / window_seconds
    bytes_per_second = total_bytes / window_seconds
    tcp_ratio = tcp_count / packet_count
    udp_ratio = udp_count / packet_count
    icmp_ratio = icmp_count / packet_count
    unique_destination_port_count = len(dest_ports)
    unique_destination_ip_count = len(dest_ips)

    return np.array(
        [
            average_packet_size,
            packets_per_second,
            bytes_per_second,
            tcp_ratio,
            udp_ratio,
            icmp_ratio,
            unique_destination_port_count,
            unique_destination_ip_count,
        ],
        dtype=np.float64,
    )
