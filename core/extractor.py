"""Converts a window of raw packets into a fixed-length numerical feature
vector suitable for the ML anomaly detector.
"""

from __future__ import annotations

from typing import Any, List, Sequence

import numpy as np
from scapy.layers.inet import ICMP, IP, TCP, UDP
from scapy.layers.inet6 import IPv6, _ICMPv6

#: Canonical ordering of feature vector components. The ML training script
#: and the detector must both honor this exact order.
FEATURE_NAMES: List[str] = [
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


def _is_icmpv6(packet: Any) -> bool:
    """True if any layer of ``packet`` is an ICMPv6 message.

    ``packet.haslayer(_ICMPv6)`` does *not* work here: Scapy's ``haslayer``
    class matching does not walk the base-class hierarchy for this private
    abstract base, so every concrete ICMPv6 message type (echo
    request/reply, destination unreachable, neighbor discovery, ...) would
    have to be enumerated and kept in sync by hand. Walking
    ``packet.layers()`` and checking ``issubclass`` against the shared base
    catches all of them uniformly, the same way ``haslayer(ICMP)`` does for
    every ICMPv4 message type in one check.
    """
    return any(issubclass(layer_cls, _ICMPv6) for layer_cls in packet.layers())


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
    other_count = 0
    dest_ports = set()
    dest_ips = set()

    for packet in packets:
        try:
            total_bytes += len(packet)
        except Exception:
            pass

        if packet.haslayer(IP):
            dest_ips.add(packet[IP].dst)
        elif packet.haslayer(IPv6):
            dest_ips.add(packet[IPv6].dst)

        if packet.haslayer(TCP):
            tcp_count += 1
            dest_ports.add(int(packet[TCP].dport))
        elif packet.haslayer(UDP):
            udp_count += 1
            dest_ports.add(int(packet[UDP].dport))
        elif packet.haslayer(ICMP) or _is_icmpv6(packet):
            icmp_count += 1
        else:
            # ARP, IPv6 without a transport layer we track above (e.g.
            # neighbor discovery not counted as ICMPv6 by _is_icmpv6 --
            # it is, since ND messages subclass _ICMPv6, but this branch
            # also catches genuinely unrecognized/malformed traffic), or
            # any other non-TCP/UDP/ICMP protocol. Previously these
            # packets vanished silently: they still counted toward
            # packet_count and total_bytes but none of the ratio buckets,
            # so tcp_ratio + udp_ratio + icmp_ratio could sum to less
            # than 1 with no indication why.
            other_count += 1

    average_packet_size = total_bytes / packet_count
    packets_per_second = packet_count / window_seconds
    bytes_per_second = total_bytes / window_seconds
    tcp_ratio = tcp_count / packet_count
    udp_ratio = udp_count / packet_count
    icmp_ratio = icmp_count / packet_count
    other_ratio = other_count / packet_count
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
            other_ratio,
            unique_destination_port_count,
            unique_destination_ip_count,
        ],
        dtype=np.float64,
    )
