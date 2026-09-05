"""Converts a window of raw packets into a fixed-length numerical feature
vector suitable for the ML anomaly detector.

This is the single source of truth for feature computation: both
ml_engine/train.py (via ml_engine/traffic_profiles.py's packet-based
dataset) and ml_engine/detector.py (at inference time) must go through
extract_features() exactly as main.py does, so there is never a
train/serve skew between how a feature was computed during training vs.
during live detection.
"""

from __future__ import annotations

import math
from typing import Any, List, Sequence

import numpy as np
from scapy.layers.inet import ICMP, IP, TCP, UDP
from scapy.layers.inet6 import IPv6, _ICMPv6

#: Canonical ordering of feature vector components. ml_engine/train.py and
#: ml_engine/detector.py must both honor this exact order; detector.py's
#: load_model() validates a loaded artifact's stored feature list against
#: this at load time.
FEATURE_NAMES: List[str] = [
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


def _shannon_entropy(counts: Sequence[int]) -> float:
    """Shannon entropy, in bits, of a discrete distribution given as raw
    counts.

    0.0 means all mass on a single value (e.g. every packet went to the
    same destination port -- typical of a normal single-service session,
    or a single-port flood). Deliberately *not* normalized by log2(k): an
    earlier version divided by log2(k) (k = number of distinct values) to
    bound this to [0, 1], but that conflates "evenly split across 2
    values" with "evenly split across 50 values" -- both normalize to
    ~1.0, even though only the latter looks like a scan. Normal web
    traffic realistically only ever uses 1-2 destination ports (80/443)
    no matter how many sites are visited, and scored ~0.79-0.90 normalized
    entropy under the old scheme -- indistinguishable from an actual port
    scan's ~1.0. Raw entropy keeps that separation: 2 ports evenly split
    caps at 1 bit, a 50-port scan reaches ~5.6 bits.
    """
    total = sum(counts)
    if total == 0:
        return 0.0

    entropy = 0.0
    for count in counts:
        if count == 0:
            continue
        p = count / total
        entropy -= p * math.log2(p)

    return entropy


def extract_features(packets: Sequence[Any], window_seconds: float = 5.0) -> np.ndarray:
    """Compute a feature vector summarizing a window of packets.

    Args:
        packets: Sequence of Scapy packets captured within the window. May
            be empty. Packets are sorted by their ``.time`` attribute
            (falling back to the given order for packets without one) to
            compute inter-arrival statistics correctly regardless of the
            order they were handed in.
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
    packet_sizes: List[int] = []
    tcp_count = 0
    udp_count = 0
    icmp_count = 0
    other_count = 0
    syn_no_ack_count = 0
    dest_ports: dict = {}
    dest_ips = set()
    src_ips = set()
    timestamps: List[float] = []

    for packet in packets:
        try:
            size = len(packet)
        except Exception:
            size = 0
        total_bytes += size
        packet_sizes.append(size)

        if hasattr(packet, "time") and packet.time is not None:
            try:
                timestamps.append(float(packet.time))
            except (TypeError, ValueError):
                pass

        if packet.haslayer(IP):
            dest_ips.add(packet[IP].dst)
            src_ips.add(packet[IP].src)
        elif packet.haslayer(IPv6):
            dest_ips.add(packet[IPv6].dst)
            src_ips.add(packet[IPv6].src)

        if packet.haslayer(TCP):
            tcp_count += 1
            tcp_layer = packet[TCP]
            dport = int(tcp_layer.dport)
            dest_ports[dport] = dest_ports.get(dport, 0) + 1

            flags = int(tcp_layer.flags)
            is_syn = flags & 0x02 != 0
            is_ack = flags & 0x10 != 0
            if is_syn and not is_ack:
                syn_no_ack_count += 1
        elif packet.haslayer(UDP):
            udp_count += 1
            dest_ports[int(packet[UDP].dport)] = dest_ports.get(int(packet[UDP].dport), 0) + 1
        elif packet.haslayer(ICMP) or _is_icmpv6(packet):
            icmp_count += 1
        else:
            # ARP, IPv6 traffic with no transport layer we recognize above,
            # or any other protocol. Counted explicitly so tcp_ratio +
            # udp_ratio + icmp_ratio + other_ratio always sums to 1,
            # instead of silently vanishing from every bucket.
            other_count += 1

    average_packet_size = total_bytes / packet_count
    packets_per_second = packet_count / window_seconds
    tcp_ratio = tcp_count / packet_count
    udp_ratio = udp_count / packet_count
    icmp_ratio = icmp_count / packet_count
    other_ratio = other_count / packet_count
    # SYN-without-ACK as a fraction of TCP traffic specifically (not of all
    # traffic): a scan is characterized by *how much of its TCP* is bare
    # SYNs, which stays informative even in a window with little TCP
    # overall. 0.0 when there is no TCP traffic to take a ratio of.
    syn_ratio = syn_no_ack_count / tcp_count if tcp_count > 0 else 0.0

    unique_destination_port_count = len(dest_ports)
    unique_destination_ip_count = len(dest_ips)
    unique_source_ip_count = len(src_ips)

    dest_port_entropy = _shannon_entropy(list(dest_ports.values()))

    # ddof=0 (population std): we're describing this exact window, not
    # estimating a std for a larger population it was sampled from.
    packet_size_std = float(np.std(packet_sizes)) if len(packet_sizes) >= 2 else 0.0

    if len(timestamps) >= 2:
        timestamps.sort()
        deltas = np.diff(timestamps)
        mean_inter_arrival = float(np.mean(deltas))
        std_inter_arrival = float(np.std(deltas))
    else:
        mean_inter_arrival = 0.0
        std_inter_arrival = 0.0

    return np.array(
        [
            average_packet_size,
            packets_per_second,
            tcp_ratio,
            udp_ratio,
            icmp_ratio,
            other_ratio,
            syn_ratio,
            unique_destination_port_count,
            unique_destination_ip_count,
            unique_source_ip_count,
            dest_port_entropy,
            packet_size_std,
            mean_inter_arrival,
            std_inter_arrival,
        ],
        dtype=np.float64,
    )
