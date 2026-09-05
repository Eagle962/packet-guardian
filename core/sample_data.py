"""Generates a synthetic PCAP file containing normal TCP/UDP/ICMP traffic
plus a SYN scan scenario and an ICMP flood scenario.

Used as an offline fallback when live packet capture is unavailable (e.g.
due to insufficient OS privileges), and to drive the PCAP integration test
gate for the rule engine and TUI.
"""

from __future__ import annotations

from pathlib import Path

from scapy.layers.inet import ICMP, IP, TCP, UDP
from scapy.utils import wrpcap

DEFAULT_SAMPLE_PATH = Path(__file__).parent.parent / "data" / "sample.pcap"

#: Base epoch timestamp for the synthetic capture; arbitrary but fixed so
#: generated files are reproducible.
BASE_TIME = 1_700_000_000.0


def _timed(packet, timestamp: float):
    packet.time = timestamp
    return packet


def _build_normal_traffic(start: float) -> list:
    packets = []
    t = start

    # A handful of normal TCP handshakes to a couple of common ports.
    for port in (80, 443):
        for client_ip in ("10.0.0.10", "10.0.0.11"):
            packets.append(_timed(IP(src=client_ip, dst="10.0.0.1") / TCP(sport=40000 + port, dport=port, flags="S"), t))
            t += 0.05
            packets.append(_timed(IP(src="10.0.0.1", dst=client_ip) / TCP(sport=port, dport=40000 + port, flags="SA"), t))
            t += 0.05
            packets.append(_timed(IP(src=client_ip, dst="10.0.0.1") / TCP(sport=40000 + port, dport=port, flags="A"), t))
            t += 1.0

    # Normal DNS-like UDP traffic.
    for _ in range(5):
        packets.append(_timed(IP(src="10.0.0.12", dst="10.0.0.1") / UDP(sport=53000, dport=53), t))
        t += 1.0

    # Normal ping traffic, well under the ICMP flood threshold.
    for _ in range(5):
        packets.append(_timed(IP(src="10.0.0.13", dst="10.0.0.1") / ICMP(type=8), t))
        t += 0.5
        packets.append(_timed(IP(src="10.0.0.1", dst="10.0.0.13") / ICMP(type=0), t))
        t += 0.5

    return packets, t


def _build_syn_scan(start: float, attacker_ip: str = "10.0.0.66", victim_ip: str = "10.0.0.1") -> list:
    packets = []
    t = start
    for port in range(1, 31):
        packets.append(_timed(IP(src=attacker_ip, dst=victim_ip) / TCP(sport=55555, dport=port, flags="S"), t))
        t += 0.02
    return packets, t


def _build_icmp_flood(start: float, attacker_ip: str = "10.0.0.77", victim_ip: str = "10.0.0.1") -> list:
    packets = []
    t = start
    for _ in range(150):
        packets.append(_timed(IP(src=attacker_ip, dst=victim_ip) / ICMP(type=8), t))
        t += 0.01
    return packets, t


def generate_sample_pcap(path: Path = DEFAULT_SAMPLE_PATH) -> Path:
    """Build the synthetic capture and write it to ``path``.

    Returns the path written to.
    """
    all_packets = []

    normal_packets, t = _build_normal_traffic(BASE_TIME)
    all_packets.extend(normal_packets)

    scan_packets, t = _build_syn_scan(t + 1.0)
    all_packets.extend(scan_packets)

    flood_packets, t = _build_icmp_flood(t + 1.0)
    all_packets.extend(flood_packets)

    more_normal_packets, _ = _build_normal_traffic(t + 1.0)
    all_packets.extend(more_normal_packets)

    all_packets.sort(key=lambda pkt: float(pkt.time))

    path.parent.mkdir(parents=True, exist_ok=True)
    wrpcap(str(path), all_packets)
    return path


if __name__ == "__main__":
    written_to = generate_sample_pcap()
    print(f"Synthetic sample PCAP written to {written_to}")
