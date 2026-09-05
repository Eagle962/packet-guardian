"""Tests for core.sniffer: BPF filtering in both offline (read_pcap) and
live (capture_live) paths. Live capture itself cannot be exercised in a
sandboxed/CI environment without root privileges and a real interface, so
capture_live is verified by confirming it passes the filter through to
Scapy's sniff() correctly rather than by capturing real traffic.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from scapy.layers.inet import ICMP, IP, TCP, UDP
from scapy.utils import wrpcap

from core.sniffer import capture_live, read_pcap


def _write_mixed_pcap(path: Path) -> None:
    packets = [
        IP(src="10.0.0.1", dst="10.0.0.2") / TCP(sport=1234, dport=80, flags="S"),
        IP(src="10.0.0.1", dst="10.0.0.2") / TCP(sport=1234, dport=443, flags="S"),
        IP(src="10.0.0.1", dst="10.0.0.2") / UDP(sport=1234, dport=53),
        IP(src="10.0.0.1", dst="10.0.0.2") / ICMP(type=8),
    ]
    wrpcap(str(path), packets)


class TestReadPcapBpfFiltering:
    def test_no_filter_yields_every_packet(self, tmp_path) -> None:
        path = tmp_path / "mixed.pcap"
        _write_mixed_pcap(path)

        seen = []
        read_pcap(str(path), None, seen.append)
        assert len(seen) == 4

    def test_tcp_filter_yields_only_tcp_packets(self, tmp_path) -> None:
        path = tmp_path / "mixed.pcap"
        _write_mixed_pcap(path)

        seen = []
        read_pcap(str(path), "tcp", seen.append)
        assert len(seen) == 2
        assert all(pkt.haslayer(TCP) for pkt in seen)

    def test_icmp_filter_yields_only_icmp_packets(self, tmp_path) -> None:
        path = tmp_path / "mixed.pcap"
        _write_mixed_pcap(path)

        seen = []
        read_pcap(str(path), "icmp", seen.append)
        assert len(seen) == 1
        assert seen[0].haslayer(ICMP)

    def test_udp_filter_yields_only_udp_packets(self, tmp_path) -> None:
        path = tmp_path / "mixed.pcap"
        _write_mixed_pcap(path)

        seen = []
        read_pcap(str(path), "udp", seen.append)
        assert len(seen) == 1
        assert seen[0].haslayer(UDP)

    def test_host_filter_yields_matching_packets_by_port(self, tmp_path) -> None:
        path = tmp_path / "mixed.pcap"
        _write_mixed_pcap(path)

        seen = []
        read_pcap(str(path), "port 443", seen.append)
        assert len(seen) == 1
        assert seen[0][TCP].dport == 443


class TestCaptureLiveBpfFiltering:
    """Live capture can't be exercised without root + a real interface in
    this environment, so we verify the integration point instead: that
    capture_live() forwards interface/filter/timeout/callback to Scapy's
    sniff() exactly as given, rather than dropping or mangling the BPF
    expression before it reaches Scapy."""

    def test_forwards_bpf_filter_to_scapy_sniff(self) -> None:
        with patch("core.sniffer.sniff") as mock_sniff:
            callback = lambda pkt: None  # noqa: E731
            capture_live("en0", "tcp and port 22", 5.0, callback)

        mock_sniff.assert_called_once_with(
            iface="en0",
            filter="tcp and port 22",
            timeout=5.0,
            prn=callback,
            store=False,
        )

    def test_forwards_none_filter_unchanged(self) -> None:
        with patch("core.sniffer.sniff") as mock_sniff:
            callback = lambda pkt: None  # noqa: E731
            capture_live(None, None, None, callback)

        mock_sniff.assert_called_once_with(
            iface=None,
            filter=None,
            timeout=None,
            prn=callback,
            store=False,
        )
