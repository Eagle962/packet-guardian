"""packet-guardian: modular network traffic anomaly detection tool.

Orchestrates packet capture (live or PCAP), rule-based detection, feature
extraction, and ML-based anomaly detection, and drives the Rich terminal
dashboard.
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Any, List, Optional

from rich.console import Console
from rich.live import Live
from scapy.error import Scapy_Exception

from core.extractor import FEATURE_NAMES, extract_features
from core.rules import IcmpFloodRule, Ruleset, SynScanRule
from core.sample_data import DEFAULT_SAMPLE_PATH, generate_sample_pcap
from core.sniffer import capture_live, read_pcap
from ml_engine.detector import ModelArtifact, load_model, predict
from ui.console import Dashboard

DEFAULT_WINDOW_SECONDS = 5.0


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="packet-guardian",
        description="Modular network traffic anomaly detection tool.",
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--pcap", type=str, default=None, help="Path to a PCAP file to analyze offline."
    )
    source_group.add_argument(
        "--interface", type=str, default=None, help="Network interface to sniff live traffic from."
    )
    parser.add_argument(
        "--bpf", type=str, default=None, help="Berkeley Packet Filter expression to restrict traffic."
    )
    parser.add_argument(
        "--window",
        type=float,
        default=DEFAULT_WINDOW_SECONDS,
        help="Feature extraction / rule window size in seconds (default: 5.0).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Live capture duration in seconds (live mode only).",
    )
    return parser.parse_args(argv)


class Orchestrator:
    """Feeds packets through the rule engine and ML detector, driving the
    dashboard as traffic windows complete."""

    def __init__(
        self,
        window_seconds: float,
        model_artifact: Optional[ModelArtifact],
        dashboard: Dashboard,
    ) -> None:
        self.window_seconds = window_seconds
        self.model_artifact = model_artifact
        self.dashboard = dashboard
        self.ruleset = Ruleset(
            [
                SynScanRule(window_seconds=window_seconds),
                IcmpFloodRule(window_seconds=window_seconds),
            ]
        )
        self._window_packets: List[Any] = []
        self._window_start: Optional[float] = None
        self.total_packets = 0

    def handle_packet(self, packet: Any) -> None:
        timestamp = float(packet.time) if hasattr(packet, "time") else time.time()
        self.total_packets += 1

        for alert in self.ruleset.process_packet(packet, timestamp=timestamp):
            self.dashboard.add_alert(alert)

        if self._window_start is None:
            self._window_start = timestamp
        self._window_packets.append(packet)

        if timestamp - self._window_start >= self.window_seconds:
            self._flush_window()

    def finalize(self) -> None:
        if self._window_packets:
            self._flush_window()

    def _flush_window(self) -> None:
        features = extract_features(self._window_packets, window_seconds=self.window_seconds)

        if self.model_artifact is not None:
            ml_result = predict(features, self.model_artifact)
        else:
            ml_result = {"anomaly_score": 0.0, "status": "UNKNOWN"}

        feature_dict = dict(zip(FEATURE_NAMES, features))
        stats = {
            "total_packets": self.total_packets,
            "packets_per_second": feature_dict["packets_per_second"],
            "bytes_per_second": feature_dict["bytes_per_second"],
            "average_packet_size": feature_dict["average_packet_size"],
            "tcp_ratio": feature_dict["tcp_ratio"],
            "udp_ratio": feature_dict["udp_ratio"],
            "icmp_ratio": feature_dict["icmp_ratio"],
            "anomaly_score": ml_result["anomaly_score"],
            "ml_status": ml_result["status"],
        }
        self.dashboard.update_stats(stats)

        self._window_packets = []
        self._window_start = None


def _load_model_or_warn(console: Console) -> Optional[ModelArtifact]:
    try:
        return load_model()
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[yellow]ML model unavailable ({exc}); continuing with rule-based detection only.[/yellow]")
        return None


def run(args: argparse.Namespace) -> int:
    console = Console()
    model_artifact = _load_model_or_warn(console)
    dashboard = Dashboard()
    orchestrator = Orchestrator(args.window, model_artifact, dashboard)

    pcap_path = args.pcap

    with Live(dashboard.render(), console=console, refresh_per_second=4) as live:
        dashboard.attach_live(live)

        if args.interface is not None:
            try:
                capture_live(args.interface, args.bpf, args.timeout, orchestrator.handle_packet)
            except (PermissionError, OSError, ValueError, Scapy_Exception) as exc:
                console.print(
                    f"[yellow]Live capture on '{args.interface}' failed ({exc}); "
                    f"falling back to synthetic PCAP offline mode.[/yellow]"
                )
                if not DEFAULT_SAMPLE_PATH.exists():
                    generate_sample_pcap(DEFAULT_SAMPLE_PATH)
                pcap_path = str(DEFAULT_SAMPLE_PATH)

        if pcap_path is not None:
            read_pcap(pcap_path, args.bpf, orchestrator.handle_packet)

        orchestrator.finalize()

    return 0


def main() -> None:
    args = parse_args()
    sys.exit(run(args))


if __name__ == "__main__":
    main()
