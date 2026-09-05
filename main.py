"""packet-guardian: modular network traffic anomaly detection tool.

Orchestrates packet capture (live or PCAP), rule-based detection, feature
extraction, and ML-based anomaly detection, and drives the Rich terminal
dashboard.
"""

from __future__ import annotations

import argparse
import sys
import threading
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
    dashboard as traffic windows complete.

    Window emission is **time-driven, not event-driven**: windows are
    flushed on a fixed schedule of ``window_seconds``-sized buckets,
    including buckets that end up containing zero packets. This matters for
    two reasons: (1) rate features (packets/sec, bytes/sec) must be divided
    by how much time actually elapsed, not by a constant, and (2) a link
    that goes idle must keep producing windows so the dashboard doesn't
    appear frozen and so a long gap isn't silently averaged away.

    - In **PCAP mode**, the clock is derived entirely from packet
      timestamps (``packet.time``): a gap between two packets causes every
      intervening empty window to be emitted before the next packet's own
      window opens.
    - In **live mode**, a background thread (see :meth:`start_live_clock`)
      drives the same schedule off wall-clock time, so idle windows are
      still emitted even though no packet ever arrives to trigger them.

    The very first packet anchors ``window_start`` to its own timestamp
    (PCAP mode) or, if live, the moment capture starts. Windows are
    therefore ``[window_start + k*window_seconds, window_start +
    (k+1)*window_seconds)`` relative to that anchor -- not to absolute wall
    time -- which keeps the scheme identical between live and offline mode.
    """

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
        self._last_timestamp: Optional[float] = None
        self.total_packets = 0

        # Guards all of the mutable state above. Only needed because live
        # mode's background clock thread (start_live_clock) and the
        # sniff() callback thread both call into this object concurrently;
        # in PCAP mode everything runs on a single thread and contention
        # never occurs, but taking the lock unconditionally keeps the two
        # code paths identical and avoids a second, untested code branch.
        self._lock = threading.Lock()
        self._timer_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def handle_packet(self, packet: Any) -> None:
        timestamp = float(packet.time) if hasattr(packet, "time") else time.time()

        with self._lock:
            self.total_packets += 1

            for alert in self.ruleset.process_packet(packet, timestamp=timestamp):
                self.dashboard.add_alert(alert)

            if self._window_start is None:
                self._window_start = timestamp

            # Emit every window that fully elapsed strictly before this
            # packet arrived -- this is what makes a gap between packets
            # (e.g. 60s of silence in PCAP mode) produce the intervening
            # empty windows instead of one wildly-averaged window.
            self._emit_due_windows(timestamp)

            self._window_packets.append(packet)
            self._last_timestamp = timestamp

    def start_live_clock(self) -> None:
        """Start a background thread that emits windows on a wall-clock
        schedule, independent of packet arrival.

        Design choice: a plain daemon thread, not
        ``scapy.sendrecv.AsyncSniffer``. ``core/sniffer.py`` is the only
        module allowed to depend on Scapy's capture APIs, and
        ``capture_live()`` is meant to stay a thin *blocking* wrapper around
        ``sniff()``. Driving window emission from an AsyncSniffer would
        require main.py to manage a Scapy object's start/stop/join
        lifecycle directly (leaking the capture API across the layering
        boundary) or push that lifecycle management into the sniffer
        wrapper (turning it into more than a thin wrapper). A plain timer
        thread needs no Scapy knowledge at all: it only reads the wall
        clock and takes the same lock ``handle_packet`` already uses.
        """
        with self._lock:
            if self._window_start is None:
                self._window_start = time.time()

        def _tick() -> None:
            while not self._stop_event.wait(self.window_seconds):
                with self._lock:
                    self._emit_due_windows(time.time())

        self._timer_thread = threading.Thread(target=_tick, daemon=True)
        self._timer_thread.start()

    def stop_live_clock(self) -> None:
        if self._timer_thread is not None:
            self._stop_event.set()
            self._timer_thread.join(timeout=self.window_seconds + 1.0)
            self._timer_thread = None
            self._stop_event = threading.Event()

    def finalize(self) -> None:
        with self._lock:
            if self._window_start is None:
                return

            end = self._last_timestamp if self._last_timestamp is not None else time.time()
            span = end - self._window_start
            if span <= 0:
                # Exactly one packet (or a burst sharing one timestamp)
                # with nothing after it to bound the window's true
                # duration. Nothing suggests this window was shorter than
                # nominal, so fall back to the configured window size
                # rather than fabricating an infinite rate from a
                # zero-length span.
                span = self.window_seconds

            packets, self._window_packets = self._window_packets, []
            self._emit_window(packets, span)
            self._window_start = None
            self._last_timestamp = None

    def _emit_due_windows(self, now: float) -> None:
        """Emit every complete ``window_seconds`` bucket that has fully
        elapsed by ``now``, oldest first. Must be called with ``_lock``
        held."""
        while self._window_start is not None and now - self._window_start >= self.window_seconds:
            packets, self._window_packets = self._window_packets, []
            self._emit_window(packets, self.window_seconds)
            self._window_start += self.window_seconds

    def _emit_window(self, packets: List[Any], span: float) -> None:
        has_traffic = len(packets) > 0
        features = extract_features(packets, window_seconds=span)

        if not has_traffic:
            # An all-zero feature vector previously scored ANOMALY against
            # the ML model, producing constant false alarms on any idle
            # link. Empty windows are still reported to the dashboard (so
            # it doesn't look frozen) but are never scored by the model.
            ml_result = {"anomaly_score": 0.0, "status": "NO_TRAFFIC"}
        elif self.model_artifact is not None:
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
            "has_traffic": has_traffic,
            "window_span": span,
        }
        self.dashboard.update_stats(stats)


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
    exit_code = 0

    try:
        with Live(dashboard.render(), console=console, refresh_per_second=4) as live:
            dashboard.attach_live(live)

            if args.interface is not None:
                orchestrator.start_live_clock()
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
                finally:
                    # Stop the wall-clock timer before switching to PCAP
                    # replay: the two modes drive the same window-start
                    # anchor from different clocks (wall time vs. packet
                    # timestamps), so they must never run concurrently.
                    orchestrator.stop_live_clock()

            if pcap_path is not None:
                read_pcap(pcap_path, args.bpf, orchestrator.handle_packet)

            orchestrator.finalize()
    except KeyboardInterrupt:
        # Verified empirically (and in Scapy 2.7.0's source): sniff()'s
        # internal _run() catches KeyboardInterrupt itself and only
        # re-raises when chainCC=True, which we never pass -- so both
        # capture_live() and read_pcap() already return normally on
        # Ctrl-C, no exception reaches here. This handler is a safety net
        # for the remaining window where our own code (not inside sniff())
        # is running on the main thread: e.g. the live-clock thread holding
        # `orchestrator._lock` while the main thread is blocked acquiring
        # it, or finalize()/dashboard rendering executing between reads.
        orchestrator.stop_live_clock()
        orchestrator.finalize()
        console.print("[yellow]Interrupted by user; final window flushed. Exiting.[/yellow]")
        exit_code = 130

    return exit_code


def main() -> None:
    args = parse_args()
    sys.exit(run(args))


if __name__ == "__main__":
    main()
