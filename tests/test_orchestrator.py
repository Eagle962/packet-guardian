"""Regression tests for main.Orchestrator's time-driven windowing.

These specifically target the confirmed bug where window emission was
event-driven (only flushed when a new packet happened to arrive) and rate
features were divided by a constant window_seconds regardless of how much
time actually elapsed. See main.py's Orchestrator docstring for the fixed
design.
"""

import time
from typing import Any, Dict, List

import pytest
from scapy.layers.inet import IP, TCP

import main as main_module
from main import Orchestrator
from ui.console import Dashboard


def make_packet(timestamp: float, src: str = "10.0.0.5", dport: int = 80):
    pkt = IP(src=src, dst="10.0.0.1") / TCP(sport=12345, dport=dport, flags="A")
    pkt.time = timestamp
    return pkt


class RecordingDashboard(Dashboard):
    """Captures every stats update instead of rendering, for assertions."""

    def __init__(self) -> None:
        super().__init__()
        self.history: List[Dict[str, Any]] = []

    def update_stats(self, stats: Dict[str, Any]) -> None:
        self.history.append(dict(stats))
        super().update_stats(stats)


class TestSparseTraffic:
    """The reported bug case: 4 packets, 60s apart, window_seconds=5.0.

    Before the fix: 2 windows emitted, both reporting pps=0.40 (packet_count
    divided by the constant window_seconds, with 2 packets accumulated per
    flush because the window only closes on the *second* packet after a big
    gap). After the fix: ~36 windows (180s span / 5s), most of them empty,
    and the aggregate rate across all windows matches the true ~0.022 pps.
    """

    def test_reports_correct_window_count_and_aggregate_rate(self) -> None:
        dashboard = RecordingDashboard()
        orchestrator = Orchestrator(window_seconds=5.0, model_artifact=None, dashboard=dashboard)

        timestamps = [0.0, 60.0, 120.0, 180.0]
        for ts in timestamps:
            orchestrator.handle_packet(make_packet(ts))
        orchestrator.finalize()

        # ~36 windows over the ~180-185s span, not the pre-fix count of 2.
        assert 30 <= len(dashboard.history) <= 40

        total_span = sum(s["window_span"] for s in dashboard.history)
        aggregate_pps = len(timestamps) / total_span
        assert aggregate_pps == pytest.approx(0.022, abs=0.005)

        # Every non-empty window holds exactly one packet in a 5s bucket,
        # i.e. pps=0.2 -- never the pre-fix 0.40 (2 packets / constant 5s).
        non_empty = [s for s in dashboard.history if s["has_traffic"]]
        assert len(non_empty) == 4
        for stats in non_empty:
            assert stats["packets_per_second"] == pytest.approx(0.2, abs=1e-9)
            assert stats["packets_per_second"] != pytest.approx(0.40, abs=1e-9)

    def test_empty_windows_are_marked_and_not_scored_by_ml(self, monkeypatch: pytest.MonkeyPatch) -> None:
        predict_calls: List[Any] = []

        def fake_predict(features, model_artifact):
            predict_calls.append(features)
            return {"anomaly_score": -1.0, "status": "ANOMALY"}

        monkeypatch.setattr(main_module, "predict", fake_predict)

        dashboard = RecordingDashboard()
        # A non-None sentinel artifact: if predict() were ever called for an
        # empty window we'd see it show up in predict_calls regardless of
        # what this object actually contains.
        orchestrator = Orchestrator(window_seconds=5.0, model_artifact=object(), dashboard=dashboard)

        orchestrator.handle_packet(make_packet(0.0))
        orchestrator.handle_packet(make_packet(60.0))
        orchestrator.finalize()

        empty_windows = [s for s in dashboard.history if not s["has_traffic"]]
        non_empty_windows = [s for s in dashboard.history if s["has_traffic"]]

        assert len(empty_windows) > 0
        for stats in empty_windows:
            assert stats["ml_status"] == "NO_TRAFFIC"

        # predict() was called exactly once per non-empty window, never for
        # an empty one.
        assert len(predict_calls) == len(non_empty_windows)


class TestDenseTraffic:
    def test_multiple_full_windows_report_correct_rates(self) -> None:
        dashboard = RecordingDashboard()
        window_seconds = 5.0
        orchestrator = Orchestrator(window_seconds=window_seconds, model_artifact=None, dashboard=dashboard)

        # 100 packets/sec for 12 seconds -> windows [0,5), [5,10), [10,12).
        rate = 100
        duration = 12.0
        count = int(rate * duration)
        for i in range(count):
            orchestrator.handle_packet(make_packet(i / rate))
        orchestrator.finalize()

        assert len(dashboard.history) == 3

        first, second, third = dashboard.history
        assert first["window_span"] == pytest.approx(5.0)
        assert second["window_span"] == pytest.approx(5.0)
        assert first["packets_per_second"] == pytest.approx(rate, rel=0.02)
        assert second["packets_per_second"] == pytest.approx(rate, rel=0.02)

        # Third window is truncated (finalize): real span, not padded to 5s.
        assert third["window_span"] < 5.0
        assert third["window_span"] == pytest.approx(duration - 10.0, abs=0.05)

        total_packets_seen = sum(
            round(s["packets_per_second"] * s["window_span"]) for s in dashboard.history
        )
        assert total_packets_seen == count


class TestEmptyCapture:
    def test_zero_packets_emits_zero_windows(self) -> None:
        dashboard = RecordingDashboard()
        orchestrator = Orchestrator(window_seconds=5.0, model_artifact=None, dashboard=dashboard)
        orchestrator.finalize()
        assert dashboard.history == []
        assert orchestrator.total_packets == 0


class TestSinglePacket:
    def test_single_packet_uses_nominal_window_as_fallback_span(self) -> None:
        dashboard = RecordingDashboard()
        window_seconds = 5.0
        orchestrator = Orchestrator(window_seconds=window_seconds, model_artifact=None, dashboard=dashboard)

        orchestrator.handle_packet(make_packet(100.0))
        orchestrator.finalize()

        assert len(dashboard.history) == 1
        stats = dashboard.history[0]
        assert stats["has_traffic"] is True
        # No second timestamp bounds this window's true duration, so it
        # falls back to the configured window size rather than a
        # divide-by-zero / infinite rate from a zero-length span.
        assert stats["window_span"] == pytest.approx(window_seconds)
        assert stats["packets_per_second"] == pytest.approx(1 / window_seconds)


class TestLiveClock:
    """Live mode must keep emitting windows on wall-clock time even when no
    packets ever arrive -- otherwise the dashboard freezes during quiet
    traffic. Uses a tiny window_seconds so the test runs fast on real time.
    """

    def test_idle_live_capture_still_emits_windows(self) -> None:
        dashboard = RecordingDashboard()
        orchestrator = Orchestrator(window_seconds=0.05, model_artifact=None, dashboard=dashboard)

        orchestrator.start_live_clock()
        try:
            time.sleep(0.35)
        finally:
            orchestrator.stop_live_clock()

        assert len(dashboard.history) >= 3
        assert all(not s["has_traffic"] for s in dashboard.history)
        assert all(s["ml_status"] == "NO_TRAFFIC" for s in dashboard.history)


class TestTruncatedFinalWindow:
    def test_multi_packet_trailing_window_uses_real_elapsed_span(self) -> None:
        dashboard = RecordingDashboard()
        window_seconds = 5.0
        orchestrator = Orchestrator(window_seconds=window_seconds, model_artifact=None, dashboard=dashboard)

        # 3 packets within 2 real seconds -- well short of the 5s nominal
        # window, and with a genuine second timestamp to bound the span.
        for ts in (0.0, 1.0, 2.0):
            orchestrator.handle_packet(make_packet(ts))
        orchestrator.finalize()

        assert len(dashboard.history) == 1
        stats = dashboard.history[0]
        assert stats["window_span"] == pytest.approx(2.0)
        assert stats["window_span"] != pytest.approx(window_seconds)
        assert stats["packets_per_second"] == pytest.approx(3 / 2.0)
