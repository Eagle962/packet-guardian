"""Rich-based terminal dashboard for packet-guardian.

This module is purely responsible for rendering. It holds no detection or
sniffing logic: callers push traffic/ML statistics and rule alerts in, and
this module turns that state into a Rich renderable.
"""

from __future__ import annotations

import datetime
from collections import deque
from typing import Any, Deque, Dict, Optional

from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

DEFAULT_STATS: Dict[str, Any] = {
    "total_packets": 0,
    "packets_per_second": 0.0,
    "bytes_per_second": 0.0,
    "average_packet_size": 0.0,
    "tcp_ratio": 0.0,
    "udp_ratio": 0.0,
    "icmp_ratio": 0.0,
    "anomaly_score": 0.0,
    "ml_status": "NORMAL",
}


class Dashboard:
    """Holds current traffic/ML stats and recent alerts, and renders them."""

    def __init__(self, max_alerts: int = 15) -> None:
        self.stats: Dict[str, Any] = dict(DEFAULT_STATS)
        self.alerts: Deque[Dict[str, Any]] = deque(maxlen=max_alerts)
        self._live: Optional[Any] = None

    def attach_live(self, live: Any) -> None:
        """Attach a ``rich.live.Live`` instance to auto-refresh on updates."""
        self._live = live

    def update_stats(self, stats: Dict[str, Any]) -> None:
        self.stats = stats
        self._refresh()

    def add_alert(self, alert: Dict[str, Any]) -> None:
        self.alerts.appendleft(alert)
        self._refresh()

    def _refresh(self) -> None:
        if self._live is not None:
            self._live.update(self.render())

    def _render_stats_panel(self) -> Panel:
        table = Table.grid(padding=(0, 2))
        table.add_column(justify="left", style="bold cyan")
        table.add_column(justify="right")

        stats = self.stats
        table.add_row("Total Packets", str(stats.get("total_packets", 0)))
        table.add_row("Packets/sec", f"{stats.get('packets_per_second', 0.0):.2f}")
        table.add_row("Avg Packet Size", f"{stats.get('average_packet_size', 0.0):.2f} bytes")
        table.add_row(
            "Protocol Ratios (TCP/UDP/ICMP)",
            f"{stats.get('tcp_ratio', 0.0):.2f} / "
            f"{stats.get('udp_ratio', 0.0):.2f} / "
            f"{stats.get('icmp_ratio', 0.0):.2f}",
        )
        table.add_row("SYN Ratio (of TCP)", f"{stats.get('syn_ratio', 0.0):.2f}")
        table.add_row("Unique Source IPs", str(stats.get("unique_source_ip_count", 0)))

        ml_status = stats.get("ml_status", "NORMAL")
        if ml_status == "ANOMALY":
            status_style = "bold red"
        elif ml_status in ("NO_TRAFFIC", "UNKNOWN"):
            status_style = "dim"
        else:
            status_style = "bold green"
        # `:.4g` rather than a fixed `:.4f`: some model families (e.g.
        # LocalOutlierFactor) can produce scores with a very wide dynamic
        # range (a "clearly anomalous" window's ratio-based score isn't
        # bounded the way e.g. IsolationForest's is), and `.4f` would
        # render those as an unreadable wall of digits.
        table.add_row("ML Anomaly Score", f"{stats.get('anomaly_score', 0.0):.4g}")
        table.add_row("ML Status", Text(str(ml_status), style=status_style))

        return Panel(table, title="Traffic / ML Stats", border_style="cyan")

    def _render_alerts_panel(self) -> Panel:
        table = Table(expand=True, show_lines=False)
        table.add_column("Time", style="bold red", no_wrap=True)
        table.add_column("Rule", style="bold red", no_wrap=True)
        table.add_column("Source IP", style="bold red", no_wrap=True)
        table.add_column("Details", style="bold red")
        table.add_column("Severity", style="bold red", no_wrap=True)

        for alert in self.alerts:
            timestamp = alert.get("timestamp")
            if isinstance(timestamp, (int, float)):
                time_str = datetime.datetime.fromtimestamp(timestamp).strftime("%H:%M:%S")
            else:
                time_str = str(timestamp)

            if "ports" in alert:
                details = f"count={alert.get('count')} ports={len(alert.get('ports', []))}"
            else:
                details = f"count={alert.get('count')}"

            table.add_row(
                time_str,
                str(alert.get("rule_name", "")),
                str(alert.get("source_ip", "")),
                details,
                str(alert.get("severity", "")),
            )

        return Panel(table, title="Rule Engine Alerts", border_style="red")

    def render(self) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(self._render_stats_panel(), name="stats"),
            Layout(self._render_alerts_panel(), name="alerts"),
        )
        return layout
