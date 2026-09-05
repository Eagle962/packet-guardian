"""Class-based, extensible rule engine for network traffic anomaly detection.

Each rule inspects a sliding window of packets (fed incrementally via
``process_packet``) and produces alert dictionaries when its detection
criteria are met. Rules are self-contained and stateful; a :class:`Ruleset`
simply fans packets out to every registered rule and collects alerts.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from typing import Any, Deque, Dict, List, Optional

from scapy.layers.inet import ICMP, IP, TCP


Alert = Dict[str, Any]


class BaseRule(ABC):
    """Base class for all detection rules.

    Subclasses must implement :meth:`process_packet`, which is called once
    per packet and should return a list of alerts (usually empty) triggered
    by that packet's arrival.
    """

    #: Human readable name reported in alerts. Subclasses should override.
    name: str = "base_rule"

    def __init__(self, window_seconds: float = 5.0) -> None:
        self.window_seconds = window_seconds

    @abstractmethod
    def process_packet(self, packet: Any, timestamp: Optional[float] = None) -> List[Alert]:
        """Process a single packet and return any alerts it triggers."""
        raise NotImplementedError

    def reset(self) -> None:
        """Clear any accumulated state. Subclasses may override."""

    def _now(self, timestamp: Optional[float]) -> float:
        return timestamp if timestamp is not None else time.time()


class SynScanRule(BaseRule):
    """Detects SYN scan behavior: many SYNs to many distinct ports.

    A source IP is flagged when, within ``window_seconds``, it sends at
    least ``syn_threshold`` SYN packets (with no ACK flag, to exclude
    normal handshake SYN-ACKs) touching at least ``unique_port_threshold``
    distinct destination ports.
    """

    name = "SYN_SCAN"

    def __init__(
        self,
        window_seconds: float = 5.0,
        syn_threshold: int = 20,
        unique_port_threshold: int = 10,
    ) -> None:
        super().__init__(window_seconds)
        self.syn_threshold = syn_threshold
        self.unique_port_threshold = unique_port_threshold
        # source_ip -> deque of (timestamp, dest_port)
        self._events: Dict[str, Deque] = defaultdict(deque)
        # source_ip -> set of already-alerted signature to avoid duplicate
        # alerts on every subsequent packet within the same window.
        self._alerted_until: Dict[str, float] = {}

    def reset(self) -> None:
        self._events.clear()
        self._alerted_until.clear()

    def process_packet(self, packet: Any, timestamp: Optional[float] = None) -> List[Alert]:
        alerts: List[Alert] = []

        if not packet.haslayer(IP) or not packet.haslayer(TCP):
            return alerts

        tcp_layer = packet[TCP]
        flags = int(tcp_layer.flags)
        syn_flag = flags & 0x02 != 0
        ack_flag = flags & 0x10 != 0

        # Only pure SYN packets (no ACK) count as scan probes; SYN-ACK
        # replies from a normal handshake are excluded.
        if not syn_flag or ack_flag:
            return alerts

        now = self._now(timestamp)
        src_ip = packet[IP].src
        dst_port = int(tcp_layer.dport)

        events = self._events[src_ip]
        events.append((now, dst_port))

        # Drop events outside the sliding window.
        while events and now - events[0][0] > self.window_seconds:
            events.popleft()

        unique_ports = {port for _, port in events}
        if (
            len(events) >= self.syn_threshold
            and len(unique_ports) >= self.unique_port_threshold
        ):
            # Avoid re-alerting on every single packet for the same
            # ongoing scan; only alert once per window of activity.
            last_alert = self._alerted_until.get(src_ip, float("-inf"))
            if now - last_alert >= self.window_seconds:
                self._alerted_until[src_ip] = now
                alerts.append(
                    {
                        "rule_name": self.name,
                        "source_ip": src_ip,
                        "count": len(events),
                        "ports": sorted(unique_ports),
                        "time_window": self.window_seconds,
                        "timestamp": now,
                        "severity": "HIGH",
                    }
                )

        return alerts


class IcmpFloodRule(BaseRule):
    """Detects ICMP flood behavior: many Echo Requests from one source."""

    name = "ICMP_FLOOD"

    def __init__(
        self,
        window_seconds: float = 5.0,
        echo_request_threshold: int = 100,
    ) -> None:
        super().__init__(window_seconds)
        self.echo_request_threshold = echo_request_threshold
        self._events: Dict[str, Deque] = defaultdict(deque)
        self._alerted_until: Dict[str, float] = {}

    def reset(self) -> None:
        self._events.clear()
        self._alerted_until.clear()

    def process_packet(self, packet: Any, timestamp: Optional[float] = None) -> List[Alert]:
        alerts: List[Alert] = []

        if not packet.haslayer(IP) or not packet.haslayer(ICMP):
            return alerts

        icmp_layer = packet[ICMP]
        # ICMP type 8 == Echo Request.
        if int(icmp_layer.type) != 8:
            return alerts

        now = self._now(timestamp)
        src_ip = packet[IP].src

        events = self._events[src_ip]
        events.append(now)

        while events and now - events[0] > self.window_seconds:
            events.popleft()

        if len(events) >= self.echo_request_threshold:
            last_alert = self._alerted_until.get(src_ip, float("-inf"))
            if now - last_alert >= self.window_seconds:
                self._alerted_until[src_ip] = now
                alerts.append(
                    {
                        "rule_name": self.name,
                        "source_ip": src_ip,
                        "count": len(events),
                        "time_window": self.window_seconds,
                        "timestamp": now,
                        "severity": "HIGH",
                    }
                )

        return alerts


class Ruleset:
    """Runs a packet through every registered rule and collects alerts."""

    def __init__(self, rules: Optional[List[BaseRule]] = None) -> None:
        self.rules: List[BaseRule] = rules if rules is not None else []

    def add_rule(self, rule: BaseRule) -> None:
        self.rules.append(rule)

    def process_packet(self, packet: Any, timestamp: Optional[float] = None) -> List[Alert]:
        alerts: List[Alert] = []
        for rule in self.rules:
            alerts.extend(rule.process_packet(packet, timestamp=timestamp))
        return alerts

    def reset(self) -> None:
        for rule in self.rules:
            rule.reset()
