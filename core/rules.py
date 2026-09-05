"""Class-based, extensible rule engine for network traffic anomaly detection.

Each rule inspects a sliding window of packets (fed incrementally via
``process_packet``) and produces alert dictionaries when its detection
criteria are met. Rules are self-contained and stateful; a :class:`Ruleset`
simply fans packets out to every registered rule and collects alerts.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections import OrderedDict, defaultdict, deque
from typing import Any, Deque, Dict, List, Optional

from scapy.layers.inet import ICMP, IP, TCP


Alert = Dict[str, Any]

#: Default cap on the number of distinct source IPs a rule will track at
#: once. Without a cap, a spoofed-source flood (packets from thousands of
#: forged IPs, each seen once) grows a rule's per-source state forever,
#: since a source's own deque is only ever pruned when that same source
#: sends another packet -- this is a memory-exhaustion vector against the
#: monitor itself.
DEFAULT_MAX_TRACKED_SOURCES = 10_000


class BaseRule(ABC):
    """Base class for all detection rules.

    Subclasses must implement :meth:`process_packet`, which is called once
    per packet and should return a list of alerts (usually empty) triggered
    by that packet's arrival.

    Provides bounded, TTL-based tracking of per-source-IP state via
    :meth:`_touch_source`, so subclasses don't each have to reimplement
    memory-bounding logic.
    """

    #: Human readable name reported in alerts. Subclasses should override.
    name: str = "base_rule"

    def __init__(
        self,
        window_seconds: float = 5.0,
        max_tracked_sources: int = DEFAULT_MAX_TRACKED_SOURCES,
        sweep_interval: Optional[float] = None,
    ) -> None:
        self.window_seconds = window_seconds
        self.max_tracked_sources = max_tracked_sources
        # How often (in the same time units as packet timestamps) to sweep
        # for sources that have gone quiet. Defaults to the detection
        # window itself: any source silent for longer than window_seconds
        # can no longer contribute to a still-open detection window, so its
        # state is safe to drop.
        self.sweep_interval = window_seconds if sweep_interval is None else sweep_interval
        # source_ip -> last time it was seen, in insertion/access order so
        # the least-recently-touched source is always at the front (LRU).
        self._last_seen: "OrderedDict[str, float]" = OrderedDict()
        self._last_sweep_at: Optional[float] = None

    @abstractmethod
    def process_packet(self, packet: Any, timestamp: Optional[float] = None) -> List[Alert]:
        """Process a single packet and return any alerts it triggers."""
        raise NotImplementedError

    def reset(self) -> None:
        """Clear any accumulated state. Subclasses should call super()."""
        self._last_seen.clear()
        self._last_sweep_at = None

    def _now(self, timestamp: Optional[float]) -> float:
        return timestamp if timestamp is not None else time.time()

    def _touch_source(self, src_ip: str, now: float, tracked_dicts: List[Dict[str, Any]]) -> None:
        """Record that ``src_ip`` was active at ``now``, and bound the total
        memory used to track sources via two mechanisms:

        1. A periodic TTL sweep (every ``sweep_interval``): any source not
           seen for longer than ``window_seconds`` is dropped, since it can
           no longer affect a sliding window of that size. This is the
           primary defense -- it reclaims state from sources that simply
           stop sending, which per-packet pruning of a single source's own
           deque can never do on its own.
        2. An LRU cap at ``max_tracked_sources``: a hard ceiling on worst
           case memory even if a flood arrives faster than the sweep
           interval, evicting the least-recently-active source first.

        ``tracked_dicts`` are the calling rule's own per-source dicts (e.g.
        its event deque and last-alert-time maps) which must be evicted in
        lockstep with ``_last_seen`` so a dropped source's state doesn't
        outlive its eviction here.
        """
        self._last_seen[src_ip] = now
        self._last_seen.move_to_end(src_ip)

        if self._last_sweep_at is None:
            self._last_sweep_at = now
        elif now - self._last_sweep_at >= self.sweep_interval:
            self._last_sweep_at = now
            stale_ips = [
                ip
                for ip, last_seen in self._last_seen.items()
                if now - last_seen > self.window_seconds
            ]
            for ip in stale_ips:
                del self._last_seen[ip]
                for tracked in tracked_dicts:
                    tracked.pop(ip, None)

        while len(self._last_seen) > self.max_tracked_sources:
            oldest_ip, _ = self._last_seen.popitem(last=False)
            for tracked in tracked_dicts:
                tracked.pop(oldest_ip, None)


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
        max_tracked_sources: int = DEFAULT_MAX_TRACKED_SOURCES,
        sweep_interval: Optional[float] = None,
    ) -> None:
        super().__init__(window_seconds, max_tracked_sources, sweep_interval)
        self.syn_threshold = syn_threshold
        self.unique_port_threshold = unique_port_threshold
        # source_ip -> deque of (timestamp, dest_port)
        self._events: Dict[str, Deque] = defaultdict(deque)
        # source_ip -> set of already-alerted signature to avoid duplicate
        # alerts on every subsequent packet within the same window.
        self._alerted_until: Dict[str, float] = {}

    def reset(self) -> None:
        super().reset()
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

        self._touch_source(src_ip, now, [self._events, self._alerted_until])

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
        max_tracked_sources: int = DEFAULT_MAX_TRACKED_SOURCES,
        sweep_interval: Optional[float] = None,
    ) -> None:
        super().__init__(window_seconds, max_tracked_sources, sweep_interval)
        self.echo_request_threshold = echo_request_threshold
        self._events: Dict[str, Deque] = defaultdict(deque)
        self._alerted_until: Dict[str, float] = {}

    def reset(self) -> None:
        super().reset()
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

        self._touch_source(src_ip, now, [self._events, self._alerted_until])

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
