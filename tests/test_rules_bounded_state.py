"""Regression test for unbounded per-source state growth in the rule
engine.

Confirmed bug: SynScanRule._events (a defaultdict(deque)) is only pruned
for a given source IP when that same IP sends another packet, so a
spoofed-source flood (many distinct source IPs, each seen once or a few
times) grows the tracked-source dict forever -- a memory-exhaustion vector
against the monitor itself. Verified against the pre-fix code: 20,000
packets from random source IPs left all 20,000 tracked forever.
"""

import random

from scapy.layers.inet import ICMP, IP, TCP

from core.rules import IcmpFloodRule, SynScanRule


def random_ip(rng: random.Random) -> str:
    return f"{rng.randint(1, 223)}.{rng.randint(0, 255)}.{rng.randint(0, 255)}.{rng.randint(0, 255)}"


class TestSynScanRuleBoundedState:
    def test_tracked_sources_stay_bounded_under_spoofed_flood(self) -> None:
        max_tracked = 500
        rule = SynScanRule(window_seconds=5.0, max_tracked_sources=max_tracked)
        rng = random.Random(42)

        for i in range(100_000):
            src_ip = random_ip(rng)
            pkt = IP(src=src_ip, dst="10.0.0.1") / TCP(
                sport=1234, dport=rng.randint(1, 65535), flags="S"
            )
            # Timestamps advance slowly enough that the TTL sweep also gets
            # exercised (not just the hard LRU cap), since packets are
            # spread across many multiples of window_seconds.
            rule.process_packet(pkt, timestamp=i * 0.001)

        assert len(rule._last_seen) <= max_tracked
        assert len(rule._events) <= max_tracked
        assert len(rule._alerted_until) <= max_tracked

    def test_reset_clears_bounded_state(self) -> None:
        rule = SynScanRule(window_seconds=5.0, max_tracked_sources=10)
        for i in range(5):
            pkt = IP(src=f"10.0.0.{i}", dst="10.0.0.1") / TCP(sport=1234, dport=80, flags="S")
            rule.process_packet(pkt, timestamp=float(i))
        assert len(rule._last_seen) == 5
        rule.reset()
        assert len(rule._last_seen) == 0
        assert len(rule._events) == 0


class TestIcmpFloodRuleBoundedState:
    def test_tracked_sources_stay_bounded_under_spoofed_flood(self) -> None:
        max_tracked = 500
        rule = IcmpFloodRule(window_seconds=5.0, max_tracked_sources=max_tracked)
        rng = random.Random(7)

        for i in range(100_000):
            src_ip = random_ip(rng)
            pkt = IP(src=src_ip, dst="10.0.0.1") / ICMP(type=8)
            rule.process_packet(pkt, timestamp=i * 0.001)

        assert len(rule._last_seen) <= max_tracked
        assert len(rule._events) <= max_tracked
        assert len(rule._alerted_until) <= max_tracked


class TestBoundedStateDoesNotBreakDetection:
    def test_scan_still_detected_with_bounded_state(self) -> None:
        # A generous cap that easily accommodates the scanning source plus
        # a bit of unrelated background noise, to confirm bounding memory
        # doesn't interfere with genuine detections.
        rule = SynScanRule(window_seconds=5.0, max_tracked_sources=50)
        rng = random.Random(1)

        # Background noise from a handful of other sources.
        for i in range(10):
            pkt = IP(src=random_ip(rng), dst="10.0.0.1") / TCP(sport=1234, dport=80, flags="S")
            rule.process_packet(pkt, timestamp=0.0)

        alerts = []
        for port in range(1, 26):
            pkt = IP(src="10.0.0.66", dst="10.0.0.1") / TCP(sport=1234, dport=port, flags="S")
            alerts.extend(rule.process_packet(pkt, timestamp=0.0))

        assert len(alerts) == 1
        assert alerts[0]["source_ip"] == "10.0.0.66"
