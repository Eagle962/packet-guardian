"""Regression test for the slow-scan blind spot.

Confirmed bug: with SynScanRule's defaults (5s window, 20-SYN threshold),
a scan of 1 port/second is undetectable no matter how long it runs or how
many ports it eventually covers, since no 5s window ever accumulates more
than ~5 SYNs. Verified: 200 sequential-port probes over 200 seconds
produced zero alerts.
"""

from scapy.layers.inet import IP, TCP

from core.rules import Ruleset, SlowSynScanRule, SynScanRule


def make_syn(src: str, dst: str, dport: int) -> object:
    return IP(src=src, dst=dst) / TCP(sport=12345, dport=dport, flags="S")


class TestFastScanStillDetected:
    def test_fast_scan_detected_by_fast_rule(self) -> None:
        rule = SynScanRule(window_seconds=5.0, syn_threshold=20, unique_port_threshold=10)
        alerts = []
        for port in range(1, 26):
            alerts.extend(rule.process_packet(make_syn("10.0.0.66", "10.0.0.1", port), timestamp=port * 0.01))
        assert len(alerts) == 1
        assert alerts[0]["rule_name"] == "SYN_SCAN"


class TestSlowScanBlindSpot:
    def test_slow_scan_undetected_by_fast_rule_alone(self) -> None:
        fast_rule = SynScanRule(window_seconds=5.0, syn_threshold=20, unique_port_threshold=10)
        alerts = []
        for port in range(1, 201):
            alerts.extend(
                fast_rule.process_packet(make_syn("10.0.0.66", "10.0.0.1", port), timestamp=float(port))
            )
        assert alerts == []

    def test_slow_scan_detected_by_slow_rule(self) -> None:
        slow_rule = SlowSynScanRule(window_seconds=300.0, syn_threshold=30, unique_port_threshold=30)
        alerts = []
        # 1 port/second, well under the fast rule's radar, but the slow
        # rule's 300s window eventually sees >=30 unique ports.
        for port in range(1, 201):
            alerts.extend(
                slow_rule.process_packet(make_syn("10.0.0.66", "10.0.0.1", port), timestamp=float(port))
            )
        assert len(alerts) == 1
        alert = alerts[0]
        assert alert["rule_name"] == "SLOW_SYN_SCAN"
        assert alert["source_ip"] == "10.0.0.66"
        assert len(alert["ports"]) >= 30

    def test_slow_rule_does_not_alert_on_normal_intermittent_traffic(self) -> None:
        # A handful of ordinary connections over a long period should not
        # look like a scan to the slow rule either.
        slow_rule = SlowSynScanRule(window_seconds=300.0, syn_threshold=30, unique_port_threshold=30)
        alerts = []
        for i, port in enumerate((80, 443, 80, 443, 22)):
            alerts.extend(
                slow_rule.process_packet(make_syn("10.0.0.5", "10.0.0.1", port), timestamp=i * 30.0)
            )
        assert alerts == []


class TestBothRulesTogetherCoverBothSpeeds:
    def test_ruleset_with_both_variants_catches_fast_and_slow_scans(self) -> None:
        fast_rule = SynScanRule(window_seconds=5.0, syn_threshold=20, unique_port_threshold=10)
        slow_rule = SlowSynScanRule(window_seconds=300.0, syn_threshold=30, unique_port_threshold=30)
        ruleset = Ruleset([fast_rule, slow_rule])

        alerts = []
        # Fast attacker: 25 ports within a fraction of a second.
        for port in range(1, 26):
            alerts.extend(ruleset.process_packet(make_syn("10.0.0.66", "10.0.0.1", port), timestamp=port * 0.01))
        # Slow attacker: 1 port/second for 200 seconds.
        for port in range(1, 201):
            alerts.extend(ruleset.process_packet(make_syn("10.0.0.77", "10.0.0.1", port), timestamp=float(port)))

        rule_names_by_source = {}
        for alert in alerts:
            rule_names_by_source.setdefault(alert["source_ip"], set()).add(alert["rule_name"])

        assert rule_names_by_source["10.0.0.66"] == {"SYN_SCAN"}
        assert rule_names_by_source["10.0.0.77"] == {"SLOW_SYN_SCAN"}
