"""Tests for core.rules: SynScanRule, IcmpFloodRule, and Ruleset."""

from scapy.layers.inet import ICMP, IP, TCP

from core.rules import IcmpFloodRule, Ruleset, SynScanRule


def make_syn(src: str, dst: str, dport: int) -> object:
    return IP(src=src, dst=dst) / TCP(sport=12345, dport=dport, flags="S")


def make_synack(src: str, dst: str, dport: int) -> object:
    return IP(src=src, dst=dst) / TCP(sport=dport, dport=12345, flags="SA")


def make_icmp_echo_request(src: str, dst: str) -> object:
    return IP(src=src, dst=dst) / ICMP(type=8)


def make_icmp_echo_reply(src: str, dst: str) -> object:
    return IP(src=src, dst=dst) / ICMP(type=0)


class TestSynScanRule:
    def test_normal_handshake_traffic_does_not_alert(self) -> None:
        rule = SynScanRule(window_seconds=5.0, syn_threshold=20, unique_port_threshold=10)
        alerts = []
        t = 0.0
        for i in range(5):
            alerts.extend(rule.process_packet(make_syn("10.0.0.1", "10.0.0.2", 80), timestamp=t))
            alerts.extend(rule.process_packet(make_synack("10.0.0.2", "10.0.0.1", 80), timestamp=t))
            t += 0.1
        assert alerts == []

    def test_syn_scan_triggers_alert(self) -> None:
        rule = SynScanRule(window_seconds=5.0, syn_threshold=20, unique_port_threshold=10)
        alerts = []
        t = 0.0
        for port in range(1, 26):
            alerts.extend(
                rule.process_packet(make_syn("10.0.0.5", "10.0.0.9", port), timestamp=t)
            )
            t += 0.01
        assert len(alerts) == 1
        alert = alerts[0]
        assert alert["rule_name"] == "SYN_SCAN"
        assert alert["source_ip"] == "10.0.0.5"
        assert alert["count"] >= 20
        assert len(alert["ports"]) >= 10
        assert alert["severity"] == "HIGH"
        assert "timestamp" in alert
        assert alert["time_window"] == 5.0

    def test_syn_scan_below_threshold_does_not_alert(self) -> None:
        rule = SynScanRule(window_seconds=5.0, syn_threshold=20, unique_port_threshold=10)
        alerts = []
        t = 0.0
        for port in range(1, 6):
            alerts.extend(
                rule.process_packet(make_syn("10.0.0.5", "10.0.0.9", port), timestamp=t)
            )
            t += 0.01
        assert alerts == []

    def test_syn_scan_events_outside_window_are_dropped(self) -> None:
        rule = SynScanRule(window_seconds=5.0, syn_threshold=20, unique_port_threshold=10)
        alerts = []
        # First burst of 15 SYNs, well before the window closes.
        for port in range(1, 16):
            alerts.extend(
                rule.process_packet(make_syn("10.0.0.5", "10.0.0.9", port), timestamp=0.0)
            )
        # Second burst, 10 seconds later (outside the 5s window), does not
        # accumulate with the first burst.
        for port in range(16, 21):
            alerts.extend(
                rule.process_packet(make_syn("10.0.0.5", "10.0.0.9", port), timestamp=10.0)
            )
        assert alerts == []

    def test_non_syn_packets_ignored(self) -> None:
        rule = SynScanRule(window_seconds=5.0, syn_threshold=1, unique_port_threshold=1)
        alerts = rule.process_packet(make_synack("10.0.0.2", "10.0.0.1", 80), timestamp=0.0)
        assert alerts == []

    def test_configurable_thresholds(self) -> None:
        rule = SynScanRule(window_seconds=2.0, syn_threshold=3, unique_port_threshold=2)
        alerts = []
        for port in (1, 2, 3):
            alerts.extend(
                rule.process_packet(make_syn("10.0.0.5", "10.0.0.9", port), timestamp=0.0)
            )
        assert len(alerts) == 1


class TestIcmpFloodRule:
    def test_normal_ping_traffic_does_not_alert(self) -> None:
        rule = IcmpFloodRule(window_seconds=5.0, echo_request_threshold=100)
        alerts = []
        t = 0.0
        for _ in range(4):
            alerts.extend(
                rule.process_packet(make_icmp_echo_request("10.0.0.1", "10.0.0.2"), timestamp=t)
            )
            alerts.extend(
                rule.process_packet(make_icmp_echo_reply("10.0.0.2", "10.0.0.1"), timestamp=t)
            )
            t += 1.0
        assert alerts == []

    def test_icmp_flood_triggers_alert(self) -> None:
        rule = IcmpFloodRule(window_seconds=5.0, echo_request_threshold=100)
        alerts = []
        t = 0.0
        for _ in range(150):
            alerts.extend(
                rule.process_packet(make_icmp_echo_request("10.0.0.7", "10.0.0.9"), timestamp=t)
            )
            t += 0.01
        assert len(alerts) == 1
        alert = alerts[0]
        assert alert["rule_name"] == "ICMP_FLOOD"
        assert alert["source_ip"] == "10.0.0.7"
        assert alert["count"] >= 100
        assert alert["severity"] == "HIGH"
        assert "timestamp" in alert
        assert alert["time_window"] == 5.0

    def test_icmp_flood_below_threshold_does_not_alert(self) -> None:
        rule = IcmpFloodRule(window_seconds=5.0, echo_request_threshold=100)
        alerts = []
        t = 0.0
        for _ in range(10):
            alerts.extend(
                rule.process_packet(make_icmp_echo_request("10.0.0.7", "10.0.0.9"), timestamp=t)
            )
            t += 0.01
        assert alerts == []

    def test_echo_replies_are_not_counted(self) -> None:
        rule = IcmpFloodRule(window_seconds=5.0, echo_request_threshold=5)
        alerts = []
        for _ in range(10):
            alerts.extend(
                rule.process_packet(make_icmp_echo_reply("10.0.0.7", "10.0.0.9"), timestamp=0.0)
            )
        assert alerts == []

    def test_configurable_threshold(self) -> None:
        rule = IcmpFloodRule(window_seconds=5.0, echo_request_threshold=3)
        alerts = []
        for _ in range(3):
            alerts.extend(
                rule.process_packet(make_icmp_echo_request("10.0.0.7", "10.0.0.9"), timestamp=0.0)
            )
        assert len(alerts) == 1


class TestRuleset:
    def test_ruleset_aggregates_alerts_from_multiple_rules(self) -> None:
        syn_rule = SynScanRule(window_seconds=5.0, syn_threshold=3, unique_port_threshold=2)
        icmp_rule = IcmpFloodRule(window_seconds=5.0, echo_request_threshold=3)
        ruleset = Ruleset([syn_rule, icmp_rule])

        alerts = []
        for port in (1, 2, 3):
            alerts.extend(
                ruleset.process_packet(make_syn("10.0.0.5", "10.0.0.9", port), timestamp=0.0)
            )
        for _ in range(3):
            alerts.extend(
                ruleset.process_packet(
                    make_icmp_echo_request("10.0.0.7", "10.0.0.9"), timestamp=0.0
                )
            )

        rule_names = {alert["rule_name"] for alert in alerts}
        assert rule_names == {"SYN_SCAN", "ICMP_FLOOD"}

    def test_ruleset_reset_clears_state(self) -> None:
        syn_rule = SynScanRule(window_seconds=5.0, syn_threshold=3, unique_port_threshold=2)
        ruleset = Ruleset([syn_rule])
        for port in (1, 2, 3):
            ruleset.process_packet(make_syn("10.0.0.5", "10.0.0.9", port), timestamp=0.0)
        ruleset.reset()
        alerts = ruleset.process_packet(make_syn("10.0.0.5", "10.0.0.9", 1), timestamp=0.0)
        assert alerts == []
