"""Regression tests for IPv6 support in the rule engine and extractor.

Confirmed bug: core/rules.py and core/extractor.py only imported
scapy.layers.inet (IPv4), so all IPv6 traffic was silently ignored -- a
SYN scan or ICMP flood conducted entirely over IPv6 was never detected,
and unique_destination_ip_count / icmp_ratio silently ignored IPv6
addresses and ICMPv6 messages.
"""

from scapy.layers.inet import TCP
from scapy.layers.inet6 import ICMPv6EchoReply, ICMPv6EchoRequest, IPv6

from core.extractor import FEATURE_NAMES, extract_features
from core.rules import IcmpFloodRule, Ruleset, SynScanRule


class TestSynScanRuleIPv6:
    def test_ipv6_syn_scan_is_detected(self) -> None:
        rule = SynScanRule(window_seconds=5.0, syn_threshold=20, unique_port_threshold=10)
        alerts = []
        for port in range(1, 26):
            pkt = IPv6(src="2001:db8::66", dst="2001:db8::1") / TCP(sport=1234, dport=port, flags="S")
            alerts.extend(rule.process_packet(pkt, timestamp=0.0))

        assert len(alerts) == 1
        assert alerts[0]["source_ip"] == "2001:db8::66"
        assert alerts[0]["rule_name"] == "SYN_SCAN"

    def test_ipv6_below_threshold_does_not_alert(self) -> None:
        rule = SynScanRule(window_seconds=5.0, syn_threshold=20, unique_port_threshold=10)
        alerts = []
        for port in range(1, 6):
            pkt = IPv6(src="2001:db8::66", dst="2001:db8::1") / TCP(sport=1234, dport=port, flags="S")
            alerts.extend(rule.process_packet(pkt, timestamp=0.0))
        assert alerts == []

    def test_mixed_ipv4_and_ipv6_scan_sources_tracked_independently(self) -> None:
        rule = SynScanRule(window_seconds=5.0, syn_threshold=20, unique_port_threshold=10)
        from scapy.layers.inet import IP

        alerts = []
        for port in range(1, 26):
            v4 = IP(src="10.0.0.66", dst="10.0.0.1") / TCP(sport=1234, dport=port, flags="S")
            v6 = IPv6(src="2001:db8::66", dst="2001:db8::1") / TCP(sport=1234, dport=port, flags="S")
            alerts.extend(rule.process_packet(v4, timestamp=0.0))
            alerts.extend(rule.process_packet(v6, timestamp=0.0))

        sources_alerted = {a["source_ip"] for a in alerts}
        assert sources_alerted == {"10.0.0.66", "2001:db8::66"}


class TestIcmpFloodRuleIPv6:
    def test_ipv6_icmp_flood_is_detected(self) -> None:
        rule = IcmpFloodRule(window_seconds=5.0, echo_request_threshold=100)
        alerts = []
        for _ in range(150):
            pkt = IPv6(src="2001:db8::77", dst="2001:db8::1") / ICMPv6EchoRequest()
            alerts.extend(rule.process_packet(pkt, timestamp=0.0))

        assert len(alerts) == 1
        assert alerts[0]["source_ip"] == "2001:db8::77"
        assert alerts[0]["rule_name"] == "ICMP_FLOOD"

    def test_ipv6_echo_replies_are_not_counted(self) -> None:
        rule = IcmpFloodRule(window_seconds=5.0, echo_request_threshold=5)
        alerts = []
        for _ in range(10):
            pkt = IPv6(src="2001:db8::77", dst="2001:db8::1") / ICMPv6EchoReply()
            alerts.extend(rule.process_packet(pkt, timestamp=0.0))
        assert alerts == []


class TestRulesetIPv6:
    def test_ruleset_detects_both_v4_and_v6_attacks(self) -> None:
        from scapy.layers.inet import IP

        syn_rule = SynScanRule(window_seconds=5.0, syn_threshold=3, unique_port_threshold=2)
        icmp_rule = IcmpFloodRule(window_seconds=5.0, echo_request_threshold=3)
        ruleset = Ruleset([syn_rule, icmp_rule])

        alerts = []
        for port in (1, 2, 3):
            pkt = IPv6(src="2001:db8::66", dst="2001:db8::1") / TCP(sport=1234, dport=port, flags="S")
            alerts.extend(ruleset.process_packet(pkt, timestamp=0.0))
        for _ in range(3):
            pkt = IP(src="10.0.0.77", dst="10.0.0.1")
            from scapy.layers.inet import ICMP

            pkt = pkt / ICMP(type=8)
            alerts.extend(ruleset.process_packet(pkt, timestamp=0.0))

        rule_names = {a["rule_name"] for a in alerts}
        assert rule_names == {"SYN_SCAN", "ICMP_FLOOD"}


class TestExtractorIPv6:
    def test_ipv6_destination_counted(self) -> None:
        packets = [
            IPv6(src="2001:db8::1", dst="2001:db8::2") / TCP(sport=1, dport=80, flags="S")
            for _ in range(5)
        ]
        features = extract_features(packets, window_seconds=1.0)
        as_dict = dict(zip(FEATURE_NAMES, features))
        assert as_dict["unique_destination_ip_count"] == 1
        assert as_dict["tcp_ratio"] == 1.0

    def test_icmpv6_counts_toward_icmp_ratio(self) -> None:
        packets = [IPv6(src="2001:db8::1", dst="2001:db8::2") / ICMPv6EchoRequest() for _ in range(4)]
        features = extract_features(packets, window_seconds=1.0)
        as_dict = dict(zip(FEATURE_NAMES, features))
        assert as_dict["icmp_ratio"] == 1.0

    def test_mixed_v4_v6_traffic_ratios_sum_correctly(self) -> None:
        from scapy.layers.inet import IP

        packets = [IP(src="10.0.0.1", dst="10.0.0.2") / TCP(sport=1, dport=80, flags="S") for _ in range(3)]
        packets += [IPv6(src="2001:db8::1", dst="2001:db8::2") / ICMPv6EchoRequest() for _ in range(3)]
        features = extract_features(packets, window_seconds=1.0)
        as_dict = dict(zip(FEATURE_NAMES, features))
        assert as_dict["tcp_ratio"] == 0.5
        assert as_dict["icmp_ratio"] == 0.5
        assert as_dict["unique_destination_ip_count"] == 2
