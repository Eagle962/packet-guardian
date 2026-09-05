"""Synthetic, labelled traffic generators for training and evaluating the
ML anomaly detector.

Every profile function returns a list of real Scapy packets (with `.time`
set) for one time window -- never a hand-written feature vector. This
matters for train/serve parity: the *only* way a window's features are
computed anywhere in this project is core.extractor.extract_features(),
so training data must be packets that go through that same function, not
a shortcut that fabricates plausible-looking numbers directly (which is
exactly what the old ml_engine/train.py did, and why its model had never
actually been evaluated against real attack signal).

Each profile is randomised (rates, sizes, host counts, jitter) via a
caller-supplied random.Random, so many independent instances can be drawn
from the same profile rather than one fixed canned sample.

A handful of `pkt.time = t` assignments below carry a `# type: ignore
[attr-defined]`: Scapy's `Packet.time` is a dynamic field set via its own
descriptor magic, which mypy cannot see through for a freshly-constructed
`IP(...) / TCP(...)`-style packet (as opposed to one that has passed
through a helper returning a looser type).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from scapy.layers.inet import ICMP, IP, TCP, UDP
from scapy.packet import Raw
from scapy.utils import wrpcap

from core.extractor import extract_features

ProfileFn = Callable[[random.Random, float], List[Any]]


def _rand_ip(rng: random.Random, prefix: str = "10.0.") -> str:
    parts = prefix.count(".")
    remaining = 4 - parts
    octets = [str(rng.randint(1, 254)) for _ in range(remaining)]
    return prefix + ".".join(octets)


def _timestamps(rng: random.Random, count: int, window_seconds: float, jitter_frac: float = 0.15) -> List[float]:
    """``count`` timestamps spread roughly evenly across [0, window_seconds),
    each perturbed by up to ``jitter_frac`` of the average spacing, so
    packets don't land on an artificially perfect grid."""
    if count <= 0:
        return []
    if count == 1:
        return [rng.uniform(0, window_seconds)]

    step = window_seconds / count
    timestamps = []
    t = 0.0
    for _ in range(count):
        jitter = rng.uniform(-jitter_frac, jitter_frac) * step
        timestamps.append(min(max(0.0, t + jitter), window_seconds))
        t += step
    timestamps.sort()
    return timestamps


def _sized_tcp(src: str, dst: str, sport: int, dport: int, flags: str, size: int):
    pkt = IP(src=src, dst=dst) / TCP(sport=sport, dport=dport, flags=flags)
    pad = size - len(pkt)
    if pad > 0:
        pkt = pkt / Raw(load=b"x" * pad)
    return pkt


def _sized_udp(src: str, dst: str, sport: int, dport: int, size: int):
    pkt = IP(src=src, dst=dst) / UDP(sport=sport, dport=dport)
    pad = size - len(pkt)
    if pad > 0:
        pkt = pkt / Raw(load=b"x" * pad)
    return pkt


# ---------------------------------------------------------------------------
# Normal traffic profiles
# ---------------------------------------------------------------------------


def profile_idle(rng: random.Random, window_seconds: float) -> List[Any]:
    """Near-idle link: at most a couple of stray packets."""
    src = _rand_ip(rng)
    dst = "10.0.0.1"
    count = rng.choice([0, 0, 1, 1, 2])
    packets = []
    for t in _timestamps(rng, count, window_seconds):
        pkt = _sized_tcp(src, dst, rng.randint(1024, 65535), 443, "A", rng.randint(40, 100))
        pkt.time = t
        packets.append(pkt)
    return packets


def profile_web_browsing(rng: random.Random, window_seconds: float, src: Optional[str] = None) -> List[Any]:
    """Bursty TCP to a handful of destinations, mixed request/response sizes."""
    src = src or _rand_ip(rng)
    dsts = [_rand_ip(rng, "93.184.") for _ in range(rng.randint(2, 5))]
    pps = rng.uniform(8, 35)
    count = max(1, int(pps * window_seconds))
    packets = []
    for t in _timestamps(rng, count, window_seconds):
        dst = rng.choice(dsts)
        port = rng.choice([80, 443, 443, 443])
        flags = rng.choice(["S", "A", "PA", "A", "FA"])
        size = rng.randint(60, 1400)
        pkt = _sized_tcp(src, dst, rng.randint(1024, 65535), port, flags, size)
        pkt.time = t
        packets.append(pkt)
    return packets


def profile_video_streaming(rng: random.Random, window_seconds: float) -> List[Any]:
    """High rate, large near-uniform packets, a single destination."""
    src = _rand_ip(rng)
    dst = _rand_ip(rng, "203.0.")
    port = rng.choice([443, 80])
    pps = rng.uniform(40, 100)
    count = max(1, int(pps * window_seconds))
    packets = []
    for t in _timestamps(rng, count, window_seconds, jitter_frac=0.05):
        size = rng.randint(1200, 1460)
        pkt = _sized_tcp(src, dst, rng.randint(1024, 65535), port, "A", size)
        pkt.time = t
        packets.append(pkt)
    return packets


def profile_bulk_file_transfer(rng: random.Random, window_seconds: float, src: Optional[str] = None) -> List[Any]:
    """Sustained large-packet TCP transfer to one or two destinations."""
    src = src or _rand_ip(rng)
    dsts = [_rand_ip(rng, "198.51.") for _ in range(rng.randint(1, 2))]
    port = rng.choice([443, 22, 80])
    pps = rng.uniform(50, 120)
    count = max(1, int(pps * window_seconds))
    packets = []
    for t in _timestamps(rng, count, window_seconds, jitter_frac=0.05):
        dst = rng.choice(dsts)
        size = rng.randint(1000, 1460)
        pkt = _sized_tcp(src, dst, rng.randint(1024, 65535), port, "PA", size)
        pkt.time = t
        packets.append(pkt)
    return packets


def profile_dns_heavy(rng: random.Random, window_seconds: float, src: Optional[str] = None) -> List[Any]:
    """Many small UDP queries to one or a few resolvers."""
    src = src or _rand_ip(rng)
    resolvers = [_rand_ip(rng, "8.8.") for _ in range(rng.randint(1, 3))]
    pps = rng.uniform(15, 45)
    count = max(1, int(pps * window_seconds))
    packets = []
    for t in _timestamps(rng, count, window_seconds):
        dst = rng.choice(resolvers)
        size = rng.randint(48, 120)
        pkt = _sized_udp(src, dst, rng.randint(1024, 65535), 53, size)
        pkt.time = t
        packets.append(pkt)
    return packets


def profile_voip_udp(rng: random.Random, window_seconds: float) -> List[Any]:
    """Small, steady-rate UDP (RTP-like) stream to one destination/port."""
    src = _rand_ip(rng)
    dst = _rand_ip(rng, "172.16.")
    port = rng.randint(10000, 20000)
    pps = rng.uniform(25, 55)
    count = max(1, int(pps * window_seconds))
    packets = []
    for t in _timestamps(rng, count, window_seconds, jitter_frac=0.05):
        size = rng.randint(120, 220)
        pkt = _sized_udp(src, dst, rng.randint(1024, 65535), port, size)
        pkt.time = t
        packets.append(pkt)
    return packets


def profile_mixed_office(rng: random.Random, window_seconds: float) -> List[Any]:
    """A blend of light web browsing, DNS, and a little bulk traffic --
    representative of an ordinary workstation's background chatter.

    All sub-traffic shares a single source IP: it's one workstation doing
    several things at once, not several different hosts. An earlier
    version let each sub-profile pick its own random source IP, which
    inflated unique_source_ip_count to 2-3 for a single-host window --
    a real source of the false positives this profile once had (see
    EVALUATION.md's iteration history).
    """
    src = _rand_ip(rng)
    packets = []
    packets += profile_web_browsing(rng, window_seconds, src=src)
    packets += profile_dns_heavy(rng, window_seconds, src=src)
    if rng.random() < 0.4:
        packets += profile_bulk_file_transfer(rng, window_seconds, src=src)
    return packets


def profile_normal_with_bursts(rng: random.Random, window_seconds: float) -> List[Any]:
    """Low baseline traffic plus a short legitimate burst (e.g. a page
    load or a quick download) -- still normal, but not perfectly steady."""
    src = _rand_ip(rng)
    dst = _rand_ip(rng, "93.184.")
    baseline_pps = rng.uniform(2, 6)
    baseline_count = max(1, int(baseline_pps * window_seconds))
    packets = []
    for t in _timestamps(rng, baseline_count, window_seconds):
        pkt = _sized_tcp(src, dst, rng.randint(1024, 65535), 443, "A", rng.randint(60, 300))
        pkt.time = t
        packets.append(pkt)

    burst_span = min(window_seconds, rng.uniform(0.3, 1.0))
    burst_start = rng.uniform(0, max(0.0, window_seconds - burst_span))
    burst_count = rng.randint(15, 40)
    for i in range(burst_count):
        t = burst_start + (i / burst_count) * burst_span
        pkt = _sized_tcp(src, dst, rng.randint(1024, 65535), 443, "A", rng.randint(200, 1400))
        pkt.time = t
        packets.append(pkt)
    return packets


# ---------------------------------------------------------------------------
# Attack traffic profiles
# ---------------------------------------------------------------------------


def profile_fast_syn_scan(rng: random.Random, window_seconds: float) -> List[Any]:
    """Single source, many destination ports on one target, all within a
    short burst -- the classic fast SYN scan the rule engine already
    catches."""
    src = _rand_ip(rng, "192.168.")
    dst = "10.0.0.1"
    n_ports = rng.randint(20, 80)
    ports = rng.sample(range(1, 65535), min(n_ports, 65534))
    burst_span = min(window_seconds, rng.uniform(0.2, 1.5))
    start = rng.uniform(0, max(0.0, window_seconds - burst_span))
    packets = []
    for i, port in enumerate(ports):
        t = start + (i / len(ports)) * burst_span
        pkt = IP(src=src, dst=dst) / TCP(sport=rng.randint(1024, 65535), dport=port, flags="S")
        pkt.time = t  # type: ignore[attr-defined]
        packets.append(pkt)
    return packets


def profile_slow_syn_scan(rng: random.Random, window_seconds: float) -> List[Any]:
    """One 5s-window snapshot of an ongoing low-and-slow scan (~1
    port/several seconds). Deliberately produces almost no signal within
    a single short window -- see EVALUATION.md for why this is expected
    to be very hard for a per-window ML detector, which is precisely why
    the rule engine has a dedicated long-window rule for this case
    instead."""
    src = _rand_ip(rng, "192.168.")
    dst = "10.0.0.1"
    count = rng.choice([0, 0, 1, 1, 1])
    packets = []
    for t in _timestamps(rng, count, window_seconds):
        port = rng.randint(1, 65535)
        pkt = IP(src=src, dst=dst) / TCP(sport=rng.randint(1024, 65535), dport=port, flags="S")
        pkt.time = t  # type: ignore[attr-defined]
        packets.append(pkt)
    return packets


def profile_port_sweep(rng: random.Random, window_seconds: float) -> List[Any]:
    """Single source probing the same port(s) across many destination
    hosts -- host discovery, as opposed to a single-host port scan."""
    src = _rand_ip(rng, "192.168.")
    n_hosts = rng.randint(50, 250)
    base = f"10.0.{rng.randint(0, 255)}."
    hosts = [f"{base}{(i % 254) + 1}" for i in range(n_hosts)]
    port = rng.choice([22, 80, 443, 3389])
    burst_span = min(window_seconds, rng.uniform(0.5, 3.0))
    start = rng.uniform(0, max(0.0, window_seconds - burst_span))
    packets = []
    for i, host in enumerate(hosts):
        t = start + (i / len(hosts)) * burst_span
        pkt = IP(src=src, dst=host) / TCP(sport=rng.randint(1024, 65535), dport=port, flags="S")
        pkt.time = t  # type: ignore[attr-defined]
        packets.append(pkt)
    return packets


def profile_icmp_flood(rng: random.Random, window_seconds: float) -> List[Any]:
    src = _rand_ip(rng, "192.168.")
    dst = "10.0.0.1"
    pps = rng.uniform(80, 220)
    count = max(1, int(pps * window_seconds))
    packets = []
    for t in _timestamps(rng, count, window_seconds, jitter_frac=0.02):
        pkt = IP(src=src, dst=dst) / ICMP(type=8)
        pkt.time = t  # type: ignore[attr-defined]
        packets.append(pkt)
    return packets


def profile_udp_flood(rng: random.Random, window_seconds: float) -> List[Any]:
    """High-rate UDP to one target, source address varying per packet
    (spoofing is common in real UDP floods)."""
    dst = "10.0.0.1"
    port = rng.choice([53, 123, 1900, 80])
    pps = rng.uniform(80, 220)
    count = max(1, int(pps * window_seconds))
    packets = []
    for t in _timestamps(rng, count, window_seconds, jitter_frac=0.02):
        src = _rand_ip(rng, "192.168.")
        size = rng.randint(20, 200)
        pkt = _sized_udp(src, dst, rng.randint(1024, 65535), port, size)
        pkt.time = t
        packets.append(pkt)
    return packets


def profile_syn_flood_single_port(rng: random.Random, window_seconds: float) -> List[Any]:
    """Many SYNs to a *single* port (not spread across ports, unlike a
    scan), source varying per packet."""
    dst = "10.0.0.1"
    port = rng.choice([80, 443])
    pps = rng.uniform(80, 220)
    count = max(1, int(pps * window_seconds))
    packets = []
    for t in _timestamps(rng, count, window_seconds, jitter_frac=0.02):
        src = _rand_ip(rng, "192.168.")
        pkt = IP(src=src, dst=dst) / TCP(sport=rng.randint(1024, 65535), dport=port, flags="S")
        pkt.time = t  # type: ignore[attr-defined]
        packets.append(pkt)
    return packets


NORMAL_PROFILES: Dict[str, ProfileFn] = {
    "idle": profile_idle,
    "web_browsing": profile_web_browsing,
    "video_streaming": profile_video_streaming,
    "bulk_file_transfer": profile_bulk_file_transfer,
    "dns_heavy": profile_dns_heavy,
    "voip_udp": profile_voip_udp,
    "mixed_office": profile_mixed_office,
    "normal_with_bursts": profile_normal_with_bursts,
}

ATTACK_PROFILES: Dict[str, ProfileFn] = {
    "fast_syn_scan": profile_fast_syn_scan,
    "slow_syn_scan": profile_slow_syn_scan,
    "port_sweep": profile_port_sweep,
    "icmp_flood": profile_icmp_flood,
    "udp_flood": profile_udp_flood,
    "syn_flood_single_port": profile_syn_flood_single_port,
}

ALL_PROFILES: Dict[str, ProfileFn] = {**NORMAL_PROFILES, **ATTACK_PROFILES}


@dataclass
class WindowSample:
    label: str  # "normal" or "attack"
    scenario: str  # profile name
    packets: List[Any]


def generate_window(profile_name: str, rng: random.Random, window_seconds: float = 5.0) -> List[Any]:
    return ALL_PROFILES[profile_name](rng, window_seconds)


def generate_dataset(
    profile_names: List[str],
    n_per_profile: int,
    seed: int,
    window_seconds: float = 5.0,
) -> List[WindowSample]:
    """Generate a labelled dataset of window samples.

    Each profile instance is generated with its own independent RNG
    derived from ``seed``, so results are fully reproducible for a given
    seed while every instance still varies (different rates, sizes, host
    counts, jitter).
    """
    master_rng = random.Random(seed)
    samples: List[WindowSample] = []
    for name in profile_names:
        label = "normal" if name in NORMAL_PROFILES else "attack"
        for _ in range(n_per_profile):
            instance_seed = master_rng.randrange(2**32)
            instance_rng = random.Random(instance_seed)
            packets = generate_window(name, instance_rng, window_seconds)
            samples.append(WindowSample(label=label, scenario=name, packets=packets))
    return samples


def build_feature_dataset(
    profile_names: List[str],
    n_per_profile: int,
    seed: int,
    window_seconds: float = 5.0,
):
    """Generate a labelled dataset and run every window through
    core.extractor.extract_features() -- the same function main.py's
    Orchestrator calls at inference time -- so training/evaluation data
    can never drift from what the live pipeline actually computes.

    Returns (X, labels, scenarios): X is an (n_samples, n_features) array,
    labels is a parallel list of "normal"/"attack", scenarios is a
    parallel list of profile names.
    """
    samples = generate_dataset(profile_names, n_per_profile, seed, window_seconds)
    X = np.array([extract_features(s.packets, window_seconds) for s in samples], dtype=np.float64)
    labels = [s.label for s in samples]
    scenarios = [s.scenario for s in samples]
    return X, labels, scenarios


def write_example_pcaps(output_dir: Path, seed: int = 0, window_seconds: float = 5.0) -> List[Path]:
    """Write one example capture per profile to ``output_dir``, for manual
    inspection -- not used by training/evaluation itself."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    written = []
    for name in ALL_PROFILES:
        packets = generate_window(name, rng, window_seconds)
        path = output_dir / f"{name}.pcap"
        if packets:
            wrpcap(str(path), packets)
            written.append(path)
    return written
