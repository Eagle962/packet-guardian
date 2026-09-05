"""Packet capture abstraction layer built on top of Scapy.

This module is the only place in the codebase that imports Scapy directly.
Callers (main.py, tests, etc.) interact exclusively through the functions
defined here so the rest of the application never depends on Scapy's API
shape.
"""

from __future__ import annotations

from typing import Callable, Optional

from scapy.all import Packet
from scapy.sendrecv import sniff

PacketCallback = Callable[[Packet], None]


def capture_live(
    interface: Optional[str],
    bpf_filter: Optional[str],
    timeout: Optional[float],
    callback: PacketCallback,
) -> None:
    """Capture packets live from a network interface.

    Args:
        interface: Name of the network interface to sniff on (e.g. "en0").
            If None, Scapy selects its default interface.
        bpf_filter: Berkeley Packet Filter expression, or None for no filter.
        timeout: Maximum time in seconds to capture for, or None to capture
            indefinitely (until interrupted).
        callback: Invoked once per captured packet.

    Raises:
        PermissionError: If the current process lacks the privileges
            required to open a live capture socket.
        OSError: For other Scapy/OS level capture failures.
    """
    sniff(
        iface=interface,
        filter=bpf_filter,
        timeout=timeout,
        prn=callback,
        store=False,
    )


def read_pcap(
    filepath: str,
    bpf_filter: Optional[str],
    callback: PacketCallback,
) -> None:
    """Replay packets from a PCAP file on disk.

    Args:
        filepath: Path to the .pcap/.pcapng file to read.
        bpf_filter: Optional BPF expression used to filter packets read
            from disk, or None for no filter.
        callback: Invoked once per packet that passes the filter.
    """
    sniff(
        offline=filepath,
        filter=bpf_filter,
        prn=callback,
        store=False,
    )
