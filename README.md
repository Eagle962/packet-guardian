# packet-guardian

A modular, defensive network traffic anomaly detection tool. It combines a
configurable **rule-based engine** (SYN scan / ICMP flood detection) with an
**unsupervised ML detector** and renders everything live in a terminal
dashboard built with [Rich](https://github.com/Textualize/rich).

packet-guardian is a monitoring tool only: it observes traffic you are
authorized to inspect and reports on it. It contains no exploitation,
malware, or unauthorized-access functionality.

## Features

- **Live or offline analysis** — sniff a live interface or replay a `.pcap`
  file through the exact same detection pipeline.
- **BPF filtering** — restrict traffic with standard Berkeley Packet Filter
  expressions.
- **Rule engine** — extensible `BaseRule` / `Ruleset` architecture with
  built-in `SynScanRule` (fast scans), `SlowSynScanRule` (low-and-slow
  scans), and `IcmpFloodRule` detectors, fully configurable thresholds and
  time windows, covering both IPv4 and IPv6 traffic. See **Detection
  Limits** below for what these rules do *not* catch.
- **ML anomaly detection** — packets are summarized into a fixed feature
  vector (`core.extractor.FEATURE_NAMES`) per time window and scored by a
  trained model.
- **Live TUI dashboard** — split-screen view of traffic/ML stats (top) and
  rule engine alerts (bottom), rendered with Rich.
- **Automatic offline fallback** — if live capture fails (e.g. missing
  capture privileges), the tool automatically generates a synthetic sample
  capture and continues in PCAP mode instead of crashing.

## Architecture

```
                         ┌─────────────────┐
   --interface / --pcap  │   core/sniffer   │  (only module that imports Scapy)
   ────────────────────► │  capture_live()  │
                         │  read_pcap()     │
                         └────────┬─────────┘
                                  │ packet stream
                                  ▼
                         ┌─────────────────┐
                         │   core/rules     │
                         │  Ruleset         │──► alerts (SYN_SCAN,
                         │  BaseRule impls  │    SLOW_SYN_SCAN, ICMP_FLOOD)
                         └────────┬─────────┘
                                  │ packet stream (windowed)
                                  ▼
                         ┌─────────────────┐
                         │  core/extractor  │
                         │ extract_features │──► FEATURE_NAMES vector
                         └────────┬─────────┘
                                  ▼
                         ┌─────────────────┐
                         │ ml_engine/       │
                         │  detector.py     │──► anomaly_score, status
                         │  (model.pkl,     │
                         │   generated --   │
                         │   not tracked)   │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │   ui/console     │  (pure rendering, no logic)
                         │   Dashboard      │──► live split-screen TUI
                         └─────────────────┘

                All of the above is wired together by main.py.
```

## File Structure

```
packet-guardian/
├── core/
│   ├── sniffer.py       # Scapy capture/replay wrapper (live + pcap) -- the
│   │                    #   only module that imports Scapy's capture APIs
│   ├── extractor.py     # packet window -> FEATURE_NAMES vector
│   ├── rules.py         # BaseRule / Ruleset / SynScanRule / SlowSynScanRule
│   │                    #   / IcmpFloodRule (IPv4 + IPv6)
│   └── sample_data.py   # synthetic sample.pcap generator (offline fallback)
├── ml_engine/
│   ├── traffic_profiles.py  # labelled synthetic traffic generator (packets,
│   │                        #   not hand-written feature vectors)
│   ├── preprocessing.py # shared log1p column transform (train + serve)
│   ├── train.py         # hyperparameter/model-family search + training
│   ├── evaluate.py      # held-out evaluation -> EVALUATION.md
│   ├── detector.py      # load_model() / predict()
│   └── model.pkl        # trained pipeline + metadata (generated, gitignored)
├── ui/
│   └── console.py       # Rich dashboard (rendering only)
├── data/
│   └── sample.pcap      # synthetic capture: normal + SYN scan + ICMP flood
├── tests/
│   ├── test_rules.py
│   ├── test_rules_bounded_state.py
│   ├── test_slow_scan.py
│   ├── test_ipv6.py
│   ├── test_extractor.py
│   ├── test_orchestrator.py
│   ├── test_traffic_profiles.py
│   ├── test_detector.py
│   └── test_ml_quality.py   # hard ML quality gates (see EVALUATION.md)
├── main.py               # CLI orchestrator
├── requirements.txt
├── EVALUATION.md         # ML detector evaluation report (generated)
└── README.md
```

## Installation

```bash
cd packet-guardian
python -m venv venv          # if you don't already have one
source venv/bin/activate
python -m pip install -r requirements.txt
```

## Usage

### PCAP (offline) mode

```bash
python main.py --pcap data/sample.pcap
```

### Live capture mode

```bash
python main.py --interface en0
```

Live capture requires elevated privileges (see **Permission Requirements**
below). If it fails, packet-guardian automatically falls back to a
synthetic sample capture rather than crashing.

### BPF filtering (either mode)

```bash
python main.py --pcap data/sample.pcap --bpf "tcp or icmp"
python main.py --interface en0 --bpf "host 10.0.0.1"
```

### Other options

```bash
python main.py --pcap data/sample.pcap --window 10   # 10s rule/feature window
python main.py --interface en0 --timeout 30           # capture for 30s
```

## Model Training

`ml_engine/model.pkl` is **not** checked into version control (see
Security & Privacy below) -- generate it locally before running live/PCAP
mode with ML detection enabled:

```bash
python -m ml_engine.train
```

This generates labelled synthetic traffic (`ml_engine/traffic_profiles.py`),
searches hyperparameters across three candidate model families
(`IsolationForest`, `OneClassSVM`, `LocalOutlierFactor`) on a validation
split, and saves the winning pipeline (log1p + `RobustScaler` +
model) plus feature/version/provenance metadata to `ml_engine/model.pkl`
via `joblib`. If no model is present (or it fails its schema check),
`main.py` prints a warning and falls back to rule-only detection rather
than crashing.

To see how well the currently-trained model actually performs, and
against what data:

```bash
python -m ml_engine.evaluate
```

This scores the model against a held-out labelled set (a seed never used
during training or hyperparameter selection) and (re)writes
[`EVALUATION.md`](EVALUATION.md) with the real numbers -- dataset
provenance, per-scenario precision/recall/FPR, ROC-AUC, and a confusion
matrix. `tests/test_ml_quality.py` enforces hard minimum-quality gates
against those same numbers.

## Detection Limits

The built-in rules cover a specific, narrow set of behaviors. Do not read
their presence as general scan/flood coverage -- each one has real, known
blind spots:

- **`SynScanRule` / `SlowSynScanRule` only look at pure SYN packets** (SYN
  set, ACK clear). They do **not** detect:
  - **FIN, NULL, or Xmas scans** (probes with FIN/no flags/FIN+PSH+URG set)
    -- a classic technique for evading exactly this kind of SYN-counting
    rule.
  - **ACK scans** (used for firewall/stateful-filter mapping, not open-port
    discovery) -- these carry the ACK flag and are explicitly excluded by
    design (to avoid flagging normal handshake traffic), which also means
    a real ACK scan passes through unnoticed.
  - **UDP port scans** -- there is no UDP-scan rule at all. A UDP sweep
    across many ports produces no alert from either rule.
  - **Distributed / low-and-slow-per-source scans**: both rules key
    entirely on a single source IP. A scan spread across many source IPs
    (a botnet each probing a handful of ports) never crosses either rule's
    per-source threshold, no matter how coordinated or effective the
    overall scan is.
  - `SlowSynScanRule`'s 300s window still has a floor: a scan slower than
    roughly `unique_port_threshold` ports per 300s (e.g. one port every
    10+ seconds) will still evade it. Raising the window further trades
    detection latency and memory for coverage of even slower scans.
- **`IcmpFloodRule` only looks at ICMP/ICMPv6 Echo Request.** It does not
  detect floods using other ICMP types, non-ICMP floods (UDP floods, TCP
  SYN floods to a single port/service rather than spread across ports),
  or application-layer denial-of-service patterns.
- **All rules are per-source-IP and stateless across restarts.** Spoofed
  source addresses can dilute a real scan below threshold by spreading
  probes across many forged IPs (see the bounded-state fix above -- this
  keeps the *monitor* from running out of memory, it does not make a
  spoofed-source scan detectable). State does not persist between runs of
  packet-guardian.

If your threat model includes any of the above, treat the rule engine as
one signal among several (alongside the ML detector and your own
monitoring), not a complete scan/flood detector.

## Writing a Custom Rule

Rules subclass `core.rules.BaseRule` and implement `process_packet`, which
receives one packet at a time and returns a list of alert dicts (usually
empty):

```python
from core.rules import BaseRule

class PortKnockRule(BaseRule):
    name = "PORT_KNOCK"

    def __init__(self, window_seconds: float = 5.0, port_threshold: int = 5) -> None:
        super().__init__(window_seconds)
        self.port_threshold = port_threshold
        # ... your own state, e.g. a dict of deques keyed by source IP

    def process_packet(self, packet, timestamp=None):
        alerts = []
        # inspect `packet`, update internal state, and append an alert
        # dict when your condition is met, e.g.:
        # alerts.append({
        #     "rule_name": self.name,
        #     "source_ip": src_ip,
        #     "count": count,
        #     "time_window": self.window_seconds,
        #     "timestamp": timestamp,
        #     "severity": "MEDIUM",
        # })
        return alerts

    def reset(self):
        # clear internal state (called between test cases, etc.)
        ...
```

Register it alongside the built-in rules in `main.py`:

```python
self.ruleset = Ruleset([
    SynScanRule(window_seconds=window_seconds),
    IcmpFloodRule(window_seconds=window_seconds),
    PortKnockRule(window_seconds=window_seconds),
])
```

## Testing

```bash
python -m pytest -q          # unit tests: rules, extractor, detector
python -m compileall .        # syntax/import sanity check
python main.py --pcap data/sample.pcap   # end-to-end integration check
```

`data/sample.pcap` is a synthetic capture containing normal TCP/UDP/ICMP
traffic plus an embedded SYN scan and ICMP flood scenario, so a correct run
of the integration check will show both a `SYN_SCAN` and an `ICMP_FLOOD`
alert in the dashboard's alert panel.

## Permission Requirements

Live packet capture (`--interface`) opens a raw socket and, on most
platforms, requires elevated privileges:

- **macOS/Linux**: run with `sudo`, or grant your Python interpreter/BPF
  device the appropriate capture capability (e.g. `setcap cap_net_raw+eip`
  on Linux).
- **Without privileges**, live capture will fail with a permission error;
  packet-guardian catches this and automatically switches to the synthetic
  PCAP fallback so the rest of the pipeline still runs.

PCAP mode (`--pcap`) requires no special privileges — it only reads a file
from disk.

## Security & Privacy Considerations

- **Only sniff traffic you are authorized to monitor.** Live packet capture
  can expose sensitive data (credentials, personal information) traversing
  the network. Do not run `--interface` mode against networks or hosts you
  do not own or have explicit permission to monitor.
- **Untrusted model files are a code-execution risk.** `ml_engine/model.pkl`
  is loaded with `joblib.load()`, which (like the `pickle` module it uses
  internally for non-array objects) can execute arbitrary code if the file
  has been tampered with. Switching from bare `pickle` to `joblib` changes
  how the array data inside the artifact is stored -- it does **not**
  change this risk. Only load model files you trained yourself (`python -m
  ml_engine.train`) or obtained from a source you trust as much as you'd
  trust running its code directly — never load a `.pkl` file from an
  untrusted or unauthenticated origin. The model is intentionally not
  checked into version control (see Model Training above); generate it
  locally instead of fetching a prebuilt one from anywhere you can't
  verify.
- This tool is strictly **defensive/monitoring** software: it detects and
  reports anomalies, and includes no exploitation, malware, or unauthorized
  access capabilities.
