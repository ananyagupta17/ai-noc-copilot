"""
Synthetic data generator for AI NOC Copilot.
Generates incidents, alerts, logs, topology, and runbooks.
Run: python scripts/generate_data.py
"""

import json
import random
import uuid
from datetime import datetime, timedelta
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from faker import Faker
import networkx as nx
from config import (
    INCIDENTS_DIR, LOGS_DIR, ALERTS_DIR, TOPOLOGY_DIR, RUNBOOKS_DIR
)

fake = Faker()
random.seed(42)
Faker.seed(42)

# ─────────────────────────────────────────────
# CONSTANTS — realistic telecom vocabulary
# ─────────────────────────────────────────────

REGIONS = ["Singapore", "Mumbai", "London", "Frankfurt", "New York", "Tokyo", "Sydney", "Dubai"]

DEVICE_TYPES = ["edge-router", "core-router", "aggregation-switch", "pe-router", "p-router"]

SEVERITIES = ["P1", "P2", "P3", "P4"]
SEVERITY_WEIGHTS = [0.1, 0.25, 0.40, 0.25]  # P1 rare, P3 most common

SYMPTOM_TYPES = [
    "packet_loss", "high_latency", "bgp_flap", "interface_down",
    "crc_errors", "cpu_spike", "memory_exhaustion", "mpls_failure",
    "dns_outage", "optical_degradation"
]

ROOT_CAUSES = [
    "fiber_cut", "optical_degradation", "hardware_failure", "bgp_misconfiguration",
    "ddos_attack", "software_bug", "power_fluctuation", "capacity_exhaustion",
    "misconfigured_acl", "upstream_provider_issue"
]

RESOLUTIONS = [
    "Traffic rerouted via backup path",
    "Fiber repaired by field team",
    "Interface reset and re-established",
    "BGP session manually reset",
    "Hardware replaced",
    "Configuration rollback applied",
    "Upstream provider resolved issue",
    "Traffic rate-limited to control load",
    "Optical amplifier replaced",
    "Failover to standby device triggered",
]

CUSTOMER_SEGMENTS = ["Enterprise MPLS", "SD-WAN", "Cloud Connect", "Internet", "Voice"]


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def random_timestamp(days_back=90):
    """Random timestamp within the last N days."""
    start = datetime.now() - timedelta(days=days_back)
    return start + timedelta(seconds=random.randint(0, days_back * 86400))

def random_device(region):
    """Generate a realistic device hostname for a region."""
    code = region[:3].upper()
    dtype = random.choice(["ER", "CR", "PE", "AGG"])
    num = random.randint(1, 8)
    return f"{code}-{dtype}-{num:02d}"

def random_ip():
    return f"10.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}"


# ─────────────────────────────────────────────
# 1. TOPOLOGY GENERATION (NetworkX)
# ─────────────────────────────────────────────
# We build a realistic backbone graph:
# - Each region has 2-3 edge routers and 1-2 core routers
# - Core routers are interconnected (backbone mesh)
# - Edge routers connect to their region's core

def generate_topology():
    G = nx.Graph()
    all_nodes = {}

    # Create nodes per region
    for region in REGIONS:
        code = region[:3].upper()
        core_count = random.randint(1, 2)
        edge_count = random.randint(2, 3)

        core_nodes = [f"{code}-CR-{i+1:02d}" for i in range(core_count)]
        edge_nodes = [f"{code}-ER-{i+1:02d}" for i in range(edge_count)]

        for node in core_nodes:
            G.add_node(node, type="core-router", region=region,
                       ip=random_ip(), vendor=random.choice(["Cisco", "Juniper", "Nokia"]))
        for node in edge_nodes:
            G.add_node(node, type="edge-router", region=region,
                       ip=random_ip(), vendor=random.choice(["Cisco", "Juniper", "Nokia"]))

        # Edge routers connect to core in same region
        for er in edge_nodes:
            cr = random.choice(core_nodes)
            G.add_edge(er, cr, link_type="access", capacity_gbps=10, region=region)

        all_nodes[region] = {"core": core_nodes, "edge": edge_nodes}

    # Connect core routers across regions (backbone mesh)
    region_list = list(all_nodes.keys())
    for i in range(len(region_list)):
        for j in range(i + 1, len(region_list)):
            # Not every region pair is connected — ~60% chance
            if random.random() < 0.6:
                r1_cores = all_nodes[region_list[i]]["core"]
                r2_cores = all_nodes[region_list[j]]["core"]
                cr1 = random.choice(r1_cores)
                cr2 = random.choice(r2_cores)
                G.add_edge(cr1, cr2, link_type="backbone",
                           capacity_gbps=random.choice([100, 400]),
                           region=f"{region_list[i]}-{region_list[j]}")

    # Serialize to JSON
    topology_data = {
        "nodes": [
            {"id": n, **G.nodes[n]} for n in G.nodes
        ],
        "edges": [
            {"source": u, "target": v, **G.edges[u, v]} for u, v in G.edges
        ],
        "summary": {
            "total_nodes": G.number_of_nodes(),
            "total_edges": G.number_of_edges(),
            "regions": REGIONS
        }
    }

    out = TOPOLOGY_DIR / "topology.json"
    out.write_text(json.dumps(topology_data, indent=2))
    print(f"  Topology: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges → {out}")
    return topology_data


# ─────────────────────────────────────────────
# 2. INCIDENT GENERATION (Faker + templates)
# ─────────────────────────────────────────────
# Each incident is a full lifecycle:
# detected → investigating → resolved, with timestamps, RCA, resolution

def generate_incidents(n=80):
    incidents = []

    for _ in range(n):
        region = random.choice(REGIONS)
        severity = random.choices(SEVERITIES, weights=SEVERITY_WEIGHTS)[0]
        symptom = random.choice(SYMPTOM_TYPES)
        root_cause = random.choice(ROOT_CAUSES)
        device = random_device(region)
        detected_at = random_timestamp()
        ttd = timedelta(minutes=random.randint(2, 30))       # time to detect
        ttr = timedelta(minutes=random.randint(15, 240))     # time to resolve

        incident = {
            "incident_id": f"INC-{fake.numerify('######')}",
            "severity": severity,
            "region": region,
            "affected_device": device,
            "symptom": symptom,
            "root_cause": root_cause,
            "description": _incident_description(symptom, region, device),
            "customer_segment": random.choice(CUSTOMER_SEGMENTS),
            "affected_customers": random.randint(1, 500) if severity in ["P1", "P2"] else random.randint(0, 50),
            "detected_at": detected_at.isoformat(),
            "acknowledged_at": (detected_at + ttd).isoformat(),
            "resolved_at": (detected_at + ttd + ttr).isoformat(),
            "mttr_minutes": int((ttd + ttr).total_seconds() / 60),
            "resolution": random.choice(RESOLUTIONS),
            "engineer": fake.name(),
            "tags": _incident_tags(symptom, root_cause),
        }
        incidents.append(incident)

    out = INCIDENTS_DIR / "incidents.json"
    out.write_text(json.dumps(incidents, indent=2))
    print(f"  Incidents: {len(incidents)} records → {out}")
    return incidents


def _incident_description(symptom, region, device):
    """Template-based descriptions so they read like real tickets."""
    templates = {
        "packet_loss": f"Packet loss detected on {device} in {region}. Customers reporting intermittent connectivity. Loss rate exceeding 15%.",
        "high_latency": f"Latency spike observed on {region} edge. RTT exceeding 250ms on {device}. SLA breach imminent.",
        "bgp_flap": f"BGP session instability on {device} ({region}). Session flapping every 2-3 minutes. Routing table unstable.",
        "interface_down": f"Interface GigabitEthernet0/1 down on {device} in {region}. Link state: DOWN. Traffic impact confirmed.",
        "crc_errors": f"CRC error rate spiking on {device}. Errors: >10000/min. Possible physical layer degradation in {region}.",
        "cpu_spike": f"CPU utilization at 95%+ on {device} in {region}. Process: BGP scanner consuming excess cycles.",
        "memory_exhaustion": f"Memory utilization critical on {device} ({region}). Free memory < 5%. Risk of process crash.",
        "mpls_failure": f"MPLS label distribution failure on {device}. LDP sessions dropping in {region}.",
        "dns_outage": f"DNS resolution failures reported from {region}. {device} DNS forwarder not responding.",
        "optical_degradation": f"Optical power levels degrading on {device} in {region}. Rx power: -28dBm (threshold: -23dBm).",
    }
    return templates.get(symptom, f"Network issue detected on {device} in {region}.")


def _incident_tags(symptom, root_cause):
    tag_map = {
        "packet_loss": ["networking", "loss", "customer-impact"],
        "bgp_flap": ["routing", "bgp", "instability"],
        "optical_degradation": ["physical", "fiber", "optical"],
        "cpu_spike": ["performance", "cpu", "device"],
    }
    base = tag_map.get(symptom, ["networking"])
    base.append(root_cause.replace("_", "-"))
    return base


# ─────────────────────────────────────────────
# 3. ALERT GENERATION (templates + Faker)
# ─────────────────────────────────────────────
# Alerts are shorter, machine-generated events.
# One incident typically spawns multiple alerts.

def generate_alerts(incidents, alerts_per_incident=(1, 6)):
    alerts = []

    for inc in incidents:
        n_alerts = random.randint(*alerts_per_incident)
        base_time = datetime.fromisoformat(inc["detected_at"])

        for i in range(n_alerts):
            alert_time = base_time + timedelta(seconds=random.randint(0, 300))
            alerts.append({
                "alert_id": f"ALT-{uuid.uuid4().hex[:8].upper()}",
                "incident_id": inc["incident_id"],  # links alert to parent incident
                "device": inc["affected_device"],
                "region": inc["region"],
                "severity": random.choice(["CRITICAL", "MAJOR", "MINOR"]),
                "alert_type": inc["symptom"],
                "message": _alert_message(inc["symptom"], inc["affected_device"]),
                "metric_value": _alert_metric(inc["symptom"]),
                "threshold_breached": True,
                "timestamp": alert_time.isoformat(),
                "source": random.choice(["Nagios", "Prometheus", "SolarWinds", "Zabbix"]),
            })

    out = ALERTS_DIR / "alerts.json"
    out.write_text(json.dumps(alerts, indent=2))
    print(f"  Alerts: {len(alerts)} records → {out}")
    return alerts


def _alert_message(symptom, device):
    messages = {
        "packet_loss":        f"CRITICAL: Packet loss {random.randint(10,40)}% on {device}",
        "high_latency":       f"MAJOR: Latency {random.randint(150,500)}ms exceeds threshold on {device}",
        "bgp_flap":           f"CRITICAL: BGP session DOWN on {device} peer {random_ip()}",
        "interface_down":     f"CRITICAL: Interface Gi0/{random.randint(0,4)} DOWN on {device}",
        "crc_errors":         f"MAJOR: CRC errors {random.randint(1000,50000)}/min on {device}",
        "cpu_spike":          f"MAJOR: CPU {random.randint(85,99)}% on {device}",
        "memory_exhaustion":  f"CRITICAL: Memory {random.randint(90,99)}% used on {device}",
        "mpls_failure":       f"CRITICAL: LDP session DOWN on {device}",
        "dns_outage":         f"MAJOR: DNS resolution timeout on {device}",
        "optical_degradation":f"CRITICAL: Optical Rx power {-random.randint(25,35)}dBm on {device}",
    }
    return messages.get(symptom, f"ALERT: Issue detected on {device}")


def _alert_metric(symptom):
    metrics = {
        "packet_loss":        {"name": "packet_loss_pct", "value": round(random.uniform(10, 45), 1)},
        "high_latency":       {"name": "rtt_ms", "value": random.randint(150, 600)},
        "bgp_flap":           {"name": "bgp_session_state", "value": 0},
        "interface_down":     {"name": "interface_status", "value": 0},
        "crc_errors":         {"name": "crc_errors_per_min", "value": random.randint(500, 60000)},
        "cpu_spike":          {"name": "cpu_utilization_pct", "value": random.randint(85, 99)},
        "memory_exhaustion":  {"name": "memory_used_pct", "value": random.randint(90, 99)},
        "optical_degradation":{"name": "rx_power_dbm", "value": -random.randint(25, 35)},
    }
    return metrics.get(symptom, {"name": "generic_metric", "value": random.randint(1, 100)})


# ─────────────────────────────────────────────
# 4. LOG GENERATION (templates)
# ─────────────────────────────────────────────
# Logs are plain text files mimicking real router/syslog output.
# Each log file corresponds to one incident.

def generate_logs(incidents, sample_size=40):
    """Generate one log file per sampled incident."""
    sampled = random.sample(incidents, min(sample_size, len(incidents)))

    for inc in sampled:
        lines = _generate_log_lines(inc)
        out = LOGS_DIR / f"{inc['incident_id']}.log"
        out.write_text("\n".join(lines))

    print(f"  Logs: {len(sampled)} log files → {LOGS_DIR}")


def _generate_log_lines(inc):
    """Generate syslog-style lines for a given incident."""
    device = inc["affected_device"]
    symptom = inc["symptom"]
    base_time = datetime.fromisoformat(inc["detected_at"])
    lines = []

    # Common preamble lines
    for i in range(random.randint(3, 6)):
        t = base_time - timedelta(minutes=random.randint(1, 10))
        lines.append(f"{t.strftime('%b %d %H:%M:%S')} {device} : %SYS-5-CONFIG_I: Configured from console by admin")

    # Symptom-specific log lines
    symptom_logs = {
        "bgp_flap": [
            f"%BGP-5-ADJCHANGE: neighbor {random_ip()} Down BGP Notification sent",
            f"%BGP-3-NOTIFICATION: sent to neighbor {random_ip()} 6/7 (Cease/connection collision resolution)",
            f"%BGP-5-ADJCHANGE: neighbor {random_ip()} Up",
            f"%BGP-5-ADJCHANGE: neighbor {random_ip()} Down Hold Timer Expired",
        ],
        "interface_down": [
            f"%LINK-3-UPDOWN: Interface GigabitEthernet0/1, changed state to down",
            f"%LINEPROTO-5-UPDOWN: Line protocol on Interface GigabitEthernet0/1, changed state to down",
            f"%LINK-3-UPDOWN: Interface GigabitEthernet0/1, changed state to up",
            f"%LINK-3-UPDOWN: Interface GigabitEthernet0/1, changed state to down",
        ],
        "crc_errors": [
            f"%ERRORS-3-RX_CRC: CRC errors on GigabitEthernet0/2: {random.randint(1000,9999)} errors/min",
            f"%PLATFORM-4-ELEMENT_WARNING: Interface Gi0/2 CRC error threshold exceeded",
            f"%ERRORS-3-RX_CRC: CRC errors on GigabitEthernet0/2: {random.randint(10000,50000)} errors/min",
        ],
        "packet_loss": [
            f"%IP-4-DUPADDR: Duplicate address {random_ip()} on GigabitEthernet0/0",
            f"%MPLS-3-BADLABEL: Received bad label {random.randint(100,999)} on interface Gi0/1",
            f"QoS: drop tail on queue 3, packet loss rate: {random.randint(10,40)}%",
        ],
        "cpu_spike": [
            f"%CPU-4-HIGH: CPU utilization: {random.randint(85,99)}%; 5min: {random.randint(80,95)}%",
            f"Scheduler: process BGP Scanner at {random.randint(50,80)}% CPU",
            f"%SYS-3-CPUHOG: Task ran for {random.randint(2000,5000)}ms",
        ],
        "optical_degradation": [
            f"%OPTICAL-3-RXPOWER_LOW: Rx power on {device} Gi0/1: -{random.randint(25,35)} dBm (threshold: -23 dBm)",
            f"%OPTICAL-4-SIGNAL_DEGRADED: Signal quality degraded on interface Gi0/1",
            f"%OPTICAL-3-LOS: Loss of signal on Gi0/1",
        ],
    }

    log_lines = symptom_logs.get(symptom, [f"%SYS-3-ERROR: Unknown error on {device}"])

    for i, msg in enumerate(log_lines):
        t = base_time + timedelta(seconds=i * random.randint(10, 60))
        lines.append(f"{t.strftime('%b %d %H:%M:%S')} {device} : {msg}")

    # Some noisy follow-up lines
    for i in range(random.randint(2, 5)):
        t = base_time + timedelta(minutes=random.randint(1, 15))
        lines.append(f"{t.strftime('%b %d %H:%M:%S')} {device} : %SYS-5-RELOAD: Reload requested by NOC engineer")

    lines.sort()  # chronological order
    return lines


# ─────────────────────────────────────────────
# 5. RUNBOOK GENERATION (manual text)
# ─────────────────────────────────────────────
# These are written as real SOPs — the RAG system retrieves these.
# Quality matters here more than quantity.

RUNBOOKS = {
    "packet_loss_troubleshooting.txt": """
RUNBOOK: Packet Loss Troubleshooting
Category: Network Performance
Severity: P1/P2
Last Updated: 2024-01-15

OVERVIEW
--------
This runbook covers investigation and remediation of packet loss events on Tata Communications backbone and edge infrastructure.

SYMPTOMS
--------
- Customer reports intermittent connectivity
- Monitoring alerts: packet_loss_pct > 5%
- Ping loss > 5% to customer CE device
- SLA breach notifications triggered

STEP 1 — INITIAL TRIAGE
------------------------
1. Identify affected device and interface from alert
2. Run ping test from NOC to affected device:
   ping <device-ip> repeat 100 size 1500
3. Check interface error counters:
   show interface GigabitEthernet0/1 | include error|drop|loss
4. Check if issue is unidirectional or bidirectional

STEP 2 — CHECK FOR CRC ERRORS
-------------------------------
CRC errors indicate physical layer problems (fiber, optics, cables).
   show interfaces | include CRC|input error
If CRC errors > 1000/min:
  → Suspect optical degradation or cable fault
  → Proceed to optical diagnostics (see optical_degradation runbook)

STEP 3 — CHECK QUEUE DROPS
---------------------------
   show interfaces | include output drop|queue
If output drops present:
  → QoS policy may be misconfigured or interface is congested
  → Check traffic rate vs interface capacity:
    show interfaces GigabitEthernet0/1 | include rate

STEP 4 — CHECK UPSTREAM/DOWNSTREAM
------------------------------------
1. Identify upstream device from topology
2. Run traceroute to isolate hop where loss begins:
   traceroute <destination-ip> source <loopback>
3. Ping each hop — loss at specific hop identifies fault location

STEP 5 — MPLS LABEL CHECK
--------------------------
   show mpls forwarding-table
   show mpls ldp neighbor
If LDP neighbors are missing or label errors present:
  → Refer to MPLS failure runbook

STEP 6 — REMEDIATION OPTIONS
------------------------------
Option A — Reroute traffic:
  - Identify backup path via topology
  - Adjust OSPF/ISIS metric to shift traffic:
    interface GigabitEthernet0/2
     ip ospf cost 100
Option B — Interface reset (use with caution):
  shutdown / no shutdown on affected interface
Option C — Escalate to field team if physical fault confirmed

STEP 7 — CUSTOMER COMMUNICATION
---------------------------------
- Update ticket with findings within 15 minutes
- For P1: Send executive update every 30 minutes
- Template: "We have identified [root cause] affecting [region]. Traffic is being rerouted via [backup path]. ETA for full resolution: [time]."

ESCALATION
----------
- L2 Network Engineering: if unresolved after 30 mins
- Fiber Operations: if physical fault confirmed
- Account Manager: if enterprise customer impacted > 1 hour
""",

    "bgp_flap_runbook.txt": """
RUNBOOK: BGP Session Instability / Flapping
Category: Routing
Severity: P1/P2
Last Updated: 2024-02-01

OVERVIEW
--------
BGP flapping causes route withdrawals and re-advertisements, leading to traffic blackholing and instability. This runbook guides investigation of BGP session drops.

SYMPTOMS
--------
- BGP session repeatedly going UP/DOWN in logs
- Routes being withdrawn and re-advertised
- Customers reporting intermittent loss on specific prefixes
- Alert: bgp_session_state = 0

STEP 1 — IDENTIFY AFFECTED SESSION
-------------------------------------
   show bgp summary
   show bgp neighbors | include BGP state|Resets|Notifications
Note the neighbor IP, AS number, and flap count.

STEP 2 — CHECK HOLD TIMER EXPIRY
----------------------------------
If session drops show "Hold Timer Expired":
  → Keepalive messages not reaching peer in time
  → Check for high CPU on router:
    show processes cpu sorted | head 10
  → Check for packet loss between peers:
    ping <bgp-peer-ip> source <loopback> repeat 1000

STEP 3 — CHECK FOR NOTIFICATION MESSAGES
------------------------------------------
   show bgp neighbors <peer-ip> | include Notification|Error
Common notification codes:
  - 6/7: Connection collision (two sessions trying to form)
  - 4/0: Hold timer expired
  - 2/2: Bad peer AS number (config mismatch)
  - 3/x: UPDATE message error (check route policy)

STEP 4 — CHECK TCP SESSION STABILITY
--------------------------------------
BGP runs over TCP. TCP issues cause BGP drops.
   show tcp brief | include <peer-ip>
If TCP session is unstable:
  → MTU mismatch on path (check with ping df-bit size 1500)
  → QoS policy dropping BGP packets (check for any rate-limiters on TCP 179)

STEP 5 — CHECK ROUTE POLICY / FILTERS
----------------------------------------
A bad route-map or prefix-list can cause UPDATE errors and resets.
   show route-map
   show bgp neighbors <peer-ip> | include policy

STEP 6 — REMEDIATION
----------------------
Option A — Increase hold timer (temporary stability):
  router bgp <ASN>
   neighbor <peer-ip> timers 10 30
Option B — Soft reset (no traffic impact):
  clear ip bgp <peer-ip> soft
Option C — Hard reset (use carefully, causes route withdrawal):
  clear ip bgp <peer-ip>
Option D — BFD check:
  show bfd neighbors detail

ESCALATION
----------
- If session with upstream provider: contact provider NOC
- If config-related: escalate to Network Engineering
- For P1 with customer impact: notify account manager immediately
""",

    "high_latency_runbook.txt": """
RUNBOOK: High Latency Investigation
Category: Network Performance
Severity: P2/P3
Last Updated: 2024-01-20

OVERVIEW
--------
Latency spikes degrade application performance and can breach SLAs. Common causes include congestion, routing changes, and hardware issues.

SYMPTOMS
--------
- RTT > 150ms on normally sub-50ms paths
- Customer complaints about slow application response
- Alert: rtt_ms threshold breached

STEP 1 — BASELINE COMPARISON
------------------------------
First confirm this is abnormal:
- Check historical latency from monitoring (last 7 days)
- Is it affecting all destinations or specific prefixes?
  traceroute <affected-destination> source <loopback>

STEP 2 — IDENTIFY CONGESTED HOP
---------------------------------
Run traceroute with timing:
  traceroute <destination> probe 10
Look for hop where RTT jumps significantly.
That hop or the link before it is the bottleneck.

STEP 3 — CHECK INTERFACE UTILIZATION
--------------------------------------
On the congested hop device:
  show interfaces GigabitEthernet0/1 | include rate|utilization
If utilization > 80%:
  → Traffic engineering needed
  → Check if recent routing change shifted traffic to this path

STEP 4 — CHECK ROUTING TABLE CHANGES
--------------------------------------
Has the routing path changed recently?
  show ip route <destination>
  show bgp <destination>
Compare with expected next-hop. If path is suboptimal:
  → Traffic may have been rerouted due to link failure elsewhere
  → Restore primary path if available

STEP 5 — CHECK FOR MICROBURSTS
--------------------------------
Short bursts can cause queuing without showing sustained high utilization.
  show interfaces GigabitEthernet0/1 | include output queue
  show queueing interface GigabitEthernet0/1

STEP 6 — REMEDIATION
----------------------
Option A — Traffic engineering: shift traffic to lower-latency path
Option B — QoS: prioritize latency-sensitive traffic
Option C — Capacity upgrade: if chronic congestion, raise ticket for link upgrade
Option D — ECMP: verify equal-cost paths are being utilized

ESCALATION
----------
- SLA breach > 30 mins: notify account manager
- Chronic congestion: capacity planning team
""",

    "optical_degradation_runbook.txt": """
RUNBOOK: Optical Signal Degradation
Category: Physical Layer
Severity: P1/P2
Last Updated: 2024-03-01

OVERVIEW
--------
Optical degradation indicates issues with fiber, connectors, amplifiers, or transceivers. Left unresolved it leads to complete link failure.

SYMPTOMS
--------
- CRC errors escalating on interface
- Rx optical power below threshold (typically < -23 dBm)
- Alert: optical_degradation
- Intermittent packet loss that worsens over time

STEP 1 — CHECK OPTICAL POWER LEVELS
--------------------------------------
   show interfaces GigabitEthernet0/1 transceiver
   show controllers GigabitEthernet0/1
Record:
  - Tx power (should be within spec, e.g. -3 to +2 dBm)
  - Rx power (should be > -23 dBm for most transceivers)
  - Temperature (high temp causes performance issues)

STEP 2 — IDENTIFY FAULT LOCATION
----------------------------------
Is Tx normal but Rx low?
  → Problem is on the incoming fiber or far-end transmitter
Is Tx also low?
  → Problem is the local transceiver — replace it
Compare readings at both ends of the link.

STEP 3 — CHECK FOR FIBER CONTAMINATION
----------------------------------------
Dirty connectors are the #1 cause of optical issues.
  → Request field team to inspect and clean connectors with IEC 61300-3-35 standard cleaner
  → Do not use compressed air alone

STEP 4 — CHECK OTDR (if available)
------------------------------------
OTDR trace will show:
  - Exact location of splice loss or break
  - Connector losses
  - Bend radius violations
Escalate OTDR results to fiber operations team.

STEP 5 — INTERIM REMEDIATION
------------------------------
While fiber is being repaired:
  - Reroute traffic to backup path immediately
  - Reduce interface speed (if partial signal): may stabilize
  - Replace transceiver if that is identified as fault

STEP 6 — ESCALATION
---------------------
- Field team dispatch for physical inspection
- Fiber operations for OTDR and repair
- Vendor support if transceiver under warranty

DO NOT attempt fiber repair without certified field team.
""",

    "mpls_failure_runbook.txt": """
RUNBOOK: MPLS / LDP Failure
Category: MPLS
Severity: P1/P2
Last Updated: 2024-02-15

OVERVIEW
--------
MPLS failures cause enterprise VPN traffic to drop or blackhole. LDP session loss is the most common trigger.

SYMPTOMS
--------
- Enterprise MPLS customers report complete loss
- Alert: mpls_failure or LDP session DOWN
- Logs: %MPLS-3-LDP_NBR_DOWN

STEP 1 — CHECK LDP SESSIONS
-----------------------------
   show mpls ldp neighbor
All PE-to-P and PE-to-PE sessions should be UP.
If session is DOWN:
  - Check IGP reachability to LDP peer (ping loopback)
  - LDP uses TCP 646 — check for ACL blocking this port

STEP 2 — CHECK LABEL FORWARDING TABLE
---------------------------------------
   show mpls forwarding-table
Entries should exist for all VPN prefixes.
Missing entries indicate LDP or BGP VPN failure.

STEP 3 — CHECK BGP VPN (L3VPN)
---------------------------------
   show bgp vpnv4 unicast all summary
If BGP VPN sessions are down:
  → Refer to BGP flap runbook
If sessions are up but routes missing:
  → Check route-target import/export config on PE

STEP 4 — CHECK VRF TABLE
--------------------------
   show ip vrf
   show ip route vrf <customer-vrf>
Customer routes should be present. If missing:
  → Route leak or redistribution issue
  → Check route-map applied to VRF

STEP 5 — REMEDIATION
----------------------
Option A — LDP session reset:
  clear mpls ldp neighbor <peer-ip>
Option B — Redistribute check:
  Verify that customer CE routes are being redistributed into VRF
Option C — Traffic reroute via backup PE if available

ESCALATION
----------
- MPLS core issues: escalate to Network Engineering
- Customer VRF issues: check with account team for recent changes
""",

    "cpu_spike_runbook.txt": """
RUNBOOK: High CPU Utilization on Network Device
Category: Device Health
Severity: P2/P3
Last Updated: 2024-01-10

OVERVIEW
--------
High CPU on routers/switches can cause control plane failures, BGP drops, and slow packet processing. Usually caused by routing instability, DDoS, or misconfiguration.

SYMPTOMS
--------
- CPU > 85% sustained for > 5 minutes
- BGP sessions starting to flap
- Slow CLI response on device
- Alert: cpu_utilization_pct threshold breached

STEP 1 — IDENTIFY TOP PROCESS
-------------------------------
   show processes cpu sorted | head 20
Common culprits:
  - BGP Scanner: routing table instability
  - IP Input: high traffic volume (possible DDoS)
  - OSPF/ISIS: topology changes / flapping
  - Crypto: if IPSec is configured

STEP 2 — BGP SCANNER HIGH
---------------------------
If BGP Scanner is top process:
  → Routing table churning due to BGP instability
  → Check for route flapping: show bgp flap-statistics
  → Apply route dampening if excessive:
    router bgp <ASN>
     bgp dampening

STEP 3 — IP INPUT HIGH (possible DDoS)
----------------------------------------
If IP Input process is high:
  → Check for traffic anomalies: show interfaces rate
  → Check for small packet flood (common in DDoS):
    show ip traffic | include fragments
  → Apply rate-limiting ACL on edge interfaces if attack confirmed

STEP 4 — OSPF/ISIS HIGH
-------------------------
  show ip ospf statistics
  show isis statistics
Excessive LSA flooding indicates topology instability.
  → Check for interface flapping upstream
  → Increase OSPF/ISIS timers to reduce churn

STEP 5 — REMEDIATION
----------------------
Option A — Process restart (non-disruptive for some):
  Consult vendor documentation before restarting
Option B — Redistribute load if possible (second supervisor)
Option C — Rate-limit control plane traffic:
  control-plane
   service-policy input COPP-POLICY

ESCALATION
----------
- Sustained > 15 mins: escalate to L2 Network Engineering
- DDoS suspected: notify Security Operations team
""",

    "dns_outage_runbook.txt": """
RUNBOOK: DNS Resolution Failure
Category: Services
Severity: P2/P3
Last Updated: 2024-01-25

OVERVIEW
--------
DNS outages affect all hostname-based services for customers. Can be caused by DNS server failure, misconfiguration, or network reachability issues.

SYMPTOMS
--------
- Customers report websites/services unreachable
- DNS queries timing out
- Alert: dns_outage on DNS forwarder device

STEP 1 — VERIFY DNS FAILURE
-----------------------------
From NOC workstation:
   nslookup google.com <affected-dns-server-ip>
If timeout:
  → DNS server not responding — check server health
If NXDOMAIN for valid domains:
  → DNS server responding but resolution failing (upstream issue)

STEP 2 — CHECK DNS SERVER REACHABILITY
----------------------------------------
   ping <dns-server-ip> repeat 100
If ping fails:
  → Network reachability issue — check routing to DNS server
  → Check if ACL is blocking UDP/TCP 53

STEP 3 — CHECK UPSTREAM RESOLVERS
-----------------------------------
DNS forwarder relies on upstream resolvers (e.g. 8.8.8.8 or internal root).
   dig @<dns-server-ip> google.com
   dig @8.8.8.8 google.com
Compare results. If upstream works but local doesn't:
  → Forwarder misconfiguration
  → Restart DNS service on forwarder

STEP 4 — FAILOVER TO SECONDARY DNS
------------------------------------
If primary DNS is confirmed down:
  → Update DHCP scope to point to secondary DNS server
  → Notify customers to flush DNS cache: ipconfig /flushdns

STEP 5 — REMEDIATION
----------------------
Option A — Restart DNS service on affected server
Option B — Point customers to secondary/tertiary DNS
Option C — If root cause is upstream: wait or use alternate resolver

ESCALATION
----------
- Server team for DNS server health
- Security team if DNS hijacking suspected
""",

    "interface_down_runbook.txt": """
RUNBOOK: Interface Down Investigation
Category: Physical/Link Layer
Severity: P1/P2
Last Updated: 2024-02-10

OVERVIEW
--------
An interface going down can cause immediate traffic loss. Investigation must be fast to minimize MTTR.

SYMPTOMS
--------
- Alert: interface_status = DOWN
- Logs: %LINK-3-UPDOWN changed state to down
- Customer traffic loss on that segment

STEP 1 — IDENTIFY INTERFACE
-----------------------------
   show interfaces GigabitEthernet0/1
Check:
  - Line protocol: up or down?
  - Input/output errors
  - Last clearing of counters

STEP 2 — PHYSICAL vs PROTOCOL
-------------------------------
Interface down + line protocol down:
  → Physical issue (cable, transceiver, far-end device)
Interface up + line protocol down:
  → Encapsulation mismatch, keepalive failure, or far-end config issue

STEP 3 — CHECK FAR END
------------------------
  - Can you reach far-end device via out-of-band?
  - Is far-end interface also showing down?
  - Check far-end logs for errors

STEP 4 — OPTICAL CHECK (for fiber links)
------------------------------------------
Check transceiver Rx power:
   show interfaces GigabitEthernet0/1 transceiver
If Rx power is low: refer to optical degradation runbook

STEP 5 — REMEDIATION
----------------------
Option A — Interface reset:
  interface GigabitEthernet0/1
   shutdown
   no shutdown
Option B — Replace cable/transceiver if physical fault
Option C — Reroute traffic to alternate interface
Option D — Dispatch field team for physical inspection

ESCALATION
----------
- Physical fault confirmed: field operations team
- Backbone link: senior network engineer immediately
""",
}


def generate_runbooks():
    for filename, content in RUNBOOKS.items():
        out = RUNBOOKS_DIR / filename
        out.write_text(content.strip())
    print(f"  Runbooks: {len(RUNBOOKS)} files → {RUNBOOKS_DIR}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("\n=== AI NOC Copilot — Synthetic Data Generator ===\n")

    print("[1/5] Generating network topology...")
    generate_topology()

    print("[2/5] Generating incidents...")
    incidents = generate_incidents(n=80)

    print("[3/5] Generating alerts...")
    generate_alerts(incidents)

    print("[4/5] Generating logs...")
    generate_logs(incidents, sample_size=40)

    print("[5/5] Writing runbooks...")
    generate_runbooks()

    print("\n✓ All synthetic data generated successfully.")
    print(f"  Data lives in: data/\n")


if __name__ == "__main__":
    main()
