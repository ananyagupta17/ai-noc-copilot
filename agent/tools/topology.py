"""
MCP Tool: Topology queries
Reads topology.json and uses NetworkX for graph traversal.
Helps agent understand device dependencies and blast radius.
"""

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import TOPOLOGY_DIR

import networkx as nx

# Load topology once at import time — it doesn't change at runtime
_TOPOLOGY_FILE = TOPOLOGY_DIR / "topology.json"
_graph = None


def _get_graph() -> nx.Graph:
    global _graph
    if _graph is None:
        data = json.loads(_TOPOLOGY_FILE.read_text())
        _graph = nx.Graph()
        for node in data["nodes"]:
            _graph.add_node(node["id"], **{k: v for k, v in node.items() if k != "id"})
        for edge in data["edges"]:
            _graph.add_edge(edge["source"], edge["target"],
                            **{k: v for k, v in edge.items() if k not in ("source", "target")})
    return _graph


def get_device_info(device_id: str) -> dict:
    """
    Get metadata for a specific device node.
    """
    G = _get_graph()
    if device_id not in G:
        return {"error": f"Device {device_id} not found in topology"}
    return {"device_id": device_id, **G.nodes[device_id]}


def get_neighbors(device_id: str) -> dict:
    """
    Get all directly connected devices.
    Agent uses this to identify what's immediately upstream/downstream.
    """
    G = _get_graph()
    if device_id not in G:
        return {"error": f"Device {device_id} not found"}

    neighbors = []
    for nbr in G.neighbors(device_id):
        edge_data = G.edges[device_id, nbr]
        neighbors.append({
            "device": nbr,
            "type": G.nodes[nbr].get("type"),
            "region": G.nodes[nbr].get("region"),
            "link_type": edge_data.get("link_type"),
            "capacity_gbps": edge_data.get("capacity_gbps"),
        })

    return {
        "device": device_id,
        "neighbor_count": len(neighbors),
        "neighbors": neighbors
    }


def get_blast_radius(device_id: str, hops: int = 2) -> dict:
    """
    Find all devices within N hops of the affected device.
    Critical for understanding how far an outage can cascade.
    """
    G = _get_graph()
    if device_id not in G:
        return {"error": f"Device {device_id} not found"}

    # BFS up to `hops` levels
    affected = {}
    for node, depth in nx.single_source_shortest_path_length(G, device_id, cutoff=hops).items():
        if node == device_id:
            continue
        affected[node] = {
            "hops_away": depth,
            "type": G.nodes[node].get("type"),
            "region": G.nodes[node].get("region"),
        }

    # Group by region for readability
    regions_affected = list({v["region"] for v in affected.values()})

    return {
        "source_device": device_id,
        "hops_analyzed": hops,
        "total_affected_devices": len(affected),
        "regions_affected": regions_affected,
        "affected_devices": affected
    }


def get_path_between(device_a: str, device_b: str) -> dict:
    """
    Find the shortest path between two devices.
    Useful for tracing where in the network the fault lies.
    """
    G = _get_graph()
    try:
        path = nx.shortest_path(G, device_a, device_b)
        return {
            "source": device_a,
            "destination": device_b,
            "path": path,
            "hop_count": len(path) - 1
        }
    except nx.NetworkXNoPath:
        return {"error": f"No path found between {device_a} and {device_b}"}
    except nx.NodeNotFound as e:
        return {"error": str(e)}


def get_region_devices(region: str) -> dict:
    """
    List all devices in a given region with their types.
    """
    G = _get_graph()
    devices = [
        {"device_id": n, "type": G.nodes[n].get("type"), "ip": G.nodes[n].get("ip")}
        for n in G.nodes
        if G.nodes[n].get("region") == region
    ]
    return {
        "region": region,
        "device_count": len(devices),
        "devices": devices
    }