from pathlib import Path

from scripts.validate_cluster import load_topology, validate_topology


def test_cluster_topology_passes_static_validation() -> None:
    topology = load_topology(Path("infra/cluster_topology.yaml"))

    result = validate_topology(topology, expected_nodes=4)

    assert result["passed"], result["errors"]
    assert result["private_ips"] == [
        "10.42.0.10",
        "10.42.0.11",
        "10.42.0.12",
        "10.42.0.13",
    ]
    assert {22, 29500, 29501}.issubset(result["ports"])
