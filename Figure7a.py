#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 2 phương án là 2 đồ thị riêng biệt, chưa đè lên nhau như paper
import json
import networkx as nx
import matplotlib.pyplot as plt

def plot_placement(json_file, out_file):
    # Load placement JSON
    with open(json_file, "r") as f:
        placement = json.load(f)

    # placement format: {"module": ["deviceId", ...], ...}
    # Build bipartite graph: services ↔ devices
    G = nx.Graph()

    for module, devices in placement.items():
        for d in devices:
            G.add_node("D{}".format(d), bipartite=0)
            G.add_node(module, bipartite=1)
            G.add_edge(module, "D{}".format(d))

    # Split by bipartite sets
    modules = [n for n in G.nodes() if not n.startswith("D")]
    devices = [n for n in G.nodes() if n.startswith("D")]

    pos = {}
    pos.update((n, (0, i)) for i, n in enumerate(modules))
    pos.update((n, (1, i)) for i, n in enumerate(devices))

    plt.figure(figsize=(8, 6))
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_size=1500,
        node_color=["#a6bddb" if n in modules else "#e34a33" for n in G.nodes()],
        font_size=8,
    )
    plt.title("Service placement: {}".format(json_file))
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()


if __name__ == "__main__":
    # Example usage
    plot_placement("PlacementPartition.json", "placement_partition.pdf")
    plot_placement("PlacementILP.json", "placement_ilp.pdf")
