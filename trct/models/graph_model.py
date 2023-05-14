from abc import ABC, abstractmethod
from collections import defaultdict, OrderedDict
from typing import Any, Dict, Iterable, List, Set, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import netgraph as ng
import numpy as np

from trct.models.base_model import BaseModel


class GraphModel(BaseModel):
    def __init__(self, src, dest):
        super().__init__(src, dest)

        self.nodes = OrderedDict()
        self.edges = set()

        self.node_to_index: dict[str, int] = defaultdict(lambda: len(self.nodes))

        self.counts = defaultdict(lambda: defaultdict(lambda: 0))

        self.node_out_counts = defaultdict(lambda: 0)
        self.node_edge_count_fraction: list[float] = [0.0]

        idx = self.node_to_index[self.u]
        self.nodes[self.u] = idx

    def log(self, ts, hops):
        super().log(ts)

        curr = self.u

        for node in hops:
            idx = self.node_to_index[node]
            self.nodes[node] = idx

            self.edges.add((curr, node))

            self.node_out_counts[curr] += 1
            self.counts[curr][node] += 1
            curr = node

        self.nef = len(self.nodes) / len(self.edges)

    def score(self, hops):
        curr = self.u
        score = []

        for node in hops:
            node_transition_prob = self.counts[curr][node] / self.node_out_counts[curr]
            global_transition_prob = self.counts[self.u][node] / self.n
            score.append([node_transition_prob, global_transition_prob, self.nef])
            curr = node

        return score

    def __repr__(self):
        return f"{self.u} -> {self.v} (#{self.n} Graph)"

    def to_matrix(self) -> np.ndarray:
        """Return the transition matrix of the model."""
        mat = np.zeros((len(self.nodes), len(self.nodes)))
        for i, u in enumerate(self.nodes):
            if not self.node_out_counts[u]:
                continue
            for j, v in enumerate(self.nodes):
                mat[i, j] = self.counts[u][v] / self.n
        return mat

    def to_graph(self) -> nx.Graph:
        return nx.DiGraph(self.to_matrix(), node_names=self.nodes)

    def get_data(self) -> dict[str, list[Any]]:
        """Return the model data."""
        return {"nef": self.node_edge_count_fraction}

    def plot(self, axes: plt.Axes, *args, **kwargs) -> None:
        """Plot the model on specified axis."""
        axes.plot(self.tss, self.node_edge_count_fraction, *args, **kwargs)
        axes.set_xlabel("Time")
        axes.set_ylabel("Node/Edge Fraction")

    def plot_graph(
        self, axes: plt.Axes, tier_mapping=defaultdict(lambda: "o"), *args, **kwargs
    ) -> None:
        """Plot the model on specified axis."""
        # TODO: add legends https://netgraph.readthedocs.io/en/latest/sphinx_gallery_output/plot_18_legends.html#sphx-glr-sphinx-gallery-output-plot-18-legends-py
        graph = self.to_graph()

        colors = {}
        for node, i in self.node_to_index.items():
            if node == self.u:
                colors[i] = "r"
            elif node == self.v:
                colors[i] = "b"
            else:
                colors[i] = "black"

        # print(colors)
        tmp = ng.Graph(
            graph,
            layout="dot",
            arrows=True,
            weighted=True,
            # node_color={x:colors[x] for x in g.nodes}, # TODO: this based on src/dest/normal
            node_edge_color=colors,
            node_labels={x: i for i, x in enumerate(graph.nodes)},
            node_shape={x: tier_mapping[x] for x in graph.nodes},  # so^>v<dph8
            edge_cmap="PRGn",
            axes=axes,
        )

    @property
    def nef(self) -> float:
        return self.node_edge_count_fraction[-1]

    @nef.setter
    def nef(self, value):
        self.node_edge_count_fraction.append(value)
