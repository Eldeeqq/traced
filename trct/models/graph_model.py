from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Set, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from trct.models.base_model import BaseModel


class GraphModel(BaseModel):
    def __init__(self, src, dest):
        super().__init__(src, dest)

        self.nodes = set()
        self.edges = set()

        self.counts = defaultdict(lambda: defaultdict(lambda: 0))

        self.node_out_counts = defaultdict(lambda: 0)
        self.node_edge_count_fraction: list[float] = [0.0]

    def log(self, ts, hops):
        super().log(ts)
        curr = self.u

        for node in hops:
            self.nodes.add(node)
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
            score.append([node_transition_prob, global_transition_prob])
            curr = node

        return score

    def __repr__(self):
        return f"{self.u} -> {self.v} (#{self.n} Graph)"

    @abstractmethod
    def to_matrix(self) -> np.ndarray:
        pass

    @abstractmethod
    def to_graph(self) -> nx.Graph:
        pass

    @abstractmethod
    def get_data(self) -> dict[str, list[Any]]:
        """Return the model data."""

    @abstractmethod
    def plot(self, axes: plt.Axes, *args, **kwargs) -> None:
        """Plot the model on specified axis."""

    @abstractmethod
    def plot_graph(self, axes: plt.Axes, *args, **kwargs) -> None:
        """Plot the model on specified axis."""

    @property
    def nef(self):
        return self.node_edge_count_fraction[-1]

    @nef.setter
    def nef(self, value):
        self.node_edge_count_fraction.append(value)
