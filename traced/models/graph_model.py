from abc import ABC, abstractmethod
from collections import defaultdict, OrderedDict
import copy
from hashlib import sha1
from typing import Any, Dict, Iterable, List, Set, Tuple, Optional
from collections import Counter


import matplotlib.pyplot as plt
import networkx as nx
import netgraph as ng
import pandas as pd
import numpy as np
import json

from traced.models.base_model import BaseModel
from traced.models.multinomial_model import MultinomialModel
from traced.models.normal_model import NormalModel


class GraphModel(BaseModel):
    def __init__(self, src, dest):
        super().__init__(src, dest)

        self.nodes = OrderedDict()
        self.edges = set()

        self.node_to_index: dict[str, int] = defaultdict(lambda: len(self.nodes))
        self.counts = defaultdict(lambda: defaultdict(lambda: 0))
        self.node_out_counts = defaultdict(lambda: 0)
        self.node_in_counts = defaultdict(lambda: 0)
        self.node_depth_counts: Dict[int, Counter[str]] = defaultdict(lambda: Counter())

        self.path_probs = MultinomialModel(self.u, self.v)
        idx = self.node_to_index[self.u]
        self.nodes[self.u] = idx

        self.local_probs: list[float] = [0]
        self.global_probs: list[float] = [0]
        self.weighted_probs: list[float] = [0]
        self.npep: list[float] = [0]

        self.ctr = 0
        self._empty_counter: list[int] = [0]
        self.unique_paths: list[str] = [""]

        self.local_anomaly_model = NormalModel(
            self.u, self.v, one_sided=True, sigma_factor=4
        )
        self.global_anomaly_model = NormalModel(
            self.u, self.v, one_sided=True, sigma_factor=4
        )
        self.weighted_anomaly_model = NormalModel(
            self.u, self.v, one_sided=True, sigma_factor=4
        )
        self.node_prob_edge_prob = NormalModel(
            self.u, self.v, one_sided=True, sigma_factor=4
        )

    def log(self, ts, hops):
        super().log(ts)

        curr = self.u

        local_prob: list[float] = []
        global_prob: list[float] = []
        wighted_prob: list[float] = []
        path_probs: list[float] = []
        npep: list[float] = []

        self.node_in_counts[curr] += 1

        for i, node in enumerate(hops):
            self.ctr += 1
            idx = self.node_to_index[node]
            self.nodes[node] = idx
            self.edges.add((curr, node))
            self.counts[curr][node] += 1
            self.node_in_counts[node] += 1
            self.node_out_counts[curr] += 1
            self.node_depth_counts[i][curr] += 1

            # probability of node being in ith position
            npep.append(
                self.node_depth_counts[i][curr] / self.node_depth_counts[i].total()
            )

            # probability of transition from ith node to i+1th node
            local_prob.append(
                (1 + self.counts[curr][node]) / (1 + self.node_out_counts[curr])
            )

            # probabilty of transition from ith node to i+1th node in the path
            path_probs.append(local_prob[-1] * npep[-1])

            # global and weighted probs
            global_prob.append((1 + self.counts[curr][node]) / (self.ctr + 1))
            wighted_prob.append(global_prob[-1] * local_prob[-1])

            curr = node

        self.local_prob = np.prod(local_prob)  # type: ignore
        self.global_prob = np.prod(global_prob)  # type: ignore
        self.weighted_prob = np.prod(wighted_prob)  # type: ignore

        self.npep.append((np.prod(npep)))  # type: ignore

        self.node_prob_edge_prob.log(ts, -np.log(np.prod(path_probs)))
        self.local_anomaly_model.log(ts, -np.log(self.local_prob))
        # self.global_anomaly_model.log(ts, -np.log(self.global_prob))
        # self.weighted_anomaly_model.log(ts, -np.log(self.weighted_prob))

        return bool(self.node_prob_edge_prob.anomaly), path_probs, float(np.mean(path_probs)) 

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
        return {
            "local_probs": self.local_probs,
            "global_probs": self.global_probs,
            "weighted_probs": self.weighted_probs,
            "path_prob": self.weighted_probs,
            "node_prob": self.npep,
            "paths": self.unique_paths,
        }

    def plot(self, axes: Optional[plt.Axes] = None, *args, **kwargs) -> None:  # type: ignore
        """Plot the model on specified axis."""
        if axes is None:
            axes: plt.Axes = plt.gca()

        df = self.to_frame()
        df["local_probs"].plot(
            ax=axes, label="local prob", c="red", alpha=0.5, *args, **kwargs
        )
        df["node_prob"].plot(
            ax=axes, label="node prob", c="blue", alpha=0.5, *args, **kwargs
        )

        axes.set_xlabel("Time")
        axes.set_ylabel("Probability")

    def plot_graph(
        self, axes: Optional[plt.Axes]=None, tier_mapping=defaultdict(lambda: "o"), *args, **kwargs # type: ignore
    ) -> None:
        """Plot the model on specified axis."""
        # TODO: add legends https://netgraph.readthedocs.io/en/latest/sphinx_gallery_output/plot_18_legends.html#sphx-glr-sphinx-gallery-output-plot-18-legends-py
        graph = self.to_graph()
        
        if axes is None:
            axes: plt.Axes = plt.gca()

        colors = {}
        for node, i in self.node_to_index.items():
            if node == self.u:
                colors[i] = "r"
            elif node == self.v:
                colors[i] = "b"
            else:
                colors[i] = "black"

        tmp = (
            ng.Graph(
                graph,
                layout="dot",
                arrows=True,
                weighted=True,
                # node_color={x:colors[x] for x in g.nodes}, # TODO: this based on src/dest/normal
                node_edge_color=colors,
                node_labels={x: i for i, x in enumerate(graph.nodes)},
                node_shape={x: tier_mapping[x] for x in graph.nodes},  # so^>v<dph8
                edge_cmap="RdYlGn",
                node_layout=kwargs["node_layout"],
                axes=axes,
            )
            if "node_layout" in kwargs
            else ng.Graph(
                graph,
                layout="dot",
                arrows=True,
                weighted=True,
                # node_color={x:colors[x] for x in g.nodes}, # TODO: this based on src/dest/normal
                node_edge_color=colors,
                node_labels={x: i for i, x in enumerate(graph.nodes)},
                node_shape={x: tier_mapping[x] for x in graph.nodes},  # so^>v<dph8
                edge_cmap="RdYlGn",
                axes=axes,
            )
        )

    def to_json(self, ip_to_geo_mapper, threshold=2) -> str:
        """Return the learned graph as json."""
        data = {}
        data["node_mapping"] = {i: node for node, i in self.node_to_index.items()}

        depths = defaultdict(lambda: 0)  # default value is current
        g = self.to_graph()
        g.nodes[0]["d"] = 0

        for src, items in nx.algorithms.traversal.bfs_successors(g, 0):
            depth = depths[src]
            for dest in items:
                depths[dest] = depth + 1
                g.nodes[dest]["d"] = depths[dest]
        layout = nx.layout.multipartite_layout(g, subset_key="d")

        Q = [
            node
            for node in g.nodes
            if data["node_mapping"][node] not in ip_to_geo_mapper
        ]

        while Q:
            node = Q.pop()
            loc = [0.0, 0.0]
            has_loc = 0
            for neighbor in g.neighbors(node):
                ip = data["node_mapping"][neighbor]
                if ip in ip_to_geo_mapper:
                    has_loc += 1
                    loc[0] += ip_to_geo_mapper[ip][0]
                    loc[1] += ip_to_geo_mapper[ip][1]
            thr = has_loc / len(list(g.neighbors(node)))
            thr = has_loc
            print(thr)
            if thr >= min(threshold, 2 / len(list(g.neighbors(node)))):
                loc[0] = loc[0] / has_loc
                loc[1] = loc[1] / has_loc
                ip_to_geo_mapper[data["node_mapping"][node]] = loc
            else:
                Q.append(node)

        final_pos = {
            n: list(ip_to_geo_mapper[data["node_mapping"][n]] + 2e-3 * layout[n])
            for n in g.nodes()
        }

        data["edges"] = []

        for u, v in self.edges:
            edge = {}
            edge["src"] = u
            edge["dest"] = v
            edge["prob"] = self.counts[u][v] / sum(self.counts[u].values())

            data["edges"].append(edge)
        data["node_pos"] = final_pos
        data["src"] = self.u
        data["dest"] = self.v

        return json.dumps(data)

    @property
    def global_prob(self):
        return self.global_probs[-1]

    @global_prob.setter
    def global_prob(self, value):
        self.global_probs.append(value)

    @property
    def local_prob(self):
        return self.local_probs[-1]

    @local_prob.setter
    def local_prob(self, value):
        self.local_probs.append(value)

    @property
    def path(self):
        return self.unique_paths

    @path.setter
    def path(self, value):
        self.unique_paths.append(value)

    @property
    def weighted_prob(self):
        return self.weighted_probs[-1]

    @weighted_prob.setter
    def weighted_prob(self, value):
        self.weighted_probs.append(value)
