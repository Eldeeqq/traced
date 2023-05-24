from abc import ABC, abstractmethod
from collections import defaultdict, OrderedDict
import copy
from hashlib import sha1
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

        idx = self.node_to_index[self.u]
        self.nodes[self.u] = idx

        self.local_probs: list[float] = [0]
        self.global_probs: list[float] = [0]
        self.weighted_probs: list[float] = [0]
        self.ctr = 0
        self._empty_counter = [0]
        self.unique_paths = []

    def log(self, ts, hops):
        super().log(ts)

        curr = self.u

        local_prob: list[float] = []
        global_prob: list[float] = []
        wighted_prob: list[float] = []

        self.path = sha1('-'.join(map(str, hops)).encode('utf-8')).hexdigest( ) # type: ignore

        for node in hops:
            self.ctr += 1
            idx = self.node_to_index[node]
            self.nodes[node] = idx

            self.edges.add((curr, node))

            self.node_out_counts[curr] += 1
            self.counts[curr][node] += 1

            local_prob.append((1+self.counts[curr][node]) / (1+self.node_out_counts[curr])) 

            global_prob.append(
                  (1+self.counts[curr][node])/(self.ctr+1)
            )
            wighted_prob.append(global_prob[-1]*local_prob[-1])

            # global_prob.append((1+self.counts[self.u][node]) / (1+self.n/len(self.edges)))
            curr = node

        self.local_prob = np.prod(local_prob) # type: ignore
        self.global_prob = np.prod(global_prob)  # type: ignore
        self.weighted_prob = np.min(wighted_prob) # type: ignore


    def score(self, hops):
        curr = self.u
        local_prob: list[float] = []
        global_prob: list[float] = []
        wighted_prob: list[float] = []

        for node in hops:
            local_prob.append(self.counts[curr][node] / self.node_out_counts[curr]) 
            global_prob.append(self.counts[curr][node] / self.n)
            # score.append([node_transition_prob, global_transition_prob])
            global_prob.append(global_prob[-1]*local_prob[-1])
            
            curr = node
        self.local_probs = np.prod(local_prob) # type: ignore
        self.global_probs = np.prod(global_prob) # type: ignore
        return local_prob, global_prob

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
            'local_probs': self.local_probs,
            'global_probs': self.global_probs,
            'weighted_probs': self.weighted_probs,
        }

    def plot(self, axes: plt.Axes, *args, **kwargs) -> None:
        """Plot the model on specified axis."""
        axes.plot( self.global_probs, label='global prob', *args, **kwargs)
        axes.plot( self.local_probs, label='makrov prob', *args, **kwargs)
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
            edge_cmap="RdYlGn",
            node_layout=kwargs["node_layout"],
            axes=axes,
        ) if 'node_layout' in kwargs else ng.Graph(
            graph,
            layout="dot",
            arrows=True,
            weighted=True,
            # node_color={x:colors[x] for x in g.nodes}, # TODO: this based on src/dest/normal
            node_edge_color=colors,
            node_labels={x: i for i, x in enumerate(graph.nodes)},
            node_shape={x: tier_mapping[x] for x in graph.nodes},  # so^>v<dph8
            edge_cmap="RdYlGn",
            axes=axes
            )
    
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
        self.unique_paths.append((self.ts, value))

    @property
    def weighted_prob(self):
        return self.weighted_probs[-1]
    
    @weighted_prob.setter
    def weighted_prob(self, value):
        self.weighted_probs.append(value)
