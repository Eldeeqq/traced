"""Graph model for trace modelling in network graph."""
from collections import defaultdict
from datetime import datetime
from typing import Any, Hashable

import networkx as nx
import pydantic
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.pyplot import Axes

from traced_v2.models.base_model import BaseModel, Visual
from traced_v2.models.normal import NormalModel
from traced_v2.models.queue import Queue
from traced_v2.utils import add_prefix, create_hash

# pylint: disable=too-many-arguments, fixme, line-too-long, too-many-instance-attributes, invalid-name


class GraphModelOutput(pydantic.BaseModel):
    """Output of the Graph model."""

    is_anomaly: bool
    segment_probs: list[float]
    segment_neg_log_probs: list[float]
    path_prob: float
    path_neg_log_prob: float


class Graph:
    """Graph model for trace modelling in network graph."""

    def __init__(self):
        self.edges = defaultdict(lambda: defaultdict(lambda: 1))
        self.nodes = set()

    def get_prob(self, src: Hashable, dest: Hashable) -> float:
        """Get the probability of an edge from src to dest."""
        return self.edges[src][dest] / (1 + sum(self.edges[src].values()))

    def add_edge(self, src: Hashable, dest: Hashable):
        """Add an edge from src to dest."""
        self.nodes.add(src)
        self.nodes.add(dest)
        self.edges[src][dest] += 1

    def get_edges(self) -> dict[Hashable, dict[Hashable, int]]:
        """Return the edges."""
        return dict(self.edges)

    def to_graph(self) -> nx.Graph:
        data = {
            src: {
                dest: {"weight": v / sum(value.values())} for dest, v in value.items()
            }
            for src, value in self.get_edges().items()
        }
        return nx.from_dict_of_dicts(data, create_using=nx.DiGraph)  # type: ignore


class ForgettingGraph(Graph):
    """Graph model for trace modelling in network graph with forgetting."""

    def __init__(self, max_size: int = 500):
        super().__init__()
        self.max_size = max_size
        self.edge_queues = defaultdict(lambda: Queue(max_size))

    def get_counts(self) -> dict[Hashable, dict[Hashable, int]]:
        """Get the counts of edges."""
        counts = {node: queue.data for node, queue in self.edge_queues.items()}
        return counts  # type: ignore

    def get_prob(self, src: str, dest: str) -> float:
        """Get the probability of an edge from src to dest."""
        counts = self.edge_queues[src].data
        return counts[dest] / (1 + sum(counts.values()))

    def add_edge(self, src: str, dest: str):
        """Add an edge from src to dest."""
        self.nodes.add(src)
        self.nodes.add(dest)
        self.edge_queues[src].add(dest)

    def get_edges(self) -> dict[Hashable, dict[Hashable, int]]:
        """Return the edges."""
        return self.get_counts()


class GraphModel(BaseModel, Visual):
    """Graph model for trace modelling in network graph."""

    REGISTRY = {
        "global": Graph(),
        "global_forgetting": ForgettingGraph(),
        "global_ip": Graph(),
        "global_forgetting_ip": ForgettingGraph(),
        "global_as": Graph(),
        "global_forgetting_as": ForgettingGraph(),
    }  # Registry for shared graphs

    REGISTRY_KEYS = list(REGISTRY.keys())

    @classmethod
    def get_or_create_subscription(
        cls, forgetting: bool = True, local: bool = True
    ) -> str:
        if not local:
            return "global_forgetting" if forgetting else "global"

        key = create_hash(str(datetime.now()))
        cls.REGISTRY[key] = ForgettingGraph() if forgetting else Graph()
        return key

    def __init__(
        self,
        src: str,
        dest: str,
        graph_subscription: None | str = None,
        parent: BaseModel | None = None,
    ) -> None:
        super().__init__(
            src, dest, subscription=parent.subscription if parent else None
        )
        graph_subscription = graph_subscription or "global_forgetting"
        self.graph: Graph = GraphModel.REGISTRY[graph_subscription]
        self.probs = []
        self.prob_model: NormalModel = NormalModel(
            src,
            dest,
            parent=self,
            one_sided=True,
            mu_0=-2,
            sigma_0=1,
            alpha_0=1,
            beta_0=1,
        )

    def log(self, ts: int, observed_value: list[Hashable]) -> GraphModelOutput:
        """Log a trace."""
        super().log_timestamp(ts)
        probs = []
        log_prob = []

        current = observed_value[0]
        for item in observed_value[1:]:
            p = self.graph.get_prob(current, item)
            log_prob.append(-np.log1p(p))
            probs.append(p)
            self.graph.add_edge(current, item)
            current = item

        self.probs.append(np.prod(probs))
        path_neg_log_prob: float = np.sum(log_prob)  # type: ignore
        output = self.prob_model.log(ts, path_neg_log_prob)  # type: ignore

        return GraphModelOutput(
            is_anomaly=output.is_anomaly,
            segment_probs=probs,
            segment_neg_log_probs=log_prob,
            path_prob=self.probs[-1],
            path_neg_log_prob=path_neg_log_prob,
        )

    def to_dict(self) -> dict[str, list[Any]]:
        """Convert the model statistics to a dictionary."""
        return {
            "path_prob": self.probs,
            **add_prefix("log_prob", self.prob_model.to_dict()),
        }

    def plot(
        self,
        ax: Figure | Axes | None = None,
        layout: dict[Hashable, tuple[float, float]] | None = None,
    ):
        ax = ax or plt.gca()

        graph = self.graph.to_graph()

        layout = layout or nx.random_layout(graph, seed=42)  # type: ignore

        nx.draw(graph, layout, with_labels=True, ax=ax)
