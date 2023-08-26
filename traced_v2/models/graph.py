"""Graph model for trace modelling in network graph."""
from collections import Counter, defaultdict
from typing import Any, Hashable

import numpy as np
from matplotlib.figure import Figure
from matplotlib.pyplot import Axes

from traced_v2.models.base_model import BaseModel, Visual

# pylint: disable=too-many-arguments, fixme, line-too-long, too-many-instance-attributes, invalid-name


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


class ForgettingGraph(Graph):
    """Graph model for trace modelling in network graph with forgetting."""

    class Queue:
        """Queue with a max size."""

        def __init__(self, max_size: int | None = None):
            self._queue = []
            self.max_size = max_size

        def add(self, item: Any) -> None:
            """Add an item to the queue."""
            if self.max_size and len(self._queue) == self.max_size:
                self._queue.pop(0)
            self._queue.append(item)

        @property
        def data(self) -> list[Any]:
            """Get the data in the queue."""
            return self._queue

    def __init__(self, max_size: int = 50):
        super().__init__()
        self.max_size = max_size
        self._edge_queues = defaultdict(lambda: self.Queue(max_size))

    def get_counts(self) -> dict[Hashable, dict[Hashable, int]]:
        """Get the counts of edges."""
        counts = defaultdict(lambda: defaultdict(lambda: 1))
        for src, dests in self._edge_queues.items():
            for dest, count in Counter(dests.data).items():
                counts[src][dest] += count
        return counts

    def get_prob(self, src: str, dest: str) -> float:
        """Get the probability of an edge from src to dest."""
        counts = self.get_counts()
        return counts[src][dest] / (1 + sum(counts[src].values()))

    def add_edge(self, src: str, dest: str):
        """Add an edge from src to dest."""
        self.nodes.add(src)
        self.nodes.add(dest)
        self._edge_queues[src].add(dest)


class GraphModel(BaseModel, Visual):
    """Graph model for trace modelling in network graph."""

    _GLOBAL_GRAPH = Graph()
    _GLOBAL_FORGETTING_GRAPH = ForgettingGraph()

    def __init__(
        self,
        src: str,
        dest: str,
        local: bool = False,
        forgetting: bool = False,
        parent: BaseModel | None = None,
    ) -> None:
        super().__init__(
            src, dest, subscription=parent.subscription if parent else None
        )
        self.graph: Graph = (
            (Graph() if local else GraphModel._GLOBAL_GRAPH)
            if not forgetting
            else (ForgettingGraph() if local else GraphModel._GLOBAL_FORGETTING_GRAPH)
        )
        self.probs = []

    def log(self, ts: int, observed_value: list[Hashable]):
        """Log a trace."""
        super().log_timestamp(ts)
        probs = []
        if not observed_value:
            self.probs.append(0)
            return
        current = observed_value[0]
        for item in observed_value[1:]:
            probs.append(np.log1p(self.graph.get_prob(current, item)))
            self.graph.add_edge(current, item)

        self.probs.append(np.sum(probs))

    def to_dict(self) -> dict[str, list[Any]]:
        """Convert the model statistics to a dictionary."""
        return {
            "probs": self.probs,
        }

    def plot(self, ax: Figure | Axes | None = None):
        pass  # TODO: implement
