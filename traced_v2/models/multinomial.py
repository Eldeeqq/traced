"""Module for the MultinomialModel class."""

from collections import defaultdict
from typing import Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from traced_v2.models.base_model import BaseModel, Visual

# pylint: disable=too-many-arguments, fixme, line-too-long, too-many-instance-attributes, invalid-name


class MultinomialModel(BaseModel, Visual):
    """Univariate Multinomial model with bayesian updating."""

    def __init__(
        self, src: str, dest: str, *args, gamma: float = 1.0, **kwargs
    ) -> None:
        super().__init__(src, dest, *args, **kwargs)

        self.seen_categories: set[Any] = set()
        self.category_counts: dict[Any, float] = defaultdict(lambda: 1.0)
        self.category_probs: dict[Any, float] = {}

        self.observed_variables: list[Any] = []
        self.probabilities: list[float] = []
        self.divergence: list[float] = []

        self.gamma: float = gamma
        self.counter: int = 1

    def log(self, ts: int, observed_value: Any) -> tuple[float, float, float]:
        """Log a new observation and return whether it is an anomaly."""
        super().log_timestamp(ts)
        self.seen_categories.add(observed_value)
        self.counter += self.gamma
        self.category_counts[observed_value] += self.gamma
        prior = (
            self.probabilities[-1]
            if self.probabilities
            else 1 / len(self.seen_categories)
        )
        posterior_prob = (self.category_counts[observed_value] + 1) / (
            self.counter + len(self.seen_categories)
        )
        self.category_probs[observed_value] = posterior_prob
        self.probabilities.append(posterior_prob)
        self.observed_variables.append(observed_value)

        d_kl = -np.sum(
            [
                p * (np.log1p(p / posterior_prob))
                for p in self.category_probs.values()
                if p != observed_value
            ]
        )
        self.divergence.append(d_kl)

        return posterior_prob, d_kl, prior - self.probabilities[-1]

    def to_dict(self) -> dict[str, list[Any]]:
        return {
            "observed_variables": self.observed_variables,
            "probabilities": self.probabilities,
        }

    def plot(self, ax: Figure | Axes | None = None):
        df = self.to_frame()
        df.plot(ax=ax, y="probabilities", c="observed_variables", kind="scatter")
