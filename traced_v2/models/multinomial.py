"""Module for the MultinomialModel class."""

import itertools
from collections import defaultdict
from typing import Any, Hashable

import matplotlib.pyplot as plt
import numpy as np
import pydantic

from matplotlib import pyplot as plt
from matplotlib import rcParams
from matplotlib.axes import Axes

from traced_v2.models.base_model import BaseModel, Visual
from traced_v2.models.normal import NormalModel
from traced_v2.models.queue import Queue


# pylint: disable=too-many-arguments, fixme, line-too-long, too-many-instance-attributes, invalid-name
class MultinomialModelOutput(pydantic.BaseModel):
    """Output of the Multinomial model."""

    variance: float
    probability: float
    observed_value: Hashable
    anomaly: bool


class MultinomialModel(BaseModel, Visual):
    """Univariate Multinomial model with bayesian updating."""

    def __init__(
        self, src: str, dest: str, parent: BaseModel | None = None, gamma: float = 1.0
    ) -> None:
        super().__init__(
            src, dest, subscription=parent.subscription if parent else None
        )

        self.seen_categories: set[Hashable] = set()
        self.category_counts: dict[Hashable, float] = defaultdict(lambda: 1.0)
        self.category_probs: dict[Hashable, float] = {}

        self.observed_variables: list[Hashable] = []
        self.probabilities: list[float] = []
        self.variance: list[float] = []
        self.kl_divergence_model: NormalModel = NormalModel(src, dest, parent=self)

        self.gamma: float = gamma
        self.counter: float = 1.0

    def log(self, ts: int, observed_value: Hashable) -> MultinomialModelOutput:
        """Log a new observation and return whether it is an anomaly."""
        super().log_timestamp(ts)
        self.seen_categories.add(observed_value)
        self.counter += self.gamma
        self.category_counts[observed_value] += self.gamma

        posterior_prob = (self.category_counts[observed_value] + 1) / (
            self.counter + len(self.seen_categories)
        )
        a = self.category_counts[observed_value]
        b = self.counter - a
        var = (a * b) / ((self.counter**2) * (self.counter + 1))

        self.variance.append(var)
        self.category_probs[observed_value] = posterior_prob

        kl = np.mean([x * np.log(posterior_prob) for x in self.category_probs.values()])
        out = self.kl_divergence_model.log(ts, float(kl))
        self.probabilities.append(posterior_prob)
        self.observed_variables.append(observed_value)

        return MultinomialModelOutput(
            probability=posterior_prob,
            observed_value=observed_value,
            variance=var,
            anomaly=out.is_anomaly,
        )

    def to_dict(self) -> dict[str, list[Any]]:
        return {
            "observed_variables": self.observed_variables,
            "variance": self.variance,
            "probabilities": self.probabilities,
            "anomalies": self.kl_divergence_model.anomalies[1:],
        }

    def plot(self, ax: Axes | None = None, **kwargs) -> None:
        df = self.to_frame()

        ax = ax or plt.gca()

        clist = rcParams["axes.prop_cycle"]
        cgen = itertools.cycle(clist)
        title = f'Probability of {kwargs.get("kind", "classes")}'
        ax.set_title(title)
        for i, gdf in df.groupby("observed_variables"):
            # if gdf.shape[0] > 2:
                # color= next(cgen)
                lower_bound = gdf["probabilities"] - 3 * gdf["variance"].apply(np.sqrt)
                upper_bound = gdf["probabilities"] + 3 * gdf["variance"].apply(np.sqrt)
                ax.fill_between(gdf.index, lower_bound, upper_bound, alpha=0.15)  # type: ignore
                gdf["probabilities"].plot(ax=ax, label=i, marker="None" if gdf.shape[0] > 2 else "o")
            # else:
            #     gdf["probabilities"].plot(ax=ax, label=i, marker="o")
        df[df["anomalies"]].plot(
            marker="x",
            linestyle="None",
            y="probabilities",
            ax=ax,
            c="black",
            label="anomaly",
        )

        ax.legend().remove()


class ForgettingMultinomialModel(MultinomialModel):
    """Multinomial model with forgetting."""

    def __init__(
        self, src: str, dest: str, parent: BaseModel | None = None, gamma: float = 1
    ) -> None:
        super().__init__(src=src, dest=dest, parent=parent, gamma=gamma)
        self.queue = Queue(max_size=1000)
        self.category_counts = self.queue.counts_queue

    def log(self, ts: int, observed_value: Hashable) -> MultinomialModelOutput:
        super().log_timestamp(ts)
        self.seen_categories.add(observed_value)
        self.queue.add(observed_value)
        self.counter = self.gamma * sum(self.category_counts.values())

        posterior_prob = (self.category_counts[observed_value] + 1) / (
            self.counter + len(self.seen_categories)
        )
        a = self.category_counts[observed_value]
        b = self.counter - a
        var = (a * b) / ((self.counter**2) * (self.counter + 1))

        self.variance.append(var)
        self.category_probs[observed_value] = posterior_prob
        self.probabilities.append(posterior_prob)
        self.observed_variables.append(observed_value)
        kl = sum([x * np.log(x / posterior_prob) for x in self.category_probs.values()])
        out = self.kl_divergence_model.log(ts, kl)

        return MultinomialModelOutput(
            probability=posterior_prob,
            observed_value=observed_value,
            variance=var,
            anomaly=out.is_anomaly,
        )
