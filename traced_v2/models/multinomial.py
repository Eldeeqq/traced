"""Module for the MultinomialModel class."""

from collections import defaultdict
from typing import Any, Hashable

import matplotlib.pyplot as plt
import numpy as np
import pydantic
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
        self.information_model: NormalModel = NormalModel(
            src, dest, parent=self, sigma_factor=3, alpha_0=15, mu_0=0
        )

        self.gamma: float = gamma
        self.counter: float = 0

    def average_crossentropy(self, value: float) -> float:
        return -np.mean(
            [x * np.log(value) for x in self.category_probs.values()] or [0]
        )

    def recompute_category(self, category: Hashable) -> None:
        posterior_prob = (self.category_counts[category] + 1) / (
            self.counter + 1 + len(self.seen_categories)
        )
        self.category_probs[category] = posterior_prob

    def append_category(self, category: Hashable) -> None:
        self.seen_categories.add(category)
        self.counter += self.gamma
        self.category_counts[category] += self.gamma

    def log(self, ts: int, observed_value: Hashable) -> MultinomialModelOutput:
        """Log a new observation and return whether it is an anomaly."""
        super().log_timestamp(ts)

        self.append_category(observed_value)

        for x in self.category_counts.keys():
            self.recompute_category(x)

        posterior_prob = self.category_probs[observed_value]
        a = self.category_counts[observed_value]
        b = self.counter - a
        var = (a * b) / ((self.counter**2) * (self.counter + 1))
        self.variance.append(var)

        self.probabilities.append(posterior_prob)
        self.observed_variables.append(observed_value)
        out = self.information_model.log(ts, self.average_crossentropy(posterior_prob))

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
            "anomalies": self.information_model.anomalies[1:],
        }

    def plot(self, ax: Axes | None = None, **kwargs) -> None:
        df = self.to_frame()

        ax = ax or plt.gca()

        # clist = rcParams["axes.prop_cycle"]
        # cgen = itertools.cycle(clist)
        title = f'Probability of {kwargs.get("kind", "classes")}'
        for i, gdf in df.groupby("observed_variables"):
            # if gdf.shape[0] > 2:
            # color= next(cgen)
            lower_bound = gdf["probabilities"] - 3 * gdf["variance"].apply(np.sqrt)
            upper_bound = gdf["probabilities"] + 3 * gdf["variance"].apply(np.sqrt)
            ax.fill_between(gdf.index, lower_bound, upper_bound, alpha=0.15)  # type: ignore
            gdf["probabilities"].plot(
                ax=ax, label=i, marker="None" if gdf.shape[0] > 2 else "o"
            )
        # else:
        #     gdf["probabilities"].plot(ax=ax, label=i, marker="o")
        anomalies = df[df["anomalies"]]
        if anomalies.shape[0] and not kwargs.get("hide_anomalies", False):
            anomalies.plot(
                marker="x",
                linestyle="None",
                y="probabilities",
                ax=ax,
                c="black",
                label="anomaly",
            )
            title += f" ({anomalies.shape[0]}, {anomalies.shape[0]/df.shape[0]:.3%})"
        ax.set_title(title)
        ax.set_ylabel("Probability")
        ax.legend()
        if df["observed_variables"].nunique() > 10:
            ax.legend().remove()


class ForgettingMultinomialModel(MultinomialModel):
    """Multinomial model with forgetting."""

    def __init__(
        self,
        src: str,
        dest: str,
        parent: BaseModel | None = None,
        gamma: float = 1,
        memory_size: int = 1000,
    ) -> None:
        super().__init__(src=src, dest=dest, parent=parent, gamma=gamma)
        self.queue = Queue(max_size=memory_size)
        self.category_counts = self.queue.counts_queue

    def append_category(self, category: Hashable) -> None:
        self.queue.add(category)
        self.counter = self.gamma * sum(self.category_counts.values())

    # def log(self, ts: int, observed_value: Hashable) -> MultinomialModelOutput:

    #     super().log_timestamp(ts)
    #     self.seen_categories.add(observed_value)
    #     self.counter = self.gamma * sum(self.category_counts.values())

    #     for x in self.category_counts.keys():
    #         self.recompute_category(x)

    #     posterior_prob =  self.category_probs[observed_value]
    #     a = self.category_counts[observed_value]
    #     b = self.counter - a
    #     var = (a * b) / ((self.counter**2) * (self.counter + 1))

    #     self.variance.append(var)
    #     self.probabilities.append(posterior_prob)
    #     self.observed_variables.append(observed_value)
    #     out = self.information_model.log(ts, self.average_crossentropy(posterior_prob))

    #     return MultinomialModelOutput(
    #         probability=posterior_prob,
    #         observed_value=observed_value,
    #         variance=var,
    #         anomaly=out.is_anomaly,
    #     )
