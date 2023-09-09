"""Module for the MultinomialModel class."""

from typing import Any, Callable

import numpy as np
import pydantic
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from traced_v2.models.base_model import BaseModel, Visual


# pylint: disable=too-many-arguments, fixme, line-too-long, too-many-instance-attributes, invalid-name
class BernoulliModelOutput(pydantic.BaseModel):
    """Output of the Bernoulli model."""

    is_anomaly: bool
    variance: float
    probability: float
    observed_value: bool


class BernoulliModel(BaseModel, Visual):
    """Univariate Bernoulli model with bayesian updating."""

    def __init__(
        self,
        src: str,
        dest: str,
        parent: BaseModel | None = None,
        scorer=Callable[[bool, float], bool],
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            src, dest, subscription=parent.subscription if parent else None
        )

        self.alpha: int = 0
        self.beta: int = 0

        self.success_probs: list[float] = []
        self.success_var: list[float] = []
        self.observed_variables: list[float] = []
        self.anomalies: list[bool] = []
        self.scorer = scorer or (lambda x, y: False)

    def to_dict(self) -> dict[str, list[Any]]:
        return {
            "success_probs": self.success_probs,
            "success_var": self.success_var,
            "observed_variables": self.observed_variables,
            "anomalies": self.anomalies,
        }

    def plot(self, ax: Axes | None = None):
        df = self.to_frame()

        ax = ax or plt.gca()

        lower_bound = df["success_probs"] - 3 * df["success_var"].apply(np.sqrt)
        upper_bound = df["success_probs"] + 3 * df["success_var"].apply(np.sqrt)
        
        ax.fill_between(df.index, lower_bound, upper_bound, facecolor="tab:blue", alpha=0.3)  # type: ignore
        ax.plot(df.index, upper_bound, color="tab:blue", alpha=0.3)
        ax.plot(df.index, lower_bound, color="tab:blue", alpha=0.3)

        df["success_probs"].plot(
            ax=ax,
            label="$P(\\mathrm{success})$",
            color="tab:blue",
            legend="probability",  # type: ignore
        )

        p = df["success_probs"].mean()

        positive = df[df["observed_variables"] == True]
        if positive.shape[0] > 1:
          
            positive.astype(int).plot(
                ax=ax,
                y="observed_variables",
                label="success",
                color="green",
                marker="o",
                linestyle="None",
                alpha=0.0025 if p>0.20 else 0.5,
            )

            positive[positive["anomalies"]].astype(int).plot(
                ax=ax,
                y="observed_variables",
                label="anomaly",
                color="black",
                marker="x",
                linestyle="None",
                alpha=0.9,
            )
        negative = df[df["observed_variables"] == False]
        if negative.shape[0] > 1:
            negative.astype(int).plot(
                ax=ax,
                y="observed_variables",
                label="failure",
                color="red",
                marker="o",
                linestyle="None",
                alpha=0.0025 if (1-p)>0.20 else 0.5,
            )
            negative[negative["anomalies"]].astype(int).plot(
                ax=ax,
                y="observed_variables",
                label="",
                color="black",
                marker="x",
                linestyle="None",
                alpha=0.9,
            )
        ax.set_ylabel("$P(\\mathrm{success})$")
        ax.set_title("Probability of success")
        ax.set_ylabel("$p$")
        legend = plt.legend()
        # set legebd alpha to 1
        for item in legend.legendHandles:  # type: ignore
            try:
                item.set_alpha(1)
                item._legmarker.set_alpha(1)
            except AttributeError:
                pass

    def log(self, ts, observed_variable: bool) -> BernoulliModelOutput:
        """Log a new observation."""
        super().log_timestamp(ts)

        self.observed_variables.append(observed_variable)
        self.alpha += observed_variable
        self.beta += 1 - observed_variable

        self.success_probs.append(self.alpha / (self.alpha + self.beta))
        self.success_var.append(
            (self.alpha * self.beta)
            / ((self.alpha + self.beta) ** 2 * (self.alpha + self.beta + 1))
        )
        self.anomalies.append(self.scorer(observed_variable, self.success_probs[-1]))
        return BernoulliModelOutput(
            is_anomaly=self.anomalies[-1],
            variance=self.success_var[-1],
            probability=self.success_probs[-1],
            observed_value=observed_variable,
        )