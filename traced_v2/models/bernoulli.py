"""Module for the MultinomialModel class."""

from collections import defaultdict
from typing import Any

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from traced_v2.models.base_model import BaseModel, Visual

# pylint: disable=too-many-arguments, fixme, line-too-long, too-many-instance-attributes, invalid-name


class BernoulliModel(BaseModel, Visual):
    """Univariate Bernoulli model with bayesian updating."""

    def to_dict(self) -> dict[str, list[Any]]:
        return {
            "success_probs": self.success_probs,
            "success_var": self.success_var,
            "observed_variables": self.observed_variables,
        }

    def plot(self, ax: Figure | Axes | None = None):
        if ax is None:
            ax: plt.Axes = plt.gca()
        df = self.to_frame()

        lower_bound = df["success_probs"] - 3 * df["success_var"].apply(np.sqrt)
        upper_bound = df["success_probs"] + 3 * df["success_var"].apply(np.sqrt)
        ax.fill_between(df.index, lower_bound, upper_bound, facecolor="tab:blue", alpha=0.3)  # type: ignore
        ax.plot(df.index, upper_bound, color="tab:blue", alpha=0.3)
        ax.plot(df.index, lower_bound, color="tab:blue", alpha=0.3)

        df["success_probs"].plot(
            ax=ax,
            label="$P(\\mathrm{success})$",
            color="tab:blue",
            legend="probability",
        )
        df[df["observed_variables"] == True].plot(
            ax=ax,
            y="observed_variables",
            label="success",
            color="green",
            marker="o",
            linestyle="None",
            alpha=0.025,
        )
        df[df["observed_variables"] == False].plot(
            ax=ax,
            y="observed_variables",
            label="failure",
            color="red",
            marker="o",
            linestyle="None",
            alpha=0.025,
        )
        ax.set_ylabel("$P(\\mathrm{success})$")
        ax.set_title(f"Probability of success")
        ax.set_ylabel("$p$")
        legend = plt.legend()
        for item in legend.legendHandles:
            try:
                item.set_alpha(1)
                item._legmarker.set_alpha(1)
            except AttributeError:
                pass

    def __init__(self, src: str, dest: str, *args, **kwargs) -> None:
        super().__init__(src, dest, *args, **kwargs)

        self.alpha: int = 0
        self.beta: int = 0

        self.success_probs: list[float] = []
        self.success_var: list[float] = []
        self.observed_variables: list[float] = []

    def log(self, ts, observed_variable: bool) -> None:
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
