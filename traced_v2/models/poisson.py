"""Poisson model for anomaly detection."""
from math import exp, factorial
from typing import Any

import numpy as np
import pydantic
from matplotlib import pyplot as plt
from matplotlib.pyplot import Axes

from traced_v2.models.base_model import BaseModel, Visual

# pylint: disable=too-many-arguments, fixme, line-too-long, too-many-instance-attributes, invalid-name


class PoissonModelOutput(pydantic.BaseModel):
    """Output of the Poisson model."""

    is_anomaly: bool
    n_anomalies: int
    pdf: float
    expected_value: float
    observed_value: float
    shift: float = 0

    @property
    def error(self) -> float:
        """Return the error of the observation."""
        return self.observed_value - (self.expected_value) + self.shift

    @property
    def error_rate(self) -> float:
        """Return the error rate of the observation."""
        return self.observed_value / self.expected_value


class PoissonModel(BaseModel, Visual):
    """Univariate Poisson model with bayesian updating."""

    def __init__(
        self,
        src: str,
        dest: str,
        alpha_0: float = 1.0,
        beta_0: float = 1.0,
        gamma: float = 1.0,
        threshold: float = 0.1,
        parent: BaseModel | None = None,
        inherit_alpha: bool = True,
        shift: float | None = None,
    ) -> None:
        super().__init__(
            src, dest, subscription=parent.subscription if parent else None
        )
        self.alpha = alpha_0
        self.beta = beta_0
        self.gamma = gamma
        self.threshold = threshold
        self.observed_values: list[float] = [alpha_0 / beta_0]
        self.anomalies: list[bool] = [False]
        self.lambdas: list[float] = [alpha_0 / beta_0]
        self.probs: list[float] = [1]
        self.inherit_alpha = inherit_alpha
        self.learn_shift = shift is None
        self.shift = shift if shift is not None else np.inf

    def pdf(self, value: float) -> float:
        """Return the probability of observing `value`."""
        return (
            self.lambdas[-1] ** value * exp(-self.lambdas[-1]) / factorial(round(value))
        )

    def log(self, ts: int, observed_value: float) -> PoissonModelOutput:
        """Log a new observation and return whether it is an anomaly."""
        super().log_timestamp(ts)

        if self.learn_shift:
            self.shift = min(observed_value, self.shift)
            # observed_value = observed_value - self.shift

        if self.n == 1 and self.inherit_alpha:
            self.alpha = observed_value

        self.observed_values.append(observed_value)
        self.alpha += self.gamma * observed_value
        self.beta += self.gamma

        self.lambdas.append(self.alpha / self.beta)
        prob = self.pdf(observed_value)
        self.probs.append(prob)
        self.anomalies.append(prob < self.threshold)
        self.n_anomalies += int(self.anomalies[-1])

        return PoissonModelOutput(
            is_anomaly=self.anomalies[-1],
            pdf=prob,
            expected_value=self.lambdas[-1] + self.shift,
            shift=self.shift,
            observed_value=observed_value,
            n_anomalies=self.n_anomalies,
        )

    def to_dict(self) -> dict[str, list[Any]]:
        """Return a dictionary with statistics from the model."""
        return {
            "observed_values": self.observed_values[1:],
            "expected_values": self.lambdas[1:],
            "probabilities": self.probs[1:],
            "anomalies": self.anomalies[1:],
        }

    def plot(self, ax: Axes | None = None, **kwargs) -> None:
        """Plot the model statistics."""
        df = self.to_frame(omit_first=True)
        if ax is None:
            ax = plt.figure().add_subplot()

        df[~df["anomalies"]].plot(
            y="observed_values",
            marker="o",
            linestyle="None",
            label="Observed",
            alpha=0.3,
            ax=ax,
        )

        anomalies = df[df["anomalies"]]

        title = f'Anomalies on {kwargs.get("kind", "RTT")}'

        if anomalies.any().any():
            anomalies.plot(
                y="observed_values",
                marker="o",
                linestyle="None",
                label="Anomalies",
                alpha=0.3,
                ax=ax,
                c="r",
            )

            title += f" ({anomalies.shape[0]}, {anomalies.shape[0]/df.shape[0]:.2%})"
        df.plot(
            y="expected_values",
            label="Expected",
            ax=ax,
            c="black",
            linestyle="dotted",
            alpha=0.5,
        )

        ax.set_title(title)

        ax.legend()

    def plot_sf_anoms(self, ax: Axes | None = None, **kwargs) -> None:
        """Plot the model statistics."""
        df = self.to_frame(omit_first=True)
        if ax is None:
            ax = plt.figure().add_subplot()

        anomalies = df[df["anomalies"]]

        df["probabilities"].plot(
            ax=ax,
            label=kwargs.get("label", "probabilities"),
            alpha=kwargs.get("alpha", 1),
        )

        title = f'Anomalies on {kwargs.get("kind", "RTT")}'

        if anomalies.any().any():
            anomalies["probabilities"].plot(
                ax=ax,
                style="ro",
                label=kwargs.get("anom_label", "anomalies"),
                alpha=kwargs.get("alpha", 1),
            )
            title += f" ({anomalies.shape[0]}, {anomalies.shape[0]/df.shape[0]:.2%})"

        ax.set_title(title)

        ax.legend()
