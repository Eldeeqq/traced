"""Poisson model for anomaly detection."""
from math import exp, factorial
from typing import Any

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

    @property
    def error(self) -> float:
        """Return the error of the observation."""
        return self.observed_value - self.expected_value

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
        threshold: float = 0.05,
        parent: BaseModel | None = None,
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

    def pdf(self, value: float) -> float:
        """Return the probability of observing `value`."""
        return (
            self.lambdas[-1] ** value * exp(-self.lambdas[-1]) / factorial(round(value))
        )

    def log(self, ts: int, observed_value: float) -> PoissonModelOutput:
        """Log a new observation and return whether it is an anomaly."""
        super().log_timestamp(ts)

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
            expected_value=self.lambdas[-1],
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

        title = f'Anomalies on {kwargs.get("kind", "TTL")}'
        ax.set_title(title)

        anomalies = df[df["anomalies"]]

        df["probabilities"].plot(ax=ax, label="probabilities")
        anomalies["probabilities"].plot(ax=ax, style="ro", label="anomalies")
        ax.legend()
