"""Exponential model for anomaly detection."""
from typing import Any

import numpy as np
import pydantic
from matplotlib import pyplot as plt
from matplotlib.pyplot import Axes

from traced_v2.models.base_model import BaseModel, Visual

# pylint: disable=too-many-arguments, fixme, line-too-long, too-many-instance-attributes, invalid-name


class ExponentialModelOutput(pydantic.BaseModel):
    """Output of the Exponential model."""

    is_anomaly: bool
    n_anomalies: int
    sf: float
    shift: float
    expected_value: float
    observed_value: float

    @property
    def error(self) -> float:
        """Return the error of the observation."""
        return self.observed_value - self.expected_value + self.shift


class ExponentialModel(BaseModel, Visual):
    """Univariate Exponential model with bayesian updating."""

    def __init__(
        self,
        src: str,
        dest: str,
        alpha_0: float = 1.0,
        beta_0: float = 1.0,
        gamma: float = 1.0,
        threshold: float = 0.1,
        parent: BaseModel | None = None,
        inherit_beta: bool = True,
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
        self.expected_values: list[float] = [None]
        self.probs: list[float] = [1]
        self.inherit_beta = inherit_beta
        self.learn_shift = shift is None
        self.shift = shift if shift is not None else np.inf

    def sf(self, value: float) -> float:
        """Return the probability of observing values larger than `value`."""
        if np.isinf(-self.lambdas[-1] * value) or np.isnan(-self.lambdas[-1] * value):
            return 1.0
        X = np.max([-self.lambdas[-1] * value, 1e-20])
        return np.exp(X)  # (value-self.shift))

    def log(self, ts: int, observed_value: float) -> ExponentialModelOutput:
        """Log a new observation and return whether it is an anomaly."""
        super().log_timestamp(ts)

        if self.learn_shift:
            self.shift = min(observed_value, self.shift)
            # observed_value = observed_value - self.shift

        if self.n == 1 and self.inherit_beta:
            self.beta = observed_value

        self.observed_values.append(observed_value)
        self.alpha += self.gamma * observed_value
        self.beta += self.gamma

        self.lambdas.append(self.beta / self.alpha)
        self.expected_values.append(1 / self.lambdas[-1])

        sf = self.sf(observed_value)
        self.probs.append(sf)
        self.anomalies.append(sf < self.threshold)
        self.n_anomalies += int(self.anomalies[-1])

        return ExponentialModelOutput(
            is_anomaly=self.anomalies[-1],
            sf=sf,
            expected_value=1 / self.lambdas[-1],
            observed_value=observed_value,
            n_anomalies=self.n_anomalies,
            shift=self.shift,
        )

    def to_dict(self) -> dict[str, list[Any]]:
        """Return a dictionary with statistics from the model."""
        return {
            "observed_values": self.observed_values[1:],
            "expected_values": self.expected_values[1:],
            "probabilities": self.probs[1:],
            "anomalies": self.anomalies[1:],
        }

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
            alpha=0.25,
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

        # ax2 = ax.twinx()
        # ax2.set_ylabel("Probability")
        # ymin, ymax = ax.get_ylim()
        # # print(ymin, (ymin+ymax_/2), ymax)
        # # print(self.sf(ymin),self.sf((ymin+ymax_/2)), self.sf(ymax))

        # ax2.set_ylim((self.sf(ymin),self.sf(ymax)))
        # # ax2.set_yticks([])
        # ax2.plot([],[])

        ax.set_title(title)

        ax.legend()
