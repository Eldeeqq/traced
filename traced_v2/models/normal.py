"""This module contains the NormalModel class, which implements a normal model with bayesian updating.
"""

from typing import Any
import numpy as np

import pydantic
from matplotlib import pyplot as plt
from matplotlib.pyplot import Axes

from traced_v2.models.base_model import BaseModel, Visual

# pylint: disable=too-many-arguments, fixme, line-too-long, too-many-instance-attributes, invalid-name


class NormalModelOutput(pydantic.BaseModel):
    """Output of the Normal model."""

    is_anomaly: bool
    n_anomalies: int
    expected_value: float
    observed_value: float
    sigma: float
    pdf: float

    @property
    def error(self):
        """Return the error of the observation."""
        return self.observed_value - self.expected_value

    @property
    def error_rate(self):
        """Return the error rate of the observation."""
        return self.observed_value / self.expected_value


class NormalModel(BaseModel, Visual):
    """Univariate Normal model with Bayesian updating."""

    def __init__(
        self,
        src: str,
        dest: str,
        alpha_0: float = 1.0,
        beta_0: float = 1.0,
        mu_0: float = 0.0,
        sigma_0: float = 1.0,
        one_sided: bool = False,
        gamma: float = 0.5,
        sigma_factor: float = 4.0,
        parent: BaseModel | None = None,
    ) -> None:
        super().__init__(
            src, dest, subscription=parent.subscription if parent else None
        )

        self.alpha = alpha_0
        self.beta = beta_0

        self.sigma_factor: float = sigma_factor
        self.one_sided: bool = one_sided
        self.gamma = gamma

        self.expected_values: list[float] = [mu_0]
        self.sigmas: list[float] = [sigma_0]

        self.observed_values: list[float] = [mu_0]
        self.anomalies: list[bool] = [False]

    def pdf(self, value: float) -> float:
        return (1 / (self.sigmas[-1] * np.sqrt(2 * np.pi))) * np.exp(
            -0.5 * ((value - self.expected_values[-1]) / self.sigmas[-1]) ** 2
        )

    def log(self, ts: int, observed_value: float) -> NormalModelOutput:
        """Log a new observation and return whether it is an anomaly."""
        super().log_timestamp(ts)

        self.observed_values.append(observed_value)
        mu = self.expected_values[-1]

        self.alpha += self.gamma
        self.beta += self.gamma * (observed_value - mu) ** 2

        mu_2 = mu + self.gamma / self.get_n() * (observed_value - mu)
        self.expected_values.append(mu_2)

        sigma = np.sqrt(self.beta / (self.alpha + 1))
        self.sigmas.append(sigma)

        ub = mu_2 + self.sigma_factor * sigma
        lb = mu_2 - self.sigma_factor * sigma

        self.anomalies.append(
            observed_value > ub or observed_value < lb
            if not self.one_sided
            else observed_value > ub
        )
        self.n_anomalies += self.anomalies[-1]

        return NormalModelOutput(
            is_anomaly=self.anomalies[-1],
            n_anomalies=self.n_anomalies,
            expected_value=mu_2,
            observed_value=observed_value,
            sigma=sigma,
            pdf=self.pdf(observed_value),
        )

    def to_dict(self) -> dict[str, list[Any]]:
        """Convert the model statistics to a dictionary."""
        return {
            "observed_values": self.observed_values[1:],
            "expected_values": self.expected_values[1:],
            "sigmas": self.sigmas[1:],
            "anomalies": self.anomalies[1:],
        }

    def plot(self, ax: Axes | None = None, **kwargs) -> None:
        """Plot the model statistics."""
        df = self.to_frame()

        ax = ax or plt.gca()

        if "resample" in kwargs:
            df = (
                df.select_dtypes(exclude=["object"]).resample(kwargs["resample"]).mean()
            )

        df["lower_bound"] = df["expected_values"] - self.sigma_factor * df["sigmas"]
        df["upper_bound"] = df["expected_values"] + self.sigma_factor * df["sigmas"]

        ax.fill_between(df.index, df["lower_bound"], df["upper_bound"], facecolor="gray", alpha=0.3)  # type: ignore

        df["lower_bound"].plot(
            ax=ax,
            color="tab:purple",
            label="$\\pm" + str(self.sigma_factor) + "\\sigma$",
            alpha=0.5,
        )
        df["upper_bound"].plot(ax=ax, color="tab:purple", alpha=0.5, label="_nolegend_")
        df["observed_values"].plot(ax=ax, label="X", color="tab:orange", alpha=0.8)
        df["expected_values"].plot(ax=ax, label="$\\mathbb{E}(X)$", color="tab:blue")

        anomalies = df[df["anomalies"]]

        if anomalies.shape[0] > 0:
            anomalies.plot(
                y="observed_values",
                ax=ax,
                color="red",
                marker="o",
                linestyle="None",
                label="anomaly",
                alpha=0.8,
            )
        title = f'Anomalies on {kwargs.get("kind", "RTT")}'
        ax.set_title(
            f"{title} ({anomalies.shape[0]}, "
            f"{100*anomalies.shape[0]/df.shape[0]:.3f}%)"
            if df.shape[0] > 0
            else "NaN" f"\n {self.src}->{self.dest} "
        )

        if "start" in kwargs:
            ax.axvline(
                kwargs["start"], color="gray", linestyle="--", label="training end"
            )

        ax.legend(fancybox=True)
        ax.set_xlabel("time")
        ax.set_ylabel("X")
