from typing import Any

import matplotlib.figure as figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import invgamma, norm

from trct.models.base_model import BaseModel


class NormalModel(BaseModel):
    def __init__(self, u, v, alpha_0=1, beta_0=1, mu_0=5, sigma_0=2):
        super().__init__(u, v)

        self.alphas: list[float] = [alpha_0]
        self.betas: list[float] = [beta_0]
        self.mus: list[float] = [mu_0]
        self.sigmas: list[float] = [sigma_0]

        self.observed_variables: list[float] = [0]
        self.upper_bound: list[float] = [0]
        self.lower_bound: list[float] = [0]
        self.anomalies: list[bool] = [False]

    def log(self, ts, obsedved_variable) -> None:
        """Log a new observation."""

        super().log(ts)

        self.observed_variable = obsedved_variable

    def __repr__(self) -> str:
        """String representation of the model."""
        return f"{self.u} -> {self.v} (#{self.n} N({self.mu:.2f}, {self.sigma:.2f}))"

    def get_data(self):
        return {
            "observed": self.observed_variables,
            "ts": self.tss,
            "alpha": self.alphas,
            "beta": self.betas,
            "mu": self.mus,
            "sigma": self.sigmas,
            "upper_bound": self.upper_bound,
            "lower_bound": self.lower_bound,
            'anomalies': self.anomalies
        }

    def plot(self, axes, **kwargs) -> None:
        """Plot the model statistics."""
        # ignore type checks

        # get data
        df = self.to_frame(omit_first=True)
        if "resample" in kwargs:
            df = (
                df.select_dtypes(exclude=["object"]).resample(kwargs["resample"]).mean()
            )
            
        df["mu"].plot(axes=axes, label="$\\mathbb{E}(X)$")

        axes.fill_between(df.index, df["lower_bound"], df["upper_bound"], facecolor="gray", alpha=0.3) # type: ignore

        df["lower_bound"].plot(
            ax=axes, color="purple", label="$\\pm3\\sigma$", alpha=0.5
        )
        df["upper_bound"].plot(
            ax=axes, color="purple", alpha=0.5, label="_nolegend_"
        )
        df["observed"].plot(axes=axes, label="X")

        anomalies = df[df["anomalies"]]
        if anomalies.shape[0] > 0:
            anomalies.plot(
                y="observed",
                ax=axes,
                color="red",
                marker="o",
                linestyle="None",
                label="anomaly",
            )

        axes.set_title(
            f"Anomalies on RTT ({anomalies.shape[0]}, {100*anomalies.shape[0]/df.shape[0]:.3f}%)"
            f"\n {self.u}->{self.v} "
        )
        axes.legend(fancybox=True)
        axes.set_xlabel("time")
        axes.set_ylabel("X")

    def score(self, observed_variable) -> tuple[bool, float, float]:
        rtt_is_outlier = observed_variable > self.ub or observed_variable < self.lb
        rtt_prob = self.pdf(observed_variable)
        rtt_mu_diff = observed_variable - self.mu
        return rtt_is_outlier, rtt_prob, rtt_mu_diff

    def pdf(self, x) -> float:
        return np.exp(-0.5 * ((x - self.mu) / self.sigma) ** 2) / (
            self.sigma * np.sqrt(2 * np.pi)
        )

    @property
    def timestamp(self):
        return self.tss[-1] if self.tss else None

    @timestamp.setter
    def timestamp(self, ts):
        self.tss.append(ts)

    @property
    def alpha(self):
        return self.alphas[-1]

    @alpha.setter
    def alpha(self, alpha):
        self.alphas.append(alpha)

    @property
    def beta(self):
        return self.betas[-1]

    @beta.setter
    def beta(self, beta):
        self.betas.append(beta)

    @property
    def mu(self):
        return self.mus[-1]

    @mu.setter
    def mu(self, mu):
        self.mus.append(mu)

    @property
    def sigma(self):
        return self.sigmas[-1]

    @sigma.setter
    def sigma(self, sigma):
        self.sigmas.append(sigma)

    @property
    def observed_variable(self):
        return self.observed_variables[-1]

    @observed_variable.setter
    def observed_variable(self, value):
        self.observed_variables.append(value)

        self.alpha = self.alpha + 1 / 2
        self.beta = self.beta + 0.5 * (value - self.mu) ** 2

        self.mu = self.mu + 1 / self.n * (value - self.mu)
        self.sigma = np.sqrt(self.beta / (self.alpha + 1))
        self.ub = self.mu + 3 * self.sigma
        self.lb = self.mu - 3 * self.sigma
        self.anomaly = value > self.ub or value < self.lb

    @property
    def n(self):
        return self.ns[-1]

    @n.setter
    def n(self, n):
        self.ns.append(n)

    @property
    def ub(self):
        return self.upper_bound[-1]

    @ub.setter
    def ub(self, ub):
        self.upper_bound.append(ub)

    @property
    def lb(self):
        return self.lower_bound[-1]

    @lb.setter
    def lb(self, lb):
        self.lower_bound.append(lb)

    @property
    def anomaly(self):
        return self.anomalies[-1]
    
    @anomaly.setter
    def anomaly(self, anomaly):
        self.anomalies.append(anomaly)