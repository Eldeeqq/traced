"""Bernoulli model for bayesian inference."""

import numpy as np
import pandas as pd

from scipy.stats import norm, invgamma
from typing import Any
import matplotlib.pyplot as plt
import matplotlib.figure as figure

from analyzer.models.base_model import BaseModel


class BernoulliModel(BaseModel):
    """Bernoulli model

    This model is used to infer the probability of `success` as a Bernoulli distirbution.

    Attributes:
        u (str): source
        v (str): destination
        ctr (int): number of observations
        successes (list[int]): number of successes
        failures (list[int]): number of failures
        success_probs (list[float]): success probabilities
        success_vars (list[float]): success variances
        tss (list[int]): timestamps
        ns (list[int]): number of observations
    """

    def __init__(self, u, v, sucess_prob=0.5, success_var=0.25):
        super().__init__(u, v)
        self.successes: list[int] = [0]
        self.failures: list[int] = [0]

        self.success_probs: list[float] = [sucess_prob]
        self.success_vars: list[float] = [success_var]

    def log(self, ts, success) -> None:
        """Log a new measurement.

        Args:
            ts (int): timestamp
            success (bool): whether the event was successful

        """
        super().log(ts)
        self.success = success

    def __repr__(self) -> str:
        """String representation of the model."""
        return f"{self.u} -> {self.v} (#{self.n} Ber({self.success_prob:.2f}))"

    def get_data(self) -> dict[str, list[Any]]:
        return {
                "successes": self.successes,
                "failures": self.failures,
                "success_prob": self.success_probs,
                "success_var": self.success_vars,
            }
    
    def probability_changes(self) -> bool:
        """Return whether the probability has changed."""
        seen = set()
        for prob in self.success_probs:
            if prob in seen:
                return True
            seen.add(prob)
        return False

    def plot(self, axes: plt.Axes, *args, **kwargs) -> None:
        """Plot the model on specified axis."""
        # get data
        df = self.to_frame(omit_first=True)

        if "resample" in kwargs:
            df = (
                df.select_dtypes(exclude=["object"]).resample(kwargs["resample"]).mean()
            )
            del kwargs["resample"]

        # https://stats.stackexchange.com/questions/4756/confidence-interval-for-bernoulli-sampling
        bound = (df["success_var"] / df["total"]).apply(np.sqrt)

        upper_bound = df["success_prob"] + bound
        lower_bound = df["success_prob"] - bound

        axes.fill_between(df.index, lower_bound, upper_bound, facecolor="gray", alpha=0.3)  # type: ignore
        axes.plot(upper_bound, color="gray", alpha=0.3)  # type: ignore
        axes.plot(lower_bound, color="gray", alpha=0.3)  # type: ignore

        df["success_prob"].plot(ax=axes, label="$P(\\mathrm{success})$", color="green")

        axes.set_ylabel("$P(\\mathrm{success})$")
        axes.set_title(f"Probability of success")
        axes.set_ylabel("$p$")
        axes.set_ylim(
            max(0, lower_bound.min()) - 0.01, min(1, upper_bound.max()) + 0.01
        )

    def score(self, success) -> float:
        """Return the score of the model given a success or failure and current state."""
        return success - self.success_prob

    @property
    def success(self):
        return self.successes[-1], self.failures[-1]

    @success.setter
    def success(self, success):
        """Update the success and failure counts and Bernouli distribution parameters."""
        a, b = self.success
        a, b = a + 1 if success else a, b + 1 if not success else b
        self.successes.append(a)
        self.failures.append(b)
        self.success_prob = a / (a + b)
        self.success_var = (a * b) / ((a + b) ** 2 * (a + b + 1))

    @property
    def success_prob(self):
        return self.success_probs[-1]

    @success_prob.setter
    def success_prob(self, success_prob):
        self.success_probs.append(success_prob)

    @property
    def success_var(self):
        return self.success_vars[-1]

    @success_var.setter
    def success_var(self, success_var):
        self.success_vars.append(success_var)
