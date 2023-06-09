import copy
from matplotlib import pyplot as plt
import numpy as np
from collections import defaultdict
from typing import Any, List, Dict, Tuple, Union
import pandas as pd
import scipy

from trct.models.base_model import BaseModel


class MultinomialModel(BaseModel):
    def __init__(self, src: str, dest: str, *args, **kwargs) -> None:
        super().__init__(src, dest, *args, **kwargs)

        self.undef: list[int] = [0]  # number of undefined observations
        self.undef_prob: list[float] = [0.0]  # number of undefined observations
        self.seen: set[int] = set()  # set of seen timestamps
        self.counts: dict[int, list[int]] = defaultdict(
            lambda: copy.deepcopy(self.undef)
        )
        self.marginal_probs: dict[int, list[float]] = defaultdict(
            lambda: copy.deepcopy(self.undef_prob)
        )
        self.marginal_var: dict[int, list[float]] = defaultdict(
            lambda: copy.deepcopy(self.undef_prob)
        )

        self.posterior_probs: dict[int, list[float]] = defaultdict(
            lambda: copy.deepcopy(self.undef_prob)
        )  # todo use 1  - uniform prior
        self.posterior_var: dict[int, list[float]] = defaultdict(
            lambda: copy.deepcopy(self.undef_prob)
        )
        self.ttls: list[int] = [0]
        self.ttl_prob: list[float] = [] #TODO add 0 abd propagate to df
        self.max_ttl_prob: list[float] = [] #TODO add 0 abd propagate to df
        self.expected_val: list[float] = []
        self.most_probable_ttl: list[float] = []
        self.anomalies: list[bool] = [False]
        self.anomalies_y: list[float] = [0.0]

    def log(self, ts: int, ttl: int):
        super().log(ts)
        self.ttl = ttl

    @property
    def k(self) -> int:
        return len(self.counts)

    def __repr__(self):
        return f"MultinomialModel(src={self.u}, dest={self.v}, k={self.k} n={self.n})"

    def get_data(self) -> dict[str, list[Any]]:
        """Return the model data."""
        return {
            "ttls": self.ttls,
            **{f"p_{i}": self.posterior_probs[i] for i in self.posterior_probs.keys()},
            **{f"var_p_{i}": self.posterior_var[i] for i in self.posterior_probs.keys()},
            **{f"m_{i}": self.marginal_probs[i] for i in self.marginal_probs.keys()},
            **{f"n_{i}": self.counts[i] for i in self.marginal_probs.keys()},
            'anomalies': self.anomalies,
            'anomalies_y': self.anomalies_y,
        }

    def plot_marginal_posterior(self, *args, **kwargs) -> None:
        """Plot the model on specified axis."""
        fig, (ax0, ax1) = plt.subplots(2, 1, sharex=True, sharey=True)

        for i in self.marginal_probs.keys():
            m = np.array(self.marginal_probs[i])
            v = np.array(self.marginal_var[i])
            mvar = 3 * np.sqrt(v / self.n)
            ax0.plot(
                self.marginal_probs[i],
                label=f"m_{i}",
            )
            ax0.fill_between(
                range(len(self.marginal_probs[i])), m - mvar, m + mvar, alpha=0.2
            )

            p = np.array(self.posterior_probs[i])
            ax1.plot(
                p,
                label=f"p_{i}",
            )
            var = 3 * np.sqrt(np.array(self.posterior_var[i]))
            ax1.fill_between(range(len(p)), p - var, p + var, alpha=0.2)
        ax1.set_title(f"Posterior")
        ax0.set_title(f"Marginal")
        ax1.legend()
        ax0.legend()

    def plot(self, axes: plt.Axes, *args, **kwargs) -> None:
        """Plot the model on specified axis."""
        df = self.to_frame()
        for i in self.posterior_probs.keys():
            df[f'p_{i}'].plot(axes=axes, label=f"p_{i}")
            thr = 3 * np.sqrt(df[f"var_p_{i}"])
            axes.fill_between(df.index, df[f'p_{i}'] - thr, df[f'p_{i}'] + thr, alpha=0.15)  # type: ignore
        anomalies = df[df['anomalies']]
        if anomalies.shape[0] > 0:
            anomalies.plot(
                y="anomalies_y",
                ax=axes,
                color="red",
                marker="o",
                linestyle="None",
                label="anomaly",
            )
        plt.plot(df.index, self.ttl_prob, label="ttl", color="black", linestyle="dashed")
        axes.set_title(f"TTL for {self.u} -> {self.v} ({anomalies.shape[0]}, {anomalies.shape[0]/df.shape[0]:.2f})")
        axes.legend(fancybox=True, loc=0)

    def score(self, ttl) -> Any:
        """Score the traceroute data."""
        return self.pdf(ttl) < 0.025

    def pdf(self, ttl) -> Any:
        """Return the probability of the ttl."""
        return self.posterior_probs[ttl][-1]

    @property
    def ttl(self) -> int:
        return self.ttls[-1]

    @ttl.setter
    def ttl(self, ttl: int) -> None:
        self.ttls.append(ttl)
        self.seen.add(ttl)

        self.anomalies.append(self.score(ttl))
        self.anomalies_y.append(self.posterior_probs[ttl][-1])
        tmp = 0
        index  = -1
        max_prob = 0
        for key in self.seen:
            new_count = self.counts[key][-1] + int(key == ttl)
            self.counts[key].append(new_count)
            marginal_prob = new_count / self.n
            self.marginal_probs[key].append(marginal_prob)
            self.marginal_var[key].append(marginal_prob * (1 - marginal_prob))

            self.posterior_probs[key].append((new_count + 1) / (self.n + self.k))
            index = index if max_prob > self.posterior_probs[key][-1] else key

            max_prob = max(max_prob, self.posterior_probs[key][-1])
            tmp += self.posterior_probs[key][-1] * key
            p = self.marginal_probs[key][-1]
            var = p * (1 - p) / (self.n + 1)
            self.posterior_var[key].append(var)
        self.ttl_prob.append(self.posterior_probs[ttl][-1])
        self.expected_val.append(tmp)
        self.max_ttl_prob.append(max_prob)
        self.most_probable_ttl.append(index)
        self.undef.append(self.undef[-1])
        self.undef_prob.append(self.undef_prob[-1])


if __name__ == "__main__":
    model = MultinomialModel("a", "b")
    data = np.argmax(
        scipy.stats.multinomial.rvs(1, [0.1, 0.2, 0.5, 0.05, 0.15], size=1000), axis=1
    )
    # data = [1, 2, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1,1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 3,2,3,3,2,2,2,2,2,3,3,2,2]

    for i, x in enumerate(data):
        model.log(i, x)
    fig = plt.figure()

    model.plot_marginal_posterior()

    x = model.get_data()
