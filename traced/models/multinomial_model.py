import copy
from matplotlib import pyplot as plt
import numpy as np
from collections import defaultdict
from typing import Any, List, Dict, Optional, Tuple, Union, Hashable
import pandas as pd
import scipy

from traced.models.base_model import BaseModel


class MultinomialModel(BaseModel):
    def __init__(self, src: str, dest: str, gamma: float=1., *args, **kwargs) -> None:
        super().__init__(src, dest, *args, **kwargs)

        self.undef: list[float] = [0.]  # number of undefined observations
        self.undef_prob: list[float] = [0.0]  # number of undefined observations
        self.seen: set[Hashable] = set()  # set of seen timestamps
        self.gamma = gamma
        self.counter = 0
        self.counts: dict[Hashable, list[float]] = defaultdict(
            lambda: copy.deepcopy(self.undef)
        )
        self.marginal_probs: dict[Hashable, list[float]] = defaultdict(
            lambda: copy.deepcopy(self.undef_prob)
        )
        self.marginal_var: dict[Hashable, list[float]] = defaultdict(
            lambda: copy.deepcopy(self.undef_prob)
        )

        self.posterior_probs: dict[Hashable, list[float]] = defaultdict(
            lambda: copy.deepcopy(self.undef_prob)
        )  # todo use 1  - uniform prior
        self.posterior_var: dict[Hashable, list[float]] = defaultdict(
            lambda: copy.deepcopy(self.undef_prob)
        )
        self.cats: list[Hashable] = [0]
        self.cat_prob: list[float] = [] #TODO add 0 abd propagate to df
        self.max_cat_prob: list[float] = [] #TODO add 0 abd propagate to df
        self.expected_val: list[float] = []
        self.most_probable_cat: list[Hashable] = []
        self.anomalies: list[bool] = [False]
        self.anomalies_y: list[float] = [0.0]

    def log(self, ts: int, cat: Hashable):
        super().log(ts)
        self.cat = cat
        # TODO: calculate
        return bool(self.anomalies[-1]), float(self.cat_prob[-1])
        

    @property
    def k(self) -> int:
        return len(self.counts)

    def __repr__(self):
        return f"MultinomialModel(src={self.u}, dest={self.v}, k={self.k} n={self.n})"

    def get_data(self) -> dict[str, list[Any]]:
        """Return the model data."""
        return {
            "cats": self.cats,
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

    def plot(self, axes: Optional[plt.Axes] = None, *args, **kwargs) -> None: # type: ignore
        """Plot the model on specified axis."""
        if axes is None:
            axes: plt.Axes = plt.gca()

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
        # plt.plot(df.index, self.cat_prob, label="cat", color="black", linestyle="dashed")
        axes.legend(fancybox=True, loc=0)

    def pdf(self, cat) -> Any:
        """Return the probability of the cat."""
        return self.posterior_probs[cat][-1]

    @property
    def cat(self) -> Hashable:
        return self.cats[-1]

    @cat.setter
    def cat(self, cat: Hashable) -> None:
        self.cats.append(cat)
        self.seen.add(cat)

        self.anomalies_y.append(self.posterior_probs[cat][-1])
        # tmp = 0
        index  = -1
        max_prob = 0
        for key in self.seen:
            new_count = float(self.counts[key][-1] + int(key == cat) * self.gamma)
            self.counter += self.gamma
            self.counts[key].append(new_count)
            marginal_prob = new_count / self.counter
            self.marginal_probs[key].append(marginal_prob)
            self.marginal_var[key].append(marginal_prob * (1 - marginal_prob))

            self.posterior_probs[key].append((new_count + 1) / (self.counter + self.k))
            index = index if max_prob > self.posterior_probs[key][-1] else key

            max_prob = max(max_prob, self.posterior_probs[key][-1])
            # tmp += self.posterior_probs[key][-1] * key
            p = self.marginal_probs[key][-1]
            var = p * (1 - p) / (self.n + 1)
            self.posterior_var[key].append(var)
        self.cat_prob.append(self.posterior_probs[cat][-1])
        # self.expected_val.append(tmp)
        self.max_cat_prob.append(max_prob)
        self.most_probable_cat.append(index)
        self.undef.append(self.undef[-1])
        self.undef_prob.append(self.undef_prob[-1])


if __name__ == "__main__":
    model = MultinomialModel("a", "b")
    import scipy.stats
    data = np.argmax(
        scipy.stats.multinomial.rvs(1, [0.1, 0.2, 0.5, 0.05, 0.15], size=1000), axis=1
    )
    # data = [1, 2, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1,1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 3,2,3,3,2,2,2,2,2,3,3,2,2]

    for i, x in enumerate(data):
        model.log(i, x)
    fig = plt.figure()

    model.plot_marginal_posterior()

    x = model.get_data()
