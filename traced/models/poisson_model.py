import copy
from matplotlib import pyplot as plt
import numpy as np
from collections import defaultdict
from typing import Any, List, Dict, Optional, Tuple, Union
import pandas as pd
import scipy
import scipy.stats
import math

from traced.models.base_model import BaseModel


class PoissonModel(BaseModel):
    def __init__(self, src: str, dest: str, alpha_0=10, beta_0=1, gamma=1, *args, **kwargs) -> None:
        super().__init__(src, dest, *args, **kwargs)
        # hyper-parameters
        self.gamma = gamma
        self.alphas: list[float] = [alpha_0] 
        self.betas: list[float]  = [beta_0]
        
        # parameters
        self.lambdas: list[float] = [0]
        self.sigmas: list[float] = [0]

        # data
        self.values: list[float] = [0]
        self.probs: list[float] = [0]
        self.anomalies: list[bool] = [False]
        self.n_anomalies = 0

    def log(self, ts: int, value: int):
        super().log(ts)
        self.value = value
        self.prob = self.pdf(value)
        self.anomaly = self.prob < 0.05
        self.n_anomalies += int(self.anomaly)

        self.alpha += self.gamma * value
        self.beta += self.gamma
        self.lambda_ = self.alpha / self.beta
        return bool(self.anomaly), float(self.prob), float(self.lambda_), float(value), self.n_anomalies

    def pdf(self, value: int) -> float:
        return self.lambda_ ** value * np.exp(-self.lambda_) / math.factorial(value)

    def get_data(self) -> dict[str, list[Any]]:
        return {
            'value': self.values,
            'alpha': self.alphas,
            'beta': self.betas,
            'lambda': self.lambdas,
            'sigma': self.sigmas,
            'prob': self.probs,
            'anomaly': self.anomalies,
        }

    def plot(self, axes: Optional[Tuple[plt.Axes, plt.Axes]] = None, **kwargs) -> None:
        if axes is None:
            _, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True) # type: ignore
        
        # axes[0].sup_title(f"{self.src} -> {self.dest}")
        axes[0].set_title(f"Values")
        axes[1].set_title(f"Probs")
        df = self.to_frame(omit_first=True)
        anomalies = df[df['anomaly']]
        df['value'].plot(ax=axes[0])
        anomalies['value'].plot(ax=axes[0], style='ro')
        axes[0].legend(['value', 'anomaly'])
        df['prob'].plot(ax=axes[1])
        anomalies['prob'].plot(ax=axes[1], style='ro')
        axes[1].legend(['prob', 'anomaly'])


    @property
    def value(self) -> float:
        return self.values[-1]
    
    @value.setter
    def value(self, value: float):
        self.values.append(value)

    @property
    def alpha(self) -> float:
        return self.alphas[-1]
    
    @alpha.setter
    def alpha(self, alpha: float):
        self.alphas.append(alpha)

    @property
    def beta(self) -> float:
        return self.betas[-1]

    @beta.setter
    def beta(self, beta: float):
        self.betas.append(beta)

    @property
    def lambda_(self) -> float:
        return self.lambdas[-1]
    
    @lambda_.setter
    def lambda_(self, lambda_: float):
        self.lambdas.append(lambda_)

    @property
    def sigma(self) -> float:
        return self.sigmas[-1]
    
    @sigma.setter
    def sigma(self, sigma: float):
        self.sigmas.append(sigma)
    
    @property
    def prob(self) -> float:
        return self.probs[-1]

    @prob.setter
    def prob(self, prob: float):
        self.probs.append(prob)
    
    @property
    def anomaly(self) -> bool:
        return self.anomalies[-1]
    
    @anomaly.setter
    def anomaly(self, anomaly: bool):
        self.anomalies.append(anomaly)
