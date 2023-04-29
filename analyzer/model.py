'''Model for a single edge in graph.'''

import numpy as np
import pandas as pd

from scipy.stats import norm, invgamma
from typing import Any
import matplotlib.pyplot as plt
import matplotlib.figure as figure

class Model:
    '''Model for a single edge in graph.
    
    Attributes:
        u (str): source
        v (str): destination
        ctr (int): number of measurements
        
        alphas (list[float]): alpha values for inverse gamma distribution
        betas (list[float]): beta values for inverse gamma distribution
        mus (list[float]): mu values for normal distribution
        sigmas (list[float]): sigma values for normal distribution
        tss (list[int]): timestamps
        ns (list[int]): number of measurements
        delta_rtts (list[float]): difference in RTT
        delta_ttls (list[float]): difference in TTL
        successes (list[int]): number of successes
        failures (list[int]): number of failures
        
        finished (bool): whether the last traceroute has finished
        timestamp (int): timestamp of last traceroute
        delta_rtt (float): difference in RTT of last traceroute
        delta_ttl (float): difference in TTL of last traceroute
        alpha (float): alpha of last traceroute
        beta (float): beta of last traceroute
        mu (float): mu value of last traceroute
        sigma (float): sigma value of last traceroute

    '''
    
    def __init__(self, u, v, alpha_0=1, beta_0=1, mu_0=5, sigma_0=2):
        '''Initialize a new model.
        
        Args:
            u (str): source
            v (str): destination
            alpha_0 (int, optional): initial alpha value for inverse gamma distribution. Defaults to 1.
            beta_0 (int, optional): initial beta value for inverse gamma distribution. Defaults to 1.
            mu_0 (int, optional): prior knowledge about expected delta rtt. Defaults to 5.
            sigma_0 (int, optional): prior knowledge about std of delta rtt. Defaults to 2.
                
        '''
        self.u: str = u
        self.v: str = v
        self.ctr: int = 0

        self.alphas: list[float] = [alpha_0]
        self.betas: list[float] = [beta_0]
        self.mus: list[float] = [mu_0]
        self.sigmas: list[float] = [sigma_0]

        self.tss: list[int] = [0]
        self.ns: list[int] = [0]

        self.delta_rtts: list[float] = [0]
        self.delta_ttls: list[float] = [0]

        self.successes: list[int] = [0]
        self.failures: list[int] = [0]
        self.success_probs: list[float] = [0.5]
        self.success_vars: list[float] = [0.25]

    def log(self, ts, delta_rtt, delta_ttl, success) -> None:
        '''Log a new measurement.
        
        Args:
            ts (int): timestamp
            delta_rtt (float): difference in RTT
            delta_ttl (float): difference in TTL
            success (bool): whether the traceroute was successful

        Raises:
            ValueError: if logged timestamps is older than last one

        '''
        if self.timestamp is not None and ts < self.timestamp:
            raise ValueError(f"Timestamps must be in order (ts>{self.timestamp})")
        
        self.ctr += 1
        self.timestamp = ts
        self.delta_rtt = delta_rtt
        self.delta_ttl = delta_ttl

        self.alpha = self.alpha + 1/2
        self.n += 1

        self.beta = self.beta + 0.5*(delta_rtt-self.mu)**2
        self.mu = self.mu + 1/self.n * (delta_rtt-self.mu)
        self.sigma = np.sqrt(self.beta/(self.alpha + 1))
        self.finished = success
    
    def __repr__(self) -> str:
        '''String representation of the model.'''
        return f"{self.u} -> {self.v} (#{self.ctr} N({self.mu:.2f}, {self.sigma:.2f}))"

    def to_frame(self, omit_first=True) -> pd.DataFrame:
        '''Convert the model statistics to a pandas DataFrame.'''
        df = pd.DataFrame({
            "delta_rtt": self.delta_rtts,
            "delta_ttl": self.delta_ttls,
            "ts": self.tss,
            "alpha": self.alphas,
            "beta": self.betas,
            "mu": self.mus,
            "sigma": self.sigmas,
            "successes": self.successes,
            "failures": self.failures,
            "success_prob": self.success_probs,
            "success_var": self.success_vars,
            # "n": self.ns
        }, index=pd.DatetimeIndex(pd.to_datetime(self.tss, unit="ms")))

        df["u"] = self.u
        df["v"] = self.v
        df["total"] = self.ctr
        
        if omit_first:
            df = df.iloc[1:]
            
        return df

    def plot(self, **kwargs) -> figure.Figure:
        '''Plot the model statistics.'''
        # ignore type checks
        
        # get data
        df = self.to_frame(omit_first=True)
        prob_changes = (df["success_prob"].unique().shape[0] > 1)
        ttl_changes = (df["delta_ttl"].unique().shape[0] > 1)
        n_axis = sum([1, prob_changes, ttl_changes])
        # create figure with 3 rows of subplots
        fig, axs = plt.subplots(n_axis, 1, sharex=True)

        # fig = plt.figure()
        fig.suptitle(f"Statistics {self.u} -> {self.v}")
        fig.set_size_inches(15, 9)

        ax: plt.Axes = axs[0] if n_axis > 1 else axs # type: ignore

        # render anomaly plot
        mu_plus_3sigma = df["mu"] + 3*df["sigma"]
        mu_minus_3sigma = df["mu"] - 3*df["sigma"]

        ax.fill_between(df.index, mu_plus_3sigma, mu_minus_3sigma, color='green', alpha=0.1) # type: ignore

        (mu_plus_3sigma + df["sigma"]).plot(ax=ax, color='purple', alpha=0.5)
        (mu_minus_3sigma - df["sigma"]).plot(ax=ax, color='purple', label='$\\pm4\\sigma$', alpha=0.5)

        df['delta_rtt'].plot(ax=ax, label="$\\Delta\\mathrm{rtt}$")
        df["mu"].plot(ax=ax, label="$\\mathbb{E}(rtt)$")

        anomalies = df[(mu_minus_3sigma >= df["delta_rtt"]) | (df["delta_rtt"] >= mu_plus_3sigma)]
        anomalies.plot(y="delta_rtt", ax=ax, color="red", marker='o', linestyle = 'None', label="anomaly")  
          
        ax.set_title(f"Anomalies on RTT ({anomalies.shape[0]}, {100*anomalies.shape[0]/df.shape[0]:.3f}%)")
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=5, fancybox=True)  
        ax.set_xlabel("time")
        ax.set_ylabel("$\\Delta\\mathrm{rtt}$")

        if prob_changes:
            # plot probability
            ax: plt.Axes = axs[1]
            df["success_prob"].plot(ax=ax, label="$P(\\mathrm{success})$", color='green')
            upper_bound = df["success_prob"] + 3*df["success_var"].apply(np.sqrt)
            lower_bound = df["success_prob"] - 3*df["success_var"].apply(np.sqrt)
            ax.fill_between(df.index, lower_bound, upper_bound, color='gray', alpha=0.2) # type: ignore
            ax.set_ylabel("$P(\\mathrm{success})$")
            ax.set_title(f"Probability of success")
            ax.set_ylabel("$p$")
            ax.set_ylim (max(0, lower_bound.min())-0.01, min(1, upper_bound.max())+0.01)

        if ttl_changes:
            # plot variance
            ax: plt.Axes = axs[-1]
            df["delta_ttl"].plot(kind='line', ax=ax, label="$\\Delta(\\mathrm{ttl})$", color='red')
            ax.set_ylabel("$\\Delta(\\mathrm{ttl})")
            ax.set_title(f"TTL changes")


        return fig

    def score(self, rtt, ttl, destination_reached) -> tuple[bool, float, float, float, float, float]:
        rtt_is_outlier = rtt > self.mu + 3*self.sigma or rtt < self.mu - 3*self.sigma
        rtt_prob = self.pdf(rtt) 
        rtt_mu_diff = rtt - self.mu
        success_prob = self.success_prob
        success_score = destination_reached - self.success_prob
        rtt_ttl_rate = rtt/ttl
        return rtt_is_outlier, rtt_prob, rtt_mu_diff, success_prob, success_score, rtt_ttl_rate
    
    def pdf(self, x)->float:
        return np.exp(-0.5*((x-self.mu)/self.sigma)**2)/(self.sigma*np.sqrt(2*np.pi))
    
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
    def delta_rtt(self):
        return self.delta_rtts[-1] if self.delta_rtts else None
    
    @delta_rtt.setter
    def delta_rtt(self, delta_rtt):
        self.delta_rtts.append(delta_rtt)

    @property
    def delta_ttl(self):
        return self.delta_ttls[-1] if self.delta_ttls else None
    
    @delta_ttl.setter
    def delta_ttl(self, delta_ttl):
        self.delta_ttls.append(delta_ttl)

    @property
    def finished(self):
        return self.successes[-1], self.failures[-1]
    
    @finished.setter
    def finished(self, success):
        a, b = self.finished
        a, b  = a + 1 if success else a, b + 1 if not success else b
        self.successes.append(a)
        self.failures.append(b)
        self.success_prob = a / (a + b)
        self.success_var = (a * b) / ((a + b)**2 * (a + b + 1))

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

    @property
    def n(self):
        return self.ns[-1]
    
    @n.setter
    def n(self, n):
        self.ns.append(n)

    