'''Model for a single traceroute.'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.figure as figure

class Model:
    '''Model for a single traceroute.
    
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
        self.success_probs: list[float] = [np.nan]
        self.success_vars: list[float] = [np.nan]

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
        prob_changes = (df["success_prob"].nunique()>1)
        ttl_changes = (df["delta_ttl"].nunique() > 1)
        # create figure with 3 rows of subplots
        fig, axs = plt.subplots(sum([1, prob_changes, ttl_changes]), 1, sharex=True)
        axs = axs.ravel()
        fig.set_size_inches(15, 9)
        fig.suptitle(f"Statistics {self.u} -> {self.v}")
        # TODO: split figures

        if not prob_changes:
            axs = (axs, axs, axs)

        # render anomaly plot
        mu_plus_3sigma = df["mu"] + 3*df["sigma"]
        mu_minus_3sigma = df["mu"] - 3*df["sigma"]

        mu_plus_3sigma.plot(ax=axs[0], color='green', alpha=0.5) 
        mu_minus_3sigma.plot(ax=axs[0], color='green',label='$\\pm3\\sigma$', alpha=0.5)

        (mu_plus_3sigma + df["sigma"]).plot(ax=axs[0], color='purple', alpha=0.5)
        (mu_minus_3sigma - df["sigma"]).plot(ax=axs[0], color='purple', label='$\\pm4\\sigma$', alpha=0.5)

        df['delta_rtt'].plot(ax=axs[0], label="$\\Delta\\mathrm{rtt}$")
        df["mu"].plot(ax=axs[0], label="$\\mathbb{E}(rtt)$")

        anomalies = df[(mu_minus_3sigma >= df["delta_rtt"]) | (df["delta_rtt"] >= mu_plus_3sigma)]
        anomalies.plot(y="delta_rtt", ax=axs[0], color="red", marker='o', linestyle = 'None', label="anomaly")  
        
          
        axs[0].set_title(f"Anomalies on RTT ({anomalies.shape[0]}, {100*anomalies.shape[0]/df.shape[0]:.3f}%)")
        axs[0].legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=5, fancybox=True)  
        axs[0].set_xlabel("time")
        axs[0].set_ylabel("$\\Delta\\mathrm{rtt}$")

        if prob_changes:

            # plot probability
            df["success_prob"].plot(ax=axs[1], label="$P(\\mathrm{success})$", color='green')
            (df['success_prob']+3*df["success_var"].apply(np.sqrt)).plot(ax=axs[1], label="$\\sigma(\\mathrm{success})$", color='orange')

            axs[2].set_title(f"Probability of success")
            axs[1].set_ylabel("$p$")

        if ttl_changes:
            # plot variance
            df["delta_ttl"].plot(ax=axs[2], label="$\\Delta(\\mathrm{ttl})$", color='red')
            axs[2].set_ylabel("$\\Delta(\\mathrm{ttl})")
            axs[2].set_title(f"TTL changes")

        return fig

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

    