from enum import Enum

from typing import Any
from matplotlib import pyplot as plt
from scipy.stats import hmean

from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np

from traced_v2.models.base_model import BaseModel, Visual
from traced_v2.models.bernoulli import BernoulliModel
from traced_v2.models.graph import GraphModel
from traced_v2.models.multinomial import MultinomialModel
from traced_v2.models.normal import NormalModel
from traced_v2.models.poisson import PoissonModel
from traced_v2.utils import create_hash, add_prefix


class Mode(Enum):
    MEAN = "mean"
    WEIGHTED_MEAN = "w_mean"
    WEIGHTED_HARMONIC_MEAN = "w_h_mean"


class TraceModel(BaseModel, Visual):
    def __init__(
        self,
        src: str,
        dest: str,
        mode: Mode,
        parent: BaseModel | None = None,
    ) -> None:
        super().__init__(
            src, dest, subscription=parent.subscription if parent else None
        )
        self.rtt_models: dict[str, NormalModel] = {}
        self.ttl_models: dict[str, PoissonModel] = {}
        self.mode: Mode = Mode(mode)
        self.final_rtt: NormalModel = NormalModel(src, dest, parent=self)
        self.final_ttl: NormalModel = NormalModel(src, dest, parent=self)

        self.rtt_sum_errors = []
        self.ttl_sum_errors = []

        self.rtt_mean_errors = []
        self.ttl_mean_errors = []


    def _calculate_anomaly_weights(self, rtt_anomalies, ttl_anomalies)-> tuple[np.ndarray, np.ndarray]:
        rtt_anom_seq_weight = np.ones_like(rtt_anomalies, dtype=float)
        ttl_anom_seq_weight = np.ones_like(ttl_anomalies, dtype=float)
        # iterate from last to first
        
        rtt_c = 1
        ttl_c = 1
        
        for i in range(len(rtt_anomalies) - 1, 0, -1):
            rtt_anom_seq_weight[i] = rtt_c
            if rtt_anomalies[i]:
                rtt_c += 1

            ttl_anom_seq_weight[i] = ttl_c
            if ttl_anomalies[i]:
                ttl_c += 1
        
        # normalize
        ttl_anom_seq_weight /= ttl_anom_seq_weight.sum() # type: ignore
        rtt_anom_seq_weight /= rtt_anom_seq_weight.sum() # type: ignore

        return rtt_anom_seq_weight, ttl_anom_seq_weight

    def _process(
        self,
        ts: int,
        rtt_score: float,
        ttl_score: float,
        rtt_error: list[float],
        ttl_error: list[float],
    ) -> tuple[bool, float, float, bool, float, float]:
        mean_rtt_error: float = np.mean(rtt_error) # type: ignore
        sum_rtt_error: float = np.sum(rtt_error)

        mean_ttl_error: float = np.mean(ttl_error) # type: ignore
        sum_ttl_error: float = np.sum(ttl_error)

        self.rtt_sum_errors.append(sum_rtt_error)
        self.ttl_sum_errors.append(sum_ttl_error)
        self.rtt_mean_errors.append(mean_rtt_error)
        self.ttl_mean_errors.append(mean_ttl_error)

        rtt = self.final_rtt.log(ts, rtt_score)
        ttl = self.final_ttl.log(ts, ttl_score)
        return (rtt[0], mean_rtt_error, sum_rtt_error, ttl[0], mean_ttl_error, sum_ttl_error)

    def log(self, ts: int, hops: list[str], rtts: list[float], ttls: list[int]) -> tuple[bool, float, float, bool, float, float]:
        super().log_timestamp(ts)

        rtt_rate = []
        ttl_rate = []

        rtt_errors = []
        ttl_errors = []

        rtt_anomalies = []
        ttl_anomalies = []

        rtt_anomaly_total = []
        ttl_anomaly_total = []

        for hop, rtt, ttl in zip(hops, rtts, ttls):
            if hop not in self.rtt_models:
                self.rtt_models[hop] = NormalModel(self.src, hop, sigma_factor=4)
                self.ttl_models[hop] = PoissonModel(self.src, hop)

            val = self.rtt_models[hop].log(ts, rtt)
            # self.anomalies[-1], self.n_anomalies, mu, observed_value, sigma

            rtt_anomalies.append(val[0])
            rtt_anomaly_total.append(val[1])
            rtt_rate.append((1 + val[3]) / (1 + val[2]))
            # rtt_rate.append(val[3] - val[2])
            rtt_errors.append(val[3] - val[2])

            val = self.ttl_models[hop].log(ts, ttl)
            # (self.anomalies[-1], prob, self.lambdas[-1], observed_value, self.n_anomalies)
            ttl_anomalies.append(val[0])
            ttl_anomaly_total.append(val[4])
            ttl_rate.append((1 + val[3]) / (1 + val[2]))
            # ttl_rate.append(val[3] - val[2])
            ttl_errors.append(val[3] - val[2])

        if self.mode == Mode.MEAN:
            return self._process(
                ts,
                float(np.mean(rtt_errors)),
                float(np.mean(ttl_errors)),
                rtt_errors,
                ttl_errors,
            )
        uniform_weights = np.ones_like(rtt_anomaly_total, dtype=float)
        uniform_weights /= uniform_weights.sum() # type: ignore

        rtt_anomaly_weights = np.log1p(rtt_anomaly_total) 
        rtt_anomaly_weights /= rtt_anomaly_weights.sum() # type: ignore

        ttl_anomaly_weights = np.log1p(ttl_anomaly_total)
        ttl_anomaly_weights /= ttl_anomaly_weights.sum() # type: ignore

        rtt_anom_seq_weight, ttl_anom_seq_weight = self._calculate_anomaly_weights(rtt_anomalies, ttl_anomalies)
        
        if self.mode == Mode.WEIGHTED_MEAN:
            rtt_w =  np.mean([uniform_weights, rtt_anomaly_weights, rtt_anom_seq_weight], axis=0).T
            ttl_w = np.mean([uniform_weights, ttl_anomaly_weights, ttl_anom_seq_weight], axis=0).T
        
        else:
            rtt_w = hmean([uniform_weights, rtt_anomaly_weights, rtt_anom_seq_weight], axis=0).T
            ttl_w = hmean([uniform_weights, ttl_anomaly_weights, ttl_anom_seq_weight], axis=0).T
        
        rtt_rate = np.array(rtt_errors)
        ttl_rate = np.array(ttl_errors)

        return self._process(ts, rtt_rate @ rtt_w, ttl_rate @ ttl_w, rtt_errors, ttl_errors)

    def to_dict(self) -> dict[str, list[Any]]:
        return {
            **add_prefix("trace_rtt", self.final_rtt.to_dict()),
            **add_prefix("trace_ttl", self.final_ttl.to_dict()),
            "rtt_sum_errors": self.rtt_sum_errors,
            "ttl_sum_errors": self.ttl_sum_errors,
            "rtt_mean_errors": self.rtt_mean_errors,
            "ttl_mean_errors": self.ttl_mean_errors,
        }

    def plot(self, ax:  Axes | None = None, **kwargs):
        if ax is None:
            ax = plt.gca()

        df = self.to_frame()

        anomaly_cnt = df["trace_ttl_anomalies"] + df["trace_rtt_anomalies"]

        if 'resample' in kwargs:
            anomaly_cnt = anomaly_cnt.resample(kwargs['resample']).sum()

        plt.plot(kind='bar', label="Number of anomalies", ax=ax, **kwargs)

class TraceAnalyzer(BaseModel):
    def __init__(self, src: str, dest: str, *args, **kwargs) -> None:
        if "ip_model" in kwargs:
            ip_model = kwargs.pop("ip_model")
        else:
            ip_model = GraphModel.get_or_create_subscription(
                local=True, forgetting=True
            )

        if "as_model" in kwargs:
            as_model = kwargs.pop("as_model")
        else:
            as_model = GraphModel.get_or_create_subscription(
                local=True, forgetting=True
            )
        self.as_model_tag = as_model

        super().__init__(src, dest, *args, **kwargs)

        self.ip_model: GraphModel = GraphModel(
            src, dest, graph_subscription=ip_model, parent=self
        )
        self.as_model: GraphModel | None = None

        self.path_complete: BernoulliModel = BernoulliModel(src, dest, parent=self)
        self.destination_reached: BernoulliModel = BernoulliModel(
            src, dest, parent=self
        )
        self.looping: BernoulliModel = BernoulliModel(src, dest, parent=self)
        self.path_probs: MultinomialModel = MultinomialModel(src, dest, parent=self)

        self.n_hops_model: PoissonModel = PoissonModel(src, dest, parent=self)
        self.trace_model: TraceModel = TraceModel(
            src, dest, parent=self, mode=Mode.WEIGHTED_HARMONIC_MEAN
        )

    def log(self, data: dict[str, Any]) -> None:
        if "asns" not in data:
            return

        if self.as_model is None and data["hops"][-1] == self.dest:
            # lazy init the asn graph model, since we do not know the as apriori
            self.as_model = GraphModel(
                data["asns"][0],
                data["asns"][-1],
                parent=self,
                graph_subscription=self.as_model_tag,
            )

        ts = data["timestamp"]
        self.log_timestamp(ts)

        self.ip_model.log(ts, data["hops"])
        if self.as_model is not None:
            self.as_model.log(ts, data["asns"])

        delta_ttls = [ttl - i for i, ttl in enumerate(data["ttls"])]
        out = self.trace_model.log(
            ts, data["hops"], data["rtts"], delta_ttls
        )

        self.path_complete.log(ts, data["path_complete"])
        self.destination_reached.log(ts, data["destination_reached"])
        self.looping.log(ts, data["looping"])

        tmp = []
        for x in data["asns"]:
            if tmp and tmp[-1] == x:
                continue
            tmp.append(x)

        self.path_probs.log(ts, create_hash("".join(map(str, tmp))))
        self.n_hops_model.log(ts, len(data["hops"]))

    def to_dict(self) -> dict[str, list[Any]]:
        return {
            **add_prefix("ip", self.ip_model.to_dict()),
            **add_prefix("as", self.as_model.to_dict() if self.as_model else {}),
            **add_prefix("path_complete", self.path_complete.to_dict()),
            **add_prefix("destination_reached", self.destination_reached.to_dict()),
            **add_prefix("looping", self.looping.to_dict()),
            **add_prefix("path_probs", self.path_probs.to_dict()),
            **add_prefix("n_hops_model", self.n_hops_model.to_dict()),
            # **add_prefix("trace_model", self.trace_model.to_dict()),
        }
