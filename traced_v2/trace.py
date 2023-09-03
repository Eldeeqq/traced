from enum import Enum
from typing import Any

import numpy as np
import pydantic
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from scipy.stats import hmean

from traced_v2.models.base_model import BaseModel, Visual
from traced_v2.models.bernoulli import BernoulliModel, BernoulliModelOutput
from traced_v2.models.graph import GraphModel, GraphModelOutput
from traced_v2.models.multinomial import ForgettingMultinomialModel, MultinomialModelOutput
from traced_v2.models.normal import NormalModel, NormalModelOutput
from traced_v2.models.poisson import PoissonModel, PoissonModelOutput
from traced_v2.utils import add_prefix, create_hash


class Mode(Enum):
    """Agggregation mode for the trace model."""

    MEAN = "mean"
    WEIGHTED_MEAN = "w_mean"
    WEIGHTED_HARMONIC_MEAN = "w_h_mean"


class TraceModelOutput(pydantic.BaseModel):
    rtt_result: NormalModelOutput
    rtt_mean_error: float
    rtt_sum_error: float
    rtt_pdfs: list[float]

    ttl_result: NormalModelOutput
    ttl_mean_error: float
    ttl_sum_error: float
    ttl_pdfs: list[float]


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

    def _calculate_anomaly_weights(
        self, rtt_anomalies, ttl_anomalies
    ) -> tuple[np.ndarray, np.ndarray]:
        rtt_anom_seq_weight = np.ones_like(rtt_anomalies, dtype=float)
        ttl_anom_seq_weight = np.ones_like(ttl_anomalies, dtype=float)
        # iterate from last to first

        rtt_c = 1
        ttl_c = 1

        # calculate the index of node in anomaly sub-sequence
        for i in range(len(rtt_anomalies) - 1, 0, -1):
            rtt_anom_seq_weight[i] = rtt_c
            if rtt_anomalies[i]:
                rtt_c += 1

            ttl_anom_seq_weight[i] = ttl_c
            if ttl_anomalies[i]:
                ttl_c += 1

        # normalize
        ttl_anom_seq_weight /= ttl_anom_seq_weight.sum()  # type: ignore
        rtt_anom_seq_weight /= rtt_anom_seq_weight.sum()  # type: ignore

        return rtt_anom_seq_weight, ttl_anom_seq_weight

    def _process(
        self,
        ts: int,
        rtt_score: float,
        ttl_score: float,
        rtt_error: list[float],
        ttl_error: list[float],
        rtt_pdf: list[float],
        ttl_pdf: list[float],
    ) -> TraceModelOutput:
        mean_rtt_error: float = np.mean(rtt_error)  # type: ignore
        sum_rtt_error: float = np.sum(rtt_error)

        mean_ttl_error: float = np.mean(ttl_error)  # type: ignore
        sum_ttl_error: float = np.sum(ttl_error)

        self.rtt_sum_errors.append(sum_rtt_error)
        self.ttl_sum_errors.append(sum_ttl_error)
        self.rtt_mean_errors.append(mean_rtt_error)
        self.ttl_mean_errors.append(mean_ttl_error)

        rtt = self.final_rtt.log(ts, rtt_score)
        ttl = self.final_ttl.log(ts, ttl_score)

        return TraceModelOutput(
            rtt_result=rtt,
            rtt_mean_error=mean_rtt_error,
            rtt_sum_error=sum_rtt_error,
            rtt_pdfs=rtt_pdf,
            ttl_result=ttl,
            ttl_mean_error=mean_ttl_error,
            ttl_sum_error=sum_ttl_error,
            ttl_pdfs=ttl_pdf,
        )

    def log(
        self, ts: int, hops: list[str], rtts: list[float], ttls: list[int]
    ) -> TraceModelOutput:
        super().log_timestamp(ts)

        rtt_errors = []
        ttl_errors = []

        rtt_anomalies = []
        ttl_anomalies = []

        rtt_anomaly_total = []
        ttl_anomaly_total = []

        rtt_pdf = []
        ttl_pdf = []
        for hop, rtt, ttl in zip(hops, rtts, ttls):
            if hop not in self.rtt_models:
                self.rtt_models[hop] = NormalModel(self.src, hop)
                self.ttl_models[hop] = PoissonModel(self.src, hop)

            val = self.rtt_models[hop].log(ts, rtt)

            rtt_anomalies.append(val.is_anomaly)
            rtt_anomaly_total.append(val.n_anomalies)
            rtt_errors.append(val.error)
            rtt_pdf.append(val.pdf)

            val = self.ttl_models[hop].log(ts, ttl)

            ttl_anomalies.append(val.is_anomaly)
            ttl_anomaly_total.append(val.n_anomalies)
            ttl_errors.append(val.error)
            ttl_pdf.append(val.pdf)

        if self.mode == Mode.MEAN:
            return self._process(
                ts,
                float(np.mean(rtt_errors)),
                float(np.mean(ttl_errors)),
                rtt_errors,
                ttl_errors,
                rtt_pdf,
                ttl_pdf,
            )
        uniform_w = np.ones_like(rtt_anomaly_total, dtype=float)
        uniform_w /= uniform_w.sum()  # type: ignore

        rtt_anomaly_w = np.log1p(rtt_anomaly_total)
        rtt_anomaly_w /= rtt_anomaly_w.sum()  # type: ignore

        ttl_anomaly_w = np.log1p(ttl_anomaly_total)
        ttl_anomaly_w /= ttl_anomaly_w.sum()  # type: ignore

        rtt_seq_w, ttl_anom_seq_w = self._calculate_anomaly_weights(
            rtt_anomalies, ttl_anomalies
        )

        if self.mode == Mode.WEIGHTED_MEAN:
            rtt_w = np.mean([uniform_w, rtt_anomaly_w, rtt_seq_w], axis=0).T
            ttl_w = np.mean([uniform_w, ttl_anomaly_w, ttl_anom_seq_w], axis=0).T

        else:
            rtt_w = hmean([uniform_w, rtt_anomaly_w, rtt_seq_w], axis=0).T
            ttl_w = hmean([uniform_w, ttl_anomaly_w, ttl_anom_seq_w], axis=0).T

        return self._process(
            ts,
            np.array(rtt_errors) @ rtt_w,
            np.array(ttl_errors) @ ttl_w,
            rtt_errors,
            ttl_errors,
            rtt_pdf,
            ttl_pdf,
        )

    def to_dict(self) -> dict[str, list[Any]]:
        return {
            **add_prefix("trace_rtt", self.final_rtt.to_dict()),
            **add_prefix("trace_ttl", self.final_ttl.to_dict()),
            "rtt_sum_errors": self.rtt_sum_errors,
            "ttl_sum_errors": self.ttl_sum_errors,
            "rtt_mean_errors": self.rtt_mean_errors,
            "ttl_mean_errors": self.ttl_mean_errors,
        }

    def plot(self, ax: Axes | None = None, **kwargs):
        df = self.to_frame()
        ax = ax or plt.gca()

        anomaly_cnt = df["trace_ttl_anomalies"] + df["trace_rtt_anomalies"]

        if "resample" in kwargs:
            anomaly_cnt = anomaly_cnt.resample(kwargs["resample"]).sum()

        plt.plot(kind="bar", label="Number of anomalies", ax=ax, **kwargs)


class TraceAnalyzerOutput(pydantic.BaseModel):
    ip_model: GraphModelOutput
    as_model: GraphModelOutput | None
    path_complete: BernoulliModelOutput 
    destination_reached: BernoulliModelOutput
    looping: BernoulliModelOutput
    n_hops_model: PoissonModelOutput
    trace_model: TraceModelOutput
    path_probs: MultinomialModelOutput


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
        self.path_probs: ForgettingMultinomialModel = ForgettingMultinomialModel(
            src, dest, parent=self
        )

        self.n_hops_model: PoissonModel = PoissonModel(src, dest, parent=self)
        self.trace_model: TraceModel = TraceModel(
            src, dest, parent=self, mode=Mode.WEIGHTED_HARMONIC_MEAN
        )
        self.missing_data: BernoulliModel = BernoulliModel(src, dest)

    def log(self, data: dict[str, Any]) -> tuple[None | TraceAnalyzerOutput, BernoulliModelOutput]:
        ts = data["timestamp"]
        num_anomalies = 0
        try:
            # TODO: Incorporate ModelOutputs into the trace model
            if self.as_model is None and data["hops"][-1] == self.dest:
                # lazy init the asn graph model, since we do not know the as apriori
                self.as_model = GraphModel(
                    data["asns"][0],
                    data["asns"][-1],
                    parent=self,
                    graph_subscription=self.as_model_tag,
                )

            self.log_timestamp(ts)

            ip_model_output = self.ip_model.log(ts, data["hops"])
            num_anomalies += int(ip_model_output.is_anomaly)
        
            as_model_output = None
            if self.as_model is not None:
                as_model_output = self.as_model.log(ts, data["asns"])
                num_anomalies += int(as_model_output.is_anomaly)

            delta_ttls = [ttl - i for i, ttl in enumerate(data["ttls"])] # substract index from TTL
            trace_output = self.trace_model.log(ts, data["hops"], data["rtts"], delta_ttls)
            num_anomalies += (int(trace_output.rtt_result.is_anomaly) + int(trace_output.ttl_result.is_anomaly))

            path_complete_output = self.path_complete.log(ts, data["path_complete"])
            # num_anomalies += int(path_complete_output.is_anomaly) # TODO: uncomment when bernoulli detect anomalies
            dest_reached = self.destination_reached.log(ts, data["destination_reached"])
            # num_anomalies += int(dest_reached.is_anomaly) # TODO: uncomment when bernoulli detect anomalies
            looping = self.looping.log(ts, data["looping"])
            # num_anomalies += int(looping.is_anomaly) # TODO: uncomment when bernoulli detect anomalies

            path_probs = self.path_probs.log(ts, create_hash("".join(map(str, set(data["asns"])))))
            # num_anomalies += int(path_probs.is_anomaly) # TODO: uncomment when multinomial detects anomalies
            n_hops = self.n_hops_model.log(ts, len(data["hops"]))
            
            num_anomalies += int(n_hops.is_anomaly)
            return TraceAnalyzerOutput(
                ip_model=ip_model_output,
                as_model=as_model_output,
                trace_model = trace_output,
                path_complete = path_complete_output,
                destination_reached=dest_reached, 
                looping=looping,
                n_hops_model=n_hops,
                path_probs=path_probs,
            ), self.missing_data.log(ts, False)
        
        except (AttributeError, KeyError) as e:
             # TODO: mark this traceroute as invalid
            pass
            
        return None, self.missing_data.log(ts, True)

    def to_dict(self) -> dict[str, list[Any]]:
        return {
            **add_prefix("ip", self.ip_model.to_dict()),
            **add_prefix("as", self.as_model.to_dict() if self.as_model else {}),
            **add_prefix("path_complete", self.path_complete.to_dict()),
            **add_prefix("destination_reached", self.destination_reached.to_dict()),
            **add_prefix("looping", self.looping.to_dict()),
            **add_prefix("path_probs", self.path_probs.to_dict()),
            **add_prefix("n_hops_model", self.n_hops_model.to_dict()),
            **add_prefix("trace_model", self.trace_model.to_dict()),
        }
