from enum import Enum
from typing import Any

import numpy as np
import pydantic
from matplotlib import pyplot as plt
from scipy.stats import hmean

from traced_v2.models.base_model import BaseModel, Visual
from traced_v2.models.exponential import ExponentialModel
from traced_v2.models.normal import NormalModel, NormalModelOutput
from traced_v2.models.poisson import PoissonModel
from traced_v2.utils import add_prefix


def filter_zeros(items: list[float]) -> list[float]:
    tmp = [x for x in items if x] or [0]

    # if len(tmp) == 1:
    #     return tmp + [0] * (len(items)//2)

    return tmp


class Mode(Enum):
    """Agggregation mode for the trace model."""

    MEAN = "mean"
    SUM = "sum"
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
        shift: bool = False,
        sigma_factor: float = 4,
        ignore_zero_errors: bool = True,
    ) -> None:
        super().__init__(
            src, dest, subscription=parent.subscription if parent else None
        )
        self.rtt_models: dict[str, ExponentialModel] = {}
        self.ttl_models: dict[str, PoissonModel] = {}
        self.mode: Mode = Mode(mode)
        self.shift = None if shift else 0
        self.final_rtt: NormalModel = NormalModel(
            src,
            dest,
            parent=self,
            alpha_0=15,
            sigma_factor=sigma_factor,
        )
        self.final_ttl: NormalModel = NormalModel(
            src,
            dest,
            parent=self,
            alpha_0=10,
            mu_0=0,
            sigma_factor=sigma_factor,
        )
        self.ignore_zero_errors = ignore_zero_errors

        self.rtt_sum_errors = []
        self.ttl_sum_errors = []

        self.rtt_mean_errors = []
        self.ttl_mean_errors = []

        self.ttl_errors = []
        self.rtt_errors = []

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
        ns = []
        for i, (hop, rtt, ttl) in enumerate(zip(hops, rtts, ttls)):
            if hop not in self.rtt_models:
                self.rtt_models[hop] = ExponentialModel(
                    self.src, hop, alpha_0=1, threshold=0.05, shift=0
                )
                self.ttl_models[hop] = PoissonModel(
                    self.src, hop, threshold=0.05, gamma=0.5, shift=self.shift
                )

            val = self.rtt_models[hop].log(ts, rtt)
            rtt_anomalies.append(val.is_anomaly)
            rtt_anomaly_total.append(val.n_anomalies)
            rtt_errors.append(val.error)
            rtt_pdf.append(val.sf)

            val = self.ttl_models[hop].log(ts, ttl)

            ttl_anomalies.append(val.is_anomaly)
            ttl_anomaly_total.append(val.n_anomalies)
            ttl_errors.append(val.error)
            ttl_pdf.append(val.pdf)
            ns.append(self.ttl_models[hop].n)

        # self.rtt_errors.append(rtt_errors)
        # self.ttl_errors.append(ttl_errors)

        if self.mode == Mode.MEAN:
            return self._process(
                ts,
                float(
                    np.mean(
                        rtt_errors
                        if not self.ignore_zero_errors
                        else filter_zeros(rtt_errors)
                    )
                ),
                float(
                    np.mean(
                        ttl_errors
                        if not self.ignore_zero_errors
                        else filter_zeros(ttl_errors)
                    )
                ),
                # float(np.mean([x for x in (ttl_errors + [1]) if x])),
                rtt_errors,
                ttl_errors,
                rtt_pdf,
                ttl_pdf,
            )
        if self.mode == Mode.SUM:
            return self._process(
                ts,
                float(np.sum(rtt_errors)),
                float(np.sum(ttl_errors)),
                rtt_errors,
                ttl_errors,
                rtt_pdf,
                ttl_pdf,
            )
        uniform_w = np.ones_like(rtt_anomaly_total, dtype=float)
        uniform_w /= uniform_w.sum()  # type: ignore

        rtt_anomaly_w = np.log1p(rtt_anomaly_total)
        rtt_anomaly_w /= rtt_anomaly_w.sum()  # type: ignore
        # rtt_anomaly_w = 1 - np.array(rtt_anomaly_w)

        ttl_anomaly_w = np.log1p(ttl_anomaly_total)
        ttl_anomaly_w /= ttl_anomaly_w.sum()  # type: ignore
        # ttl_anomaly_w = 1 - np.array(ttl_anomaly_w)

        rtt_seq_w, ttl_anom_seq_w = self._calculate_anomaly_weights(
            rtt_anomalies, ttl_anomalies
        )

        if self.mode == Mode.WEIGHTED_MEAN:
            rtt_w = np.mean([rtt_anomaly_w, rtt_seq_w], axis=0).T
            ttl_w = np.mean([ttl_anomaly_w, ttl_anom_seq_w], axis=0).T

        else:
            rtt_w = hmean([rtt_anomaly_w, rtt_seq_w]).T
            ttl_w = hmean([ttl_anomaly_w, ttl_anom_seq_w]).T

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
            # "ttl_errors": self.ttl_errors,
            # "rtt_errors": self.rtt_errors,
        }

    def plot(self, fig: plt.Figure | None = None, **kwargs):
        fig = fig or plt.figure()

        ax1, ax2 = fig.subplots(2, 1, sharex=True)

        self.final_rtt.plot(ax=ax1, **kwargs, kind="RTT")
        ax1.legend(bbox_to_anchor=(1.02, 0.95), loc=2, borderaxespad=0.0)
        self.final_ttl.plot(ax=ax2, **kwargs, kind="TTL")
        ax2.legend(bbox_to_anchor=(1.02, 0.95), loc=2, borderaxespad=0.0)

        # ax = ax or plt.gca()

        # anomaly_cnt = df["trace_ttl_anomalies"] + df["trace_rtt_anomalies"]

        # if "resample" in kwargs:
        #     anomaly_cnt = anomaly_cnt.resample(kwargs["resample"]).sum()

        # plt.plot(kind="bar", label="Number of anomalies", ax=ax, **kwargs)
        plt.suptitle(
            f"Number of anomalies on RTT and TTL for {self.src} -> {self.dest}"
        )
