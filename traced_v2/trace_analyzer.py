from typing import Any

import numpy as np
import pandas as pd
import pydantic

from traced_v2.models.base_model import BaseModel
from traced_v2.models.bernoulli import BernoulliModel, BernoulliModelOutput
from traced_v2.models.graph import GraphModel, GraphModelOutput
from traced_v2.models.multinomial import (ForgettingMultinomialModel,
                                          MultinomialModelOutput)
from traced_v2.models.poisson import PoissonModel, PoissonModelOutput
from traced_v2.models.trace_model import Mode, TraceModel, TraceModelOutput
from traced_v2.utils import add_prefix, create_hash


def jaccaard_similarity(list1, list2):
    s1 = set(list1)
    s2 = set(list2)
    if len(s2) == 0:
        return 0
    return len(s1.intersection(s2)) / len(s1.union(s2))


class AnomalyReport(pydantic.BaseModel):
    ip_model_anomaly: bool
    as_model_anomaly: bool
    trace_rtt_anomaly: bool
    trace_ttl_anomaly: bool
    path_complete_anomaly: bool
    destination_reached_anomaly: bool
    looping_anomaly: bool
    as_path_probs_anomaly: bool
    ip_path_probs_anomaly: bool
    n_hops_model_anomaly: bool

    @property
    def total(self) -> int:
        return sum(self.dict().values())


class TraceAnalyzerOutput(pydantic.BaseModel):
    ip_model: GraphModelOutput
    as_model: GraphModelOutput | None
    path_complete: BernoulliModelOutput
    destination_reached: BernoulliModelOutput
    looping: BernoulliModelOutput
    n_hops_model: PoissonModelOutput
    trace_model: TraceModelOutput
    path_probs: MultinomialModelOutput
    ip_path_probs: MultinomialModelOutput
    anomalies: AnomalyReport


class TraceAnalyzer(BaseModel):
    def __init__(self, src: str, dest: str, *args, **kwargs) -> None:
        if "ip_model" in kwargs:
            ip_model = kwargs.pop("ip_model")
        else:
            ip_model = GraphModel.get_or_create_subscription(
                local=False, forgetting=True
            )

        if "as_model" in kwargs:
            as_model = kwargs.pop("as_model")
        else:
            as_model = GraphModel.get_or_create_subscription(
                local=False, forgetting=True
            )
        self.as_model_tag = as_model

        super().__init__(src, dest, *args, **kwargs)

        self.ip_model: GraphModel = GraphModel(
            src, dest, graph_subscription=ip_model, parent=self
        )
        # self.ip_hash_hierarchy: HashHierarchy = HashHierarchy()

        self.as_model: GraphModel = GraphModel(
            self.src,
            self.dest,
            parent=self,
            graph_subscription=self.as_model_tag,
        )
        # self.as_hash_hierarchy: HashHierarchy = HashHierarchy()

        self.path_complete: BernoulliModel = BernoulliModel(
            src,
            dest,
            parent=self,
            scorer=lambda x, p: x if p <= 0.1 else not x if p >= 0.9 else False,
        )
        self.destination_reached: BernoulliModel = BernoulliModel(
            src,
            dest,
            parent=self,
            scorer=lambda x, p: x if p <= 0.2 else not x if p >= 0.8 else False,
        )
        self.looping: BernoulliModel = BernoulliModel(
            src, dest, parent=self, scorer=lambda x, _: x
        )

        self.path_probs: ForgettingMultinomialModel = ForgettingMultinomialModel(
            src, dest, parent=self
        )
        self.ip_path_probs: ForgettingMultinomialModel = ForgettingMultinomialModel(
            src, dest, parent=self
        )

        self.n_hops_model: PoissonModel = PoissonModel(
            src, dest, parent=self, threshold=0.05
        )
        self.trace_model: TraceModel = TraceModel(
            src,
            dest,
            mode=Mode.MEAN,
            parent=self,
        )
        self.missing_data: BernoulliModel = BernoulliModel(
            src, dest, scorer=lambda x, _: x
        )
        self.n_anomalies: list[AnomalyReport] = []
        self.anomalies_model = PoissonModel(src, dest, parent=self, threshold=0.025)
        self.ip_path_dict = {}
        self.as_path_dict = {}

    def check_data_validity(self, data: dict[str, Any]) -> bool:
        """Checks that are needed data is present in the data object."""
        for key in [
            "hops",
            "rtts",
            "ttls",
            "path_complete",
            "destination_reached",
            "looping",
        ]:
            if key not in data:
                print(f"Missing key {key} in data")
                return False
        if not data["hops"] or not data["asns"]:
            print("Empty hops or asns")
            return False

        return True

    def find_closest_as_path(
        self,
        hashed: str,
        path: list[str | int],
        path_complete: bool,
        ip: bool = False,
        threshold: float = 0.8,
    ) -> str:
        hashes = self.as_path_dict if not ip else self.ip_path_dict

        if path_complete:
            if hashed not in hashes:
                hashes[hashed] = set(path)
        else:
            if hashes:
                scores = [jaccaard_similarity(path, hashes[x]) for x in hashes]
                max_idx = np.argmax(scores)
                if scores[max_idx] > threshold:
                    hashed = list(hashes.keys())[max_idx]

        return hashed

    def log(self, data: dict[str, Any]) -> TraceAnalyzerOutput | AnomalyReport:
        ts = data["timestamp"]

        if not self.check_data_validity(data):
            if (
                not self.missing_data.timestamps
                or self.missing_data.timestamps[-1] != ts
            ):
                self.missing_data.log(ts, True)
            return AnomalyReport(**{k: True for k in AnomalyReport.__fields__.keys()})

        self.log_timestamp(ts)

        ip_model_output = self.ip_model.log(ts, data["hops"])

        as_model_output = self.as_model.log(ts, data["asns"])

        delta_ttls = [
            ttl - i for i, ttl in enumerate(data["ttls"])
        ]  # substract index from TTL
        trace_output = self.trace_model.log(ts, data["hops"], data["rtts"], delta_ttls)

        path_complete_output = self.path_complete.log(ts, data["path_complete"])

        dest_reached = self.destination_reached.log(ts, data["destination_reached"])

        looping = self.looping.log(ts, data["looping"])

        path = set(map(str, [data["src"]] + data["asns"]))
        as_path_hash = create_hash("".join(path))
        final_as_hash = self.find_closest_as_path(
            as_path_hash, path, data["path_complete"], ip=False
        )

        path_probs = self.path_probs.log(
            # ts, create_hash("".join(map(str, set(data["asns"]))))
            ts,
            final_as_hash,
        )
        path = set([data["src"]] + data["hops"])
        ip_path_hash = create_hash("".join(path))
        final_ip_hash = self.find_closest_as_path(
            ip_path_hash, path, data["path_complete"], ip=True
        )

        ip_path_probs = self.ip_path_probs.log(
            # ts, create_hash("".join(map(str, set(data["hops"]))))
            ts,
            final_ip_hash,
        )
        n_hops = self.n_hops_model.log(ts, len(data["hops"]))

        anomalies = AnomalyReport(
            ip_model_anomaly=ip_model_output.is_anomaly,
            as_model_anomaly=as_model_output.is_anomaly,
            trace_rtt_anomaly=trace_output.rtt_result.is_anomaly,
            trace_ttl_anomaly=trace_output.ttl_result.is_anomaly,
            path_complete_anomaly=path_complete_output.is_anomaly,
            destination_reached_anomaly=dest_reached.is_anomaly,
            looping_anomaly=looping.is_anomaly,
            as_path_probs_anomaly=path_probs.anomaly,
            ip_path_probs_anomaly=ip_path_probs.anomaly,
            n_hops_model_anomaly=n_hops.is_anomaly,
        )
        self.anomalies_model.log(ts, anomalies.total)

        return TraceAnalyzerOutput(
            ip_model=ip_model_output,
            as_model=as_model_output,
            trace_model=trace_output,
            path_complete=path_complete_output,
            destination_reached=dest_reached,
            looping=looping,
            n_hops_model=n_hops,
            path_probs=path_probs,
            ip_path_probs=ip_path_probs,
            anomalies=anomalies,
        )

    def to_dict(self) -> dict[str, list[Any]]:
        return {
            **add_prefix("ip", self.ip_model.to_dict()),
            **add_prefix("as", self.as_model.to_dict() if self.as_model else {}),
            **add_prefix("path_complete", self.path_complete.to_dict()),
            **add_prefix("destination_reached", self.destination_reached.to_dict()),
            **add_prefix("looping", self.looping.to_dict()),
            **add_prefix("as_path_probs", self.path_probs.to_dict()),
            **add_prefix("ip_path_probs", self.ip_path_probs.to_dict()),
            **add_prefix("n_hops_model", self.n_hops_model.to_dict()),
            **add_prefix("trace_model", self.trace_model.to_dict()),
        }


class MultiTraceAnalyzer(BaseModel):
    def __init__(self, src: str, dest: str, *args, **kwargs) -> None:
        super().__init__(src, dest, *args, **kwargs)
        self.trace_analyzer: dict[str, TraceAnalyzer] = {}
        self.n_anomalies: PoissonModel = PoissonModel(
            src, dest, parent=self, threshold=0.025
        )
        self._anomaly_reports: list[Any] = []

    def log(self, data: dict[str, Any]) -> TraceAnalyzerOutput | AnomalyReport:
        ts = data["timestamp"]
        super().log_timestamp(ts)
        src_site = data["src"]
        dest_site = data["dest"]
        key = f"{src_site}-{dest_site}"
        subscription = GraphModel.get_or_create_subscription(
            local=True, forgetting=True, key=key
        )
        subscription_as = GraphModel.get_or_create_subscription(
            local=True, forgetting=True, key=f"{key}_as"
        )
        if key not in self.trace_analyzer:
            self.trace_analyzer[key] = TraceAnalyzer(
                src_site, dest_site, ip_model=subscription, as_model=subscription_as
            )
        out = self.trace_analyzer[key].log(data)

        tmp = out.anomalies if isinstance(out, TraceAnalyzerOutput) else out
        self.n_anomalies.log(ts, tmp.total)

        payload = {
            **tmp.dict(),
            "src": src_site,
            "dest": dest_site,
            "src_ip": data.get("src"),
            "dest_ip": data.get("dest"),
            "timestamp": ts,
        }

        anomalies = payload
        self._anomaly_reports.append(anomalies)
        return out

    @property
    def anomaly_reports(self):
        tmp = pd.DataFrame(columns=list(AnomalyReport.__fields__.keys()))
        tmp = pd.concat([tmp, pd.DataFrame(self._anomaly_reports)], ignore_index=True)
        # tmp.set_index("timestamp", inplace=True)
        # tmp.sort_index(inplace=True)
        # tmp['timestamp'] = pd.to_datetime(tmp['timestamp'], unit='ms')
        tmp.rename(
            columns={
                "as_path_probs_anomaly": "AS Path Anomaly",
                "as_model_anomaly": "AS Transition Anomaly",
                "ip_path_probs_anomaly": "IP Path Anomaly",
                "ip_model_anomaly": "IP Transition Anomaly",
                "destination_reached_anomaly": "Destination Reached Anomaly",
                "path_complete_anomaly": "Path Complete Anomaly",
                "looping_anomaly": "Looping Anomaly",
                "trace_rtt_anomaly": "RTT delay Anomaly",
                "trace_ttl_anomaly": "TTL delay Anomaly",
                "n_hops_model_anomaly": "Number of hops Anomaly",
            },
            inplace=True,
        )
        return tmp

    def to_dict(self) -> dict[str, list[Any]]:
        return {
            **add_prefix("n_anomalies", self.n_anomalies.to_dict()),
            **add_prefix("anomaly_reports", self.anomaly_reports.to_dict()),  # type: ignore
        }
