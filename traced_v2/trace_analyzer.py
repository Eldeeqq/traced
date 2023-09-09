from typing import Any

import pydantic


from traced_v2.models.base_model import BaseModel
from traced_v2.models.bernoulli import BernoulliModel, BernoulliModelOutput
from traced_v2.models.graph import GraphModel, GraphModelOutput
from traced_v2.models.multinomial import ForgettingMultinomialModel, MultinomialModelOutput
from traced_v2.models.poisson import PoissonModel, PoissonModelOutput
from traced_v2.utils import add_prefix, create_hash, remove_duplicates
from traced_v2.models.trace_model import TraceModel, TraceModelOutput, Mode


class TraceAnalyzerOutput(pydantic.BaseModel):
    ip_model: GraphModelOutput
    as_model: GraphModelOutput | None
    path_complete: BernoulliModelOutput 
    destination_reached: BernoulliModelOutput
    looping: BernoulliModelOutput
    n_hops_model: PoissonModelOutput
    trace_model: TraceModelOutput
    path_probs: MultinomialModelOutput
    n_anomalies: int


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

        self.path_complete: BernoulliModel = BernoulliModel(src, dest, parent=self, scorer=lambda x, p: x if p <= 0.2 else not x if p >= 0.8 else False)
        self.destination_reached: BernoulliModel = BernoulliModel(
            src, dest, parent=self, scorer=lambda x, p: x if p <= 0.2 else not x if p >= 0.8 else False  
        )
        self.looping: BernoulliModel = BernoulliModel(src, dest, parent=self, scorer=lambda x, _: x)

        self.path_probs: ForgettingMultinomialModel = ForgettingMultinomialModel(
            src, dest,  parent=self
        )
        self.ip_path_probs: ForgettingMultinomialModel = ForgettingMultinomialModel(
            src, dest, parent=self
        )

        self.n_hops_model: PoissonModel = PoissonModel(src, dest, parent=self)
        self.trace_model: TraceModel = TraceModel(
            src, dest, mode=Mode.WEIGHTED_HARMONIC_MEAN, #parent=self,
        )
        self.missing_data: BernoulliModel = BernoulliModel(src, dest, scorer=lambda x, _: x)

    def log(self, data: dict[str, Any]) -> tuple[None | TraceAnalyzerOutput, BernoulliModelOutput]:
        ts = data["timestamp"]
        num_anomalies = 0
        try:
            # TODO: better handling of missing data
            if self.as_model is None and data["hops"][-1] == self.dest:
                # lazy init the asn graph model, since we do not know the as apriori
                self.as_model = GraphModel(
                    data["asns"][0],
                    data["asns"][-1],
                    # parent=self,
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
            num_anomalies += int(path_complete_output.is_anomaly) # TODO: uncomment when bernoulli detect anomalies
            dest_reached = self.destination_reached.log(ts, data["destination_reached"])
            num_anomalies += int(dest_reached.is_anomaly) # TODO: uncomment when bernoulli detect anomalies
            looping = self.looping.log(ts, data["looping"])
            num_anomalies += int(looping.is_anomaly) # TODO: uncomment when bernoulli detect anomalies

            path_probs = self.path_probs.log(ts, create_hash("".join(map(str, remove_duplicates(data["asns"])))))
            ip_path_probs = self.ip_path_probs.log(ts, create_hash("".join(map(str, remove_duplicates(data["hops"])))))
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
                n_anomalies=num_anomalies
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
         #   **add_prefix("trace_model", self.trace_model.to_dict()),
        }
