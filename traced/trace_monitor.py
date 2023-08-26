import json
from collections import defaultdict
from hashlib import sha1
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from traced.models.bernoulli_model import BernoulliModel
from traced.models.graph_model import GraphModel
from traced.models.multinomial_model import MultinomialModel
from traced.models.normal_model import NormalModel
from traced.models.poisson_model import PoissonModel


class MultiSiteTraceMonitor:
    def __init__(self):
        self.trace_models: Dict[Tuple[str, str], SiteTraceMonitor] = {}
    
    def process(self, files):
        for file in tqdm(files):
            with open(file, "r") as f:
                json_data = json.load(f)
            self.log(json_data)

    def log(self, data):
        src_site: str = data.get("src_site", "NA")
        dest_site: str = data["dest_site"]

        if (src_site, dest_site) not in self.trace_models:
            self.trace_models[(src_site, dest_site)] = SiteTraceMonitor(
                src_site, dest_site
            )
        site_monitor = self.trace_models[(src_site, dest_site)]
        site_monitor.log(data)




class SiteTraceMonitor:
    def __init__(self, src_site, dest_site):
        self.src = src_site
        self.dest = dest_site

        self.trace_models: Dict[Tuple[str, str], TraceMonitor] = {}

    def log(self, data):
        src: str = data["src"]
        dest: str = data["dest"]

        src_asn = data.get("src_asn", "NA")

        if len(data["asns"]) < 1:
            return
        dest_asn = data["asns"][-1]

        if (src, dest) not in self.trace_models:
            self.trace_models[(src, dest)] = TraceMonitor(
                src, dest, src_asn=src_asn, dest_asn=dest_asn
            )

        self.trace_models[(src, dest)].log(data)

    def process(self, files):
        for file in tqdm(files):
            with open(file, "r") as f:
                json_data = json.load(f)
            self.log(json_data)


class TraceMonitor:
    def __init__(self, u, v, src_asn=None, dest_asn=None):
        self.src = u
        self.dest = v

        self.hops_ip_model = GraphModel(u, v)
        self.hops_asn_model = GraphModel(
            src_asn if src_asn else u, dest_asn if dest_asn else v
        )

        self.rtt_models: dict[str, NormalModel] = {}
        self.destination_reached: dict[str, BernoulliModel] = {}

        self.path_complete_model: BernoulliModel = BernoulliModel(u, v)
        self.destination_reached_model: BernoulliModel = BernoulliModel(u, v)
        self.path_probs: MultinomialModel = MultinomialModel(u, v)

        self.n_hops_model: PoissonModel = PoissonModel(u, v)

        self.ttl_poissons = {}
        self.n = 0
        self.detected = defaultdict(lambda: {})

    def process(self, files):
        for file in tqdm(files):
            with open(file, "r") as f:
                json_data = json.load(f)
            self.log(json_data)

    def log(self, data):
        is_anomaly = False
        anom_cnt = 0
        total_cnt = 0

        ts = data["timestamp"]
        self.detected[self.n]["ts"] = ts
        if not data["n_hops"]:
            # TODO: mark as anomaly?
            return

        self.detected[self.n]["n_hops"] = self.n_hops_model.log(ts, len(data["hops"]))

        asn_hash = sha1(
            "".join(list(map(str, data["asns"]))).encode("utf-8")
        ).hexdigest()
        self.detected[self.n]["ip_hash"] = sha1(
            "".join(list(map(str, data["hops"]))).encode("utf-8")
        ).hexdigest()

        tmp = []

        for x in data["asns"]:
            if tmp and tmp[-1] == x:
                continue
            tmp.append(x)
        self.detected[self.n]["unique_asns"] = sha1(
            "".join(list(map(str, tmp))).encode("utf-8")
        ).hexdigest()

        tmp = self.detected[self.n]["ip"] = self.hops_ip_model.log(ts, data["hops"])

        is_anomaly = is_anomaly or tmp[0]
        anom_cnt += int(tmp[0])
        total_cnt += 1

        tmp = self.detected[self.n]["asn"] = self.hops_asn_model.log(ts, data["asns"])
        is_anomaly = is_anomaly or tmp[0]
        anom_cnt += int(tmp[0])
        total_cnt += 1

        rtts = []
        ttls = []
        index = 1
        rtt_anom_count = 0
        ttl_anom_count = 0
        for node, rtt, ttl in zip(data["hops"], data["rtts"], data["ttls"]):
            if node not in self.rtt_models:
                self.rtt_models[node] = NormalModel(self.src, node, sigma_factor=4)
                self.ttl_poissons[node] = PoissonModel(self.src, node)

            tmp = self.rtt_models[node].log(ts, np.log1p(rtt))
            is_anomaly = is_anomaly or tmp[0]
            rtt_anom_count += int(tmp[0])
            rtts.append(tmp)
            tmp = self.ttl_poissons[node].log(ts, ttl - index)
            is_anomaly = is_anomaly or tmp[0]
            ttl_anom_count += int(tmp[0])
            ttls.append(tmp)
            index += 1
        x = rtt_anom_count / len(data["rtts"])
        anom_cnt += x
        total_cnt += x

        x = ttl_anom_count / len(data["ttls"])
        anom_cnt += x
        total_cnt += x

        self.detected[self.n]["ttls"] = list(zip(*ttls))
        self.detected[self.n]["rtts"] = list(zip(*rtts))
        asn_hash = sha1(
            "".join(["src"] + list(map(str, data["asns"]))).encode("utf-8")
        ).hexdigest()
        tmp = self.detected[self.n]["path"] = self.path_probs.log(ts, asn_hash)
        is_anomaly = is_anomaly or tmp[0]
        anom_cnt += int(tmp[0])
        total_cnt += 1

        tmp = self.detected[self.n]["reached"] = self.destination_reached_model.log(
            ts, data["destination_reached"]
        )
        is_anomaly = is_anomaly or tmp[0]
        anom_cnt += int(tmp[0])
        total_cnt += 1

        self.detected[self.n]["has_anomaly"] = is_anomaly
        self.detected[self.n]["anom_share"] = (1 + anom_cnt) / (1 + total_cnt)
        self.detected[self.n]["path_hash"] = asn_hash
        self.n += 1

    def plot(self):
        fig, axes = plt.subplots(2, 2, sharex=False, sharey=False)
        self.hops_ip_model.plot(axes=axes[0][0])
        self.hops_asn_model.plot(axes=axes[0][1])

        self.destination_reached_model.plot(axes=axes[1][0])
        self.path_complete_model.plot(axes=axes[1][1])

        fig.suptitle(f"{self.src} -> {self.dest}")

        fig.show()
        for node in self.rtt_models:
            # if self.rtt_models[node].n < 5:
            #     continue

            fig = plt.figure(figsize=(10, 5))
            self.rtt_models[node].plot(plt.gca())
            fig.show()
            fig = plt.figure(figsize=(10, 5))
            # self.ttl_models[node].plot(plt.gca())
            fig.show()
