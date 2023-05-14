import numpy as np
import json
import matplotlib.pyplot as plt

from trct.models.normal_model import NormalModel
from trct.models.graph_model import GraphModel
from trct.models.bernoulli_model import BernoulliModel
from trct.models.multinomial_model import MultinomialModel

class TraceMonitor:

    def __init__(self, u, v, dest_asn=None):
        self.src = u
        self.dest = v

        self.hops_ip_model = GraphModel(u, v)
        self.hops_asn_model = GraphModel(u, dest_asn if dest_asn else v)

        self.rtt_models: dict[str, NormalModel] = {}
        self.ttl_models: dict[str, MultinomialModel] = {}

        self.path_complete_model: BernoulliModel = BernoulliModel(u, v)
        self.destination_reached_model: BernoulliModel = BernoulliModel(u, v)

    def process(self, files):
        for file in files:
            with open(file, "r") as f:
                json_data = json.load(f)
            self.log(json_data)
    
    def log(self, data):
        ts = data["timestamp"]
        
        # destination_reached = data["destination_reached"]
        
        self.hops_ip_model.log(ts, data["hops"])
        self.hops_asn_model.log(ts, data["asns"])

        for node, rtt, ttl in zip(data["hops"], data["rtts"], data["ttls"]):
            
            if node not in self.rtt_models:
                self.rtt_models[node] = NormalModel(self.src, node)
                self.ttl_models[node] = MultinomialModel(self.src, node)

            self.rtt_models[node].log(ts, rtt)
            self.ttl_models[node].log(ts, ttl)

        self.path_complete_model.log(ts, data["path_complete"])
        self.destination_reached_model.log(ts, data["destination_reached"])

    def plot(self):
        fig, axes = plt.subplots(2, 2, sharex=False, sharey=False)
        self.hops_ip_model.plot(axes=axes[0][0])
        self.hops_asn_model.plot(axes=axes[0][1])

        self.destination_reached_model.plot(axes=axes[1][0])
        self.path_complete_model.plot(axes=axes[1][1])
        
        fig.suptitle(f"{self.src} -> {self.dest}")

        fig.show() 
        for node in self.rtt_models:
            if self.rtt_models[node].n < 5:
                continue
            
            fig = plt.figure(figsize=(10, 5))
            self.rtt_models[node].plot(plt.gca())
            fig.show()
            fig = plt.figure(figsize=(10, 5))
            self.ttl_models[node].plot(plt.gca())
            fig.show()


    def score(self, data):
        # todo: implement
        pass