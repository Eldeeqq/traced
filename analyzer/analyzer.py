import networkx as nx
import json
from hashlib import sha1
from collections import defaultdict
from typing import Iterator
import numpy as np
from analyzer.model import Model


class TraceAnalyzer:

    def __init__(self, file_iter):
        self.G : nx.Graph = nx.DiGraph()
        self.file_iter: Iterator[str] = file_iter
        self.hash_trace: dict[str, tuple[str, str, list[str]]] = {}
        self.models: dict[tuple[str, str], Model] = {}
        self.hash_counter: dict[str, int] = defaultdict(lambda: 0)
        self.markov_probs: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(lambda: 0))
        self.src_dest_freq: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(lambda: 0))
        self.sources: dict[str, int] = defaultdict(lambda: 0)
        self.destinations: dict[str, int] = defaultdict(lambda: 0)
        self.data = []
        self.n = 0

    @staticmethod
    def hash_route(src, dest, hops) -> str:
        return sha1((f"{src}{' '.join(hops)}{dest}").encode('ascii')).hexdigest()

    def process(self):
        for file in self.file_iter:
            with open(file, "r") as f:
                json_data = json.load(f)
            self.process_traceroute(json_data)

    def get_edge_model(self, u, v) -> Model:
        self.markov_probs[u][v] += 1
        if (u, v) not in self.models:
                    self.G.add_edge(u, v)
                    self.models[(u, v)] = Model(u, v)
        return self.models[(u, v)] 
    
    def get_probs(self, node):
         node_out = self.markov_probs[node]
         n = sum(node_out.values())
         return {x: node_out[x]/n for x in node_out}
    
    def graph_probs(self, ):
        return {(x,y):p for x in self.markov_probs for y,p in self.get_probs(x).items()}
    
    def score(self, src, dest, hops) -> float:
        route_hash = self.hash_route(src, dest, hops)
        return self.hash_counter[route_hash] / self.n

    def process_traceroute(self, data):
        self.n += 1
        src = data["src"]
        dest = data["dest"]
        hops = data["hops"]    

        self.src_dest_freq[src][dest] += 1
        self.sources[src] += 1
        self.destinations[dest] += 1

        route_hash = self.hash_route(src, dest, hops)
        self.hash_counter[route_hash] += 1
        self.hash_trace[route_hash] = (src, dest, hops)
        
        hop_gen = iter(hops)
        ttl_gen = iter(np.diff(np.array(data['ttls']), prepend=0))
        rtt_gen = iter(np.diff(np.array(data['rtts']), prepend=0))

        for node, ttls, rtt, in zip(hops, data['ttls'], data['rtts']):
            model = self.get_edge_model(src, node)
            model.log(data["timestamp"], rtt, ttls, data['destination_reached'])
             
        # curr = src
        # while True:
        #     try:
        #         next_hop = next(hop_gen)
        #         model = self.get_edge_model(curr, next_hop)
        #         model.log(data["timestamp"], next(rtt_gen), next(ttl_gen), data['destination_reached'])
        #         curr = next_hop
        #     except StopIteration:
        #         break
        
        