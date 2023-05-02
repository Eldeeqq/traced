import networkx as nx
import json
from hashlib import sha1
from collections import defaultdict
from typing import Iterable
import numpy as np
from analyzer.model import Model
from typing import Any
import pandas as pd
import seaborn as sns
import tqdm


class TraceAnalyzer:
    def __init__(self, files):
        self.G: nx.Graph = nx.DiGraph()
        self.files: Iterable[str] = files
        self.hash_trace: dict[str, tuple[str, str, list[str]]] = {}
        self.models: dict[tuple[str, str], Model] = {}
        self.hash_counter: dict[str, int] = defaultdict(lambda: 0)
        self.markov_probs: dict[str, dict[str, float]] = defaultdict(
            lambda: defaultdict(lambda: 1)
        )
        self.src_dest_freq: dict[str, dict[str, int]] = defaultdict(
            lambda: defaultdict(lambda: 0)
        )
        self.sources: dict[str, int] = defaultdict(lambda: 0)
        self.destinations: dict[str, int] = defaultdict(lambda: 0)
        self.scores: dict[str, dict[int, Any]] = defaultdict(lambda: {})
        self.data = []
        self.n = 0

    @staticmethod
    def hash_route(src, dest, hops) -> str:
        return sha1((f"{src}{' '.join(hops)}{dest}").encode("ascii")).hexdigest()

    def process(self, progressbar=None):
        files: Iterable[str] = tqdm.tqdm(self.files) if progressbar else self.files

        for file in files:
            with open(file, "r") as f:
                json_data = json.load(f)
            self.process_traceroute(json_data)

    def process_and_save(self, folder, progressbar=None):
        files = tqdm.tqdm(self.files) if progressbar else self.files

        for file in files:
            with open(file, "r") as f:
                json_data = json.load(f)
            score = self.process_traceroute(json_data)

            # rtt_is_outlier, rtt_prob, rtt_mu_diff, success_prob, success_score, rtt_ttl_rate
            json_data["outlier"] = score[0].tolist()
            json_data["rtts_prob"] = score[1].tolist()
            json_data["rtt_error"] = score[2].tolist()
            json_data["success_prob"] = score[3].tolist()
            json_data["success_score"] = score[4].tolist()
            json_data["rtt_ttl_rate"] = score[5].tolist()

            with open(f"{folder}/{file.split('/')[-1]}", "w") as f:
                json.dump(json_data, f)

    def get_edge_model(self, u, v) -> Model:
        self.markov_probs[u][v] += 1
        if (u, v) not in self.models:
            self.G.add_edge(u, v)
            self.models[(u, v)] = Model(u, v)
        return self.models[(u, v)]

    def get_probs(self, node, scale=False):
        node_out = self.markov_probs[node]
        n = sum(node_out.values())
        probs = {x: node_out[x] / n for x in node_out}
        if scale:
            frac = max(probs.values()) - min(probs.values())
            probs = {k: v / frac for k, v in probs.items()}
        return probs

    def graph_probs(self, scale=False):
        return {
            (x, y): p
            for x in self.markov_probs
            for y, p in self.get_probs(x, scale=scale).items()
        }

    def process_traceroute(self, data):
        self.n += 1
        src = data["src"]
        dest = data["dest"]
        hops = data["hops"]
        ts = data["timestamp"]

        self.src_dest_freq[src][dest] += 1
        self.sources[src] += 1
        self.destinations[dest] += 1

        route_hash = self.hash_route(src, dest, hops)
        self.hash_counter[route_hash] += 1
        self.hash_trace[route_hash] = (src, dest, hops)

        # hop_gen = iter(hops)
        # ttl_gen = iter(np.diff(np.array(data['ttls']), prepend=0))
        # rtt_gen = iter(np.diff(np.array(data['rtts']), prepend=0))

        scores = []

        for (
            node,
            ttl,
            rtt,
        ) in zip(hops, data["ttls"], data["rtts"]):
            model = self.get_edge_model(src, node)
            scores.append(model.score(rtt, ttl, data["destination_reached"]))
            model.log(ts, rtt, ttl, data["destination_reached"])

        self.scores[route_hash][ts] = np.array(scores).T
        # curr = src
        # while True:
        #     try:
        #         next_hop = next(hop_gen)
        #         model = self.get_edge_model(curr, next_hop)
        #         model.log(data["timestamp"], next(rtt_gen), next(ttl_gen), data['destination_reached'])
        #         curr = next_hop
        #     except StopIteration:
        #         break
        return np.array(scores).T

    def src_dest_hist(self):
        df = pd.DataFrame(self.src_dest_freq).fillna(0).astype(int).T
        df = df.loc[:, (df != 0).any(axis=0)]
        return sns.heatmap(
            df,
            cmap="Greens",
            annot=False,
            fmt="d",
            linewidths=1,
            cbar=True,
            square=False,
        )
