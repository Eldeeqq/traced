from itertools import count
from pathlib import Path
from time import sleep
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
import requests
from tqdm.auto import tqdm

path = Path(__file__).parent / "ip2geo.csv"


class IP2Geo:
    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(IP2Geo, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        self.path = path
        try:
            self._df = pd.read_csv(self.path)
            self.known_ips = set(self._df["ip"])

        except FileNotFoundError:
            self._df = pd.DataFrame()
            self.known_ips = set()

    @property
    def df(self):
        # tmp = self._df.copy() if self._df.shape[0] < 0 else self._df.copy()
        return self._df.copy()

    @staticmethod
    def get_metadata(ip_address):
        # url = f"https://api.ip>geolocation.io/ipgeo?&apiKey=aa32d3d50bde4cd8ab692ffde4e756f2&ip={ip_address}"
        url = f"http://api.ipstack.com/{ip_address}?access_key=5e5c0c8224d9d080011f2fb88830d054"
        # print(url)
        # payload = json.dumps(ip_addresses)
        headers = {"Content-Type": "application/json"}
        response = requests.request("GET", url, headers=headers)
        return response.json()

    def get_metadata_bulk(self, ip_addresses: list[str], show_progress: bool = True):
        if show_progress:
            ip_addresses = tqdm(ip_addresses)  # type: ignore

        df = []
        index = count()
        for ip in ip_addresses:
            tmp = pd.DataFrame(self.get_metadata(ip), index=[next(index)])
            tmp["ip"] = ip
            tmp["longitude"] = tmp["longitude"].astype(float)
            tmp["latitude"] = tmp["latitude"].astype(float)
            df.append(tmp)
            sleep(0.5)
        return pd.concat(df)

    def interpolate_missing(self, df: pd.DataFrame, graph: nx.Graph):
        known = set(df[~df["latitude"].isna()]["ip"]) | self.known_ips

        queue = []
        for node in graph.nodes():
            if node not in known:
                cnt = 0
                for neighbor in graph.neighbors(node):
                    if neighbor in known:
                        cnt += 1
                queue.append((node, cnt))

        for node, _ in sorted(queue, key=lambda x: x[1], reverse=True):
            neighbors = list(graph.neighbors(node))
            if len(neighbors) == 0:
                continue

            long = 0
            lat = 0

            for neighbor in neighbors:
                if neighbor in known:
                    long += df[df["ip"] == neighbor]["longitude"].values[0]
                    lat += df[df["ip"] == neighbor]["latitude"].values[0]

            long /= len(neighbors)
            lat /= len(neighbors)
            df.loc[df["ip"] == node, "longitude"] = long
            df.loc[df["ip"] == node, "latitude"] = lat
            known.add(node)

    def add_noise(self, df: pd.DataFrame, min: int, max: int):
        df["longitude"] += np.random.randint(min, max, len(df)) * 1e-4
        df["latitude"] += np.random.randint(min, max, len(df)) * 1e-4

    def get_missing_node_metadata(self, graph: nx.Graph):
        ips = list(graph.nodes)
        ips = set(ips) - self.known_ips

        if not ips:
            return

        meta = self.get_metadata_bulk(list(ips))
        self.interpolate_missing(meta, graph)
        self.add_noise(meta, 1, 3)
        self.known_ips |= ips
        self._df = pd.concat([self._df, meta])
        self.save()

    def save(self):
        self._df.to_csv(self.path, index=False)

    def __del__(self):
        self.save()

    def get_ip_geo(self, ip_address: str) -> dict[str, Any]:
        if ip_address in self.known_ips:
            return self._df[self._df["ip"] == ip_address].iloc[0].to_dict()
        else:
            return {"latitude": 0, "longitude": 0}
