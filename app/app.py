import sys

import streamlit as st

sys.path.insert(0, "..")
from abc import abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Any

import dill
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from matplotlib.axes import Axes
from streamlit_folium import st_folium
from tqdm.auto import tqdm
from traced_v2.models.base_model import BaseModel, Visual
from traced_v2.models.bernoulli import BernoulliModelOutput
from traced_v2.models.poisson import PoissonModel
from traced_v2.site_analyzer import SiteAnalyzer
from traced_v2.trace_analyzer import TraceAnalyzer, TraceAnalyzerOutput

from utils import *


st.set_page_config(
    layout="wide",
    page_title="Traced demo",
    page_icon=":globe_with_meridians:",
    initial_sidebar_state="expanded",
)
TOC.reset()

plot_all_anomalies()

analyzer: SiteAnalyzer = get_site_to_site()
sources = list(analyzer.site_to_site.keys())

source = st.sidebar.selectbox("Source", sources, help="Select source site")

if not sources or source not in analyzer.site_to_site:
    st.warning("There has been an issue. Try different site.")
    st.stop()

destinations = list(analyzer.site_to_site[source].keys())
dest = st.sidebar.selectbox("Destination", destinations, help="Select destination site")

if not destinations or dest not in analyzer.site_to_site[source]:
    st.warning("There has been an issue. Try different site.")
    st.stop()


st.subheader(f"{source} -> {dest}")

traces = list(analyzer.site_to_site[source][dest].trace_analyzer.keys())
n = analyzer.site_to_site[source][dest].n
site_to_site = analyzer.site_to_site[source][dest]
plot_site_to_site_anomalies(source, dest)


trace = st.selectbox(
    "Route",
    traces,
    format_func=lambda x: f"({100*site_to_site.trace_analyzer[x].n/n:2.1f}%) {x} ",
)

if not trace and trace not in analyzer.site_to_site[source][dest].trace_analyzer:
    st.warning("There has been an issue. Try different route.")
    st.stop()

geo_plot = plot_paths_folium(source, dest, trace)

TOC.header("Map with routers and paths")
st_folium(geo_plot, use_container_width=True, returned_objects=[])

trace_model = analyzer.site_to_site[source][dest].trace_analyzer[trace]
TOC.placeholder(sidebar=True)
plot_n_anomalies(source, dest, trace)

tabs = st.tabs(["AS sequences", "IP sequences", "Number of hops", "RTT", "TTL", "Destination reached", "Path complete", "Path looping"])
with tabs[0]:
    plot_asn_section(source, dest, trace)
with tabs[1]:
    plot_ip_section(source, dest, trace)
with tabs[2]:
    plot_n_hops_section(source, dest, trace)
with tabs[3]:
    plot_rtt_section(source, dest, trace)
with tabs[4]:
    plot_ttl_section(source, dest, trace)
with tabs[5]:
    plot_destination_reached_section(source, dest, trace)
with tabs[6]:
    plot_path_complete_section(source, dest, trace)
with tabs[7]:
    plot_path_looping_section(source, dest, trace)

TOC.generate()
# st.selectbox("cols",get_trace_model_df(source,dest, trace).columns)
