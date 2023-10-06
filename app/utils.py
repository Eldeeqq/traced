import tempfile
from enum import Enum
from pathlib import Path

import dill
import folium
import jinja2
import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import pyvis
import streamlit as st
import streamlit.components.v1 as components
from folium import plugins
from matplotlib import pyplot as plt
from networkx.drawing.nx_pydot import graphviz_layout
from streamlit_folium import st_folium
from traced_v2.ip import IP2Geo
from traced_v2.site_analyzer import SiteAnalyzer
from traced_v2.trace_analyzer import TraceAnalyzer

from templates import POPUP_TEMPLATE
from toc import TOC

column_name_mapping = {
    "trace_rtt_anomaly": "RTT anomaly",
    "trace_ttl_anomaly": "TTL anomaly",

    "as_model_anomaly" : "AS hash anomaly",
    "as_path_probs_anomaly" : "AS transition anomaly",

    "ip_model_anomaly" : "IP hash anomaly",
    "ip_path_probs_anomaly" : "IP transition anomaly",
    "looping_anomaly": "Looping anomaly",
    "destination_reached_anomaly": "Destination reached anomaly",
    "path_complete_anomaly": "Path complete anomaly",
    "n_hops_model_anomaly": "#hops anomaly",

}


@st.cache_resource()
def get_ip2geo():
    return IP2Geo()


@st.cache_resource()
def get_site_to_site() -> SiteAnalyzer:
    # with open("site_to_site_full.dill", "rb") as f:
    #     return dill.load(f)
    site_to_site = SiteAnalyzer("src", "dest")
    from collections import defaultdict

    loaded = defaultdict(lambda: defaultdict(lambda: None))
    files = sorted(list(Path("data").rglob("*-*-*2.dill")), key=lambda x: x.name)

    for file in files:
        with file.open("rb") as f:
            # _, src, dest, _ = file.name.split("-")
            tmp = dill.load(f)
        if tmp.src not in site_to_site.site_to_site:
            site_to_site.site_to_site[tmp.src] = {}
        site_to_site.site_to_site[tmp.src][tmp.dest] = tmp

    return site_to_site


@st.cache_resource(max_entries=3)
def get_trace_model(src_site, dest_site, route) -> TraceAnalyzer:
    site_to_site = get_site_to_site()
    return site_to_site.site_to_site[src_site][dest_site].trace_analyzer[route]


@st.cache_resource(max_entries=3)
def plot_paths_folium(src_site, dest_site, route):
    analyzer = get_site_to_site()
    site_to_site = analyzer.site_to_site[src_site][dest_site]
    trace_analyzer = site_to_site.trace_analyzer[route]

    graph = trace_analyzer.ip_model.graph.to_graph()
    ip2geo = get_ip2geo()
    ip2geo.get_missing_node_metadata(graph)

    df = ip2geo.df

    curr_df = df[df["ip"].isin(graph.nodes)]  # .dropna()
    center_lon = curr_df["longitude"].dropna().mean()
    center_lat = curr_df["latitude"].dropna().mean()
    if np.isnan(center_lat) or np.isnan(center_lon):
        return

    m = folium.Map(
        location=[center_lat, center_lon], zoom_start=3, tiles="OpenStreetMap"
    )
    layer = folium.FeatureGroup(name=f"Nodes")
    if curr_df.shape[0] == 0:
        return m
    bearings = [0, 45, 90, 135, 180, 225, 270, 315]

    edges = folium.FeatureGroup(name=f"Edges")
    for u, v, data in graph.edges(data=True):
        u_data = ip2geo.get_ip_geo(u)
        v_data = ip2geo.get_ip_geo(v)
        color = "#3632a8"

        folium.PolyLine(
            [
                [u_data["latitude"], u_data["longitude"]],
                [v_data["latitude"], v_data["longitude"]],
            ],
            color=color,
            weight=8 * data["weight"],
            tooltip=data["weight"],
            opacity=0.4,
            bearings=bearings,
        ).add_to(edges)
    edges.add_to(m)

    for node in graph.nodes:
        data = ip2geo.get_ip_geo(node)
        if not np.isnan(data["latitude"]) and not np.isnan(data["longitude"]):
            is_bogon = np.isclose(data["latitude"], 0, atol=1e-3) and np.isclose(
                data["longitude"], 0, atol=1e-3
            )
            popup = POPUP_TEMPLATE.render(data=data) if not is_bogon else ""
            color = "darkgreen" if not is_bogon else "darkred"
            icon = "server" if not is_bogon else "question"
            folium.Marker(
                [data["latitude"], data["longitude"]],
                popup=popup,
                tooltip=node,
                icon=folium.Icon(color=color, icon=icon, prefix="fa"),
            ).add_to(layer)
        else:
            st.toast(f"`{node}` is missing.")
    m.add_child(layer)
    m.add_child(plugins.MiniMap())
    m.add_child(plugins.Draw())
    m.add_child(folium.LayerControl())
    m.add_child(plugins.Fullscreen())
    return m


@st.cache_resource(max_entries=3)
def plot_asn_path(src_site, dest_site, route):
    trace_model = get_trace_model(src_site, dest_site, route)

    fig = plt.figure(figsize=(16, 5))
    trace_model.path_probs.plot(ax=fig.gca(), kind="AS Number sequence probabilities")
    return fig


@st.cache_resource(max_entries=3)
def plot_ip_path(src_site, dest_site, route):
    trace_model = get_trace_model(src_site, dest_site, route)

    fig = plt.figure(figsize=(16, 5))
    trace_model.ip_path_probs.plot(
        ax=fig.gca(), kind="AS Number sequence probabilities"
    )
    return fig


@st.cache_resource(max_entries=3)
def plot_as_path_probs(src_site, dest_site, route):
    trace_model = get_trace_model(src_site, dest_site, route)

    fig = plt.figure(figsize=(16, 5))
    trace_model.as_model.prob_model.plot(
        ax=fig.gca(), kind="AS Number sequence transition log-probabilities"
    )
    return fig


@st.cache_resource(max_entries=3)
def plot_ip_path_probs(src_site, dest_site, route):
    trace_model = get_trace_model(src_site, dest_site, route)

    fig = plt.figure(figsize=(16, 5))
    trace_model.ip_model.prob_model.plot(
        ax=fig.gca(), kind="IP sequence transition log-probabilities"
    )
    return fig


@st.cache_resource(max_entries=3)
def plot_rtt(src_site, dest_site, route):
    trace_model = get_trace_model(src_site, dest_site, route)

    fig = plt.figure(figsize=(16, 5))
    trace_model.trace_model.final_rtt.plot(
        ax=fig.gca(), kind="Aggregated RTT errors (ms)"
    )
    plt.ylabel("RTT error (ms)")
    return fig


@st.cache_resource(max_entries=3)
def plot_ttl(src_site, dest_site, route):
    trace_model = get_trace_model(src_site, dest_site, route)

    fig = plt.figure(figsize=(16, 5))
    trace_model.trace_model.final_ttl.plot(
        ax=fig.gca(), kind="Aggregated TTL error (hops)"
    )
    plt.ylabel("TTL error (hops)")
    return fig


@st.cache_resource(max_entries=3)
def get_trace_model_df(src_site, dest_site, route):
    return get_trace_model(src_site, dest_site, route).to_frame()


@st.cache_resource(max_entries=3)
def plot_destination_reached(src_site, dest_site, route):
    model = get_trace_model(src_site, dest_site, route)
    fig = plt.figure(figsize=(16, 5))
    model.destination_reached.plot(ax=fig.gca())
    plt.title("Destination Reached")
    plt.ylabel("Probability")
    return fig


@st.cache_resource(max_entries=3)
def plot_path_complete(src_site, dest_site, route):
    model = get_trace_model(src_site, dest_site, route)
    fig = plt.figure(figsize=(16, 5))
    model.path_complete.plot(ax=fig.gca())
    plt.ylabel("Probability")
    plt.title("Path Complete")
    return fig


@st.cache_resource(max_entries=3)
def plot_looping(src_site, dest_site, route):
    model = get_trace_model(src_site, dest_site, route)
    fig = plt.figure(figsize=(16, 5))
    model.looping.plot(ax=fig.gca())
    plt.ylabel("Probability")
    plt.title("Looping")
    return fig


@st.cache_resource(max_entries=40)
def get_plot(
    src_site,
    dest_site,
    route,
    y,
    color,
    hover_data,
    cmap="BlueRed",
    opacity=0.5,
    symbol=None,
    **kwargs,
):
    df = get_trace_model_df(src_site, dest_site, route)
    tmp = {"symbol": symbol} if symbol else {}
    fig = df.plot(
        backend="plotly",
        kind="scatter",
        x=df.index,  # type: ignore
        y=y,
        color=color,
        hover_data=hover_data,
        opacity=opacity,
        color_continuous_scale=cmap,
        color_discrete_sequence=px.colors.qualitative.G10,
        **tmp,
    )
    fig.update_layout(**kwargs)  # type: ignore
    fig.update_traces(marker=dict(size=8.5))  # type: ignore
    fig.update_layout(
        legend=dict(
            orientation="h",
            y=-0.2,
        )
    )
    return fig


def plot_asn_section(src_site, dest_site, route):
    TOC.header("AS Number sequences")
    st.pyplot(plot_asn_path(src_site, dest_site, route))

    df = get_trace_model_df(src_site, dest_site, route)
    st.metric("# unique AS sequences", df["as_path_probs_observed_variables"].nunique())
    TOC.subheader("ASN sequences transition probabilities")
    primary, per_as, per_ip, as_hier = st.tabs(
        ["Primary", "Per AS", "Per IP", "AS Hierarchy"]
    )
    primary.pyplot(plot_as_path_probs(src_site, dest_site, route))

    extras = dict(
        title="AS Number sequence transition log-probabilities",
        xaxis_title="Time",
        yaxis_title="Negative Log probability",
        legend_title_text="IP sequence hash",
    )
    per_as.plotly_chart(
        get_plot(
            src_site,
            dest_site,
            route,
            y="as_log_prob_observed_values",
            color="as_path_probs_observed_variables",
            hover_data=["as_path_probs_observed_variables"],
            opacity=0.5,
            **extras,
        ),
        use_container_width=True,
    )

    per_ip.plotly_chart(
        get_plot(
            src_site,
            dest_site,
            route,
            y="as_log_prob_observed_values",
            color="ip_path_probs_observed_variables",
            hover_data=["ip_path_probs_observed_variables"],
            opacity=0.5,
            **extras,
        ),
        use_container_width=True,
    )

    plot_as_hierarchy(src_site, dest_site, route, as_hier)

@st.cache_resource(max_entries=1)
def _plot_as_hierarchy(src_site, dest_site, route):
    model = get_trace_model(src_site, dest_site, route)
    G = model.as_hash_hierarchy.hash_graph
    G = prune_graph(G)

    fig = plt.figure(figsize=(10, 10))
    node_labels = {
        n: f'{n}\n {G.nodes[n]["count"]}' if G.nodes[n].get("count", 0) > 0 else ""
        for n in G.nodes
    }

    pos = graphviz_layout(
        G,
    )

    colors = [G.nodes[n]["color"] for n in G.nodes()]
    nx.draw(G, with_labels=False, pos=pos, node_color=colors, ax=fig.gca())
    nx.draw_networkx_edge_labels(
        G,
        pos=pos,
        edge_labels=nx.get_edge_attributes(G, "weight"),
        font_size=7,
        ax=fig.gca(),
    )
    nx.draw_networkx_labels(
        G, pos=pos, labels=node_labels, font_size=7, font_color="black", ax=fig.gca()
    )
    return fig

def plot_as_hierarchy(src_site, dest_site, route, as_hier):
        model = get_trace_model(src_site, dest_site, route)
        fig = _plot_as_hierarchy(src_site, dest_site, route)

        left, right = as_hier.columns((2, 3))
        path = left.selectbox(
            "Select path", model.as_hash_hierarchy.hash_to_path.keys(), key="as_path"
        )
        left.table(pd.DataFrame(model.as_hash_hierarchy.hash_to_path[path]))
        path2 = left.selectbox(
            "Select second path",
            model.as_hash_hierarchy.hash_to_path.keys(),
            key="as_path2",
        )
        left.table(pd.DataFrame(model.as_hash_hierarchy.hash_to_path[path2]))
        right.pyplot(
            fig, use_container_width=True
        )

def prune_graph(G: nx.DiGraph):
    source = [n for n in G.nodes if G.in_degree(n) == 0][0]

    pruned = nx.DiGraph()
    pruned.add_node(source)

    pruned.nodes[source]["color"] = "red"

    # dfs search
    def rec(src_node, cur_node, n=1):
        out_degree = G.out_degree(cur_node)
        if out_degree == 1 and G.nodes[cur_node].get("count", 0) == 0:
            for x in G.successors(cur_node):
                rec(src_node, x, n + 1)
            return

        color = "yellow" if out_degree != 0 else "green"
        pruned.add_node(cur_node)
        pruned.add_edge(src_node, cur_node, weight=n)
        pruned.nodes[cur_node]["color"] = color
        pruned.nodes[cur_node]["count"] = G.nodes[cur_node].get("count", 0)

        for x in G.successors(cur_node):
            rec(cur_node, x, n=1)
        return

    for node in G.successors(source):
        rec(source, node)
    return pruned


def plot_ip_section(src_site, dest_site, route):
    TOC.header("IP sequences")
    fig = plot_ip_path(src_site, dest_site, route)
    plt.title("Probability of IP sequence ")
    st.pyplot(fig)
    df = get_trace_model_df(src_site, dest_site, route)
    st.metric("# unique IP sequences", df["ip_path_probs_observed_variables"].nunique())

    TOC.subheader("IP sequences transition probabilities")
    primary, per_as, per_ip, ip_hierarchy = st.tabs(
        ["Primary", "Per AS", "Per IP", "IP Hierarchy"]
    )
    primary.pyplot(plot_ip_path_probs(src_site, dest_site, route))

    extras = dict(
        title="AS Number sequence transition log-probabilities",
        xaxis_title="Time",
        yaxis_title="Negative Log probability",
        legend_title_text="IP sequence hash",
    )
    per_as.plotly_chart(
        get_plot(
            src_site,
            dest_site,
            route,
            y="ip_log_prob_observed_values",
            color="as_path_probs_observed_variables",
            hover_data=["as_path_probs_observed_variables"],
            opacity=0.5,
            **extras,
        ),
        use_container_width=True,
    )

    per_ip.plotly_chart(
        get_plot(
            src_site,
            dest_site,
            route,
            y="ip_log_prob_observed_values",
            color="ip_path_probs_observed_variables",
            hover_data=["ip_path_probs_observed_variables"],
            opacity=0.5,
            **extras,
        ),
        use_container_width=True,
    )

    plot_ip_hirarchy(src_site, dest_site, route, ip_hierarchy)

@st.cache_resource(max_entries=1)
def _plot_ip_hierarchy(src_site, dest_site, route):
    model = get_trace_model(src_site, dest_site, route)
    G = model.ip_hash_hierarchy.hash_graph
    G = prune_graph(G)

    fig = plt.figure(figsize=(10, 10))
    node_labels = {
        n: f'{n}\n {G.nodes[n]["count"]}' if G.nodes[n].get("count", 0) > 0 else ""
        for n in G.nodes
    }

    pos = graphviz_layout(
        G,
    )

    colors = [G.nodes[n]["color"] for n in G.nodes()]
    nx.draw(G, with_labels=False, pos=pos, node_color=colors, ax=fig.gca())
    nx.draw_networkx_edge_labels(
        G,
        pos=pos,
        edge_labels=nx.get_edge_attributes(G, "weight"),
        font_size=7,
        ax=fig.gca(),
    )
    nx.draw_networkx_labels(
        G, pos=pos, labels=node_labels, font_size=7, font_color="black", ax=fig.gca()
    )
    return fig

def plot_ip_hirarchy(src_site, dest_site, route, ip_hier):
    model = get_trace_model(src_site, dest_site, route)
    fig = _plot_ip_hierarchy(src_site, dest_site, route)
    left, right = ip_hier.columns((2, 3))
    path = left.selectbox(
        "Select path", model.ip_hash_hierarchy.hash_to_path.keys(), key="ip_path"
    )
    left.table(pd.DataFrame(model.ip_hash_hierarchy.hash_to_path[path]))
    path2 = left.selectbox(
        "Select second path",
        model.ip_hash_hierarchy.hash_to_path.keys(),
        key="ip_path2",
    )
    left.table(pd.DataFrame(model.ip_hash_hierarchy.hash_to_path[path2]))
    right.pyplot(
        fig, use_container_width=True
    )  

def plot_n_hops_section(src_site, dest_site, route):
    TOC.header("Number of hops")

    fig = plt.figure(figsize=(16, 5))
    trace_model = get_trace_model(src_site, dest_site, route)
    trace_model.n_hops_model.plot(ax=fig.gca())
    plt.title("Probability of number of hops")
    st.write(fig)

    primary, per_as, per_ip = st.tabs(["Observed", "Per AS", "Per IP"])
    extras = dict(
        title="Number of hops per traceroute",
        xaxis_title="Time",
        yaxis_title="N-hops",
    )

    primary.plotly_chart(
        get_plot(
            src_site,
            dest_site,
            route,
            y="n_hops_model_observed_values",
            color="n_hops_model_probabilities",
            hover_data=["n_hops_model_observed_values"],
            opacity=1,
            coloraxis_colorbar_title_text="Probability of N-hops",
            **extras,
        ),
        use_container_width=True,
    )
    per_as.plotly_chart(
        get_plot(
            src_site,
            dest_site,
            route,
            y="n_hops_model_observed_values",
            color="as_path_probs_observed_variables",
            hover_data=["as_path_probs_observed_variables"],
            legend_title_text="ASN sequence hash",
            opacity=0.5,
            **extras,
        ),
        use_container_width=True,
    )
    per_ip.plotly_chart(
        get_plot(
            src_site,
            dest_site,
            route,
            y="n_hops_model_observed_values",
            color="ip_path_probs_observed_variables",
            hover_data=["ip_path_probs_observed_variables"],
            legend_title_text="IP sequence hash",
            opacity=0.5,
            **extras,
        ),
        use_container_width=True,
    )

    TOC.subheader("N-hops probability")
    prob, per_as_p, per_ip_p = st.tabs(["Probabilities", "Per AS", "Per IP"])
    extras = dict(
        title="Probality of number of hops per traceroute",
        xaxis_title="Time",
        yaxis_title="Probability (N-hops)",
    )

    prob.plotly_chart(
        get_plot(
            src_site,
            dest_site,
            route,
            y="n_hops_model_probabilities",
            color="n_hops_model_observed_values",
            hover_data=["n_hops_model_observed_values"],
            opacity=1,
            coloraxis_colorbar_title_text="Value of N-hops",
            **extras,
        ),
        use_container_width=True,
    )
    per_as_p.plotly_chart(
        get_plot(
            src_site,
            dest_site,
            route,
            y="n_hops_model_probabilities",
            color="as_path_probs_observed_variables",
            hover_data=["as_path_probs_observed_variables"],
            opacity=0.5,
            legend_title_text="AS sequence hash",
            **extras,
        ),
        use_container_width=True,
    )
    per_ip_p.plotly_chart(
        get_plot(
            src_site,
            dest_site,
            route,
            y="n_hops_model_probabilities",
            color="ip_path_probs_observed_variables",
            hover_data=["ip_path_probs_observed_variables"],
            opacity=0.5,
            legend_title_text="IP sequence hash",
            **extras,
        ),
        use_container_width=True,
    )


def plot_rtt_section(src_site, dest_site, route):
    TOC.header("Round Trip Time")
    st.pyplot(plot_rtt(src_site, dest_site, route), use_container_width=True)

    df = get_trace_model_df(src_site, dest_site, route)
    with st.expander("Anomaly summary"):
        st.dataframe(
            df[df["trace_model_trace_rtt_anomalies"]]["trace_model_rtt_sum_errors"]
            .rename("RTT anomalies")
            .agg(["count", "sum", "mean", "min", "max"])
            .to_frame(),
            use_container_width=True,
        )

    default, summed, mean = st.tabs(["RTT", "Sum RTT", "Mean RTT"])
    default.plotly_chart(
        get_plot(
            src_site,
            dest_site,
            route,
            y="trace_model_rtt_sum_errors",
            color="trace_model_rtt_mean_errors",
            hover_data=[
                "as_path_probs_observed_variables",
                "ip_path_probs_observed_variables",
            ],
            opacity=0.5,
            legend_title_text="Sum of RTT error (ms)",
            title="Sum RTT error (per path)",
            xaxis_title="Time",
            yaxis_title="Sum RTT error (ms)",
            coloraxis_colorbar_title_text="Mean RTT error (ms)",
        ),
        use_container_width=True,
    )

    summed.plotly_chart(
        get_plot(
            src_site,
            dest_site,
            route,
            y="trace_model_rtt_sum_errors",
            color="trace_model_trace_rtt_anomalies",
            hover_data=["ip_path_probs_observed_variables"],
            symbol="trace_model_trace_rtt_anomalies",
            opacity=0.5,
            legend_title_text="Anomaly",
            title="Sum TTL error (per path)",
            xaxis_title="Time",
            yaxis_title="Sum TTL error (ms)",
        ),
        use_container_width=True,
    )
    mean.plotly_chart(
        get_plot(
            src_site,
            dest_site,
            route,
            y="trace_model_rtt_mean_errors",
            color="trace_model_trace_rtt_anomalies",
            hover_data=[
                "as_path_probs_observed_variables",
                "as_path_probs_observed_variables",
                "trace_model_rtt_sum_errors",
            ],
            opacity=0.5,
            coloraxis_colorbar_title_text="ASN sequence hash",
            legend_title_text="Anomaly",
            symbol="trace_model_trace_rtt_anomalies",
            title="Mean RTT error (per path)",
            xaxis_title="Time",
            yaxis_title="Mean RTT error (ms)",
        ),
        use_container_width=True,
    )


def plot_ttl_section(src_site, dest_site, route):
    TOC.header("Time to Live")
    st.pyplot(plot_ttl(src_site, dest_site, route), use_container_width=True)

    df = get_trace_model_df(src_site, dest_site, route)
    with st.expander("Anomaly summary"):
        st.dataframe(
            df[df["trace_model_trace_ttl_anomalies"]]["trace_model_ttl_sum_errors"]
            .rename("TTL anomalies")
            .agg(["count", "sum", "mean", "min", "max"])
            .to_frame(),
            use_container_width=True,
        )

    default, summed, mean = st.tabs(["TTL", "Sum TTL", "Mean TTL"])
    default.plotly_chart(
        get_plot(
            src_site,
            dest_site,
            route,
            y="trace_model_ttl_sum_errors",
            color="trace_model_ttl_mean_errors",
            hover_data=[
                "as_path_probs_observed_variables",
                "ip_path_probs_observed_variables",
            ],
            opacity=0.5,
            legend_title_text="Sum of TTL error (ms)",
            title="Sum TTL error (per path)",
            xaxis_title="Time",
            yaxis_title="Sum TTL error (ms)",
            coloraxis_colorbar_title_text="Mean TTL error (ms)",
            symbol="trace_model_trace_ttl_anomalies",
        ),
        use_container_width=True,
    )
    summed.plotly_chart(
        get_plot(
            src_site,
            dest_site,
            route,
            y="trace_model_ttl_sum_errors",
            color="trace_model_trace_ttl_anomalies",
            hover_data=["ip_path_probs_observed_variables"],
            symbol="trace_model_trace_ttl_anomalies",
            opacity=0.5,
            legend_title_text="Anomaly",
            title="Sum TTL error (per path)",
            xaxis_title="Time",
            yaxis_title="Sum TTL error (ms)",
        ),
        use_container_width=True,
    )
    mean.plotly_chart(
        get_plot(
            src_site,
            dest_site,
            route,
            y="trace_model_ttl_mean_errors",
            color="trace_model_trace_ttl_anomalies",
            hover_data=[
                "as_path_probs_observed_variables",
                "as_path_probs_observed_variables",
                "trace_model_ttl_sum_errors",
            ],
            opacity=0.5,
            coloraxis_colorbar_title_text="ASN sequence hash",
            legend_title_text="Anomaly",
            symbol="trace_model_trace_ttl_anomalies",
            title="Mean TTL error (per path)",
            xaxis_title="Time",
            yaxis_title="Mean TTL error (ms)",
        ),
        use_container_width=True,
    )


def plot_destination_reached_section(src_site, dest_site, route):
    TOC.header("Destination reached")
    df = get_trace_model_df(src_site, dest_site, route)
    st.pyplot(
        plot_destination_reached(src_site, dest_site, route), use_container_width=True
    )
    with st.expander("Anomaly summary"):
        left, middle, right, _ = st.columns((0.3, 1, 1, 0.05))
        left.metric(
            "Number of anomalies", df[df["destination_reached_anomalies"]].shape[0]
        )
        middle.subheader("AS sequences with multiple outcomes")
        try:
            tmp = df.groupby("as_path_probs_observed_variables")[
                "destination_reached_observed_variables"
            ].agg(["count", pd.Series.nunique, pd.Series.mode])
            middle.dataframe(tmp[tmp["nunique"] > 1], use_container_width=True)

        except ValueError:
            pass
        right.subheader("IP sequences with multiple outcomes")
        try:
            tmp = df.groupby("ip_path_probs_observed_variables")[
                "destination_reached_observed_variables"
            ].agg(["count", pd.Series.nunique, pd.Series.mode])
            right.dataframe(tmp[tmp["nunique"] > 1], use_container_width=True)

        except ValueError:
            pass

    normal, per_as, per_ip = st.tabs(["Distribution", "Per AS", "Per IP"])
    normal.plotly_chart(
        get_plot(
            src_site,
            dest_site,
            route,
            y="destination_reached_observed_variables",
            color="destination_reached_observed_variables",
            hover_data=[
                "ip_path_probs_observed_variables",
                "as_path_probs_observed_variables",
            ],
            opacity=0.1,
            legend_title_text="",
            title="Probability of reaching destination",
            xaxis_title="Time",
            yaxis_title="Probability",
            coloraxis_colorbar_title_text="Destination reached",
        ),
        use_container_width=True,
    )
    per_as.plotly_chart(
        get_plot(
            src_site,
            dest_site,
            route,
            y="destination_reached_observed_variables",
            color="as_path_probs_observed_variables",
            hover_data=[
                "as_path_probs_observed_variables",
                "ip_path_probs_observed_variables",
            ],
            opacity=0.5,
            legend_title_text="AS sequence hash",
            title="Probability of reaching destination (Per AS)",
            xaxis_title="Time",
            yaxis_title="Probability",
            coloraxis_colorbar_title_text="Destination reached",
        ),
        use_container_width=True,
    )
    per_ip.plotly_chart(
        get_plot(
            src_site,
            dest_site,
            route,
            y="destination_reached_observed_variables",
            color="ip_path_probs_observed_variables",
            hover_data=[
                "as_path_probs_observed_variables",
                "ip_path_probs_observed_variables",
            ],
            opacity=0.5,
            legend_title_text="IP sequence hash",
            title="Probability of reaching destination (Per IP)",
            xaxis_title="Time",
            yaxis_title="Probability",
            coloraxis_colorbar_title_text="Destination reached",
        ),
        use_container_width=True,
    )


def plot_path_complete_section(src_site, dest_site, route):
    TOC.header("Path complete")
    df = get_trace_model_df(src_site, dest_site, route)
    st.pyplot(plot_path_complete(src_site, dest_site, route), use_container_width=True)
    with st.expander("Anomaly summary"):
        left, middle, right, _ = st.columns((0.3, 1, 1, 0.05))
        left.metric("Number of anomalies", df[df["path_complete_anomalies"]].shape[0])
        middle.subheader("AS sequences with multiple outcomes")
        tmp = df.groupby("as_path_probs_observed_variables")[
            "path_complete_observed_variables"
        ].agg(["count", pd.Series.nunique, pd.Series.mode])
        middle.dataframe(
            tmp[tmp["nunique"] > 1],
            use_container_width=True,
        )
        right.subheader("IP sequences with multiple outcomes")
        tmp = df.groupby("ip_path_probs_observed_variables")[
            "path_complete_observed_variables"
        ].agg(["count", pd.Series.nunique, pd.Series.mode])
        right.dataframe(
            tmp[tmp["nunique"] > 1],
            use_container_width=True,
        )
    normal, per_as, per_ip = st.tabs(["Distribution", "Per AS", "Per IP"])
    normal.plotly_chart(
        get_plot(
            src_site,
            dest_site,
            route,
            y="path_complete_observed_variables",
            color="path_complete_observed_variables",
            hover_data=[
                "ip_path_probs_observed_variables",
                "as_path_probs_observed_variables",
            ],
            opacity=0.1,
            legend_title_text="",
            title="Probability of reaching path being complete",
            xaxis_title="Time",
            yaxis_title="Probability",
            coloraxis_colorbar_title_text="Path complete",
        ),
        use_container_width=True,
    )
    per_as.plotly_chart(
        get_plot(
            src_site,
            dest_site,
            route,
            y="path_complete_observed_variables",
            color="as_path_probs_observed_variables",
            hover_data=[
                "as_path_probs_observed_variables",
                "ip_path_probs_observed_variables",
            ],
            opacity=0.5,
            legend_title_text="AS sequence hash",
            title="Probability of reaching path beign complete (Per AS)",
            xaxis_title="Time",
            yaxis_title="Probability",
            coloraxis_colorbar_title_text="Path complete",
        ),
        use_container_width=True,
    )
    per_ip.plotly_chart(
        get_plot(
            src_site,
            dest_site,
            route,
            y="path_complete_observed_variables",
            color="ip_path_probs_observed_variables",
            hover_data=[
                "as_path_probs_observed_variables",
                "ip_path_probs_observed_variables",
            ],
            opacity=0.5,
            legend_title_text="IP sequence hash",
            title="Probability of reaching path being complete (Per IP)",
            xaxis_title="Time",
            yaxis_title="Probability",
            coloraxis_colorbar_title_text="Path being complete",
        ),
        use_container_width=True,
    )


def plot_path_looping_section(src_site, dest_site, route):
    TOC.header("Looping")
    df = get_trace_model_df(src_site, dest_site, route)
    st.pyplot(plot_looping(src_site, dest_site, route), use_container_width=True)
    with st.expander("Anomaly summary"):
        left, middle, right, _ = st.columns((0.3, 1, 1, 0.05))
        left.metric("Number of anomalies", df[df["looping_anomalies"]].shape[0])
        middle.subheader("AS sequences with multiple outcomes")
        tmp = df.groupby("as_path_probs_observed_variables")[
            "looping_observed_variables"
        ].agg(["count", pd.Series.nunique, pd.Series.mode])
        middle.dataframe(
            tmp[tmp["nunique"] > 1],
            use_container_width=True,
        )
        right.subheader("IP sequences with multiple outcomes")
        tmp = df.groupby("ip_path_probs_observed_variables")[
            "looping_observed_variables"
        ].agg(["count", pd.Series.nunique, pd.Series.mode])
        right.dataframe(
            tmp[tmp["nunique"] > 1],
            use_container_width=True,
        )
    normal, per_as, per_ip = st.tabs(["Distribution", "Per AS", "Per IP"])
    normal.plotly_chart(
        get_plot(
            src_site,
            dest_site,
            route,
            y="looping_observed_variables",
            color="looping_observed_variables",
            hover_data=[
                "ip_path_probs_observed_variables",
                "as_path_probs_observed_variables",
            ],
            opacity=0.1,
            legend_title_text="",
            title="Probability of path looping",
            xaxis_title="Time",
            yaxis_title="Probability",
            coloraxis_colorbar_title_text="Path complete",
        ),
        use_container_width=True,
    )
    per_as.plotly_chart(
        get_plot(
            src_site,
            dest_site,
            route,
            y="looping_observed_variables",
            color="as_path_probs_observed_variables",
            hover_data=[
                "as_path_probs_observed_variables",
                "ip_path_probs_observed_variables",
            ],
            opacity=0.5,
            legend_title_text="AS sequence hash",
            title="Probability of path loopinhg (Per AS)",
            xaxis_title="Time",
            yaxis_title="Probability",
            coloraxis_colorbar_title_text="Looping",
        ),
        use_container_width=True,
    )
    per_ip.plotly_chart(
        get_plot(
            src_site,
            dest_site,
            route,
            y="looping_observed_variables",
            color="ip_path_probs_observed_variables",
            hover_data=[
                "as_path_probs_observed_variables",
                "ip_path_probs_observed_variables",
            ],
            opacity=0.5,
            legend_title_text="IP sequence hash",
            title="Probability of reaching path being complete (Per IP)",
            xaxis_title="Time",
            yaxis_title="Probability",
            coloraxis_colorbar_title_text="Path being complete",
        ),
        use_container_width=True,
    )



def plot_n_anomalies(src_site, dest_site, route):
    model = get_trace_model(src_site, dest_site, route)
    df = pd.DataFrame([x.dict() for x in model.n_anomalies], index=pd.to_datetime(model.timestamps, unit='ms'))
    X = df.stack().reset_index()
    if X.empty:
        return
    TOC.header("Number of anomalies on route ")
    first, second = st.tabs(["Observed", "Model"])
    
    X = X[X[0] == True]
    X.columns = ["timestamp", "anomaly", "value"]
    X["value"] = X["value"].astype(int)
    period = st.selectbox("Period", ["1H","30min", "12H", "1D", "1W", "2W", "1M"])
    stacked = st.toggle("Stacked", True)
    mode=  "stack" if stacked else "group"
    X = X.groupby("anomaly").resample(period, on='timestamp').sum(numeric_only=True).reset_index()
    fig = X.plot(kind='bar', barmode=mode,  x="timestamp", y="value",  color='anomaly', backend='plotly')
    first.plotly_chart(fig, use_container_width=True)
    
    fig = plt.figure(figsize=(12, 6))
    model.anomalies_model.plot(ax=fig.gca())
    plt.title("Probabilities of number of anomalies")
    second.pyplot(fig, use_container_width=True)

@st.cache_resource(max_entries=20, show_spinner=False)
def collect_df(src, dest, route):
    model = get_trace_model(src, dest, route)
    df = pd.DataFrame([x.dict() for x in model.n_anomalies], 
            index=pd.to_datetime(model.timestamps, unit='ms')).rename(columns=column_name_mapping)

    X = df.stack().reset_index()

    if X.empty:
        return X

    X = X[X[0] == True]
    X.columns = ["timestamp", "anomaly", "value"]
    X["value"] = X["value"].astype(int)
    X["route"] = route
    return X

def collect_all_df(src, dest):
    site_to_site: SiteAnalyzer = get_site_to_site()
    dfs = []
    for k in site_to_site.site_to_site[src][dest].trace_analyzer:
        dfs.append(collect_df(src, dest, k))
    return pd.concat(dfs, ignore_index=True)    

def plot_site_to_site_anomalies(src_site, dest_site):
    df = collect_all_df(src_site, dest_site)

    if df.empty:
        st.toast(":warning: No anomaly data for this site pair")
        return

    TOC.subheader("Site to site anomalies")
    period = st.selectbox("Period", [ "1H", "30min","3H", "8H", "12H", "1D", "1W", "2W", "1M"], key="period_1")
    stacked = st.toggle("Stacked", True, key="toggle1")
    mode =  "stack" if stacked else "group"
    X = (df[["timestamp", "value", "anomaly"]].groupby("anomaly").resample(period, on='timestamp').sum(numeric_only=True).reset_index())
    fig = X.plot(kind='bar', barmode=mode,  x="timestamp", y="value",  color='anomaly', backend='plotly')    
    st.plotly_chart(fig, use_container_width=True)

def plot_all_anomalies():
    TOC.header("Global anomalies")   
    model = get_site_to_site()
    dfs = [collect_all_df(src, dest) for src in model.site_to_site for dest in model.site_to_site[src]]
    df = pd.concat(dfs, ignore_index=True)
    period = st.selectbox("Period", ["1H","30min", "3H", "8H", "12H", "1D", "1W", "2W", "1M"], key="period_0")
    stacked = st.toggle("Stacked", True, key="toggle_0")
    mode =  "stack" if stacked else "group"
    X = (df[["timestamp", "value", "anomaly"]].groupby("anomaly").resample(period, on='timestamp').sum(numeric_only=True).reset_index())
    fig = X.plot(kind='bar', barmode=mode,  x="timestamp", y="value",  color='anomaly', backend='plotly')    
    st.plotly_chart(fig, use_container_width=True)