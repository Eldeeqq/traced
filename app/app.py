import glob
import random
import sys

sys.path.insert(0, "..")


import streamlit as st
from streamlit_folium import st_folium
from utils import (get_site_to_site, plot_all_anomalies, plot_asn_section,
                   plot_destination_reached_section, plot_ip_section,
                   plot_n_anomalies, plot_n_hops_section,
                   plot_path_complete_section, plot_path_looping_section,
                   plot_paths_folium, plot_rtt_section,
                   plot_site_to_site_anomalies, plot_ttl_section,
                   plot_global_paths,
                   plot_site_paths)

from traced_v2.site_analyzer import SiteAnalyzer



st.set_page_config(
    layout="wide",
    page_title="Traced demo",
    page_icon=":globe_with_meridians:",
    initial_sidebar_state="expanded",
)


files = list(glob.glob("../data/pic-CSCS_LCG2/*.json"))
sample = random.choice(files)

# st.header("Sample record")
# with open(sample) as f:
#     st.json(f.read())


analyzer: SiteAnalyzer = get_site_to_site()
sources = list(analyzer.site_to_site.keys())
st.sidebar.title("Select site pair")
source = st.sidebar.selectbox("Source", sources, help="Select source site")

if not sources or source not in analyzer.site_to_site:
    st.warning("There has been an issue. Try different site.")
    st.stop()

destinations = list(analyzer.site_to_site[source].keys())
destinations = destinations[1:] + [destinations[0]]
dest = st.sidebar.selectbox("Destination", destinations, help="Select destination site")

if not destinations or dest not in analyzer.site_to_site[source]:
    st.warning("There has been an issue. Try different site.")
    st.stop()

st.sidebar.toggle(
    "Show help", True, key="help", help="Shows help across the dashboard"
)
st.sidebar.success(
    """How to use the dashboard:

- You can chage the source and destination site using the selectboxes on the left.
- Note that whenever anything changes, the dashboard will re-do the plots and it might take a while. This is indicated by the loading icon in the right corner of the page.
If this icon is active, please wait until the dashboard is updated.
- There are help sections explaining the plots. You can show/hide them using the toggle on the left.
- There is a known bug with the map visualisation, which sometimes does not work. If this happens, please refresh the page, or restart the app.
    """
)


st.header("Traced Dashboard")

st.subheader("Global aggregation of anomalies")

with st.expander("Info", st.session_state["help"]):
    st.markdown(
        """
    This section shows the global aggregation of anomalies. 

    The bar plot contains the number of anomalies detected at specific time. You can change the aggregation period using the selectbox.
    The anomalies can be of multiple types and the segments of the bar plot are colored accordingly.

    Type of anomalies:
    - `AS Path Anomaly`: Unusual AS path
    - `AS Transition Anomaly`: Unusual AS transition
    - `IP Path Anomaly`: Unusual IP path
    - `IP Transition Anomaly`: Unusual IP transition
    - `Destination Reached Anomaly`: The result of the traceroute is different than expected
    - `Path Complete Anomaly`: The traceroute did not reach the destination
    - `Looping Anomaly`: The traceroute is looping
    - `RTT delay Anomaly`: The RTT delay is unusual
    - `TTL delay Anomaly`: The TTL delay is unusual
    - `Number of hops Anomaly`: The number of hops is unusual
    """
    )
plot_all_anomalies()
with st.expander("Global map visualisation"):
    st_folium(plot_global_paths(), use_container_width=True, returned_objects=[])

st.divider()

st.subheader(f"Site to site results: {source} -> {dest}")
traces = list(analyzer.site_to_site[source][dest].trace_analyzer.keys())
n = analyzer.site_to_site[source][dest].n
site_to_site = analyzer.site_to_site[source][dest]

with st.expander("Info", st.session_state["help"]):
    st.markdown(
        """
    This section shows the site level aggregation of anomalies. 

    The bar plot contains the number of anomalies detected at specific time. You can change the aggregation period using the selectbox.
    The anomalies can be of multiple types and the segments of the bar plot are colored accordingly.

    Type of anomalies:
    - `AS Path Anomaly`: Unusual AS path
    - `AS Transition Anomaly`: Unusual AS transition
    - `IP Path Anomaly`: Unusual IP path
    - `IP Transition Anomaly`: Unusual IP transition
    - `Destination Reached Anomaly`: The result of the traceroute is different than expected
    - `Path Complete Anomaly`: The traceroute did not reach the destination
    - `Looping Anomaly`: The traceroute is looping
    - `RTT delay Anomaly`: The RTT delay is unusual
    - `TTL delay Anomaly`: The TTL delay is unusual
    - `Number of hops Anomaly`: The number of hops is unusual
    """
    )
plot_site_to_site_anomalies(source, dest)
with st.expander("Site level map visualisation"):
    st_folium(plot_site_paths(source, dest), use_container_width=True, returned_objects=[])
st.divider()
st.subheader("Device to device results")

with st.expander("Info", st.session_state["help"]):
    st.markdown(
        """
    This section shows the results of the models on the device level. 

    Below you can select a route between two devices for inspection. The routes are ordered based on the number of occurences.

    Directly below that you can see a map visualising the devices that occured on the path. The map is interactive and you can zoom in and out and. To get further information about a device, click on the marker. To see the transition probability between two devices, hover on the edge between the devices. There can be overlaping edges, in that case you need to zoom in. The source device is coloured green with play icon, the destination device is coloured red with stop icon. The devices in between are coloured blue with server icon. If the device is coloured orange, it means that the geolocation of the device is not known. 

    Further below are the results from each of the model with some additional information.

    """
    )
trace = st.selectbox(
    "Route",
    sorted(traces, key=lambda x: site_to_site.trace_analyzer[x].n, reverse=True),
    format_func=lambda x: f"({100*site_to_site.trace_analyzer[x].n/n:2.1f}%) {x} ",
    placeholder="Choose a device pair",
)

if not trace and trace not in analyzer.site_to_site[source][dest].trace_analyzer:
    st.warning("There has been an issue. Try different route.")
    st.stop()

with st.expander("Info", st.session_state["help"]):
    st.markdown(
        """
    This section shows the device level aggregation of anomalies. 

    The bar plot contains the number of anomalies detected at specific time. You can change the aggregation period using the selectbox.
    The anomalies can be of multiple types and the segments of the bar plot are colored accordingly.

    Type of anomalies:
    - `AS Path Anomaly`: Unusual AS path
    - `AS Transition Anomaly`: Unusual AS transition
    - `IP Path Anomaly`: Unusual IP path
    - `IP Transition Anomaly`: Unusual IP transition
    - `Destination Reached Anomaly`: The result of the traceroute is different than expected
    - `Path Complete Anomaly`: The traceroute did not reach the destination
    - `Looping Anomaly`: The traceroute is looping
    - `RTT delay Anomaly`: The RTT delay is unusual
    - `TTL delay Anomaly`: The TTL delay is unusual
    - `Number of hops Anomaly`: The number of hops is unusual
    """
    )
plot_n_anomalies(source, dest, trace)

try:
    with st.expander("Device level map visualisation"):

        geo_plot = plot_paths_folium(source, dest, trace)
        st_folium(geo_plot, use_container_width=True, returned_objects=[])
except AttributeError:
    st.toast("Unable to plot geo map")
    pass

trace_model = analyzer.site_to_site[source][dest].trace_analyzer[trace]

tabs = st.tabs(
    [
        "AS sequences",
        "IP sequences",
        "Number of hops",
        "RTT",
        "TTL",
        "Destination reached",
        "Path complete",
        "Path looping",
    ]
)
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

# st.selectbox("cols",get_trace_model_df(source,dest, trace).columns)
