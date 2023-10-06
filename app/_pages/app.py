import folium
import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_folium import st_folium
from streamlit_plotly_events import plotly_events

st.set_page_config(layout="wide")

st.header("Site-to-Site Anomalies")


@st.cache_data
def get_data():
    df = pd.read_csv("data/score.csv", index_col=0)
    df2 = pd.read_csv("data/scored_data.csv")
    df2["ts"] = pd.to_datetime(df2["ts"], unit="ms")
    return df, df2


@st.cache_resource
def get_plot(df):
    fig = px.line(
        df,
        y=["mean", "mean_mix", "hmean_mix"],
        markers="lines+markers",
        color_discrete_sequence=[
            "#0068c9",
            "#ff2b2b",
            "#29b09d",
            "#7defa1",
            "#ff8700",
            "#ffd16a",
            "#6d3fc0",
            "#d5dae5",
        ],
    )
    fig.update_layout(
        width=1500,
        height=500,
        yaxis_title="anomaly score",
        xaxis_title="timestep",
    )
    fig.update_xaxes(
        rangeslider_visible=True,
    )
    return fig


df, df2 = get_data()

with st.spinner("Creating plot"):
    with st.container():
        # Writes a component similar to st.write()
        fig = get_plot(df)
        selected_points = plotly_events(
            fig,
            key="plotly_events",
            override_height=500,
            click_event=True,
            hover_event=False,
            select_event=True,
        )


if not selected_points:
    st.write("No points selected")
else:
    # st.write(selected_points)
    row = df2[df2["ts"] == selected_points[0]["x"]]
    st.write(row.columns)
    st.write(row[["ttl_probs", "rtt_probs"]])

with st.container():
    m = folium.Map(location=[39.949610, -75.150282], zoom_start=16)
    folium.Marker(
        [39.949610, -75.150282], popup="Liberty Bell", tooltip="Liberty Bell"
    ).add_to(m)

    # call to render Folium map in Streamlit
    st_data = st_folium(
        m,
        width=725,
        height=500,
        key="folium1",
        return_on_hover=False,
        use_container_width=True,
        returned_objects=[],  # ["last_object_clicked"]
    )
