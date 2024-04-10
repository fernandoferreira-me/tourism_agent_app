import streamlit as st
import folium
import os
import pandas as pd

from streamlit_folium import folium_static
from folium.plugins import MeasureControl
import numpy as np

from dotenv import load_dotenv
from pathlib import Path
from agent import Agent
import json

def load_secrets():
    load_dotenv()
    env_path = Path(".") / ".env"
    load_dotenv(dotenv_path=env_path)

    open_ai_key = os.getenv("OPENAI_API_KEY")

    return {
        "OPENAI_API_KEY": open_ai_key
    }

CENTER_START = [48.86, 2.34]
ZOOM_START = 9
   
load_secrets()

agent = Agent(open_ai_api_key=os.getenv("OPENAI_API_KEY"))


st.set_page_config(layout="wide")
st.title("Tourism Intelligent App")


def initialize_session_state():
    if "center" not in st.session_state:
        st.session_state["center"] = CENTER_START
    if "zoom" not in st.session_state:
        st.session_state["zoom"] = ZOOM_START
    if "markers" not in st.session_state:
        st.session_state["markers"] = []


def initialize_map(center, zoom):
    if 'map' not in st.session_state or st.session_state.map is None:
        m = folium.Map(location=center,
                       zoom_start=zoom,
                       scrollWheelZoom=False)
        st.session_state.map = m
    return st.session_state.map

def reset_session_state():
    # Delete all the items in Session state besides center and zoom
    for key in st.session_state.keys():
        if key in ["center", "zoom"]:
            continue
        del st.session_state[key]
    initialize_session_state()

initialize_session_state()
m = initialize_map([48.86, 2.34], 9)



# Creating two columns for the layout
col1, col2 = st.columns(2)

with col1:
    # Button to add a marker
    query = st.text_area("Where would like to go?")
    button = st.button('Add Marker')
    box = st.container(height=300)
    with box:
        container = st.empty()
        container.header("Itinerary")

if button and query:
    itinerary = agent.get_itinerary(query)
    try:
        days = json.loads(itinerary['mapping_list'])
        days = days[0]['days']
    except KeyError:
        pass
    new_center = [json.loads(itinerary['center_location'])[k] for k in ['lat', 'lon']]
    zoom_start = json.loads(itinerary['center_location'])['zoom']
    locations =[]
    for points in days:
        for point in points.get('locations'):
            locations.append([point['lat'], point['lon']])
    reset_session_state()
    m = initialize_map(new_center, zoom_start)
    st.session_state["markers"] = [folium.Marker(location=location) for location in locations]
    with box:
        container.write(itinerary['agent_suggestion'])

with col2:
    fg = folium.FeatureGroup(name="Markers")
    for marker in st.session_state["markers"]:
        fg.add_child(marker)
    fg.add_to(m)
    folium_static(m)