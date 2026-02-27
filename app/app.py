import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import folium_static
import plotly.express as px
import plotly.graph_objects as go
import joblib
import sys
from pathlib import Path
# Page configuration
st.set_page_config(
    page_title="Kenya EV Charging Station Optimizer",
    page_icon="⚡",
    layout="wide"
)
# Title and introduction
st.title("⚡ Kenya EV Charging Station Location Optimizer")
st.markdown("""
This application uses population data and machine learning to recommend optimal locations 
for electric vehicle charging stations across Kenyan counties.
""")
# Load data
@st.cache_data
def load_data():
    county_df = pd.read_csv("./utilities/county_data_clean.csv")
    kenya_stations = pd.read_csv("./utilities/kenya_stations_clean.csv")
    return county_df, kenya_stations

county_df, kenya_stations = load_data()
# Load model
@st.cache_resource
def load_models():
    additional_station_model = joblib.load("../ml_models/additional_station_model.joblib")
    county_kmeans_models = joblib.load("../ml_models/county_models.joblib")
    return additional_station_model, county_kmeans_models

additional_station_model, county_kmeans_models = load_models()
# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Dashboard", "Infrastructure Gap Analysis", "Station Placement", "Future Projections", "About"]
)
st.sidebar.header("User Input")

selected_county = st.sidebar.selectbox(
    "Select County",
    county_df["county"].unique()
)

population_input = st.sidebar.number_input(
    "Enter Population (Optional)",
    min_value=0,
    value=0
)

existing_stations_input = st.sidebar.number_input(
    "Enter Existing Stations (Optional)",
    min_value=0,
    value=0
)

predict_button = st.sidebar.button("Predict Stations")
