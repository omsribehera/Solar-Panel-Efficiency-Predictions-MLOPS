import streamlit as st
import pandas as pd
import joblib
import os
from typing import Optional
from eda import show_eda_tab
from predict import show_predict_tab
import logging

# Configure logging
logging.basicConfig(
    filename="App/app.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def show_home_tab():
    """Display the Home tab with a welcome message and quick start info."""
    logging.info("User navigated to Home tab.")
    st.title("⚡ Solar Power System Analyzer")
    st.markdown("""
    Welcome to the Solar Power System Analyzer! This app allows you to:
    - Explore your solar data with interactive EDA tools
    - Predict outcomes using your trained model
    - Download results and visualizations
    
    **Get started by selecting a tab from the sidebar.**
    """)

def show_about_tab():
    """Display the About tab with project and author info."""
    logging.info("User navigated to About tab.")
    st.title("About")
    st.markdown("""
    **Solar Power System Analyzer**
    
    - Built with Streamlit, ZenML, and MLflow
    - Modular, production-ready, and extensible
    - Developed for robust solar data analysis and ML workflows
    
    **Author:** THAMIZHARASU SARAVANAN
    **GitHub:** [your-github-link](https://github.com/THAMIZH-ARASU)
    """)

st.set_page_config(page_title="Solar Power System Analyzer", layout="wide")

# Sidebar - App navigation
st.sidebar.title("Navigation")
tab = st.sidebar.radio("Go to", ("Home", "Prediction", "EDA", "About"))
logging.info(f"Tab selected: {tab}")

# Main app
if tab == "Home":
    show_home_tab()
elif tab == "Prediction":
    show_predict_tab()
elif tab == "EDA":
    show_eda_tab()
elif tab == "About":
    show_about_tab() 
