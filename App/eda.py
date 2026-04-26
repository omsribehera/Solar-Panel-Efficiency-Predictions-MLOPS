import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import logging

# Configure logging
logging.basicConfig(
    filename="App/app.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# Add AnalyzeSrc to sys.path for import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../Analysis/AnalyzeSrc')))

from UnivariateAnalysis import UnivariateAnalyzer, NumericalUnivariateAnalysis, CategoricalUnivariateAnalysis
from BivariateAnalysis import BivariateAnalyzer, NumericalVsNumericalAnalysis, CategoricalVsNumericalAnalysis, CategoricalVsCategoricalAnalysis
from MultivariateAnalysis import SimpleMultivariateAnalysis
from MissingValueAnalysis import SimpleMissingValuesAnalysis

def show_eda_tab():
    """
    Display the EDA (Exploratory Data Analysis) tab in the Streamlit app using AnalyzeSrc utilities.
    Includes missing value analysis, univariate, bivariate, and multivariate visualizations.
    """
    st.header("Exploratory Data Analysis (EDA)")
    uploaded_file = st.file_uploader("Upload a CSV file for EDA", type=["csv"])
    if uploaded_file is not None:
        logging.info(f"EDA: File uploaded: {uploaded_file.name}")
        df = pd.read_csv(uploaded_file)
        st.subheader("Data Preview")
        st.dataframe(df.head())

        st.subheader("Missing Values Analysis")
        if st.button("Show Missing Values Analysis"):
            logging.info("EDA: Missing Values Analysis triggered.")
            analyzer = SimpleMissingValuesAnalysis()
            with st.expander("Missing Values Count and Heatmap"):
                try:
                    analyzer.analyze(df)
                    st.pyplot(plt.gcf())
                    plt.clf()
                except Exception as e:
                    logging.error(f"EDA: Error in missing values analysis: {e}")
                    st.error(f"Error: {e}")

        st.subheader("Univariate Analysis")
        feature = st.selectbox("Select a feature for univariate analysis", df.columns)
        if feature:
            if pd.api.types.is_numeric_dtype(df[feature]):
                analyzer = UnivariateAnalyzer(NumericalUnivariateAnalysis())
            else:
                analyzer = UnivariateAnalyzer(CategoricalUnivariateAnalysis())
            if st.button("Show Univariate Analysis"):
                logging.info(f"EDA: Univariate Analysis for {feature}.")
                with st.expander(f"Univariate Analysis for {feature}"):
                    try:
                        analyzer.execute_analysis(df, feature)
                        st.pyplot(plt.gcf())
                        plt.clf()
                    except Exception as e:
                        logging.error(f"EDA: Error in univariate analysis: {e}")
                        st.error(f"Error: {e}")

        st.subheader("Bivariate Analysis")
        feature1 = st.selectbox("Feature 1", df.columns, key="biv1")
        feature2 = st.selectbox("Feature 2", df.columns, key="biv2")
        if feature1 and feature2 and feature1 != feature2:
            if pd.api.types.is_numeric_dtype(df[feature1]) and pd.api.types.is_numeric_dtype(df[feature2]):
                analyzer = BivariateAnalyzer(NumericalVsNumericalAnalysis())
            elif pd.api.types.is_object_dtype(df[feature1]) and pd.api.types.is_numeric_dtype(df[feature2]):
                analyzer = BivariateAnalyzer(CategoricalVsNumericalAnalysis())
            elif pd.api.types.is_object_dtype(df[feature1]) and pd.api.types.is_object_dtype(df[feature2]):
                analyzer = BivariateAnalyzer(CategoricalVsCategoricalAnalysis())
            else:
                analyzer = None
            if analyzer and st.button("Show Bivariate Analysis"):
                logging.info(f"EDA: Bivariate Analysis for {feature1} vs {feature2}.")
                with st.expander(f"Bivariate Analysis: {feature1} vs {feature2}"):
                    try:
                        analyzer.execute_analysis(df, feature1, feature2)
                        st.pyplot(plt.gcf())
                        plt.clf()
                    except Exception as e:
                        logging.error(f"EDA: Error in bivariate analysis: {e}")
                        st.error(f"Error: {e}")

        st.subheader("Multivariate Analysis")
        if st.button("Show Multivariate Analysis"):
            logging.info("EDA: Multivariate Analysis triggered.")
            analyzer = SimpleMultivariateAnalysis()
            with st.expander("Correlation Heatmap and Pairplot"):
                try:
                    analyzer.analyze(df.select_dtypes(include=['number']))
                    st.pyplot(plt.gcf())
                    plt.clf()
                except Exception as e:
                    logging.error(f"EDA: Error in multivariate analysis: {e}")
                    st.error(f"Error: {e}")
    else:
        st.info("Please upload a CSV file to see EDA visualizations.") 