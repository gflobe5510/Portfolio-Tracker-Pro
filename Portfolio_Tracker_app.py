import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
from portfolio import (
    load_local_data,
    calculate_metrics,
    plot_price_chart,
    plot_bar_chart,
    plot_pie_chart
)

# Set Streamlit page config (must be first Streamlit command)
st.set_page_config(page_title="📊 Portfolio Tracker Pro+", layout="wide", initial_sidebar_state="expanded")

# Background image via CSS injection
def set_bg():
    with open("hector-j-rivas-1FxMET2U5dU-unsplash.jpg", "rb") as image_file:
        encoded = image_file.read().encode("base64")
    st.markdown(
        f"""<style>
            .stApp {{
                background-image: url('data:image/png;base64,{encoded}');
                background-size: cover;
            }}
        </style>""",
        unsafe_allow_html=True
    )

set_bg()

# Load ticker list
ticker_df = pd.read_csv("replacement_data/security_list.csv")

# Sidebar - Portfolio and benchmark selection
st.sidebar.header("Portfolio Configuration")
tickers = ticker_df["Symbol"].tolist()
selected_tickers = st.sidebar.multiselect("Select Portfolio Assets", tickers[:50], default=["AAPL", "MSFT"])
selected_bench = st.sidebar.selectbox("Select Benchmark", ["SPY"])

start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2023-12-31"))

# Sidebar - Weights
weights = {}
for ticker in selected_tickers:
    weights[ticker] = st.sidebar.slider(f"{ticker} Weight (%)", 0, 100, 100 // len(selected_tickers), key=f"weight_{ticker}")
total_weight = sum(weights.values())
if total_weight != 100:
    for ticker in weights:
        weights[ticker] = round(weights[ticker] / total_weight * 100, 2)

# Load data from local CSVs
try:
    price_data, bench_data = load_local_data(selected_tickers, selected_bench, start_date, end_date)
except Exception as e:
    st.error(f"Failed to load asset data: {e}")
    st.stop()

# Tabs for visualization
tab1, tab2, tab3 = st.tabs(["📈 Performance", "📊 Allocation", "📄 Metrics"])

with tab1:
    st.subheader("Portfolio Performance vs Benchmark")
    plot_price_chart(price_data, bench_data)

with tab2:
    st.subheader("Portfolio Allocation")
    plot_pie_chart(weights)

with tab3:
    st.subheader("Portfolio Metrics")
    stats = calculate_metrics(price_data, weights)
    st.dataframe(stats)
