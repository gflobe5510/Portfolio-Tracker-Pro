
import streamlit as st
import pandas as pd
import numpy as np
import os
from portfolio import (
    load_local_data,
    calculate_metrics,
    plot_price_chart,
    plot_bar_chart,
    plot_pie_chart,
    generate_pdf_report
)

# Set page config FIRST
st.set_page_config(
    page_title="📊 Portfolio Tracker Pro+ (Offline Mode)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Background styling
def set_background():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("https://images.unsplash.com/photo-1504384308090-c894fdcc538d");
            background-size: cover;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_background()

# Load ticker metadata
ticker_df = pd.read_csv("replacement_data/security_list.csv")

# Sidebar controls
st.sidebar.header("Controls")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))
risk_free_rate = st.sidebar.slider("Risk-Free Rate (%)", 0.0, 10.0, 2.0)

# Ticker selection
st.markdown("## 📈 Portfolio Tracker Pro+ (Offline Mode)")
st.markdown("This version uses local CSV files for simulation instead of fetching data from live APIs.")
selected_tickers = st.multiselect(
    "Select your assets:",
    options=ticker_df["Ticker"].tolist(),
    default=ticker_df["Ticker"].tolist()[:5],
    max_selections=10
)

benchmark = st.selectbox("Select benchmark:", options=selected_tickers if selected_tickers else ["None"])

# Weight sliders
st.sidebar.header("Portfolio Allocation")
weights = {}
for ticker in selected_tickers:
    weights[ticker] = st.sidebar.slider(
        f"{ticker} Weight (%)", 0, 100, 100 // len(selected_tickers), key=f"weight_{ticker}"
    )

# Normalize weights
total_weight = sum(weights.values())
if total_weight != 100:
    for ticker in weights:
        weights[ticker] = (weights[ticker] / total_weight) * 100

# Load data and analysis
tab1, tab2, tab3 = st.tabs(["Overview", "Analytics", "Forecast"])
with st.status("🔄 Loading market data...", expanded=True):
    try:
        data, bench_data = load_local_data(selected_tickers, start_date, end_date, benchmark)
        if data is None or data.empty:
            st.error("❌ Data loading failed: No valid data found.")
        else:
            with tab1:
                st.success("✅ Data loaded successfully!")
                plot_price_chart(data, bench_data)
                st.dataframe(data.tail())

            with tab2:
                metrics_df = calculate_metrics(data, weights, risk_free_rate)
                st.dataframe(metrics_df)
                plot_bar_chart(metrics_df)
                plot_pie_chart(weights)

            with tab3:
                st.markdown("### Forecasting Placeholder")
    except Exception as e:
        st.error(f"❌ Data loading failed: {e}")
