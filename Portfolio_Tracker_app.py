
import streamlit as st
import pandas as pd
import numpy as np
from portfolio import (
    load_data,
    calculate_metrics,
    plot_price_chart,
    plot_bar_chart,
    plot_pie_chart,
    optimize_portfolio
)
from pdf_utils import create_pdf_report

# Set page config
st.set_page_config(page_title="📊 Portfolio Tracker Pro+", layout="wide", initial_sidebar_state="expanded")

def main():
    st.title("📊 Portfolio Tracker Pro+")

    # Load data
    selected_tickers = ["AAPL", "MSFT"]
    start_date = "2020-01-01"
    end_date = "2023-01-01"
    benchmark = "SPY"

    with st.spinner("🔄 Loading market data..."):
        data, bench_data = load_data(selected_tickers, start_date, end_date, benchmark)

    if data is None or data.empty:
        st.error("Failed to load asset data")
        return

    st.success("✅ Data loaded successfully!")

    # Display a chart
    st.plotly_chart(plot_price_chart(data, bench_data), use_container_width=True)

    # Display metrics
    metrics = calculate_metrics(data)
    st.write(metrics)

    # Portfolio optimization section
    if st.button("Optimize Portfolio"):
        optimized_weights = optimize_portfolio(data)
        st.write("Optimized Weights:", optimized_weights)

if __name__ == "__main__":
    main()
