# 1. First import Streamlit (MUST BE FIRST)
import streamlit as st

# 2. Set page config (MUST BE IMMEDIATELY AFTER)
st.set_page_config(
    page_title="📊 Portfolio Tracker Pro+",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 3. Now import other libraries
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from tenacity import retry, stop_after_attempt, wait_exponential
import time

# Enhanced data loading with better error handling
@st.cache_data(ttl=3600)
def load_data(tickers, start_date, end_date, benchmark=None):
    """Load financial data with robust error handling"""
    ticker_list = [t.strip().upper() for t in tickers.split(",")] if isinstance(tickers, str) else tickers
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def download_tickers():
        try:
            data = yf.download(
                ticker_list,
                start=start_date - timedelta(days=7),  # Buffer for timezone issues
                end=end_date + timedelta(days=1),
                group_by='ticker',
                progress=False,
                auto_adjust=True,
                threads=True,
                timeout=10
            )
            return data
        except Exception as e:
            st.error(f"Download failed: {str(e)}")
            return None
    
    # Download main assets
    data = download_tickers()
    if data is None:
        return None, None
    
    # Process downloaded data
    close_data = pd.DataFrame()
    for ticker in ticker_list:
        try:
            if len(ticker_list) == 1:
                if not data.empty and 'Close' in data.columns:
                    close_data[ticker] = data['Close']
            elif ticker in data.columns.get_level_values(0):
                if 'Close' in data[ticker]:
                    close_data[ticker] = data[ticker]['Close']
        except Exception as e:
            st.warning(f"Could not process {ticker}: {str(e)}")
    
    # Download benchmark if specified
    benchmark_data = None
    if benchmark:
        try:
            bench_data = yf.download(
                benchmark,
                start=start_date - timedelta(days=7),
                end=end_date + timedelta(days=1),
                progress=False,
                auto_adjust=True
            )
            if not bench_data.empty and 'Close' in bench_data.columns:
                benchmark_data = bench_data['Close']
                benchmark_data.name = f"Benchmark ({benchmark})"
        except Exception as e:
            st.warning(f"Could not download benchmark: {str(e)}")
    
    return close_data.ffill().dropna(), benchmark_data.ffill().dropna() if benchmark_data is not None else None

# Rest of your functions (normalize_prices, calculate_metrics, etc.) remain the same
# [Previous versions of these functions can be inserted here]

# Main app UI
st.title("📊 Portfolio Performance Tracker Pro+")

# Input controls
col1, col2 = st.columns(2)
with col1:
    tickers = st.text_input("Enter tickers (comma separated)", "AAPL, MSFT, GOOGL")
    start_date = st.date_input("Start date", datetime.now() - timedelta(days=365))
with col2:
    benchmark = st.text_input("Benchmark ticker (optional)", "^GSPC")
    end_date = st.date_input("End date", datetime.now())

# Analysis button
if st.button("Run Analysis", type="primary"):
    with st.spinner("Loading market data..."):
        data, bench = load_data(tickers, start_date, end_date, benchmark if benchmark else None)
        
        if data is not None and not data.empty:
            st.success("Data loaded successfully!")
            
            # Display results
            st.subheader("Performance Overview")
            plot_price_chart(data, bench)
            
            metrics = calculate_metrics(data, bench)
            if metrics is not None:
                st.subheader("Performance Metrics")
                st.dataframe(style_performance(metrics), use_container_width=True)
                plot_bar_chart(metrics)
        else:
            st.error("Failed to load data. Please check your tickers and try again.")

# Add some spacing
st.markdown("---")
st.caption("Note: Market data provided by Yahoo Finance. Past performance is not indicative of future results.")
