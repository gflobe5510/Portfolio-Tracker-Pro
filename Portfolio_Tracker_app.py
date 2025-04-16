
import streamlit as st
import pandas as pd
from portfolio import load_data, calculate_metrics, plot_price_chart, plot_bar_chart
from datetime import datetime

st.set_page_config(page_title="📊 Portfolio Tracker Pro+", layout="wide", initial_sidebar_state="expanded")

st.title("📊 Portfolio Tracker Pro+ (Offline Mode)")
st.markdown("This version uses local CSV files for simulation instead of fetching data from live APIs.")

# Load ticker list
ticker_df = pd.read_csv("replacement_data/security_list.csv")
available_tickers = ticker_df["Ticker"].tolist()

selected_tickers = st.multiselect("Select your assets:", available_tickers, default=available_tickers[:5])
selected_bench = st.selectbox("Select benchmark:", available_tickers)

start_date = st.date_input("Start Date", datetime.today().replace(year=datetime.today().year - 1))
end_date = st.date_input("End Date", datetime.today())

# Portfolio Allocation Sliders
st.sidebar.header("Portfolio Allocation")
weights = {}
for ticker in selected_tickers:
    weights[ticker] = st.sidebar.slider(
        f"{ticker} Weight (%)",
        0, 100, int(100/len(selected_tickers)),
        key=f"weight_{ticker}"
    )

total_weight = sum(weights.values())
if total_weight == 0:
    st.sidebar.error("Total weight cannot be 0% - resetting to equal weights")
    equal_weight = 100/len(selected_tickers)
    for ticker in weights:
        weights[ticker] = equal_weight
elif total_weight != 100:
    st.sidebar.warning(f"Total weights sum to {total_weight}%. Normalizing to 100%.")
    for ticker in weights:
        weights[ticker] = (weights[ticker] / total_weight) * 100

with st.status("🔄 Loading market data...", expanded=True) as status:
    try:
        data, bench_data = load_data(
            selected_tickers,
            start_date,
            end_date,
            selected_bench
        )
        if data is None or data.empty:
            st.error("❌ Failed to load asset data")
        else:
            status.update(label="✅ Data loaded successfully", state="complete")
            st.plotly_chart(plot_price_chart(data, bench_data))
            cumulative, returns = calculate_metrics(data, weights)
            st.subheader("📈 Cumulative Return")
            st.line_chart(cumulative)
            st.subheader("📊 Portfolio Contribution")
            st.plotly_chart(plot_bar_chart(data.pct_change().fillna(0)))
    except Exception as e:
        st.error(f"Data loading failed: {e}")
