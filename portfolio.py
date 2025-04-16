import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_exponential

@st.cache_data(ttl=3600)
def load_data(tickers, start, end, benchmark=None):
    try:
        ticker_list = [tickers] if isinstance(tickers, str) else tickers
        
        @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
        def download_assets():
            return yf.download(ticker_list, start=start, end=end, group_by='ticker', progress=False, auto_adjust=True)
        
        data = download_assets()
        
        if isinstance(data.columns, pd.MultiIndex):
            close_data = pd.concat(
                [data[ticker]["Close"].rename(ticker) 
                 for ticker in ticker_list 
                 if "Close" in data[ticker]],
                axis=1
            )
        else:
            close_data = data[["Close"]].rename(columns={"Close": ticker_list[0]})

        benchmark_data = None
        if benchmark:
            try:
                @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
                def download_bench():
                    return yf.download(benchmark, start=start, end=end, progress=False, auto_adjust=True)
                
                bench_df = download_bench()
                if not bench_df.empty:
                    benchmark_data = bench_df["Close"].copy()
                    benchmark_data.name = f"Benchmark ({benchmark})"
            except Exception as e:
                st.warning(f"⚠️ Benchmark warning: {str(e)}")
                benchmark_data = None
        
        return close_data.dropna(), benchmark_data.dropna() if benchmark_data is not None else None
    
    except Exception as e:
        st.error(f"❌ Data loading failed: {e}")
        return None, None

def normalize_prices(data):
    """Normalize prices to base=100 for consistent scaling."""
    return (data / data.iloc[0]) * 100

def calculate_metrics(data, benchmark=None, risk_free_rate=0.02):
    returns = data.pct_change().dropna()
    metrics = {}
    
    annual_returns = (1 + returns.mean()) ** 252 - 1
    metrics["Annualized Return"] = annual_returns
    
    annual_volatility = returns.std() * np.sqrt(252)
    metrics["Annualized Volatility"] = annual_volatility
    
    sharpe_ratio = (annual_returns - risk_free_rate) / annual_volatility
    metrics["Sharpe Ratio"] = sharpe_ratio
    
    cumulative_returns = (1 + returns).cumprod()
    peak = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_returns / peak) - 1
    metrics["Max Drawdown"] = drawdown.min()
    
    if benchmark is not None:
        benchmark_returns = benchmark.pct_change().dropna()
        benchmark_annual_return = (1 + benchmark_returns.mean()) ** 252 - 1
        metrics["Benchmark Return"] = benchmark_annual_return
        metrics["Alpha"] = annual_returns - benchmark_annual_return
    
    return pd.DataFrame(metrics)

def plot_price_chart(data, benchmark=None):
    """Plot normalized price trends with benchmark comparison."""
    normalized_data = normalize_prices(data)
    fig = px.line(normalized_data, title="📈 Normalized Price Trends (Base=100)")
    
    if benchmark is not None:
        if isinstance(benchmark, pd.DataFrame) and benchmark.shape[1] == 1:
            benchmark_series = benchmark.iloc[:, 0]
        elif isinstance(benchmark, pd.Series):
            benchmark_series = benchmark
        else:
            st.warning("⚠️ Benchmark format not supported for plotting.")
            return
        
        normalized_benchmark = (benchmark_series / benchmark_series.iloc[0]) * 100
        fig.add_scatter(
            x=normalized_benchmark.index,
            y=normalized_benchmark,
            name=benchmark_series.name if benchmark_series.name else "Benchmark",
            line=dict(color="gray", dash="dot")
        )
    
    fig.update_yaxes(title_text="Normalized Price (Base=100)")
    st.plotly_chart(fig, use_container_width=True)

def plot_bar_chart(metrics):
    df = metrics.reset_index().melt(id_vars="index", var_name="Metric", value_name="Value")
    fig = px.bar(
        df,
        x="Metric",
        y="Value",
        color="index",
        barmode="group",
        title="📊 Performance Metrics"
    )
    st.plotly_chart(fig, use_container_width=True)

def style_performance(df):
    def highlight_sharpe(val):
        if pd.isna(val): return ''
        return 'background-color: #006400; color: white;' if val > 0 else ''
    
    def highlight_drawdown(val):
        if pd.isna(val): return ''
        return 'background-color: #ff0000; color: white;' if val < 0 else ''
    
    def clean_nans(val):
        return f"{val:.2%}" if pd.notna(val) else '—'

    styled = df.style.format(clean_nans).applymap(highlight_sharpe, subset=["Sharpe Ratio"]).applymap(highlight_drawdown, subset=["Max Drawdown"])
    return styled
