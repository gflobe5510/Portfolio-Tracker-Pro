
Copy
# IMPORTANT: This must be the VERY FIRST Streamlit command
import streamlit as st
st.set_page_config(
    page_title="📊 Portfolio Tracker Pro+",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Now import all other libraries
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_exponential
from scipy.optimize import minimize
from itertools import cycle

@st.cache_data(ttl=3600)
def load_data(tickers, start, end, benchmark=None):
    """Enhanced data loading function with better error handling and yfinance compatibility"""
    try:
        # Convert single ticker to list for consistent processing
        ticker_list = [tickers] if isinstance(tickers, str) else list(tickers)
        
        @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
        def download_assets():
            """Retryable download function with improved yfinance parameters"""
            data = yf.download(
                ticker_list,
                start=start,
                end=end,
                group_by='ticker',
                progress=False,
                auto_adjust=True,
                threads=True,
                timeout=30  # Increased timeout
            , threads=False, progress=False)
            
            # Handle single ticker case
            if len(ticker_list) == 1:
                if 'Close' in data.columns:
                    return data[['Close']].rename(columns={'Close': ticker_list[0]})
                else:
                    raise ValueError(f"No 'Close' data for {ticker_list[0]}")
            
            # Handle multiple tickers
            close_data = pd.DataFrame()
            for ticker in ticker_list:
                if ticker in data.columns.get_level_values(0):
                    if 'Close' in data[ticker]:
                        close_data[ticker] = data[ticker]['Close']
                    else:
                        st.warning(f"No 'Close' data available for {ticker}")
            
            return close_data

        # Download main assets
        close_data = download_assets()
        
        # Validate we got some data
        if close_data is None or close_data.empty:
            raise ValueError("No data returned for selected tickers")

        # Download benchmark if specified
        benchmark_data = None
        if benchmark:
            try:
                @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
                def download_bench():
                    bench_df = yf.download(
                        benchmark,
                        start=start,
                        end=end,
                        progress=False,
                        auto_adjust=True,
                        threads=True,
                        timeout=30
                    , threads=False, progress=False)
                    if not bench_df.empty and 'Close' in bench_df.columns:
                        return bench_df['Close'].copy()
                    return None
                
                benchmark_data = download_bench()
                if benchmark_data is not None:
                    benchmark_data.name = f"Benchmark ({benchmark})"
                else:
                    st.warning(f"Could not load benchmark data for {benchmark}")
            
            except Exception as e:
                st.warning(f"Benchmark download warning: {str(e)}")
                benchmark_data = None
        
        # Return data with proper cleaning
        return close_data.ffill().dropna(), benchmark_data.ffill().dropna() if benchmark_data is not None else None
    
    except Exception as e:
        st.error(f"Data loading failed: {str(e)}")
        return None, None

def normalize_prices(data):
    """Normalize prices to base=100 for consistent scaling."""
    if data is None or data.empty:
        return None
    return (data / data.iloc[0]) * 100

def calculate_metrics(data, benchmark=None, risk_free_rate=0.02):
    """Calculate performance metrics with enhanced robustness"""
    if data is None or data.empty:
        return None
        
    try:
        returns = data.pct_change().dropna()
        if returns.empty:
            return None
            
        metrics = {}
        
        # Annualized metrics
        annual_returns = (1 + returns.mean()) ** 252 - 1
        metrics["Annualized Return"] = annual_returns
        
        annual_volatility = returns.std() * np.sqrt(252)
        metrics["Annualized Volatility"] = annual_volatility
        
        sharpe_ratio = (annual_returns - risk_free_rate) / annual_volatility
        metrics["Sharpe Ratio"] = sharpe_ratio
        
        # Drawdown calculation
        cumulative_returns = (1 + returns).cumprod()
        peak = cumulative_returns.expanding(min_periods=1).max()
        drawdown = (cumulative_returns / peak) - 1
        metrics["Max Drawdown"] = drawdown.min()
        
        # Benchmark comparison if available
        if benchmark is not None:
            benchmark_returns = benchmark.pct_change().dropna()
            if not benchmark_returns.empty:
                benchmark_annual_return = (1 + benchmark_returns.mean()) ** 252 - 1
                metrics["Benchmark Return"] = benchmark_annual_return
                metrics["Alpha"] = annual_returns - benchmark_annual_return
        
        return pd.DataFrame(metrics)
    
    except Exception as e:
        st.error(f"Metrics calculation failed: {str(e)}")
        return None

def plot_price_chart(data, benchmark=None):
    """Plot normalized price trends with benchmark comparison."""
    if data is None or data.empty:
        st.warning("No data available for price chart")
        return
        
    normalized_data = normalize_prices(data)
    if normalized_data is None:
        return
        
    fig = px.line(normalized_data, title="📈 Normalized Price Trends (Base=100)")
    
    if benchmark is not None:
        try:
            if isinstance(benchmark, pd.DataFrame) and benchmark.shape[1] == 1:
                benchmark_series = benchmark.iloc[:, 0]
            elif isinstance(benchmark, pd.Series):
                benchmark_series = benchmark
            else:
                st.warning("Benchmark format not supported for plotting.")
                return
            
            normalized_benchmark = normalize_prices(benchmark_series)
            if normalized_benchmark is not None:
                fig.add_scatter(
                    x=normalized_benchmark.index,
                    y=normalized_benchmark,
                    name=benchmark_series.name if benchmark_series.name else "Benchmark",
                    line=dict(color="gray", dash="dot")
                )
        except Exception as e:
            st.warning(f"Could not plot benchmark: {str(e)}")
    
    fig.update_layout(
        yaxis_title="Normalized Price (Base=100)",
        xaxis_title="Date",
        legend_title="Assets"
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_bar_chart(metrics):
    """Plot performance metrics as bar chart with error handling"""
    if metrics is None or metrics.empty:
        st.warning("No metrics available for bar chart")
        return
        
    try:
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
    except Exception as e:
        st.error(f"Could not create bar chart: {str(e)}")

def style_performance(df):
    """Style performance metrics table with conditional formatting"""
    if df is None or df.empty:
        return None
        
    def highlight_sharpe(val):
        if pd.isna(val): return ''
        return 'background-color: #006400; color: white;' if val > 0 else ''
    
    def highlight_drawdown(val):
        if pd.isna(val): return ''
        return 'background-color: #ff0000; color: white;' if val < 0 else ''
    
    def clean_nans(val):
        return f"{val:.2%}" if pd.notna(val) else '—'

    try:
        styled = df.style.format(clean_nans)\
                   .applymap(highlight_sharpe, subset=["Sharpe Ratio"])\
                   .applymap(highlight_drawdown, subset=["Max Drawdown"])
        return styled
    except Exception as e:
        st.error(f"Could not style metrics: {str(e)}")
        return df.style.format(clean_nans)
