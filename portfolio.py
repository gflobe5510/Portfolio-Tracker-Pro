import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from datetime import datetime

# Configure logger
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_local_data(tickers, benchmark, start_date, end_date):
    """Load and filter price data from local CSV"""
    try:
        # Validate inputs
        if not tickers:
            raise ValueError("No tickers selected")
        if not benchmark:
            raise ValueError("No benchmark selected")
        
        # Read only necessary columns to optimize memory
        usecols = ["Date"] + list(set(tickers + [benchmark]))
        logger.info(f"Loading columns: {usecols}")
        
        # Read CSV with chunking for large files
        chunks = []
        for chunk in pd.read_csv("price_data.csv", parse_dates=["Date"], usecols=usecols, chunksize=10000):
            # Filter dates within each chunk
            mask = (chunk["Date"] >= pd.to_datetime(start_date)) & (chunk["Date"] <= pd.to_datetime(end_date))
            filtered_chunk = chunk[mask].copy()
            if not filtered_chunk.empty:
                chunks.append(filtered_chunk)
        
        if not chunks:
            raise ValueError("No data available for selected date range")
        
        df = pd.concat(chunks)
        df = df.drop_duplicates(subset=["Date"]).sort_values("Date")
        df.set_index("Date", inplace=True)
        
        # Check for missing data
        missing_tickers = set(tickers) - set(df.columns)
        if missing_tickers:
            st.warning(f"Missing data for: {', '.join(missing_tickers)}")
            tickers = [t for t in tickers if t in df.columns]
        
        if benchmark not in df.columns:
            raise ValueError(f"Benchmark {benchmark} not found in data")
        
        return df[tickers], df[[benchmark]]
    
    except Exception as e:
        logger.error(f"Error in load_local_data: {str(e)}")
        st.error(f"Data loading error: {str(e)}")
        raise

def calculate_metrics(data, weights):
    """Calculate portfolio performance metrics"""
    try:
        if data.empty:
            return pd.DataFrame({
                "Metric": ["Annual Return", "Volatility", "Sharpe Ratio"],
                "Value": ["N/A", "N/A", "N/A"]
            })
        
        # Calculate daily returns
        returns = data.pct_change().dropna()
        
        # Calculate weighted portfolio returns
        weighted_returns = pd.Series(0, index=returns.index)
        for ticker, weight in weights.items():
            if ticker in returns.columns:
                weighted_returns += returns[ticker] * (weight / 100)
        
        if len(weighted_returns) < 5:  # Minimum data points requirement
            return pd.DataFrame({
                "Metric": ["Annual Return", "Volatility", "Sharpe Ratio"],
                "Value": ["Insufficient data", "Insufficient data", "Insufficient data"]
            })
        
        # Calculate metrics
        cumulative_returns = (1 + weighted_returns).cumprod()
        trading_days = len(cumulative_returns)
        
        # Annualized return
        annual_return = cumulative_returns.iloc[-1] ** (252 / trading_days) - 1
        
        # Annualized volatility
        volatility = weighted_returns.std() * np.sqrt(252)
        
        # Sharpe ratio (assuming 0% risk-free rate)
        sharpe_ratio = annual_return / volatility if volatility != 0 else 0
        
        # Max drawdown
        peak = cumulative_returns.cummax()
        drawdown = (cumulative_returns - peak) / peak
        max_drawdown = drawdown.min()
        
        return pd.DataFrame({
            "Metric": [
                "Annual Return", 
                "Annual Volatility", 
                "Sharpe Ratio",
                "Max Drawdown",
                "Trading Days"
            ],
            "Value": [
                f"{annual_return:.2%}",
                f"{volatility:.2%}",
                f"{sharpe_ratio:.2f}",
                f"{max_drawdown:.2%}",
                str(trading_days)
            ]
        })
        
    except Exception as e:
        logger.error(f"Error in calculate_metrics: {str(e)}")
        st.error(f"Metrics calculation error: {str(e)}")
        return pd.DataFrame()

def plot_price_chart(data, benchmark):
    """Plot normalized price performance chart"""
    try:
        if data.empty or benchmark.empty:
            st.warning("No data available to plot")
            return
        
        fig = go.Figure()
        
        # Normalize all data to starting point = 100
        norm_data = (data / data.iloc[0]) * 100
        norm_bench = (benchmark / benchmark.iloc[0]) * 100
        
        # Add portfolio components
        for col in norm_data.columns:
            fig.add_trace(go.Scatter(
                x=norm_data.index,
                y=norm_data[col],
                name=col,
                mode='lines',
                hovertemplate="%{y:.2f}%<extra></extra>",
                line=dict(width=1.5)
            ))
        
        # Add benchmark
        fig.add_trace(go.Scatter(
            x=norm_bench.index,
            y=norm_bench[benchmark.columns[0]],
            name=f"{benchmark.columns[0]} (Benchmark)",
            mode='lines',
            line=dict(dash='dot', width=2, color='black'),
            hovertemplate="%{y:.2f}%<extra></extra>"
        ))
        
        # Add portfolio weighted average if multiple assets
        if len(data.columns) > 1:
            portfolio_avg = norm_data.mean(axis=1)
            fig.add_trace(go.Scatter(
                x=portfolio_avg.index,
                y=portfolio_avg,
                name="Portfolio Average",
                mode='lines',
                line=dict(width=3, color='royalblue'),
                hovertemplate="%{y:.2f}%<extra></extra>"
            ))
        
        fig.update_layout(
            title="Normalized Price Performance (Base = 100)",
            xaxis_title="Date",
            yaxis_title="Normalized Value (%)",
            hovermode="x unified",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=20, r=20, t=60, b=20),
            plot_bgcolor='rgba(255,255,255,0.8)',
            paper_bgcolor='rgba(255,255,255,0.5)',
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        logger.error(f"Error in plot_price_chart: {str(e)}")
        st.error(f"Chart error: {str(e)}")

def plot_pie_chart(weights):
    """Plot portfolio allocation pie chart"""
    try:
        if not weights:
            st.warning("No allocation data to display")
            return
        
        # Prepare data
        labels = list(weights.keys())
        values = list(weights.values())
        
        # Create pie chart
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.4,
            marker_colors=px.colors.qualitative.Plotly,
            textinfo='percent+label',
            insidetextorientation='radial',
            hoverinfo='label+percent+value',
            textfont_size=14
        )])
        
        fig.update_layout(
            title="Portfolio Allocation",
            uniformtext_minsize=12,
            uniformtext_mode='hide',
            margin=dict(t=50, b=20, l=20, r=20),
            legend=dict(
                font=dict(size=12)
            ),
            plot_bgcolor='rgba(255,255,255,0.8)',
            paper_bgcolor='rgba(255,255,255,0.5)',
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        logger.error(f"Error in plot_pie_chart: {str(e)}")
        st.error(f"Pie chart error: {str(e)}")
