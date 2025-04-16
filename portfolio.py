import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional, Union
import numpy as np
import plotly.express as px

def load_data(tickers: Union[List[str], str], start: str, end: str, benchmark: str = None) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """Load price data for tickers and optional benchmark"""
    try:
        # Load main ticker data
        data = yf.download(tickers, start=start, end=end, progress=False)["Adj Close"]
        if isinstance(data, pd.Series):
            data = data.to_frame()
        data = data.dropna(how='all', axis=1)
        
        # Load benchmark data if specified
        bench_data = None
        if benchmark:
            bench_data = yf.download(benchmark, start=start, end=end, progress=False)["Adj Close"]
            
        return data, bench_data
    except Exception as e:
        raise RuntimeError(f"Data loading failed: {str(e)}")

def calculate_metrics(data: pd.DataFrame, bench_data: pd.Series = None, risk_free: float = 0.02) -> pd.DataFrame:
    """Calculate portfolio performance metrics"""
    returns = data.pct_change().dropna()
    metrics = pd.DataFrame(index=data.columns)
    metrics["Annualized Return"] = returns.mean() * 252
    metrics["Annualized Volatility"] = returns.std() * np.sqrt(252)
    metrics["Sharpe Ratio"] = (metrics["Annualized Return"] - risk_free) / metrics["Annualized Volatility"]
    
    # Calculate max drawdown
    cumulative = (1 + returns).cumprod()
    peak = cumulative.expanding(min_periods=1).max()
    drawdown = (cumulative - peak) / peak
    metrics["Max Drawdown"] = drawdown.min()
    
    if bench_data is not None:
        bench_returns = bench_data.pct_change().dropna()
        metrics["Alpha"] = metrics["Annualized Return"] - bench_returns.mean() * 252
        # Calculate beta for each asset
        betas = []
        for col in returns.columns:
            cov = returns[col].cov(bench_returns)
            var = bench_returns.var()
            betas.append(cov / var if var != 0 else np.nan)
        metrics["Beta"] = betas
    
    return metrics

def plot_price_chart(data: pd.DataFrame, benchmark: pd.Series = None):
    """Plot price history chart"""
    fig, ax = plt.subplots(figsize=(10, 4))
    data.plot(ax=ax)
    if benchmark is not None:
        benchmark.plot(ax=ax, color='gray', linestyle='--', label='Benchmark')
    plt.title("Asset Price History")
    plt.ylabel("Adjusted Close Price")
    plt.xlabel("Date")
    plt.legend()
    return fig

def plot_bar_chart(metrics: pd.DataFrame):
    """Plot performance metrics as bar chart"""
    fig, ax = plt.subplots(figsize=(10, 4))
    metrics[["Annualized Return", "Annualized Volatility"]].plot(kind="bar", ax=ax)
    plt.title("Return & Volatility")
    plt.ylabel("Annualized")
    return fig

def portfolio_performance(weights, mean_returns, cov_matrix):
    """Compute return, volatility, and Sharpe ratio for given weights"""
    ret = np.dot(weights, mean_returns) * 252
    vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    sharpe = ret / vol
    return ret, vol, sharpe

def optimize_portfolio(data: pd.DataFrame, max_allocation: Optional[float] = None) -> Tuple[Dict[str, float], Tuple[float, float, float]]:
    """Optimize portfolio weights for maximum Sharpe ratio"""
    from scipy.optimize import minimize
    returns = data.pct_change().dropna()
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    num_assets = len(data.columns)

    def negative_sharpe(weights):
        ret, vol, _ = portfolio_performance(weights, mean_returns, cov_matrix)
        return -ret/vol  # Negative Sharpe for minimization

    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    bounds = [(0, max_allocation if max_allocation else 1.0)] * num_assets
    result = minimize(negative_sharpe, [1 / num_assets] * num_assets, 
                     bounds=bounds, constraints=constraints)

    if not result.success:
        raise RuntimeError('Optimization failed')

    opt_weights = dict(zip(data.columns, result.x))
    performance = portfolio_performance(result.x, mean_returns, cov_matrix)
    return opt_weights, performance

def monte_carlo_simulation(mu: float, sigma: float, days: int, simulations: int, start_value: float) -> Tuple[np.ndarray, np.ndarray]:
    """Run Monte Carlo simulation for portfolio projection"""
    dt = 1/days
    results = np.zeros((days+1, simulations))
    results[0] = start_value
    
    for t in range(1, days+1):
        shock = np.random.normal(mu*dt, sigma*np.sqrt(dt), simulations)
        results[t] = results[t-1] * (1 + shock)
        
    return results[-1], results

def get_monte_carlo_stats(ending_values: np.ndarray, start_value: float) -> Dict[str, float]:
    """Calculate statistics from Monte Carlo simulation results"""
    return {
        "mean": float(np.mean(ending_values)),
        "median": float(np.median(ending_values)),
        "min": float(np.min(ending_values)),
        "max": float(np.max(ending_values)),
        "pct_change": float((np.mean(ending_values) - start_value) / start_value),
        "std_dev": float(np.std(ending_values))
    }

def plot_monte_carlo_histogram(ending_values: np.ndarray, start_value: float):
    """Create histogram visualization of Monte Carlo results"""
    fig = px.histogram(ending_values, nbins=50)
    fig.update_layout(
        title=f"Monte Carlo Simulation Results (Start Value: {start_value})",
        xaxis_title="Portfolio Value",
        yaxis_title="Frequency"
    )
    fig.add_vline(x=start_value, line_dash="dash", line_color="red")
    fig.add_vline(x=np.mean(ending_values), line_dash="dash", line_color="green")
    return fig