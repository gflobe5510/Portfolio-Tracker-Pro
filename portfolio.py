
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.optimize import minimize

def load_data(tickers, start_date, end_date, benchmark):
    # Simulated data for development
    dates = pd.date_range(start=start_date, end=end_date)
    data = pd.DataFrame(index=dates)
    for ticker in tickers:
        data[ticker] = np.random.rand(len(dates)) * 100
    bench_data = pd.Series(np.random.rand(len(dates)) * 100, index=dates, name=benchmark)
    return data, bench_data

def calculate_metrics(data):
    returns = data.pct_change().dropna()
    return {
        "mean_returns": returns.mean().to_dict(),
        "volatility": returns.std().to_dict(),
    }

def plot_price_chart(data, benchmark_series):
    fig = go.Figure()
    for col in data.columns:
        fig.add_trace(go.Scatter(x=data.index, y=data[col], mode='lines', name=col))
    fig.add_trace(go.Scatter(x=benchmark_series.index, y=benchmark_series, mode='lines', name=benchmark_series.name))
    fig.update_layout(title="Price Chart", xaxis_title="Date", yaxis_title="Price")
    return fig

def plot_bar_chart(data):
    return go.Figure(data=[go.Bar(x=data.index, y=data.values)])

def plot_pie_chart(weights_dict):
    labels = list(weights_dict.keys())
    values = list(weights_dict.values())
    fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
    return fig

def optimize_portfolio(price_data):
    returns = price_data.pct_change().dropna()
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    num_assets = len(mean_returns)

    def portfolio_volatility(weights):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
    bounds = tuple((0, 1) for _ in range(num_assets))
    initial_guess = num_assets * [1. / num_assets]

    result = minimize(portfolio_volatility, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    return dict(zip(price_data.columns, result.x))
