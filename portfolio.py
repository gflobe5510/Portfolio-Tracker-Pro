import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import matplotlib.pyplot as plt

def load_data(tickers, start, end, benchmark_symbol):
    """
    Load historical data for selected tickers and benchmark.

    Parameters:
        tickers (list): List of stock tickers.
        start (str): Start date in 'YYYY-MM-DD'.
        end (str): End date in 'YYYY-MM-DD'.
        benchmark_symbol (str): Benchmark ticker symbol (e.g., '^GSPC').

    Returns:
        tuple: (data_df, benchmark_df)
    """
    all_tickers = tickers + [benchmark_symbol]
    df = yf.download(all_tickers, start=start, end=end, progress=False, group_by='ticker', auto_adjust=True)

    if df.empty:
        return None, None

    if isinstance(df.columns, pd.MultiIndex):
        data = pd.concat([df[ticker]['Close'] for ticker in tickers], axis=1)
        data.columns = tickers

        benchmark = df[benchmark_symbol]['Close'].rename(benchmark_symbol)
    else:
        data = df['Close'].to_frame(name=tickers[0])
        benchmark = df['Close'].to_frame(name=benchmark_symbol)

    return data, benchmark

def calculate_metrics(data, weights):
    """
    Calculate portfolio returns, volatility, and Sharpe ratio.

    Parameters:
        data (DataFrame): Asset price data.
        weights (dict): Ticker weights.

    Returns:
        dict: Portfolio metrics.
    """
    df = data.copy()
    returns = df.pct_change().dropna()
    weighted_returns = returns.mul([weights[t] / 100 for t in df.columns], axis=1)
    portfolio_returns = weighted_returns.sum(axis=1)

    cumulative_return = (1 + portfolio_returns).prod() - 1
    volatility = portfolio_returns.std() * np.sqrt(252)
    sharpe_ratio = (portfolio_returns.mean() / portfolio_returns.std()) * np.sqrt(252)

    return {
        "Cumulative Return": cumulative_return,
        "Volatility": volatility,
        "Sharpe Ratio": sharpe_ratio
    }

def plot_price_chart(data, benchmark=None):
    """
    Plot the normalized price chart of assets and benchmark.

    Parameters:
        data (DataFrame): Price data for assets.
        benchmark (Series): Benchmark price data.
    """
    fig = go.Figure()
    for ticker in data.columns:
        fig.add_trace(go.Scatter(x=data.index, y=data[ticker] / data[ticker].iloc[0], mode='lines', name=ticker))

    if benchmark is not None:
        fig.add_trace(go.Scatter(x=benchmark.index, y=benchmark / benchmark.iloc[0],
                                 mode='lines', name=benchmark.name, line=dict(dash='dot')))

    fig.update_layout(title="Normalized Price Performance", xaxis_title="Date", yaxis_title="Normalized Price")
    return fig

def plot_bar_chart(metrics):
    """
    Plot a bar chart for key portfolio metrics.

    Parameters:
        metrics (dict): Dictionary with metric names and values.
    """
    names = list(metrics.keys())
    values = list(metrics.values())

    fig, ax = plt.subplots()
    ax.barh(names, values, color='skyblue')
    ax.set_title('Portfolio Metrics')
    ax.set_xlabel('Value')
    return fig

def optimize_portfolio(price_data, risk_free_rate=0.01):
    import numpy as np
    from scipy.optimize import minimize

    daily_returns = price_data.pct_change().dropna()
    mean_returns = daily_returns.mean()
    cov_matrix = daily_returns.cov()

    num_assets = len(mean_returns)

    def portfolio_performance(weights):
        returns = np.sum(mean_returns * weights) * 252
        volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
        sharpe_ratio = (returns - risk_free_rate) / volatility
        return -sharpe_ratio

    constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
    bounds = tuple((0, 1) for _ in range(num_assets))
    initial_weights = num_assets * [1. / num_assets,]

    result = minimize(portfolio_performance, initial_weights, method='SLSQP',
                      bounds=bounds, constraints=[constraints])
    return result.x