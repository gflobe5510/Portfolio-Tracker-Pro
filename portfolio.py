import pandas as pd
import plotly.graph_objects as go

def load_local_data(tickers, benchmark, start_date, end_date):
    df = pd.read_csv("replacement_data/price_data.csv", parse_dates=["Date"])
    df = df[(df["Date"] >= pd.to_datetime(start_date)) & (df["Date"] <= pd.to_datetime(end_date))]
    df.set_index("Date", inplace=True)

    data = df[tickers]
    bench_data = df[[benchmark]]
    return data, bench_data

def calculate_metrics(data, weights):
    df = data.pct_change().dropna()
    weighted_returns = sum(df[ticker] * (weights[ticker] / 100) for ticker in weights)
    cumulative = (1 + weighted_returns).cumprod()
    annual_return = cumulative[-1] ** (1 / (len(cumulative) / 252)) - 1
    volatility = weighted_returns.std() * (252 ** 0.5)
    sharpe_ratio = annual_return / volatility if volatility else 0
    return pd.DataFrame({
        "Annual Return": [f"{annual_return:.2%}"],
        "Volatility": [f"{volatility:.2%}"],
        "Sharpe Ratio": [f"{sharpe_ratio:.2f}"]
    })

def plot_price_chart(data, benchmark):
    fig = go.Figure()
    for col in data.columns:
        fig.add_trace(go.Scatter(x=data.index, y=data[col], name=col))
    fig.add_trace(go.Scatter(x=benchmark.index, y=benchmark[benchmark.columns[0]], name=benchmark.columns[0], line=dict(dash="dot")))
    fig.update_layout(title="Normalized Prices", xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig, use_container_width=True)

def plot_pie_chart(weights):
    labels = list(weights.keys())
    values = list(weights.values())
    fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
    fig.update_layout(title="Portfolio Allocation")
    st.plotly_chart(fig, use_container_width=True)
