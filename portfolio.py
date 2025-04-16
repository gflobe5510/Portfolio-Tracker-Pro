
import pandas as pd

PRICE_DATA_CSV = "replacement_data/price_data.csv"

def load_data(tickers, start_date, end_date, benchmark=None):
    df = pd.read_csv(PRICE_DATA_CSV, parse_dates=["Date"])
    df = df[df["Ticker"].isin(tickers)]
    df = df[(df["Date"] >= pd.to_datetime(start_date)) & (df["Date"] <= pd.to_datetime(end_date))]
    df_pivot = df.pivot(index="Date", columns="Ticker", values="Adj Close").dropna(how="all")

    bench_df = None
    if benchmark and benchmark in df["Ticker"].unique():
        bench_df = df[df["Ticker"] == benchmark].set_index("Date")["Adj Close"]

    return df_pivot, bench_df

def calculate_metrics(prices, weights):
    normed = prices / prices.iloc[0]
    weighted = (normed * list(weights.values())).sum(axis=1)
    returns = weighted.pct_change().dropna()
    cumulative = (1 + returns).cumprod()
    return cumulative, returns

def plot_price_chart(prices, benchmark=None):
    import plotly.graph_objects as go
    fig = go.Figure()
    for col in prices.columns:
        fig.add_scatter(x=prices.index, y=prices[col], name=col)
    if benchmark is not None:
        fig.add_scatter(x=benchmark.index, y=benchmark, name=benchmark.name, line=dict(color="gray", dash="dot"))
    fig.update_layout(title="Adjusted Close Price Over Time", xaxis_title="Date", yaxis_title="Price")
    return fig

def plot_bar_chart(returns):
    import plotly.express as px
    bar_df = returns.tail(1).T
    bar_df.columns = ["Return"]
    fig = px.bar(bar_df, y="Return", title="Final Portfolio Value Contribution")
    return fig
