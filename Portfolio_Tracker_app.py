import streamlit as st

st.set_page_config(
    page_title="Portfolio Tracker Pro+",
    layout="wide",
    initial_sidebar_state="expanded"
)

import pandas as pd
import numpy as np
import base64
import os
from pathlib import Path
from datetime import date, datetime, timedelta
import plotly.express as px
import plotly.figure_factory as ff
from portfolio import load_data, calculate_metrics, plot_price_chart, plot_bar_chart
from pdf_utils import create_pdf_report
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from itertools import cycle




# ========== DEPENDENCY HANDLING ==========
def check_dependencies():
    """Check and initialize optional dependencies"""
    dependencies = {
        'prophet': False,
        'statsmodels': False
    }
    
    try:
        from prophet import Prophet
        dependencies['prophet'] = True
    except ImportError:
        pass
        
    try:
        from statsmodels.tsa.arima.model import ARIMA
        dependencies['statsmodels'] = True
    except ImportError:
        pass
    
    return dependencies

deps = check_dependencies()

# ========== CONFIGURATION ==========
    page_title="📊 Portfolio Tracker Pro+",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== BACKGROUND IMAGE ==========
def set_background(image_file):
    """Set background image - simplified working version"""
    try:
        with open(image_file, "rb") as f:
            img_data = f.read()
        b64_encoded = base64.b64encode(img_data).decode()
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("data:image/jpeg;base64,{b64_encoded}");
                background-size: cover;
                background-position: center;
                background-attachment: fixed;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    except Exception as e:
        st.warning(f"Background image error: {str(e)}")
        st.markdown("""
        <style>
        .stApp {
            background-color: #f0f2f6;
        }
        </style>
        """, unsafe_allow_html=True)

# Set background (assuming image is in same directory as script)
set_background("hector-j-rivas-1FxMET2U5dU-unsplash.jpg")

# ========== BENCHMARK OPTIONS ==========
BENCHMARK_OPTIONS = {
    "🟦 S&P 500 (SPY)": "SPY",
    "🟩 Nasdaq 100 (QQQ)": "QQQ",
    "🟨 Gold (GLD)": "GLD",
    "🟪 Bitcoin (BTC-USD)": "BTC-USD",
    "⬜ No Benchmark": None
}

# ========== CACHE DATA LOADING ==========
@st.cache_data
def load_ticker_list():
    """Fetch 500+ tickers including stocks, ETFs, and cryptocurrencies"""
    try:
        # Predefined list of 500+ tickers (stocks, ETFs, crypto)
        return [
            # Top 100 Stocks
            "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "BRK-B", "JPM", "JNJ",
            "V", "PG", "UNH", "HD", "MA", "DIS", "BAC", "PYPL", "CMCSA", "XOM",
            "VZ", "ADBE", "CSCO", "PFE", "CVX", "ABT", "NFLX", "PEP", "CRM", "TMO",
            "WMT", "KO", "MRK", "INTC", "PEP", "T", "ABBV", "COST", "AVGO", "QCOM",
            "DHR", "MDT", "MCD", "BMY", "NKE", "LIN", "HON", "AMGN", "SBUX", "LOW",
            "ORCL", "TXN", "UPS", "UNP", "PM", "IBM", "RTX", "CAT", "GS", "AMD",
            "SPGI", "INTU", "ISRG", "PLD", "DE", "NOW", "SCHW", "BLK", "AMT", "ADI",
            "MDLZ", "GE", "LMT", "BKNG", "TJX", "AXP", "SYK", "MMC", "GILD", "CB",
            "ZTS", "CI", "ADP", "TGT", "DUK", "SO", "MO", "MMM", "BDX", "EOG",
            "EL", "CL", "APD", "FIS", "AON", "ITW", "PNC", "BSX", "ICE", "WM",
            
            # ETFs (100+)
            "SPY", "QQQ", "IWM", "DIA", "VTI", "VOO", "VEA", "VWO", "VUG", "VO",
            "VB", "VTV", "VYM", "VXUS", "BND", "BNDX", "VGK", "VPL", "VEU", "VSS",
            "VGT", "VPU", "VIS", "VNQ", "VAW", "VHT", "VOX", "VCR", "VDC", "VDE",
            "VFH", "VHT", "VIG", "VONG", "VONV", "VOT", "VIOG", "VIOV", "VBR", "VBK",
            "VONE", "VTHR", "VONG", "VONV", "VOT", "VIOG", "VIOV", "VBR", "VBK", "VONE",
            "ARKK", "ARKQ", "ARKW", "ARKG", "ARKF", "ARKX", "GLD", "SLV", "USO", "UNG",
            "TAN", "ICLN", "LIT", "REMX", "BOTZ", "ROBO", "AIQ", "QQQJ", "QQJG", "QQQN",
            "XLE", "XLF", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLK", "XLC",
            "XBI", "IBB", "FXI", "EWZ", "EWJ", "EWH", "EWY", "EWT", "EWW", "EWG",
            "EWU", "EWP", "EWQ", "EWL", "EWM", "EWN", "EWK", "EWD", "EWC", "EWA",
            "EEM", "EFA", "IEMG", "IEFA", "IEUR", "IEUS", "IEF", "TLT", "HYG", "LQD",
            
            # Cryptocurrencies (50+)
            "BTC-USD", "ETH-USD", "BNB-USD", "ADA-USD", "DOGE-USD", "XRP-USD", "DOT-USD",
            "SOL-USD", "MATIC-USD", "SHIB-USD", "AVAX-USD", "LTC-USD", "UNI-USD", "LINK-USD",
            "ATOM-USD", "XLM-USD", "ETC-USD", "BCH-USD", "VET-USD", "FIL-USD", "THETA-USD",
            "XMR-USD", "EOS-USD", "AAVE-USD", "XTZ-USD", "ALGO-USD", "MKR-USD", "KSM-USD",
            "DASH-USD", "ZEC-USD", "COMP-USD", "YFI-USD", "SUSHI-USD", "SNX-USD", "RUNE-USD",
            "NEAR-USD", "GRT-USD", "ENJ-USD", "CHZ-USD", "BAT-USD", "MANA-USD", "ANKR-USD",
            "ICX-USD", "SC-USD", "STORJ-USD", "HNT-USD", "OMG-USD", "ZIL-USD", "IOST-USD",
            
            # International Stocks (100+)
            "BABA", "TSM", "ASML", "NVO", "SAP", "RY", "SHOP", "TD", "BNS", "BAM",
            "ENB", "CNQ", "SU", "TRI", "CP", "ATD", "L", "WCN", "CSU", "OTEX",
            "NVS", "HSBC", "UL", "AZN", "GSK", "BP", "SHEL", "RIO", "BHP", "NGLOY",
            "TM", "SONY", "HMC", "NTT", "MFG", "SMFG", "MUFG", "LYG", "SAN", "BBVA",
            "TOT", "TTE", "VIVHY", "PBR", "ITUB", "BBD", "BSBR", "ERJ", "GGB", "SID",
            "YPF", "TEO", "GGAL", "BMA", "EDN", "IRS", "PAM", "TGS", "SUPV", "CRESY",
            "CEPU", "LOMA", "BIOX", "BOLT", "CAAP", "CELU", "CRESY", "CTIO", "DESP", "DX",
            "GGAL", "IRS", "LOMA", "PAM", "SUPV", "TEO", "TGS", "YPF", "BMA", "EDN",
            "CEPU", "CRESY", "GGAL", "IRS", "LOMA", "PAM", "SUPV", "TEO", "TGS", "YPF",
            
            # Small/Mid-Cap Stocks (100+)
            "AFRM", "UPST", "SOFI", "RIVN", "LCID", "FUBO", "PLTR", "HOOD", "COIN", "DASH",
            "RBLX", "SNOW", "DDOG", "ZM", "PTON", "DOCU", "TWLO", "OKTA", "NET", "CRWD",
            "ZS", "MDB", "SPOT", "SQ", "PYPL", "SHOP", "U", "ESTC", "ASAN", "CLOV",
            "WISH", "SDC", "BLNK", "CHPT", "QS", "NKLA", "HYLN", "WKHS", "GOEV", "RIDE",
            "FSR", "LCID", "NIO", "XPEV", "LI", "F", "GM", "STLA", "HMC", "TM",
            "RKLB", "ASTS", "SPCE", "VORB", "RDW", "ASTR", "MNTS", "BKSY", "LILM", "JOBY",
            "DNA", "BEAM", "CRSP", "EDIT", "NTLA", "VERV", "IOVA", "KYMR", "RXRX", "TXG",
            "TWST", "CDNA", "PACB", "NVTA", "GH", "SDGR", "ME", "SGFY", "HIMS", "OSCR",
            "AMWL", "TDOC", "CURI", "LFST", "VWE", "BYND", "TTCF", "STKL", "OATLY", "DNUT",
            "IMGN", "KROS", "RCUS", "ARCT", "BCRX", "KPTI", "SAGE", "SRPT", "BPMC", "CABA"
        ]
    except:
        # Fallback to a smaller list if the main list fails
        return ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "BRK-B", "SPY", "QQQ", "BTC-USD", "ETH-USD"]

# ========== UTILITY FUNCTIONS ==========
def normalize(df):
    return df / df.iloc[0] * 100

def plot_comparison_chart(portfolio_df, benchmark_df):
    if isinstance(benchmark_df, pd.Series):
        benchmark_df = benchmark_df.to_frame()

    normalized_portfolio = normalize(portfolio_df)
    normalized_benchmark = normalize(benchmark_df)

    df_combined = pd.concat([
        normalized_portfolio.rename(columns=lambda c: "Your Portfolio"),
        normalized_benchmark.rename(columns=lambda c: "Benchmark" if not c else c)
    ], axis=1)

    df_melted = df_combined.reset_index().melt(id_vars="Date", var_name="Asset", value_name="Normalized Value")
    fig = px.line(df_melted, x="Date", y="Normalized Value", color="Asset", title="📈 Your Portfolio vs Benchmark")
    fig.update_layout(template="plotly_dark")
    return fig

def plot_correlation_matrix(data):
    corr = data.pct_change().corr()
    fig = ff.create_annotated_heatmap(
        z=corr.values,
        x=list(corr.columns),
        y=list(corr.index),
        annotation_text=corr.round(2).values,
        colorscale='Blues',
        showscale=True
    )
    fig.update_layout(title='📊 Asset Correlation Matrix')
    st.plotly_chart(fig, use_container_width=True)

def forecast_prophet(data, periods=365, weekly_seasonality=False, changepoint_scale=0.05):
    """Enhanced Prophet forecasting with tunable parameters"""
    from prophet import Prophet
    forecasts = {}
    current_date = pd.Timestamp.now().normalize()
    
    for ticker in data.columns:
        df = data[ticker].reset_index()
        df.columns = ['ds', 'y']
        df['ds'] = pd.to_datetime(df['ds'])
        df = df[df['ds'] <= current_date]
        
        model = Prophet(
            daily_seasonality=False,
            yearly_seasonality=True,
            weekly_seasonality=weekly_seasonality,
            changepoint_prior_scale=changepoint_scale
        )
        model.fit(df)
        
        future = model.make_future_dataframe(
            periods=periods,
            freq='B',
            include_history=False
        )
        
        forecast = model.predict(future)
        forecast_df = forecast.set_index('ds')['yhat']
        forecast_df.index = pd.to_datetime(forecast_df.index)
        forecasts[ticker] = forecast_df
    
    return pd.DataFrame(forecasts)

def forecast_arima(data, periods=30):
    from statsmodels.tsa.arima.model import ARIMA
    forecasts = {}


    for ticker in data.columns:
        # Get data and ensure proper datetime index
        ts_data = data[ticker][data[ticker].index <= current_date]
        ts_data = ts_data.asfreq('B').ffill()  # Business day frequency
        
        # Model with error handling
        try:
            model = ARIMA(ts_data, order=(5,1,0))
            model_fit = model.fit()
            
            # Generate forecast with proper dates
            forecast = model_fit.get_forecast(steps=periods)
            forecast_df = forecast.predicted_mean
            
            # Create date index aligned with business days
            last_date = ts_data.index[-1]
            forecast_dates = pd.date_range(
                start=last_date + pd.offsets.BDay(1),
                periods=periods,
                freq='B'
            )
            forecasts[ticker] = forecast_df
            
        except Exception as e:
            st.warning(f"ARIMA failed for {ticker}: {str(e)}")
            forecasts[ticker] = pd.Series(np.nan, index=pd.date_range(
                start=current_date,
                periods=periods,
                freq='B'
            ))
    
    return pd.DataFrame(forecasts)

def plot_pie_chart(tickers, weights=None):
    """Enhanced pie chart showing actual allocations"""
    if weights is None:
        weights = {ticker: 1/len(tickers) for ticker in tickers}
    
    # Create DataFrame for plotting
    plot_data = pd.DataFrame({
        'Ticker': list(weights.keys()),
        'Weight': list(weights.values())
    }).sort_values('Weight', ascending=False)
    
    fig = px.pie(
        plot_data,
        names='Ticker',
        values='Weight',
        title="🧮 Portfolio Allocation",
        color_discrete_sequence=px.colors.qualitative.Plotly,
        hole=0.3
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig, use_container_width=True)

def style_metrics(df):
    """Enhanced metric styling with sorting and highlighting"""
    # Sort by Sharpe ratio (or Alpha if available) descending
    sort_column = 'Alpha' if 'Alpha' in df.columns else 'Sharpe Ratio'
    styled = df.sort_values(sort_column, ascending=False)
    
    # Apply styling
    styled = styled.style.format("{:.2%}")\
        .highlight_max(axis=1, props='background-color: #c8e6c9; color: black')\
        .highlight_min(axis=1, props='background-color: #ffcdd2; color: black')
    
    return styled

# ========== MANUAL PORTFOLIO OPTIMIZATION ==========
def calculate_expected_returns(data):
    """Calculate annualized expected returns"""
    return data.pct_change().mean() * 252

def calculate_covariance_matrix(data):
    """Calculate annualized covariance matrix"""
    return data.pct_change().cov() * 252

def portfolio_performance(weights, expected_returns, cov_matrix):
    """Calculate portfolio performance metrics"""
    returns = np.sum(weights * expected_returns)
    volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe = returns / volatility
    return returns, volatility, sharpe

def optimize_portfolio(data):
    """Manual portfolio optimization without pypfopt"""
    expected_returns = calculate_expected_returns(data)
    cov_matrix = calculate_covariance_matrix(data)
    n_assets = len(data.columns)
    
    # Constraints
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for asset in range(n_assets))
    
    # Initial guess
    init_guess = n_assets * [1./n_assets]
    
    # Optimization
    def negative_sharpe(weights):
        return -portfolio_performance(weights, expected_returns, cov_matrix)[2]
    
    opt_results = minimize(negative_sharpe,
                         init_guess,
                         method='SLSQP',
                         bounds=bounds,
                         constraints=constraints)
    
    optimal_weights = opt_results.x
    perf = portfolio_performance(optimal_weights, expected_returns, cov_matrix)
    
    # Format results
    weights_dict = {data.columns[i]: optimal_weights[i] for i in range(len(optimal_weights))}
    return weights_dict, perf

# ========== MAIN APP ==========
def main():
    # Header
    col1, col2 = st.columns([5, 1])
    with col1:
        st.title("📊 Portfolio Tracker Pro+")
        st.caption("Advanced portfolio analytics with optimization and forecasting")
    with col2:
        if st.button("ℹ️ How to Use"):
            st.session_state.show_help = not st.session_state.get('show_help', False)

    if st.session_state.get('show_help'):
        with st.expander("📚 Portfolio Tracker Guide", expanded=True):
            st.markdown("""
            ### **Getting Started**
            
            **1. Select Date Range**  
            - Choose your analysis period using the date pickers in the sidebar
            
            **2. Pick Assets**  
            - Search and select from 500+ stocks, ETFs, and cryptocurrencies (type to filter)  
            - Maximum of 10 assets can be selected
            
            **3. Set Portfolio Allocation**  
            - Adjust weights using sliders in the sidebar  
            - Weights automatically normalize to 100%
            
            **4. Select Benchmark (Optional)**  
            - Compare against major indices or assets  
            - Choose "No Benchmark" to disable comparison
            
            ### **Advanced Features**
            
            **📊 Correlation Matrix**  
            - Visualize how assets move in relation to each other
            
            **🔮 Forecasting**  
            - Prophet: Best for long-term trend forecasting  
            - ARIMA: Best for short-term predictions
            - Adjust seasonality and flexibility in settings
            
            **⚖️ Portfolio Optimization**  
            - Calculates optimal weights using Modern Portfolio Theory  
            - Visualizes the efficient frontier
            
            ### **Installation Tips**  
            For full functionality:
            ```bash
            pip install prophet statsmodels fpdf yfinance
            ```
            """)
            if st.button("Close Guide"):
                st.session_state.show_help = False

    # Show dependency warnings
    if not deps['prophet']:
        st.warning("Prophet not installed - will use ARIMA for forecasting if available")
    if not deps['statsmodels']:
        st.warning("statsmodels not installed - forecasting features limited")

    # Control Panel
    with st.sidebar:
        st.header("Controls")
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", date(2020, 1, 1))
        with col2:
            end_date = st.date_input("End Date", date.today())
            
        # Date validation
        if start_date >= end_date:
            st.error("End date must be after start date")
            st.stop()
        if (end_date - start_date).days < 30:
            st.warning("Very short date range selected - results may be unreliable")
            
        risk_free = st.slider("Risk-Free Rate (%)", 0.0, 10.0, 2.0, 0.1) / 100
        
        selected_bench = st.selectbox(
            "Benchmark",
            options=list(BENCHMARK_OPTIONS.keys()),
            index=0
        )

    # Ticker Selection
    all_tickers = load_ticker_list()
    selected_tickers = st.multiselect(
        "Select Assets (Min 1, Max 10) - Type to search",
        all_tickers,
        default=["AAPL", "MSFT"],
        max_selections=10
    )

    # Data Processing
    if not selected_tickers:
        st.warning("Please select at least one asset")
        st.stop()

    # Portfolio Allocation Sliders
    st.sidebar.header("Portfolio Allocation")
    weights = {}
    for ticker in selected_tickers:
        weights[ticker] = st.sidebar.slider(
            f"{ticker} Weight (%)",
            0, 100, int(100/len(selected_tickers)),
            key=f"weight_{ticker}"
        )
    
    # Robust weight normalization
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
                BENCHMARK_OPTIONS[selected_bench]
            )
            
            st.warning(f"DEBUG - Attempted to load tickers: {selected_tickers}")
            if data is None or data.empty:
                st.error("Failed to load asset data")
                st.stop()
                
            # Calculate weighted portfolio returns
            weighted_returns = pd.DataFrame()
            for ticker in selected_tickers:
                if ticker in data.columns:  # Check if ticker exists in data
                    weighted_returns[ticker] = data[ticker].pct_change() * (weights[ticker]/100)
            
            if weighted_returns.empty:
                st.error("No valid data available for selected tickers")
                st.stop()
                
            portfolio_value = (1 + weighted_returns.sum(axis=1)).cumprod()
            portfolio_value.name = "Your Portfolio"
                
            status.update(label="✅ Data loaded successfully", state="complete")
            
        except Exception as e:
            st.error(f"Critical error: {str(e)}")
            st.stop()

    # Metrics Calculation
    metrics = calculate_metrics(data, bench_data, risk_free)
    
    if metrics is None or metrics.empty:
        st.error("Could not calculate performance metrics")
        st.stop()
    
    st.subheader("📈 Performance Metrics")
    st.dataframe(
        style_metrics(metrics),
        use_container_width=True
    )

    # Expanded Charts with new features
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Price Trends", 
        "Performance Analysis", 
        "Portfolio Allocation",
        "Portfolio vs Benchmark",
        "Correlation Matrix",
        "Forecasting",
        "Portfolio Optimization"
    ])
    
    with tab1:
        plot_price_chart(data, bench_data)
        
    with tab2:
        plot_bar_chart(metrics)
        
    with tab3:
        plot_pie_chart(selected_tickers, weights)
    
    with tab4:
        if bench_data is not None:
            fig = plot_comparison_chart(portfolio_value.to_frame(), bench_data)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Please select a benchmark to enable comparison")
    
    with tab5:
        st.subheader("📊 Asset Correlation Matrix")
        plot_correlation_matrix(data)
        st.caption("Measures how assets move in relation to each other (-1 to +1 scale)")
    
    with tab6:
        st.subheader("🔮 Price Forecasting")
        st.info('After clicking "Run Forecast" you must return to the Forecasting tab to see the results.')

        if deps['prophet'] or deps['statsmodels']:
            if deps['prophet'] and deps['statsmodels']:
                model_choice = st.radio("Select Model", ["Prophet", "ARIMA"])
            elif deps['prophet']:
                model_choice = "Prophet"
            else:
                model_choice = "ARIMA"
            
            periods = st.number_input("Forecast Periods", 30, 365, 90)
            
            # Advanced forecast settings
            with st.expander("⚙️ Forecast Settings", expanded=False):
                weekly_season = st.checkbox("Include weekly seasonality", False)
                change_scale = st.slider("Model flexibility (changepoint scale)", 0.01, 0.5, 0.05, 0.01)
            
            if st.button("Run Forecast", key="forecast_button"):
                with st.spinner(f"Running {model_choice} forecast..."):
                    try:
                        if model_choice == "Prophet":
                            forecast_df = forecast_prophet(
                                data, 
                                periods=periods,
                                weekly_seasonality=weekly_season,
                                changepoint_scale=change_scale
                            )
                        else:
                            forecast_df = forecast_arima(data, periods)
                        
                        # Get last year of historical data for context
                        last_year_start = pd.Timestamp.now().normalize() - pd.DateOffset(days=365)
                        last_year_data = data[data.index >= last_year_start]
                        
                        # Filter for available columns only
                        available_cols = [col for col in selected_tickers if col in last_year_data.columns]
                        if not available_cols:
                            st.error("No valid tickers available for forecasting")
                            return

                        # Create plot with clear distinction between historical and forecast
                        fig = px.line(
                            last_year_data.reset_index(),
                            x='Date',
                            y=available_cols,
                            title=f"{model_choice} Forecast ({periods} days)",
                            labels={'value': 'Price'},
                            line_dash_sequence=['solid']*len(available_cols)
                        )
                        
                        # Add forecast data with dashed lines using color cycling
                        color_cycle = cycle(px.colors.qualitative.Plotly)
                        for ticker in available_cols:
                            fig.add_scatter(
                                x=forecast_df.index,
                                y=forecast_df[ticker],
                                mode='lines',
                                line=dict(dash='dash', color=next(color_cycle)),
                                name=f"{ticker} (Forecast)",
                                showlegend=True
                            )
                        
                        # Add vertical line at current date
                        last_historical_date = data.index[-1].to_pydatetime()
                        fig.add_vline(
                            x=last_historical_date,
                            line_width=2,
                            line_dash="dash",
                            line_color="red"
                        )
                        
                        # Add shaded forecast region
                        fig.add_vrect(
                            x0=last_historical_date,
                            x1=forecast_df.index[-1].to_pydatetime(),
                            fillcolor="LightSalmon",
                            opacity=0.2,
                            layer="below",
                            line_width=0
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Forecasting failed: {str(e)}")
        else:
            st.warning("Install forecasting packages: pip install prophet statsmodels")
    
    with tab7:
        st.subheader("⚖️ Portfolio Optimization")
        if st.button("Calculate Optimal Portfolio"):
            with st.spinner("Optimizing portfolio..."):
                try:
                    weights, performance = optimize_portfolio(data)
                    
                    st.success("Optimal weights found!")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("### Optimal Weights")
                        for k, v in weights.items():
                            st.write(f"{k}: {v:.2%}")
                    
                    with col2:
                        st.write("### Portfolio Performance")
                        st.write(f"Expected Return: {performance[0]:.2%}")
                        st.write(f"Volatility: {performance[1]:.2%}")
                        st.write(f"Sharpe Ratio: {performance[2]:.2f}")
                    
                    # Plot efficient frontier
                    expected_returns = calculate_expected_returns(data)
                    cov_matrix = calculate_covariance_matrix(data)
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    # Generate random portfolios
                    n_portfolios = 10000
                    results = np.zeros((3, n_portfolios))
                    
                    for i in range(n_portfolios):
                        weights = np.random.random(len(data.columns))
                        weights /= np.sum(weights)
                        ret, vol, sharpe = portfolio_performance(weights, expected_returns, cov_matrix)
                        results[0,i] = vol
                        results[1,i] = ret
                        results[2,i] = sharpe
                    
                    # Plot random portfolios
                    ax.scatter(results[0,:], results[1,:], c=results[2,:], cmap='viridis_r', alpha=0.3)
                    
                    # Plot optimal portfolio
                    ax.scatter(performance[1], performance[0], marker="*", color="r", s=300, label="Optimal")
                    
                    ax.set_title("Efficient Frontier")
                    ax.set_xlabel("Volatility")
                    ax.set_ylabel("Return")
                    ax.legend()
                    st.pyplot(fig)
                    
                except Exception as e:
                    st.error(f"Optimization failed: {str(e)}")

    # PDF Report
    st.markdown("---")
    with st.expander("📄 Export Options"):
        if metrics is None or metrics.empty:
            st.error("Metrics unavailable. Cannot generate report.")
        elif st.button("Generate PDF Report"):
            with st.spinner("Compiling report..."):
                try:
                    pdf_path = create_pdf_report(
                        metrics,
                        start_date=str(start_date),
                        end_date=str(end_date),
                        title=f"Portfolio Analysis: {', '.join(selected_tickers)}"
                    )
                    with open(pdf_path, "rb") as f:
                        st.download_button(
                            "⬇️ Download Full Report",
                            data=f,
                            file_name="portfolio_analysis.pdf",
                            mime="application/pdf"
                        )
                except Exception as e:
                    st.error(f"Report generation failed: {str(e)}")

if __name__ == "__main__":
    main()
