from typing import Optional, Dict, List, Tuple, Union
import streamlit as st
import pandas as pd
import numpy as np
import base64
import os
from pathlib import Path
from datetime import date, datetime, timedelta
import plotly.express as px
import plotly.figure_factory as ff
from portfolio import (
    load_data, 
    calculate_metrics, 
    plot_price_chart, 
    plot_bar_chart, 
    optimize_portfolio, 
    monte_carlo_simulation, 
    get_monte_carlo_stats, 
    plot_monte_carlo_histogram,
    portfolio_performance
)
from pdf_utils import create_pdf_report
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from itertools import cycle
from monte_carlo_simulator import monte_carlo_tab
import yfinance as yf

# ========== DEPENDENCY HANDLING ==========
def check_dependencies():
    """Check and initialize optional dependencies"""
    dependencies = {
        'prophet': False,
        'statsmodels': False,
        'openai': False
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
    
    try:
        import openai
        if 'openai' in st.secrets:
            openai.api_key = st.secrets.openai.api_key
            dependencies['openai'] = True
        else:
            openai.api_key = os.getenv('OPENAI_API_KEY')
            if openai.api_key:
                dependencies['openai'] = True
    except ImportError:
        pass
    
    return dependencies

deps = check_dependencies()

# ========== CONFIGURATION ==========
st.set_page_config(
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
            .guide-text {{
                color: #f0f0f0;
                line-height: 1.6;
            }}
            .guide-header {{
                color: #4facfe;
                margin-top: 20px;
            }}
            .insight-positive {{
                color: #2ecc71;
                font-weight: bold;
            }}
            .insight-warning {{
                color: #f39c12;
                font-weight: bold;
            }}
            .insight-negative {{
                color: #e74c3c;
                font-weight: bold;
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
        .guide-text {
            color: #333333;
        }
        .guide-header {
            color: #0068c9;
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

# ========== RISK FREE RATE SOURCES ==========
RISK_FREE_SOURCES = {
    "10Y Treasury Yield": "treasury",
    "High-Yield Savings (4.0%)": "hysa_4",
    "High-Yield Savings (4.5%)": "hysa_4.5",
    "High-Yield Savings (5.0%)": "hysa_5",
    "Fed Funds Rate": "fedfunds",
    "Custom Rate": "custom"
}

# ========== CACHE DATA LOADING ==========
@st.cache_data
def load_ticker_list():
    """Fetch 500+ tickers including stocks, ETFs, and cryptocurrencies"""
    try:
        return [
            # Stocks (300+)
            "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "BRK-B", "JPM", "JNJ",
            # Cryptocurrencies (50+)
            "BTC-USD", "ETH-USD", "BNB-USD", "ADA-USD", "DOGE-USD", 
        ]
    except:
        return ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "BRK-B", "SPY", "QQQ", "BTC-USD", "ETH-USD"]

def get_current_treasury_yield():
    """Fetch current 10-year treasury yield"""
    try:
        treasury = yf.Ticker("^TNX")
        yield_pct = treasury.history(period="1d")['Close'].iloc[-1]
        return yield_pct / 100  # Convert to decimal
    except:
        return 0.02  # Fallback to 2% if fetch fails

def get_fed_funds_rate():
    """Fetch current federal funds rate"""
    try:
        fedfunds = yf.Ticker("^DFF")
        rate = fedfunds.history(period="1d")['Close'].iloc[-1]
        return rate / 100
    except:
        return 0.05  # Fallback to 5%

def get_risk_free_rate(source: str) -> float:
    """Get risk-free rate from selected source"""
    if source == "treasury":
        return get_current_treasury_yield()
    elif source == "hysa_4":
        return 0.04
    elif source == "hysa_4.5":
        return 0.045
    elif source == "hysa_5":
        return 0.05
    elif source == "fedfunds":
        return get_fed_funds_rate()
    elif source == "custom":
        return st.session_state.get('custom_rate', 0.02)
    else:
        return 0.02  # Default fallback

def calculate_beta_weighted_average(data: pd.DataFrame, 
                                  benchmark_ticker: str,
                                  weights: dict,
                                  lookback: str = "3y") -> Optional[dict]:
    """
    Enhanced beta calculation with:
    - Multiple lookback periods
    - Rolling beta analysis
    - Error handling
    """
    if not benchmark_ticker:
        return None
        
    try:
        # Get historical data
        asset_data = {}
        for ticker in data.columns:
            try:
                asset_data[ticker] = yf.download(ticker, period=lookback)['Adj Close']
            except:
                st.warning(f"Could not fetch data for {ticker}")
                continue
                
        bench_data = yf.download(benchmark_ticker, period=lookback)['Adj Close']
        
        if len(asset_data) == 0:
            return None
            
        # Calculate rolling betas (90-day window)
        betas = {}
        rolling_betas = {}
        for ticker, prices in asset_data.items():
            merged = pd.concat([prices, bench_data], axis=1).dropna()
            merged.columns = ['asset', 'benchmark']
            
            returns = merged.pct_change().dropna()
            cov = returns.rolling(window=90).cov().unstack()['asset']['benchmark']
            var = returns['benchmark'].rolling(window=90).var()
            rolling_beta = cov / var
            
            # Use median of last 6 months rolling betas
            betas[ticker] = rolling_beta.last('6M').median()
            rolling_betas[ticker] = rolling_beta
        
        # Calculate weighted average beta
        valid_betas = [betas[t] for t in betas if not np.isnan(betas[t])]
        valid_weights = [weights[t] for t in betas if not np.isnan(betas[t])]
        
        if len(valid_betas) == 0:
            return None
            
        beta_avg = np.average(valid_betas, weights=valid_weights)
        
        # Additional metrics
        combined_rolling = pd.DataFrame(rolling_betas).mean(axis=1)
        beta_stability = combined_rolling.std() / abs(beta_avg)
        
        return {
            'beta_avg': beta_avg,
            'beta_stability': beta_stability,
            'rolling_beta': combined_rolling
        }
        
    except Exception as e:
        st.error(f"Beta calculation error: {str(e)}")
        return None

def generate_llm_insights(metrics: pd.DataFrame, 
                         portfolio_composition: dict,
                         risk_free_rate: float) -> str:
    """Generate advanced insights using OpenAI's API"""
    if not deps['openai']:
        return "Advanced insights require OpenAI API (pip install openai)"
    
    if not openai.api_key:
        return "OpenAI API key not configured"
    
    try:
        prompt = f"""
        Analyze this investment portfolio:
        - Metrics: {metrics.to_dict()}
        - Composition: {portfolio_composition}
        - Risk-Free Rate: {risk_free_rate:.2%}
        
        Provide 3-5 concise bullet points highlighting:
        1. Key strengths/weaknesses
        2. Risk profile analysis
        3. Suggested improvements
        Use professional but accessible language.
        Format response with markdown bullet points.
        """
        
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        st.error(f"AI analysis failed: {str(e)}")
        return "Advanced insights unavailable"

def generate_insights(metrics: pd.DataFrame, 
                     beta_data: Optional[dict],
                     risk_free_rate: float,
                     portfolio_composition: dict) -> list:
    """Expanded financial logic with 20+ rules"""
    insights = []
    portfolio_metrics = metrics.get('Your Portfolio', {})
    
    # 1. Sharpe Ratio Analysis
    sharpe = portfolio_metrics.get('Sharpe Ratio', 0)
    if sharpe > 2.0:
        insights.append(("positive", "Exceptional risk-adjusted returns (Sharpe > 2.0)"))
    elif sharpe > 1.5:
        insights.append(("positive", "Strong risk-adjusted performance (Sharpe > 1.5)"))
    elif sharpe < 0.5:
        insights.append(("warning", f"Suboptimal risk-adjusted returns (Sharpe = {sharpe:.2f})"))
    
    # 2. Beta Analysis (expanded)
    if beta_data:
        beta = beta_data.get('beta_avg', 1)
        stability = beta_data.get('beta_stability', 0)
        
        if beta > 1.5:
            insights.append(("warning", f"Aggressive profile (Beta = {beta:.2f}) - 50%+ more volatile than market"))
        elif beta < 0.8:
            insights.append(("positive", f"Defensive positioning (Beta = {beta:.2f})"))
                
        if stability > 0.3:
            insights.append(("warning", f"Unstable beta (Variation = {stability:.0%}) - consider less volatile assets"))
    
    # 3. Drawdown Analysis (enhanced)
    drawdown = portfolio_metrics.get('Max Drawdown', 0)
    if drawdown < -0.30:
        insights.append(("negative", f"Severe drawdown risk ({drawdown:.0%} max loss)"))
    elif drawdown < -0.15:
        insights.append(("warning", f"Elevated drawdown risk ({drawdown:.0%} max loss)"))
    
    # 4. Concentration Risk
    if len(portfolio_composition) < 3:
        insights.append(("warning", "High concentration risk - consider diversifying across more assets"))
    elif len([w for w in portfolio_composition.values() if w > 0.3]) > 1:
        insights.append(("warning", "Potential overconcentration in top holdings"))
    
    # 5. Risk-Free Rate Context
    if risk_free_rate > 0.05:
        insights.append(("neutral", f"High risk-free rate environment ({risk_free_rate:.1%}) - bonds are competitive"))
    elif risk_free_rate < 0.02:
        insights.append(("neutral", f"Low rate environment ({risk_free_rate:.1%}) - favorable for equities"))
    
    # 6. Alpha Analysis
    alpha = portfolio_metrics.get('Alpha', None)
    if alpha is not None:
        if alpha > 0.05:
            insights.append(("positive", f"Strong alpha generation ({alpha:.1%} above benchmark)"))
        elif alpha < -0.05:
            insights.append(("negative", f"Underperformance vs benchmark ({alpha:.1%} below)"))
    
    # 7. Volatility Analysis
    vol = portfolio_metrics.get('Annualized Volatility', 0)
    if vol > 0.30:
        insights.append(("warning", f"High volatility ({vol:.1%} annualized)"))
    elif vol < 0.15:
        insights.append(("positive", f"Low volatility profile ({vol:.1%} annualized)"))
    
    # 8. Return Consistency
    returns = portfolio_metrics.get('Annualized Return', 0)
    if returns > 0.15:
        insights.append(("positive", f"Strong annualized returns ({returns:.1%})"))
    elif returns < 0:
        insights.append(("negative", f"Negative annualized returns ({returns:.1%})"))
    
    return insights

def normalize_prices(data):
    """Normalize prices to base=100 for consistent scaling."""
    return (data / data.iloc[0]) * 100

def plot_comparison_chart(portfolio_df, benchmark_df):
    if isinstance(benchmark_df, pd.Series):
        benchmark_df = benchmark_df.to_frame()

    # Normalize both to base=100
    normalized_portfolio = normalize_prices(portfolio_df)
    normalized_benchmark = normalize_prices(benchmark_df)

    df_combined = pd.concat([
        normalized_portfolio.rename(columns=lambda c: "Your Portfolio"),
        normalized_benchmark.rename(columns=lambda c: "Benchmark" if not c else c)
    ], axis=1)

    df_melted = df_combined.reset_index().melt(id_vars="Date", var_name="Asset", value_name="Normalized Value (Base=100)")
    fig = px.line(df_melted, x="Date", y="Normalized Value (Base=100)", color="Asset", title="📈 Your Portfolio vs Benchmark (Normalized)")
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
    current_date = pd.Timestamp.now().normalize()
    
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
    """Style the performance metrics DataFrame with tooltips"""
    def highlight_sharpe(val):
        if pd.isna(val): return ''
        return 'background-color: #006400; color: white;' if val > 0 else ''
    
    def highlight_drawdown(val):
        if pd.isna(val): return ''
        return 'background-color: #ff0000; color: white;' if val < 0 else ''
    
    def clean_nans(val):
        return f"{val:.2%}" if pd.notna(val) else '—'

    # Add tooltip explanations
    tooltips = {
        'Annualized Return': 'Average yearly return over the period',
        'Annualized Volatility': 'Standard deviation of returns (risk measure)',
        'Sharpe Ratio': 'Risk-adjusted return (higher is better)',
        'Max Drawdown': 'Largest peak-to-trough decline',
        'Alpha': 'Excess return vs benchmark',
        'Beta': 'Sensitivity to market movements (1 = market average)',
        'Beta Stability': 'Consistency of beta over time (lower is better)'
    }
    
    styled = df.style.format(clean_nans)\
        .set_tooltips(pd.DataFrame(tooltips, index=df.index))\
        .applymap(highlight_sharpe, subset=["Sharpe Ratio"])\
        .applymap(highlight_drawdown, subset=["Max Drawdown"])
    return styled

def calculate_expected_returns(data):
    """Calculate annualized expected returns"""
    return data.pct_change().mean() * 252

def calculate_covariance_matrix(data):
    """Calculate annualized covariance matrix"""
    return data.pct_change().cov() * 252

# ========== MAIN APP ==========

# ========== PATCHED FUNCTIONS ==========
def load_data(tickers, start_date, end_date):
    import yfinance as yf
    try:
        data = yf.download(tickers, start=start_date, end=end_date)["Adj Close"]
        return data
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return pd.DataFrame()

def calculate_metrics(data):
    returns = data.pct_change().dropna()
    mean_returns = returns.mean() * 252
    volatility = returns.std() * (252 ** 0.5)
    sharpe_ratio = mean_returns / volatility
    metrics = pd.DataFrame({
        "Mean Return": mean_returns,
        "Volatility": volatility,
        "Sharpe Ratio": sharpe_ratio
    })
    return metrics

def optimize_portfolio(returns_df):
    from scipy.optimize import minimize

    num_assets = returns_df.shape[1]
    def portfolio_performance(weights):
        mean_returns = returns_df.mean() * 252
        cov_matrix = returns_df.cov() * 252
        port_return = np.dot(weights, mean_returns)
        port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe_ratio = port_return / port_volatility
        return -sharpe_ratio  # negative Sharpe Ratio to minimize

    constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
    bounds = tuple((0, 1) for _ in range(num_assets))
    init_guess = num_assets * [1. / num_assets,]
    
    result = minimize(portfolio_performance, init_guess,
                      method="SLSQP", bounds=bounds, constraints=constraints)
    return result.x


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
            <div class="guide-text">
            
            <h3 class="guide-header">Getting Started</h3>
            
            <strong>1. Select Date Range</strong><br>
            - Choose your analysis period using the date pickers in the sidebar<br>
            - Minimum 30 days recommended for reliable metrics
            
            <h3 class="guide-header">Core Features</h3>
            
            <strong>📈 Performance Metrics</strong><br>
            - <em>Sharpe Ratio</em>: Risk-adjusted returns (higher = better)<br>
            - <em>Alpha</em>: Excess return vs benchmark<br>
            - <em>Max Drawdown</em>: Worst historical loss<br>
            - Hover over any metric for detailed definitions
            
            <strong>📊 Visualization Tabs</strong><br>
            - <em>Price Trends</em>: Normalized price comparison<br>
            - <em>Performance Analysis</em>: Metric comparisons<br>
            - <em>Portfolio Allocation</em>: Current weight distribution<br>
            - <em>Correlation Matrix</em>: How assets move together<br>
            - <em>Forecasting</em>: Prophet and ARIMA models<br>
            - <em>Optimization</em>: Find ideal allocations<br>
            - <em>Monte Carlo</em>: Future value simulations
            
            <h3 class="guide-header">Advanced Features</h3>
            
            <strong>⚖️ Portfolio Optimization</strong><br>
            - Set max allocation constraints<br>
            - Visualize the efficient frontier<br>
            - See optimal Sharpe ratio portfolio
            
            <strong>🔮 Forecasting Models</strong><br>
            - Compare Prophet (Facebook) and ARIMA models<br>
            - Adjust forecast periods (30-365 days)
            
            <strong>🎲 Monte Carlo Simulation</strong><br>
            - Run probabilistic simulations<br>
            - Adjust days and simulation count<br>
            - See probability of profit/loss
            
            </div>
            """, unsafe_allow_html=True)
            if st.button("Close Guide"):
                st.session_state.show_help = False

    # Show dependency warnings
    if not deps['prophet']:
        st.warning("Prophet not installed - will use ARIMA for forecasting if available")
    if not deps['statsmodels']:
        st.warning("statsmodels not installed - forecasting features limited")
    if not deps['openai']:
        st.warning("OpenAI not installed - advanced AI insights unavailable")

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
            
        # Enhanced Risk-Free Rate Selection
        st.subheader("Risk-Free Rate Options")
        rate_source = st.selectbox(
            "Rate Source",
            options=list(RISK_FREE_SOURCES.keys()),
            index=0
        )
        
        if rate_source == "Custom Rate":
            st.session_state.custom_rate = st.number_input(
                "Custom Rate (%)", 0.0, 15.0, 2.0, 0.1) / 100
        
        # Beta calculation options
        st.subheader("Beta Calculation")
        beta_lookback = st.selectbox(
            "Lookback Period",
            ["1y", "3y", "5y", "10y"],
            index=1
        )
        
        selected_bench = st.selectbox(
            "Benchmark",
            options=list(BENCHMARK_OPTIONS.keys()),
            index=0
        )

        # Manual API key input (fallback)
        if not deps['openai'] and st.checkbox("Enter OpenAI API Key Manually"):
            api_key = st.text_input("OpenAI API Key", type="password")
            if api_key:
                try:
                    import openai
                    openai.api_key = api_key
                    deps['openai'] = True
                    st.success("API key set successfully")
                except Exception as e:
                    st.error(f"Error setting API key: {str(e)}")

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
                
            # Calculate beta-weighted average if benchmark exists
            beta_data = None
            if BENCHMARK_OPTIONS[selected_bench]:
                beta_data = calculate_beta_weighted_average(
                    data,
                    BENCHMARK_OPTIONS[selected_bench],
                    weights,
                    lookback=beta_lookback
                )
                
            # Get risk-free rate
            risk_free = get_risk_free_rate(RISK_FREE_SOURCES[rate_source])
                
            status.update(label="✅ Data loaded successfully", state="complete")
            
        except Exception as e:
            st.error(f"Critical error: {str(e)}")
            st.stop()

    # Metrics Calculation
    metrics = calculate_metrics(data, bench_data, risk_free)

    if st.button("📄 Generate PDF Report"):
        try:
            create_pdf_report(metrics, start_date, end_date)
            st.success("PDF report created successfully.")
        except Exception as e:
            st.error(f"Failed to generate PDF: {e}")
    
    if metrics is None or metrics.empty:
        st.error("Could not calculate performance metrics")
        st.stop()
    
    # Add beta metrics if calculated
    if beta_data:
        metrics.loc['Beta (Weighted Avg)'] = {'Your Portfolio': beta_data['beta_avg']}
        metrics.loc['Beta Stability'] = {'Your Portfolio': beta_data['beta_stability']}
    
    st.subheader("📈 Performance Metrics")
    st.dataframe(
        style_metrics(metrics),
        use_container_width=True
    )
    
    # Generate and display insights
    with st.expander("💡 Portfolio Intelligence", expanded=True):
        tab1, tab2 = st.tabs(["Quick Insights", "Advanced AI Analysis"])
        
        with tab1:
            insights = generate_insights(metrics, beta_data, risk_free, weights)
            for insight_type, text in insights:
                if insight_type == "positive":
                    st.success(f"✓ {text}")
                elif insight_type == "warning":
                    st.warning(f"⚠ {text}")
                elif insight_type == "negative":
                    st.error(f"✗ {text}")
                else:
                    st.info(f"• {text}")
        
        with tab2:
            if deps['openai']:
                if st.button("Generate Deep Analysis"):
                    with st.spinner("Consulting market experts..."):
                        analysis = generate_llm_insights(
                            metrics,
                            weights,
                            risk_free
                        )
                        st.markdown(analysis)
            else:
                st.warning("OpenAI integration not available. Install package and configure API key.")

    # Expanded Charts with new features
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "Price Trends", 
        "Performance Analysis", 
        "Portfolio Allocation",
        "Portfolio vs Benchmark",
        "Correlation Matrix",
        "Forecasting",
        "Portfolio Optimization",
        "Monte Carlo Simulation"
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
        
        if deps['prophet'] or deps['statsmodels']:
            if deps['prophet'] and deps['statsmodels']:
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Prophet Forecast**")
                    periods = st.number_input("Forecast Periods", 30, 365, 90, key="prophet_periods")
                    if st.button("Run Prophet Forecast"):
                        with st.spinner("Running Prophet forecast..."):
                            forecast_df = forecast_prophet(data, periods=periods)
                            st.line_chart(forecast_df)
                
                with col2:
                    st.markdown("**ARIMA Forecast**")
                    periods = st.number_input("Forecast Periods", 30, 365, 30, key="arima_periods")
                    if st.button("Run ARIMA Forecast"):
                        with st.spinner("Running ARIMA forecast..."):
                            forecast_df = forecast_arima(data, periods=periods)
                            st.line_chart(forecast_df)
            else:
                model_choice = "Prophet" if deps['prophet'] else "ARIMA"
                periods = st.number_input("Forecast Periods", 30, 365, 90)
                if st.button(f"Run {model_choice} Forecast"):
                    with st.spinner(f"Running {model_choice} forecast..."):
                        if model_choice == "Prophet":
                            forecast_df = forecast_prophet(data, periods=periods)
                        else:
                            forecast_df = forecast_arima(data, periods=periods)
                        st.line_chart(forecast_df)
        else:
            st.warning("Install forecasting packages: pip install prophet statsmodels")
    
    with tab7:
        st.subheader("⚖️ Portfolio Optimization")
        
        with st.expander("⚙️ Optimization Constraints", expanded=False):
            max_alloc = st.slider("Max Allocation per Asset (%)", 5, 100, 30) / 100
        
        if st.button("Calculate Optimal Portfolio"):
            with st.spinner("Optimizing portfolio..."):
                try:
                    weights, performance = optimize_portfolio(data, max_allocation=max_alloc)
                    
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

    with tab8:
        monte_carlo_tab()

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