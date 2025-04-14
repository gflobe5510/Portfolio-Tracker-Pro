# 📊 Portfolio Tracker Pro  
**Live App:** [https://portfoliotrackerpro.streamlit.app/](https://portfoliotrackerpro.streamlit.app/)  
**GitHub:** [https://github.com/gflobe5510](https://github.com/gflobe5510)

An interactive, Python-powered dashboard for exploring asset performance, visualizing risk, and forecasting future returns. Built for analysts, investors, and financial professionals who need fast, clean, and reliable portfolio insights.

---

## 🔎 Overview

**Portfolio Tracker Pro** helps users:

- Compare over **500 stocks, ETFs, and indices**
- Normalize price series to visualize **relative growth**
- Analyze assets with interactive **line, bar, and pie charts**
- Select a **custom benchmark** for alpha/beta analysis
- Run time series forecasting using both **ARIMA** and **Facebook Prophet**
- Generate PDF reports with charts, metrics, and summaries
- Export raw or normalized data to **CSV**

Designed for performance, readability, and real-world analytical workflows, the app is built using Streamlit and a modular Python backend.

---

## 📈 Features

✅ 500+ asset universe via searchable dropdown  
✅ Normalized vs. raw price comparisons  
✅ Custom benchmark selection (e.g., SPY)  
✅ Risk metrics:  
  • **Sharpe Ratio**  
  • **Alpha & Beta** (vs. benchmark)  
  • **Volatility**, **Max Drawdown**  
✅ Time series forecasting using:  
  • **ARIMA** via `statsmodels`  
  • **Prophet** via Meta’s forecasting package  
✅ Correlation matrix for asset relationships  
✅ PDF reporting with performance charts  
✅ One-click export to CSV and PDF  
✅ Built-in forecasting tips for user guidance

---

## 🧱 Architecture

📁 portfolio_tracker_pro/ │ ├── Portfolio_Tracker_app.py # Streamlit frontend interface ├── portfolio.py # Charting, analytics, metrics ├── pdf_utils.py # PDF report generation ├── data/ │ └── tickers.csv # Ticker universe (~500+ assets) ├── assets/ # App styling, screenshots, branding ├── requirements.txt # Package dependencies └── README.md # This file




