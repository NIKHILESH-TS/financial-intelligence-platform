# 🧠 AlphaIntel™
## AI-Powered Real-Time Financial Intelligence Platform

AlphaIntel is a cloud-hosted financial intelligence platform that combines real-time market analytics, NLP-based news sentiment analysis, quantitative risk diagnostics, and machine learning forecasting into a unified decision-support dashboard.

🌐 Live Demo: https://alphaintel-platform.streamlit.app/

---

## Problem Statement

Investors and analysts often rely on multiple disconnected tools for market data, news monitoring, risk assessment, and forecasting.

AlphaIntel brings these capabilities together into a single platform, enabling users to evaluate financial assets through a unified analytical workflow.

---

## Key Features

### 📈 Real-Time Market Analytics
- Live financial market data using Yahoo Finance
- Interactive candlestick chart visualization
- 20-Day Simple Moving Average (SMA) trend tracking

### ⚖️ Quantitative Risk Diagnostics
- Annualized Volatility Analysis
- Maximum Drawdown (Max DD)
- Sharpe Ratio Calculation

### 📰 NLP News Intelligence
- Real-time financial news retrieval
- VADER-based sentiment analysis
- Bullish / Bearish sentiment classification

### 🤖 Machine Learning Forecasting
- Random Forest Regression model
- Autoregressive lag-based feature engineering
- Next-day price prediction
- Confidence interval estimation

### 🔍 Model Explainability
- Feature importance visualization
- Transparent forecasting pipeline
- Explainable ML outputs

### 🌎 Asset-Agnostic Design
Supports any Yahoo Finance asset including:
- Stocks (AAPL, TSLA, NVDA, META)
- Indian Equities (RELIANCE.NS, TCS.NS, INFY.NS)
- Market Indices (^GSPC, ^NSEI)
- Cryptocurrencies (BTC-USD, ETH-USD)

---

## Technology Stack

**Frontend**
- Streamlit
- Plotly

**Data Engineering**
- Pandas
- NumPy
- BeautifulSoup

**Machine Learning**
- Scikit-Learn
- Random Forest Regressor

**Natural Language Processing**
- VADER Sentiment Analysis

**Data Source**
- Yahoo Finance

---

## System Architecture

Data Collection
→ Data Cleaning & Processing
→ Risk Analytics Engine
→ NLP Sentiment Engine
→ Machine Learning Forecasting
→ Interactive Dashboard Visualization

---

## Running Locally

```bash
git clone https://github.com/NIKHILESH-TS/financial-intelligence-platform.git
cd financial-intelligence-platform

pip install -r requirements.txt

streamlit run app.py
```

---

## Disclaimer

This platform is intended for educational and analytical purposes only. The forecasts, sentiment signals, and quantitative indicators presented should not be interpreted as financial advice or investment recommendations.

---

## Author

Nikhilesh TS

AI-Powered Financial Analytics • Machine Learning • Data Science