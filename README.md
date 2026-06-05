# Portfolio Management & Risk Analytics Dashboard

A Streamlit dashboard for portfolio analytics, designed for junior portfolio management / asset management / risk analyst applications.

## Why this version exists

This version avoids manual ticker parsing issues. Instead of writing tickers in a text box like `AAPL:Equity`, the app uses an editable table with two columns:

- `Ticker`
- `Asset Class`

It also has a synthetic demo-data fallback. Therefore, the dashboard can still run even if Yahoo Finance rejects a ticker, blocks a request, or returns missing data.

## Features

- Editable investment universe table
- Preset portfolios
- Yahoo Finance data download with fallback demo data
- Equal-weighted portfolio
- Minimum-volatility portfolio
- Maximum-Sharpe portfolio
- Risk parity / inverse volatility portfolio
- Walk-forward monthly backtest
- Transaction costs in basis points
- Allocation constraints with max weight per asset
- Performance metrics: return, volatility, Sharpe ratio, drawdown, VaR, CVaR
- Benchmark comparison
- Correlation matrix
- Asset class allocation
- HTML report export

## Run locally

```bash
cd portfolio-dashboard-working
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## GitHub setup

```bash
git init
git add .
git commit -m "Initial portfolio dashboard"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/portfolio-dashboard-working.git
git push -u origin main
```

## Suggested CV bullet

Developed a Python-based portfolio management and risk analytics dashboard using Streamlit. The tool compares equal-weighted, minimum-volatility, maximum-Sharpe and risk-parity allocations, with walk-forward rebalancing, transaction costs, VaR/CVaR, drawdown analysis, benchmark comparison and asset-class exposure monitoring.

## Important limitation

This project is intended as a portfolio analytics and educational tool. It does not produce investment advice and should not be presented as a model designed to beat the market.
