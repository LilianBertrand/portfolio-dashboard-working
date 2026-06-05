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

## Live Demo

https://portfolio-dashboard-working-asc33z9mg3zqsd3sbh9adh.streamlit.app/
