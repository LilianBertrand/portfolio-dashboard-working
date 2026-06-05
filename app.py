from __future__ import annotations
import pandas as pd
import streamlit as st
from src.data import PRESETS, clean_universe, load_prices
from src.analytics import returns_from_prices, portfolio_returns, strategy_weights, simple_backtest, metrics_table, asset_class_weights
from src.plots import performance_chart, weights_bar, corr_heatmap

st.set_page_config(page_title="Portfolio Management Dashboard", layout="wide")

st.title("Portfolio Management & Risk Analytics Dashboard")
st.caption("Robust Streamlit version: no manual ticker parsing required. Use the editable asset table or a preset portfolio.")

with st.sidebar:
    st.header("Configuration")
    preset_name = st.selectbox("Preset portfolio", list(PRESETS.keys()), index=0)
    start = st.date_input("Start date", pd.to_datetime("2018-01-01"))
    end = st.date_input("End date", pd.Timestamp.today())
    benchmark = st.text_input("Benchmark ticker", value="SPY").upper().strip().replace(",", "")
    method = st.selectbox("Main allocation method", ["Equal Weighted", "Minimum Volatility", "Maximum Sharpe", "Risk Parity / Inverse Vol"], index=2)
    max_weight = st.slider("Maximum weight per asset", 0.10, 1.00, 0.35, 0.05)
    lookback_days = st.slider("Rebalancing lookback window", 126, 756, 252, 21)
    cost_bps = st.slider("Transaction cost, bps", 0.0, 50.0, 5.0, 1.0)
    rf = st.number_input("Risk-free rate", value=0.02, step=0.005, format="%.3f")
    use_live_data = st.toggle("Use Yahoo Finance live data", value=True)

st.subheader("1. Investment universe")
st.write("Edit the table directly. This avoids the previous comma/colon parsing bug. Keep at least two rows.")

base = PRESETS[preset_name].copy()
universe = st.data_editor(
    base,
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "Ticker": st.column_config.TextColumn("Ticker", help="Example: AAPL, MSFT, TLT, GLD"),
        "Asset Class": st.column_config.SelectboxColumn("Asset Class", options=["Equity", "Bond", "Gold", "Cash", "Commodity", "Other"]),
    },
    key=f"universe_editor_{preset_name}",
)
universe = clean_universe(universe)

if len(universe) < 2:
    st.error("Please keep at least two valid assets in the table.")
    st.stop()

tickers = universe["Ticker"].tolist()
all_tickers = list(dict.fromkeys(tickers + ([benchmark] if benchmark else [])))

try:
    prices, msg = load_prices(all_tickers, str(start), str(end), use_live_data=use_live_data)
    st.info(msg)
except Exception as exc:
    st.error(f"The analysis could not be completed: {exc}")
    st.stop()

available_assets = [t for t in tickers if t in prices.columns]
if len(available_assets) < 2:
    st.error("Less than two portfolio assets were available after data loading. Switch off live data to use demo data, or use the preset portfolio.")
    st.stop()

prices_assets = prices[available_assets].dropna()
returns = returns_from_prices(prices_assets)

if benchmark in prices.columns:
    bench_returns = returns_from_prices(prices[[benchmark]].dropna())[benchmark]
    bench_returns = bench_returns.reindex(returns.index).dropna()
else:
    bench_returns = None

st.subheader("2. Portfolio construction")

methods = ["Equal Weighted", "Minimum Volatility", "Maximum Sharpe", "Risk Parity / Inverse Vol"]
strategy_returns = {}
weights_dict = {}
for m in methods:
    w = strategy_weights(returns, m, max_weight, rf)
    weights_dict[m] = pd.Series(w, index=returns.columns)
    strategy_returns[m] = portfolio_returns(returns, w)

bt_returns, bt_weights, costs = simple_backtest(returns, method, max_weight, rf, lookback_days, cost_bps)
strategy_returns[f"Walk-forward {method}"] = bt_returns

if bench_returns is not None and len(bench_returns) > 0:
    strategy_returns[f"Benchmark {benchmark}"] = bench_returns

metric_df = metrics_table(strategy_returns, rf)

c1, c2, c3, c4 = st.columns(4)
main_name = f"Walk-forward {method}"
if main_name in metric_df.index:
    row = metric_df.loc[main_name]
    c1.metric("Annualized Return", f"{row['Annualized Return']:.2%}")
    c2.metric("Annualized Volatility", f"{row['Annualized Volatility']:.2%}")
    c3.metric("Sharpe Ratio", f"{row['Sharpe Ratio']:.2f}")
    c4.metric("Max Drawdown", f"{row['Max Drawdown']:.2%}")

st.plotly_chart(performance_chart(strategy_returns), use_container_width=True)

st.subheader("3. Metrics")
st.dataframe(metric_df.style.format("{:.2%}", subset=["Annualized Return", "Annualized Volatility", "Max Drawdown", "VaR 95%", "CVaR 95%"]).format("{:.2f}", subset=["Sharpe Ratio"]), use_container_width=True)

st.subheader("4. Allocation")
col1, col2 = st.columns(2)
selected_weights = weights_dict[method]
with col1:
    st.plotly_chart(weights_bar(selected_weights, f"{method} weights"), use_container_width=True)
with col2:
    class_w = asset_class_weights(selected_weights, universe[universe["Ticker"].isin(returns.columns)])
    st.plotly_chart(weights_bar(class_w, "Asset class allocation"), use_container_width=True)

st.subheader("5. Risk diagnostics")
st.plotly_chart(corr_heatmap(returns), use_container_width=True)

st.subheader("6. Rebalancing details")
st.write("Last walk-forward weights")
if not bt_weights.empty:
    st.dataframe(bt_weights.tail(12).style.format("{:.2%}"), use_container_width=True)
st.write(f"Total estimated transaction costs over backtest: **{costs.sum():.2%}**")

st.subheader("7. Export")
report_html = f"""
<h1>Portfolio Management & Risk Analytics Dashboard</h1>
<p><b>Universe:</b> {', '.join(available_assets)}</p>
<p><b>Main strategy:</b> {method}</p>
<p><b>Benchmark:</b> {benchmark}</p>
<h2>Performance Metrics</h2>
{metric_df.to_html(float_format=lambda x: f'{x:.4f}')}
<h2>Current Weights</h2>
{selected_weights.to_frame('Weight').to_html(float_format=lambda x: f'{x:.4f}')}
<h2>Asset Class Allocation</h2>
{class_w.to_frame('Weight').to_html(float_format=lambda x: f'{x:.4f}')}
"""
st.download_button("Download HTML report", report_html, file_name="portfolio_report.html", mime="text/html")

st.success("Dashboard completed successfully.")
