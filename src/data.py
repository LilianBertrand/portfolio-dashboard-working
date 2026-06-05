from __future__ import annotations
import numpy as np
import pandas as pd

DEFAULT_UNIVERSE = pd.DataFrame({
    "Ticker": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "JPM", "XOM", "TLT", "GLD", "SHY"],
    "Asset Class": ["Equity", "Equity", "Equity", "Equity", "Equity", "Equity", "Equity", "Bond", "Gold", "Cash"],
})

PRESETS = {
    "Balanced demo portfolio": DEFAULT_UNIVERSE,
    "Simple test portfolio": pd.DataFrame({"Ticker": ["AAPL", "MSFT", "TLT", "GLD"], "Asset Class": ["Equity", "Equity", "Bond", "Gold"]}),
    "ETF allocation portfolio": pd.DataFrame({"Ticker": ["SPY", "QQQ", "TLT", "IEF", "GLD", "SHY"], "Asset Class": ["Equity", "Equity", "Bond", "Bond", "Gold", "Cash"]}),
}

def clean_universe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Ticker"] = out["Ticker"].astype(str).str.upper().str.strip().str.replace(",", "", regex=False)
    out["Asset Class"] = out["Asset Class"].astype(str).str.strip().replace("", "Other")
    out = out[out["Ticker"].str.len() > 0]
    out = out.drop_duplicates(subset=["Ticker"], keep="first")
    return out.reset_index(drop=True)

def synthetic_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    dates = pd.bdate_range(start=start, end=end)
    if len(dates) < 60:
        dates = pd.bdate_range(end=pd.Timestamp.today(), periods=900)
    rng = np.random.default_rng(42)
    prices = pd.DataFrame(index=dates)
    for i, t in enumerate(tickers):
        if t in {"TLT", "IEF", "BND"}:
            mu, sigma = 0.00012, 0.006
        elif t in {"GLD", "IAU"}:
            mu, sigma = 0.00018, 0.009
        elif t in {"SHY", "BIL", "CASH"}:
            mu, sigma = 0.00007, 0.001
        else:
            mu, sigma = 0.00035, 0.018
        shocks = rng.normal(mu, sigma, len(dates))
        prices[t] = 100 * np.exp(np.cumsum(shocks)) * (1 + i * 0.01)
    return prices

def load_prices(tickers: list[str], start: str, end: str, use_live_data: bool = True) -> tuple[pd.DataFrame, str]:
    tickers = [str(t).upper().strip().replace(",", "") for t in tickers if str(t).strip()]
    tickers = list(dict.fromkeys(tickers))
    if len(tickers) < 2:
        raise ValueError("Please keep at least two valid tickers in the asset table.")
    if use_live_data:
        try:
            import yfinance as yf
            raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False, group_by="column", threads=False)
            if isinstance(raw.columns, pd.MultiIndex):
                if "Close" in raw.columns.get_level_values(0):
                    prices = raw["Close"].copy()
                else:
                    prices = raw.xs("Close", level=1, axis=1).copy()
            else:
                prices = raw[["Close"]].copy() if "Close" in raw.columns else raw.copy()
                if len(tickers) == 1:
                    prices.columns = tickers
            if isinstance(prices, pd.Series):
                prices = prices.to_frame(tickers[0])
            prices = prices.dropna(axis=1, how="all").ffill().dropna()
            valid = [c for c in prices.columns if c in tickers]
            prices = prices[valid]
            if prices.shape[1] >= 2 and len(prices) >= 60:
                return prices, "Live Yahoo Finance data loaded successfully."
        except Exception as exc:
            msg = f"Live data failed, fallback demo data used. Details: {exc}"
            return synthetic_prices(tickers, start, end), msg
    return synthetic_prices(tickers, start, end), "Demo data mode: synthetic price series used, so the dashboard always runs."
