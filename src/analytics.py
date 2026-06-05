from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import minimize

TRADING_DAYS = 252

def returns_from_prices(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna(how="all").fillna(0.0)

def ann_return(r: pd.Series | pd.DataFrame):
    return (1 + r).prod() ** (TRADING_DAYS / len(r)) - 1

def ann_vol(r: pd.Series | pd.DataFrame):
    return r.std() * np.sqrt(TRADING_DAYS)

def max_drawdown(r: pd.Series) -> float:
    wealth = (1 + r).cumprod()
    dd = wealth / wealth.cummax() - 1
    return float(dd.min())

def var_cvar(r: pd.Series, alpha: float = 0.05) -> tuple[float, float]:
    q = float(r.quantile(alpha))
    cvar = float(r[r <= q].mean()) if (r <= q).any() else q
    return q, cvar

def portfolio_returns(returns: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    return returns.dot(weights)

def metrics_table(strategy_returns: dict[str, pd.Series], rf: float = 0.02) -> pd.DataFrame:
    rows = []
    for name, r in strategy_returns.items():
        ar = float(ann_return(r))
        av = float(ann_vol(r))
        sr = (ar - rf) / av if av > 0 else np.nan
        v, cv = var_cvar(r)
        rows.append({
            "Strategy": name,
            "Annualized Return": ar,
            "Annualized Volatility": av,
            "Sharpe Ratio": sr,
            "Max Drawdown": max_drawdown(r),
            "VaR 95%": v,
            "CVaR 95%": cv,
        })
    return pd.DataFrame(rows).set_index("Strategy")

def equal_weight(n: int) -> np.ndarray:
    return np.repeat(1 / n, n)

def min_vol_weights(returns: pd.DataFrame, max_weight: float = 0.35) -> np.ndarray:
    n = returns.shape[1]
    cov = returns.cov().values * TRADING_DAYS
    def obj(w): return float(np.sqrt(w @ cov @ w))
    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1},)
    bounds = [(0, max_weight)] * n
    x0 = equal_weight(n)
    res = minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=cons, options={"maxiter": 500})
    return res.x if res.success else x0

def max_sharpe_weights(returns: pd.DataFrame, max_weight: float = 0.35, rf: float = 0.02) -> np.ndarray:
    n = returns.shape[1]
    mu = ann_return(returns).values
    cov = returns.cov().values * TRADING_DAYS
    def obj(w):
        ret = float(w @ mu)
        vol = float(np.sqrt(w @ cov @ w))
        return -((ret - rf) / vol) if vol > 0 else 1e6
    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1},)
    bounds = [(0, max_weight)] * n
    x0 = equal_weight(n)
    res = minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=cons, options={"maxiter": 500})
    return res.x if res.success else x0

def inv_vol_weights(returns: pd.DataFrame, max_weight: float = 0.35) -> np.ndarray:
    vol = returns.std().replace(0, np.nan)
    inv = 1 / vol
    w = (inv / inv.sum()).fillna(1 / len(vol)).values
    # cap iteratively and redistribute excess
    for _ in range(20):
        excess = np.maximum(w - max_weight, 0).sum()
        w = np.minimum(w, max_weight)
        uncapped = w < max_weight - 1e-12
        if excess <= 1e-12 or not uncapped.any():
            break
        w[uncapped] += excess * w[uncapped] / w[uncapped].sum()
    return w / w.sum()

def strategy_weights(returns: pd.DataFrame, method: str, max_weight: float, rf: float) -> np.ndarray:
    if method == "Equal Weighted":
        return equal_weight(returns.shape[1])
    if method == "Minimum Volatility":
        return min_vol_weights(returns, max_weight)
    if method == "Maximum Sharpe":
        return max_sharpe_weights(returns, max_weight, rf)
    if method == "Risk Parity / Inverse Vol":
        return inv_vol_weights(returns, max_weight)
    return equal_weight(returns.shape[1])

def simple_backtest(returns: pd.DataFrame, method: str, max_weight: float, rf: float, lookback_days: int = 252, cost_bps: float = 5.0) -> tuple[pd.Series, pd.DataFrame, pd.Series]:
    if len(returns) <= lookback_days + 20:
        w = strategy_weights(returns, method, max_weight, rf)
        return portfolio_returns(returns, w), pd.DataFrame([w], columns=returns.columns, index=[returns.index[0]]), pd.Series(0.0, index=returns.index)
    month_ends = returns.resample("ME").last().index
    w_prev = np.zeros(returns.shape[1])
    out = []
    weights_log = []
    costs = []
    for i, date in enumerate(month_ends):
        hist = returns.loc[:date].tail(lookback_days)
        if len(hist) < 60:
            continue
        w = strategy_weights(hist, method, max_weight, rf)
        turnover = np.abs(w - w_prev).sum()
        cost = turnover * cost_bps / 10000
        start = date
        end = month_ends[i + 1] if i + 1 < len(month_ends) else returns.index[-1]
        period = returns.loc[(returns.index > start) & (returns.index <= end)]
        if period.empty:
            continue
        pr = portfolio_returns(period, w).copy()
        pr.iloc[0] -= cost
        out.append(pr)
        weights_log.append(pd.Series(w, index=returns.columns, name=date))
        costs.append(pd.Series(cost, index=[period.index[0]]))
        w_prev = w
    if not out:
        w = strategy_weights(returns, method, max_weight, rf)
        return portfolio_returns(returns, w), pd.DataFrame([w], columns=returns.columns, index=[returns.index[0]]), pd.Series(0.0, index=returns.index)
    return pd.concat(out).sort_index(), pd.DataFrame(weights_log), pd.concat(costs).reindex(returns.index).fillna(0.0)

def asset_class_weights(weights: pd.Series, universe: pd.DataFrame) -> pd.Series:
    mapping = universe.set_index("Ticker")["Asset Class"].to_dict()
    classes = pd.Series(weights.index.map(lambda x: mapping.get(x, "Other")), index=weights.index)
    return weights.groupby(classes).sum().sort_values(ascending=False)
