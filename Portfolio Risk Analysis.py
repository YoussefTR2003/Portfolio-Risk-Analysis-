import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy import stats

# -----------------------------
# 1) Parameters
# -----------------------------
TICKERS = ["JNJ", "TSLA"]
START = "2024-01-01"
END = "2025-12-01"
TRADING_DAYS = 252

WEIGHTS = pd.Series({"JNJ": 0.50, "TSLA": 0.50})  # must sum to 1
V0 = 100000
RF = 0.06

ALPHA = 0.05                 # 5% tail -> 95% VaR
VAR_WINDOW = 252             # rolling window for VaR estimation

# -----------------------------
# 2) Helpers
# -----------------------------
def download_prices(tickers, start, end):
    prices = yf.download(tickers, start=start, end=end, auto_adjust=True)["Close"].dropna()
    # Ensure columns order + name consistency
    prices = prices[tickers]
    return prices

def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices / prices.shift(1)).dropna()

def compute_drawdown(log_returns: pd.Series) -> pd.Series:
    """Drawdown from log-returns."""
    index = np.exp(log_returns.cumsum()) * 100
    running_max = index.cummax()
    return (index - running_max) / running_max

def portfolio_log_return(log_ret: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """Daily portfolio log-return (weighted sum)."""
    w = weights.reindex(log_ret.columns)
    w = w / w.sum()  # safety
    return (log_ret * w).sum(axis=1)

def var_parametric_normal(mu: float, sigma: float, v0: float, alpha: float, horizon_days: int = 1) -> float:
    """
    Parametric Normal VaR (positive number = loss threshold).
    """
    z = stats.norm.ppf(alpha)  # negative
    mu_h = horizon_days * mu
    sigma_h = np.sqrt(horizon_days) * sigma
    return v0 * (-(mu_h + z * sigma_h))

def rolling_var_parametric(port_log_ret: pd.Series, v0: float, alpha: float, window: int, horizon_days: int = 1) -> pd.Series:
    """
    Rolling Parametric Normal VaR (time series).
    """
    mu_roll = port_log_ret.rolling(window).mean()
    sigma_roll = port_log_ret.rolling(window).std(ddof=1)

    z = stats.norm.ppf(alpha)  # negative
    mu_h = horizon_days * mu_roll
    sigma_h = np.sqrt(horizon_days) * sigma_roll

    var_series = v0 * (-(mu_h + z * sigma_h))
    return var_series.dropna()

# -----------------------------
# 3) Download & compute returns
# -----------------------------
prices = download_prices(TICKERS, START, END)
log_ret = compute_log_returns(prices)

# -----------------------------
# 4) Asset statistics
# -----------------------------
ann_vol = log_ret.std() * np.sqrt(TRADING_DAYS)
skewness = log_ret.skew()
excess_kurt = log_ret.kurtosis()

print("\n=== Asset Statistics ===")
print("Annualized Volatility:\n", ann_vol)
print("\nSkewness:\n", skewness)
print("\nExcess Kurtosis:\n", excess_kurt)

rolling_vol = log_ret.rolling(TRADING_DAYS).std() * np.sqrt(TRADING_DAYS)

# -----------------------------
# 5) Asset risk metrics (Drawdown, Calmar)
# -----------------------------
dd_assets = pd.DataFrame({t: compute_drawdown(log_ret[t]) for t in TICKERS})
max_dd_assets = dd_assets.min()

ann_return_assets = log_ret.mean() * TRADING_DAYS
calmar_assets = ann_return_assets / max_dd_assets.abs()

print("\n=== Risk Metrics (Assets) ===")
print("Max Drawdown:\n", max_dd_assets)
print("\nAnnualized Return:\n", ann_return_assets)
print("\nCalmar Ratio:\n", calmar_assets)

# -----------------------------
# 6) Portfolio metrics
# -----------------------------
port_lr = portfolio_log_return(log_ret, WEIGHTS)

port_ann_return = port_lr.mean() * TRADING_DAYS
port_ann_vol = port_lr.std(ddof=1) * np.sqrt(TRADING_DAYS)
port_sharpe = (port_ann_return - RF) / port_ann_vol

dd_port = compute_drawdown(port_lr)
port_max_dd = dd_port.min()
port_calmar = port_ann_return / abs(port_max_dd)

# Point VaR (based on full sample)
mu_d, sigma_d = port_lr.mean(), port_lr.std(ddof=1)
var_1d = var_parametric_normal(mu_d, sigma_d, V0, alpha=ALPHA, horizon_days=1)
var_252d = var_parametric_normal(mu_d, sigma_d, V0, alpha=ALPHA, horizon_days=252)

print("\n=== Portfolio Metrics ===")
print(f"Weights: {WEIGHTS.to_dict()}")
print(f"Annualized Return: {port_ann_return:.4f}")
print(f"Annualized Volatility: {port_ann_vol:.4f}")
print(f"Sharpe (RF={RF:.2%}): {port_sharpe:.4f}")
print(f"Max Drawdown: {port_max_dd:.4f}")
print(f"Calmar Ratio: {port_calmar:.4f}")
print(f"VaR 95% (1-day, parametric): {var_1d:,.2f} $")
print(f"VaR 95% (252-day, parametric): {var_252d:,.2f} $")

# Rolling VaR series (what you wanted to plot)
var_1d_roll = rolling_var_parametric(port_lr, V0, alpha=ALPHA, window=VAR_WINDOW, horizon_days=1)
var_252d_roll = rolling_var_parametric(port_lr, V0, alpha=ALPHA, window=VAR_WINDOW, horizon_days=252)

# -----------------------------
# 7) Plots
# -----------------------------

# (A) Return distributions (AAPL, TSLA)
fig, ax = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

for i, t in enumerate(TICKERS):
    ax[i].hist(log_ret[t], bins=60, density=True, alpha=0.6, label=t)
    mu, sig = log_ret[t].mean(), log_ret[t].std(ddof=1)
    x = np.linspace(log_ret[t].min(), log_ret[t].max(), 200)
    ax[i].plot(x, stats.norm.pdf(x, mu, sig), linestyle="--", label="Normal fit")
    ax[i].set_title(f"{t} Return Distribution")
    ax[i].grid(True, alpha=0.3)
    ax[i].legend()

plt.suptitle("Distribution of Daily Log-Returns (Skewness & Excess Kurtosis)")
plt.tight_layout()
plt.show()

# (B) Rolling annualized volatility
rolling_vol.plot(figsize=(10, 4))
plt.title("Rolling Annualized Volatility (252 trading days)")
plt.ylabel("Volatility")
plt.grid(True, alpha=0.3)
plt.show()

# (C) Drawdowns assets
dd_assets.plot(figsize=(10, 4))
plt.title("Drawdowns (AAPL vs TSLA)")
plt.ylabel("Drawdown")
plt.grid(True, alpha=0.3)
plt.show()

# (D) Portfolio drawdown
dd_port.plot(figsize=(10, 4))
plt.title("Portfolio Drawdown")
plt.ylabel("Drawdown")
plt.grid(True, alpha=0.3)
plt.show()

# (E) Portfolio value index
port_index = np.exp(port_lr.cumsum()) * 100
port_index.plot(figsize=(10, 4))
plt.title("Portfolio Value Index (Base = 100)")
plt.ylabel("Index")
plt.grid(True, alpha=0.3)
plt.show()

# (F) Rolling VaR evolution (1-day vs 252-day)
plt.figure(figsize=(10, 4))
plt.plot(var_1d_roll, label="VaR 95% (1-day, rolling)")
plt.plot(var_252d_roll, label="VaR 95% (252-day, rolling)", linestyle="--")
plt.title(f"Rolling Parametric VaR (Normal) - window={VAR_WINDOW} days")
plt.ylabel("VaR ($)")
plt.xlabel("Date")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()
