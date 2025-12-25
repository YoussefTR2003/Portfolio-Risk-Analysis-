# Portfolio Risk Analysis (Python)

This project implements a **portfolio risk and performance analysis framework** in Python.  
It allows the user to freely select financial assets and portfolio weights, and evaluates the resulting portfolio in terms of **return, volatility, drawdown, and Value at Risk (VaR)**.

The goal is to understand how **portfolio composition affects risk metrics**, using real market data.

---

##  Features

### Asset-level analysis
- Daily **log-returns**
- Annualized volatility
- Skewness
- Excess kurtosis
- Rolling annualized volatility (252 trading days)
- Drawdown & maximum drawdown
- Calmar ratio

### Portfolio-level analysis
- Flexible asset universe (user-defined tickers)
- Fixed-weight portfolio construction
- Annualized return & volatility
- Sharpe ratio (with configurable risk-free rate)
- Portfolio drawdown & maximum drawdown
- Calmar ratio

### Risk measures
- **Parametric Value at Risk (VaR)** under the Normality assumption
- 1-day VaR
- 252-day VaR (illustrative long-horizon risk)
- **Rolling VaR** to capture time-varying risk

### Visualizations
- Return distributions with Normal fit
- Rolling volatility
- Asset and portfolio drawdowns
- Portfolio value index (base 100)
- Rolling VaR evolution

---

##  Methodology

- Prices are downloaded from **Yahoo Finance** via `yfinance`
- Returns are computed as **log-returns**
- Portfolio returns are computed as a **weighted sum of asset log-returns**
- VaR is computed using the **parametric (Gaussian) approach**:

\[
VaR_{\alpha} = V_0 \times \left( -(\mu_h + z_{\alpha}\sigma_h) \right)
\]

where:
- \( \mu_h \) is the expected return over the holding period  
- \( \sigma_h \) is the volatility over the holding period  
- \( z_{\alpha} \) is the Normal quantile  
- \( V_0 \) is the portfolio value  

---

## ⚠️ Important Notes

- VaR assumes **normally distributed returns**, which may underestimate tail risk in the presence of excess kurtosis.
- The 252-day VaR is shown for educational purposes; in practice, short-horizon VaR is more common.
- This project is designed for **learning and demonstration**, not for live trading.

---

##  Installation

```bash
pip install yfinance pandas numpy matplotlib scipy
