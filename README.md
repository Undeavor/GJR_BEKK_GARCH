# 📈 Multi-Asset Portfolio Optimization: MGARCH-GJR + VAR

A complete quantitative portfolio optimization pipeline combining **multivariate GARCH with leverage effects (BEKK-GJR)** for dynamic covariance modeling and a **Vector Autoregression (VAR)** model for return forecasting — applied to CAC 40 and other equities.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Theoretical Background](#theoretical-background)
- [Backtest Strategies](#backtest-strategies)
- [Project Files](#project-files)
- [Installation](#installation)
- [Usage](#usage)
- [Interactive Dashboard](#interactive-dashboard)
- [Output Files](#output-files)
- [Dependencies](#dependencies)

---

## Overview

This project implements an end-to-end pipeline for dynamic portfolio construction:

1. **Covariance modeling** via BEKK-GJR(1,1) — a multivariate GARCH model that captures time-varying volatility, cross-asset correlation dynamics, and asymmetric leverage effects (negative shocks have a stronger impact than positive ones).
2. **Return forecasting** via a VAR model estimated at each rebalancing step.
3. **Portfolio optimization** by maximizing the Sharpe ratio subject to allocation constraints, with optional L1 transaction cost penalization.
4. **Three backtest strategies** that simulate realistic investment scenarios — all-in, DCA with full rebalancing, and DCA with fractional rebalancing.

---

## Architecture

```
raw_to_data.py              →  data.npz
        ↓
data_to_cov_and_returns.py  →  cov_and_modelparams.npz
        ↓
cov_to_weights.py           →  backtest plots + metrics
        ↑
   dashboard.py (Streamlit UI wrapping the whole pipeline)
```

| Module | Role |
|---|---|
| `raw_to_data.py` | Download historical price data via `yfinance` and save log-returns |
| `data_to_cov_and_returns.py` | Fit BEKK-GJR(1,1) via MLE, compute conditional covariances, export model params |
| `cov_to_weights.py` | Run backtests using forecasted covariances + VAR returns, compute Sharpe ratios |
| `dashboard.py` | Streamlit interactive interface to configure and visualize all strategies |
| `yfinance_tickers.py` | Curated ticker dictionary (CAC 40, US large-cap, ETFs, crypto, indices) |

---

## Theoretical Background

### Volatility Model: BEKK-GJR(1,1)

The model captures the full joint conditional covariance matrix $H_t$ at each time step:

$$H_t = CC^\top + A^\top \varepsilon_{t-1}\varepsilon_{t-1}^\top A + B^\top H_{t-1} B + G^\top \left(\varepsilon_{t-1}\varepsilon_{t-1}^\top \odot \mathbf{1}_{\varepsilon_{t-1}<0}\right) G$$

- **$CC^\top$** — unconditional baseline covariance (positive semi-definite by construction)
- **$A^\top \varepsilon\varepsilon^\top A$** — ARCH term: reaction to recent squared innovations
- **$B^\top H_{t-1} B$** — GARCH term: persistence of past volatility
- **$G^\top (\ldots) G$** — GJR asymmetry term: amplified response to negative shocks (leverage effect)

Parameters are estimated by **maximum log-likelihood**, accelerated with `numba` JIT compilation.

### Return Forecasting: VAR(p)

At each rebalancing step, a VAR model of order $p=3$ is fitted on the available in-sample log-returns. The one-step-ahead forecast $\hat{\mu}_{t+1}$ is used as the expected return vector for portfolio optimization.

### Portfolio Optimization

Given $\hat{\mu}_{t+1}$ and $H_t$, portfolio weights $w$ are found by solving:

$$\max_w \quad \hat{\mu}_{t+1}^\top w - \frac{1}{2} w^\top H_t w - \lambda_{tc} \|w - w_{t-1}\|_1$$

subject to:
$$\sum_i w_i = 1, \quad -1 \leq w_i \leq 1 \quad \forall i$$

The L1 penalty $\lambda_{tc}$ penalizes large portfolio turnover, directly reflecting transaction costs. Solved via `cvxpy` with the OSQP solver.

### Performance Metric: Realized Sharpe Ratio

$$\text{Sharpe} = \sqrt{252} \cdot \frac{\bar{r}_e}{\sigma_e}$$

where $r_e = r_t - r_f$ are excess daily returns over the risk-free rate (3% annually).

---

## Backtest Strategies

| Strategy | Description | Transaction Costs |
|---|---|---|
| **All-In** | Full capital invested on day 1, daily rebalancing with optimal weights | Simulated at 0.5% per unit of turnover |
| **Regu (DCA)** | Fixed periodic contribution, full portfolio rebalanced each period | 0.1% per unit of turnover |
| **OnlyRegu** | Fixed periodic contribution, only new cash is allocated optimally; existing holdings left untouched | 0.1% only on new cash |
| **1/n (Benchmark)** | Equal-weight portfolio, no rebalancing | None |

Each strategy produces four portfolio value series:
- **Sans frais** — no transaction costs (theoretical upper bound)
- **Avec frais (Brut)** — costs applied to unconstrained weights
- **Avec frais (Optimisé)** — costs applied to turnover-penalized weights
- **1/n** — equal-weight benchmark

---

## Project Files

```
.
├── raw_to_data.py                # Step 1: download price data
├── data_to_cov_and_returns.py    # Step 2: fit BEKK-GJR model
├── cov_to_weights.py             # Step 3: backtest strategies
├── dashboard.py                  # Streamlit UI
├── yfinance_tickers.py           # Ticker reference dictionary
├── requirements.txt              # Python dependencies
├── data.npz                      # (generated) raw price data
├── cov_and_modelparams.npz       # (generated) model params + covariances
└── backtest_results.npz          # (generated) backtest portfolio values
```

---

## Installation

**Python 3.9+ recommended.**

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
pip install -r requirements.txt
```

Full dependency list:

```
numpy pandas matplotlib yfinance scipy statsmodels numba cvxpy streamlit plotly
```

> **Note on `numba`:** The first run will trigger JIT compilation of the GARCH likelihood functions, which may take 30–60 seconds. Subsequent runs are significantly faster.

---

## Usage

### Step 1 — Download price data

```bash
python raw_to_data.py
```

Fetches daily closing prices for 18 CAC 40 tickers from Yahoo Finance (2010–2025) and saves them to `data.npz`. Edit the `syms` list in the script to change the asset universe.

### Step 2 — Fit the BEKK-GJR model

```bash
python data_to_cov_and_returns.py
```

Fits the BEKK-GJR(1,1) model via MLE on the training set (85% of data). Saves model parameters $(C, A, B, G)$, conditional covariance matrices, and the full return series to `cov_and_modelparams.npz`.

> This step is computationally intensive. Estimated runtime: **5–30 minutes** depending on the number of assets and available hardware.

### Step 3 — Run backtest

```bash
python cov_to_weights.py
```

Prompts you to choose a strategy:

```
Stratégie (allin/regu/onlyregu) :
```

Outputs a matplotlib chart of portfolio values over the test period, with realized Sharpe ratios in the legend.

---

## Interactive Dashboard

A hosted Streamlit app is available at:

```
https://projetetude28.streamlit.app/
```

The dashboard allows you to:
- Select tickers from the full `yfinance_tickers.py` dictionary (CAC 40, US stocks, ETFs, crypto…)
- Set the initial capital and DCA contribution amount
- Choose the backtest strategy
- Visualize portfolio evolution and compare Sharpe ratios in real time

To run locally:

```bash
streamlit run dashboard.py
```

---

## Output Files

| File | Content |
|---|---|
| `data.npz` | `data` (DataFrame), `n_dims` |
| `cov_and_modelparams.npz` | `y`, `test_size`, `A`, `B`, `C`, `G`, `H_train`, `n_dims` |
| `backtest_results.npz` | Portfolio value arrays for each strategy and variant |

---

## Supported Asset Universes

`yfinance_tickers.py` provides ready-to-use ticker dictionaries for:

| Universe | Examples |
|---|---|
| CAC 40 | `AI.PA`, `MC.PA`, `BNP.PA`, `SAN.PA`, … |
| US Large Cap | `AAPL`, `MSFT`, `NVDA`, `JPM`, … |
| US ETFs | `SPY`, `QQQ`, `TLT`, `GLD`, … |
| Europe | `ULVR.L`, `VOW3.DE`, `SAP.DE`, … |
| Crypto | `BTC-USD`, `ETH-USD`, `SOL-USD`, … |
| Indices | `^GSPC`, `^FCHI`, `^GDAXI`, … |

---

## Dependencies

| Package | Role |
|---|---|
| `numpy` | Numerical computation |
| `pandas` | Data manipulation |
| `yfinance` | Historical price download |
| `scipy` | MLE optimization |
| `numba` | JIT-compiled GARCH likelihood |
| `statsmodels` | VAR model estimation |
| `cvxpy` | Convex portfolio optimization |
| `matplotlib` / `plotly` | Visualization |
| `streamlit` | Interactive dashboard |

---

## License

MIT — see `LICENSE` for details.
