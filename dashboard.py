import streamlit as st
import numpy as np
import pandas as pd 
import yfinance as yf
import matplotlib.pyplot as plt 

from data_to_cov_and_returns import (
    fit_bekk_gjr, MGARCH_GJR, compute_bekk_gjr_covariances
)
from cov_to_weights import (
    strat_all_in, strat_regu, strat_only_regu, 
    realized_sharpe_from_portfolio_values
)

# -------------------------
# LANGUE
# -------------------------
if "lang" not in st.session_state:
    st.session_state.lang = "FR"

if st.sidebar.button("FR / EN"):
    st.session_state.lang = "EN" if st.session_state.lang == "FR" else "FR"

lang = st.session_state.lang

TEXT = {
    "FR": {
        "title": "Backtest GJR-BEKK-GARCH",
        "params": "Paramètres",
        "tickers": "Tickers (séparés par virgule)",
        "capital": "Capital initial",
        "ratio": "Proportion test",
        "start": "Date début",
        "end": "Date fin",
        "strategy": "Stratégie",
        "run": "Lancer analyse",
        "results": "Résultats",
        "download": "Télécharger résultats"
    },
    "EN": {
        "title": "GJR-BEKK-GARCH Backtest",
        "params": "Parameters",
        "tickers": "Tickers (comma separated)",
        "capital": "Initial capital",
        "ratio": "Test ratio",
        "start": "Start date",
        "end": "End date",
        "strategy": "Strategy",
        "run": "Run analysis",
        "results": "Results",
        "download": "Download results"
    }
}

st.title(TEXT[lang]["title"])

# -------------------------
# CHARGEMENT DONNÉES SAUVEGARDÉES
# -------------------------
backtest = np.load("backtest_results.npz", allow_pickle=True)

tickers = backtest["tickers"]
y = backtest["y"]
test_size = backtest["test_size"]
initial_capital = backtest["initial_capital"]
start_date = pd.to_datetime(str(backtest["start_date"]))
end_date = pd.to_datetime(str(backtest["end_date"]))

# -------------------------
# SIDEBAR PARAMÈTRES
# -------------------------
with st.sidebar:
    st.header(TEXT[lang]["params"])

    tickers_input = st.text_input(
        TEXT[lang]["tickers"],
        value=",".join(tickers)
    )
    tickers = [t.strip() for t in tickers_input.split(",")]

    initial_capital = st.number_input(
        TEXT[lang]["capital"], 
        value=int(initial_capital), 
        step=1000
    )

    test_ratio = st.slider(
        TEXT[lang]["ratio"], 
        0.05, 0.3, 
        int(test_size)/len(y)
    )

    start_date = st.date_input(TEXT[lang]["start"], value=start_date)
    end_date = st.date_input(TEXT[lang]["end"], value=end_date)

    strategie = st.selectbox(
        TEXT[lang]["strategy"],
        ["allin", "regu", "onlyregu"]
    )

    run_button = st.button(TEXT[lang]["run"])

# -------------------------
# MAIN
# -------------------------
st.info("Configure parameters in the sidebar and run the analysis." if lang=="EN" 
        else "Configurez les paramètres dans la barre latérale puis lancez l'analyse.")

if run_button:

    with st.spinner("Loading data..."):
        data = yf.download(tickers, start='2010-01-01', end='2025-12-06', auto_adjust=True)["Close"]
        y = np.log(data).diff().dropna()
        y_matrix = y.values

        test_size = int(test_ratio * len(y_matrix))
        y_train = y_matrix[:-test_size]

    st.success("Data loaded")

    # -------------------------
    # MODEL
    # -------------------------
    with st.spinner("Estimating model..."):
        result = fit_bekk_gjr(y_train, len(tickers))
        model = MGARCH_GJR.from_params(result.x, len(tickers))

        H_train = list(
            compute_bekk_gjr_covariances(
                y_train, model.C, model.A, model.B, model.G
            )
        )

    st.success("Model estimated")

    # -------------------------
    # BACKTEST
    # -------------------------
    if strategie == "allin":
        opt, opt_pf, opt_fees, ref = strat_all_in(
            len(tickers), test_size, y_matrix, H_train,
            model.A, model.B, model.C, model.G,
            initial_capital
        )

    elif strategie == "regu":
        opt, opt_pf, opt_fees, ref, _ = strat_regu(
            len(tickers), test_size, y_matrix, H_train,
            model.A, model.B, model.C, model.G,
            initial_capital/test_size
        )

    else:
        opt, opt_pf, opt_fees, ref, _ = strat_only_regu(
            len(tickers), test_size, y_matrix, H_train,
            model.A, model.B, model.C, model.G,
            initial_capital/test_size
        )

    # -------------------------
    # METRICS
    # -------------------------
    sharpe = realized_sharpe_from_portfolio_values(opt)

    col1, col2, col3 = st.columns(3)
    col1.metric("Sharpe", f"{sharpe:.2f}")
    col2.metric("Final value", f"{opt[-1]:.0f}")
    col3.metric("Return (%)", f"{(opt[-1]/initial_capital - 1)*100:.1f}")

    # -------------------------
    # PLOT
    # -------------------------
    st.header(TEXT[lang]["results"])

    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(opt, label="Optimized")
    ax.plot(opt_pf, label="Optimized + fees")
    ax.plot(opt_fees, label="Fees optimized")
    ax.plot(ref, label="1/n", linestyle='--')

    ax.set_xlabel("Time")
    ax.set_ylabel("Portfolio value")
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)

    # -------------------------
    # SAVE
    # -------------------------
    save_path = "backtest_results.npz"

    np.savez_compressed(
        save_path,
        tickers=tickers,
        y=y_matrix,
        test_size=test_size,
        initial_capital=initial_capital,
        opt=opt
    )

    st.download_button(
        TEXT[lang]["download"],
        data=open(save_path, "rb"),
        file_name=save_path
    )
