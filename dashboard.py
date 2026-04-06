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
# CONFIG PAGE
# -------------------------
st.set_page_config(layout="wide")

# -------------------------
# LANGUE
# -------------------------
if "lang" not in st.session_state:
    st.session_state.lang = "FR"

with st.sidebar:
    if st.button("FR / EN"):
        st.session_state.lang = "EN" if st.session_state.lang == "FR" else "FR"

lang = st.session_state.lang

TEXT = {
    "FR": {
        "title": "Backtest GJR-BEKK-GARCH yfinance - Sauvegarde auto",
        "tickers": "Entrez les tickers séparés par une virgule",
        "capital": "Capital initial",
        "ratio": "Proportion des données pour test",
        "start": "Début des données",
        "end": "Fin des données",
        "strategy": "Choisissez la stratégie",
        "train": "Lancer l'entrainement",
        "backtest": "Lancer le backtest",
        "results": "Résultats backtest",
        "params": "Paramètres",
        "info": "Configurez les paramètres dans la barre latérale puis lancez l'entraînement et le backtest.",
        "expander": "Explication des stratégies",
        "data_loading": "Téléchargement des données...",
        "model_estimation": "Estimation du modèle BEKK-GJR...",
        "data_loaded": "Données téléchargées",
        "model_done": "Modèle estimé",
        "backtest_done": "Backtest terminé",
        "days": "Jours",
        "portfolio": "Valeur du portefeuille",
        "invested": "Argent investi",
        "opt_no_fees": "Portefeuille optimisé sans frais",
        "opt_with_fees": "Portefeuille optimisé avec frais",
        "opt_fees": "Portefeuille optimisant les frais",
        "benchmark": "Portefeuille 1/n",
        "title_allin": "Backtest : stratégie All-in",
        "title_regu": "Backtest : stratégie REGU",
        "title_onlyregu": "Backtest : stratégie ONLYREGU"
    },
    "EN": {
        "title": "GJR-BEKK-GARCH Backtest yfinance - Auto save",
        "tickers": "Enter tickers separated by commas",
        "capital": "Initial capital",
        "ratio": "Test data proportion",
        "start": "Start date",
        "end": "End date",
        "strategy": "Choose strategy",
        "train": "Run training",
        "backtest": "Run backtest",
        "results": "Backtest results",
        "params": "Parameters",
        "info": "Configure parameters in the sidebar then run training and backtest.",
        "expander": "Strategy explanation",
        "data_loading": "Downloading data...",
        "model_estimation": "Estimating BEKK-GJR model...",
        "data_loaded": "Data loaded",
        "model_done": "Model estimated",
        "backtest_done": "Backtest completed",
        "days": "Days",
        "portfolio": "Portfolio value",
        "invested": "Invested capital",
        "opt_no_fees": "Optimized portfolio (no fees)",
        "opt_with_fees": "Optimized portfolio (with fees)",
        "opt_fees": "Fees-optimized portfolio",
        "benchmark": "Equal weight portfolio",
        "title_allin": "Backtest: All-in strategy",
        "title_regu": "Backtest: Regular investment strategy",
        "title_onlyregu": "Backtest: Only regular investment"
    }
}

st.title(TEXT[lang]["title"])

# -------------------------
# INFOS
# -------------------------
st.info(TEXT[lang]["info"])

# -------------------------
# CHARGEMENT BACKTEST
# -------------------------
backtest = np.load("backtest_results.npz", allow_pickle=True)

tickers = backtest["tickers"]
n_dims = backtest["n_dims"]
y = backtest["y"]
test_size = backtest["test_size"]
H_train = list(backtest["H_train"]) 
C = backtest["C"]
A = backtest["A"]
B = backtest["B"]
G = backtest["G"]
initial_capital=backtest["initial_capital"]
start_date=backtest["start_date"]
end_date=backtest["end_date"]

# ALL-IN
allin_opt = backtest["allin_opt"]
allin_opt_puis_frais = backtest["allin_opt_puis_frais"]
allin_opt_avec_frais = backtest["allin_opt_avec_frais"]
allin_ref = backtest["allin_ref"]

# REGU
regu_opt = backtest["regu_opt"]
regu_opt_puis_frais = backtest["regu_opt_puis_frais"]
regu_opt_avec_frais = backtest["regu_opt_avec_frais"]
regu_ref = backtest["regu_ref"]

# ONLY REGU 
only_regu_opt = backtest["only_regu_opt"]
only_regu_opt_puis_frais = backtest["only_regu_opt_puis_frais"]
only_regu_opt_avec_frais = backtest["only_regu_opt_avec_frais"]
only_regu_ref = backtest["only_regu_ref"]

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
    n_dims = len(tickers)

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

    start_date = st.date_input(
        TEXT[lang]["start"], 
        value=pd.to_datetime(str(start_date))
    )

    end_date = st.date_input(
        TEXT[lang]["end"], 
        value=pd.to_datetime(str(end_date))
    )

    strategie = st.selectbox(
        TEXT[lang]["strategy"],
        ["allin", "regu", "onlyregu"]
    )

    st.markdown("---")

    train_button = st.button(TEXT[lang]["train"])
    backtest_button = st.button(TEXT[lang]["backtest"])

# -------------------------
# EXPLICATION
# -------------------------
with st.expander(TEXT[lang]["expander"]):
    st.write("""
    allin : investissement total initial  
    regu : investissement progressif  
    onlyregu : investissement régulier pur  
    """)

# -------------------------
# TRAIN
# -------------------------
if train_button:

    with st.spinner(TEXT[lang]["data_loading"]):
        data = yf.download(tickers, start='2010-01-01', end='2025-12-06', auto_adjust=True)["Close"]
        y = np.log(data).diff().dropna()
        y_matrix = y.values
        test_size = int(test_ratio * len(y_matrix))
        y_train = y_matrix[:-test_size]
        z = data.dropna()
        z_matrix=z.values

    st.success(f"{TEXT[lang]['data_loaded']} : {y_train.shape[0]} obs, {n_dims} tickers")

    with st.spinner(TEXT[lang]["model_estimation"]):
        result = fit_bekk_gjr(y_train, n_dims)
        model = MGARCH_GJR.from_params(result.x, n_dims)
        H_train = list(compute_bekk_gjr_covariances(y_train, model.C, model.A, model.B, model.G))

    st.success(TEXT[lang]["model_done"])

    allin_opt, allin_opt_puis_frais, allin_opt_avec_frais, allin_ref = strat_all_in(
        n_dims, test_size, y_matrix, H_train,
        model.A, model.B, model.C, model.G, initial_capital
    )

    regu_opt, regu_opt_puis_frais, regu_opt_avec_frais, regu_ref, total = strat_regu(
        n_dims, test_size, y_matrix, H_train,
        model.A, model.B, model.C, model.G, initial_capital/test_size
    )

    only_regu_opt, only_regu_opt_puis_frais, only_regu_opt_avec_frais, only_regu_ref, total = strat_only_regu(
        n_dims, test_size, y_matrix, H_train,
        model.A, model.B, model.C, model.G, initial_capital/test_size
    )

    st.success(TEXT[lang]["backtest_done"])

# -------------------------
# BACKTEST DISPLAY
# -------------------------
if backtest_button:

    st.header(TEXT[lang]["results"])
    fig, ax = plt.subplots(figsize=(10,6))

    if strategie == "allin":
        sharpe_opt = realized_sharpe_from_portfolio_values(allin_opt)
        sharpe_opt_puis_frais = realized_sharpe_from_portfolio_values(allin_opt_puis_frais)
        sharpe_opt_avec_frais = realized_sharpe_from_portfolio_values(allin_opt_avec_frais)
        sharpe_ref = realized_sharpe_from_portfolio_values(allin_ref)

        ax.plot(allin_opt, label=f"{TEXT[lang]['opt_no_fees']}, S={sharpe_opt:.2f}")
        ax.plot(allin_opt_puis_frais, label=f"{TEXT[lang]['opt_with_fees']}, S={sharpe_opt_puis_frais:.2f}")
        ax.plot(allin_opt_avec_frais, label=f"{TEXT[lang]['opt_fees']}, S={sharpe_opt_avec_frais:.2f}")
        ax.plot(allin_ref, label=f"{TEXT[lang]['benchmark']}, S={sharpe_ref:.2f}", linestyle='--')
        ax.set_title(TEXT[lang]["title_allin"])
    elif strategie == "regu":
        total = [i * initial_capital / test_size for i in range(len(regu_opt))]
        ax.plot(regu_opt, label=TEXT[lang]["opt_no_fees"])
        ax.plot(regu_opt_puis_frais, label=TEXT[lang]["opt_with_fees"])
        ax.plot(regu_opt_avec_frais, label=TEXT[lang]["opt_fees"])
        ax.plot(regu_ref, label=TEXT[lang]["benchmark"], linestyle='--')
        ax.plot(total, label=TEXT[lang]["invested"])
        
        ax.set_title(TEXT[lang]["title_regu"])
    else:
        total = [i * initial_capital / test_size for i in range(len(regu_opt))]
        ax.plot(only_regu_opt, label=TEXT[lang]["opt_no_fees"])
        ax.plot(only_regu_opt_puis_frais, label=TEXT[lang]["opt_with_fees"])
        ax.plot(only_regu_opt_avec_frais, label=TEXT[lang]["opt_fees"])
        ax.plot(only_regu_ref, label=TEXT[lang]["benchmark"], linestyle='--')
        ax.plot(total, label=TEXT[lang]["invested"])
        
        ax.set_title(TEXT[lang]["title_onlyregu"])
        
    ax.set_xlabel(TEXT[lang]["days"])
    ax.set_ylabel(TEXT[lang]["portfolio"])    
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
