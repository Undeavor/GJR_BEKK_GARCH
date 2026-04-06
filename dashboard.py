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
        "results": "Résultats backtest"
    },
    "EN": {
        "title": "Backtest GJR-BEKK-GARCH yfinance - Auto save",
        "tickers": "Enter tickers separated by commas",
        "capital": "Initial capital",
        "ratio": "Test data proportion",
        "start": "Start date",
        "end": "End date",
        "strategy": "Choose strategy",
        "train": "Run training",
        "backtest": "Run backtest",
        "results": "Backtest results"
    }
}

st.title(TEXT[lang]["title"])

# -------------------------
# INFOS
# -------------------------
st.info(
    "Configure parameters in the sidebar then run training and backtest."
    if lang == "EN"
    else "Configurez les paramètres dans la barre latérale puis lancez l'entraînement et le backtest."
)

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
    st.header("Paramètres")

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
with st.expander("Explication des stratégies"):
    st.write("""
    allin : investissement total initial  
    regu : investissement progressif  
    onlyregu : investissement régulier pur  
    """)

# -------------------------
# TRAIN
# -------------------------
if train_button:

    with st.spinner("Téléchargement des données..."):
        data = yf.download(tickers, start='2010-01-01', end='2025-12-06', auto_adjust=True)["Close"]
        y = np.log(data).diff().dropna()
        y_matrix = y.values
        test_size = int(test_ratio * len(y_matrix))
        y_train = y_matrix[:-test_size]
        z = data.dropna()
        z_matrix=z.values

    st.success(f"Données téléchargées : {y_train.shape[0]} obs, {n_dims} tickers")

    with st.spinner("Estimation du modèle BEKK-GJR..."):
        result = fit_bekk_gjr(y_train, n_dims)
        model = MGARCH_GJR.from_params(result.x, n_dims)
        H_train = list(compute_bekk_gjr_covariances(y_train, model.C, model.A, model.B, model.G))

    st.success("Modèle estimé")

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

    st.success("Backtest terminé")

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

        ax.plot(allin_opt, label=f'Portefeuille optimisé sans frais, S={sharpe_opt:.2f}')
        ax.plot(allin_opt_puis_frais, label=f'Portefeuille optimisé avec frais, S={sharpe_opt_puis_frais:.2f}')
        ax.plot(allin_opt_avec_frais, label=f'Portefeuille optimisant les frais, S={sharpe_opt_avec_frais:.2f}')
        ax.plot(allin_ref, label=f'Portefeuille 1/n, S={sharpe_ref:.2f}', linestyle='--')
        ax.set_title("Backtest : Portefeuille stratégie All-in")
    elif strategie == "regu":
        total = [i * initial_capital / test_size for i in range(len(regu_opt))]
        ax.plot(regu_opt, label='Portefeuille optimisé sans frais')
        ax.plot(regu_opt_puis_frais, label='Portefeuille optimisé avec frais')
        ax.plot(regu_opt_avec_frais, label='Portefeuille optimisant les frais')
        ax.plot(regu_ref, label='Portefeuille 1/n', linestyle='--')
        ax.plot(total, label='Argent investi')
        ax.set_title("Backtest : Portefeuille stratégie REGU")
    else:
        total = [i * initial_capital / test_size for i in range(len(regu_opt))]
        ax.plot(only_regu_opt, label='Portefeuille optimisé sans frais')
        ax.plot(only_regu_opt_puis_frais, label='Portefeuille optimisé avec frais')
        ax.plot(only_regu_opt_avec_frais, label='Portefeuille optimisant les frais')
        ax.plot(only_regu_ref, label='Portefeuille 1/n', linestyle='--')
        ax.plot(total, label='Argent investi')
        ax.set_title("Backtest : Portefeuille stratégie ONLYREGU")
        
    ax.set_xlabel("Jours")
    ax.set_ylabel("Valeur du portefeuille")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
