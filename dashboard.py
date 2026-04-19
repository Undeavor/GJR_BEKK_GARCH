import streamlit as st
import numpy as np
import pandas as pd 
import yfinance as yf
import matplotlib.pyplot as plt 
import os

from data_to_cov_and_returns import (
    fit_bekk_gjr, MGARCH_GJR, compute_bekk_gjr_covariances
)
from cov_to_weights import (
    strat_all_in, strat_regu, strat_only_regu, 
    realized_sharpe_from_portfolio_values
)
from yfinance_tickers import TICKERS_DICT, TICKER_NAMES

# -------------------------
# CONFIG PAGE
# -------------------------
st.set_page_config(layout="wide", page_title="GJR-BEKK Backtester")

# -------------------------
# LANGUE ET ÉTAT
# -------------------------
if "lang" not in st.session_state:
    st.session_state.lang = "FR"

with st.sidebar:
    if st.button("FR / EN"):
        st.session_state.lang = "EN" if st.session_state.lang == "FR" else "FR"

lang = st.session_state.lang

TEXT = {
    "FR": {
        "title": "Backtest GJR-BEKK-GARCH yfinance",
        "tickers": "Tickers (ex: AAPL, MSFT, GOOG)",
        "ticker_search_header": "Recherche de ticker",
        "ticker_search_input": "Cherchez une entreprise",
        "ticker_search_matches": "Correspondances",
        "ticker_search_none": "Aucune correspondance trouvée",
        "capital": "Capital initial",
        "ratio": "Proportion backtest",
        "start": "Début historique",
        "end": "Fin historique",
        "strategy": "Stratégie",
        "train": "1. Lancer l'entraînement",
        "backtest": "2. Afficher les résultats",
        "results": "Performance du portefeuille",
        "params": "Paramètres",
        "info": "Configurez les tickers dans la barre latérale, puis lancez l'entraînement.",
        "intro": "Analyse basée sur GJR-BEKK-GARCH. Frais : 0.5% | Sans risque : 3%/an",
        "expander": "Aide sur les stratégies",
        "data_loading": "Téléchargement...",
        "model_estimation": "Estimation GARCH (peut être long)...",
        "data_loaded": "Données prêtes",
        "model_done": "Modèle optimisé",
        "training_done": "Calculs terminés !",
        "days": "Jours de trading",
        "portfolio": "Valeur ($)",
        "invested": "Total investi",
        "opt_no_fees": "Optimisé (Sans frais)",
        "opt_with_fees": "Naïf avec frais (Orange)",
        "opt_fees": "Optimisé avec frais (Vert)",
        "benchmark": "Benchmark 1/n",
        "title_allin": "Stratégie : All-in",
        "title_regu": "Stratégie : REGU (DCA)",
        "title_onlyregu": "Stratégie : ONLYREGU"
    },
    "EN": {
        "title": "GJR-BEKK-GARCH Backtest",
        "tickers": "Tickers (e.g., AAPL, MSFT, GOOG)",
        "ticker_search_header": "Ticker Search",
        "ticker_search_input": "Search company",
        "ticker_search_matches": "Matches",
        "ticker_search_none": "No matches found",
        "capital": "Initial capital",
        "ratio": "Backtest ratio",
        "start": "Start date",
        "end": "End date",
        "strategy": "Strategy",
        "train": "1. Run Training",
        "backtest": "2. Show Results",
        "results": "Portfolio Performance",
        "params": "Settings",
        "info": "Configure tickers in the sidebar, then run training.",
        "intro": "GJR-BEKK-GARCH analysis. Fees: 0.5% | Risk-free: 3%/y",
        "expander": "Strategy help",
        "data_loading": "Downloading...",
        "model_estimation": "Estimating GARCH...",
        "data_loaded": "Data ready",
        "model_done": "Model estimated",
        "training_done": "Calculations done!",
        "days": "Trading Days",
        "portfolio": "Value ($)",
        "invested": "Total Invested",
        "opt_no_fees": "Optimized (No fees)",
        "opt_with_fees": "Naive with fees (Orange)",
        "opt_fees": "Optimized with fees (Green)",
        "benchmark": "1/n Benchmark",
        "title_allin": "Strategy: All-in",
        "title_regu": "Strategy: REGU (DCA)",
        "title_onlyregu": "Strategy: ONLYREGU"
    }
}

st.title(TEXT[lang]["title"])
st.markdown(TEXT[lang]["intro"])

# -------------------------
# CHARGEMENT DES DONNÉES PRÉCÉDENTES
# -------------------------
SAVE_FILE = "backtest_results.npz"
if os.path.exists(SAVE_FILE):
    backtest_data = np.load(SAVE_FILE, allow_pickle=True)
    stored_tickers = list(backtest_data["tickers"])
    stored_capital = float(backtest_data["initial_capital"])
    stored_ratio = float(backtest_data["test_size"] / len(backtest_data["y"]))
    stored_start = pd.to_datetime(str(backtest_data["start_date"]))
    stored_end = pd.to_datetime(str(backtest_data["end_date"]))
else:
    stored_tickers = ["AAPL", "MSFT", "GOOG"]
    stored_capital = 10000
    stored_ratio = 0.2
    stored_start = pd.to_datetime("2020-01-01")
    stored_end = pd.to_datetime("2024-01-01")

# -------------------------
# SIDEBAR
# -------------------------
with st.sidebar:
    st.header(TEXT[lang]["ticker_search_header"])
    search_input = st.text_input(TEXT[lang]["ticker_search_input"])
    if search_input:
        matches = [n for n in TICKER_NAMES if search_input.lower() in n.lower()][:5]
        if matches:
            selected = st.selectbox(TEXT[lang]["ticker_search_matches"], matches)
            st.code(f"{TICKERS_DICT[selected]}")
        else:
            st.write(TEXT[lang]["ticker_search_none"])
            
    st.header(TEXT[lang]["params"])
    tickers_input = st.text_input(TEXT[lang]["tickers"], value=",".join(stored_tickers))
    current_tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    
    initial_capital = st.number_input(TEXT[lang]["capital"], value=int(stored_capital), step=1000)
    test_ratio = st.slider(TEXT[lang]["ratio"], 0.05, 0.4, stored_ratio)
    start_date = st.date_input(TEXT[lang]["start"], value=stored_start)
    end_date = st.date_input(TEXT[lang]["end"], value=stored_end)
    
    strategie = st.selectbox(TEXT[lang]["strategy"], ["allin", "regu", "onlyregu"])
    
    st.markdown("---")
    train_button = st.button(TEXT[lang]["train"], use_container_width=True)
    backtest_button = st.button(TEXT[lang]["backtest"], use_container_width=True)

# -------------------------
# LOGIQUE D'ENTRAÎNEMENT
# -------------------------
if train_button:
    with st.spinner(TEXT[lang]["data_loading"]):
        df = yf.download(current_tickers, start=start_date, end=end_date, auto_adjust=True)["Close"]
        y_matrix = np.log(df).diff().dropna().values
        n_dims = len(current_tickers)
        t_size = int(test_ratio * len(y_matrix))
        y_train = y_matrix[:-t_size]

    with st.spinner(TEXT[lang]["model_estimation"]):
        result = fit_bekk_gjr(y_train, n_dims)
        model = MGARCH_GJR.from_params(result.x, n_dims)
        H_train = list(compute_bekk_gjr_covariances(y_train, model.C, model.A, model.B, model.G))

    with st.spinner("Running strategies..."):
        # Calcul All-in
        res_allin = strat_all_in(n_dims, t_size, y_matrix, H_train, model.A, model.B, model.C, model.G, initial_capital)
        # Calcul Regu
        res_regu = strat_regu(n_dims, t_size, y_matrix, H_train, model.A, model.B, model.C, model.G, initial_capital/t_size)
        # Calcul Only Regu
        res_only = strat_only_regu(n_dims, t_size, y_matrix, H_train, model.A, model.B, model.C, model.G, initial_capital/t_size)

    np.savez_compressed(
        SAVE_FILE,
        tickers=current_tickers, n_dims=n_dims, y=y_matrix, test_size=t_size,
        H_train=H_train, C=model.C, A=model.A, B=model.B, G=model.G,
        initial_capital=initial_capital, start_date=start_date, end_date=end_date,
        allin_opt=res_allin[0], allin_opt_pf=res_allin[1], allin_opt_af=res_allin[2], allin_ref=res_allin[3],
        regu_opt=res_regu[0], regu_opt_pf=res_regu[1], regu_opt_af=res_regu[2], regu_ref=res_regu[3],
        only_opt=res_only[0], only_opt_pf=res_only[1], only_opt_af=res_only[2], only_ref=res_only[3]
    )
    st.success(TEXT[lang]["training_done"])

# -------------------------
# AFFICHAGE RÉSULTATS
# -------------------------
if backtest_button:
    if not os.path.exists(SAVE_FILE):
        st.error("Aucun modèle entraîné. Cliquez sur 'Lancer l'entraînement' d'abord.")
    else:
        data = np.load(SAVE_FILE, allow_pickle=True)
        st.header(TEXT[lang]["results"])
        
        fig, ax = plt.subplots(figsize=(10, 5))
        
        if strategie == "allin":
            p_opt, p_brut, p_opti, p_ref = data["allin_opt"], data["allin_opt_pf"], data["allin_opt_af"], data["allin_ref"]
            st.subheader(TEXT[lang]["title_allin"])
        elif strategie == "regu":
            p_opt, p_brut, p_opti, p_ref = data["regu_opt"], data["regu_opt_pf"], data["regu_opt_af"], data["regu_ref"]
            st.subheader(TEXT[lang]["title_regu"])
        else:
            p_opt, p_brut, p_opti, p_ref = data["only_opt"], data["only_opt_pf"], data["only_opt_af"], data["only_ref"]
            st.subheader(TEXT[lang]["title_onlyregu"])

        # Plotting
        ax.plot(p_opt, label=f"{TEXT[lang]['opt_no_fees']} (S={realized_sharpe_from_portfolio_values(p_opt):.2f})")
        ax.plot(p_brut, label=f"{TEXT[lang]['opt_with_fees']} (S={realized_sharpe_from_portfolio_values(p_brut):.2f})", alpha=0.7)
        ax.plot(p_opti, label=f"{TEXT[lang]['opt_fees']} (S={realized_sharpe_from_portfolio_values(p_opti):.2f})", linewidth=2)
        ax.plot(p_ref, label=f"{TEXT[lang]['benchmark']} (S={realized_sharpe_from_portfolio_values(p_ref):.2f})", linestyle="--", color="black")
        
        if strategie != "allin":
            invested = [i * (float(data["initial_capital"])/len(p_opt)) for i in range(len(p_opt))]
            ax.fill_between(range(len(p_opt)), invested, color='gray', alpha=0.2, label=TEXT[lang]["invested"])

        ax.set_xlabel(TEXT[lang]["days"])
        ax.set_ylabel(TEXT[lang]["portfolio"])
        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig)
