import streamlit as st
import numpy as np
import pandas as pd 
import yfinance as yf
import matplotlib.pyplot as plt 
import os

# Tes imports personnalisés
from data_to_cov_and_returns import (
    fit_bekk_gjr, MGARCH_GJR, compute_bekk_gjr_covariances
)
# Assure-toi que cov_to_weights contient bien tes nouvelles fonctions strat_...
from cov_to_weights import (
    strat_all_in, strat_regu, strat_only_regu, 
    realized_sharpe_from_portfolio_values
)
from yfinance_tickers import TICKERS_DICT, TICKER_NAMES

# -------------------------
# CONFIG PAGE & LANGUE
# -------------------------
st.set_page_config(layout="wide", page_title="GJR-BEKK-GARCH Backtester")

if "lang" not in st.session_state:
    st.session_state.lang = "FR"

with st.sidebar:
    if st.button("FR / EN"):
        st.session_state.lang = "EN" if st.session_state.lang == "FR" else "FR"

lang = st.session_state.lang

# (Dictionnaire TEXT inchangé par rapport à ta version, il fonctionne très bien)
TEXT = {
    "FR": {
        "title": "Backtest GJR-BEKK-GARCH - Stratégies Avancées",
        "tickers": "Entrez les tickers (séparés par virgule)",
        "ticker_search_header": "Recherche de ticker",
        "ticker_search_input": "Rechercher une entreprise",
        "ticker_search_matches": "Correspondances trouvées",
        "ticker_search_none": "Aucune correspondance",
        "capital": "Capital initial",
        "ratio": "Proportion test (Backtest)",
        "start": "Début des données",
        "end": "Fin des données",
        "strategy": "Choisissez la stratégie",
        "train": "Lancer l'entraînement et Backtest",
        "backtest": "Afficher le dernier graphique",
        "results": "Résultats du Backtest",
        "params": "Paramètres",
        "info": "1. Configurez les tickers. 2. Lancez l'entraînement. 3. Visualisez.",
        "intro": "Modèle GJR-BEKK-GARCH avec pénalité de turnover et contrainte Long-Only.",
        "expander": "Aide sur les stratégies",
        "data_loading": "Téléchargement yfinance...",
        "model_estimation": "Optimisation GARCH (patientez)...",
        "data_loaded": "Données récupérées",
        "model_done": "Modèle estimé",
        "training_done": "Calculs terminés !",
        "days": "Jours de trading",
        "portfolio": "Valeur du portefeuille ($)",
        "invested": "Capital investi",
        "opt_no_fees": "Sans frais (Théorique)",
        "opt_with_fees": "Brut avec frais (Naïf)",
        "opt_fees": "Optimisé avec frais (Turnover penalty)",
        "benchmark": "Benchmark 1/n",
        "title_allin": "Stratégie : All-in",
        "title_regu": "Stratégie : REGU (Injections)",
        "title_onlyregu": "Stratégie : ONLYREGU (Injections pures)"
    },
    "EN": {
        "title": "GJR-BEKK-GARCH Backtest - Advanced Strategies",
        "tickers": "Enter tickers (separated by commas)",
        "ticker_search_header": "Ticker Search",
        "ticker_search_input": "Search for a company",
        "ticker_search_matches": "Matches found",
        "ticker_search_none": "No matches",
        "capital": "Initial capital",
        "ratio": "Backtest ratio",
        "start": "Start date",
        "end": "End date",
        "strategy": "Choose strategy",
        "train": "Run Training & Backtest",
        "backtest": "Show Latest Chart",
        "results": "Backtest Results",
        "params": "Parameters",
        "info": "1. Set tickers. 2. Run training. 3. Visualize results.",
        "intro": "GJR-BEKK-GARCH model with turnover penalty and Long-Only constraint.",
        "expander": "Strategy help",
        "data_loading": "Downloading from yfinance...",
        "model_estimation": "GARCH Optimization (wait)...",
        "data_loaded": "Data received",
        "model_done": "Model estimated",
        "training_done": "Calculations complete!",
        "days": "Trading days",
        "portfolio": "Portfolio Value ($)",
        "invested": "Invested capital",
        "opt_no_fees": "No Fees (Theoretical)",
        "opt_with_fees": "Raw with fees (Naive)",
        "opt_fees": "Optimized with fees (Turnover penalty)",
        "benchmark": "1/n Benchmark",
        "title_allin": "Strategy: All-in",
        "title_regu": "Strategy: REGU (Injections)",
        "title_onlyregu": "Strategy: ONLYREGU (Pure injections)"
    }
}

st.title(TEXT[lang]["title"])
st.markdown(TEXT[lang]["intro"])
st.info(TEXT[lang]["info"])

# -------------------------
# SIDEBAR / PARAMÈTRES
# -------------------------
with st.sidebar:
    st.header(TEXT[lang]["ticker_search_header"])
    search_input = st.text_input(TEXT[lang]["ticker_search_input"])
    if search_input:
        matches = [name for name in TICKER_NAMES if search_input.lower() in name.lower()]
        if matches:
            selected = st.selectbox(TEXT[lang]["ticker_search_matches"], matches)
            st.code(TICKERS_DICT[selected])
    
    st.header(TEXT[lang]["params"])
    # Valeurs par défaut si le fichier n'existe pas encore
    tickers_val = "AAPL,MSFT,GOOG"
    cap_val = 10000
    ratio_val = 0.2

    # Tentative de chargement pour pré-remplir les champs
    if os.path.exists("backtest_results.npz"):
        try:
            bt_load = np.load("backtest_results.npz", allow_pickle=True)
            tickers_val = ",".join(bt_load["tickers"])
            cap_val = int(bt_load["initial_capital"])
            ratio_val = float(bt_load["test_size"] / len(bt_load["y"]))
        except: pass

    tickers_input = st.text_input(TEXT[lang]["tickers"], value=tickers_val)
    tickers = [t.strip().upper() for t in tickers_input.split(",")]
    
    initial_capital = st.number_input(TEXT[lang]["capital"], value=cap_val, step=1000)
    test_ratio = st.slider(TEXT[lang]["ratio"], 0.05, 0.4, ratio_val)
    start_date = st.date_input(TEXT[lang]["start"], value=pd.to_datetime("2018-01-01"))
    end_date = st.date_input(TEXT[lang]["end"], value=pd.to_datetime("2024-01-01"))
    
    strategie_type = st.selectbox(TEXT[lang]["strategy"], ["allin", "regu", "onlyregu"])
    
    st.markdown("---")
    train_button = st.button(TEXT[lang]["train"])
    backtest_button = st.button(TEXT[lang]["backtest"])

# -------------------------
# CŒUR DU CALCUL (TRAIN)
# -------------------------
if train_button:
    with st.spinner(TEXT[lang]["data_loading"]):
        # Téléchargement
        df = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)["Close"]
        y_df = np.log(df).diff().dropna()
        y_matrix = y_df.values
        n_dims = len(tickers)
        test_size = int(test_ratio * len(y_matrix))
        y_train = y_matrix[:-test_size]

    st.success(f"{TEXT[lang]['data_loaded']} ({len(y_matrix)} jours)")

    with st.spinner(TEXT[lang]["model_estimation"]):
        # Estimation BEKK
        result = fit_bekk_gjr(y_train, n_dims)
        model = MGARCH_GJR.from_params(result.x, n_dims)
        # Calcul de la suite de covariances sur le train
        H_train = list(compute_bekk_gjr_covariances(y_train, model.C, model.A, model.B, model.G))

    # Lancement du Backtest sélectionné
    if strategie_type == "allin":
        res = strat_all_in(n_dims, test_size, y_matrix, H_train, model.A, model.B, model.C, model.G, initial_capital)
    elif strategie_type == "regu":
        res = strat_regu(n_dims, test_size, y_matrix, H_train, model.A, model.B, model.C, model.G, initial_capital/test_size)
    else:
        res = strat_only_regu(n_dims, test_size, y_matrix, H_train, model.A, model.B, model.C, model.G, initial_capital/test_size)

    # Sauvegarde
    np.savez_compressed(
        "backtest_results.npz",
        tickers=tickers, n_dims=n_dims, y=y_matrix, test_size=test_size,
        results_matrix=res, initial_capital=initial_capital,
        strat_name=strategie_type, start_date=str(start_date), end_date=str(end_date)
    )
    st.success(TEXT[lang]["training_done"])

# -------------------------
# AFFICHAGE DU GRAPHIQUE
# -------------------------
if backtest_button or train_button:
    if os.path.exists("backtest_results.npz"):
        data = np.load("backtest_results.npz", allow_pickle=True)
        res = data["results_matrix"]
        strat_name = str(data["strat_name"])
        
        st.header(f"{TEXT[lang]['results']} - {strat_name.upper()}")
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Définition des labels selon la stratégie
        if strat_name == "allin":
            labels = [TEXT[lang]["opt_no_fees"], TEXT[lang]["opt_with_fees"], TEXT[lang]["opt_fees"], TEXT[lang]["benchmark"]]
        else:
            labels = [TEXT[lang]["opt_no_fees"], TEXT[lang]["opt_with_fees"], TEXT[lang]["opt_fees"], TEXT[lang]["benchmark"], TEXT[lang]["invested"]]

        # Plot des courbes
        for j in range(len(labels)):
            # Calcul du Sharpe pour la légende
            # Le Sharpe n'a pas de sens sur la courbe "Argent investi" (j=4)
            if j < 4:
                s_val = realized_sharpe_from_portfolio_values(res[:, j])
                ax.plot(res[:, j], label=f"{labels[j]} (S={s_val:.2f})")
            else:
                ax.plot(res[:, j], label=labels[j], linestyle=':', color='gray')

        ax.set_xlabel(TEXT[lang]["days"])
        ax.set_ylabel(TEXT[lang]["portfolio"])
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    else:
        st.warning("Aucun résultat trouvé. Veuillez d'abord lancer l'entraînement.")
