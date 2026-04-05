# dashboard_save.py
import streamlit as st
import numpy as np
import pandas as pd 
import yfinance as yf
import matplotlib.pyplot as plt 
from data_to_cov_and_returns import (
    fit_bekk_gjr, MGARCH_GJR, compute_bekk_gjr_covariances
)
from cov_to_weights import (strat_all_in, realized_sharpe_from_portfolio_values)
import os

st.title("Backtest CAC40 - Sauvegarde automatique")

backtest = np.load("backtest_results.npz", allow_pickle=True)

tickers = backtest["tickers"]
n_dims = backtest["n_dims"]
y = backtest["y"]
#z=backtest["z"]
test_size = backtest["test_size"]
H_train = list(backtest["H_train"]) 
C = backtest["C"]
A = backtest["A"]
B = backtest["B"]
G = backtest["G"]
initial_capital=backtest["initial_capital"]
start_date=backtest["start_date"]
end_date=backtest["end_date"]

portefeuille_opt = backtest["portefeuille_opt"]
portefeuille_opt_frais = backtest["portefeuille_opt_frais"]
portefeuille_ref = backtest["portefeuille_ref"]

# 1️⃣ Choix des tickers
tickers_input = st.text_input(
    "Entrez les tickers séparés par une virgule",
    value=",".join(tickers)
)
tickers = [t.strip() for t in tickers_input.split(",")]
n_dims = len(tickers)

# 2️⃣ Paramètres du backtest
initial_capital = st.number_input("Capital initial", value=int(initial_capital), step=1000)
test_ratio = st.slider("Proportion des données pour test", 0.05, 0.3, int(test_size)/len(y))
start_date = st.date_input("Début des données", value=pd.to_datetime(str(start_date)))
end_date = st.date_input("Fin des données", value=pd.to_datetime(str(end_date)))

# 3️⃣ Bouton pour lancer le backtest
if st.button("Lancer le backtest"):

    with st.spinner("Téléchargement des données..."):
        data = yf.download(tickers, start='2010-01-01', end='2025-12-06', auto_adjust=True)["Close"]
        y = np.log(data).diff().dropna()
        y_matrix = y.values
        test_size = int(test_ratio * len(y_matrix))
        y_train = y_matrix[:-test_size]
        z = data.dropna()
        z_matrix=z.values

    st.success(f"Données téléchargées : {y_train.shape[0]} obs, {n_dims} tickers")

    # 4️⃣ Fit BEKK-GJR
    with st.spinner("Estimation du modèle BEKK-GJR..."):
        result = fit_bekk_gjr(y_train, n_dims)
        model = MGARCH_GJR.from_params(result.x, n_dims)
        H_train = list(compute_bekk_gjr_covariances(y_train, model.C, model.A, model.B, model.G))

    st.success("Modèle estimé")

    # 5️⃣ Backtest stratégie ALL-IN
    portefeuille_opt, portefeuille_opt_frais, portefeuille_ref = strat_all_in(
        n_dims=n_dims,
        test_size=test_size,
        y=y_matrix,
        H_train=H_train,
        A=model.A,
        B=model.B,
        C=model.C,
        G=model.G,
        allin=initial_capital
    )

    # 6️⃣ Sauvegarde dans un fichier NPZ
    save_path = "backtest_results.npz"
    np.savez_compressed(
        save_path,
        tickers=tickers,
        n_dims=n_dims,
        y=y_matrix,
        z=z_matrix,
        test_size=test_size,
        H_train=H_train,
        C=model.C,
        A=model.A,
        B=model.B,
        G=model.G,
        initial_capital=initial_capital,
        start_date=start_date,
        end_date=end_date,
        portefeuille_opt=portefeuille_opt,
        portefeuille_opt_frais=portefeuille_opt_frais,
        portefeuille_ref=portefeuille_ref
    )

    st.success(f"Résultats sauvegardés dans {save_path}")
    st.download_button("Télécharger le fichier NPZ", data=open(save_path, "rb"), file_name=save_path)

if st.button("Résultats backtest"):
    sharpe_opt = realized_sharpe_from_portfolio_values(portefeuille_opt)
    sharpe_opt_frais = realized_sharpe_from_portfolio_values(portefeuille_opt_frais)
    sharpe_ref = realized_sharpe_from_portfolio_values(portefeuille_ref)

    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(portefeuille_opt, label=f'Portefeuille optimisé sans frais, S={sharpe_opt:.2f}')
    ax.plot(portefeuille_opt_frais, label=f'Portefeuille optimisé avec frais, S={sharpe_opt_frais:.2f}')
    ax.plot(portefeuille_ref, label=f'Portefeuille 1/n, S={sharpe_ref:.2f}', linestyle='--')
    ax.set_title("Backtest : Portefeuille stratégie All-in")
    ax.set_xlabel("Jours")
    ax.set_ylabel("Valeur du portefeuille")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)