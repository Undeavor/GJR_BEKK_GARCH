import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
import cvxpy as cp

# --- Configuration Globale ---
RF_ANNUAL_LOG = np.log(1 + 0.03)
N_TRADING_DAYS = 252
RF_DAILY_LOG = RF_ANNUAL_LOG / N_TRADING_DAYS

def realized_sharpe_from_portfolio_values(portfolio_values, rf_log=RF_DAILY_LOG):
    portfolio_values = np.array(portfolio_values)
    if len(portfolio_values) < 2: return 0.0
    # Rendements arithmétiques pour un Sharpe standard
    returns = portfolio_values[1:] / portfolio_values[:-1] - 1
    excess = returns - (np.exp(rf_log) - 1)
    std = np.std(excess, ddof=1)
    if std == 0: return 0.0
    return np.sqrt(N_TRADING_DAYS) * np.mean(excess) / std

def predict_next_returns_from_returns(returns, lags=3):
    returns = np.array(returns)
    if len(returns) <= lags + 1:
        return np.zeros(returns.shape[1])
    try:
        model = VAR(returns)
        model_fit = model.fit(lags)
        forecast = model_fit.forecast(returns[-lags:], steps=1)
        return forecast[0]
    except:
        # Fallback stable sur moyenne mobile courte
        return np.mean(returns[-min(15, len(returns)):], axis=0)

def optimal_sharpe_weights(mu, Sigma, prev_weights=None, lambda_tc=0.0):
    n = len(mu)
    w = cp.Variable(n)
    
    # Ajout d'une petite ridge pour la stabilité numérique de Sigma
    Sigma_reg = Sigma + 1e-6 * np.eye(n)
    risk = cp.quad_form(w, Sigma_reg)

    turnover_penalty = 0
    if prev_weights is not None and lambda_tc > 0:
        turnover_penalty = lambda_tc * cp.norm1(w - prev_weights)

    # Objectif : Max rendement - Risque (gamma=0.5) - Pénalité Turnover
    objective = cp.Maximize(mu @ w - 0.5 * risk - turnover_penalty)

    constraints = [
        cp.sum(w) == 1,
        w >= 0,  # Long-Only pour éviter l'explosion des frais
        w <= 1
    ]

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.OSQP)

    if w.value is None:
        return np.ones(n) / n
    return np.array(w.value).flatten()

# --- Stratégie 1 : ALL-IN (Investissement unique au départ) ---
def strat_all_in(n_dims, test_size, y, H_train, A, B, C, G, initial_cash):
    history = [[initial_cash] * 4]
    w_prev_brut = np.ones(n_dims) / n_dims
    w_prev_opti = np.ones(n_dims) / n_dims
    w_ref = np.ones(n_dims) / n_dims
    frais_taux = 0.005

    for i in range(test_size):
        epsilon = y[i - test_size - 1]
        I_neg = np.diag((epsilon < 0))
        H_new = (C @ C.T + A.T @ np.outer(epsilon, epsilon) @ A +
                 B.T @ H_train[-1] @ B + G.T @ (np.outer(epsilon, epsilon) * I_neg) @ G)
        H_train.append(H_new)

        mu_next = predict_next_returns_from_returns(y[:i - test_size])
        
        # Calcul des poids
        w_sans = optimal_sharpe_weights(mu_next, H_new)
        w_opti = optimal_sharpe_weights(mu_next, H_new, prev_weights=w_prev_opti, lambda_tc=frais_taux)
        
        market_returns = np.exp(y[i - test_size])

        # 1. Sans frais
        v_sans = history[-1][0] * np.dot(w_sans, market_returns)
        # 2. Brut (Subit les frais de w_sans)
        f_brut = history[-1][1] * frais_taux * np.sum(np.abs(w_sans - w_prev_brut))
        v_brut = (history[-1][1] - f_brut) * np.dot(w_sans, market_returns)
        # 3. Optimisé (Turnover régularisé)
        f_opti = history[-1][2] * frais_taux * np.sum(np.abs(w_opti - w_prev_opti))
        v_opti = (history[-1][2] - f_opti) * np.dot(w_opti, market_returns)
        # 4. 1/n
        v_ref = history[-1][3] * np.dot(w_ref, market_returns)

        history.append([v_sans, v_brut, v_opti, v_ref])
        w_prev_brut, w_prev_opti = w_sans, w_opti

    return np.array(history)

# --- Stratégie 2 : REGU (Injections de cash quotidiennes) ---
def strat_regu(n_dims, test_size, y, H_train, A, B, C, G, regu_amount):
    history = [[0, 0, 0, 0, 0]] # [Sans, Brut, Opti, 1/n, Total Investi]
    w_prev_brut = np.zeros(n_dims)
    w_prev_opti = np.zeros(n_dims)
    w_ref = np.ones(n_dims) / n_dims
    frais_taux = 0.001 # Frais réduits pour injections régulières

    for i in range(test_size):
        epsilon = y[i - test_size - 1]
        I_neg = np.diag((epsilon < 0))
        H_new = (C @ C.T + A.T @ np.outer(epsilon, epsilon) @ A +
                 B.T @ H_train[-1] @ B + G.T @ (np.outer(epsilon, epsilon) * I_neg) @ G)
        H_train.append(H_new)

        mu_next = predict_next_returns_from_returns(y[:i - test_size])
        market_returns = np.exp(y[i - test_size])
        
        # Gestion du cash : On ajoute regu_amount avant de calculer les nouveaux poids
        total_investi = history[-1][4] + regu_amount
        
        # Optimisation
        w_sans = optimal_sharpe_weights(mu_next, H_new)
        w_opti = optimal_sharpe_weights(mu_next, H_new, prev_weights=w_prev_opti, lambda_tc=frais_taux)

        # Calcul des valeurs (Injection + Rendement - Frais)
        v_sans = (history[-1][0] + regu_amount) * np.dot(w_sans, market_returns)
        
        f_brut = (history[-1][1] + regu_amount) * frais_taux * np.sum(np.abs(w_sans - w_prev_brut))
        v_brut = (history[-1][1] + regu_amount - f_brut) * np.dot(w_sans, market_returns)
        
        f_opti = (history[-1][2] + regu_amount) * frais_taux * np.sum(np.abs(w_opti - w_prev_opti))
        v_opti = (history[-1][2] + regu_amount - f_opti) * np.dot(w_opti, market_returns)
        
        v_ref = (history[-1][3] + regu_amount) * np.dot(w_ref, market_returns)

        history.append([v_sans, v_brut, v_opti, v_ref, total_investi])
        w_prev_brut, w_prev_opti = w_sans, w_opti

    return np.array(history)

# --- Stratégie 3 : ONLYREGU (On ne rééquilibre que le nouveau cash) ---
def strat_only_regu(n_dims, test_size, y, H_train, A, B, C, G, regu_amount):
    history = [[0, 0, 0, 0, 0]]
    w_prev_brut = np.zeros(n_dims)
    w_prev_opti = np.zeros(n_dims)
    w_ref = np.ones(n_dims) / n_dims
    frais_taux = 0.001

    for i in range(test_size):
        epsilon = y[i - test_size - 1]
        H_new = (C @ C.T + A.T @ np.outer(epsilon, epsilon) @ A + B.T @ H_train[-1] @ B)
        H_train.append(H_new)

        mu_next = predict_next_returns_from_returns(y[:i - test_size])
        market_returns = np.exp(y[i - test_size])
        
        # Poids cible pour l'injection
        w_target = optimal_sharpe_weights(mu_next, H_new)
        
        # Poids combinés (Ancien portefeuille + Nouveau cash)
        # On ne force pas de rééquilibrage global, on dilue juste l'ancien w avec le nouveau
        def combine(w_old, val_old, w_new, val_new):
            if val_old + val_new == 0: return np.ones(n_dims)/n_dims
            return (w_old * val_old + w_new * val_new) / (val_old + val_new)

        w_sans = combine(w_prev_brut, history[-1][0], w_target, regu_amount)
        w_opti = combine(w_prev_opti, history[-1][2], w_target, regu_amount)

        v_sans = (history[-1][0] + regu_amount) * np.dot(w_sans, market_returns)
        # Frais uniquement sur l'achat du nouveau cash
        f_opti = regu_amount * frais_taux 
        v_opti = (history[-1][2] + regu_amount - f_opti) * np.dot(w_opti, market_returns)
        v_ref = (history[-1][3] + regu_amount) * np.dot(w_ref, market_returns)

        history.append([v_sans, v_opti, v_opti, v_ref, history[-1][4] + regu_amount])
        w_prev_brut, w_prev_opti = w_sans, w_opti

    return np.array(history)

# --- Main Execution ---
def main():
    data = np.load("cov_and_modelparams.npz")
    y, test_size = data["y"], data["test_size"]
    A, B, C, G = data["A"], data["B"], data["C"], data["G"]
    H_train, n_dims = list(data["H_train"]), data["n_dims"]

    choice = input("Stratégie (allin/regu/onlyregu) : ").strip().lower()
    initial_total = 10000

    if choice == "allin":
        res = strat_all_in(n_dims, test_size, y, H_train, A, B, C, G, initial_total)
        labels = ["Sans frais", "Avec frais (Brut)", "Avec frais (Optimisé)", "1/n"]
    elif choice in ["regu", "onlyregu"]:
        func = strat_regu if choice == "regu" else strat_only_regu
        res = func(n_dims, test_size, y, H_train, A, B, C, G, initial_total/test_size)
        labels = ["Sans frais", "Avec frais (Brut)", "Avec frais (Optimisé)", "1/n", "Investi"]
    else:
        return

    plt.figure(figsize=(12, 6))
    for j in range(len(labels)):
        s = realized_sharpe_from_portfolio_values(res[:, j]) if j < 4 else 0
        plt.plot(res[:, j], label=f"{labels[j]} (S={s:.2f})")
    
    plt.title(f"Backtest Portfolio - {choice.upper()}")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
