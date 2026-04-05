import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR

RF_ANNUAL = 0.03
N_TRADING_DAYS = 252
RF_DAILY = (1 + RF_ANNUAL)**(1/N_TRADING_DAYS) - 1

def realized_sharpe_from_portfolio_values(portfolio_values, rf=RF_DAILY):
    portfolio_values = np.array(portfolio_values)
    returns = portfolio_values[1:] / portfolio_values[:-1] - 1
    excess_returns = returns - rf
    sharpe = np.mean(excess_returns) / np.std(excess_returns, ddof=1)
    return sharpe

def predict_next_returns_from_returns(returns, lags=3):
    model = VAR(returns)
    model_fit = model.fit(lags)
    forecast = model_fit.forecast(y=returns[-lags:], steps=1)
    return forecast[0] 

def adjust_weights(weights, min_weight=1e-3):
    weights = np.array(weights, dtype=float)
    
    # Forcer un minimum
    weights = np.maximum(weights, min_weight)
    
    # Renormaliser pour que la somme = 1
    weights /= np.sum(weights)
    
    return weights

def optimal_sharpe_weights(returns, cov_matrix, rf, prev_weights=None, lambda_tc=0.01): #je capte pas pq ca marche mieux que la version de chatgpt
    """
    Calcule les poids du portefeuille qui maximisent le Sharpe dans le cas général, de manière vectorisée.
    Paramètres :
    -----------
    returns : array_like
        Rendements attendus des actifs (R_i)
    cov_matrix : array_like
        Matrice de covariance des rendements (sigma_i^2 et cov(r_i,r_j))
    rf : float
        Taux sans risque
    Retour :
    -------
    weights : np.array
        Poids optimaux normalisés
    """
    returns = np.array(returns)
    cov_matrix = np.array(cov_matrix)

    # Matrice pour le calcul vectorisé de sum_term
    sigma_diag = np.diag(cov_matrix)
    n = len(returns)

    # Calcul des termes numerators / denominators pour tous i,j avec i != j
    cov_matrix_half = cov_matrix / 2
    numerator = sigma_diag[:, None] - cov_matrix_half
    denominator = sigma_diag - cov_matrix_half.T

    # Éviter division par zéro sur la diagonale
    np.fill_diagonal(denominator, 1)  # n’importe quelle valeur non nulle

    # somme des termes hors diagonale
    sum_term = np.sum(numerator / denominator * cov_matrix_half, axis=1) - np.diag(numerator / denominator * cov_matrix_half)

    # w' vectorisé
    w_prime = (rf-np.mean(returns)) / (sigma_diag + sum_term) #je capte pas pq c'est une moyenne des rendements, surtout pk ca marche mieux avec 

    if prev_weights is not None:
        penalty = lambda_tc * (w_prime - prev_weights)
        w_prime = w_prime - penalty

    # Normalisation
    weights = w_prime / np.sum(w_prime)

    return weights

#ALLIN
def strat_all_in(n_dims,test_size,y,H_train,A,B,C,G,allin):
    #en supposant qu'on ait tout en liquide au départ (ca change rien à l'implémentation)
    valeur_portefeuille=[[allin,allin,allin, allin]]  #sans frais, avec frais, 1/n
    w_ref = 1/n_dims * np.ones(n_dims) #strat 1/n
    frais_transaction = 0.01

    for i in range(test_size):  
        if i==0:
            w_prev = np.zeros(n_dims)
            w_prev_frais = np.zeros(n_dims)
        else:
            w_prev = w
            w_prev_frais = w_frais
        epsilon = y[i-test_size-1] # on prend epsilon_{t} pour calculer H_{t+1}
        I_neg = np.diag((epsilon < 0))  # Indicator for negative residuals

        H_new= (C @ C.T +
                A.T @ np.outer(epsilon, epsilon) @ A +
                B.T @ H_train[-1] @ B +
                G.T @ (np.outer(epsilon, epsilon) * I_neg) @ G)
        H_train.append(H_new)

        next_epsilon = predict_next_returns_from_returns(y[:i-test_size-1])
        #Calcul des poids pour le lendemain avec la prédiction de rendements et de covariance
        w=optimal_sharpe_weights(next_epsilon, H_new, RF_DAILY)
        w_frais=optimal_sharpe_weights(next_epsilon,H_new, RF_DAILY, w_prev_frais, frais_transaction)
        print(w,w_frais)

        #ATTENTION à bien prendre en compte que les rendements sont ici logarithmiques 
        #On calcule le rendement effectif du portefeuille à t+1 sans frais
        next_val = valeur_portefeuille[-1][0] * sum(w * np.exp(y[i-test_size]))

        #On calcule le rendement effectif du portefeuille à t+1 puis les frais totaux 
        frais_totaux = frais_transaction * np.sum(np.abs(w - w_prev))
        next_val_frais = valeur_portefeuille[-1][1] * sum(w * np.exp(y[i-test_size])) * (1-frais_totaux)

        #On calcule le rendement effectif du portefeuille à t+1 en même temps que les frais totaux 
        frais_totaux_opti = frais_transaction * np.sum(np.abs(w_frais - w_prev_frais))
        next_val_frais_opti = valeur_portefeuille[-1][2] * sum(w_frais * np.exp(y[i-test_size]))* (1-frais_totaux_opti)

        #On calcule pour le portefeuille de ref, en 1/n, qui est sans frais sauf au premier jour
        next_val_ref = valeur_portefeuille[-1][3] * sum(w_ref * np.exp(y[i-test_size]))

        #on update la valeur du portefeuille (les poids changent pas donc pas de frais de transaction)
        valeur_portefeuille.append([next_val,next_val_frais,next_val_frais_opti,next_val_ref])


    valeurs = np.array(valeur_portefeuille)  # shape (test_size, 3)
    portefeuille_opt = valeurs[:, 0]
    portefeuille_opt_puis_frais = valeurs[:, 1]
    portefeuille_opt_avec_frais = valeurs[:, 2]
    portefeuille_ref = valeurs[:, 3]

    return portefeuille_opt,portefeuille_opt_puis_frais,portefeuille_opt_avec_frais,portefeuille_ref

#REGU
def strat_regu(n_dims, test_size, y, H_train, A, B, C, G, regu):
    #en supposant qu'on ait tout en liquide au départ (ca change rien à l'implémentation)
    valeur_portefeuille=[[0,0,0,0,0]]  #sans frais, avec frais, 1/n
    w_ref = 1/n_dims * np.ones(n_dims) #strat 1/n
    frais_transaction = 0.001 #la double transaction nécessaire à vendre un actif A puis acheter un actif B est géré par le système de frais pénalisant ||w-w_prev||

    for i in range(test_size):  
        if i==0:
            w_prev = np.zeros(n_dims)
            w_prev_frais = np.zeros(n_dims)
            montants_prev_frais=np.zeros(n_dims)
            montants_prev_frais_opti=np.zeros(n_dims)
            montants_actuel_frais=np.zeros(n_dims)
            montants_actuel_frais_opti=np.zeros(n_dims)
        else:
            w_prev = w
            w_prev_frais = w_frais
            montants_prev_frais=w_prev*next_val_frais
            montants_prev_frais_opti=w_prev_frais*next_val_frais_opti
            montants_actuel_frais=w_prev*next_val_frais
            montants_actuel_frais_opti=w_prev_frais*next_val_frais_opti
        epsilon = y[i-test_size-1] # on prend epsilon_{t} pour calculer H_{t+1}
        I_neg = np.diag((epsilon < 0))  # Indicator for negative residuals

        H_new= (C @ C.T +
                A.T @ np.outer(epsilon, epsilon) @ A +
                B.T @ H_train[-1] @ B +
                G.T @ (np.outer(epsilon, epsilon) * I_neg) @ G)
        H_train.append(H_new)

        #on ajoute regu 
        regu_acc = regu 
        montants_inter_frais=np.zeros(n_dims)
        for k in range(len(montants_prev_frais)):
            if montants_prev_frais[k] < montants_actuel_frais[k] and regu_acc>=montants_actuel_frais[k]-montants_prev_frais[k]:
                montants_inter_frais[k]=montants_actuel_frais[k]
                regu_acc=regu_acc - (montants_actuel_frais[k]-montants_prev_frais[k])
            elif montants_prev_frais[k] < montants_actuel_frais[k]:
                montants_inter_frais[k]=montants_prev_frais[k]+regu_acc
                regu_acc=0
            else:
                montants_inter_frais[k]=montants_prev_frais[k]

        regu_acc = regu 
        montants_inter_frais_opti=np.zeros(n_dims)
        for k in range(len(montants_prev_frais_opti)):
            if montants_prev_frais_opti[k] < montants_actuel_frais_opti[k] and regu_acc>=montants_actuel_frais_opti[k]-montants_prev_frais_opti[k]:
                montants_inter_frais_opti[k]=montants_actuel_frais_opti[k]
                regu_acc=regu_acc - (montants_actuel_frais_opti[k]-montants_prev_frais_opti[k])
            elif montants_prev_frais_opti[k] < montants_actuel_frais_opti[k]:
                montants_inter_frais_opti[k]=montants_prev_frais_opti[k]+regu_acc
                regu_acc=0
            else:
                montants_inter_frais_opti[k]=montants_prev_frais_opti[k]

        w_inter_frais=montants_inter_frais/(valeur_portefeuille[-1][1]+regu)
        w_inter_frais_opti=montants_inter_frais_opti/(valeur_portefeuille[-1][2]+regu)
            
        next_epsilon = predict_next_returns_from_returns(y[:i-test_size-1])
        #Calcul des poids pour le lendemain
        w=optimal_sharpe_weights(next_epsilon, H_new, 0.03)
        w_frais=optimal_sharpe_weights(next_epsilon,H_new, 0.03)

        #On calcule le rendement effectif du portefeuille à t+1 sans frais
        next_val = (valeur_portefeuille[-1][0]+regu) * sum(w * np.exp(y[i-test_size]))

        #On calcule le rendement effectif du portefeuille à t+1 puis les frais totaux 
        frais_totaux = frais_transaction * np.sum(np.abs(w - w_inter_frais))
        next_val_frais = (valeur_portefeuille[-1][1] +regu*(1-frais_transaction)) * sum(w * np.exp(y[i-test_size])) * (1-frais_totaux)

        #On calcule le rendement effectif du portefeuille à t+1 en même temps que les frais totaux 
        frais_totaux_opti = frais_transaction * np.sum(np.abs(w_frais - w_inter_frais_opti))
        next_val_frais_opti = (valeur_portefeuille[-1][2] +regu*(1-frais_transaction)) * sum(w_frais * np.exp(y[i-test_size])) * (1-frais_totaux_opti)

        #On calcule pour le portefeuille de ref, en 1/n, qui est sans frais sauf au premier jour
        next_val_ref = (valeur_portefeuille[-1][3] +regu*(1-frais_transaction))* sum(w_ref * np.exp(y[i-test_size]))

        #Argent total investi 
        next_total = valeur_portefeuille[-1][4] + regu 

        #on update la valeur du portefeuille (les poids changent pas donc pas de frais de transaction)
        valeur_portefeuille.append([next_val,next_val_frais,next_val_frais_opti,next_val_ref, next_total])


    valeurs = np.array(valeur_portefeuille)  # shape (test_size, 4)
    portefeuille_opt = valeurs[:, 0]
    portefeuille_opt_puis_frais = valeurs[:, 1]
    portefeuille_opt_avec_frais = valeurs[:, 2]
    portefeuille_ref = valeurs[:, 3]
    portefeuille_total = valeurs[:,4]

    return portefeuille_opt, portefeuille_opt_puis_frais, portefeuille_opt_avec_frais, portefeuille_ref, portefeuille_total

def strat_only_regu(n_dims, test_size, y, H_train, A, B, C, G, regu):
    #en supposant qu'on ait tout en liquide au départ (ca change rien à l'implémentation)
    valeur_portefeuille=[[0,0,0,0,0]]  #sans frais, avec frais, 1/n
    w_ref = 1/n_dims * np.ones(n_dims) #strat 1/n
    frais_transaction = 0.001 #la double transaction nécessaire à vendre un actif A puis acheter un actif B est géré par le système de frais pénalisant ||w-w_prev||

    for i in range(test_size):  
        if i==0:
            w_prev = np.zeros(n_dims)
            w_prev_frais = np.zeros(n_dims)
            w_prev_frais_opti=np.zeros(n_dims)
        else:
            w_prev = w
            w_prev_frais = w_frais
            w_prev_frais_opti = w_frais_opti
        epsilon = y[i-test_size-1] # on prend epsilon_{t} pour calculer H_{t+1}
        I_neg = np.diag((epsilon < 0))  # Indicator for negative residuals

        H_new= (C @ C.T +
                A.T @ np.outer(epsilon, epsilon) @ A +
                B.T @ H_train[-1] @ B +
                G.T @ (np.outer(epsilon, epsilon) * I_neg) @ G)
        H_train.append(H_new)
            
        next_epsilon = predict_next_returns_from_returns(y[:i-test_size-1])
        #Calcul des poids pour le lendemain
        w_regu=optimal_sharpe_weights(next_epsilon, H_new, 0.03)
        w_regu_frais=optimal_sharpe_weights(next_epsilon,H_new, 0.03)

        w=(w_prev*valeur_portefeuille[-1][0]+w_regu*regu)/(valeur_portefeuille[-1][0]+regu)
        w_frais=(w_prev_frais*valeur_portefeuille[-1][1]+w_regu*regu)/(valeur_portefeuille[-1][1]+regu)
        w_frais_opti=(w_prev_frais_opti*valeur_portefeuille[-1][2]+w_regu_frais*regu)/(valeur_portefeuille[-1][2]+regu)

        #On calcule le rendement effectif du portefeuille à t+1 sans frais
        next_val = (valeur_portefeuille[-1][0]+regu) * sum(w * np.exp(y[i-test_size]))

        #On calcule le rendement effectif du portefeuille à t+1 puis les frais totaux 
        next_val_frais = (valeur_portefeuille[-1][1] +regu* (1-frais_transaction)) * sum(w_frais * np.exp(y[i-test_size])) 

        #On calcule le rendement effectif du portefeuille à t+1 en même temps que les frais totaux 
        next_val_frais_opti = (valeur_portefeuille[-1][2] +regu* (1-frais_transaction)) * sum(w_frais_opti * np.exp(y[i-test_size]))

        #On calcule pour le portefeuille de ref, en 1/n, qui est sans frais sauf au premier jour
        next_val_ref = (valeur_portefeuille[-1][3] +regu*(1-frais_transaction))* sum(w_ref * np.exp(y[i-test_size]))

        #Argent total investi 
        next_total = valeur_portefeuille[-1][4] + regu 

        #on update la valeur du portefeuille (les poids changent pas donc pas de frais de transaction)
        valeur_portefeuille.append([next_val,next_val_frais,next_val_frais_opti,next_val_ref, next_total])


    valeurs = np.array(valeur_portefeuille)  # shape (test_size, 4)
    portefeuille_opt = valeurs[:, 0]
    portefeuille_opt_puis_frais = valeurs[:, 1]
    portefeuille_opt_avec_frais = valeurs[:, 2]
    portefeuille_ref = valeurs[:, 3]
    portefeuille_total = valeurs[:,4]

    return portefeuille_opt, portefeuille_opt_puis_frais, portefeuille_opt_avec_frais, portefeuille_ref, portefeuille_total

def main():
    cov_and_modelparams = np.load("cov_and_modelparams.npz")

    n_dims=cov_and_modelparams["n_dims"]
    y=cov_and_modelparams["y"]
    test_size=cov_and_modelparams["test_size"]
    A=cov_and_modelparams["A"]
    B=cov_and_modelparams["B"]
    C=cov_and_modelparams["C"]
    G=cov_and_modelparams["G"]
    H_train=list(cov_and_modelparams["H_train"])

    strategie = input("Choisissez votre stratégie de portefeuille (allin/regu/onlyregu) : ").strip().lower()

    allin = 10000

    if strategie == "allin":
        portefeuille_opt, portefeuille_opt_puis_frais, portefeuille_opt_avec_frais, portefeuille_ref = strat_all_in(
            n_dims, test_size, y, H_train, A, B, C, G, allin
        )
        sharpe_opt = realized_sharpe_from_portfolio_values(portefeuille_opt)
        sharpe_opt_frais = realized_sharpe_from_portfolio_values(portefeuille_opt_puis_frais)
        sharpe_opt_avec_frais = realized_sharpe_from_portfolio_values(portefeuille_opt_avec_frais)
        sharpe_ref = realized_sharpe_from_portfolio_values(portefeuille_ref)

        # Plot strat ALL-IN avec Sharpe dans la légende
        plt.figure(figsize=(10,6))
        plt.plot(portefeuille_opt, label=f'Portefeuille sans frais, S={sharpe_opt:.2f}')
        plt.plot(portefeuille_opt_puis_frais, label=f'Portefeuille avec frais, S={sharpe_opt_frais:.2f}')
        plt.plot(portefeuille_opt_avec_frais, label=f'Portefeuille optimisant les frais, S={sharpe_opt_avec_frais:.2f}')
        plt.plot(portefeuille_ref, label=f'Portefeuille 1/n, S={sharpe_ref:.2f}', linestyle='--')
        plt.title("Backtest : Portefeuille stratégie All-in")
        plt.xlabel("Jours")
        plt.ylabel("Valeur du portefeuille")
        plt.legend()
        plt.grid(True)
        plt.show()
    elif strategie == "regu":
        portefeuille_opt, portefeuille_opt_puis_frais, portefeuille_opt_avec_frais, portefeuille_ref, total = strat_regu(
            n_dims, test_size, y, H_train, A, B, C, G, int(allin/test_size)
        )
        sharpe_opt = realized_sharpe_from_portfolio_values(portefeuille_opt)
        sharpe_opt_frais = realized_sharpe_from_portfolio_values(portefeuille_opt_puis_frais)
        sharpe_opt_avec_frais = realized_sharpe_from_portfolio_values(portefeuille_opt_avec_frais)
        sharpe_ref = realized_sharpe_from_portfolio_values(portefeuille_ref)

        # Plot strat REGU
        plt.figure(figsize=(10,6))
        plt.plot(portefeuille_opt, label=f'Portefeuille sans frais, S={sharpe_opt:.2f}')
        plt.plot(portefeuille_opt_puis_frais, label=f'Portefeuille avec frais, S={sharpe_opt_frais:.2f}')
        plt.plot(portefeuille_opt_avec_frais, label=f'Portefeuille en optimisant les frais, S={sharpe_opt_avec_frais:.2f}')
        plt.plot(portefeuille_ref, label=f'Portefeuille 1/n, S={sharpe_ref:.2f}', linestyle='--')
        plt.plot(total, label='Argent investi')
        plt.title("Backtest : Portefeuille stratégie REGU")
        plt.xlabel("Jours")
        plt.ylabel("Valeur du portefeuille")
        plt.legend()
        plt.grid(True)
        plt.show()
    elif strategie == "onlyregu":
        portefeuille_opt, portefeuille_opt_puis_frais, portefeuille_opt_avec_frais, portefeuille_ref, total = strat_only_regu(
            n_dims, test_size, y, H_train, A, B, C, G, int(allin/test_size)
        )
        sharpe_opt = realized_sharpe_from_portfolio_values(portefeuille_opt)
        sharpe_opt_frais = realized_sharpe_from_portfolio_values(portefeuille_opt_puis_frais)
        sharpe_opt_avec_frais = realized_sharpe_from_portfolio_values(portefeuille_opt_avec_frais)
        sharpe_ref = realized_sharpe_from_portfolio_values(portefeuille_ref)

        # Plot strat REGU
        plt.figure(figsize=(10,6))
        plt.plot(portefeuille_opt, label=f'Portefeuille sans frais, S={sharpe_opt:.2f}')
        plt.plot(portefeuille_opt_puis_frais, label=f'Portefeuille avec frais, S={sharpe_opt_frais:.2f}')
        plt.plot(portefeuille_opt_avec_frais, label=f'Portefeuille en optimisant les frais, S={sharpe_opt_avec_frais:.2f}')
        plt.plot(portefeuille_ref, label=f'Portefeuille 1/n, S={sharpe_ref:.2f}', linestyle='--')
        plt.plot(total, label='Argent investi')
        plt.title("Backtest : Portefeuille stratégie REGU")
        plt.xlabel("Jours")
        plt.ylabel("Valeur du portefeuille")
        plt.legend()
        plt.grid(True)
        plt.show()
    else:
        print("Stratégie non reconnue. Choisissez 'allin' ou 'regu'.")
        exit()




if __name__ == "__main__":
    main()
