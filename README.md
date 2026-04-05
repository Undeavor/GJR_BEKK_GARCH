# Multi-Asset Portfolio Optimization: MGARCH-GJR & VAR Approach

Ce projet implémente un pipeline complet d'optimisation de portefeuille basé sur la modélisation avancée de la volatilité multidimensionnelle. Il combine la puissance du modèle **MGARCH BEKK-GJR** pour la dynamique des covariances et un modèle **VAR** (Vector Autoregression) pour la prédiction des rendements.

---

## Architecture du Projet

Le dépôt est articulé autour de trois modules Python principaux :

1.  **`data_to_cov_and_returns.py`** : 
    * **Extraction** : Téléchargement automatique des données via `yfinance`.
    * **Modélisation** : Estimation des paramètres du modèle BEKK-GJR(1,1) par maximum de vraisemblance.
    * **Performance** : Utilisation de `numba` (compilation JIT) pour accélérer les calculs de vraisemblance et de simulation.
    * **Sortie** : Génération d'un fichier `.npz` contenant les paramètres du modèle et les matrices de covariance.

2.  **`cov_to_weights.py`** :
    * **Prédiction** : Utilisation d'un modèle `statsmodels.tsa.api.VAR` pour prévoir les rendements à $t+1$.
    * **Optimisation** : Calcul vectorisé des poids optimaux pour maximiser le ratio de Sharpe.
    * **Contraintes** : Gestion des frais de transaction via une pénalisation $L_1$ sur la variation des poids.

3.  **`dashboard.py`** :
    * **Interface** : Application interactive `Streamlit`.
    * **Pilotage** : Permet de configurer les tickers, le capital initial, et de comparer les stratégies en temps réel.

---

## Fondamentaux Théoriques

### 1. Modèle de Volatilité (BEKK-GJR)
Contrairement à un GARCH standard, le modèle BEKK-GJR capture les corrélations dynamiques et l'**effet de levier** (asymétrie) :
$$H_t = CC^\top + A^\top \epsilon_{t-1}\epsilon_{t-1}^\top A + B^\top H_{t-1} B + G^\top (\epsilon_{t-1}\epsilon_{t-1}^\top \odot I_{\epsilon_{t-1}<0}) G$$

### 2. Optimisation avec Frais
Le script intègre une régularisation pour limiter le "turnover" (rotation du portefeuille) :
$$\text{Poids Target} = w_{opt} - \lambda_{tc} \cdot (w_{opt} - w_{prev})$$
Ceci permet de s'assurer que les gains théoriques ne sont pas absorbés par les coûts de courtage.

---

## Stratégies de Backtest

| Stratégie | Description |
| :--- | :--- |
| **All-in** | Investissement total au jour 1, suivi d'un rééquilibrage dynamique quotidien. |
| **Regu** | Investissement progressif (DCA) avec réoptimisation de l'ensemble de l'encours. |
| **OnlyRegu** | Investissement progressif où seule la nouvelle tranche de capital est optimisée, figeant le reste pour minimiser les frais. |
| **1/n (Ref)** | Benchmark équipondéré servant de base de comparaison de performance. |

---

## 🚀 Installation et Utilisation

### Installation des dépendances
```bash
pip install numpy pandas matplotlib yfinance statsmodels scipy numba streamlit
```
## 🖥️ Utilisation
### Mode Interactif (Recommandé)

Lancez l'interface de contrôle pour configurer et visualiser vos backtests :
```Bash

streamlit run dashboard.py
```
### Mode Script

Pour entraîner le modèle et générer les paramètres de base :
```Bash

python data_to_cov_and_returns.py
```
Ensuite les utiliser pour générer les graphes associés aux stratégies :
```Bash

python cov_to_weights.py
```
