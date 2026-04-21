# 📈 Optimisation de Portefeuille Multi-Actifs : MGARCH-GJR + VAR

Un pipeline complet d'optimisation quantitative de portefeuille combinant le modèle **GARCH multivarié avec effets de levier (BEKK-GJR)** pour la modélisation dynamique des covariances et un modèle **VAR (Vector Autoregression)** pour la prévision des rendements — appliqué aux actions du CAC 40 et à d'autres marchés.

---

## Table des matières

- [Vue d'ensemble](#vue-densemble)
- [Architecture](#architecture)
- [Fondements théoriques](#fondements-théoriques)
- [Stratégies de backtest](#stratégies-de-backtest)
- [Fichiers du projet](#fichiers-du-projet)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Dashboard interactif](#dashboard-interactif)
- [Fichiers générés](#fichiers-générés)
- [Dépendances](#dépendances)

---

## Vue d'ensemble

Ce projet implémente un pipeline de bout en bout pour la construction dynamique de portefeuille :

1. **Modélisation des covariances** via BEKK-GJR(1,1) — un modèle GARCH multivarié capturant la volatilité conditionnelle, les corrélations dynamiques entre actifs et l'asymétrie due à l'effet de levier (les chocs négatifs ont un impact plus fort que les positifs).
2. **Prévision des rendements** via un modèle VAR estimé à chaque pas de rééquilibrage.
3. **Optimisation du portefeuille** en maximisant le ratio de Sharpe sous contraintes d'allocation, avec pénalisation L1 optionnelle des coûts de transaction.
4. **Trois stratégies de backtest** simulant des scénarios d'investissement réalistes : all-in, DCA avec rééquilibrage total, et DCA avec rééquilibrage partiel.

---

## Architecture

```
raw_to_data.py              →  data.npz
        ↓
data_to_cov_and_returns.py  →  cov_and_modelparams.npz
        ↓
cov_to_weights.py           →  graphiques + métriques
        ↑
   dashboard.py  (interface Streamlit encapsulant le pipeline)
```

| Module | Rôle |
|---|---|
| `raw_to_data.py` | Téléchargement des données de prix via `yfinance`, sauvegarde des log-rendements |
| `data_to_cov_and_returns.py` | Estimation BEKK-GJR(1,1) par MV, calcul des covariances conditionnelles, export des paramètres |
| `cov_to_weights.py` | Backtests avec covariances prévues + rendements VAR, calcul des ratios de Sharpe |
| `dashboard.py` | Interface Streamlit pour configurer et visualiser toutes les stratégies |
| `yfinance_tickers.py` | Dictionnaire de tickers (CAC 40, grandes capitalisations US, ETFs, crypto, indices) |

---

## Fondements théoriques

### Modèle de volatilité : BEKK-GJR(1,1)

Le modèle capture la matrice de covariance conditionnelle jointe $H_t$ à chaque instant :

$$H_t = CC^\top + A^\top \varepsilon_{t-1}\varepsilon_{t-1}^\top A + B^\top H_{t-1} B + G^\top \left(\varepsilon_{t-1}\varepsilon_{t-1}^\top \odot \mathbf{1}_{\varepsilon_{t-1}<0}\right) G$$

- **$CC^\top$** — covariance inconditionnelle de base (semi-définie positive par construction)
- **$A^\top \varepsilon\varepsilon^\top A$** — terme ARCH : réaction aux innovations récentes au carré
- **$B^\top H_{t-1} B$** — terme GARCH : persistance de la volatilité passée
- **$G^\top (\ldots) G$** — terme d'asymétrie GJR : réponse amplifiée aux chocs négatifs (effet de levier)

Les paramètres sont estimés par **maximum de log-vraisemblance**, accéléré par la compilation JIT `numba`.

### Prévision des rendements : VAR(p)

À chaque étape de rééquilibrage, un modèle VAR d'ordre $p=3$ est estimé sur les log-rendements disponibles en échantillon. La prévision à un pas $\hat{\mu}_{t+1}$ est utilisée comme vecteur de rendements espérés pour l'optimisation.

### Optimisation du portefeuille

Étant donnés $\hat{\mu}_{t+1}$ et $H_t$, les poids $w$ sont obtenus en résolvant :

$$\max_w \quad \hat{\mu}_{t+1}^\top w - \frac{1}{2} w^\top H_t w - \lambda_{tc} \|w - w_{t-1}\|_1$$

sous les contraintes :
$$\sum_i w_i = 1, \quad -1 \leq w_i \leq 1 \quad \forall i$$

La pénalité L1 $\lambda_{tc}$ pénalise le turnover excessif, reflétant directement les coûts de courtage. Le problème est résolu via `cvxpy` avec le solveur OSQP.

### Métrique de performance : Ratio de Sharpe réalisé

$$\text{Sharpe} = \sqrt{252} \cdot \frac{\bar{r}_e}{\sigma_e}$$

où $r_e = r_t - r_f$ sont les rendements journaliers excédentaires par rapport au taux sans risque (3% annualisé).

---

## Stratégies de backtest

| Stratégie | Description | Frais de transaction |
|---|---|---|
| **All-In** | Capital total investi au jour 1, rééquilibrage quotidien avec poids optimaux | 0,5% par unité de turnover |
| **Regu (DCA)** | Versement périodique fixe, portefeuille entier rééquilibré à chaque période | 0,1% par unité de turnover |
| **OnlyRegu** | Versement périodique fixe, seul le nouveau cash est alloué de manière optimale ; l'encours existant n'est pas touché | 0,1% uniquement sur le nouveau cash |
| **1/n (Référence)** | Portefeuille équipondéré, sans rééquilibrage | Aucun |

Chaque stratégie produit quatre séries de valeurs de portefeuille :
- **Sans frais** — borne théorique haute, sans coûts de transaction
- **Avec frais (Brut)** — frais appliqués aux poids non contraints
- **Avec frais (Optimisé)** — frais appliqués aux poids pénalisés par le turnover
- **1/n** — benchmark équipondéré

---

## Fichiers du projet

```
.
├── raw_to_data.py                # Étape 1 : téléchargement des données de prix
├── data_to_cov_and_returns.py    # Étape 2 : estimation du modèle BEKK-GJR
├── cov_to_weights.py             # Étape 3 : backtests des stratégies
├── dashboard.py                  # Interface Streamlit
├── yfinance_tickers.py           # Dictionnaire de référence des tickers
├── requirements.txt              # Dépendances Python
├── data.npz                      # (généré) données de prix brutes
├── cov_and_modelparams.npz       # (généré) paramètres du modèle + covariances
└── backtest_results.npz          # (généré) valeurs de portefeuille des backtests
```

---

## Installation

**Python 3.9+ recommandé.**

```bash
git clone https://github.com/votre-utilisateur/votre-repo.git
cd votre-repo
pip install -r requirements.txt
```

Liste complète des dépendances :

```
numpy pandas matplotlib yfinance scipy statsmodels numba cvxpy streamlit plotly
```

> **Note sur `numba` :** La première exécution déclenchera la compilation JIT des fonctions de vraisemblance GARCH, ce qui peut prendre 30 à 60 secondes. Les exécutions suivantes sont nettement plus rapides.

---

## Utilisation

### Étape 1 — Télécharger les données de prix

```bash
python raw_to_data.py
```

Télécharge les cours de clôture journaliers de 18 tickers du CAC 40 depuis Yahoo Finance (2010–2025) et les sauvegarde dans `data.npz`. Modifiez la liste `syms` dans le script pour changer l'univers d'actifs.

### Étape 2 — Estimer le modèle BEKK-GJR

```bash
python data_to_cov_and_returns.py
```

Estime le modèle BEKK-GJR(1,1) par maximum de vraisemblance sur l'échantillon d'entraînement (85% des données). Sauvegarde les paramètres du modèle $(C, A, B, G)$, les matrices de covariance conditionnelles et la série complète des rendements dans `cov_and_modelparams.npz`.

> Cette étape est coûteuse en calcul. Durée estimée : **5 à 30 minutes** selon le nombre d'actifs et le matériel disponible.

### Étape 3 — Lancer le backtest

```bash
python cov_to_weights.py
```

Le script vous invite à choisir une stratégie :

```
Stratégie (allin/regu/onlyregu) :
```

Génère un graphique matplotlib des valeurs de portefeuille sur la période de test, avec les ratios de Sharpe réalisés affichés dans la légende.

---

## Dashboard interactif

Une application Streamlit hébergée est disponible à l'adresse :

```
https://projetetude28.streamlit.app/
```

Le dashboard permet de :
- Sélectionner des tickers depuis le dictionnaire complet de `yfinance_tickers.py` (CAC 40, actions US, ETFs, crypto…)
- Définir le capital initial et le montant de versement périodique (DCA)
- Choisir la stratégie de backtest
- Visualiser l'évolution du portefeuille et comparer les ratios de Sharpe en temps réel

Pour lancer le dashboard en local :

```bash
streamlit run dashboard.py
```

---

## Fichiers générés

| Fichier | Contenu |
|---|---|
| `data.npz` | `data` (prix de clôture), `n_dims` |
| `cov_and_modelparams.npz` | `y`, `test_size`, `A`, `B`, `C`, `G`, `H_train`, `n_dims` |
| `backtest_results.npz` | Tableaux de valeurs de portefeuille pour chaque stratégie et variante |

---

## Univers d'actifs supportés

`yfinance_tickers.py` fournit des dictionnaires de tickers prêts à l'emploi :

| Univers | Exemples |
|---|---|
| CAC 40 | `AI.PA`, `MC.PA`, `BNP.PA`, `SAN.PA`, … |
| Grandes cap. US | `AAPL`, `MSFT`, `NVDA`, `JPM`, … |
| ETFs US | `SPY`, `QQQ`, `TLT`, `GLD`, … |
| Europe | `ULVR.L`, `VOW3.DE`, `SAP.DE`, … |
| Cryptomonnaies | `BTC-USD`, `ETH-USD`, `SOL-USD`, … |
| Indices | `^GSPC`, `^FCHI`, `^GDAXI`, … |

---

## Dépendances

| Package | Rôle |
|---|---|
| `numpy` | Calcul numérique |
| `pandas` | Manipulation des données |
| `yfinance` | Téléchargement des données historiques |
| `scipy` | Optimisation MLE |
| `numba` | Compilation JIT de la vraisemblance GARCH |
| `statsmodels` | Estimation du modèle VAR |
| `cvxpy` | Optimisation convexe du portefeuille |
| `matplotlib` / `plotly` | Visualisation |
| `streamlit` | Dashboard interactif |
