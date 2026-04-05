Multi-Asset Portfolio Optimization with MGARCH-GJR & VAR

Ce projet propose un pipeline complet de finance quantitative pour la gestion de portefeuille. Il combine la modélisation de la volatilité conditionnelle par un modèle MGARCH BEKK-GJR (Multivariate GARCH avec effets de levier) et la prédiction des rendements via un modèle VAR (Vector Autoregression) pour optimiser l'allocation d'actifs sous contraintes de frais de transaction.
📋 Sommaire

    Architecture du Projet

    Fonctionnalités Clés

    Modèles Mathématiques

    Stratégies de Gestion

    Installation et Utilisation

    Visualisation (Dashboard)

🏗 Architecture du Projet

Le dépôt est articulé autour de trois scripts Python principaux :

    data_to_cov_and_returns.py : Moteur d'estimation. Télécharge les données historiques via l'API Yahoo Finance, estime les paramètres du modèle BEKK-GJR et génère les matrices de covariance conditionnelle.

    cov_to_weights.py : Moteur de stratégie. Contient les algorithmes d'optimisation du ratio de Sharpe, l'intégration des modèles de prédiction VAR et les backtests des différentes stratégies d'investissement.

    dashboard.py : Interface utilisateur. Application Streamlit permettant de piloter les backtests, de modifier les paramètres de marché et de visualiser les performances de manière interactive.

✨ Fonctionnalités Clés

    Modélisation Avancée de la Volatilité : Implémentation d'un modèle BEKK-GJR(1,1) pour capturer les corrélations dynamiques et l'asymétrie des chocs de volatilité (leverage effect).

    Prédiction de Rendements : Utilisation d'un modèle Vector Autoregressive (VAR) pour prévoir les rendements à court terme, intégrés dans l'optimisation.

    Optimisation de Sharpe Vectorisée : Calcul rapide des poids optimaux pour maximiser le rendement ajusté au risque.

    Gestion des Frais de Transaction : Intégration d'une pénalité L1​ dans la fonction d'optimisation pour limiter le turnover excessif du portefeuille.

    Persistance des Données : Sauvegarde automatique des résultats et des paramètres du modèle au format .npz pour éviter les calculs redondants.

⚗️ Modèles Mathématiques
1. BEKK-GJR (1,1)

La matrice de covariance conditionnelle Ht​ est modélisée comme :
Ht​=CC⊤+A⊤ϵt−1​ϵt−1⊤​A+B⊤Ht−1​B+G⊤(ϵt−1​ϵt−1⊤​⊙Iϵt−1​<0​)G

Où G capture l'impact asymétrique des rendements négatifs sur la volatilité future.
2. Optimisation de Portefeuille

Le script calcule les poids w maximisant :
S=σp​E[Rp​]−Rf​​

Une régularisation est appliquée pour minimiser λ∑∣wt​−wt−1​∣, réduisant ainsi l'impact des coûts de transaction réels.
📈 Stratégies de Gestion

Le projet supporte trois modes d'investissement :

    All-In : Investissement de la totalité du capital à t=0 avec rééquilibrage dynamique selon les prédictions quotidiennes.

    Regu (Investissement Progressif) : Injection régulière de capital (Dollar Cost Averaging) avec réoptimisation complète du portefeuille à chaque étape.

    OnlyRegu : Investissement progressif où seule la nouvelle tranche de capital est optimisée selon les conditions de marché, limitant ainsi les frais sur le stock existant.

🚀 Installation et Utilisation
Prérequis
Bash

pip install numpy pandas matplotlib yfinance statsmodels scipy numba streamlit

Exécution du Pipeline

    Estimation et Backtest (CLI) :
    Lancez l'entraînement du modèle et visualisez les résultats par console :
    Bash

    python cov_to_weights.py

    Interface Interactive (Streamlit) :
    Pour une analyse visuelle et un paramétrage dynamique :
    Bash

    streamlit run dashboard.py

📊 Visualisation (Dashboard)

L'interface Streamlit permet de :

    Modifier l'univers d'actifs (ex: AI.PA, BNP.PA, MC.PA).

    Ajuster le capital initial et la période de test.

    Comparer les performances face à un benchmark Equipondéré (1/n).

    Analyser l'impact des frais de transaction sur la valeur liquidative finale.

Note : Ce projet est à but éducatif et de recherche. Les performances passées ne préjugent pas des résultats futurs.
