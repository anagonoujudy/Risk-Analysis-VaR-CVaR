Projet Finance Quantitative – Optimisation de Portefeuille et Gestion du Risque

======================================================================

📌 Contexte

Un opérateur souhaite investir 1 000 000 € aujourd’hui dans un portefeuille diversifié.

Composition du portefeuille :
- 5 actions internationales : TotalEnergies, Apple, LVMH, BNP Paribas, Vinci
- 1 obligation d’État française offrant un profil défensif et une protection partielle contre l’inflation

Horizon d’investissement : 1 an, avec l’objectif de revendre le portefeuille à un prix supérieur.

Problématique centrale :
Les marchés financiers sont volatils et peuvent subir des chocs importants.
L’opérateur doit donc évaluer et maîtriser le risque de perte potentielle associée à ce projet d’investissement.
Comment quantifier ce risque et optimiser la répartition du capital ?

======================================================================

📊 Description du projet

L'objectif de ce projet est de construire un portefeuille diversifié d'actifs financiers et d’évaluer son risque à l'aide de mesures de risque extrême : Value-at-Risk (VaR) et Conditional Value-at-Risk (CVaR). 

Le projet s’appuie sur des méthodes classiques de finance quantitative (Markowitz et CAPM) ainsi que sur des simulations Monte Carlo pour simuler des scénarios possibles de performance du portefeuille sur un horizon d’un an.

Le portefeuille étudié vise à profiter d'une diversification sectorielle tout en protégeant le capital contre l'inflation.

======================================================================

🛠️ Méthodologie

1️⃣ Traitement des données
- Importation des historiques de prix CSV.
- Fusion des données pour créer un DataFrame unique avec toutes les dates et les cours de chaque actif.
- Remplissage des valeurs manquantes par interpolation linéaire.
- Visualisation des prix et rendements.

2️⃣ Estimation des rendements et risques
- Calcul des rendements logarithmiques journaliers.
- Calcul des moyennes et volatilités annualisés.
- Estimation des paramètres alpha et beta via CAPM.

3️⃣ Construction du portefeuille optimal
- Optimisation du portefeuille via Markowitz (maximisation du Sharpe ratio).
- Deux versions : sans vente à découvert et avec vente à découvert.
- Visualisation de l'évolution historique des portefeuilles.

4️⃣ Simulation Monte Carlo
- Simulation de milliers de scénarios de rendements sur 1 jour, 10 jours et 1 an.

5️⃣ Évaluation du risque
- Calcul de la VaR et de la CVaR pour différents horizons et niveaux de confiance.
- Présentation des résultats sous forme de DataFrame et graphiques.

6️⃣ Optimisation centrée sur le risque
- Optimisation du portefeuille en minimisant la CVaR pour obtenir un profil risque-optimisé.

======================================================================

📈 Résultats attendus
- Poids optimaux pour chaque actif.
- VaR et CVaR pour différentes périodes et niveaux de confiance.
- Comparaison entre portefeuille Markowitz et portefeuille minimisant la CVaR.

======================================================================

🛠️ Technologies et librairies utilisées
- Python 3.x
- pandas, numpy, matplotlib, scipy

======================================================================

💻 Utilisation
1. Placer les fichiers CSV des historiques de prix dans le dossier 'data/'.
2. Exécuter le script Python principal.
3. Visualiser les graphiques et tableaux de résultats générés.

======================================================================

📁 Structure du projet recommandée

Finance-Quant-Portfolio/
│
├── data/                  # Fichiers CSV des historiques de prix
├── notebooks/             # Notebooks d’analyse exploratoire ou tests
├── src/                   # Scripts Python (calcul rendements, optimisation, Monte Carlo)
├── plots/                 # Graphiques générés
├── reports/               # Beamer
├── README.txt             # Ce fichier
└── requirements.txt       # Librairies Python nécessaires

======================================================================

Auteur :
Judith ANAGONOU– Ingénieur Financier 