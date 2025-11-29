import pandas as pd
from functools import reduce
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# IMPORTATION DES DONNEES
# Liste des fichiers CSV
files = ['data/TotalEnergiesSE Stock Price History.csv',
        'data/Apple Stock Price History.csv', 
        'data/Louis Vuitton Stock Price History.csv', 
        'data/BNP Paribas Stock Price History.csv', 
        'data/Vinci Stock Price History.csv', 
        'data/FR_OAT.csv',
        'data/Nasdaq 100 Historical Data.csv',
        'data/CAC 40 Historical Data.csv']
titres = ['TotalEnergies', 'Apple', 'LVMH', 'BNP', 'Vinci', 'OAT','Nasdaq','CAC 40']

# Initialisation d'une liste pour stocker les DataFrames
full_data = []

# Chargement de chaque fichier
for f, titre in zip(files, titres):
    mydata = pd.read_csv(f, parse_dates=['Date']) 
    mydata = mydata[['Date', 'Price']].rename(columns={'Price': titre})
    full_data.append(mydata)


# TRAITEMENT DES DONNEES
# Fusion de tous les fichiers sur la colonne Date en conservant toutes les dates
titres_df = reduce(lambda left, right: pd.merge(left, right, on='Date', how='outer'), full_data)
# Trier par date
titres_df = titres_df.sort_values('Date').reset_index(drop=True)
titres_df.head(10)

# Interpolation des valeurs manquantes avec la moyenne des valeurs précédente et suivante
titres_df.interpolate(method='linear', inplace=True)
print(titres_df.info())

# Conversion des colonnes Nasdaq et Cac 40 en float
titres_df['Nasdaq'] = titres_df['Nasdaq'].str.replace(",", "").astype(float)
titres_df['CAC 40'] = titres_df['CAC 40'].str.replace(",", "").astype(float)


#-------------------------------------------------------------------------------------
#Affichage des données
# Sélection des colonnes des actifs du portefeuille à constituer
actifs = ['TotalEnergies', 'Apple', 'LVMH', 'BNP', 'Vinci', 'OAT']
actifs_df = titres_df[actifs]

for actif in actifs:
    plt.figure(figsize=(12, 6))  # Nouveau graphique pour chaque actif
    plt.plot(titres_df['Date'], actifs_df[actif], label=actif)
    plt.title(f"Évolution du cours de {actif}")
    plt.xlabel("Date")
    plt.ylabel("Prix")
    plt.legend()
    plt.grid(True)
    plt.show()

# CALCUL DES RENDEMENTS LOGARITHMIQUES
returns_df = np.log(titres_df[titres] / titres_df[titres].shift(1)).dropna()
returns_df.insert(0, 'Date', titres_df['Date'].reset_index(drop=True))
print(returns_df.head(10))

# Affichage des rendements
plt.figure(figsize=(12, 6))  # Taille du graphique
returns_df_actif = returns_df[actifs]
for actif_rdt in returns_df_actif:
    plt.plot(returns_df['Date'], returns_df_actif[actif_rdt], label=actif_rdt)

plt.title("Évolution des rendements des actifs")
plt.xlabel("Date")
plt.ylabel("Rendements")
plt.legend()
plt.grid(True)
plt.show()

# Statistiques
returns_df.set_index('Date', inplace=True)

trading_days = 252

# DataFrame pour stocker les résultats
actif_stats = pd.DataFrame(index=actifs, columns=["Mean_annualized (%)", "Vol_annualized (%)"])

for actif in actifs:
    daily_returns = returns_df[actif]

    # Moyenne annualisée des rendements log
    mean_ann = daily_returns.mean() * trading_days

    # Volatilité annualisée
    vol_ann = daily_returns.std() * np.sqrt(trading_days)
    actif_stats.loc[actif] = [
        f"{mean_ann*100:.4f}", 
        f"{vol_ann*100:.4f}"
    ]


# Affichage
print(actif_stats)

# Calcul de la covariance annualisées
cov = returns_df.cov() * trading_days

# Calcul de alpha et beta pour chaque actif
# Liste des actifs et leur indice de référence
actifs_indices = {
    'TotalEnergies': 'CAC 40',
    'LVMH': 'CAC 40',
    'BNP': 'CAC 40',
    'Vinci': 'CAC 40',
    'Apple': 'Nasdaq',
}

# DataFrame pour stocker alpha et beta
alpha_beta_df = pd.DataFrame(columns=['Alpha', 'Beta'])

for actif, indice in actifs_indices.items():
    R_actif = returns_df[actif]
    R_indice = returns_df[indice]

    # Calcul du beta
    beta = np.cov(R_actif, R_indice)[0, 1] / np.var(R_indice)
    
    # Calcul de l'alpha
    alpha = R_actif.mean() - beta * R_indice.mean()
    
    alpha_beta_df.loc[actif] = [round(alpha, 4), round(beta, 4)]

print(alpha_beta_df)

# Construction du portefeuille optimal via Markowitz
daily_returns = returns_df[actifs]
mu = daily_returns.mean() * trading_days
cov = daily_returns.cov() * trading_days

# Fonction objectif : maximiser le Sharpe ratio (rendement / volatilité)

def sharpe_ratio(weights, mu, cov, rf=0.0):
    port_return = np.dot(weights, mu)
    port_vol = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
    return - (port_return - rf) / port_vol  # négatif pour minimiser

# Contraintes : somme des poids = 1, long only
# Sans possibilité de vente à découvert
constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
bounds = tuple((0,1) for _ in actifs)

# Initial guess
init_guess = np.array([1/len(actifs)]*len(actifs))
opt = minimize(sharpe_ratio, init_guess, args=(mu, cov), method='SLSQP', bounds=bounds, constraints=constraints)
weights_opt_Sdcvt = opt.x

print("Poids optimaux :")
for actif, w in zip(actifs, weights_opt_Sdcvt):
    print(f"{actif}: {w:.4f}")

# Avec possibilité de vente à découvert
# Bornes : autoriser vente à découvert (-100% à +100% par actif)
bounds = tuple((-1, 1) for _ in actifs)

# Initial guess
init_guess = np.array([1/len(actifs)]*len(actifs))

# Optimisation
opt = minimize(sharpe_ratio, init_guess, args=(mu, cov), method='SLSQP', bounds=bounds, constraints=constraints)
weights_opt_dcvt = opt.x

print("Poids optimaux :")
for actif, w in zip(actifs, weights_opt_dcvt):
    print(f"{actif}: {w:.4f}")


# Visualisation historique des deux portefeuilles

daily_returns = returns_df[actifs]
# Calcul du rendement quotidien du portefeuille
returns_df['Portfolio_dcvt'] = daily_returns.dot(weights_opt_dcvt)
returns_df['Portfolio_Sdcvt'] = daily_returns.dot(weights_opt_Sdcvt)

# Calcul de la valeur cumulée (base 100)
valeur_dcvt = (1 + returns_df['Portfolio_dcvt']).cumprod() * 100
valeur_Sdcvt = (1 + returns_df['Portfolio_Sdcvt']).cumprod() * 100

# Tracé des deux courbes
plt.figure(figsize=(12,6))
plt.plot(returns_df.index, valeur_dcvt, label='Portfolio avec vente à découvert')
plt.plot(returns_df.index, valeur_Sdcvt, label='Portfolio avec sans vente à découvert')
plt.xlabel('Date')
plt.ylabel('Valeur du portefeuille (base 100)')
plt.title('Comparaison des portefeuilles')
plt.legend()
plt.grid(True)
plt.show()

# Estimation de la Var et de le CVaR par la simulation Monte Carlo


def monte_carlo_var_cvar_multi_conf(returns, weights, horizon=1, confidence_levels=[0.05, 0.01], n_simulations=100000):
    """
    Calcule la VaR et la CVaR d'un portefeuille par simulation Monte Carlo
    pour plusieurs niveaux de confiance.

    Parameters
    ----------
    returns : DataFrame ou array
        Rendements quotidiens des actifs (colonnes = actifs)
    weights : array ou Series
        Poids du portefeuille
    horizon : int
        Nombre de jours pour l'horizon (1, 10, 252, etc.)
    confidence_levels : list
        Liste des niveaux de confiance (ex: [0.05, 0.01] pour 95% et 99%)
    n_simulations : int
        Nombre de tirages Monte Carlo

    Returns
    -------
    results : dict
        Dictionnaire contenant VaR et CVaR pour chaque niveau de confiance
    """
    port_rend = returns.dot(weights).values

    # Simulation Monte Carlo multi-journée
    simulated_horizon_returns = np.array([
        np.prod(1 + np.random.choice(port_rend, size=horizon, replace=True)) - 1
        for _ in range(n_simulations)
    ])

    results = {}
    for cl in confidence_levels:
        VaR = -np.percentile(simulated_horizon_returns, cl*100)
        CVaR = -simulated_horizon_returns[simulated_horizon_returns <= -VaR].mean()
        results[f"VaR_{int((1-cl)*100)}%"] = VaR
        results[f"CVaR_{int((1-cl)*100)}%"] = CVaR

    return results

# Exemple d'utilisation
results_1j = monte_carlo_var_cvar_multi_conf(daily_returns, weights_opt_dcvt, horizon=1, confidence_levels=[0.05, 0.01])
results_10j = monte_carlo_var_cvar_multi_conf(daily_returns, weights_opt_dcvt, horizon=10, confidence_levels=[0.05, 0.01])
results_1an = monte_carlo_var_cvar_multi_conf(daily_returns, weights_opt_dcvt, horizon=252, confidence_levels=[0.05, 0.01])


# Créer un DataFrame pour afficher proprement
def display_var_cvar(results_dict, horizon_label):
    VarEtCVar = pd.DataFrame(results_dict, index=[horizon_label]).T
    VarEtCVar.columns = [horizon_label]
    VarEtCVar[horizon_label] = VarEtCVar[horizon_label].round(4)
    return VarEtCVar

# Affichage joli
VarEtCVar_1j = display_var_cvar(results_1j, '1 jour')
VarEtCVar_10j = display_var_cvar(results_10j, '10 jours')
VarEtCVar_1an = display_var_cvar(results_1an, '1 an')

print("=== VaR et CVaR Monte Carlo ===\n")
print("\n",VarEtCVar_1j)
print("\n", VarEtCVar_10j)
print("\n", VarEtCVar_1an)

# OPTIMISATION DU PORTEFEUILLE EN MINIMISANT LA CVar
