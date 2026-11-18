import numpy as np
import matplotlib.pyplot as plt

# 1. Paramètres du modèle (Portefeuille fictif d'assurance auto/habitation)
np.random.seed(42)          # Pour que tes résultats soient reproductibles
n_simulations = 100_000     # Nombre d'années simulées

lambda_poisson = 50         # Fréquence : en moyenne 50 sinistres par an
mu_lognorm = 8.0            # Sévérité : paramètre de position (exp(8) ≈ 2981 €)
sigma_lognorm = 0.8         # Sévérité : volatilité des montants de sinistres

# 2. Simulation Monte-Carlo (Fréquence x Sévérité)
pertes_totales = np.zeros(n_simulations)

for i in range(n_simulations):
    # Combien de sinistres cette année ? (Loi de Poisson)
    nb_sinistres = np.random.poisson(lambda_poisson)
    
    # Quel coût pour chaque sinistre ? (Loi Log-normale)
    couts_sinistres = np.random.lognormal(mean=mu_lognorm, sigma=sigma_lognorm, size=nb_sinistres)
    
    # Perte totale de l'assureur sur l'année
    pertes_totales[i] = np.sum(couts_sinistres)

# 3. Calcul des métriques de risque
perte_moyenne = np.mean(pertes_totales)
var_995 = np.percentile(pertes_totales, 99.5)  # Quantile à 99,5 % (Solvabilité II)
tvar_995 = np.mean(pertes_totales[pertes_totales >= var_995])  # Expected Shortfall (moyenne des 0,5% pires cas)

# 4. Intervalle de confiance à 95% de la VaR par Bootstrap (1 000 rééchantillonnages)
vars_boot = [np.percentile(np.random.choice(pertes_totales, size=n_simulations, replace=True), 99.5) for _ in range(1000)]
ic_bas, ic_haut = np.percentile(vars_boot, [2.5, 97.5])

print("--- RÉSULTATS DU MODÈLE DE BASE ---")
print(f"Perte moyenne annuelle : {perte_moyenne:,.0f} €")
print(f"VaR 99,5 % : {var_995:,.0f} € (IC 95% : [{ic_bas:,.0f} € ; {ic_haut:,.0f} €])")
print(f"TVaR 99,5 % (Expected Shortfall) : {tvar_995:,.0f} €\n")

# 5. Analyse de sensibilité (Choc de +10 % sur la fréquence et +10 % sur la sévérité)
print("--- ANALYSE DE SENSIBILITÉ ---")
for nom, lam_test, mu_test in [
    ("Choc Fréquence (+10% sinistres)", lambda_poisson * 1.10, mu_lognorm),
    ("Choc Sévérité (+10% coût moyen)", lambda_poisson, mu_lognorm + np.log(1.10)),
]:
    pertes_choc = [np.sum(np.random.lognormal(mu_test, sigma_lognorm, np.random.poisson(lam_test))) for _ in range(n_simulations)]
    var_choc = np.percentile(pertes_choc, 99.5)
    hausse = (var_choc - var_995) / var_995 * 100
    print(f"{nom} -> Nouvelle VaR 99,5% : {var_choc:,.0f} € (+{hausse:.1f} %)")

# 6. Visualisation graphique
plt.figure(figsize=(10, 5))
plt.hist(pertes_totales, bins=100, color='steelblue', alpha=0.75, edgecolor='black')
plt.axvline(perte_moyenne, color='green', linestyle='--', linewidth=2, label=f'Moyenne : {perte_moyenne:,.0f} €')
plt.axvline(var_995, color='red', linestyle='-', linewidth=2, label=f'VaR 99,5% : {var_995:,.0f} €')
plt.title("Distribution des pertes annuelles simulées (100 000 scénarios Monte-Carlo)")
plt.xlabel("Perte annuelle totale (€)")
plt.ylabel("Fréquence")
plt.legend()
plt.tight_layout()
plt.savefig("distribution_pertes.png", dpi=300)
plt.show()