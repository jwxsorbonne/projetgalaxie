"""
Script de démonstration des unités utilisées dans la simulation N-corps.

Ce script montre comment les unités ont été converties pour utiliser :
- Années-lumière (ly) pour les distances
- Masses solaires (M_sun) pour les masses
- Années pour le temps
- Années-lumière par an (ly/an) pour les vitesses
"""

import numpy as np

# === Constantes physiques ===
print("="*70)
print("UNITÉS ET CONSTANTES PHYSIQUES")
print("="*70)

# En unités SI
G_SI = 6.67430e-11  # m^3 kg^-1 s^-2
M_SUN_kg = 1.98847e30  # kg
LY_m = 9.4607e15  # m
YEAR_s = 3.15576e7  # s

print("\nUnités SI (Système International):")
print(f"  G = {G_SI:.5e} m³/(kg·s²)")
print(f"  1 masse solaire = {M_SUN_kg:.5e} kg")
print(f"  1 année-lumière = {LY_m:.5e} m")
print(f"  1 année = {YEAR_s:.5e} s")

# Conversion de G
# Pour que v = sqrt(G*M/r) soit cohérent:
# v [ly/an] = sqrt(G_new [ly³/(M_sun·an²)] * M [M_sun] / r [ly])
# Calculons G_new à partir de l'orbite terrestre
AU_m = 1.496e11  # m (Unité Astronomique)
v_earth_ms = 29780  # m/s
r_earth_ly = AU_m / LY_m
v_earth_lyan = (v_earth_ms / LY_m) * YEAR_s
M_sun = 1.0
G_new = v_earth_lyan**2 * r_earth_ly / M_sun

print("\nUnités de la simulation:")
print(f"  G = {G_new:.6e} ly³/(M_sun·an²)")
print(f"  Distance: années-lumière (ly)")
print(f"  Masse: masses solaires (M_sun)")
print(f"  Temps: années (an)")
print(f"  Vitesse: années-lumière par an (ly/an)")

# === Exemples de conversion ===
print("\n" + "="*70)
print("EXEMPLES DE CONVERSION")
print("="*70)

# Exemple 1: Distance Terre-Soleil
AU_ly = AU_m / LY_m

print(f"\n1. Distance Terre-Soleil:")
print(f"   {AU_m:.3e} m = {AU_ly:.6e} ly")

# Exemple 2: Vitesse orbitale de la Terre
v_earth_lyan_display = v_earth_lyan

print(f"\n2. Vitesse orbitale de la Terre:")
print(f"   {v_earth_ms} m/s = {v_earth_lyan_display:.6f} ly/an")

# Exemple 3: Vérification avec une orbite circulaire
print(f"\n3. Vérification: orbite circulaire autour du Soleil (M=1 M_sun)")
M_star = 1.0  # masse solaire
r_ly = AU_ly  # distance en ly
v_circular = np.sqrt(G_new * M_star / r_ly)
print(f"   Distance: {r_ly:.6e} ly")
print(f"   Vitesse circulaire calculée: {v_circular:.6f} ly/an")
print(f"   Vitesse de la Terre: {v_earth_lyan:.6f} ly/an")
print(f"   Différence: {abs(v_circular - v_earth_lyan)/v_earth_lyan * 100:.2f}%")

# === Ordres de grandeur galactiques ===
print("\n" + "="*70)
print("ORDRES DE GRANDEUR GALACTIQUES")
print("="*70)

# Trou noir supermassif (Sagittarius A*)
M_sgr_a = 4.15e6  # masses solaires
print(f"\n1. Sagittarius A* (trou noir au centre de la Voie Lactée):")
print(f"   Masse: {M_sgr_a:.2e} M_sun")

# Distance du Soleil au centre galactique
d_sol_centre_ly = 26000  # années-lumière
v_sol_orbital = np.sqrt(G_new * M_sgr_a / d_sol_centre_ly)
print(f"\n2. Orbite du Soleil autour du centre galactique:")
print(f"   Distance: {d_sol_centre_ly:.0f} ly")
print(f"   Vitesse orbitale (si seul trou noir): {v_sol_orbital:.4f} ly/an")
v_sol_real = 220000  # m/s vitesse réelle
v_sol_real_lyan = (v_sol_real / LY_m) * YEAR_s
print(f"   Vitesse réelle: {v_sol_real_lyan:.4f} ly/an")
print(f"   Note: La différence vient de la masse totale de la galaxie,")
print(f"         pas seulement celle du trou noir central")

# === Simulation d'exemple ===
print("\n" + "="*70)
print("PARAMÈTRES DE SIMULATION TYPIQUES")
print("="*70)

print("\nPour une petite galaxie simulée:")
M_bh = 1e6  # masses solaires
r_min = 0.001  # ly
r_max = 1.0  # ly

print(f"  Masse du trou noir central: {M_bh:.1e} M_sun")
print(f"  Rayon orbital minimum: {r_min} ly")
print(f"  Rayon orbital maximum: {r_max} ly")

v_min = np.sqrt(G_new * M_bh / r_max)
v_max = np.sqrt(G_new * M_bh / r_min)

print(f"  Vitesse minimale (à r_max): {v_min:.4f} ly/an")
print(f"  Vitesse maximale (à r_min): {v_max:.4f} ly/an")

dt_suggested = 0.001  # années
print(f"\n  Pas de temps suggéré: {dt_suggested} an ({dt_suggested * 365.25:.1f} jours)")

print("\n" + "="*70)
