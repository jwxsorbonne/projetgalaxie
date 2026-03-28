import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from galaxy_generator import generate_galaxy

# --- Importation des différentes implémentations ---
from simulation_naive import Corps, NCorps
from simulation_vectorisee import GalaxyVectorized
import simulation_numba
import rk4numba
import verletnumba
import precond
import precond1  
import precond2  

def get_iterations(N):
    """
    Détermine le nombre d'itérations en fonction de la taille du système.
    Plus N est petit, plus on fait de boucles pour lisser les erreurs de mesure.
    """
    if N <= 300: return 10
    elif N <= 1000: return 5
    elif N <= 5000: return 3
    else: return 2

def run_benchmark():
    # Différentes tailles de systèmes (N) à tester
    Ns = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 5000, 8000, 10000]
    dt = 0.01
    
    # Dictionnaire pour stocker les résultats en secondes
    results = {
        "Naïve (Boucles)": [],
        "Vectorisée (NumPy)": [],
        "Numba (Euler)": [],
        "Numba (Verlet)": [],
        "Numba (RK4)": [],
        "Numba (Grille/Precond)": [],
        "Numba (2D Grille/Precond1)": [],
        "Numba (2D Grille/Precond2)": []  
    }

    print("Démarrage du benchmark optimisé... Cela peut prendre quelques minutes.")
    
    for N in Ns:
        print(f"\n--- Test avec N = {N} étoiles ---")
        masses0, positions0, velocities0, colors = generate_galaxy(n_stars=N)
        iterations = get_iterations(N)
        print(f"(Mesure moyenne sur {iterations} itérations)")
        
        # --- FACTORY PATTERN ---
        # On stocke des fonctions (lambda) qui recréent les objets à la demande.
        # L'appel de .copy() dans la lambda garantit que chaque instanciation utilise des données neuves.
        algorithms = {
            "Naïve (Boucles)": (
                lambda: NCorps([Corps(masses0[i], positions0[i].copy(), velocities0[i].copy(), colors[i]) for i in range(N)]), 
                1000 # N max supporté
            ),
            "Vectorisée (NumPy)": (
                lambda: GalaxyVectorized(masses0.copy(), positions0.copy(), velocities0.copy()), 
                5000
            ),
            "Numba (Euler)": (
                lambda: simulation_numba.GalaxyNumba(masses0.copy(), positions0.copy(), velocities0.copy(), use_parallel=True), 
                float('inf')
            ),
            "Numba (Verlet)": (
                lambda: verletnumba.GalaxyNumba(masses0.copy(), positions0.copy(), velocities0.copy(), use_parallel=True), 
                float('inf')
            ),
            "Numba (RK4)": (
                lambda: rk4numba.GalaxyNumba(masses0.copy(), positions0.copy(), velocities0.copy(), use_parallel=True), 
                float('inf')
            ),
            "Numba (Grille/Precond)": (
                lambda: precond.GalaxyNumba(masses0.copy(), positions0.copy(), velocities0.copy()), 
                float('inf')
            ),
            "Numba (2D Grille/Precond1)": (
                lambda: precond1.GalaxyNumba(masses0.copy(), positions0.copy(), velocities0.copy()), 
                float('inf')
            ),
            "Numba (2D Grille/Precond2)": (
                lambda: precond2.GalaxyNumba(masses0.copy(), positions0.copy(), velocities0.copy()), 
                float('inf')
            )
        }

        # --- DRY PRINCIPLE ---
        # Une seule boucle pour tester tous les algorithmes
        for name, (init_func, max_N) in algorithms.items():
            if N > max_N:
                results[name].append(None)
                if name == "Naïve (Boucles)":
                    print("Naïve: Ignoré (Trop long pour N > 1000)")
                continue
                
            # 1. WARM-UP (Compilation JIT)
            sys_warmup = init_func()
            sys_warmup.step(dt) 
            
            # 2. ÉVITER LA DÉRIVE D'ÉTAT (Reset State)
            # On recrée une nouvelle instance propre pour le vrai test
            sys_test = init_func()
            
            # 3. RÉSOUDRE LE BIAIS D'ÉCHANTILLONNAGE (Boucle + Moyenne)
            t0 = time.perf_counter()
            for _ in range(iterations):
                sys_test.step(dt)
            t1 = time.perf_counter()
            
            # Calcul du temps moyen par pas
            avg_time = (t1 - t0) / iterations
            results[name].append(avg_time)
            print(f"{name}: {avg_time:.6f} s")
    
    # === Sauvegarde et Affichage (Identique à ton code original) ===
    with open("results.txt", "w") as f:
        header = "N\t" + "\t".join(results.keys())
        f.write(header + "\n")
        for i, N in enumerate(Ns):
            row = [f"{N}"]
            for key in results:
                val = results[key][i]
                if val is None:
                    row.append("NA")
                else:
                    row.append(f"{val:.6f}")
            f.write("\t".join(row) + "\n")
      
    df = pd.DataFrame(results, index=Ns)
    df.index.name = "Nb Étoiles (N)"
    
    print("\n" + "="*70)
    print("RÉSUMÉ DES TEMPS D'EXÉCUTION (en secondes / pas)")
    print("="*70)
    pd.set_option('display.width', 200)        
    pd.set_option('display.max_columns', None) 
    print(df)
    
    plt.figure(figsize=(12, 7))
    markers = ['o', 's', '^', 'D', 'v', '*', 'X', 'p'] 
    for i, col in enumerate(df.columns):
        mask = df[col].notna()
        plt.plot(df.index[mask], df[col][mask], marker=markers[i], linewidth=2, markersize=8, label=col)

    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel("Nombre d'étoiles (N)", fontsize=12)
    plt.ylabel("Temps d'exécution par pas de temps (secondes)", fontsize=12)
    plt.title("Comparaison des performances des algorithmes N-Corps", fontsize=15, fontweight='bold')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig("benchmark_ncorps.png", dpi=150)
    print("\nLe graphique a été sauvegardé avec succès sous le nom 'benchmark_ncorps.png'")
    
if __name__ == "__main__":
    run_benchmark()
