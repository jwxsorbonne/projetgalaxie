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

def run_benchmark():
    # Différentes tailles de systèmes (N) à tester
    Ns = [100,200,300,400,500,600,700,800,900,1000, 2000, 5000, 8000, 10000]
    dt = 0.1
    
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

    print("Démarrage du benchmark... Cela peut prendre quelques minutes.")
    
    for N in Ns:
        print(f"\n--- Test avec N = {N} étoiles ---")
        masses0, positions0, velocities0, colors = generate_galaxy(n_stars=N)
        
        # 1. Naïve (O(N^2) - très lent en Python pur, on ignore pour N > 1000)
        if N <= 1000:
            masses = masses0.copy()
            positions = positions0.copy()
            velocities = velocities0.copy()
            bodies = [Corps(masses[i], positions[i], velocities[i], colors[i]) for i in range(N)]
            sys_naive = NCorps(bodies)
            
            t0 = time.perf_counter()
            sys_naive.step(dt)
            t1 = time.perf_counter()
            
            results["Naïve (Boucles)"].append(t1 - t0)
            print(f"Naïve: {t1 - t0:.4f} s")
        else:
            results["Naïve (Boucles)"].append(None)
            print("Naïve: Ignoré (Trop long pour N > 1000)")
            
        # 2. Vectorisée (O(N^2) accéléré par C/NumPy, mais lourd en mémoire RAM)
        if N <= 5000:
            masses = masses0.copy()
            positions = positions0.copy()
            velocities = velocities0.copy()
            sys_vec = GalaxyVectorized(masses, positions, velocities)
            sys_vec.step(dt) # Initialisation 
            
            t0 = time.perf_counter()
            sys_vec.step(dt)
            t1 = time.perf_counter()
            
            results["Vectorisée (NumPy)"].append(t1 - t0)
            print(f"Vectorisée: {t1 - t0:.4f} s")
        else:
            results["Vectorisée (NumPy)"].append(None)

        # 3. Numba (Euler - O(N^2) compilé)
        masses = masses0.copy()
        positions = positions0.copy()
        velocities = velocities0.copy()
        sys_euler = simulation_numba.GalaxyNumba(masses, positions, velocities, use_parallel=True)
        sys_euler.step(dt) # Warm-up (Compilation JIT pour ne pas fausser le test)
        
        t0 = time.perf_counter()
        sys_euler.step(dt)
        t1 = time.perf_counter()
        results["Numba (Euler)"].append(t1 - t0)
        print(f"Numba (Euler): {t1 - t0:.4f} s")
        masses = masses0.copy()
        positions = positions0.copy()
        velocities = velocities0.copy()
        # 4. Numba (Verlet - 2 évaluations partielles de force)
        sys_verlet = verletnumba.GalaxyNumba(masses, positions, velocities, use_parallel=True)
        sys_verlet.step(dt) # Warm-up
        
        t0 = time.perf_counter()
        sys_verlet.step(dt)
        t1 = time.perf_counter()
        results["Numba (Verlet)"].append(t1 - t0)
        print(f"Numba (Verlet): {t1 - t0:.4f} s")
        masses = masses0.copy()
        positions = positions0.copy()
        velocities = velocities0.copy()
        # 5. Numba (RK4 - 4 évaluations de force par pas)
        sys_rk4 = rk4numba.GalaxyNumba(masses, positions, velocities, use_parallel=True)
        sys_rk4.step(dt) # Warm-up
        
        t0 = time.perf_counter()
        sys_rk4.step(dt)
        t1 = time.perf_counter()
        results["Numba (RK4)"].append(t1 - t0)
        print(f"Numba (RK4): {t1 - t0:.4f} s")
        masses = masses0.copy()
        positions = positions0.copy()
        velocities = velocities0.copy()
        # 6. Numba (Grille/Precond - force lointaine simplifiée)
        sys_pre = precond.GalaxyNumba(masses, positions, velocities)
        sys_pre.step(dt) # Warm-up
        
        t0 = time.perf_counter()
        sys_pre.step(dt)
        t1 = time.perf_counter()
        results["Numba (Grille/Precond)"].append(t1 - t0)
        print(f"Numba (Grille/Precond): {t1 - t0:.4f} s")
        masses = masses0.copy()
        positions = positions0.copy()
        velocities = velocities0.copy()
        # 7. Numba (2D Grille/Precond1)
        sys_newpre = precond1.GalaxyNumba(masses, positions, velocities)
        sys_newpre.step(dt) # Warm-up
        
        t0 = time.perf_counter()
        sys_newpre.step(dt)
        t1 = time.perf_counter()
        results["Numba (2D Grille/Precond1)"].append(t1 - t0)
        print(f"Numba (2D Grille/Precond1): {t1 - t0:.4f} s")
        masses = masses0.copy()
        positions = positions0.copy()
        velocities = velocities0.copy()
        # 8. Numba (2D Grille/Precond2)
        sys_improved = precond2.GalaxyNumba(masses, positions, velocities)
        sys_improved.step(dt) # Warm-up (Très important pour Numba)
        
        t0 = time.perf_counter()
        sys_improved.step(dt)
        t1 = time.perf_counter()
        results["Numba (2D Grille/Precond2)"].append(t1 - t0)
        print(f"Numba (2D Grille/Precond2): {t1 - t0:.4f} s")
    
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
      
    # === Création du tableau de résultats ===
    df = pd.DataFrame(results, index=Ns)
    df.index.name = "Nb Étoiles (N)"
    
    print("\n" + "="*70)
    print("RÉSUMÉ DES TEMPS D'EXÉCUTION (en secondes / pas)")
    print("="*70)
    pd.set_option('display.width', 200)        
    pd.set_option('display.max_columns', None) 
    print(df)
    
    # === Création du graphique ===
    plt.figure(figsize=(12, 7))
    
    # AJOUT : Ajout du marqueur 'p' (pentagone) pour la 8ème courbe
    markers = ['o', 's', '^', 'D', 'v', '*', 'X', 'p'] 
    for i, col in enumerate(df.columns):
        # On ne trace que les valeurs valides (on ignore le "None" de la version Naïve)
        mask = df[col].notna()
        plt.plot(df.index[mask], df[col][mask], marker=markers[i], linewidth=2, markersize=8, label=col)

    # Échelle logarithmique pour les X et les Y, standard pour observer des croissances O(N^2)
    plt.yscale('log')
    plt.xscale('log')
    
    plt.xlabel("Nombre d'étoiles (N)", fontsize=12)
    plt.ylabel("Temps d'exécution par pas de temps (secondes)", fontsize=12)
    plt.title("Comparaison des performances des algorithmes N-Corps", fontsize=15, fontweight='bold')
    
    # Ajout d'une grille pour mieux lire les échelles log
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig("benchmark_ncorps.png", dpi=150)
    print("\nLe graphique a été sauvegardé avec succès sous le nom 'benchmark_ncorps.png'")
    
if __name__ == "__main__":
    run_benchmark()