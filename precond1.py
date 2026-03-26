import numpy as np
import time
from numba import njit, prange
from galaxy_generator import generate_galaxy

try:
    from visualizer3d_vbo import Visualizer3D
    print("Using VBO Visualizer")
except ImportError:
    from visualizer3d_sans_vbo import Visualizer3D
    print("Using Standard Visualizer (No VBO)")

G = 1.560339e-13 

@njit(parallel=True, fastmath=True)
def compute_forces_grid_parallel(positions, masses, bounds_max, grid_res):
    """
   Calcule les accélérations gravitationnelles pour un système de particules
   en utilisant une méthode de grille 2D combinée avec une structure CSR.

   Paramètres
   ----------
   positions : ndarray (n, 3)
       Positions des particules dans l'espace.
   masses : ndarray (n,)
       Masses des particules.
   bounds_max : float
       Limite spatiale du domaine ([-bounds_max, bounds_max]).
   grid_res : int
       Résolution de la grille (nombre de cellules par axe).

   Principe
   --------
   1. Discrétisation de l'espace en une grille 2D.
   2. Regroupement des particules par cellule (structure CSR).
   3. Calcul du centre de masse pour chaque cellule.
   4. Approximation des interactions lointaines via le centre de masse.
   5. Calcul exact des interactions proches.

   Retour
   ------
   acc : ndarray (n, 3)
       Accélérations gravitationnelles pour chaque particule.
   Remarque
   --------
   L'utilisation de CSR permet un accès mémoire contigu,
   améliorant significativement les performances (cache CPU).
   """
    n = len(positions)
    acc = np.zeros((n, 3), dtype=np.float32)

    num_cells = grid_res * grid_res
    
    cell_counts = np.zeros(num_cells, dtype=np.int32) 
    particle_cells = np.zeros(n, dtype=np.int32)     
    
    cell_masses = np.zeros(num_cells, dtype=np.float32)
    cell_coms = np.zeros((num_cells, 3), dtype=np.float32)

    box_size = 2.0 * bounds_max
    cell_size = box_size / grid_res
    R_threshold = np.sqrt(cell_size**2 + cell_size**2)

    for i in range(n):
        px, py, pz = positions[i]
        
        px_c = max(min(px, bounds_max - 1e-5), -bounds_max)
        py_c = max(min(py, bounds_max - 1e-5), -bounds_max)

        cx = int((px_c + bounds_max) / cell_size)
        cy = int((py_c + bounds_max) / cell_size)
        c_idx = cx * grid_res + cy

        particle_cells[i] = c_idx
        cell_counts[c_idx] += 1

        cell_masses[c_idx] += masses[i]
        cell_coms[c_idx, 0] += masses[i] * px
        cell_coms[c_idx, 1] += masses[i] * py
        cell_coms[c_idx, 2] += masses[i] * pz

    row_ptr = np.zeros(num_cells + 1, dtype=np.int32)
    cumsum = 0
    for c in range(num_cells):
        row_ptr[c] = cumsum
        cumsum += cell_counts[c]
    row_ptr[num_cells] = cumsum

    col_indices = np.zeros(n, dtype=np.int32)
    cell_counters = np.zeros(num_cells, dtype=np.int32) 
    
    for i in range(n):
        c_idx = particle_cells[i]
        local_idx = cell_counters[c_idx]
        col_indices[row_ptr[c_idx] + local_idx] = i
        cell_counters[c_idx] += 1

    for c in range(num_cells):
        if cell_masses[c] > 0:
            cell_coms[c, 0] /= cell_masses[c]
            cell_coms[c, 1] /= cell_masses[c]
            cell_coms[c, 2] /= cell_masses[c]

    for i in prange(n):
        px, py, pz = positions[i]
        acc_x, acc_y, acc_z = 0.0, 0.0, 0.0

        for gx in range(grid_res):
            for gy in range(grid_res):
                c_idx = gx * grid_res + gy
                
                if cell_masses[c_idx] == 0:
                    continue
                
                dx_com = cell_coms[c_idx, 0] - px
                dy_com = cell_coms[c_idx, 1] - py
                dz_com = cell_coms[c_idx, 2] - pz
                dist_sq_com = dx_com**2 + dy_com**2 + dz_com**2
                dist_com = np.sqrt(dist_sq_com)
                
                if 0.5 * dist_com > R_threshold:
                    if dist_sq_com > 1e-10:
                        f = (G * cell_masses[c_idx]) / (dist_sq_com * dist_com)
                        acc_x += f * dx_com
                        acc_y += f * dy_com
                        acc_z += f * dz_com
                else:
                    start = row_ptr[c_idx]
                    end = row_ptr[c_idx + 1]
                    
                    for p in range(start, end):
                        j = col_indices[p]
                        if i != j:
                            dx = positions[j, 0] - px
                            dy = positions[j, 1] - py
                            dz = positions[j, 2] - pz
                            dist_sq = dx*dx + dy*dy + dz*dz
                            if dist_sq > 1e-10:
                                dist = np.sqrt(dist_sq)
                                f = (G * masses[j]) / (dist_sq * dist)
                                acc_x += f * dx
                                acc_y += f * dy
                                acc_z += f * dz
                                
        acc[i, 0] = acc_x
        acc[i, 1] = acc_y
        acc[i, 2] = acc_z

    return acc


class GalaxyNumba:
    def __init__(self, masses, positions, velocities):
        self.masses = np.array(masses, dtype=np.float32)
        self.positions = np.array(positions, dtype=np.float32)
        self.velocities = np.array(velocities, dtype=np.float32)
        
        self.bounds_max = 5.0  
        self.grid_res = 20    
        
        self.acc = compute_forces_grid_parallel(self.positions, self.masses, self.bounds_max, self.grid_res)
        print("Compilation done.")

    def step(self, dt):
        """Applique la méthode de Verlet vectorisée pour mettre à jour"""
        self.positions += self.velocities * dt + 0.5 * self.acc * (dt**2)
        new_acc = compute_forces_grid_parallel(self.positions, self.masses, self.bounds_max, self.grid_res)
        self.velocities += 0.5 * (self.acc + new_acc) * dt
        self.acc = new_acc

def run_simulation():
    N_STARS = 8000 
    DT = 0.1
    
    print(f"Generating {N_STARS} stars...")
    masses, positions, velocities, colors = generate_galaxy(n_stars=N_STARS)
    
    galaxy = GalaxyNumba(masses, positions, velocities) 
    
    pos_array = galaxy.positions
    max_range = np.max(np.abs(pos_array)) * 1.5
    bounds = ((-max_range, max_range), (-max_range, max_range), (-max_range, max_range))
    luminosities = np.ones(len(masses))
    
    vis = Visualizer3D(positions, colors, luminosities, bounds)
    
    frame_count = 0
    last_time = time.time()
    
    def update_loop(dt_unused):
        nonlocal frame_count, last_time
        
        t_start = time.time()
        galaxy.step(DT)
        t_end = time.time()
        
        calc_time = t_end - t_start
        frame_count += 1
        
        if frame_count % 10 == 0:
            current_time = time.time()
            elapsed = current_time - last_time
            fps = 10 / elapsed
            print(f"FPS: {fps:.2f} | Calc Time: {calc_time*1000:.3f} ms | N: {N_STARS} | Mode: 2D Grid CSR-Verlet")
            last_time = current_time
            
        return galaxy.positions

    print("Starting simulation...")
    vis.run(updater=update_loop, dt=DT)

if __name__ == "__main__":
    run_simulation()
