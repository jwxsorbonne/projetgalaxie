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

# ---  Numba JIT  ---
@njit(fastmath=True)
def compute_forces_numba(positions, masses):
    n = len(positions)
    acc = np.zeros((n, 3), dtype=np.float32)
    

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
                

            # r_ji

            dx = positions[j, 0] - positions[i, 0]
            dy = positions[j, 1] - positions[i, 1]
            dz = positions[j, 2] - positions[i, 2]
            
            dist_sq = dx*dx + dy*dy + dz*dz
            dist = np.sqrt(dist_sq)
            
            if dist < 1e-5:
                continue
                
            # F = G * m / r^3 * vec_r
            f = (G * masses[j]) / (dist_sq * dist)
            
            acc[i, 0] += f * dx
            acc[i, 1] += f * dy
            acc[i, 2] += f * dz
            
    return acc

# parallel=True 
@njit(parallel=True, fastmath=True)
def compute_forces_parallel(positions, masses):
    n = len(positions)
    acc = np.zeros((n, 3), dtype=np.float32)
    
    for i in prange(n):
        for j in range(n):
            if i == j:
                continue
            
            dx = positions[j, 0] - positions[i, 0]
            dy = positions[j, 1] - positions[i, 1]
            dz = positions[j, 2] - positions[i, 2]
            
            dist_sq = dx*dx + dy*dy + dz*dz
            dist = np.sqrt(dist_sq)
            
            if dist < 1e-5:
                continue
            
            f = (G * masses[j]) / (dist_sq * dist)
            
            acc[i, 0] += f * dx
            acc[i, 1] += f * dy
            acc[i, 2] += f * dz
            
    return acc

class GalaxyNumba:
    def __init__(self, masses, positions, velocities, use_parallel=True):
        self.masses = np.array(masses, dtype=np.float32)
        self.positions = np.array(positions, dtype=np.float32)
        self.velocities = np.array(velocities, dtype=np.float32)
        self.use_parallel = use_parallel
        
        
        print("Compiling Numba functions (Warmup)...")
        if use_parallel:
            compute_forces_parallel(self.positions, self.masses)
        else:
            compute_forces_numba(self.positions, self.masses)
        print("Compilation done.")

    def step(self, dt):
        if self.use_parallel:
            acc = compute_forces_parallel(self.positions, self.masses)
        else:
            acc = compute_forces_numba(self.positions, self.masses)
            
        self.positions += self.velocities * dt + 0.5 * acc * (dt**2)
        self.velocities += acc * dt

def run_simulation():

    N_STARS = 5000
    DT = 0.01
    USE_PARALLEL = True  
    
    print(f"Generating {N_STARS} stars...")
    masses, positions, velocities, colors = generate_galaxy(n_stars=N_STARS)
    
    galaxy = GalaxyNumba(masses, positions, velocities, use_parallel=USE_PARALLEL)
        
    
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
            print(f"FPS: {fps:.2f} | Calc Time: {calc_time*1000:.3f} ms | N: {N_STARS} | Parallel: {USE_PARALLEL}")
            last_time = current_time
            
        return galaxy.positions

    print("Starting simulation...")
    vis.run(updater=update_loop, dt=DT)

if __name__ == "__main__":
    run_simulation()