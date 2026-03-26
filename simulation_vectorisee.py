import numpy as np
import time
from galaxy_generator import generate_galaxy

try:
    from visualizer3d_vbo import Visualizer3D
    print("Using VBO Visualizer")
except ImportError:
    from visualizer3d_sans_vbo import Visualizer3D
    print("Using Standard Visualizer (No VBO)")

# G = 1.560339e-13 ly^3 / (M_sun * an^2)
G = 1.560339e-13 

class GalaxyVectorized:
    def __init__(self, masses, positions, velocities):
        self.masses = np.array(masses, dtype=np.float32)       # Shape: (N,)
        self.positions = np.array(positions, dtype=np.float32) # Shape: (N, 3)
        self.velocities = np.array(velocities, dtype=np.float32) # Shape: (N, 3)
        self.n_bodies = len(masses)

    def compute_accelerations(self):
        """
        - positions: (N, 3)
        - pos_j (N, 1, 3) - pos_i (1, N, 3) -> diff (N, N, 3)
        """
        #  r_ji = p_j - p_i
        # (1, N, 3) - (N, 1, 3) = (N, N, 3)
        diff = self.positions[None, :, :] - self.positions[:, None, :]
        
        dist = np.linalg.norm(diff, axis=2)
        

        np.fill_diagonal(dist, np.inf)
        
        # force_scalar shape: (N, N)
        force_scalar = G * self.masses[None, :] / (dist**3)
       
        # diff (N, N, 3) * force_scalar (N, N, 1) -> (N, N, 3)
        acc = np.sum(diff * force_scalar[:, :, np.newaxis], axis=1)
        
        return acc

    def step(self, dt):
        """update
        """
        acc = self.compute_accelerations()
        
        # p(t+dt) = p(t) + v(t)dt + 0.5*a(t)dt^2
        self.positions += self.velocities * dt + 0.5 * acc * (dt**2)
        
        # v(t+dt) = v(t) + a(t)dt
        self.velocities += acc * dt

def run_simulation():
    N_STARS =500    
    DT = 0.001           
    
    print(f"Generating galaxy with {N_STARS} stars (Vectorized)...")
    
    masses, positions, velocities, colors = generate_galaxy(n_stars=N_STARS)
    
    galaxy = GalaxyVectorized(masses, positions, velocities)
    
    pos_array = galaxy.positions
    max_range = np.max(np.abs(pos_array)) * 1.5
    bounds = ((-max_range, max_range), (-max_range, max_range), (-max_range, max_range))
    luminosities = np.ones(len(masses))
    
    vis = Visualizer3D(
        positions, 
        colors, 
        luminosities, 
        bounds
    )

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
            print(f"FPS: {fps:.2f} | Calc Time: {calc_time*1000:.2f} ms | N: {N_STARS}")
            last_time = current_time
            
        return galaxy.positions

    vis.run(updater=update_loop, dt=DT)

if __name__ == "__main__":
    run_simulation()