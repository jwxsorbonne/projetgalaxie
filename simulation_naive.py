
import numpy as np
import time
from galaxy_generator import generate_galaxy
try:
    from visualizer3d_vbo import Visualizer3D
    print("Using VBO Visualizer")
except ImportError:
    from visualizer3d_sans_vbo import Visualizer3D
    print("Using Standard Visualizer (No VBO)")


G = 1.560339e-13 

class Corps:
    """
    (Body)
    """
    def __init__(self, mass, position, velocity, color):
        self.mass = mass
        self.position = np.array(position, dtype=np.float32)
        self.velocity = np.array(velocity, dtype=np.float32)
        self.color = color
        self.acceleration = np.zeros(3, dtype=np.float32)

    def update(self, dt):
        """
        p(t+dt) = p(t) + dt*v(t) + 0.5*dt^2*a(t)
        v(t+dt) = v(t) + dt*a(t)
        """
        self.position += (self.velocity * dt) + (0.5 * (dt**2) * self.acceleration)
        
        self.velocity += self.acceleration * dt

class NCorps:
    """
    (N-Body System)
   
    """
    def __init__(self, bodies):
        self.bodies = bodies 

    def compute_gravity(self):
        """
        acceleration O(N^2)
        """
        n = len(self.bodies)
        
        for i in range(n):
            body_i = self.bodies[i]
            total_acc = np.zeros(3, dtype=np.float32)
            
            for j in range(n):
                if i == j:
                    continue 
                
                body_j = self.bodies[j]
                
                r_vec = body_j.position - body_i.position
                
                distance = np.linalg.norm(r_vec)
                
                if distance < 1e-5:
                    continue

             
                acc_contribution = G * body_j.mass * r_vec / (distance**3)
                total_acc += acc_contribution
            
            body_i.acceleration = total_acc

    def step(self, dt):
        """
        simulation of step
        """
        self.compute_gravity()
        
        for body in self.bodies:
            body.update(dt)

def run_simulation():
    N_STARS = 500 
    DT = 0.01          
    
    print(f"Generating galaxy with {N_STARS} stars...")
    
  
    masses, positions, velocities, colors = generate_galaxy(n_stars=N_STARS)
    
    bodies_list = []
    for i in range(len(masses)):
        b = Corps(masses[i], positions[i], velocities[i], colors[i])
        bodies_list.append(b)
        
    system = NCorps(bodies_list)
    

    pos_array = np.array(positions)
    max_range = np.max(np.abs(pos_array)) * 1.5
    bounds = ((-max_range, max_range), (-max_range, max_range), (-max_range, max_range))
    
    luminosities = np.ones(len(bodies_list))
    
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
        
        system.step(DT)
        

        new_positions = np.array([b.position for b in system.bodies], dtype=np.float32)
        
        t_end = time.time()
        calc_time = t_end - t_start
        frame_count += 1
        
        if frame_count % 10 == 0:
            current_time = time.time()
            elapsed = current_time - last_time
            fps = 10 / elapsed
            print(f"FPS: {fps:.2f} | Calc Time per step: {calc_time*1000:.2f} ms | Bodies: {len(bodies_list)}")
            last_time = current_time
            
        return new_positions

    print("Starting simulation... Close the window to stop.")
    vis.run(updater=update_loop, dt=DT)

if __name__ == "__main__":
    run_simulation()