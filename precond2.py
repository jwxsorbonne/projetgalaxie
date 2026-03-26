import os
import sys
import time
import random
import numpy as np
from numba import njit, prange

script_dir = os.path.dirname(__file__)
sys.path.append(script_dir)

from galaxy_generator import generate_galaxy

try:
    from visualizer3d_vbo import Visualizer3D
    print("Using VBO Visualizer")
except ImportError:
    from visualizer3d_sans_vbo import Visualizer3D
    print("Using Standard Visualizer (No VBO)")

G = 1.560339e-13
EPS2 = 1e-8


@njit(cache=True)
def _compute_bounds_xy(positions, margin=0.10):
    n = positions.shape[0]
    max_abs_x = 0.0
    max_abs_y = 0.0
    for i in range(n):
        ax = abs(positions[i, 0])
        ay = abs(positions[i, 1])
        if ax > max_abs_x:
            max_abs_x = ax
        if ay > max_abs_y:
            max_abs_y = ay

    max_abs = max(max_abs_x, max_abs_y)
    if max_abs < 1e-6:
        max_abs = 1.0

    half_extent = max_abs * (1.0 + margin)
    return -half_extent, half_extent, -half_extent, half_extent


@njit(cache=True)
def _build_grid_2d(positions, xmin, xmax, ymin, ymax, grid_res):
    n = positions.shape[0]
    num_cells = grid_res * grid_res
    dx = (xmax - xmin) / grid_res
    dy = (ymax - ymin) / grid_res

    cell_counts = np.zeros(num_cells, dtype=np.int32)
    cell_x = np.empty(n, dtype=np.int32)
    cell_y = np.empty(n, dtype=np.int32)
    cell_idx_of_particle = np.empty(n, dtype=np.int32)

    inv_dx = 1.0 / dx
    inv_dy = 1.0 / dy

    for i in range(n):
        px = positions[i, 0]
        py = positions[i, 1]

        cx = int((px - xmin) * inv_dx)
        cy = int((py - ymin) * inv_dy)

        if cx < 0:
            cx = 0
        elif cx >= grid_res:
            cx = grid_res - 1
        if cy < 0:
            cy = 0
        elif cy >= grid_res:
            cy = grid_res - 1

        cidx = cx * grid_res + cy
        cell_x[i] = cx
        cell_y[i] = cy
        cell_idx_of_particle[i] = cidx
        cell_counts[cidx] += 1

    row_ptr = np.empty(num_cells + 1, dtype=np.int32)
    row_ptr[0] = 0
    for c in range(num_cells):
        row_ptr[c + 1] = row_ptr[c] + cell_counts[c]

    col_indices = np.empty(n, dtype=np.int32)
    cursor = row_ptr[:-1].copy()
    for i in range(n):
        cidx = cell_idx_of_particle[i]
        p = cursor[cidx]
        col_indices[p] = i
        cursor[cidx] = p + 1

    occupied_count = 0
    for c in range(num_cells):
        if cell_counts[c] > 0:
            occupied_count += 1

    occupied_cells = np.empty(occupied_count, dtype=np.int32)
    k = 0
    for c in range(num_cells):
        if cell_counts[c] > 0:
            occupied_cells[k] = c
            k += 1

    return dx, dy, row_ptr, col_indices, cell_counts, cell_idx_of_particle, occupied_cells


@njit(cache=True)
def _compute_cell_properties(positions, masses, row_ptr, col_indices, occupied_cells, num_cells):
    cell_mass = np.zeros(num_cells, dtype=np.float32)
    cell_com = np.zeros((num_cells, 3), dtype=np.float32)

    for kk in range(occupied_cells.shape[0]):
        c = occupied_cells[kk]
        start = row_ptr[c]
        end = row_ptr[c + 1]
        msum = 0.0
        cx = 0.0
        cy = 0.0
        cz = 0.0
        for p in range(start, end):
            j = col_indices[p]
            m = masses[j]
            msum += m
            cx += m * positions[j, 0]
            cy += m * positions[j, 1]
            cz += m * positions[j, 2]
        cell_mass[c] = msum
        inv_m = 1.0 / msum
        cell_com[c, 0] = cx * inv_m
        cell_com[c, 1] = cy * inv_m
        cell_com[c, 2] = cz * inv_m

    return cell_mass, cell_com


@njit(parallel=True, fastmath=True, cache=True)
def compute_forces_grid_parallel(positions, masses, grid_res, theta, margin=0.10):
    n = positions.shape[0]
    acc = np.zeros((n, 3), dtype=np.float32)

    xmin, xmax, ymin, ymax = _compute_bounds_xy(positions, margin)
    dx, dy, row_ptr, col_indices, cell_counts, cell_of_particle, occupied_cells = _build_grid_2d(
        positions, xmin, xmax, ymin, ymax, grid_res
    )

    num_cells = grid_res * grid_res
    cell_mass, cell_com = _compute_cell_properties(
        positions, masses, row_ptr, col_indices, occupied_cells, num_cells
    )

    cell_diag2 = dx * dx + dy * dy
    theta2 = theta * theta

    for i in prange(n):
        px = positions[i, 0]
        py = positions[i, 1]
        pz = positions[i, 2]
        my_cell = cell_of_particle[i]

        ax = 0.0
        ay = 0.0
        az = 0.0

        for kk in range(occupied_cells.shape[0]):
            c = occupied_cells[kk]
            start = row_ptr[c]
            end = row_ptr[c + 1]
            count = end - start
            if count == 0:
                continue

            dxc = cell_com[c, 0] - px
            dyc = cell_com[c, 1] - py
            dzc = cell_com[c, 2] - pz
            dist2c = dxc * dxc + dyc * dyc + dzc * dzc + EPS2

            # Always open the particle's own cell; otherwise use standard opening-angle test s / d < theta.
            if c != my_cell and cell_diag2 < theta2 * dist2c:
                inv_dist3 = 1.0 / (dist2c * np.sqrt(dist2c))
                fac = G * cell_mass[c] * inv_dist3
                ax += fac * dxc
                ay += fac * dyc
                az += fac * dzc
            else:
                for p in range(start, end):
                    j = col_indices[p]
                    if j == i:
                        continue
                    dxp = positions[j, 0] - px
                    dyp = positions[j, 1] - py
                    dzp = positions[j, 2] - pz
                    dist2 = dxp * dxp + dyp * dyp + dzp * dzp + EPS2
                    inv_dist3 = 1.0 / (dist2 * np.sqrt(dist2))
                    fac = G * masses[j] * inv_dist3
                    ax += fac * dxp
                    ay += fac * dyp
                    az += fac * dzp

        acc[i, 0] = ax
        acc[i, 1] = ay
        acc[i, 2] = az

    return acc


class GalaxyNumba:
    def __init__(self, masses, positions, velocities, grid_res=None, theta=0.80, margin=0.10):
        self.masses = np.asarray(masses, dtype=np.float32)
        self.positions = np.asarray(positions, dtype=np.float32)
        self.velocities = np.asarray(velocities, dtype=np.float32)

        n = len(self.masses)
        if grid_res is None:
            # heuristic tuned for thin-disk galaxies: enough cells to localize the dense center
            # without making the occupied-cell loop too expensive.
            grid_res = max(16, min(28, int(0.70 * np.sqrt(n))))

        self.grid_res = int(grid_res)
        self.theta = np.float32(theta)
        self.margin = np.float32(margin)
        self.acc = compute_forces_grid_parallel(self.positions, self.masses, self.grid_res, self.theta, self.margin)
        print(f"Compilation done. grid_res={self.grid_res}, theta={self.theta:.2f}, margin={self.margin:.2f}")

    def step(self, dt):
        self.positions += self.velocities * dt + 0.5 * self.acc * (dt * dt)
        new_acc = compute_forces_grid_parallel(self.positions, self.masses, self.grid_res, self.theta, self.margin)
        self.velocities += 0.5 * (self.acc + new_acc) * dt
        self.acc = new_acc


def run_simulation():
    N_STARS = 5000
    DT = 0.01

    masses, positions, velocities, colors = generate_galaxy(n_stars=N_STARS)
    galaxy = GalaxyNumba(masses, positions, velocities)

    pos_array = galaxy.positions
    max_range = np.max(np.abs(pos_array)) * 1.5
    bounds = ((-max_range, max_range), (-max_range, max_range), (-max_range, max_range))
    luminosities = np.ones(len(masses), dtype=np.float32)

    vis = Visualizer3D(positions, colors, luminosities, bounds)

    frame_count = 0
    last_time = time.time()

    def update_loop(_dt_unused):
        nonlocal frame_count, last_time
        t0 = time.time()
        galaxy.step(DT)
        t1 = time.time()
        frame_count += 1
        if frame_count % 10 == 0:
            now = time.time()
            fps = 10.0 / (now - last_time)
            print(
                f"FPS: {fps:.2f} | Calc Time: {(t1 - t0)*1000:.3f} ms | "
                f"N: {N_STARS} | Mode: 2D Grid OpeningAngle-Verlet"
            )
            last_time = now
        return galaxy.positions

    vis.run(updater=update_loop, dt=DT)


if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)
    run_simulation()
