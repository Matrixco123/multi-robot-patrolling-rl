
# Sequential Coordinated HIF Persistent Patrolling
# Robot 1 chooses first; Robot 2 sees Robot 1 destination and avoids it if possible.

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def build_grid_graph(rows=10, cols=10):
    neighbors = [[] for _ in range(rows * cols)]
    for r in range(rows):
        for c in range(cols):
            n = r * cols + c
            if r > 0: neighbors[n].append((r - 1) * cols + c)
            if r < rows - 1: neighbors[n].append((r + 1) * cols + c)
            if c > 0: neighbors[n].append(r * cols + c - 1)
            if c < cols - 1: neighbors[n].append(r * cols + c + 1)
    return neighbors

class Robot:
    def __init__(self, start, rng):
        self.position = start
        self.rng = rng

    def choose_next(self, neighbors, idleness, forbidden=None):
        nbrs = neighbors[self.position]
        if forbidden is not None:
            filt = [n for n in nbrs if n != forbidden]
            if filt:
                nbrs = filt
        nbrs = np.array(nbrs)
        vals = idleness[nbrs]
        mx = vals.max()
        candidates = nbrs[vals == mx]
        return int(self.rng.choice(candidates))

class Simulation:
    def __init__(self, rows=10, cols=10, timesteps=1_000_000, seed=42):
        self.rows = rows
        self.cols = cols
        self.timesteps = timesteps
        self.n_nodes = rows * cols

        self.rng = np.random.default_rng(seed)
        self.neighbors = build_grid_graph(rows, cols)

        self.idleness = np.zeros(self.n_nodes, dtype=np.int32)

        starts = [0, self.n_nodes - 1]
        self.robots = [Robot(starts[0], self.rng), Robot(starts[1], self.rng)]

        self.visit_count = np.zeros(self.n_nodes, dtype=np.int64)
        self.idle_sum = np.zeros(self.n_nodes, dtype=np.float64)
        self.max_idle_per_node = np.zeros(self.n_nodes, dtype=np.int32)

        for s in starts:
            self.visit_count[s] += 1

        self.sum_iai = 0.0
        self.sum_max = 0.0
        self.global_max = 0

        self.store_start = int(0.9 * timesteps)
        tail_size = timesteps - self.store_start

        self.iai_tail = np.empty(tail_size, dtype=np.float32)
        self.max_tail = np.empty(tail_size, dtype=np.int32)

    def coordinated_move(self):
        r1, r2 = self.robots

        next1 = r1.choose_next(self.neighbors, self.idleness)

        next2 = r2.choose_next(
            self.neighbors,
            self.idleness,
            forbidden=next1
        )

        r1.position = next1
        r2.position = next2

        return [next1, next2]

    def step(self):
        self.idleness += 1

        self.idle_sum += self.idleness
        self.max_idle_per_node = np.maximum(
            self.max_idle_per_node,
            self.idleness
        )

        visited = self.coordinated_move()

        self.idleness[visited] = 0

        for v in visited:
            self.visit_count[v] += 1

    def run(self):
        idx = 0

        for t in tqdm(range(self.timesteps), desc="Running"):
            self.step()

            iai = self.idleness.mean()
            mx = self.idleness.max()

            self.sum_iai += iai
            self.sum_max += mx
            self.global_max = max(self.global_max, int(mx))

            if t >= self.store_start:
                self.iai_tail[idx] = iai
                self.max_tail[idx] = mx
                idx += 1

def plot_time_series(sim):
    window = 10000
    stride = 10

    iai = sim.iai_tail[-window:][::stride]
    mx = sim.max_tail[-window:][::stride]

    ts = np.arange(
        sim.timesteps - window,
        sim.timesteps
    )[::stride]

    plt.figure(figsize=(12,5))
    plt.plot(ts, iai, linewidth=0.8)
    plt.title("IAI (Final 10,000 Timesteps)")
    plt.xlabel("Timestep")
    plt.ylabel("IAI")
    plt.grid(True)

    plt.figure(figsize=(12,5))
    plt.plot(ts, mx, linewidth=0.8)
    plt.title("Max Idle (Final 10,000 Timesteps)")
    plt.xlabel("Timestep")
    plt.ylabel("Max Idleness")
    plt.grid(True)

def plot_heatmaps(sim):
    visit = sim.visit_count.reshape(sim.rows, sim.cols)

    mean_idle = (
        sim.idle_sum / sim.timesteps
    ).reshape(sim.rows, sim.cols)

    max_idle = sim.max_idle_per_node.reshape(
        sim.rows,
        sim.cols
    )

    plt.figure(figsize=(8,6))
    plt.imshow(visit)
    plt.colorbar(label="Visit Count")
    plt.title("Visit Frequency Heatmap")

    plt.figure(figsize=(8,6))
    plt.imshow(mean_idle)
    plt.colorbar(label="Mean Idleness")
    plt.title("Mean Idleness Heatmap")

    plt.figure(figsize=(8,6))
    plt.imshow(max_idle)
    plt.colorbar(label="Maximum Observed Idleness")
    plt.title("Maximum Idleness Heatmap")

    plt.show()

def main():
    sim = Simulation(
        rows=10,
        cols=10,
        timesteps=1_000_000,
        seed=42
    )

    sim.run()

    print("\n===== RESULTS =====")
    print("Mean IAI:", sim.sum_iai / sim.timesteps)
    print("Mean MaxIdle:", sim.sum_max / sim.timesteps)
    print("Maximum Observed Idleness:", sim.global_max)
    print("Final IAI:", sim.iai_tail[-1])
    print("Final MaxIdle:", sim.max_tail[-1])

    print("\nDiagnostics")
    print("IAI range:", sim.iai_tail.min(), sim.iai_tail.max())
    print("MaxIdle range:", sim.max_tail.min(), sim.max_tail.max())

    plot_time_series(sim)
    plot_heatmaps(sim)

if __name__ == "__main__":
    main()
