import numpy as np

def build_grid_graph(rows, cols):
    n_nodes = rows * cols
    neighbors = [[] for _ in range(n_nodes)]
    for i in range(rows):
        for j in range(cols):
            node_idx = i * cols + j
            if i > 0: neighbors[node_idx].append((i - 1) * cols + j)
            if i < rows - 1: neighbors[node_idx].append((i + 1) * cols + j)
            if j > 0: neighbors[node_idx].append(i * cols + (j - 1))
            if j < cols - 1: neighbors[node_idx].append(i * cols + (j + 1))
    return neighbors


class Robot:
    def __init__(self, initial_position, rng):
        self.position = initial_position
        self.rng = rng

    def choose_next(self, neighbors, idleness, forbidden=None):
        current_neighbors = neighbors[self.position]
        if forbidden is not None:
            filtered = [n for n in current_neighbors if n != forbidden]
            if filtered:
                current_neighbors = filtered
        if len(current_neighbors) == 0:
            return self.position
        nbrs = np.array(current_neighbors)
        vals = idleness[nbrs]
        max_idle = vals.max()
        candidates = nbrs[vals == max_idle]
        return int(self.rng.choice(candidates))


class CoordinatedHIFSimulator:
    def __init__(self, seed=42, idleness_init=None, robot1_pos=0, robot2_pos=99, neighbors_override=None):
        self.rng = np.random.default_rng(seed)
        self.rows, self.cols = 10, 10
        self.n_nodes = 100
        self.neighbors = neighbors_override if neighbors_override is not None else build_grid_graph(self.rows, self.cols)
        self.idleness = np.zeros(self.n_nodes, dtype=np.int32) if idleness_init is None else idleness_init.astype(np.int32).copy()
        self.robot1 = Robot(robot1_pos, self.rng)
        self.robot2 = Robot(robot2_pos, self.rng)

    def step(self):
        self.idleness += 1
        next1 = self.robot1.choose_next(self.neighbors, self.idleness)
        next2 = self.robot2.choose_next(self.neighbors, self.idleness, forbidden=next1)
        self.robot1.position = next1
        self.robot2.position = next2
        self.idleness[next1] = 0
        self.idleness[next2] = 0
        return self.idleness.mean(), self.idleness.max()

def run_single_simulation(args):
    """Worker function — runs ONE full 1,000,000-step simulation and returns (X_row, Y_row)."""
    seed, r1_pos, r2_pos, total_steps, tail_start, neighbors_override = args
    n_nodes = 100
    initial_idleness = np.zeros(n_nodes, dtype=np.int32)

    sim = CoordinatedHIFSimulator(
        seed=seed, idleness_init=initial_idleness,
        robot1_pos=r1_pos, robot2_pos=r2_pos,
        neighbors_override=neighbors_override
    )

    max_iai = -np.inf
    max_maxidle = -np.inf
    X_row = None

    for t in range(total_steps):
        iai, max_idle = sim.step()
        if t == tail_start:
            idle_snapshot = sim.idleness.astype(np.float32).copy()
            r1_onehot = np.zeros(n_nodes, dtype=np.float32)
            r2_onehot = np.zeros(n_nodes, dtype=np.float32)
            r1_onehot[sim.robot1.position] = 1.0
            r2_onehot[sim.robot2.position] = 1.0
            X_row = np.concatenate([idle_snapshot, r1_onehot, r2_onehot])
        if t >= tail_start:
            if iai > max_iai: max_iai = iai
            if max_idle > max_maxidle: max_maxidle = max_idle

    return X_row, np.array([max_iai, max_maxidle], dtype=np.float32)