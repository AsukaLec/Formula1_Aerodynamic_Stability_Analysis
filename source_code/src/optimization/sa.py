import numpy as np

from src.utils.config import RANDOM_STATE


class SA:
    """Simulated Annealing for continuous / mixed-integer optimisation.

    Implements exponential cooling schedule with Metropolis acceptance.
    """

    def __init__(
        self,
        bounds=None,
        discrete_indices=None,
        T_start=100.0,
        T_end=0.01,
        cooling_rate=0.95,
        steps_per_temp=20,
        max_iter=2000,
        early_stop_iters=None,
        tol=1e-6,
        adaptive_step=True,
        maximise=True,
        seed=RANDOM_STATE,
    ):
        self.bounds = np.asarray(bounds, dtype=np.float64)
        self.dim = self.bounds.shape[0]
        self.discrete_indices = set(discrete_indices or [])
        self.T_start = T_start
        self.T_end = T_end
        self.cooling_rate = cooling_rate
        self.steps_per_temp = steps_per_temp
        self.max_iter = max_iter
        self.early_stop_iters = early_stop_iters
        self.tol = tol
        self.adaptive_step = adaptive_step
        self.maximise = maximise
        self.rng = np.random.RandomState(seed)

        self.fitness_fn = None
        self.history = {"best_fitness": [], "mean_fitness": [], "best_position": []}
        self.best_position = None
        self.best_fitness = -np.inf if maximise else np.inf
        self.n_iters = 0

    def set_fitness(self, fn):
        self.fitness_fn = fn

    def _clamp(self, x):
        return np.clip(x, self.bounds[:, 0], self.bounds[:, 1])

    def _round_discrete(self, x):
        xc = x.copy()
        for idx in self.discrete_indices:
            xc[idx] = round(xc[idx])
        return self._clamp(xc)

    def _neighbour(self, x, T_frac):
        """Generate a neighbour by Gaussian perturbation, scaled by temperature."""
        rng_range = self.bounds[:, 1] - self.bounds[:, 0]
        scale = rng_range * (0.1 * T_frac + 0.01)
        noise = self.rng.normal(0, scale, size=self.dim)
        neighbour = x + noise
        return self._round_discrete(self._clamp(neighbour))

    def _evaluate(self, x):
        return self.fitness_fn(self._round_discrete(x).reshape(1, -1)).item()

    def _should_accept(self, delta_fitness, T):
        if self.maximise:
            delta = -delta_fitness
        else:
            delta = delta_fitness
        if delta < 0:
            return True
        prob = np.exp(-delta / max(T, 1e-12))
        return self.rng.rand() < prob

    def optimize(self, verbose=False):
        if self.fitness_fn is None:
            raise ValueError("Call set_fitness(fn) before optimize().")

        x = self.rng.uniform(self.bounds[:, 0], self.bounds[:, 1])
        x = self._round_discrete(x)
        x_fitness = self._evaluate(x)

        self.best_position = x.copy()
        self.best_fitness = float(x_fitness)

        T = self.T_start
        stall_counter = 0
        early_stop = self.early_stop_iters
        total_steps = 0

        gen = 0
        while total_steps < self.max_iter and T > self.T_end:
            for _ in range(self.steps_per_temp):
                if total_steps >= self.max_iter:
                    break
                total_steps += 1

                T_frac = (np.log(T) - np.log(self.T_end)) / \
                         max(np.log(self.T_start) - np.log(self.T_end), 1e-12)
                neighbour = self._neighbour(x, T_frac)
                neighbour_fitness = self._evaluate(neighbour)

                delta = neighbour_fitness - x_fitness
                if self._should_accept(delta, T):
                    x = neighbour
                    x_fitness = neighbour_fitness

                if (self.maximise and x_fitness > self.best_fitness) or \
                   (not self.maximise and x_fitness < self.best_fitness):
                    self.best_fitness = float(x_fitness)
                    self.best_position = x.copy()
                    stall_counter = 0
                else:
                    stall_counter += 1

                self.history["best_fitness"].append(float(self.best_fitness))
                self.history["best_position"].append(self.best_position.copy())

                if early_stop and stall_counter >= early_stop:
                    if verbose:
                        print(f"  SA early stopped at step {total_steps}")
                    self.n_iters = total_steps
                    return self.best_position, self.best_fitness, self.history, self.n_iters

            T *= self.cooling_rate
            gen += 1

            if verbose and gen % 20 == 0:
                print(f"  SA T={T:.4f}  step={total_steps}  best={self.best_fitness:.4f}  "
                      f"current={x_fitness:.4f}")

        self.n_iters = total_steps
        return self.best_position, self.best_fitness, self.history, self.n_iters
