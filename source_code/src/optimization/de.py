import numpy as np

from src.utils.config import RANDOM_STATE


class DE:
    """Differential Evolution for continuous / mixed-integer optimisation.

    Implements DE/rand/1/bin strategy with boundary clamping.
    """

    def __init__(
        self,
        pop_size=50,
        bounds=None,
        discrete_indices=None,
        F=0.8,
        CR=0.9,
        max_iter=100,
        early_stop_iters=None,
        tol=1e-6,
        maximise=True,
        seed=RANDOM_STATE,
    ):
        self.pop_size = pop_size
        self.bounds = np.asarray(bounds, dtype=np.float64)
        self.dim = self.bounds.shape[0]
        self.discrete_indices = set(discrete_indices or [])
        self.F = F
        self.CR = CR
        self.max_iter = max_iter
        self.early_stop_iters = early_stop_iters
        self.tol = tol
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

    def _evaluate(self, x):
        return self.fitness_fn(self._round_discrete(x).reshape(1, -1)).item()

    def _init_population(self):
        pop = self.rng.uniform(
            self.bounds[:, 0], self.bounds[:, 1], size=(self.pop_size, self.dim)
        )
        for i in range(self.pop_size):
            for idx in self.discrete_indices:
                pop[i, idx] = round(pop[i, idx])
            pop[i] = self._clamp(pop[i])
        return pop

    def optimize(self, verbose=False):
        if self.fitness_fn is None:
            raise ValueError("Call set_fitness(fn) before optimize().")

        pop = self._init_population()
        fitness = np.array([self._evaluate(pop[i]) for i in range(self.pop_size)])

        if self.maximise:
            best_idx = np.argmax(fitness)
        else:
            best_idx = np.argmin(fitness)
        self.best_position = pop[best_idx].copy()
        self.best_fitness = float(fitness[best_idx])

        stall_counter = 0
        early_stop = self.early_stop_iters

        for gen in range(self.max_iter):
            for i in range(self.pop_size):
                candidates = [j for j in range(self.pop_size) if j != i]
                a, b, c = self.rng.choice(candidates, size=3, replace=False)

                mutant = pop[a] + self.F * (pop[b] - pop[c])
                mutant = self._clamp(mutant)

                cross_mask = self.rng.rand(self.dim) < self.CR
                j_rand = self.rng.randint(self.dim)
                cross_mask[j_rand] = True

                trial = np.where(cross_mask, mutant, pop[i])
                trial = self._round_discrete(trial)
                trial = self._clamp(trial)

                trial_fitness = self._evaluate(trial)

                if (self.maximise and trial_fitness > fitness[i]) or \
                   (not self.maximise and trial_fitness < fitness[i]):
                    pop[i] = trial
                    fitness[i] = trial_fitness

            if self.maximise:
                current_best_idx = np.argmax(fitness)
                current_best = fitness[current_best_idx]
            else:
                current_best_idx = np.argmin(fitness)
                current_best = fitness[current_best_idx]

            if (self.maximise and current_best > self.best_fitness + self.tol) or \
               (not self.maximise and current_best < self.best_fitness - self.tol):
                self.best_fitness = float(current_best)
                self.best_position = pop[current_best_idx].copy()
                stall_counter = 0
            else:
                stall_counter += 1

            self.history["best_fitness"].append(float(self.best_fitness))
            self.history["mean_fitness"].append(float(np.mean(fitness)))
            self.history["best_position"].append(self.best_position.copy())

            if verbose and gen % 20 == 0:
                print(f"  DE gen {gen:3d}/{self.max_iter}  best={self.best_fitness:.4f}  "
                      f"mean={np.mean(fitness):.4f}")

            if early_stop and stall_counter >= early_stop:
                if verbose:
                    print(f"  DE early stopped at gen {gen}")
                self.n_iters = gen + 1
                break
        else:
            self.n_iters = self.max_iter

        return self.best_position, self.best_fitness, self.history, self.n_iters
