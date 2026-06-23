import numpy as np

from src.utils.config import RANDOM_STATE


class GA:
    """Genetic Algorithm for continuous / mixed-integer optimisation.

    Supports:
      - Tournament selection
      - Simulated binary crossover (SBX) for continuous variables
      - Uniform crossover for discrete / binary variables
      - Gaussian mutation
      - Elitism
      - Boundary clamping
    """

    def __init__(
        self,
        pop_size=50,
        bounds=None,
        discrete_indices=None,
        p_crossover=0.8,
        p_mutation=0.1,
        mutation_eta=0.5,
        tournament_k=3,
        elitism=2,
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
        self.p_crossover = p_crossover
        self.p_mutation = p_mutation
        self.mutation_eta = mutation_eta
        self.tournament_k = tournament_k
        self.elitism = elitism
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

    def _init_population(self):
        pop = self.rng.uniform(
            self.bounds[:, 0], self.bounds[:, 1], size=(self.pop_size, self.dim)
        )
        for i in range(self.pop_size):
            for idx in self.discrete_indices:
                pop[i, idx] = round(pop[i, idx])
            pop[i] = self._clamp(pop[i])
        return pop

    def _evaluate(self, x):
        return self.fitness_fn(self._round_discrete(x).reshape(1, -1)).item()

    def _tournament_select(self, pop, fitness):
        idx = self.rng.choice(self.pop_size, size=self.tournament_k, replace=False)
        if self.maximise:
            best = idx[np.argmax(fitness[idx])]
        else:
            best = idx[np.argmin(fitness[idx])]
        return pop[best].copy()

    def _crossover_sbx(self, p1, p2):
        u = self.rng.random(self.dim)
        beta = np.where(u <= 0.5,
                        (2 * u) ** (1.0 / 21),
                        (1.0 / (2 * (1 - u))) ** (1.0 / 21))
        c1 = 0.5 * ((1 + beta) * p1 + (1 - beta) * p2)
        c2 = 0.5 * ((1 - beta) * p1 + (1 + beta) * p2)
        return self._clamp(c1), self._clamp(c2)

    def _crossover_uniform(self, p1, p2):
        mask = self.rng.rand(self.dim) < 0.5
        c1 = np.where(mask, p1, p2)
        c2 = np.where(mask, p2, p1)
        return self._clamp(c1), self._clamp(c2)

    def _crossover(self, p1, p2):
        is_discrete = np.zeros(self.dim, dtype=bool)
        for idx in self.discrete_indices:
            is_discrete[idx] = True
        c1, c2 = self._crossover_sbx(p1, p2)
        for idx in self.discrete_indices:
            mask = self.rng.rand() < 0.5
            c1[idx] = p1[idx] if mask else p2[idx]
            c2[idx] = p2[idx] if mask else p1[idx]
        return self._clamp(c1), self._clamp(c2)

    def _mutate(self, x, gen):
        scale = self.mutation_eta * (1.0 - gen / max(1, self.max_iter))
        for j in range(self.dim):
            if self.rng.rand() < self.p_mutation:
                rng_j = self.bounds[j, 1] - self.bounds[j, 0]
                noise = self.rng.normal(0, rng_j * scale)
                x[j] += noise
        return self._clamp(x)

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
            new_pop = np.empty_like(pop)

            elite_indices = np.argsort(-fitness) if self.maximise else np.argsort(fitness)
            for e in range(min(self.elitism, self.pop_size)):
                new_pop[e] = pop[elite_indices[e]].copy()

            for i in range(self.elitism, self.pop_size, 2):
                p1 = self._tournament_select(pop, fitness)
                p2 = self._tournament_select(pop, fitness)
                if self.rng.rand() < self.p_crossover:
                    c1, c2 = self._crossover(p1, p2)
                else:
                    c1, c2 = p1.copy(), p2.copy()
                c1 = self._mutate(c1, gen)
                c2 = self._mutate(c2, gen)
                c1 = self._round_discrete(c1)
                c2 = self._round_discrete(c2)
                new_pop[i] = c1
                if i + 1 < self.pop_size:
                    new_pop[i + 1] = c2

            for i in range(self.pop_size):
                for idx in self.discrete_indices:
                    new_pop[i, idx] = round(new_pop[i, idx])
                new_pop[i] = self._clamp(new_pop[i])

            fitness = np.array([self._evaluate(new_pop[i]) for i in range(self.pop_size)])
            pop = new_pop

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
                print(f"  GA gen {gen:3d}/{self.max_iter}  best={self.best_fitness:.4f}  "
                      f"mean={np.mean(fitness):.4f}")

            if early_stop and stall_counter >= early_stop:
                if verbose:
                    print(f"  GA early stopped at gen {gen}")
                self.n_iters = gen + 1
                break
        else:
            self.n_iters = self.max_iter

        return self.best_position, self.best_fitness, self.history, self.n_iters
