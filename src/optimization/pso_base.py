import numpy as np

from src.utils.config import (
    RANDOM_STATE,
    PSO_N_PARTICLES,
    PSO_W,
    PSO_C1,
    PSO_C2,
    PSO_MAX_ITER,
    PSO_EARLY_STOP_ITERS,
    PSO_TOL,
)


class Particle:
    def __init__(self, pos, vel=None):
        self.pos = np.asarray(pos, dtype=np.float64).copy()
        self.vel = np.asarray(vel, dtype=np.float64).copy() if vel is not None else np.zeros_like(pos)
        self.pbest_pos = self.pos.copy()
        self.pbest_fitness = -np.inf
        self.fitness = -np.inf


class PSOBase:
    """Standard Particle Swarm Optimization.

    Minimises fitness_fn by default; set maximise=True for maximisation problems.

    Discrete variables: indices listed in `discrete_indices` are rounded to nearest
    integer within their bounds during evaluation and after position update.
    """

    def __init__(
        self,
        n_particles=PSO_N_PARTICLES,
        bounds=None,
        discrete_indices=None,
        w=PSO_W,
        c1=PSO_C1,
        c2=PSO_C2,
        max_iter=PSO_MAX_ITER,
        early_stop_iters=PSO_EARLY_STOP_ITERS,
        tol=PSO_TOL,
        boundary_method="clamp",
        maximise=True,
        seed=RANDOM_STATE,
        v_clamp=None,
    ):
        """
        Parameters
        ----------
        n_particles : int
            Swarm size.
        bounds : np.ndarray of shape (dim, 2)
            [lower, upper] for each dimension.
        discrete_indices : list of int or None
            Indices of discrete variables. Rounded during evaluation and after position update.
        w : float
            Inertia weight.
        c1, c2 : float
            Cognitive and social learning factors.
        max_iter : int
            Maximum iterations.
        early_stop_iters : int or None
            Stop if gbest fitness does not improve for this many consecutive iterations.
            Default: max_iter // 5 (min 10).
        tol : float
            Minimum improvement threshold for early stopping.
        boundary_method : str
            'clamp' -> clip to bounds; 'reflect' -> bounce back.
        maximise : bool
            If True, PSO maximises fitness. If False, minimises.
        seed : int
            Random seed.
        v_clamp : float or None
            Max velocity magnitude per dimension. If None, computed as 0.2 * (ub - lb).
        """
        self.n_particles = n_particles
        self.bounds = np.asarray(bounds, dtype=np.float64)
        self.dim = self.bounds.shape[0]
        self.discrete_indices = sorted(set(discrete_indices or []))
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.max_iter = max_iter
        self.early_stop_iters = early_stop_iters if early_stop_iters is not None else PSO_EARLY_STOP_ITERS
        self.tol = tol
        self.boundary_method = boundary_method
        self.maximise = maximise
        self.seed = seed

        if v_clamp is None:
            self.v_clamp = 0.2 * (self.bounds[:, 1] - self.bounds[:, 0])
        else:
            self.v_clamp = np.full(self.dim, v_clamp, dtype=np.float64)

        self.rng = np.random.RandomState(seed)

        self.particles = []
        self.gbest_pos = None
        self.gbest_fitness = -np.inf if maximise else np.inf
        self.history = {"gbest_fitness": [], "gbest_pos": [], "mean_fitness": [], "iteration": []}
        self._converged = False
        self._final_iter = 0

    def _init_particles(self):
        self.particles = []
        for _ in range(self.n_particles):
            pos = np.zeros(self.dim, dtype=np.float64)
            for d in range(self.dim):
                lb, ub = self.bounds[d]
                pos[d] = self.rng.uniform(lb, ub)
            vel = np.zeros(self.dim, dtype=np.float64)
            for d in range(self.dim):
                vel[d] = self.rng.uniform(-self.v_clamp[d], self.v_clamp[d])
            self.particles.append(Particle(pos, vel))

    def _round_discrete(self, pos):
        pos = pos.copy()
        for d in self.discrete_indices:
            lb, ub = self.bounds[d]
            # Round to nearest integer within [lb, ub]
            rounded = np.round(pos[d])
            pos[d] = np.clip(rounded, lb, ub)
        return pos

    def _enforce_bounds(self, pos, vel=None):
        pos = pos.copy()
        vel_out = vel.copy() if vel is not None else None
        for d in range(self.dim):
            lb, ub = self.bounds[d]
            if pos[d] < lb:
                if self.boundary_method == "reflect":
                    pos[d] = lb + (lb - pos[d])
                    if vel_out is not None:
                        vel_out[d] = -vel_out[d]
                else:
                    pos[d] = lb
            elif pos[d] > ub:
                if self.boundary_method == "reflect":
                    pos[d] = ub - (pos[d] - ub)
                    if vel_out is not None:
                        vel_out[d] = -vel_out[d]
                else:
                    pos[d] = ub
        if vel_out is not None:
            return pos, vel_out
        return pos

    def _clamp_velocity(self, vel):
        vel = vel.copy()
        for d in range(self.dim):
            vel[d] = np.clip(vel[d], -self.v_clamp[d], self.v_clamp[d])
        return vel

    def _is_better(self, a, b):
        if self.maximise:
            return a > b
        else:
            return a < b

    def optimize(self, fitness_fn, verbose=True, collect_candidates=False):
        """Run PSO optimisation.

        Parameters
        ----------
        fitness_fn : callable
            Function that takes (N, dim) array and returns (N,) fitness values.
        verbose : bool
            Print progress.
        collect_candidates : bool
            If True, collect all evaluated (position, fitness) pairs for
            high-performance region analysis. Stored in result["candidates"].

        Returns
        -------
        dict with keys: gbest_pos, gbest_fitness, history, n_iter, converged, candidates
        """
        self._init_particles()

        candidates_all = [] if collect_candidates else None

        # Evaluate initial population
        X0 = np.array([p.pos for p in self.particles])
        f0 = np.asarray(fitness_fn(X0), dtype=np.float64)
        for i, p in enumerate(self.particles):
            p.fitness = f0[i]
            p.pbest_fitness = f0[i]
            p.pbest_pos = p.pos.copy()
            if self._is_better(p.fitness, self.gbest_fitness):
                self.gbest_fitness = p.fitness
                self.gbest_pos = p.pos.copy()

        if collect_candidates:
            candidates_all.append((X0.copy(), f0.copy()))

        self.history["gbest_fitness"].append(self.gbest_fitness)
        self.history["gbest_pos"].append(self.gbest_pos.copy())
        self.history["mean_fitness"].append(f0.mean())
        self.history["iteration"].append(0)

        if verbose:
            print(f"  Iter   0: gbest={self.gbest_fitness:.6f}, mean={f0.mean():.6f}")

        gbest_before = self.gbest_fitness
        stall_count = 0
        self._converged = False

        for t in range(1, self.max_iter + 1):
            w = self._get_inertia(t) if hasattr(self, "_get_inertia") else self.w

            # Build position matrix for batch evaluation
            new_positions = np.zeros((self.n_particles, self.dim), dtype=np.float64)

            for i, p in enumerate(self.particles):
                r1 = self.rng.rand(self.dim)
                r2 = self.rng.rand(self.dim)

                cognitive = self.c1 * r1 * (p.pbest_pos - p.pos)
                social = self.c2 * r2 * (self.gbest_pos - p.pos)
                vel = w * p.vel + cognitive + social
                vel = self._clamp_velocity(vel)
                pos = p.pos + vel
                pos, vel = self._enforce_bounds(pos, vel)
                pos = self._round_discrete(pos)

                p.vel = vel
                new_positions[i] = pos

            # Batch fitness evaluation
            fitness = np.asarray(fitness_fn(new_positions), dtype=np.float64)

            if collect_candidates:
                candidates_all.append((new_positions.copy(), fitness.copy()))

            for i, p in enumerate(self.particles):
                p.pos = new_positions[i]
                p.fitness = fitness[i]

                if self._is_better(fitness[i], p.pbest_fitness):
                    p.pbest_fitness = fitness[i]
                    p.pbest_pos = p.pos.copy()

                if self._is_better(fitness[i], self.gbest_fitness):
                    self.gbest_fitness = fitness[i]
                    self.gbest_pos = p.pos.copy()

            # Early stopping: check if gbest improved beyond tolerance this iteration
            if abs(self.gbest_fitness - gbest_before) <= self.tol:
                stall_count += 1
            else:
                stall_count = 0
            gbest_before = self.gbest_fitness

            self.history["gbest_fitness"].append(self.gbest_fitness)
            self.history["gbest_pos"].append(self.gbest_pos.copy())
            self.history["mean_fitness"].append(fitness.mean())
            self.history["iteration"].append(t)

            if verbose and (t % 10 == 0 or t == self.max_iter):
                print(f"  Iter {t:4d}: gbest={self.gbest_fitness:.6f}, mean={fitness.mean():.6f}")

            if stall_count >= self.early_stop_iters:
                self._converged = True
                if verbose:
                    print(f"  Converged at iter {t} (stall={stall_count})")
                break

        self._final_iter = t
        result = {
            "gbest_pos": self.gbest_pos,
            "gbest_fitness": self.gbest_fitness,
            "history": self.history,
            "n_iter": t,
            "converged": self._converged,
        }
        if collect_candidates:
            result["candidates"] = candidates_all
        return result

    def _get_inertia(self, t):
        """Override in subclasses for adaptive inertia strategies."""
        return self.w
