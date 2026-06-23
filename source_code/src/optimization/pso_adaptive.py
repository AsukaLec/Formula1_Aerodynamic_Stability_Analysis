from src.optimization.pso_base import PSOBase
from src.utils.config import (
    RANDOM_STATE,
    PSO_N_PARTICLES,
    PSO_W_START,
    PSO_W_END,
    PSO_W_ALPHA,
    PSO_C1,
    PSO_C2,
    PSO_MAX_ITER,
    PSO_EARLY_STOP_ITERS,
    PSO_TOL,
)


class PSOAdaptive(PSOBase):
    """Adaptive Inertia Weight PSO.

    Inertia decays from w_start to w_end over the optimisation run:
        w(t) = w_end + (w_start - w_end) * (1 - t/T)^alpha

    alpha=1 -> linear decay
    alpha>1 -> faster early decay (more exploitation early)
    alpha<1 -> slower early decay (more exploration early)
    """

    def __init__(
        self,
        n_particles=PSO_N_PARTICLES,
        bounds=None,
        discrete_indices=None,
        w_start=PSO_W_START,
        w_end=PSO_W_END,
        alpha=PSO_W_ALPHA,
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
        super().__init__(
            n_particles=n_particles,
            bounds=bounds,
            discrete_indices=discrete_indices,
            w=w_start,
            c1=c1,
            c2=c2,
            max_iter=max_iter,
            early_stop_iters=early_stop_iters,
            tol=tol,
            boundary_method=boundary_method,
            maximise=maximise,
            seed=seed,
            v_clamp=v_clamp,
        )
        self.w_start = w_start
        self.w_end = w_end
        self.alpha = alpha

    def _get_inertia(self, t):
        progress = t / self.max_iter
        return self.w_end + (self.w_start - self.w_end) * (1 - progress) ** self.alpha
