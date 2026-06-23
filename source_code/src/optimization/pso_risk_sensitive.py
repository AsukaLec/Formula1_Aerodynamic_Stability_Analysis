import numpy as np

from src.optimization.pso_base import PSOBase
from src.optimization.pso_adaptive import PSOAdaptive
from src.optimization.fitness import (
    FitnessRiskSensitive,
    FitnessDensityPenalty,
    load_ensemble_fitness,
    load_xgb_fitness,
)
from src.utils.config import (
    RANDOM_STATE, PSO_PARAM_BOUNDS, PSO_DISCRETE_INDICES,
    PSO_N_PARTICLES, PSO_W_START, PSO_W_END, PSO_W_ALPHA,
    PSO_C1, PSO_C2, PSO_MAX_ITER, PSO_EARLY_STOP_ITERS, PSO_TOL,
    PSO_RISK_LAMBDA_VALUES, PSO_DENSITY_LAMBDA, PSO_DENSITY_K,
)


DEFAULT_BOUNDS = np.array(PSO_PARAM_BOUNDS, dtype=np.float64)
DISCRETE_INDICES = list(PSO_DISCRETE_INDICES)


class PSORiskSensitive:
    """Risk-sensitive PSO runner combining AdaptivePSO with uncertainty-aware fitness.

    Supports sweeping over multiple lambda values:
        lambda=0  -> baseline (no risk penalty)
        lambda>0  -> penalise uncertain regions (μ - λ*σ)
    """

    def __init__(
        self,
        n_particles=PSO_N_PARTICLES,
        bounds=None,
        w_start=PSO_W_START,
        w_end=PSO_W_END,
        alpha=PSO_W_ALPHA,
        c1=PSO_C1,
        c2=PSO_C2,
        max_iter=PSO_MAX_ITER,
        early_stop_iters=PSO_EARLY_STOP_ITERS,
        tol=PSO_TOL,
        seed=RANDOM_STATE,
        maximise=True,
        use_adaptive=True,
    ):
        self.bounds = np.asarray(bounds, dtype=np.float64) if bounds is not None else DEFAULT_BOUNDS
        self.pso_kwargs = dict(
            n_particles=n_particles,
            bounds=self.bounds,
            discrete_indices=DISCRETE_INDICES,
            c1=c1, c2=c2,
            max_iter=max_iter,
            early_stop_iters=early_stop_iters,
            tol=tol,
            seed=seed,
            maximise=maximise,
        )
        self.pso_class = PSOAdaptive if use_adaptive else PSOBase
        if use_adaptive:
            self.pso_kwargs.update(w_start=w_start, w_end=w_end, alpha=alpha)
        else:
            self.pso_kwargs.update(w=w_start)

    def optimize(self, fitness_fn, verbose=True):
        pso = self.pso_class(**self.pso_kwargs)
        return pso.optimize(fitness_fn, verbose=verbose)

    @classmethod
    def sweep_lambda(
        cls,
        lambda_values=None,
        n_particles=PSO_N_PARTICLES,
        max_iter=PSO_MAX_ITER,
        w_start=PSO_W_START,
        w_end=PSO_W_END,
        seed=RANDOM_STATE,
        use_xgb_baseline=True,
        verbose=True,
    ):
        """Sweep across multiple lambda values and compare results.

        Returns dict: {lambda: result_dict} for each lambda, plus optional 'xgb_baseline'.
        """
        if lambda_values is None:
            lambda_values = PSO_RISK_LAMBDA_VALUES

        bounds = DEFAULT_BOUNDS
        results = {}

        if use_xgb_baseline:
            if verbose:
                print("\n" + "=" * 60)
                print("  Baseline: XGBoost (no uncertainty)")
                print("=" * 60)
            xgb_fitness = load_xgb_fitness()
            pso = PSOAdaptive(
                n_particles=n_particles, bounds=bounds,
                discrete_indices=DISCRETE_INDICES,
                w_start=w_start, w_end=w_end, max_iter=max_iter, seed=seed,
                maximise=True,
            )
            results["xgb_baseline"] = pso.optimize(xgb_fitness, verbose=verbose)

        for lam in lambda_values:
            if verbose:
                print("\n" + "=" * 60)
                print(f"  Risk-Sensitive PSO: lambda={lam}")
                print("=" * 60)
            fitness = load_ensemble_fitness(lambda_risk=lam)
            pso = PSOAdaptive(
                n_particles=n_particles, bounds=bounds,
                discrete_indices=DISCRETE_INDICES,
                w_start=w_start, w_end=w_end, max_iter=max_iter, seed=seed,
                maximise=True,
            )
            results[f"risk_lambda_{lam}"] = pso.optimize(fitness, verbose=verbose)

        return results

    @classmethod
    def run_with_density_penalty(
        cls,
        lambda_risk=1.0,
        lambda_density=PSO_DENSITY_LAMBDA,
        k=PSO_DENSITY_K,
        n_particles=PSO_N_PARTICLES,
        max_iter=PSO_MAX_ITER,
        w_start=PSO_W_START,
        w_end=PSO_W_END,
        seed=RANDOM_STATE,
        verbose=True,
    ):
        """Run PSO with density-augmented fitness: μ(x) - λ1*σ(x) - λ2*D(x)."""
        import os, pickle
        from src.utils.config import PROCESSED_DIR, MLP_HIDDEN_UNITS, NN_LR, NN_WEIGHT_DECAY, NN_BATCH_SIZE, NN_MAX_EPOCHS, NN_EARLY_STOP_PATIENCE, DEEP_ENSEMBLE_DIR, DEEP_ENSEMBLE_M
        from src.models.deep_ensemble import DeepEnsemble
        from src.optimization.fitness import ModelWrapper

        bounds = DEFAULT_BOUNDS

        # Load training data for density reference
        X_train = np.load(os.path.join(PROCESSED_DIR, "X_train_mm.npy")).astype(np.float32)

        ensemble = DeepEnsemble.load(
            input_dim=5, hidden_units=MLP_HIDDEN_UNITS, M=DEEP_ENSEMBLE_M,
            lr=NN_LR, weight_decay=NN_WEIGHT_DECAY,
            batch_size=NN_BATCH_SIZE, max_epochs=NN_MAX_EPOCHS,
            patience=NN_EARLY_STOP_PATIENCE,
        )
        with open(os.path.join(PROCESSED_DIR, "scaler_mm.pkl"), "rb") as f:
            scaler = pickle.load(f)
        wrapper = ModelWrapper(ensemble, scaler, model_type="ensemble", y_transform="x100")

        fitness = FitnessDensityPenalty(
            wrapper,
            lambda_risk=lambda_risk,
            lambda_density=lambda_density,
            k=k,
            X_train=X_train,
        )

        pso = PSOAdaptive(
            n_particles=n_particles, bounds=bounds,
            discrete_indices=DISCRETE_INDICES,
            w_start=w_start, w_end=w_end, max_iter=max_iter, seed=seed,
            maximise=True,
        )
        return pso.optimize(fitness, verbose=verbose)
