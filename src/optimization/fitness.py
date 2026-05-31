import os
import pickle
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

from src.utils.config import (
    MODELS_OUTPUT_DIR, DEEP_ENSEMBLE_DIR, FEATURE_COLS, PROCESSED_DIR,
    RANDOM_STATE,
)


class ModelWrapper:
    """Unified prediction interface for tree models (StandardScaler) and NN models (MinMaxScaler)."""

    def __init__(self, model, scaler, model_type, y_transform="identity"):
        """
        Parameters
        ----------
        model : object
            Trained model with a .predict(X) method.
        scaler : object
            Scaler with .transform(X) method, or None for raw input.
        model_type : str
            One of 'xgb', 'rf', 'ridge', 'mlp', 'tabnet', 'ensemble'.
        y_transform : str or callable
            'identity' -> return prediction as-is.
            'x100'      -> multiply prediction by 100 (NN models with y/100 scaling).
            callable    -> custom transform.
        """
        self.model = model
        self.scaler = scaler
        self.model_type = model_type
        self.y_transform = y_transform
        self._is_ensemble = (model_type == "ensemble")
        if hasattr(self.model, "eval"):
            self.model.eval()

    def _scale_input(self, X):
        X = np.asarray(X, dtype=np.float32)
        if self.scaler is not None:
            X = self.scaler.transform(X)
        return X

    def predict(self, X):
        """Returns predicted stability_index for each row in X (original scale)."""
        X_scaled = self._scale_input(X)
        preds = self.model.predict(X_scaled)
        if hasattr(preds, "numpy"):
            preds = preds.numpy()
        preds = np.asarray(preds, dtype=np.float64)
        return self._apply_y_transform(preds)

    def predict_with_uncertainty(self, X):
        """Returns (mu, sigma) for each row in X (original scale). Requires ensemble model."""
        if not self._is_ensemble:
            raise ValueError("predict_with_uncertainty requires model_type='ensemble'")
        X_scaled = self._scale_input(X)
        mu_scaled, sigma_scaled = self.model.predict_with_uncertainty(X_scaled)
        mu = self._apply_y_transform(np.asarray(mu_scaled, dtype=np.float64))
        sigma = self._apply_y_transform(np.asarray(sigma_scaled, dtype=np.float64))
        return mu, sigma

    def _apply_y_transform(self, y):
        if self.y_transform == "identity" or self.y_transform is None:
            return np.asarray(y, dtype=np.float64)
        elif self.y_transform == "x100":
            return np.asarray(y, dtype=np.float64) * 100.0
        elif callable(self.y_transform):
            return np.asarray(self.y_transform(y), dtype=np.float64)
        else:
            raise ValueError(f"Unknown y_transform: {self.y_transform}")


class FitnessStandard:
    """Standard fitness: f(x) = mu(x). PSO maximizes this.
       We want HIGHER stability_index, so we return mu(x) directly."""

    def __init__(self, model_wrapper):
        self.model = model_wrapper

    def evaluate(self, X):
        return self.model.predict(X)

    def __call__(self, X):
        return self.evaluate(X)


class FitnessRiskSensitive:
    """Risk-sensitive fitness: f(x) = mu(x) - lambda * sigma(x).
       Penalises OOD/high-uncertainty regions. Requires ensemble model."""

    def __init__(self, model_wrapper, lambda_risk=1.0):
        if not model_wrapper._is_ensemble:
            raise ValueError("FitnessRiskSensitive requires an ensemble model (model_type='ensemble')")
        self.model = model_wrapper
        self.lambda_risk = lambda_risk

    def evaluate(self, X):
        mu, sigma = self.model.predict_with_uncertainty(X)
        return mu - self.lambda_risk * sigma

    def __call__(self, X):
        return self.evaluate(X)


class FitnessDensityPenalty(FitnessRiskSensitive):
    """Fitness with density penalty: f(x) = mu(x) - lambda_risk * sigma(x) - lambda_density * D(x).
       D(x) = average distance to K nearest neighbours in training set.
       Penalises regions far from training data."""

    def __init__(self, model_wrapper, lambda_risk=1.0, lambda_density=0.5,
                 k=10, X_train=None):
        super().__init__(model_wrapper, lambda_risk=lambda_risk)
        self.lambda_density = lambda_density
        self.k = k
        if X_train is not None:
            self._nn = NearestNeighbors(n_neighbors=k)
            self._nn.fit(X_train)
        else:
            self._nn = None

    def evaluate(self, X):
        base = super().evaluate(X)
        if self._nn is not None:
            dist, _ = self._nn.kneighbors(X, n_neighbors=min(self.k, self._nn._fit_X.shape[0]))
            density = dist.mean(axis=1)
        else:
            density = 0.0
        return base - self.lambda_density * density


class FitnessBoundPenalty:
    """Wrapper that adds a soft penalty for out-of-bounds positions.
       Useful when bounds enforcement during PSO is insufficient."""

    def __init__(self, base_fitness, bounds, penalty_scale=10.0):
        self.base_fitness = base_fitness
        self.bounds = np.asarray(bounds)
        self.penalty_scale = penalty_scale

    def evaluate(self, X):
        X = np.asarray(X, dtype=np.float64)
        fitness = self.base_fitness.evaluate(X)
        penalty = self._compute_bound_penalty(X)
        return fitness - penalty

    def __call__(self, X):
        return self.evaluate(X)

    def _compute_bound_penalty(self, X):
        lower = self.bounds[:, 0]
        upper = self.bounds[:, 1]
        below = np.maximum(0, lower - X)
        above = np.maximum(0, X - upper)
        return self.penalty_scale * (below.sum(axis=1) + above.sum(axis=1))


def load_xgb_fitness():
    """Load XGBoost model + StandardScaler, return FitnessStandard."""
    import xgboost as xgb
    with open(os.path.join(MODELS_OUTPUT_DIR, "M3_XGBoost.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(PROCESSED_DIR, "scaler_ss.pkl"), "rb") as f:
        scaler = pickle.load(f)
    wrapper = ModelWrapper(model, scaler, model_type="xgb", y_transform="identity")
    return FitnessStandard(wrapper)


def load_ensemble_fitness(lambda_risk=1.0):
    """Load DeepEnsemble + MinMaxScaler, return FitnessRiskSensitive."""
    from src.models.deep_ensemble import DeepEnsemble
    from src.utils.config import MLP_HIDDEN_UNITS, NN_LR, NN_WEIGHT_DECAY, NN_BATCH_SIZE, NN_MAX_EPOCHS, NN_EARLY_STOP_PATIENCE, DEEP_ENSEMBLE_M

    ensemble = DeepEnsemble.load(
        input_dim=5, hidden_units=MLP_HIDDEN_UNITS, M=DEEP_ENSEMBLE_M,
        lr=NN_LR, weight_decay=NN_WEIGHT_DECAY,
        batch_size=NN_BATCH_SIZE, max_epochs=NN_MAX_EPOCHS,
        patience=NN_EARLY_STOP_PATIENCE,
    )
    with open(os.path.join(PROCESSED_DIR, "scaler_mm.pkl"), "rb") as f:
        scaler = pickle.load(f)
    wrapper = ModelWrapper(ensemble, scaler, model_type="ensemble", y_transform="x100")
    return FitnessRiskSensitive(wrapper, lambda_risk=lambda_risk)


def load_rf_fitness():
    """Load RandomForest model + StandardScaler, return FitnessStandard."""
    from sklearn.ensemble import RandomForestRegressor
    with open(os.path.join(MODELS_OUTPUT_DIR, "M2_RandomForest.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(PROCESSED_DIR, "scaler_ss.pkl"), "rb") as f:
        scaler = pickle.load(f)
    wrapper = ModelWrapper(model, scaler, model_type="rf", y_transform="identity")
    return FitnessStandard(wrapper)


def benchmark_rastrigin(dim=5):
    """Return a FitnessStandard-like callable for the Rastrigin benchmark function.

    Rastrigin: f(x) = 10*n + sum(x_i^2 - 10*cos(2*pi*x_i))
    Domain: [-5.12, 5.12]^n
    Global minimum at x=0 with f=0 (we negate so PSO maximises toward 0).
    """
    class _Rastrigin:
        def evaluate(self, X):
            X = np.asarray(X, dtype=np.float64)
            n = X.shape[1]
            return -(10 * n + (X**2 - 10 * np.cos(2 * np.pi * X)).sum(axis=1))
        def __call__(self, X):
            return self.evaluate(X)
    return _Rastrigin()


def benchmark_sphere(dim=5):
    """Sphere function: f(x) = sum(x_i^2). Global min at 0 with f=0. Negated for maximisation."""
    class _Sphere:
        def evaluate(self, X):
            X = np.asarray(X, dtype=np.float64)
            return -(X**2).sum(axis=1)
        def __call__(self, X):
            return self.evaluate(X)
    return _Sphere()


# --- Multi-Objective Composite Fitness --------------------------------------


class FitnessMultiObjective(FitnessRiskSensitive):
    """Multi-objective fitness: weighted sum of stability + efficiency - power - uncertainty.

    F(x) = w1 * stability(x)/100 + w2 * efficiency(x) - w3 * power(x) - lambda * sigma(x)

    This creates a non-trivial Pareto landscape because:
      - stability ↑  usually means  downforce ↑ → drag ↑ → efficiency ↓, power ↑
      - efficiency ↑ usually means  drag ↓ → downforce ↓ → stability ↓
      - power ↓       usually means  speed ↓ or drag ↓ → stability ↓

    The three objectives are fundamentally in tension — PSO must trade off.
    """

    def __init__(self, model_wrapper, w_stability=0.5, w_efficiency=0.3, w_power=0.2,
                 lambda_risk=1.0, eps=1e-6):
        super().__init__(model_wrapper, lambda_risk=lambda_risk)
        self.w_stability = w_stability
        self.w_efficiency = w_efficiency
        self.w_power = w_power
        self.eps = eps

        # Normalisation constants (computed once from training data)
        self._max_efficiency = None
        self._max_power = None

    def _set_norm_constants(self, max_efficiency, max_power):
        """Set normalisation constants. Called once after fitness construction."""
        self._max_efficiency = max_efficiency if max_efficiency else 1.0
        self._max_power = max_power if max_power else 1.0

    def _compute_objectives(self, X):
        """Compute the three raw objectives for each row in X.

        Returns
        -------
        stability : np.ndarray (N,)
        efficiency : np.ndarray (N,)
        power : np.ndarray (N,)
        """
        X = np.asarray(X, dtype=np.float64)
        mu, sigma = self.model.predict_with_uncertainty(X)
        stability = np.clip(mu, 0, 100)

        downforce = X[:, 3]
        drag = X[:, 4]
        speed = X[:, 0]

        efficiency = downforce / (drag + self.eps)

        if self._max_power is not None:
            power = (drag * speed) / self._max_power
        else:
            power = (drag * speed) / 1.0

        return stability, efficiency, power, sigma

    def evaluate(self, X):
        stability, efficiency, power, sigma = self._compute_objectives(X)

        eff_norm = np.clip(efficiency / (self._max_efficiency or 1.0), 0, 1)
        pow_norm = np.clip(power, 0, 1)

        fitness = (
            self.w_stability * stability / 100.0
            + self.w_efficiency * eff_norm
            - self.w_power * pow_norm
            - self.lambda_risk * sigma / 100.0
        )
        return np.asarray(fitness, dtype=np.float64)

    def evaluate_components(self, X):
        """Return (stability, efficiency_raw, power_raw, sigma, fitness)."""
        stability, efficiency, power, sigma = self._compute_objectives(X)
        fitness = self.evaluate(X)
        return stability, efficiency, power, sigma, fitness

    def __call__(self, X):
        return self.evaluate(X)
