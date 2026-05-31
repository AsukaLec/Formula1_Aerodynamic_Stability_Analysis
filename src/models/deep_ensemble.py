import numpy as np
import os
import pickle
import torch

from src.models.mlp import MLPTrainer
from src.utils.config import DEEP_ENSEMBLE_M, DEEP_ENSEMBLE_DIR, RANDOM_STATE


class DeepEnsemble:
    def __init__(self, input_dim=5, hidden_units=None, M=None,
                 lr=1e-3, weight_decay=1e-5, batch_size=256,
                 max_epochs=200, patience=20):
        self.M = M or DEEP_ENSEMBLE_M
        self.trainers = []
        self.input_dim = input_dim
        self.hidden_units = hidden_units
        self.name = "M6_DeepEnsemble"

        for i in range(self.M):
            trainer = MLPTrainer(
                input_dim=input_dim, hidden_units=hidden_units,
                lr=lr, weight_decay=weight_decay,
                batch_size=batch_size, max_epochs=max_epochs,
                patience=patience, seed=RANDOM_STATE + i * 100,
            )
            self.trainers.append(trainer)

    def train(self, X, y, X_val=None, y_val=None, sample_weight=None):
        for i, trainer in enumerate(self.trainers):
            trainer.train(X, y, X_val, y_val, sample_weight=sample_weight)

    def predict(self, X):
        preds = np.column_stack([t.predict(X) for t in self.trainers])
        return preds.mean(axis=1)

    def predict_with_uncertainty(self, X):
        preds = np.column_stack([t.predict(X) for t in self.trainers])
        mu = preds.mean(axis=1)
        sigma = preds.std(axis=1, ddof=1)
        return mu, sigma

    def eval(self):
        for t in self.trainers:
            t.eval()
        return self

    def save(self, output_dir=None):
        output_dir = output_dir or DEEP_ENSEMBLE_DIR
        os.makedirs(output_dir, exist_ok=True)
        for i, trainer in enumerate(self.trainers):
            path = os.path.join(output_dir, f"mlp_{i}.pt")
            torch.save(trainer.model.state_dict(), path)

    @classmethod
    def load(cls, input_dim=5, hidden_units=None, M=None,
             load_dir=None, **trainer_kwargs):
        obj = cls(input_dim=input_dim, hidden_units=hidden_units, M=M,
                  **trainer_kwargs)
        load_dir = load_dir or DEEP_ENSEMBLE_DIR
        for i, trainer in enumerate(obj.trainers):
            path = os.path.join(load_dir, f"mlp_{i}.pt")
            trainer.model.load_state_dict(torch.load(path, map_location=trainer.device, weights_only=True))
        return obj
