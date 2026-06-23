import numpy as np
import torch
from pytorch_tabnet.tab_model import TabNetRegressor

from src.utils.config import (
    RANDOM_STATE, TABNET_N_D, TABNET_N_A, TABNET_N_STEPS, TABNET_GAMMA,
    TABNET_LR, TABNET_MAX_EPOCHS, NN_BATCH_SIZE,
)


class TabNetModel:
    def __init__(self, n_d=None, n_a=None, n_steps=None, gamma=None,
                 lr=None, max_epochs=None):
        self.n_d = n_d or TABNET_N_D
        self.n_a = n_a or TABNET_N_A
        self.n_steps = n_steps or TABNET_N_STEPS
        self.gamma = gamma or TABNET_GAMMA
        self.lr = lr or TABNET_LR
        self.max_epochs = max_epochs or TABNET_MAX_EPOCHS
        self.name = "M5_TabNet"

        self.model = TabNetRegressor(
            n_d=self.n_d, n_a=self.n_a, n_steps=self.n_steps,
            gamma=self.gamma, seed=RANDOM_STATE,
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=self.lr),
        )

    def train(self, X, y, X_val=None, y_val=None, sample_weight=None):
        X_np = np.asarray(X, dtype=np.float32)
        y_np = np.asarray(y, dtype=np.float32).reshape(-1, 1)
        eval_set = None
        eval_name = None
        eval_metric = None
        if X_val is not None and y_val is not None:
            eval_set = [(np.asarray(X_val, dtype=np.float32),
                         np.asarray(y_val, dtype=np.float32).reshape(-1, 1))]
            eval_name = ["valid"]
            eval_metric = ["mse"]
        self.model.fit(
            X_np, y_np,
            eval_set=eval_set, eval_name=eval_name, eval_metric=eval_metric,
            batch_size=NN_BATCH_SIZE,
            max_epochs=self.max_epochs,
            patience=20,
            weights=sample_weight,
        )

    def predict(self, X):
        return self.model.predict(np.asarray(X, dtype=np.float32)).flatten()

    def eval(self):
        return self
