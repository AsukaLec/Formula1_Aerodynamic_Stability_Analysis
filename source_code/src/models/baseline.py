import numpy as np
import warnings
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

from src.utils.config import RANDOM_STATE, RF_N_ESTIMATORS, RF_MAX_DEPTH, XGB_N_ESTIMATORS, XGB_MAX_DEPTH, XGB_LR

warnings.filterwarnings("ignore", category=UserWarning)


class RidgeModel:
    def __init__(self, alpha=1.0):
        self.model = Ridge(alpha=alpha, random_state=RANDOM_STATE)
        self.name = "M1_Ridge"

    def train(self, X, y, sample_weight=None):
        self.model.fit(X, y, sample_weight=sample_weight)

    def predict(self, X):
        return self.model.predict(X)


class RandomForestModel:
    def __init__(self, n_estimators=None, max_depth=None):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators or RF_N_ESTIMATORS,
            max_depth=max_depth or RF_MAX_DEPTH,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        self.name = "M2_RandomForest"

    def train(self, X, y, sample_weight=None):
        self.model.fit(X, y, sample_weight=sample_weight)

    def predict(self, X):
        return self.model.predict(X)


class XGBoostModel:
    def __init__(self, n_estimators=None, max_depth=None, learning_rate=None):
        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators or XGB_N_ESTIMATORS,
            max_depth=max_depth or XGB_MAX_DEPTH,
            learning_rate=learning_rate or XGB_LR,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0,
        )
        self.name = "M3_XGBoost"

    def train(self, X, y, sample_weight=None):
        self.model.fit(X, y, sample_weight=sample_weight)

    def predict(self, X):
        return self.model.predict(X)
