import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.utils.config import (
    RANDOM_STATE, NN_BATCH_SIZE, NN_MAX_EPOCHS, NN_EARLY_STOP_PATIENCE,
    NN_LR, NN_WEIGHT_DECAY, MLP_HIDDEN_UNITS, MLP_DROPOUT,
)

_ = torch.manual_seed(RANDOM_STATE)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_units=None, dropout=MLP_DROPOUT):
        super().__init__()
        units = hidden_units or MLP_HIDDEN_UNITS
        layers = []
        prev = input_dim
        for h in units:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class MLPTrainer:
    def __init__(self, input_dim=5, hidden_units=None, dropout=MLP_DROPOUT,
                 lr=NN_LR, weight_decay=NN_WEIGHT_DECAY,
                 batch_size=NN_BATCH_SIZE, max_epochs=NN_MAX_EPOCHS,
                 patience=NN_EARLY_STOP_PATIENCE, seed=RANDOM_STATE):
        self.input_dim = input_dim
        self.hidden_units = hidden_units or MLP_HIDDEN_UNITS
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.seed = seed

        torch.manual_seed(seed)
        self.model = MLP(input_dim, hidden_units, dropout)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self._train_losses = []
        self._val_losses = []

    def train(self, X, y, X_val=None, y_val=None, sample_weight=None):
        X_t = torch.from_numpy(np.asarray(X, dtype=np.float32))
        y_t = torch.from_numpy(np.asarray(y, dtype=np.float32))
        if sample_weight is not None:
            w_t = torch.from_numpy(np.asarray(sample_weight, dtype=np.float32))
            ds = TensorDataset(X_t, y_t, w_t)
        else:
            ds = TensorDataset(X_t, y_t)
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        best_val_loss = float("inf")
        best_state = None
        wait = 0

        for epoch in range(self.max_epochs):
            self.model.train()
            epoch_loss = 0.0
            n_batches = 0
            for batch in loader:
                if len(batch) == 3:
                    xb, yb, wb = batch
                    xb, yb, wb = xb.to(self.device), yb.to(self.device), wb.to(self.device)
                    pred = self.model(xb)
                    loss = (wb * (pred - yb) ** 2).mean()
                else:
                    xb, yb = batch
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    pred = self.model(xb)
                    loss = nn.MSELoss()(pred, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            train_loss = epoch_loss / n_batches
            self._train_losses.append(train_loss)

            val_loss = None
            if X_val is not None and y_val is not None:
                val_loss = self._eval_loss(X_val, y_val)
                self._val_losses.append(val_loss)

            if val_loss is not None:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    wait = 0
                else:
                    wait += 1
                if wait >= self.patience:
                    break
            else:
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

        if best_state is not None:
            self.model.load_state_dict(best_state)

    def _eval_loss(self, X, y):
        self.model.eval()
        with torch.no_grad():
            X_t = torch.from_numpy(np.asarray(X, dtype=np.float32)).to(self.device)
            y_t = torch.from_numpy(np.asarray(y, dtype=np.float32)).to(self.device)
            pred = self.model(X_t)
            loss = nn.MSELoss()(pred, y_t)
        return loss.item()

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            X_t = torch.from_numpy(np.asarray(X, dtype=np.float32)).to(self.device)
            pred = self.model(X_t)
        return pred.cpu().numpy()

    def eval(self):
        self.model.eval()
        return self
