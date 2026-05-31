import numpy as np
import time
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mse = mean_squared_error(y_true, y_pred)
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "MSE": float(mse),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mse)),
    }


def compute_regional_metrics(y_true, y_pred, threshold=95.0):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    low_mask = y_true < threshold
    high_mask = y_true >= threshold
    result = {
        "R2_low": float(r2_score(y_true[low_mask], y_pred[low_mask])) if low_mask.sum() > 1 else float("nan"),
        "MSE_low": float(mean_squared_error(y_true[low_mask], y_pred[low_mask])) if low_mask.sum() > 0 else float("nan"),
        "R2_high": float(r2_score(y_true[high_mask], y_pred[high_mask])) if high_mask.sum() > 1 else float("nan"),
        "MSE_high": float(mean_squared_error(y_true[high_mask], y_pred[high_mask])) if high_mask.sum() > 0 else float("nan"),
    }
    return result


def measure_inference_latency(model, X_sample, warmup=100, repeats=1000):
    if hasattr(model, "eval"):
        model.eval()
    import torch
    if isinstance(X_sample, np.ndarray):
        X_input = torch.from_numpy(X_sample).float()
    else:
        X_input = X_sample

    for _ in range(warmup):
        _ = model.predict(X_sample)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        _ = model.predict(X_sample)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return {"latency_ms": round((elapsed / repeats) * 1000, 3)}
