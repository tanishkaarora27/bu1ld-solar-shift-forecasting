"""Compact learned model (sklearn MLP) under the frozen contract. Shared definition for both stages."""
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def build(hidden, alpha, seed, max_iter):
    # Fixed compute cap: max_iter epochs, no early stopping (no held-out-dependent stopping rule).
    mlp = MLPRegressor(hidden_layer_sizes=tuple(hidden), alpha=alpha, activation="relu", solver="adam",
                       learning_rate_init=1e-3, batch_size=256, max_iter=max_iter, early_stopping=False,
                       n_iter_no_change=max_iter + 1, tol=0.0, random_state=seed)
    return make_pipeline(StandardScaler(), mlp)  # scaler fit on train sites only


def ensemble_predict(models, X):
    return np.mean([m.predict(X) for m in models], axis=0)
