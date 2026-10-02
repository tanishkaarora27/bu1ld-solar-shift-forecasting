import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load_contract():
    return json.loads((ROOT / "experiment_contract.json").read_text())


def data_dir():
    return Path(os.environ.get("SOLAR_DATA_DIR", ROOT / "data" / "raw"))


def load_site(name, contract):
    cols = [contract["target"]] + contract["covariates"]
    df = pd.read_csv(data_dir() / f"{name}.csv", parse_dates=["date"]).set_index("date").sort_index()
    df = df[cols].replace(contract["missing_value_code"], np.nan).asfreq("D")
    df = df.interpolate(limit=contract["max_interp_gap_days"], limit_area="inside")
    return df


def make_windows(df, contract, start, end):
    """One sample per target date in [start, end].

    Inputs use only days <= target_date - horizon (history may start before `start`,
    but targets never do). Returns X, y, persistence prediction, target dates.
    """
    H = contract["history_window_days"]
    h = contract["horizon_days"]
    vals = df.to_numpy(dtype=float)
    idx = df.index
    X, y, pers, dates = [], [], [], []
    for i in range(H - 1, len(df) - h):
        tdate = idx[i + h]
        if tdate < pd.Timestamp(start) or tdate > pd.Timestamp(end):
            continue
        hist = vals[i - H + 1 : i + 1]
        target = vals[i + h, 0]
        if np.isnan(hist).any() or np.isnan(target):
            continue
        doy = tdate.dayofyear
        cal = [np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)]
        X.append(np.concatenate([hist.ravel(), cal]))
        y.append(target)
        pers.append(vals[i, 0])
        dates.append(tdate)
    return np.array(X), np.array(y), np.array(pers), np.array(dates)


def stack_sites(site_names, contract, split):
    start, end = contract["splits_by_target_date"][split]
    parts = []
    for s in site_names:
        X, y, p, d = make_windows(load_site(s, contract), contract, start, end)
        parts.append((s, X, y, p, d))
    return parts


def metrics(y, pred, pers_pred):
    err = pred - y
    mae = float(np.mean(np.abs(err)))
    pmae = float(np.mean(np.abs(pers_pred - y)))
    return {
        "n": int(len(y)),
        "MAE": mae,
        "RMSE": float(np.sqrt(np.mean(err**2))),
        "bias": float(np.mean(err)),
        "skill_vs_persistence": float(1 - mae / pmae) if pmae > 0 else float("nan"),
    }
