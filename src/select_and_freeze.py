"""STAGE 1: choose the compact learned model using TRAIN + VALIDATION sites only; write the freeze file.

This script never loads held-out sites. Commit its output (learned_model_freeze.json) BEFORE running
src/eval_heldout_learned.py. Usage: python -m src.select_and_freeze
"""
import json
import time

import numpy as np

from .common import ROOT, load_contract, stack_sites
from .learned_model import build, ensemble_predict

GRID = {"hidden": [[16], [32], [64], [32, 16]], "alpha": [1e-4, 1e-2]}
MAX_ITER = 100  # compute cap (epochs per fit)


def complexity(h):
    return (sum(h), len(h))  # simpler = fewer units, then fewer layers


def main():
    c = load_contract()
    train = stack_sites(c["sites"]["train"], c, "train")
    val = stack_sites(c["sites"]["validation"], c, "validation")  # NOTE: no held-out access here
    Xtr = np.vstack([p[1] for p in train]); ytr = np.concatenate([p[2] for p in train])

    log, t0 = [], time.time()
    for h in GRID["hidden"]:
        for a in GRID["alpha"]:
            models = [build(h, a, s, MAX_ITER).fit(Xtr, ytr) for s in c["seeds"]]
            mae = float(np.mean([np.mean(np.abs(ensemble_predict(models, X) - y)) for _, X, y, _, _ in val]))
            log.append({"hidden": h, "alpha": a, "val_mean_MAE": mae})
            print(h, a, round(mae, 4))
    best = min(log, key=lambda r: (round(r["val_mean_MAE"], 6), complexity(r["hidden"]), r["alpha"]))
    out = {
        "freeze_version": "1.0",
        "contract_file": "experiment_contract.json (unchanged, commit a1194ca93f718a76a97dbc92b5bb88005f3f15d2)",
        "model": "sklearn MLPRegressor ensemble (mean of one model per seed), StandardScaler fit on train sites",
        "inputs": "identical to Ridge: flattened 7-day history of target+covariates + sin/cos day-of-year",
        "seeds": c["seeds"],
        "compute_cap": {"max_iter_epochs_per_fit": MAX_ITER, "batch_size": 256, "early_stopping": False,
                        "fits_in_selection": len(log) * len(c["seeds"])},
        "search_grid": GRID,
        "selection_rule": "lowest mean validation-site MAE of seed-ensemble; ties (to 1e-6) to fewer units, then fewer layers, then smaller alpha. Held-out never used.",
        "selection_log": log,
        "selected": {"hidden": best["hidden"], "alpha": best["alpha"], "val_mean_MAE": best["val_mean_MAE"]},
        "comparators_fixed": ["persistence", "ridge (alpha=0.01 from baseline commit)"],
        "reporting_note": "A null or worse learned-model result is reported as-is in the final artifact.",
        "heldout_evaluated": False,
    }
    (ROOT / "learned_model_freeze.json").write_text(json.dumps(out, indent=2))
    print("selected:", out["selected"], f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
