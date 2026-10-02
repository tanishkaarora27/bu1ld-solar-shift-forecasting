"""Persistence + Ridge baselines under the frozen geographic-shift contract.

Model selection uses VALIDATION sites only. Held-out sites are evaluated once, after selection.
Usage: python -m src.run_baselines
"""
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .common import ROOT, load_contract, metrics, stack_sites

OUT = ROOT / "results"


def main():
    c = load_contract()
    train = stack_sites(c["sites"]["train"], c, "train")
    val = stack_sites(c["sites"]["validation"], c, "validation")
    held = stack_sites(c["sites"]["heldout"], c, "heldout")

    Xtr = np.vstack([p[1] for p in train])
    ytr = np.concatenate([p[2] for p in train])

    best_alpha, best_mae = None, np.inf
    sel_log = []
    for a in c["baselines"]["ridge"]["alpha_grid"]:
        m = make_pipeline(StandardScaler(), Ridge(alpha=a)).fit(Xtr, ytr)
        mean_mae = float(np.mean([np.mean(np.abs(m.predict(X) - y)) for _, X, y, _, _ in val]))
        sel_log.append({"alpha": a, "val_mean_MAE": mean_mae})
        if mean_mae < best_mae:
            best_alpha, best_mae = a, mean_mae
    model = make_pipeline(StandardScaler(), Ridge(alpha=best_alpha)).fit(Xtr, ytr)

    rows, worst = [], []
    for split, parts in [("train", train), ("validation", val), ("heldout", held)]:
        for site, X, y, pers, dates in parts:
            ridge_pred = model.predict(X)
            for name, pred in [("persistence", pers), ("ridge", ridge_pred)]:
                rows.append({"split": split, "site": site, "model": name, **metrics(y, pred, pers)})
            if split == "heldout":  # keep failure cases visible
                err = np.abs(ridge_pred - y)
                for j in np.argsort(-err)[:10]:
                    worst.append({"site": site, "date": str(dates[j].date()), "y": y[j],
                                  "ridge": ridge_pred[j], "persistence": pers[j], "abs_err_ridge": err[j]})

    OUT.mkdir(exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "baseline_metrics_per_site.csv", index=False)
    pd.DataFrame(worst).to_csv(OUT / "heldout_worst_cases_ridge.csv", index=False)
    summary = df.groupby(["split", "model"])[["MAE", "RMSE", "bias", "skill_vs_persistence"]].mean().round(4)
    summary.to_csv(OUT / "baseline_metrics_summary.csv")
    (OUT / "model_selection.json").write_text(json.dumps(
        {"rule": c["model_selection_rule"], "selected_alpha": best_alpha, "grid": sel_log}, indent=2))
    print(summary)
    print("selected ridge alpha:", best_alpha)


if __name__ == "__main__":
    main()
