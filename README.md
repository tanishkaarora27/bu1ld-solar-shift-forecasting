# bu1ld-solar-shift-forecasting

**Question (frozen):** Can a model trained on some Indian sites forecast next-day NASA POWER
solar irradiance (`ALLSKY_SFC_SW_DWN`) at geographically unseen sites, and does a compact learned
time-series model beat persistence and a simple non-neural baseline under that shift?

This first commit is the **contract + baseline gate only**. No learned time-series model, and nothing
is tuned on the held-out sites.

## Contract
Everything is frozen in [`experiment_contract.json`](experiment_contract.json): sites (train / validation /
held-out), date range, target, covariates, missing-data handling, 7-day history, 1-day horizon,
metrics, seeds, and the model-selection rule (validation sites only).

## Reproduce
```bash
pip install -r requirements.txt
python -m src.build_data --verify     # raw CSVs are committed; checks them against data/MANIFEST.json
python -m src.run_baselines           # persistence + Ridge, per-site metrics in results/
python -m pytest tests                # contract sanity checks (disjoint sites, no temporal overlap)
```
To re-download from NASA POWER instead: `python -m src.build_data` (rewrites the manifest).

## Outputs (`results/`)
- `baseline_metrics_per_site.csv`: MAE, RMSE, bias, skill vs persistence for every site and model
- `baseline_metrics_summary.csv`: unweighted mean across sites per split
- `model_selection.json`: Ridge alpha grid scored on validation sites only
- `heldout_worst_cases_ridge.csv`: worst held-out days, kept on purpose (failure cases are not cleaned away)

## Baseline results
Mean across sites in each split; MAE and RMSE in kWh/m^2/day, next-day horizon, 7-day history.
Ridge alpha selected on validation sites only (alpha = 0.01).

| Split | Model | MAE | RMSE | Bias | MAE skill vs persistence |
|---|---|---|---|---|---|
| Train | Persistence | 0.5815 | 0.8558 | 0.0007 | 0.0 |
| Train | Ridge | 0.5360 | 0.7535 | 0.0000 | 0.0752 |
| Validation | Persistence | 0.6918 | 1.0148 | -0.0033 | 0.0 |
| Validation | Ridge | 0.6346 | 0.9008 | 0.0551 | 0.0830 |
| Held-out | Persistence | 0.6779 | 0.9799 | -0.0006 | 0.0 |
| Held-out | Ridge | 0.6700 | 0.8990 | -0.1056 | 0.0062 |

**Observations.** Ridge's MAE gain over persistence (about 8% on train and validation sites) almost
disappears on the held-out sites (0.6%), while its RMSE gain is partly retained (about 8%). Ridge also
underpredicts at the held-out sites (bias -0.11). Ridge is insensitive to alpha in the grid (validation MAE
differs by about 0.0002 for alpha 0.01 to 10), so alpha = 0.01 was picked as the strict minimum. Worst
held-out days are kept in `results/heldout_worst_cases_ridge.csv`. No learned time-series model has been
trained yet; this commit is the baseline gate only.
