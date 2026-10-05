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


## Learned model: held-out result

Scored once at freeze commit `caeae740902f72e5225b76ee626a5a9fa0835d27` (baseline reference `a1194ca93f718a76a97dbc92b5bb88005f3f15d2`). Model: sklearn MLPRegressor ensemble (seeds 0/1/2), hidden (32, 16), alpha 1e-4, selected on validation sites only. Persistence and Ridge (alpha 0.01) are fixed comparators. Contract, seeds, compute cap and selection rule were unchanged.

Held-out, mean across 4 sites (kWh/m^2/day):

| Model | MAE | RMSE | Bias | MAE skill vs persistence |
|---|---|---|---|---|
| Persistence | 0.6779 | 0.9799 | -0.0006 | 0.0000 |
| Ridge | 0.6700 | 0.8990 | -0.1056 | 0.0062 |
| Learned MLP | 0.6633 | 0.9289 | +0.1697 | 0.0147 |

Reading: the MLP's MAE is only about 1% better than Ridge, its RMSE is worse than Ridge, and it overpredicts (bias +0.17). It beats persistence at 3 of 4 held-out sites and Ridge at 1 of 4, so this is a marginal, non-robust result. MAE skill is 0.157 on train sites but 0.015 held-out, so little of the in-sample gain transfers to unseen sites. Fits reached the frozen 100-epoch cap (sklearn ConvergenceWarning); this was left unchanged. Full output: `results/learned_metrics_per_site.csv`, `results/learned_metrics_summary.csv`, `results/learned_run_info.json`, `results/learned_run_log.txt`.

Note: `python -m src.build_data --verify` can report a mismatch on a fresh clone because the committed CSVs use LF line endings while the manifest hashes correspond to CRLF; the data content is identical.