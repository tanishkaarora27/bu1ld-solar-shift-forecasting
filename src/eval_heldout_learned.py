"""STAGE 2: score the FROZEN learned model on held-out sites (once). Refuses to run unless the freeze
file is committed and the working tree is clean, so the freeze SHA is on record before scoring.
Usage: python -m src.eval_heldout_learned
"""
import json
import subprocess

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .common import ROOT, load_contract, metrics, stack_sites
from .learned_model import build, ensemble_predict


def git(*a):
    return subprocess.check_output(["git", *a], cwd=ROOT, text=True).strip()


def main():
    assert git("ls-files", "learned_model_freeze.json"), "freeze file is not committed"
    assert not git("status", "--porcelain"), "working tree not clean: commit the freeze first"
    freeze_sha = git("rev-parse", "HEAD")
    fz = json.loads((ROOT / "learned_model_freeze.json").read_text())
    c = load_contract()
    sel = fz["selected"]
    train = stack_sites(c["sites"]["train"], c, "train")
    val = stack_sites(c["sites"]["validation"], c, "validation")
    held = stack_sites(c["sites"]["heldout"], c, "heldout")
    Xtr = np.vstack([p[1] for p in train]); ytr = np.concatenate([p[2] for p in train])

    models = [build(sel["hidden"], sel["alpha"], s, fz["compute_cap"]["max_iter_epochs_per_fit"]).fit(Xtr, ytr)
              for s in fz["seeds"]]
    ridge = make_pipeline(StandardScaler(), Ridge(alpha=0.01)).fit(Xtr, ytr)

    rows = []
    for split, parts in [("train", train), ("validation", val), ("heldout", held)]:
        for site, X, y, pers, _ in parts:
            for name, pred in [("persistence", pers), ("ridge", ridge.predict(X)),
                               ("learned_mlp", ensemble_predict(models, X))]:
                rows.append({"split": split, "site": site, "model": name, **metrics(y, pred, pers)})
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results" / "learned_metrics_per_site.csv", index=False)
    summ = df.groupby(["split", "model"])[["MAE", "RMSE", "bias", "skill_vs_persistence"]].mean().round(4)
    summ.to_csv(ROOT / "results" / "learned_metrics_summary.csv")
    (ROOT / "results" / "learned_run_info.json").write_text(json.dumps(
        {"freeze_commit_sha": freeze_sha, "selected": sel, "heldout_scored_after_freeze": True}, indent=2))
    print(summ); print("freeze SHA:", freeze_sha)


if __name__ == "__main__":
    main()
