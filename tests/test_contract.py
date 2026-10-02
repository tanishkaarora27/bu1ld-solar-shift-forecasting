import pandas as pd

from src.common import load_contract


def test_sites_disjoint():
    s = load_contract()["sites"]
    a, b, c = (set(s[k]) for k in ("train", "validation", "heldout"))
    assert not (a & b) and not (a & c) and not (b & c)


def test_train_ends_before_eval_starts():
    sp = load_contract()["splits_by_target_date"]
    assert pd.Timestamp(sp["train"][1]) < pd.Timestamp(sp["validation"][0])
    assert pd.Timestamp(sp["train"][1]) < pd.Timestamp(sp["heldout"][0])
