"""Download NASA POWER daily data for every site in the contract and write an immutable manifest.

Usage:
    python -m src.build_data            # fetch + write data/raw/*.csv + data/MANIFEST.json
    python -m src.build_data --verify   # check committed CSVs against MANIFEST.json
"""
import argparse
import hashlib
import json
import time
from datetime import date

import pandas as pd
import requests

from .common import ROOT, data_dir, load_contract

MANIFEST = ROOT / "data" / "MANIFEST.json"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch_site(name, lat, lon, contract, retries=4):
    params = [contract["target"]] + contract["covariates"]
    r = contract["date_range"]
    query = {
        "parameters": ",".join(params),
        "community": contract["data_source"]["community"],
        "longitude": lon,
        "latitude": lat,
        "start": r["start"].replace("-", ""),
        "end": r["end"].replace("-", ""),
        "format": "JSON",
    }
    for attempt in range(retries):
        try:
            resp = requests.get(contract["data_source"]["endpoint"], params=query, timeout=120)
            resp.raise_for_status()
            data = resp.json()["properties"]["parameter"]
            df = pd.DataFrame(data)
            df.index = pd.to_datetime(df.index, format="%Y%m%d")
            df.index.name = "date"
            return df[params], resp.url
        except Exception:  # noqa: BLE001
            if attempt == retries - 1:
                raise
            time.sleep(5 * (attempt + 1))


def build():
    c = load_contract()
    out = data_dir()
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"fetched_on": date.today().isoformat(), "contract_version": c["contract_version"], "files": {}}
    for split, sites in c["sites"].items():
        for name, (lat, lon) in sites.items():
            df, url = fetch_site(name, lat, lon, c)
            path = out / f"{name}.csv"
            df.to_csv(path)
            manifest["files"][name] = {
                "split": split, "lat": lat, "lon": lon, "rows": len(df),
                "missing_cells": int((df == c["missing_value_code"]).sum().sum()),
                "sha256": sha256(path), "request_url": url,
            }
            print(f"{split:10s} {name:10s} rows={len(df)}")
    MANIFEST.write_text(json.dumps(manifest, indent=2))


def verify():
    m = json.loads(MANIFEST.read_text())
    bad = [n for n, f in m["files"].items() if sha256(data_dir() / f"{n}.csv") != f["sha256"]]
    if bad:
        raise SystemExit(f"Manifest mismatch for: {bad}")
    print(f"OK: {len(m['files'])} files match MANIFEST.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    verify() if args.verify else build()
