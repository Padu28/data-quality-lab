"""Run every experiment and write raw results to results/*.csv.

    python run.py            # full study (about 30-40 min on a 2-core laptop)
    python run.py --quick    # small smoke run, used in CI (under 1 min)
    python report.py         # turn the CSVs into results/REPORT.md and figures
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import pandas as pd
import sklearn

from dqlab import data, experiments as ex

FULL = {
    "seeds": [0, 1, 2, 3, 4],
    "large_seeds": [0, 1, 2],
    "noise_rates": [0.0, 0.1, 0.2, 0.3, 0.4],
    "missing_rates": [0.1, 0.3],
    "synthetic_n_logreg": 100_000,   # the large-scale run
    "synthetic_n_hgb": 20_000,
    "synthetic_n_missing": 10_000,   # KNN imputation is O(n^2), so this one is smaller
    "ablation_rate": 0.2,
}
QUICK = {
    "seeds": [0, 1],
    "large_seeds": [0],
    "noise_rates": [0.0, 0.2],
    "missing_rates": [0.2],
    "synthetic_n_logreg": 3_000,
    "synthetic_n_hgb": 1_000,
    "synthetic_n_missing": 1_000,
    "ablation_rate": 0.2,
}
SMALL = ["breast_cancer", "digits", "wine"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="tiny configuration for smoke tests / CI")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    cfg = QUICK if args.quick else FULL
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()

    # 1. label noise ----------------------------------------------------------------------------
    rows = []
    for name in SMALL + ["synthetic"]:
        X, y, _ = data.load(name, n_synthetic=cfg["synthetic_n_logreg"])
        seeds = cfg["large_seeds"] if name == "synthetic" else cfg["seeds"]
        log(f"label noise | logreg | {name} {X.shape}")
        rows += ex.label_noise(X, y, name, "logreg", cfg["noise_rates"], seeds)
    for name in SMALL + ["synthetic"]:
        X, y, _ = data.load(name, n_synthetic=cfg["synthetic_n_hgb"])
        seeds = cfg["large_seeds"] if name == "synthetic" else cfg["seeds"]
        rates = [r for r in cfg["noise_rates"] if r > 0] if name == "synthetic" else cfg["noise_rates"]
        log(f"label noise | hgb    | {name} {X.shape}")
        rows += ex.label_noise(X, y, name, "hgb", rates, seeds)
    pd.DataFrame(rows).to_csv(out / "label_noise.csv", index=False)

    # 2. missing values -------------------------------------------------------------------------
    rows = []
    for name in SMALL + ["synthetic"]:
        X, y, _ = data.load(name, n_synthetic=cfg["synthetic_n_missing"])
        seeds = cfg["large_seeds"] if name == "synthetic" else cfg["seeds"]
        log(f"missing values | {name} {X.shape}")
        rows += ex.missing_values(X, y, name, cfg["missing_rates"], seeds)
    pd.DataFrame(rows).to_csv(out / "missing_values.csv", index=False)

    # 3. ablation + error analysis --------------------------------------------------------------
    rows, per_class, conf = [], [], []
    for name in SMALL + ["synthetic"]:
        X, y, _ = data.load(name, n_synthetic=cfg["synthetic_n_logreg"])
        seeds = cfg["large_seeds"] if name == "synthetic" else cfg["seeds"]
        log(f"ablation + error analysis | {name}")
        rows += ex.detector_ablation(X, y, name, "logreg", cfg["ablation_rate"], seeds)
        pc, cf = ex.error_analysis(X, y, name, "logreg", cfg["ablation_rate"])
        per_class += pc
        conf += cf
    pd.DataFrame(rows).to_csv(out / "ablation_folds.csv", index=False)
    pd.DataFrame(per_class).to_csv(out / "error_per_class.csv", index=False)
    pd.DataFrame(conf).to_csv(out / "error_confidence.csv", index=False)

    meta = {"config": "quick" if args.quick else "full", **cfg,
            "python": platform.python_version(), "scikit_learn": sklearn.__version__,
            "machine": platform.machine(), "runtime_minutes": round((time.perf_counter() - t_start) / 60, 1)}
    (out / "run_metadata.json").write_text(json.dumps(meta, indent=2))
    log(f"done in {meta['runtime_minutes']} min -> {out}/")


if __name__ == "__main__":
    main()
