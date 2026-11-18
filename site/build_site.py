"""Build the interactive results explorer (docs/index.html) from the raw CSVs in results/.

    python site/build_site.py

Every number on the page is aggregated here from results/*.csv (mean and 95% CI across seeds),
then embedded as JSON into site/template.html. The output is one self-contained HTML file with no
external dependencies, served by GitHub Pages from the docs/ folder.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from dqlab.stats import mean_ci  # noqa: E402

RES = ROOT / "results"


def ci(values, scale=100.0, nd=2):
    m, h = mean_ci(list(values))
    return [round(m * scale, nd), round(h * scale, nd)]


def label_noise():
    df = pd.read_csv(RES / "label_noise.csv")
    out = {}
    for (ds, model), g in df.groupby(["dataset", "model"]):
        key = f"{ds}|{model}"
        methods = {}
        for m, gm in g.groupby("method"):
            pts = []
            for r, gr in gm.groupby("noise_rate"):
                pt = {"rate": round(r * 100), "acc": ci(gr.test_acc)}
                if r > 0 and m in ("argmax_prune", "class_threshold_prune"):
                    pt["precision"] = ci(gr.precision, 1, 3)
                    pt["recall"] = ci(gr.recall, 1, 3)
                    pt["f1"] = ci(gr.f1, 1, 3)
                pts.append(pt)
            methods[m] = sorted(pts, key=lambda p: p["rate"])
        out[key] = {"dataset": ds, "model": model, "clean": round(g.clean_acc.mean() * 100, 2), "methods": methods}
    return out


def missing():
    df = pd.read_csv(RES / "missing_values.csv")
    out = {}
    for (ds, mech, rate), g in df.groupby(["dataset", "mechanism", "missing_rate"]):
        rows = {}
        for imp in ["mean", "median", "knn", "iterative"]:
            a = g[(g.imputer == imp) & (~g.indicator)]
            b = g[(g.imputer == imp) & (g.indicator)]
            rows[imp] = {"acc": ci(a.test_acc), "acc_ind": ci(b.test_acc), "rmse": ci(a.impute_rmse, 1, 3)}
        h = g[g.imputer.str.startswith("none")]
        out[f"{ds}|{mech}|{round(rate * 100)}"] = {
            "clean": round(g.clean_acc.mean() * 100, 2), "imputers": rows, "hgb": ci(h.test_acc)}
    return out


def ablation():
    df = pd.read_csv(RES / "ablation_folds.csv")
    return {ds: [{"folds": int(k), "f1": ci(gk.f1, 1, 3), "seconds": round(gk.seconds.mean(), 3)}
                 for k, gk in g.groupby("folds")] for ds, g in df.groupby("dataset")}


def errors():
    df = pd.read_csv(RES / "error_confidence.csv")
    return {ds: [{"group": r.group, "count": int(r["count"]), "given": round(r.mean_given_label_conf, 3),
                  "true": round(r.mean_true_label_conf, 3)} for _, r in g.iterrows()]
            for ds, g in df.groupby("dataset")}


def main():
    ln = pd.read_csv(RES / "label_noise.csv")
    mv = pd.read_csv(RES / "missing_values.csv")
    ab = pd.read_csv(RES / "ablation_folds.csv")
    meta = json.loads((RES / "run_metadata.json").read_text())
    data = {
        "meta": {"configs": len(ln) + len(mv) + len(ab), "seeds": len(meta["seeds"]),
                 "runtime": meta["runtime_minutes"], "sklearn": meta["scikit_learn"],
                 "max_rows": meta["synthetic_n_logreg"], "datasets": 4},
        "label_noise": label_noise(),
        "missing": missing(),
        "ablation": ablation(),
        "errors": errors(),
    }
    html = (ROOT / "site" / "template.html").read_text(encoding="utf-8")
    html = html.replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")))
    out = ROOT / "docs" / "index.html"
    out.parent.mkdir(exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out.relative_to(ROOT)} ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
