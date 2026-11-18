"""Turn results/*.csv into results/REPORT.md plus figures. Every number in the report is computed
here from the raw CSVs, so the write-up can always be regenerated from the data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from dqlab.stats import mean_ci, paired_test  # noqa: E402

METHOD_ORDER = ["no_repair", "argmax_prune", "class_threshold_prune", "oracle_prune"]
METHOD_LABEL = {"no_repair": "No repair", "argmax_prune": "Argmax disagreement + prune",
                "class_threshold_prune": "Confident learning + prune", "oracle_prune": "Oracle prune (upper bound)"}


def fmt(values, pct=True):
    m, h = mean_ci(values)
    s = 100 if pct else 1
    return f"{m * s:.1f} ± {h * s:.1f}" if pct else f"{m:.3f} ± {h:.3f}"


def label_noise_section(df, figs):
    out = ["## 1. Label noise: detect, prune, retrain\n",
           "Test accuracy (%) on the clean test set, mean ± 95% CI across seeds. "
           "p = paired t-test of confident learning vs. no repair.\n"]
    headline = {}
    for (model, dataset), g in df.groupby(["model", "dataset"]):
        rates = sorted(r for r in g.noise_rate.unique())
        out.append(f"\n**{dataset} · {model}**\n")
        out.append("| noise | " + " | ".join(METHOD_LABEL[m] for m in METHOD_ORDER) + " | p |")
        out.append("|---" * (len(METHOD_ORDER) + 2) + "|")
        for r in rates:
            gr = g[g.noise_rate == r]
            cells = []
            for m in METHOD_ORDER:
                v = gr[gr.method == m].sort_values("seed").test_acc.values
                cells.append(fmt(v) if len(v) else "–")
            a = gr[gr.method == "class_threshold_prune"].sort_values("seed").test_acc.values
            b = gr[gr.method == "no_repair"].sort_values("seed").test_acc.values
            p = paired_test(a, b) if r > 0 else np.nan
            out.append(f"| {int(r * 100)}% | " + " | ".join(cells) + f" | {p:.3f} |" if r > 0
                       else f"| {int(r * 100)}% | " + " | ".join(cells) + " | – |")
            if r > 0:
                headline[(model, dataset, r)] = (b.mean(), a.mean(),
                                                 gr[gr.method == "oracle_prune"].test_acc.mean())

    det = df[(df.noise_rate > 0) & df.method.isin(["argmax_prune", "class_threshold_prune"])]
    out.append("\n### Detection quality (does the detector find the right rows?)\n")
    out.append("| model | dataset | noise | detector | precision | recall | F1 |")
    out.append("|---|---|---|---|---|---|---|")
    for (model, dataset, r, m), g in det.groupby(["model", "dataset", "noise_rate", "method"]):
        out.append(f"| {model} | {dataset} | {int(r * 100)}% | {METHOD_LABEL[m].split(' +')[0]} | "
                   f"{fmt(g.precision.values, False)} | {fmt(g.recall.values, False)} | {fmt(g.f1.values, False)} |")

    # figure: accuracy vs noise for each dataset (logreg)
    lr = df[df.model == "logreg"]
    datasets = sorted(lr.dataset.unique())
    fig, axes = plt.subplots(1, len(datasets), figsize=(4 * len(datasets), 3.4), squeeze=False)
    for ax, ds in zip(axes[0], datasets):
        g = lr[lr.dataset == ds]
        for m in METHOD_ORDER:
            s = g[g.method == m].groupby("noise_rate").test_acc
            ax.plot(s.mean().index * 100, s.mean().values * 100, marker="o", label=METHOD_LABEL[m])
        ax.set_title(ds)
        ax.set_xlabel("label noise (%)")
        ax.grid(alpha=.3)
    axes[0][0].set_ylabel("test accuracy (%)")
    axes[0][-1].legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(figs / "label_noise_accuracy.png", dpi=130)
    plt.close(fig)
    out.append("\n![accuracy vs noise](figures/label_noise_accuracy.png)\n")
    return out, headline


def missing_section(df, figs):
    out = ["\n## 2. Missing values: which imputer, and does a missing-indicator help?\n",
           "Logistic regression test accuracy (%) and imputation RMSE (in standard deviations, "
           "on masked training cells). HGB with native NaN handling is shown as a no-imputation reference.\n"]
    for (dataset, mech, rate), g in df.groupby(["dataset", "mechanism", "missing_rate"]):
        out.append(f"\n**{dataset} · {mech} · {int(rate * 100)}% missing** "
                   f"(clean accuracy {g.clean_acc.mean() * 100:.1f}%)\n")
        out.append("| imputer | accuracy | + indicator | RMSE |")
        out.append("|---|---|---|---|")
        for imp in ["mean", "median", "knn", "iterative"]:
            a = g[(g.imputer == imp) & (~g.indicator)].sort_values("seed")
            b = g[(g.imputer == imp) & (g.indicator)].sort_values("seed")
            out.append(f"| {imp} | {fmt(a.test_acc.values)} | {fmt(b.test_acc.values)} | "
                       f"{fmt(a.impute_rmse.values, False)} |")
        h = g[g.imputer.str.startswith("none")]
        out.append(f"| HGB native NaN | {fmt(h.test_acc.values)} | – | – |")

    # figure: RMSE by imputer, MCAR vs MNAR, averaged over datasets at the highest rate
    top = df[(df.missing_rate == df.missing_rate.max()) & (~df.indicator) & df.impute_rmse.notna()]
    piv = top.groupby(["imputer", "mechanism"]).impute_rmse.mean().unstack()
    ax = piv.loc[["mean", "median", "knn", "iterative"]].plot.bar(figsize=(5.5, 3.4), rot=0)
    ax.set_ylabel("imputation RMSE (std devs)")
    ax.set_title(f"Imputation error at {int(df.missing_rate.max() * 100)}% missing (avg over datasets)")
    ax.grid(alpha=.3, axis="y")
    ax.legend(title="mechanism", loc="upper left")
    ax.set_ylim(0, piv.values.max() * 1.25)
    plt.tight_layout()
    plt.savefig(figs / "imputation_rmse.png", dpi=130)
    plt.close()
    out.append("\n![imputation error](figures/imputation_rmse.png)\n")

    # figure: the missing-indicator effect under MNAR at the highest rate (mean imputation)
    hi = df[(df.mechanism == "MNAR") & (df.missing_rate == df.missing_rate.max()) & (df.imputer == "mean")]
    g = hi.groupby(["dataset", "indicator"]).test_acc.mean().unstack() * 100
    clean = hi.groupby("dataset").clean_acc.mean() * 100
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    x = np.arange(len(g))
    ax.bar(x - 0.2, g[False], 0.4, label="mean imputation")
    ax.bar(x + 0.2, g[True], 0.4, label="mean imputation + missing indicator")
    ax.scatter(x, clean.loc[g.index], marker="_", s=600, color="black", label="no missing data", zorder=3)
    ax.set_xticks(x, g.index)
    ax.set_ylabel("test accuracy (%)")
    ax.set_ylim(max(0, g.values.min() - 10), 101)
    ax.set_title(f"MNAR, {int(df.missing_rate.max() * 100)}% missing: telling the model what was missing")
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False)
    ax.grid(alpha=.3, axis="y")
    fig.tight_layout()
    fig.savefig(figs / "mnar_indicator.png", dpi=130)
    plt.close(fig)
    out.append("\n![missing indicator under MNAR](figures/mnar_indicator.png)\n")
    return out


def ablation_section(df, figs):
    out = ["\n## 3. Ablation: number of cross-validation folds for the detector\n",
           "| dataset | folds | precision | recall | F1 | prune → accuracy (%) | seconds |",
           "|---|---|---|---|---|---|---|"]
    for (ds, k), g in df.groupby(["dataset", "folds"]):
        out.append(f"| {ds} | {k} | {fmt(g.precision.values, False)} | {fmt(g.recall.values, False)} | "
                   f"{fmt(g.f1.values, False)} | {fmt(g.test_acc.values)} | {g.seconds.mean():.2f} |")
    fig, ax = plt.subplots(figsize=(5, 3.4))
    for ds, g in df.groupby("dataset"):
        s = g.groupby("folds").f1.mean()
        ax.plot(s.index, s.values, marker="o", label=ds)
    ax.set_xlabel("folds")
    ax.set_ylabel("detection F1")
    ax.set_xticks(sorted(df.folds.unique()))
    ax.grid(alpha=.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figs / "ablation_folds.png", dpi=130)
    plt.close(fig)
    out.append("\n![folds ablation](figures/ablation_folds.png)\n")
    return out


def error_section(per_class, conf):
    out = ["\n## 4. Error analysis (20% noise, logistic regression, seed 0)\n",
           "Confidence profile: the out-of-fold probability the model gives the *given* (possibly wrong) "
           "label vs. the *true* label.\n",
           "| dataset | group | rows | mean P(given label) | mean P(true label) |", "|---|---|---|---|---|"]
    for _, r in conf.iterrows():
        out.append(f"| {r.dataset} | {r.group} | {r['count']} | {r.mean_given_label_conf:.3f} | "
                   f"{r.mean_true_label_conf:.3f} |")
    out.append("\nHardest true classes (lowest recall of flipped labels, per dataset):\n")
    out.append("| dataset | true class | flipped | caught | recall |")
    out.append("|---|---|---|---|---|")
    for ds, g in per_class.groupby("dataset"):
        for _, r in g.sort_values("recall").head(2).iterrows():
            out.append(f"| {ds} | {r.true_class} | {r.noisy} | {r.caught} | {r.recall:.2f} |")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    args = ap.parse_args()
    res = Path(args.results)
    figs = res / "figures"
    figs.mkdir(exist_ok=True)
    meta = json.loads((res / "run_metadata.json").read_text())

    ln = pd.read_csv(res / "label_noise.csv")
    mv = pd.read_csv(res / "missing_values.csv")
    ab = pd.read_csv(res / "ablation_folds.csv")
    pc = pd.read_csv(res / "error_per_class.csv")
    cf = pd.read_csv(res / "error_confidence.csv")

    lines = ["# Results\n",
             f"Generated by `report.py` from the CSVs in this folder. Config: **{meta['config']}**, "
             f"seeds {meta['seeds']} (large dataset: {meta['large_seeds']}), "
             f"scikit-learn {meta['scikit_learn']}, runtime {meta['runtime_minutes']} min.\n"]
    sec, headline = label_noise_section(ln, figs)
    lines += sec
    lines += missing_section(mv, figs)
    lines += ablation_section(ab, figs)
    lines += error_section(pc, cf)
    (res / "REPORT.md").write_text("\n".join(lines) + "\n")

    # short machine-readable summary used to keep the README in sync
    summary = {f"{m}|{d}|{r}": {"no_repair": round(a * 100, 1), "confident_learning": round(b * 100, 1),
                                "oracle": round(c * 100, 1)} for (m, d, r), (a, b, c) in headline.items()}
    (res / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {res / 'REPORT.md'}")


if __name__ == "__main__":
    main()
