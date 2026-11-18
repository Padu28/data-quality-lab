# data-quality-lab

![ci](https://github.com/Padu28/data-quality-lab/actions/workflows/ci.yml/badge.svg)

**How much do wrong labels and missing values hurt a model, and which repairs actually win the
accuracy back?** A controlled study: corrupt clean data in known ways, apply detection and repair
methods, retrain, and measure against the ground truth. 1,428 experiment configurations, 4 datasets
(up to 100,000 rows), 2 model families, 5 random seeds, and every number reported with a 95%
confidence interval. The full results are in [results/REPORT.md](results/REPORT.md).

## Key findings

**1. Finding and removing mislabelled rows recovers most of the lost accuracy.**
With 40% of training labels flipped, gradient boosting on `digits` falls from 97.5% to 86.0%.
Confident-learning detection + pruning brings it back to **93.2%** (paired t-test p < 0.001). An
oracle that knows exactly which labels are wrong reaches 96.2%. On `breast_cancer` with 30% noise,
pruning lifts accuracy from 77.5% to **89.7%** (p = 0.002).

![accuracy vs noise](results/figures/label_noise_accuracy.png)

**2. The more precise detector is not always the better repair.** Confident learning flags rows
with much higher precision than the simple "model disagrees with the label" rule (0.88 vs 0.73 on
the 20,000-row synthetic set at 20% noise). But it also misses more of them (recall 0.78 vs 0.99).
When the repair is "drop the flagged rows", missing a wrong label costs more than dropping a clean
one, so the high-recall baseline often wins downstream: logistic regression on `digits` at 20% noise
reaches **95.5%** with the baseline vs 91.7% with confident learning. The right detector depends on
what you do with its output: prune → favour recall; send to human review → favour precision.

**3. When values are missing *because* of their value, the imputer matters less than telling the
model a value was missing.** Under missing-not-at-random (MNAR) data with 30% missing, mean
imputation drops the synthetic task's accuracy from 83.6% to 63.4%. Adding missing-indicator columns
recovers **79.8% (+16.4 points)**; on `breast_cancer` it goes from 84.6% to **95.7%**. Under
missing-completely-at-random data, the same indicators slightly *hurt* (e.g. `digits`, 30% MCAR:
89.9% → 87.6%), because they add 64 uninformative features. The fix depends on why data is missing.

![missing indicator under MNAR](results/figures/mnar_indicator.png)

**4. Lower imputation error does not mean higher accuracy.** On the synthetic set with 30% MNAR,
iterative imputation reconstructs the missing values best (RMSE 1.13 std devs vs 1.58 for mean
imputation), yet the model reaches only 74.4%. Mean imputation with missing indicators, the
*worst* reconstruction, reaches 79.8%. Imputation quality and task quality need to be measured
separately.

**5. More cross-validation folds help small datasets, not large ones.** For the detector's
out-of-fold probabilities, going from 2 to 10 folds raises detection F1 on `digits` from 0.74 to
0.82. On the 100,000-row synthetic set F1 stays at 0.748 while compute rises 8× (0.45 s → 3.75 s).

**6. What the detector misses is what the model finds hard.** Wrongly-labelled rows the detector
catches get a mean out-of-fold probability of 0.62–0.83 on their *true* class. The ones it misses
get only 0.31–0.50, so they are ambiguous examples where the model itself is unsure. False alarms
are clean rows whose correct label the model rates low (0.14–0.32). This suggests better detection will come more from a stronger base model than from a smarter threshold.

**Caveat:** pruning is not free. On clean data (0% noise) it removes hard but correct examples,
e.g. gradient boosting on `digits` drops from 97.5% to 96.1% with the baseline detector. Only prune
when there is reason to expect label noise.

## What is in the study

| experiment | question | variables |
|---|---|---|
| Label noise | Detect, prune, retrain: how much accuracy comes back? | noise 0–40% · no repair / argmax rule / confident learning / oracle · logistic regression, gradient boosting |
| Missing values | Which imputer, and do missing indicators help? | MCAR vs MNAR · 10% / 30% · mean, median, KNN, iterative (MICE) · ± indicator · native-NaN gradient boosting reference |
| Ablation | How many CV folds does the detector need? | 2, 3, 5, 10 folds |
| Error analysis | Which wrong labels slip through, and why? | per-class recall, confidence profile of hits, misses and false alarms |

**Protocol.** Stratified 75/25 split. The test set is never corrupted in label experiments, so
accuracy is always measured against true labels. Corruption is applied with a seeded RNG and a
ground-truth mask, so detection is scored with precision, recall and F1. Detection uses
out-of-fold probabilities (K-fold), so no model scores rows it was trained on. Each configuration
is run across seeds and compared with a paired t-test against the no-repair baseline.

**Datasets.** `breast_cancer` (569 × 30), `digits` (1,797 × 64), `wine` (178 × 13) from
scikit-learn, plus a 10-class synthetic tabular problem (`make_classification`, 40 features) used at
100,000 rows for logistic regression, 20,000 for gradient boosting and 10,000 for the KNN-heavy
imputation runs. All datasets ship with scikit-learn, so the study runs offline.

**Methods.** Confident learning follows [Northcutt et al., 2021](https://arxiv.org/abs/1911.00068):
per-class thresholds t_j = mean predicted probability of class j over rows labelled j. A row is
flagged when it is confidently some class other than its given label. Implemented from scratch in
[`dqlab/detect.py`](dqlab/detect.py).

## Run it

```bash
pip install -r requirements.txt
pytest                      # 20 unit tests
python run.py --quick       # smoke run, about 2 minutes
python run.py               # full study, about 30 minutes on a 2-core machine
python report.py            # regenerate results/REPORT.md and figures from the CSVs
```

Every number in this README and in `REPORT.md` is computed by `report.py` from the raw CSVs in
[`results/`](results/), so the write-up can always be regenerated and checked.

## Layout

```
dqlab/
  data.py         dataset loaders
  corrupt.py      label flipping, MCAR / MNAR masking (with ground-truth masks)
  detect.py       out-of-fold probabilities, argmax rule, confident learning
  repair.py       pruning (never drops a whole class), imputers, missing indicators
  experiments.py  the four experiments
  stats.py        confidence intervals, paired t-tests
run.py            runs everything, writes results/*.csv
report.py         CSVs → results/REPORT.md + figures
tests/            unit tests (pytest); CI also runs the pipeline end to end
```

## Limitations and next steps

- Noise is uniform (a flipped label goes to any other class). Real annotation errors are
  class-dependent (a 4 is mislabelled as a 9, rarely as a 0). Next: class-conditional noise
  matrices.
- Repairs are compared one at a time. Next: label noise and missing values together, as in real
  data.
- Re-weighting instead of pruning, and relabelling flagged rows with the model's prediction.
- A large real dataset (e.g. Forest Covertype, 581K rows; loader included in `dqlab/data.py`).
