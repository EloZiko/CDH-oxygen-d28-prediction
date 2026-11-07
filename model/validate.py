"""Measure the performance of the trained model on a cohort (external validation).

Usage:
    uv run model/validate.py data/validation_cohort.csv
    uv run model/validate.py data/validation_cohort.csv --min-predictors 4
"""
import argparse

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

import cdh_model

# Resamples used for the 95% confidence interval of the AUC (2000, as in the article).
N_BOOTSTRAP = 2000


def bootstrap_auc_ci(y, score, seed=42):
    """95% confidence interval of the AUC: recompute it on 2000 samples of
    patients drawn with replacement, keep the 2.5th and 97.5th percentiles."""
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(N_BOOTSTRAP):
        sample = rng.integers(0, len(y), len(y))
        if len(np.unique(y[sample])) == 2:  # the AUC needs both outcomes
            aucs.append(roc_auc_score(y[sample], score[sample]))
    return np.percentile(aucs, [2.5, 97.5])


parser = argparse.ArgumentParser(description="Measure the performance of the trained model on a cohort.")
parser.add_argument("cohort", help="CSV file of the cohort (format: data/README.md)")
parser.add_argument("--min-predictors", type=int, default=cdh_model.MIN_PREDICTORS, choices=range(5),
                    help="keep patients with at least this many of the 4 predictors (default: 2)")
args = parser.parse_args()

model = cdh_model.load_model()
X, y = cdh_model.load_cohort(args.cohort, args.min_predictors)
y = y.to_numpy()

# The AUC ranks patients by the model output; the other metrics use the class.
score = model.predict_proba(X)[:, 1]
high_risk = cdh_model.predict_high_risk(model, X).astype(int)
low, high = bootstrap_auc_ci(y, score)

print(f"Patients:    {len(y)} ({y.sum()} with oxygen dependency at day 28)")
print(f"AUC:         {roc_auc_score(y, score):.3f} (95% CI {low:.3f} to {high:.3f})")
print(f"Accuracy:    {accuracy_score(y, high_risk):.3f}")
print(f"Sensitivity: {recall_score(y, high_risk):.3f}")
print(f"Specificity: {recall_score(y, high_risk, pos_label=0):.3f}")
print(f"Precision:   {precision_score(y, high_risk):.3f}")
