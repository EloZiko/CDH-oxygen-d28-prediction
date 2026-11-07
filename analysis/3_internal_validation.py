"""Internal validation: nested cross-validation on the Paris cohort.

    uv run analysis/3_internal_validation.py data/development_cohort.csv [--group CatBoost ...]

For every variant, the outer loop (10 x 3-fold) measures the performance and
the inner loop (3 x 3-fold) tunes the hyperparameters. Writes
results/internal/GROUP.csv: the mean and SD of each metric over the 30 outer
test folds (article, Tables 2, S2-S4 and S7). Takes hours.
"""
import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import cross_validate

import pipelines

OUTPUT = Path(__file__).resolve().parents[1] / "results" / "internal"


def evaluate(group, X, y):
    """Run the nested cross-validation of every variant of a group."""
    rows = []
    for variant, (pipeline, space) in pipelines.variants(group).items():
        print(f"== {variant} ==", flush=True)
        scores = cross_validate(
            pipelines.tuned(group, pipeline, space), X, y, cv=pipelines.OUTER_CV, scoring=pipelines.METRICS,
            n_jobs=-1 if group == "TabPFN" else None)  # TabPFN: the outer folds in parallel
        row = {"variant": variant}
        for metric in pipelines.METRICS:
            row[metric] = scores[f"test_{metric}"].mean()
            row[f"{metric}_std"] = scores[f"test_{metric}"].std()
        print(f"   AUC {row['roc_auc']:.3f} (+/- {row['roc_auc_std']:.3f})")
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("development", help="Paris cohort (output of 1_clean_data.py)")
    parser.add_argument("--group", choices=pipelines.groups(), nargs="+", default=pipelines.groups(),
                        help="groups to run (default: all)")
    args = parser.parse_args()

    X, y = pipelines.load_cohort(args.development)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for group in args.group:
        evaluate(group, X, y).to_csv(OUTPUT / f"{group}.csv", index=False)
        print(f"Written: {OUTPUT / f'{group}.csv'}")


if __name__ == "__main__":
    main()
