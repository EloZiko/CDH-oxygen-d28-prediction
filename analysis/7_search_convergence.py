"""Figure S4: does the random hyperparameter search try enough configurations?

    uv run analysis/7_search_convergence.py data/development_cohort.csv

For each classifier, with recursive feature elimination (left) and with the
four predictors (right), tries 400 configurations on Paris and plots the best
score found so far; the dashed line is the number tried in the analysis
(Table S6). Writes results/figures/figure_S4_search_convergence.png and the
curves in results/tables/search_convergence.csv. Takes several hours.
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write image files, no window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV

import pipelines

RESULTS = Path(__file__).resolve().parents[1] / "results"
N_CANDIDATES = 400
# One row of the figure per classifier: with RFE, then with the four predictors.
# TabPFN is absent: it has nothing to tune.
ROWS = [("LogR_RFE", "LogR"), ("RF_RFE", "RF"), ("HGB_RFE", "HGB"), ("XGB_RFE", "XGB"), ("CatBoost",)]
TITLES = {"LogR": "Logistic Regression", "RF": "RandomForest", "HGB": "HistGradientBoosting",
          "XGB": "XGBoost", "CatBoost": "CatBoost"}


def best_so_far(group, X, y):
    """Best balanced accuracy found after 1, 2, ..., 400 configurations (k-nearest-neighbours imputation)."""
    pipeline, space = pipelines.variants(group)[f"{group}_KNNimpute"]
    search = RandomizedSearchCV(pipeline, space, n_iter=N_CANDIDATES, cv=pipelines.INNER_CV,
                                scoring=pipelines.SEARCH_SCORING, n_jobs=-1, random_state=42).fit(X, y)
    return np.maximum.accumulate(search.cv_results_["mean_test_score"])  # results are in the order tried


def plot(ax, group, best):
    """Draw one convergence curve, with the number of configurations used in the analysis."""
    ax.plot(np.arange(1, N_CANDIDATES + 1), best, marker=".")
    ax.axvline(pipelines.N_ITER[group], color="grey", linestyle="--", label="n_iter used")
    with_rfe = " (with RFE)" if group.endswith("_RFE") else ""
    ax.set(xlabel="n_iter", ylabel="Best inner CV score (cumulative)",
           title=f"RandomizedSearchCV convergence for {TITLES[group.removesuffix('_RFE')]}{with_rfe}")
    ax.legend()
    ax.grid(True, alpha=0.3)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("development", help="Paris cohort (output of 1_clean_data.py)")
    args = parser.parse_args()
    X, y = pipelines.load_cohort(args.development)

    fig = plt.figure(figsize=(14, 24))
    grid = fig.add_gridspec(len(ROWS), 2)
    curves = []
    for row, groups in enumerate(ROWS):
        for column, group in enumerate(groups):
            print(f"== {group} ==", flush=True)
            best = best_so_far(group, X, y)
            curves.append(pd.DataFrame({"group": group, "n_iter": np.arange(1, N_CANDIDATES + 1),
                                        "best_balanced_accuracy": best}))
            plot(fig.add_subplot(grid[row, column] if len(groups) == 2 else grid[row, :]), group, best)

    for folder in ("figures", "tables"):
        (RESULTS / folder).mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(RESULTS / "figures" / "figure_S4_search_convergence.png", dpi=200, bbox_inches="tight")
    pd.concat(curves).to_csv(RESULTS / "tables" / "search_convergence.csv", index=False)
    print(f"Written: {RESULTS / 'figures' / 'figure_S4_search_convergence.png'}")


if __name__ == "__main__":
    main()
