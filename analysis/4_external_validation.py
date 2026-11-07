"""External validation: fit each variant on Paris, then test it unchanged on Leuven.

    uv run analysis/4_external_validation.py DEVELOPMENT VALIDATION [--group ...] [--min-predictors N]

Writes in results/external/ (results/external_minN/ with --min-predictors N):
the metrics (GROUP.csv), the chosen hyperparameters (GROUP_hyperparameters.csv),
each patient's predicted probability (GROUP_predictions.csv) and the fitted
models (models/, used by 6_shap.py). These files are patient-level: never
publish results/.
"""
import argparse
import pickle
from pathlib import Path

import pandas as pd
from sklearn.metrics import get_scorer

import pipelines

RESULTS = Path(__file__).resolve().parents[1] / "results"


def validate(group, X, y, X_val, y_val, output):
    """Fit every variant of a group on the whole Paris cohort and measure it on Leuven."""
    metrics, hyperparameters, predictions = [], [], []
    for variant, (pipeline, space) in pipelines.variants(group).items():
        print(f"== {variant} ==", flush=True)
        model = pipelines.tuned(group, pipeline, space).fit(X, y)  # the search runs on Paris only
        if space is not None:
            hyperparameters.append({"variant": variant, **model.best_params_})
            model = model.best_estimator_
        (output / "models" / f"{variant}.pickle").write_bytes(pickle.dumps(model))

        row = {"variant": variant}
        for metric, scorer in pipelines.METRICS.items():
            row[metric] = get_scorer(scorer)(model, X_val, y_val)
        print(f"   AUC {row['roc_auc']:.4f}")
        metrics.append(row)
        predictions.append(pd.DataFrame({
            "variant": variant, "y_true": y_val.to_numpy(), "y_score": model.predict_proba(X_val)[:, 1]}))

    pd.DataFrame(metrics).to_csv(output / f"{group}.csv", index=False)
    pd.DataFrame(hyperparameters).to_csv(output / f"{group}_hyperparameters.csv", index=False)
    pd.concat(predictions).to_csv(output / f"{group}_predictions.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("development", help="Paris cohort (output of 1_clean_data.py)")
    parser.add_argument("validation", help="Leuven cohort (output of 1_clean_data.py)")
    parser.add_argument("--group", choices=pipelines.groups(), nargs="+", default=pipelines.groups(),
                        help="groups to run (default: all)")
    parser.add_argument("--min-predictors", type=int, choices=range(5), default=2,
                        help="predictors a Leuven patient needs (Table S1); Paris always uses 2")
    args = parser.parse_args()

    X, y = pipelines.load_cohort(args.development)
    X_val, y_val = pipelines.load_cohort(args.validation, args.min_predictors)
    print(f"Paris: {len(y)} patients, Leuven: {len(y_val)} patients")
    output = RESULTS / ("external" if args.min_predictors == 2 else f"external_min{args.min_predictors}")
    (output / "models").mkdir(parents=True, exist_ok=True)
    for group in args.group:
        validate(group, X, y, X_val, y_val, output)
        print(f"Written: {output / group}*.csv")


if __name__ == "__main__":
    main()
