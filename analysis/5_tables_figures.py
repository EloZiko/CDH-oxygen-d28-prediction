"""Tables and figures of the article, from the results of scripts 3 and 4.

    uv run analysis/5_tables_figures.py data/development_cohort.csv data/validation_cohort.csv

Writes the article Tables 2, S1-S4, S7 and S8 in results/tables/ and
Figures 2 and S5 in results/figures/. Results that have not been computed yet
are left empty.
"""
import argparse
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write image files, no window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from statsmodels.nonparametric.smoothers_lowess import lowess

import pipelines

RESULTS = Path(__file__).resolve().parents[1] / "results"
N_BOOTSTRAP = 2000  # resamples of the Leuven patients for each confidence interval
MIN_PREDICTORS = [2, 0, 1, 3, 4]  # 2: main analysis; the others: Table S1
SHORT_NAMES = {"accuracy": "ACC", "precision": "PRE", "specificity": "SPE", "roc_auc": "AUC"}
IMPUTERS = [name for name in pipelines.IMPUTATIONS if name != "SMOTE"]
# The configuration retained for each classifier (article, Table 2).
RETAINED = {family: f"{family}_{'SimpleImputer' if family == 'LogR' else 'NoImpute'}"
            for family in pipelines.FAMILIES}

# The original analysis drew every bootstrap sample from one random generator,
# variant after variant, in this order. It also included a CatBoost with
# recursive feature elimination that the article does not report (6 variants):
# its samples are drawn and thrown away, so that every interval of the article
# comes out exactly the same.
DRAW_ORDER = {
    2: ["CatBoost_RFE", "CatBoost", "HGB_RFE", "HGB", "LogR_RFE", "LogR",
        "RF_RFE", "RF", "TabPFN", "XGB_RFE", "XGB"],
    **{n: ["CatBoost", "HGB", "LogR", "RF", "TabPFN", "XGB"] for n in (0, 1, 3, 4)},
}
NOT_REPORTED = {"CatBoost_RFE": 6}
GRID = np.linspace(0, 1, 101)  # predicted probabilities at which calibration curves are drawn


def external_folder(min_predictors):
    """Folder of the external results, for a given number of predictors needed."""
    return RESULTS / ("external" if min_predictors == 2 else f"external_min{min_predictors}")


def classification_metrics(y_true, y_score):
    """Compute every metric from predicted probabilities, calling 0.5 and above "oxygen"."""
    y_pred = (y_score >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "specificity": recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        "npv": precision_score(y_true, y_pred, pos_label=0, zero_division=0),
        "f1_0": f1_score(y_true, y_pred, pos_label=0, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_score),
    }


def external_intervals(validation_path):
    """Give every external metric of every variant a 95% bootstrap confidence interval."""
    rng = np.random.default_rng(42)  # one generator for everything, as originally
    intervals = {}  # {(min_predictors, variant): {metric: (value, low, high)}}
    for min_predictors in MIN_PREDICTORS:
        n = len(pipelines.load_cohort(validation_path, min_predictors)[1])
        for group in DRAW_ORDER[min_predictors]:
            path = external_folder(min_predictors) / f"{group}_predictions.csv"
            predictions = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["variant", "y_true", "y_score"])
            names = [None] * NOT_REPORTED[group] if group in NOT_REPORTED else pipelines.names(group)
            for variant in names:
                rows = predictions[predictions["variant"] == variant]
                y_true, y_score = rows["y_true"].to_numpy(), rows["y_score"].to_numpy()
                samples = []
                for _ in range(N_BOOTSTRAP):
                    index = rng.integers(0, n, size=n)  # drawn even with nothing to compute, to keep the order
                    if len(rows) and len(np.unique(y_true[index])) == 2:  # one outcome only: no AUC
                        samples.append(classification_metrics(y_true[index], y_score[index]))
                if len(rows):
                    point = classification_metrics(y_true, y_score)
                    intervals[min_predictors, variant] = {
                        metric: (point[metric], *np.percentile([s[metric] for s in samples], [2.5, 97.5]))
                        for metric in pipelines.METRICS}
    return intervals


def collect(intervals):
    """Gather, for every variant, the internal mean and SD and the external value and CI."""
    rows = []
    for family in pipelines.FAMILIES:  # in the order of the article's tables
        for group in [family] + ([f"{family}_RFE"] if family in pipelines.RFE_FAMILIES else []):
            path = RESULTS / "internal" / f"{group}.csv"
            internal = pd.read_csv(path).set_index("variant") if path.exists() else pd.DataFrame()
            for variant in pipelines.names(group):
                row = {"Model": family, "RFE": "Yes" if group.endswith("_RFE") else "No",
                       "Imputer": variant.removeprefix(f"{group}_"), "variant": variant}
                for metric in pipelines.METRICS:
                    done = variant in internal.index
                    row[f"{metric} mean"] = internal.at[variant, metric] if done else np.nan
                    row[f"{metric} sd"] = internal.at[variant, f"{metric}_std"] if done else np.nan
                    value, low, high = intervals.get((2, variant), {}).get(metric, (np.nan,) * 3)
                    row[f"{metric} external"], row[f"{metric} low"], row[f"{metric} high"] = value, low, high
                rows.append(row)
    return pd.DataFrame(rows)


def internal_text(row, metric):
    """Write an internal result as in the article, e.g. 0.878 (± 0.076)."""
    if pd.isna(row[f"{metric} mean"]):
        return ""
    return f"{row[f'{metric} mean']:.3f} (± {row[f'{metric} sd']:.3f})"


def external_text(value, low, high):
    """Write an external result as in the article, e.g. 0.827 [0.734, 0.908]."""
    return "" if pd.isna(value) else f"{value:.3f} [{low:.3f}, {high:.3f}]"


def external_cell(row, metric):
    """Write the external result of one variant for one metric."""
    return external_text(row[f"{metric} external"], row[f"{metric} low"], row[f"{metric} high"])


def rounded(value):
    """Round to three decimals, as the original analysis did before averaging or subtracting."""
    return float(f"{value:.3f}") if pd.notna(value) else np.nan


def difference(a, b):
    """Write b - a with its sign, e.g. +0.012."""
    return "" if pd.isna(a) or pd.isna(b) else f"{b - a:+.3f}"


def table(rows, metrics):
    """Format the internal, then external, results of some variants for some metrics."""
    out = rows[["Model", "Imputer"]].copy()
    for where, cell in (("Internal", internal_text), ("External", external_cell)):
        for metric in metrics:
            out[f"{SHORT_NAMES[metric]} ({where})"] = [cell(row, metric) for _, row in rows.iterrows()]
    return out


def variant_row(results, family, rfe, imputer):
    """Find the row of one variant in the collected results."""
    match = results[(results["Model"] == family) & (results["RFE"] == rfe) & (results["Imputer"] == imputer)]
    return match.iloc[0]


def compare(pairs, name_a, name_b, keys):
    """Put the AUC of the same model under two settings side by side, with the difference."""
    rows = []
    for a, b in pairs:
        rows.append({
            **{key: a[key] for key in keys},
            f"AUC Internal ({name_a})": internal_text(a, "roc_auc"),
            f"AUC Internal ({name_b})": internal_text(b, "roc_auc"),
            "Δ Internal": difference(a["roc_auc mean"], b["roc_auc mean"]),
            f"AUC External ({name_a})": external_cell(a, "roc_auc"),
            f"AUC External ({name_b})": external_cell(b, "roc_auc"),
            "Δ External": difference(rounded(a["roc_auc external"]), rounded(b["roc_auc external"])),
        })
    return pd.DataFrame(rows)


def clinical_criterion(cohort_path):
    """AUC of the counselling criterion: o/e LHR < 35%, or < 45% with liver herniation."""
    X, y = pipelines.load_cohort(cohort_path)
    lhr, liver = X["oe_lhr"], X["liver_herniation"]
    meets = (lhr < 35) | ((lhr < 45) & (lhr >= 35) & (liver == 1))  # a missing o/e LHR does not meet it
    return roc_auc_score(y, meets.astype(int))


def table_2(retained, development, validation):
    """Table 2: AUC of the clinical criterion and of the retained configuration of each classifier."""
    criterion = pd.DataFrame([{"Model": "LHR o/e", "Imputer": "",
                               "AUC (Internal)": f"{clinical_criterion(development):.3f}",
                               "AUC (External)": f"{clinical_criterion(validation):.3f}"}])
    return pd.concat([criterion, table(retained, ["roc_auc"])])


def table_s1(main_analysis, intervals):
    """Table S1: external AUC by the number of predictors a Leuven patient needs."""
    s1 = main_analysis[["Model", "Imputer"]].copy()
    mean = {"Model": "Mean", "Imputer": ""}
    for n in sorted(MIN_PREDICTORS):
        cells = [intervals.get((n, variant), {}).get("roc_auc", (np.nan,) * 3) for variant in main_analysis["variant"]]
        s1[f"AUC (min. {n} predictors)"] = [external_text(*cell) for cell in cells]
        mean[f"AUC (min. {n} predictors)"] = f"{np.nanmean([rounded(cell[0]) for cell in cells]):.3f}"
    return pd.concat([s1, pd.DataFrame([mean])])


def table_s3(results):
    """Table S3: simple imputation with and without SMOTE oversampling."""
    pairs = [(variant_row(results, family, "No", "SimpleImputer"), variant_row(results, family, "No", "SMOTE"))
             for family in pipelines.FAMILIES]
    return compare(pairs, "SimpleImputer", "SMOTE", ["Model"])


def table_s4(results):
    """Table S4: the four predictors against recursive feature elimination."""
    pairs = [(variant_row(results, family, "No", imputer), variant_row(results, family, "Yes", imputer))
             for family in pipelines.RFE_FAMILIES for imputer in IMPUTERS if imputer in pipelines.imputations(family)]
    return compare(pairs, "No RFE", "RFE", ["Model", "Imputer"])


def logit(p):
    """Log-odds of a probability, kept away from 0 and 1."""
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def calibration(y_true, y_score):
    """Observed/expected ratio, calibration slope and calibration-in-the-large."""
    binomial = sm.families.Binomial()
    # Slope: logistic regression of the outcome on the predicted log-odds.
    slope = sm.GLM(y_true, sm.add_constant(logit(y_score)), family=binomial).fit().params[1]
    # Calibration-in-the-large: the same with the slope fixed at 1.
    in_the_large = sm.GLM(y_true, np.ones((len(y_true), 1)), family=binomial,
                          offset=logit(y_score)).fit().params[0]
    return {"O/E ratio": y_true.sum() / y_score.sum(), "Calibration slope": slope,
            "Calibration-in-the-large": in_the_large}


def calibration_curve(y_true, y_score):
    """Observed proportion against predicted probability, smoothed by loess."""
    return np.clip(lowess(y_true, y_score, frac=0.7, it=0, xvals=GRID), 0, 1)


def calibration_results(predictions):
    """Table S8 and the curves of Figure S5, with 95% bootstrap intervals."""
    rng = np.random.default_rng(42)  # one generator: the table first, then the curves, as originally
    table_s8 = []
    for family, (y_true, y_score) in predictions.items():
        samples = {"O/E ratio": [], "Calibration slope": [], "Calibration-in-the-large": []}
        for _ in range(N_BOOTSTRAP):
            index = rng.integers(0, len(y_true), size=len(y_true))
            if len(np.unique(y_true[index])) < 2:
                continue
            samples["O/E ratio"].append(y_true[index].sum() / y_score[index].sum())
            try:
                values = calibration(y_true[index], y_score[index])
            except Exception:  # the logistic fit fails on a few resamples
                continue
            samples["Calibration slope"].append(values["Calibration slope"])
            samples["Calibration-in-the-large"].append(values["Calibration-in-the-large"])
        point = calibration(y_true, y_score)
        table_s8.append({"Model": family, **{name: external_text(point[name], *np.percentile(values, [2.5, 97.5]))
                                             for name, values in samples.items()}})
    curves = {}
    for family, (y_true, y_score) in predictions.items():
        resampled = []
        for _ in range(N_BOOTSTRAP):
            index = rng.integers(0, len(y_true), size=len(y_true))
            if len(np.unique(y_true[index])) == 2:
                resampled.append(calibration_curve(y_true[index], y_score[index]))
        curves[family] = (calibration_curve(y_true, y_score), *np.percentile(resampled, [2.5, 97.5], axis=0), y_score)
    return pd.DataFrame(table_s8), curves


def retained_predictions():
    """Load the Leuven predictions of the configuration retained for each classifier."""
    predictions = {}
    for family, variant in RETAINED.items():
        path = external_folder(2) / f"{family}_predictions.csv"
        if path.exists():
            rows = pd.read_csv(path).query("variant == @variant")
            predictions[family] = (rows["y_true"].to_numpy(dtype=float), rows["y_score"].to_numpy(dtype=float))
    return predictions


def figure_roc(predictions, path):
    """Figure 2: ROC curves of the retained configurations on the Leuven cohort."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1)
    for family, (y_true, y_score) in predictions.items():
        fpr, tpr, _ = roc_curve(y_true, y_score)
        ax.plot(fpr, tpr, linewidth=2, label=f"{family} (AUC = {roc_auc_score(y_true, y_score):.3f})")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", xlim=(0, 1), ylim=(0, 1.02))
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def figure_calibration(curves, path):
    """Figure S5: one calibration curve per classifier, with its 95% band."""
    fig, axes = plt.subplots(2, 3, figsize=(12.6, 8.4))
    for ax, (family, (curve, low, high, y_score)) in zip(axes.flat, curves.items()):
        ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1)  # perfect calibration
        ax.fill_between(GRID, low, high, color="steelblue", alpha=0.25, linewidth=0)
        ax.plot(GRID, curve, color="steelblue", linewidth=2)
        ax.plot(y_score, np.full_like(y_score, -0.04), "|", color="black", alpha=0.6)  # one tick per patient
        ax.set(xlim=(0, 1), ylim=(-0.05, 1.02), title=family,
               xlabel="Predicted probability", ylabel="Observed proportion")
    for ax in axes.flat[len(curves):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("development", help="Paris cohort (output of 1_clean_data.py)")
    parser.add_argument("validation", help="Leuven cohort (output of 1_clean_data.py)")
    args = parser.parse_args()
    tables, figures = RESULTS / "tables", RESULTS / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    warnings.filterwarnings("ignore", category=RuntimeWarning)  # means of empty columns, for missing results

    intervals = external_intervals(args.validation)
    results = collect(intervals)
    main_analysis = results[(results["RFE"] == "No") & results["Imputer"].isin(IMPUTERS)]
    retained = main_analysis[main_analysis["variant"].isin(RETAINED.values())]

    table_2(retained, args.development, args.validation).to_csv(tables / "table_2.csv", index=False)
    table_s1(main_analysis, intervals).to_csv(tables / "table_S1.csv", index=False)
    table(main_analysis, ["roc_auc"]).to_csv(tables / "table_S2.csv", index=False)
    table_s3(results).to_csv(tables / "table_S3.csv", index=False)
    table_s4(results).to_csv(tables / "table_S4.csv", index=False)
    table(main_analysis, SHORT_NAMES).to_csv(tables / "table_S7.csv", index=False)

    predictions = retained_predictions()
    if predictions:
        figure_roc(predictions, figures / "figure_2_roc.png")
        table_s8, curves = calibration_results(predictions)
        table_s8.to_csv(tables / "table_S8.csv", index=False)
        figure_calibration(curves, figures / "figure_S5_calibration.png")
    print(f"Written: {tables}/ and {figures}/")


if __name__ == "__main__":
    main()
