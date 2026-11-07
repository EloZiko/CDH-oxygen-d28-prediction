"""SHAP values: how much each variable pushes the predictions up or down.

    uv run analysis/6_shap.py data/development_cohort.csv data/validation_cohort.csv

Uses the models fitted by 4_external_validation.py. Writes Figure 4 (final
CatBoost model, on Paris) and Figure S3 (models with recursive feature
elimination, on Leuven) in results/figures/, and the mean SHAP values quoted
in the article in results/tables/shap_catboost.csv.
"""
import argparse
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write image files, no window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from catboost import Pool

import pipelines
from cdh_model import PREDICTORS

RESULTS = Path(__file__).resolve().parents[1] / "results"
LABELS = {"oe_lhr": "o/e LHR (%)", "liver_herniation": "Liver herniation", "dvp": "DAP (mm)",
          "oe_tflv": "o/e TFLV (%)"}
# Figure S3: the models with recursive feature elimination, as named in the figure.
FIGURE_S3 = {"LogR_RFE_SimpleImputer": "Logistic Regression", "RF_RFE_NoImpute": "RandomForest",
             "HGB_RFE_NoImpute": "HistGradientBoosting", "XGB_RFE_NoImpute": "XGBoost"}
RED, RED_EDGE, BLUE, BLUE_EDGE = "#F7A6A0", "#D97D75", "#A9C9EC", "#7FA6D6"


def load_model(variant):
    """Load a model fitted by 4_external_validation.py."""
    return pickle.loads((RESULTS / "external" / "models" / f"{variant}.pickle").read_bytes())


def catboost_shap(X):
    """SHAP values of the final CatBoost model for each patient and predictor."""
    catboost = load_model("CatBoost_NoImpute").named_steps["classifier"].model_
    return catboost.get_feature_importance(Pool(X[PREDICTORS].to_numpy()), type="ShapValues")[:, :-1]  # last column: average


def figure_4(X, figures, tables):
    """Figure 4: mean absolute SHAP value of each predictor, red when higher values mean higher risk."""
    values = catboost_shap(X)
    importance = np.abs(values).mean(axis=0)
    # Direction: correlation between a predictor and its SHAP values, over the patients where it is known.
    direction = [np.corrcoef(X[name][X[name].notna()], values[X[name].notna().to_numpy(), i])[0, 1]
                 for i, name in enumerate(PREDICTORS)]
    pd.DataFrame({"predictor": PREDICTORS, "mean_abs_shap": importance,
                  "higher_value_higher_risk": np.array(direction) > 0}).to_csv(tables / "shap_catboost.csv", index=False)

    order = np.argsort(importance)  # the largest at the top
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.barh(range(len(order)), importance[order], height=0.55, linewidth=1.1,
            color=[RED if direction[i] > 0 else BLUE for i in order],
            edgecolor=[RED_EDGE if direction[i] > 0 else BLUE_EDGE for i in order])
    ax.set_yticks(range(len(order)), [LABELS[PREDICTORS[i]] for i in order])
    for position, i in enumerate(order):
        ax.text(importance[i] * 1.02, position, f"{importance[i]:.3f}", va="center", fontsize=9.5)
    ax.set_xlim(0, importance.max() * 1.18)
    ax.set_xlabel("mean(|SHAP value|)")
    ax.set_title("CatBoost (No Imputation): Global Feature Importance", loc="left", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, facecolor=RED, edgecolor=RED_EDGE, label="Positive correlation"),
                       plt.Rectangle((0, 0), 1, 1, facecolor=BLUE, edgecolor=BLUE_EDGE, label="Negative correlation")],
              loc="lower right", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(figures / "figure_4_shap_catboost.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def rfe_shap(variant, X, X_val):
    """SHAP values on Leuven of the variables kept by recursive feature elimination."""
    model = load_model(variant)
    rfe = model.named_steps["classifier"]
    kept = np.asarray(pipelines.CANDIDATES)[rfe.support_]
    impute = model.named_steps.get("impute")
    selected = pd.DataFrame((impute.transform(X_val) if impute else X_val.to_numpy())[:, rfe.support_], columns=kept)
    if variant.startswith("LogR"):
        # A linear model is explained against the Paris patients, as its reference.
        reference = pd.DataFrame(impute.transform(X)[:, rfe.support_], columns=kept)
        explanation = shap.LinearExplainer(rfe.estimator_, reference)(selected)
    else:
        estimator = rfe.estimator_.model_ if variant.startswith("XGB") else rfe.estimator_
        explanation = shap.TreeExplainer(estimator)(selected)
    if explanation.values.ndim == 3:
        explanation = explanation[..., 1]  # some explainers return both classes: keep "oxygen"
    return explanation


def figure_s3(variant, title, X, X_val, figures):
    """Figure S3: mean absolute SHAP values (left) and one dot per patient (right)."""
    explanation = rfe_shap(variant, X, X_val)
    fig, axes = plt.subplots(1, 2, figsize=(30 / 2.54, 10 / 2.54))  # 30 x 10 cm
    plt.sca(axes[0])
    shap.plots.bar(explanation, show=False)
    axes[0].set_xlabel("mean |SHAP values|", fontsize=10)
    plt.sca(axes[1])
    shap.plots.beeswarm(explanation, show=False)
    axes[1].set_xlabel("SHAP value", fontsize=10)
    axes[1].set_yticklabels([])
    for ax in axes:
        ax.tick_params(labelsize=9)
    fig.suptitle(f"{title} SHAP analysis", fontsize=16)
    plt.tight_layout()
    plt.savefig(figures / f"figure_S3_{variant}.png", dpi=300, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("development", help="Paris cohort (output of 1_clean_data.py)")
    parser.add_argument("validation", help="Leuven cohort (output of 1_clean_data.py)")
    args = parser.parse_args()
    X, _ = pipelines.load_cohort(args.development)
    X_val, _ = pipelines.load_cohort(args.validation)
    figures, tables = RESULTS / "figures", RESULTS / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)

    figure_4(X, figures, tables)
    for variant, title in FIGURE_S3.items():
        figure_s3(variant, title, X, X_val, figures)
    print(f"Written: {figures}/ and {tables / 'shap_catboost.csv'}")


if __name__ == "__main__":
    main()
