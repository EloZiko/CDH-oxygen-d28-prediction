# Code of the article

This folder holds the code of every result of the article: data cleaning,
model comparison, hyperparameter search, internal and external validation,
tables and figures. It needs the original patient data, which are not public
(see [`data/README.md`](../data/README.md)): exact reproduction of the
results requires access to them.

## Installation

```bash
uv sync --group analysis
```

- On macOS, XGBoost needs the OpenMP library: `brew install libomp`.
- TabPFN needs a licence key (free account on <https://ux.priorlabs.ai>),
  read from the environment variable `TABPFN_TOKEN`, and runs much faster on
  a GPU.

## Scripts

Run them in this order, from the root of the repository.

| Script                     | What it does                        | Article                                       |
| -------------------------- | ----------------------------------- | --------------------------------------------- |
| `1_clean_data.py`          | Raw databases to cohort files       | Figure 1 (numbers of fetuses)                 |
| `2_table_1.py`             | Describes and compares the cohorts  | Table 1                                       |
| `3_internal_validation.py` | Nested cross-validation, Paris      | Tables 2, S2-S4, S7                           |
| `4_external_validation.py` | Fit on Paris, test on Leuven        | Tables 2, S1-S4, S7, S8                       |
| `5_tables_figures.py`      | Tables, confidence intervals, plots | Tables 2, S1-S4, S7, S8, Figures 2, 3, S2, S5 |
| `6_shap.py`                | SHAP values                         | Figures 4, S3                                 |
| `7_search_convergence.py`  | Convergence of the random search    | Figure S4                                     |

[`pipelines.py`](pipelines.py) holds what they share: the candidate
variables, the six classifiers and their variants (imputation, SMOTE,
recursive feature elimination), the hyperparameter search spaces (Table S5),
the number of candidates per search (Table S6) and the cross-validation
settings.

Scripts 3 and 7 take hours: the original analysis ran on a computing cluster.

```bash
DEV=data/development_cohort.csv
VAL=data/validation_cohort.csv

uv run --group analysis analysis/1_clean_data.py RAW_PARIS.csv RAW_LEUVEN.csv
uv run --group analysis analysis/2_table_1.py $DEV $VAL
uv run --group analysis analysis/3_internal_validation.py $DEV
uv run --group analysis analysis/4_external_validation.py $DEV $VAL
for n in 0 1 3 4; do  # Table S1: 0 to 4 predictors needed in Leuven
    uv run --group analysis analysis/4_external_validation.py $DEV $VAL \
        --min-predictors $n --group LogR RF HGB XGB CatBoost TabPFN
done
uv run --group analysis analysis/5_tables_figures.py $DEV $VAL
uv run --group analysis analysis/6_shap.py $DEV $VAL
uv run --group analysis analysis/7_search_convergence.py $DEV
```

Scripts 3 and 4 accept `--group` to run some classifiers only, for example
`--group CatBoost`.

## Outputs

Everything is written to `results/`, which git ignores: it contains
predictions for each patient. Never publish this folder.

The hyperparameters of the final model, fixed in
[`model/cdh_model.py`](../model/cdh_model.py), are those selected by script 4 for
`CatBoost_NoImpute` (`results/external/CatBoost_hyperparameters.csv`).
