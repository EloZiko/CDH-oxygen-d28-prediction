"""Table 1: characteristics of the Paris and Leuven cohorts.

    uv run analysis/2_table_1.py data/development_cohort.csv data/validation_cohort.csv

Describes the analysed patients and compares the two cohorts: Mann-Whitney U
test for continuous variables, Fisher's exact test for the others. Writes
results/tables/table_1.csv.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu

from cdh_model import OUTCOME, PREDICTORS

RESULTS = Path(__file__).resolve().parents[1] / "results"

# The rows of Table 1, section by section.
SECTIONS = {
    "Clinical variables": {
        "maternal_weight": "Maternal weight (kg)",
        "maternal_height": "Maternal height (m)",
        "maternal_bmi": "Maternal BMI (kg/m2)",
        "ivf": "IVF (%)",
        "fetal_sex": "Fetal sex (% coded 1)",
        "parity": "Parity",
    },
    "Ultrasound variables": {
        "ga_first_us_referral": "GA at first US in the reference centre (weeks)",
        "oe_lhr": "o/e LHR at the reference centre (%)",
        "liver_herniation": "Liver herniation (%)",
        "dvp": "Deepest amniotic fluid pocket (mm)",
        "cervical_length": "Cervical length (mm)",
        "placental_localisation": "Placental localisation",
    },
    "MRI variables": {
        "mri_performed": "MRI performed (%)",
        "oe_tflv": "o/e TFLV (%)",
    },
    "Procedural variables": {
        "feto": "In-utero balloon treatment, FETO (%)",
    },
    "Outcome": {
        OUTCOME: "Oxygen at day 28 (%)",
    },
}
CONTINUOUS = ["maternal_weight", "maternal_height", "maternal_bmi", "ga_first_us_referral",
              "oe_lhr", "dvp", "cervical_length", "oe_tflv"]
# Variables with more than two values get one row per level, tested against all the other levels.
LEVELS = {
    "parity": {"0": lambda value: value == 0, "1": lambda value: value == 1, "≥ 2": lambda value: value >= 2},
    "placental_localisation": {"Anterior": lambda value: value == 1, "Posterior": lambda value: value == 2,
                               "Fundal": lambda value: value == 3},
}


def load(path):
    """Read a cohort file, keeping the analysed patients (known outcome, at least 2 predictors)."""
    cohort = pd.read_csv(path)
    cohort = cohort.dropna(subset=[OUTCOME]).dropna(subset=PREDICTORS, thresh=2)
    cohort["maternal_bmi"] = cohort["maternal_weight"] / cohort["maternal_height"] ** 2
    return cohort


def describe(values, continuous):
    """Summarise one variable in one cohort: mean (+/- SD), or percentage of 1."""
    values = values.dropna()  # among the patients for whom it is documented
    if len(values) == 0:
        return "not available"
    if continuous:
        return f"{values.mean():.2f} (+/- {values.std(ddof=1):.2f})"
    return f"{100 * values.mean():.2f}%"


def p_value(a, b, continuous):
    """Compare one variable between the two cohorts."""
    a, b = a.dropna(), b.dropna()
    if len(a) < 2 or len(b) < 2:
        return np.nan
    if continuous:
        return mannwhitneyu(a, b, alternative="two-sided").pvalue
    table = [[int(a.sum()), int(len(a) - a.sum())], [int(b.sum()), int(len(b) - b.sum())]]
    if any(sum(column) == 0 for column in zip(*table)):
        return np.nan  # a value never observed: the test is not defined
    return fisher_exact(table)[1]


def p_text(p):
    """Write a p-value as in the article."""
    if np.isnan(p):
        return "-"
    return "< 0.01" if p < 0.01 else "> 0.99" if p > 0.99 else f"{p:.2f}"


def table_1(paris, leuven):
    """Build Table 1, one row per variable (or per level), with section titles."""

    def row(label, a, b, continuous):
        return [label, describe(a, continuous), describe(b, continuous), p_text(p_value(a, b, continuous))]

    rows = []
    for section, variables in SECTIONS.items():
        rows.append([section, "", "", ""])
        for variable, label in variables.items():
            a, b = paris[variable], leuven[variable]
            if variable not in LEVELS:
                rows.append(row(label, a, b, continuous=variable in CONTINUOUS))
                continue
            rows.append([label, "", "", ""])
            for level, is_level in LEVELS[variable].items():
                rows.append(row(f"    {level}", a.dropna().map(is_level).astype(float),
                                b.dropna().map(is_level).astype(float), continuous=False))
    columns = ["Features", f"Paris (n={len(paris)})", f"Leuven (n={len(leuven)})", "p-value"]
    return pd.DataFrame(rows, columns=columns)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("development", help="Paris cohort (output of 1_clean_data.py)")
    parser.add_argument("validation", help="Leuven cohort (output of 1_clean_data.py)")
    args = parser.parse_args()

    output = RESULTS / "tables"
    output.mkdir(parents=True, exist_ok=True)
    table_1(load(args.development), load(args.validation)).to_csv(output / "table_1.csv", index=False)
    print(f"Written: {output / 'table_1.csv'}")


if __name__ == "__main__":
    main()
