"""Clean the raw databases of Paris and Leuven into the two cohort files.

    uv run analysis/1_clean_data.py RAW_PARIS.csv RAW_LEUVEN.csv

Writes data/development_cohort.csv and data/validation_cohort.csv (format in
data/README.md) and prints how many fetuses are left after each step
(article, Figure 1). The raw databases are not public: this code shows how
they were cleaned.
"""
import argparse
import re
from pathlib import Path

import pandas as pd

from cdh_model import OUTCOME, PREDICTORS

DATA = Path(__file__).resolve().parents[1] / "data"

# Column of the raw databases -> column of the cohort files. First the 16
# candidate variables, in the order the models received them, then the
# outcome, then the variables only used to describe the cohorts (Table 1).
COLUMNS = {
    "Parity": "parity",
    "Weight": "maternal_weight",
    "Length": "maternal_height",
    "IVF": "ivf",
    "Sexe": "fetal_sex",
    "oe_lhr CRMR": "oe_lhr",
    "Liver herniation": "liver_herniation",
    "deepest_amniotic_fluid_pocket": "dvp",
    "Cervical length": "cervical_length",
    "Placental localisation": "placental_localisation",
    "Other malformations (including IUGR<5eP)": "other_malformations",
    "Cardiac anomaly clear": "cardiac_anomaly",
    "Anomaly in clear": "genetic_anomaly",
    "CGH/exome comment": "cgh_exome_anomaly",
    "MRI": "mri_performed",
    "TFLV %": "oe_tflv",
    "oxygen_d28": OUTCOME,
    "First US at referral CRMR (GA)": "ga_first_us_referral",
    "Cardiac scan": "cardiac_scan_performed",
    "Exome /Panel": "exome_panel_performed",
    "feto": "feto",
}

# Columns meant to hold 0 or 1, where clinicians sometimes typed a note instead.
YES_NO_COLUMNS = ["Cardiac anomaly clear", "Anomaly in clear", "CGH/exome comment"]

# Anomalies that still count as an isolated CDH: growth restriction (IUGR,
# RCIU in French) and single umbilical artery (AOU). As in the original
# analysis, only these abbreviations are recognised.
MINOR_ANOMALIES = ["RCIU", "IUGR", "AOU"]

NUMBER = re.compile(r"-?\d+(\.\d+)?")
NORMAL = re.compile(r"\bnorma+l\b|\bn[ée]gati", re.IGNORECASE)  # "normal", "négatif"; not "anormal"
NOT_DONE = re.compile(r"^NF\b|non souhait|[ée]chec", re.IGNORECASE)  # not done, declined, failed


def to_number(column):
    """Read numbers written as text; codes such as NR, NF or nvt become missing values."""

    def convert(value):
        if not isinstance(value, str):
            return value
        value = value.replace(",", ".")  # decimal comma
        return float(value) if NUMBER.fullmatch(value) else float("nan")

    return column.map(convert).astype(float)


def leading_number(column):
    """Keep the number written before a comment, as in "33 (comment)"."""
    return to_number(column.map(
        lambda value: re.match(r"\s*([\d.,]*)", value).group(1) if isinstance(value, str) else value))


def to_yes_no(column, empty):
    """Turn a yes/no column into 1/0, reading the notes clinicians sometimes typed instead."""

    def convert(value):
        if isinstance(value, str):
            value = value.strip()
            if NUMBER.fullmatch(value.replace(",", ".")):
                value = float(value.replace(",", "."))
        if not isinstance(value, str):
            return empty if pd.isna(value) else float(value != 0)  # 0: no, any other number: yes
        if not value:
            return empty  # a cell holding only spaces
        if NORMAL.search(value):  # e.g. "normal karyotype": no anomaly
            return 0.0
        if NOT_DONE.search(value):  # test not done, failed or declined: unknown
            return float("nan")
        return 1.0  # any other note describes an anomaly

    return column.map(convert).astype(float)


def has_major_anomaly(text):
    """Tell whether a list of associated anomalies, separated by commas, holds a major one."""
    if not isinstance(text, str) or text.strip() == "0":
        return False
    return any(not any(minor in item for minor in MINOR_ANOMALIES) for item in text.split(","))


def read_paris(path):
    """Read the Paris database (Antoine-Béclère, Kremlin-Bicêtre) and apply its inclusion criteria."""
    data = pd.read_csv(path, sep=";", header=1, encoding="mac_roman")  # column names on line 2
    data = data.rename(columns={"IVF ": "IVF"}).dropna(how="all")
    flow = {"Records in the database": len(data)}

    # Same steps, in the same order, as the original cleaning.
    data = data.dropna(subset=["Date of birth(mother)"])
    data = data.dropna(subset=["Sexe", "IVF"], how="all")
    flow["With maternal date of birth, and fetal sex or IVF recorded"] = len(data)
    data = data[~data["Other malforation in clear including heart"].map(has_major_anomaly)]
    flow["Without associated major anomaly"] = len(data)

    for column in ["Weight", "Length", "First US at referral CRMR (GA)", "oe_lhr CRMR", "Liver herniation",
                   "deepest_amniotic_fluid_pocket", "Placental localisation", "Cardiac scan",
                   "Exome /Panel", "MRI", "TFLV %"]:
        data[column] = to_number(data[column])
    for column in ["IVF", "Cervical length"]:
        data[column] = leading_number(data[column])  # e.g. "0 (IUI)": insemination, not IVF
    for column in YES_NO_COLUMNS:
        data[column] = to_yes_no(data[column], empty=float("nan"))
    data["Side"] = to_number(data["Side"].replace("G", "0"))  # "G" (gauche): left, coded 0
    data["feto"] = data["Date_FETO"].notna().astype(float)  # FETO done when its date is recorded
    return data, flow


def read_leuven(path):
    """Read the Leuven database and convert it to the units and codes of the Paris one."""
    data = pd.read_csv(path, sep=";", header=2, encoding="latin1")  # column names on line 3
    data = data.rename(columns={
        "IVF ": "IVF",
        "First US at referral fetal medicine center  (GA)": "First US at referral CRMR (GA)",
    }).dropna(how="all")
    flow = {"Records in the database": len(data)}

    data["Parity"] = to_number(data["Parity"].replace("more", "3"))  # "more": 3 or more
    data["Placental localisation"] = to_number(
        data["Placental localisation"].replace({"anterior": "1", "posterior": "2", "fundal": "3"}))
    for column in ["First US at referral CRMR (GA)", "oe_lhr CRMR", "Liver herniation", "Cardiac scan"]:
        data[column] = to_number(data[column])  # "nvt" (not applicable), "missing": missing
    for column in YES_NO_COLUMNS:
        data[column] = to_yes_no(data[column], empty=0.0)  # no description: no anomaly

    lhr = data["oe_lhr CRMR"]
    data["oe_lhr CRMR"] = lhr.where(lhr >= 1, lhr * 100)  # some written as fractions: 0.35 -> 35%
    data["TFLV %"] = data["TFLV %"].where(data["TFLV %"] >= 1)  # values below 1% are not valid
    data["deepest_amniotic_fluid_pocket"] = data["deepest_amniotic_fluid_pocket"] * 10  # cm -> mm
    data["Length"] = data["Length"] / 100  # cm -> m
    data["oxygen_d28"] = data["oxygen_d28"].where(data["Surv_d28"] != 0)  # the outcome exists in survivors only
    data["feto"] = data["Intervention"]
    return data, flow


def clean(centre, path):
    """Keep the left-sided CDH of one centre in the common format, counting the fetuses at each step."""
    data, flow = (read_paris if centre == "paris" else read_leuven)(path)
    data = data[data["Side"] == 0]
    flow["Left-sided CDH"] = len(data)
    cohort = data[list(COLUMNS)].rename(columns=COLUMNS).reset_index(drop=True)

    with_outcome = cohort.dropna(subset=[OUTCOME])
    flow["Oxygen status at day 28 documented"] = len(with_outcome)
    flow["At least 2 of the 4 predictors (analysed)"] = len(with_outcome.dropna(subset=PREDICTORS, thresh=2))
    return cohort, flow


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paris", help="raw Paris database, as CSV")
    parser.add_argument("leuven", help="raw Leuven database, as CSV")
    args = parser.parse_args()

    outputs = {"paris": DATA / "development_cohort.csv", "leuven": DATA / "validation_cohort.csv"}
    for centre, output in outputs.items():
        cohort, flow = clean(centre, getattr(args, centre))
        cohort.to_csv(output, index=False)
        print(f"{centre.capitalize()}: {output}")
        for step, count in flow.items():
            print(f"{count:6d}  {step}")


if __name__ == "__main__":
    main()
