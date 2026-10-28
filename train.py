"""Train the final model on the development cohort.

Usage:
    python train.py data/development_cohort.csv

Saves the model for Python (models/catboost_o2_d28.cbm) and for the web tool
(docs/model.js).
"""
import argparse

import cdh_model

parser = argparse.ArgumentParser(description="Train the final model on the development cohort.")
parser.add_argument("cohort", help="CSV file of the development cohort (format: data/README.md)")
args = parser.parse_args()

X, y = cdh_model.load_cohort(args.cohort)
print(f"Development cohort: {len(y)} patients, {y.sum()} with oxygen dependency at day 28")

model = cdh_model.train(X, y)

cdh_model.MODEL_PATH.parent.mkdir(exist_ok=True)
model.save_model(str(cdh_model.MODEL_PATH))
cdh_model.export_for_web(model)
print(f"Saved {cdh_model.MODEL_PATH.relative_to(cdh_model.ROOT)} "
      f"and {cdh_model.WEB_MODEL_PATH.relative_to(cdh_model.ROOT)}")
