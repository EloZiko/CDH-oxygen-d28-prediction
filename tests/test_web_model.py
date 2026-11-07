"""The web tool must give exactly the same results as CatBoost in Python.

The JavaScript of docs/ runs here in V8, the JavaScript engine of Chrome
(package mini-racer), on random patients with missing values and with values
lying exactly on the split borders of the trees.
"""
import json

import numpy as np
import pandas as pd
import pytest
from py_mini_racer import MiniRacer

import cdh_model


@pytest.fixture(scope="module")
def model():
    if not cdh_model.MODEL_PATH.exists():
        pytest.skip("no trained model: run model/train.py first")
    return cdh_model.load_model()


@pytest.fixture(scope="module")
def web():
    engine = MiniRacer()
    engine.eval(cdh_model.WEB_MODEL_PATH.read_text())
    engine.eval((cdh_model.ROOT / "docs" / "predict.js").read_text())
    return engine


@pytest.fixture(scope="module")
def patients(model):
    rng = np.random.default_rng(0)
    n = 2000
    random_patients = pd.DataFrame({
        "oe_lhr": rng.uniform(5, 100, n),
        "liver_herniation": rng.integers(0, 2, n).astype(float),
        "dvp": rng.uniform(0, 200, n),
        "oe_tflv": rng.uniform(5, 100, n),
    })
    random_patients = random_patients.mask(rng.random(random_patients.shape) < 0.3)  # 30% missing

    # Each border, and the closest numbers below and above it.
    on_borders = []
    for feature, borders in model.get_borders().items():
        for border in borders:
            for value in (np.nextafter(border, -np.inf), border, np.nextafter(border, np.inf)):
                patient = random_patients.iloc[len(on_borders) % n].copy()
                patient.iloc[feature] = value
                on_borders.append(patient)

    return pd.concat([random_patients, pd.DataFrame(on_borders)], ignore_index=True)


def run_web(web, function, patients):
    """Call a function of predict.js on every patient."""
    rows = json.dumps(patients.to_numpy().tolist())  # NaN is written as NaN, valid JavaScript
    return json.loads(web.eval(f"JSON.stringify({rows}.map((p) => {function}(MODEL, p)))"))


def test_same_class(model, web, patients):
    expected = cdh_model.predict_high_risk(model, patients)
    assert run_web(web, "isHighRisk", patients) == expected.tolist()


def test_same_score(model, web, patients):
    expected = model.predict(patients, prediction_type="RawFormulaVal")
    np.testing.assert_allclose(run_web(web, "rawScore", patients), expected, rtol=0, atol=1e-12)


def test_same_shap_values(model, web, patients):
    expected = cdh_model.explain(model, patients)
    np.testing.assert_allclose(run_web(web, "shapValues", patients), expected, rtol=0, atol=1e-12)
