"""What the analysis scripts share: the data, the six classifiers and their variants.

A "group" is one classifier with one way of choosing its variables: the four
predictors (e.g. "CatBoost") or recursive feature elimination among the 16
candidate variables (e.g. "LogR_RFE"). A "variant" adds one way of handling
missing values (e.g. "CatBoost_NoImpute").
"""
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline as ImbPipeline
from scipy.stats import loguniform, randint, uniform
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.experimental import enable_iterative_imputer  # noqa: F401 (makes IterativeImputer available)
from sklearn.feature_selection import RFE
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import RandomizedSearchCV, RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from cdh_model import OUTCOME, PREDICTORS

# The 16 candidate variables, in the order the models received them.
CANDIDATES = [
    "parity", "maternal_weight", "maternal_height", "ivf", "fetal_sex",
    "oe_lhr", "liver_herniation", "dvp", "cervical_length", "placental_localisation",
    "other_malformations", "cardiac_anomaly", "genetic_anomaly", "cgh_exome_anomaly",
    "mri_performed", "oe_tflv",
]
# Variables with a few possible values (yes/no, categories). When SMOTE creates an
# artificial patient, it gives them the most common value among similar patients,
# since an in-between value (0.4 of a liver herniation) would make no sense.
CATEGORICAL = [
    "parity", "ivf", "fetal_sex", "liver_herniation", "placental_localisation",
    "other_malformations", "cardiac_anomaly", "genetic_anomaly", "cgh_exome_anomaly",
    "mri_performed",
]

# Nested cross-validation: the outer loop measures performance, the inner one
# tunes the hyperparameters.
OUTER_CV = RepeatedStratifiedKFold(n_splits=3, n_repeats=10, random_state=123)
INNER_CV = RepeatedStratifiedKFold(n_splits=3, n_repeats=3, random_state=42)
SEARCH_SCORING = "balanced_accuracy"

# The metrics reported in the article.
METRICS = {
    "accuracy": "accuracy",
    "f1": make_scorer(f1_score),
    "recall": make_scorer(recall_score),  # sensitivity
    "precision": make_scorer(precision_score),  # positive predictive value
    "specificity": make_scorer(recall_score, pos_label=0),
    "npv": make_scorer(precision_score, pos_label=0),  # negative predictive value
    "f1_0": make_scorer(f1_score, pos_label=0),
    "roc_auc": "roc_auc",
}

FAMILIES = ["LogR", "RF", "HGB", "XGB", "CatBoost", "TabPFN"]
RFE_FAMILIES = ["LogR", "RF", "HGB", "XGB"]
IMPUTATIONS = ["NoImpute", "SimpleImputer", "KNNimpute", "ItImpute", "ItImpute_Knn", "SMOTE"]

# Hyperparameters searched for each classifier (article, Table S5).
CLASSIFIER_SPACE = {
    "LogR": {
        "C": loguniform(0.001, 100),
        "solver": ["saga"],
        "l1_ratio": [round(x * 0.2, 1) for x in range(0, 6)],
    },
    "RF": {
        "n_estimators": randint(50, 300),
        "max_depth": [3, 5, 7, 10, None],
        "min_samples_leaf": randint(1, 10),
        "min_samples_split": randint(2, 15),
    },
    "HGB": {
        "max_iter": randint(50, 300),
        "early_stopping": [True],
        "n_iter_no_change": [5],
        "max_depth": [5, 7, 10, None],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_leaf_nodes": [15, 31, 63, 127],
        "min_samples_leaf": randint(3, 10),
        "l2_regularization": uniform(0, 10),
    },
    "XGB": {
        "n_estimators": randint(20, 150),
        "max_depth": randint(2, 6),
        "learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3],
        "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
        "min_child_weight": randint(1, 10),
    },
    "CatBoost": {
        "iterations": randint(20, 100),
        "depth": randint(2, 5),
        "learning_rate": loguniform(0.01, 0.2),
        "l2_leaf_reg": loguniform(1.0, 100.0),
    },
    "TabPFN": {},  # nothing to tune
}

# Number of configurations tried by each random search (article, Table S6).
N_ITER = {
    "LogR": 70, "LogR_RFE": 70, "RF": 160, "RF_RFE": 160, "HGB": 130, "HGB_RFE": 130,
    "XGB": 70, "XGB_RFE": 150, "CatBoost": 150, "TabPFN": 5,
}


def load_cohort(path, min_predictors=2):
    """Read a cohort file, keeping the patients with a known outcome and enough predictors."""
    cohort = pd.read_csv(path)
    cohort = cohort.dropna(subset=[OUTCOME]).dropna(subset=PREDICTORS, thresh=min_predictors)
    return cohort[CANDIDATES].astype(float), cohort[OUTCOME]


class Impute(BaseEstimator, TransformerMixin):
    """Fill in missing values on standardised data, then put the values back on their own scale."""

    def __init__(self, imputer=None, unscale=True):
        self.imputer = imputer
        self.unscale = unscale  # False for logistic regression, which keeps standardised values

    def fit(self, X, y=None):
        self.scaler_ = StandardScaler().fit(X)
        self.imputer_ = clone(self.imputer).fit(self.scaler_.transform(X))
        return self

    def transform(self, X):
        imputed = self.imputer_.transform(self.scaler_.transform(X))
        return self.scaler_.inverse_transform(imputed) if self.unscale else imputed


def keep_predictors():
    """Keep only the four predictors, chosen a priori on clinical grounds."""
    columns = [CANDIDATES.index(name) for name in PREDICTORS]
    return ColumnTransformer([("selection", "passthrough", columns)], remainder="drop")


class CatBoostWrapper(ClassifierMixin, BaseEstimator):
    """CatBoost, with the interface scikit-learn expects."""

    def __init__(self, iterations=100, depth=3, learning_rate=0.1, l2_leaf_reg=3.0):
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.l2_leaf_reg = l2_leaf_reg

    def fit(self, X, y):
        self.model_ = CatBoostClassifier(
            iterations=self.iterations, depth=self.depth, learning_rate=self.learning_rate,
            l2_leaf_reg=self.l2_leaf_reg, bagging_temperature=1.0, subsample=1.0,
            random_state=42, verbose=0, auto_class_weights="Balanced",  # settings of the original analysis
        ).fit(X, y)
        self.classes_ = self.model_.classes_
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)


class XGBoostWrapper(ClassifierMixin, BaseEstimator):
    """XGBoost, with class weights computed on the patients it is trained on."""

    def __init__(self, n_estimators=100, max_depth=6, learning_rate=0.3, subsample=1.0,
                 colsample_bytree=1.0, min_child_weight=1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True  # XGBoost handles missing values itself
        return tags

    def fit(self, X, y):
        from xgboost import XGBClassifier  # imported here: only the XGBoost runs need it

        y = np.asarray(y)
        n_positive = (y == 1).sum()
        self.model_ = XGBClassifier(
            n_estimators=self.n_estimators, max_depth=self.max_depth,
            learning_rate=self.learning_rate, subsample=self.subsample,
            colsample_bytree=self.colsample_bytree, min_child_weight=self.min_child_weight,
            random_state=42, scale_pos_weight=(y == 0).sum() / n_positive if n_positive else 1.0,
        ).fit(X, y)
        self.classes_ = self.model_.classes_
        self.feature_importances_ = self.model_.feature_importances_
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)


class PermutationRFE(ClassifierMixin, BaseEstimator):
    """Recursive feature elimination for HistGradientBoosting, which has no built-in feature importance."""

    def __init__(self, estimator, n_features_to_select=10, n_step=1):
        self.estimator = estimator
        self.n_features_to_select = n_features_to_select
        self.n_step = n_step

    def fit(self, X, y):
        X = np.asarray(X)
        remaining = np.arange(X.shape[1])
        while len(remaining) > self.n_features_to_select:
            # A variable matters if shuffling its values lowers the score.
            fitted = clone(self.estimator).fit(X[:, remaining], y)
            importance = permutation_importance(
                fitted, X[:, remaining], y, scoring="balanced_accuracy", n_repeats=5, random_state=42)
            n_remove = min(self.n_step, len(remaining) - self.n_features_to_select)
            remaining = np.sort(remaining[np.argsort(importance.importances_mean)[n_remove:]])
        self.support_ = remaining
        self.estimator_ = clone(self.estimator).fit(X[:, remaining], y)
        return self

    @property
    def classes_(self):
        return self.estimator_.classes_

    def predict(self, X):
        return self.estimator_.predict(np.asarray(X)[:, self.support_])

    def predict_proba(self, X):
        return self.estimator_.predict_proba(np.asarray(X)[:, self.support_])


def tabpfn():
    """TabPFN, a network pre-trained on synthetic tables: there is nothing to tune."""
    from tabpfn import TabPFNClassifier  # needs a licence key in TABPFN_TOKEN (https://ux.priorlabs.ai)

    liver_herniation = PREDICTORS.index("liver_herniation")  # the only categorical predictor
    return TabPFNClassifier(n_estimators=16, categorical_features_indices=[liver_herniation],
                            balance_probabilities=True, device="auto", random_state=42)


def classifier(family, rfe):
    """Build the classifier of a family, inside recursive feature elimination if asked."""
    if family == "LogR":
        model = LogisticRegression(random_state=42, max_iter=10000, class_weight="balanced")
    elif family == "RF":
        model = RandomForestClassifier(random_state=42, class_weight="balanced")
    elif family == "HGB":
        model = HistGradientBoostingClassifier(random_state=42, class_weight="balanced")
    elif family == "XGB":
        model = XGBoostWrapper()
    elif family == "CatBoost":
        model = CatBoostWrapper()
    else:
        model = tabpfn()
    if not rfe:
        return model
    if family == "HGB":
        return PermutationRFE(model, n_features_to_select=5, n_step=5)
    # The search tunes how many variables to keep; these starting values are the original ones.
    return RFE(model, n_features_to_select=5 if family == "XGB" else len(CANDIDATES), step=5)


def imputation_step(family, imputation, rfe):
    """Build the step that fills in missing values for one variant."""
    # Fixed random state as in the original analysis; it has no effect with these settings.
    random_state = 42 if family == "LogR" and not rfe else None
    if imputation in ("SimpleImputer", "SMOTE"):
        imputer = SimpleImputer(strategy="most_frequent")
    elif imputation == "KNNimpute":
        imputer = KNNImputer(n_neighbors=5)
    elif imputation == "ItImpute":
        imputer = IterativeImputer(random_state=random_state)
    else:
        imputer = IterativeImputer(estimator=KNeighborsRegressor(n_neighbors=2), random_state=random_state)
    return Impute(imputer, unscale=family != "LogR")


def search_space(family, imputation, rfe):
    """List the hyperparameters searched for one variant, or None if there is nothing to tune."""
    prefix = "classifier__estimator__" if rfe else "classifier__"
    space = {prefix + name: values for name, values in CLASSIFIER_SPACE[family].items()}
    if rfe:
        space["classifier__n_features_to_select"] = randint(1, 10)
    if imputation in ("SimpleImputer", "SMOTE"):
        space["impute__imputer__strategy"] = ["mean", "median", "most_frequent"]
    elif imputation == "KNNimpute":
        space["impute__imputer__n_neighbors"] = [1, 2, 3, 4, 5]
    elif imputation == "ItImpute_Knn":
        space["impute__imputer__estimator__n_neighbors"] = [1, 2, 3, 4, 5]
    if imputation == "SMOTE":
        space["Over_sample__sampling_strategy"] = [{0: n, 1: n} for n in [30, 50, 70, 100]]
    return space or None


def imputations(group):
    """List the ways of handling missing values tried for a group."""
    if group.startswith("LogR"):
        return IMPUTATIONS[1:]  # logistic regression does not accept missing values
    return IMPUTATIONS


def names(group):
    """Name the variants of a group, e.g. CatBoost_NoImpute."""
    return [f"{group}_{imputation}" for imputation in imputations(group)]


def variants(group):
    """Build every variant of a group: {name: (pipeline, search space)}."""
    family, rfe = group.removesuffix("_RFE"), group.endswith("_RFE")
    result = {}
    for imputation in imputations(group):
        # These step names come from the original analysis and must not change: the
        # random search draws its candidates in the alphabetical order of the
        # parameter names ("impute__...", "classifier__...", "Over_sample__...").
        steps = [] if imputation == "NoImpute" else [("impute", imputation_step(family, imputation, rfe))]
        if imputation == "SMOTE":
            categorical = [CANDIDATES.index(name) for name in CATEGORICAL]
            steps.append(("Over_sample", SMOTENC(categorical_features=categorical, random_state=42)))
        if not rfe:
            steps.append(("keep_predictors", keep_predictors()))
        steps.append(("classifier", classifier(family, rfe)))
        pipeline = (ImbPipeline if imputation == "SMOTE" else Pipeline)(steps)
        result[f"{group}_{imputation}"] = (pipeline, search_space(family, imputation, rfe))
    return result


def groups():
    """List all groups: the six classifiers, then the four with recursive feature elimination."""
    return FAMILIES + [f"{family}_RFE" for family in RFE_FAMILIES]


def tuned(group, pipeline, space):
    """Wrap a variant in its hyperparameter search (the inner cross-validation)."""
    if space is None:
        return pipeline  # TabPFN variants with nothing to tune are used as they are
    return RandomizedSearchCV(
        pipeline, space, n_iter=N_ITER[group], cv=INNER_CV, scoring=SEARCH_SCORING,
        n_jobs=1 if group == "TabPFN" else -1,  # TabPFN: one GPU, so one job at a time
        random_state=42,
    )
