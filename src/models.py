"""The four candidate models. All of them predict finish_position.

Nothing here predicts "won" directly - 16 winners against 343 rows is not enough
to fit a classifier on. Win and podium probabilities come out of src.simulate
afterwards.

Everything follows the sklearn estimator API so backtest.py can just loop.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.base import BaseEstimator as _Est
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parents[1]

RANDOM_STATE = 42

# --------------------------------------------------------------------------
# Feature set
# --------------------------------------------------------------------------
# 14 features. src.features also exports has_sprint_data, track_temp_mean and
# was_wet - all three scored 0.000 mutual information in the EDA, so they are out.
# Decided before the backtest ran, not from looking at the folds.
MODEL_FEATURES = [
    "quali_pct_off_pole",
    "grid_position",
    "quali_teammate_delta_pct",
    "fp_best_pct_off_fastest",
    "fp_longrun_pace_pct",
    "sprint_finish_position",
    "sprint_positions_gained",
    "form_quali_pace_ewm",
    "form_finish_pos_ewm",
    "form_race_pace_ewm",
    "team_form_pace_ewm",
    "positions_gained_ewm",
    "dnf_rate_shrunk",
    "championship_position",
]

EXCLUDED_FEATURES = {
    "has_sprint_data": "zero mutual information with finish_position (EDA §5)",
    "track_temp_mean": "zero mutual information with finish_position (EDA §5)",
    "was_wet": "zero mutual information with finish_position (EDA §5)",
}

TARGET = "finish_position"

FEATURE_DEFINITIONS = {
    "quali_pct_off_pole": "Qualifying gap to pole, as a percentage of the pole lap.",
    "grid_position": "Starting position on the grid (1 = pole).",
    "quali_teammate_delta_pct": "Driver's qualifying gap minus their teammate's, same session.",
    "fp_best_pct_off_fastest": "Best clean practice lap, as a percentage off the fastest in the field.",
    "fp_longrun_pace_pct": "Median lap of the driver's longest clean practice stint, % off the field best.",
    "sprint_finish_position": "Finishing position in that weekend's sprint (imputed when there was none).",
    "sprint_positions_gained": "Sprint grid minus sprint finish.",
    "form_quali_pace_ewm": "Exponentially weighted mean of past qualifying gaps (halflife 3 rounds).",
    "form_finish_pos_ewm": "Exponentially weighted mean of past finishing positions.",
    "form_race_pace_ewm": "Exponentially weighted mean of past clean-air race pace, % off session best.",
    "team_form_pace_ewm": "Team-level exponentially weighted qualifying pace — two cars per estimate.",
    "positions_gained_ewm": "Exponentially weighted mean of past (grid − finish); racecraft and strategy.",
    "dnf_rate_shrunk": "Beta-Binomial posterior retirement probability, shrunk toward the field rate.",
    "championship_position": "Championship standing going into the session, from cumulative points.",
}


# --------------------------------------------------------------------------
# Model D: the no-fit baseline
# --------------------------------------------------------------------------
class GridPositionBaseline(BaseEstimator, RegressorMixin):
    """Everyone finishes where they started. Nothing to fit.

    Wrapped as an estimator so the backtest can score it alongside the rest.
    This is the bar the fitted models have to clear.
    """

    def __init__(self, grid_index: int = 1):
        self.grid_index = grid_index

    def fit(self, X, y=None):  # noqa: ARG002 - y unused by design
        self.is_fitted_ = True
        self.n_features_in_ = np.shape(X)[1]
        return self

    def predict(self, X):
        values = X.to_numpy() if isinstance(X, pd.DataFrame) else np.asarray(X)
        return values[:, self.grid_index].astype("float64")


def _ridge_pipeline() -> GridSearchCV:
    """Ridge on standardised features, alpha picked by inner CV.

    The search only ever sees the training split it is handed, so no test fold
    leaks into tuning.
    """
    inner = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", Ridge(random_state=None)),
        ]
    )
    return GridSearchCV(
        inner,
        param_grid={"model__alpha": [0.1, 1.0, 3.0, 10.0, 30.0, 100.0]},
        scoring="neg_mean_absolute_error",
        cv=KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=1,
    )


def _forest_pipeline() -> Pipeline:
    # No scaler - trees don't care about scale.
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=500,
                    min_samples_leaf=5,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def _boosted_pipeline() -> tuple[Pipeline, str, dict]:
    """Boosting with deliberately tiny trees - there are only ~300 rows.

    LightGBM first, but its macOS wheel needs libomp which isn't always around.
    Falls back to sklearn's histogram booster (same family, ships its own
    OpenMP); the hyperparameters map across one-to-one.
    """
    params = {
        "n_estimators": 400,
        "learning_rate": 0.03,
        "num_leaves": 7,
        "min_child_samples": 15,
        "reg_lambda": 5,
        "random_state": RANDOM_STATE,
    }
    try:
        from lightgbm import LGBMRegressor

        model = LGBMRegressor(**params, verbose=-1)
        backend = "lightgbm.LGBMRegressor"
        reported = params
    except (ImportError, OSError):
        # OSError = wheel is installed but libomp is missing.
        equivalent = {
            "max_iter": params["n_estimators"],
            "learning_rate": params["learning_rate"],
            "max_leaf_nodes": params["num_leaves"],
            "min_samples_leaf": params["min_child_samples"],
            "l2_regularization": params["reg_lambda"],
            "random_state": params["random_state"],
        }
        model = HistGradientBoostingRegressor(**equivalent)
        backend = "sklearn.HistGradientBoostingRegressor (LightGBM unavailable: no libomp)"
        reported = equivalent

    return (
        Pipeline([("impute", SimpleImputer(strategy="median")), ("model", model)]),
        backend,
        reported,
    )


def get_models() -> dict[str, _Est]:
    """The four candidates, keyed by the names used in the reports."""
    boosted, _, _ = _boosted_pipeline()
    return {
        "A_ridge": _ridge_pipeline(),
        "B_random_forest": _forest_pipeline(),
        "C_gradient_boosting": boosted,
        "D_grid_baseline": GridPositionBaseline(
            grid_index=MODEL_FEATURES.index("grid_position")
        ),
    }


def model_hyperparameters(name: str) -> dict:
    """Hyperparameters for one candidate, JSON-serialisable."""
    if name == "A_ridge":
        return {
            "estimator": "sklearn.linear_model.Ridge",
            "alpha_grid": [0.1, 1.0, 3.0, 10.0, 30.0, 100.0],
            "alpha_selected_by": "5-fold CV on the training split (never on a test fold)",
            "preprocessing": ["SimpleImputer(median)", "StandardScaler"],
        }
    if name == "B_random_forest":
        return {
            "estimator": "sklearn.ensemble.RandomForestRegressor",
            "n_estimators": 500,
            "min_samples_leaf": 5,
            "random_state": RANDOM_STATE,
            "preprocessing": ["SimpleImputer(median)"],
        }
    if name == "C_gradient_boosting":
        _, backend, params = _boosted_pipeline()
        return {"estimator": backend, **params, "preprocessing": ["SimpleImputer(median)"]}
    if name == "D_grid_baseline":
        return {"estimator": "GridPositionBaseline", "note": "predicts finish = grid; no fitting"}
    raise KeyError(f"unknown model {name!r}")


def boosting_backend() -> str:
    return _boosted_pipeline()[1]


def split_xy(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """X, y - with the columns always in the same order."""
    return frame.loc[:, MODEL_FEATURES], frame[TARGET].astype("float64")


__all__ = [
    "EXCLUDED_FEATURES",
    "FEATURE_DEFINITIONS",
    "MODEL_FEATURES",
    "RANDOM_STATE",
    "TARGET",
    "GridPositionBaseline",
    "boosting_backend",
    "get_models",
    "model_hyperparameters",
    "split_xy",
]
