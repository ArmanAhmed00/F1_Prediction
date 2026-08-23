#!/usr/bin/env python3
"""Walk-forward backtest. This is where the actual evidence comes from.

For k in 6..11: train on everything before round k, predict round k's race,
score it, move on. Nothing from round k or later is visible during training, so
each fold tests the same procedure that gets run for round 12.

Sprints are used for training (they're real sessions and the dataset is small)
but never as a test fold - the question is how well we predict a Grand Prix.

Six races is a demonstration, not a validation. Reported as such.

    uv run python src/backtest.py
"""

from __future__ import annotations

import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold, cross_val_predict

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

from src import viz  # noqa: E402
from src.features import load_model_table  # noqa: E402
from src.models import MODEL_FEATURES, TARGET, get_models, split_xy  # noqa: E402
from src.simulate import simulate_race  # noqa: E402

REPORTS = BASE / "reports"

TEST_ROUNDS = list(range(6, 12))  # 6..11 inclusive
N_SIMS = 20_000
SEED = 42
CV = KFold(n_splits=5, shuffle=True, random_state=SEED)

# B1 and B2 predict with probability 1.0, so log loss blows up to infinity as
# soon as they're wrong. Everything gets clipped to this range before scoring -
# models and baselines alike, so the comparison stays fair.
EPS = 1e-6

# A-C are the fitted candidates. D is both model D and baseline B4 - same thing,
# scored twice. Selection only looks at A-C, since "can a fitted model beat the
# grid baseline" is the whole question and D can't be its own yardstick.
FITTED_CANDIDATES = ["A_ridge", "B_random_forest", "C_gradient_boosting"]
GRID_BASELINE = "D_grid_baseline"

BASELINE_LABELS = {
    "B1_pole_wins": "B1  pole-sitter wins, p=1.0",
    "B2_champ_leader_wins": "B2  championship leader wins, p=1.0",
    "B3_uniform": "B3  uniform 1/n",
    "D_grid_baseline": "B4  predicted position = grid position",
}


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------
def _clip(p):
    return np.clip(np.asarray(p, dtype="float64"), EPS, 1.0 - EPS)


def log_loss_binary(y_true, p) -> float:
    p = _clip(p)
    y = np.asarray(y_true, dtype="float64")
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def brier(y_true, p) -> float:
    p = _clip(p)
    y = np.asarray(y_true, dtype="float64")
    return float(np.mean((p - y) ** 2))


def score_probabilities(test: pd.DataFrame, p_win, p_podium, pred_pos=None) -> dict:
    """All the metrics for one fold.

    No accuracy here on purpose - at ~5% positives, "nobody wins" scores 95%.
    """
    y_win = (test[TARGET] == 1).to_numpy(dtype="float64")
    y_podium = (test[TARGET] <= 3).to_numpy(dtype="float64")
    p_win = np.asarray(p_win, dtype="float64")

    out = {
        "win_log_loss": log_loss_binary(y_win, p_win),
        "win_brier": brier(y_win, p_win),
        "podium_log_loss": log_loss_binary(y_podium, p_podium),
        "podium_brier": brier(y_podium, p_podium),
        "p_actual_winner": float(p_win[y_win == 1][0]) if y_win.any() else np.nan,
    }

    # Uniform has no meaningful argmax, so top-1 is undefined, not zero.
    if np.allclose(p_win, p_win[0]):
        out["top1_hit"] = np.nan
    else:
        out["top1_hit"] = float(y_win[int(np.argmax(p_win))])

    if pred_pos is None:
        # B1-B3 give probabilities but no finishing order, so rank metrics
        # don't exist for them.
        out["spearman_rho"] = np.nan
        out["mae_position"] = np.nan
    else:
        pred_pos = np.asarray(pred_pos, dtype="float64")
        actual = test[TARGET].to_numpy(dtype="float64")
        out["spearman_rho"] = float(spearmanr(pred_pos, actual).statistic)
        out["mae_position"] = float(np.mean(np.abs(pred_pos - actual)))
    return out


# --------------------------------------------------------------------------
# Baselines that emit probabilities directly
# --------------------------------------------------------------------------
def baseline_probabilities(name: str, test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    n = len(test)
    if name == "B1_pole_wins":
        p_win = (test["grid_position"] == test["grid_position"].min()).to_numpy(dtype="float64")
        p_podium = (test["grid_position"] <= 3).to_numpy(dtype="float64")
    elif name == "B2_champ_leader_wins":
        champ = test["championship_position"]
        p_win = (champ == champ.min()).to_numpy(dtype="float64")
        p_podium = (champ.rank(method="min") <= 3).to_numpy(dtype="float64")
    elif name == "B3_uniform":
        p_win = np.full(n, 1.0 / n)
        p_podium = np.full(n, 3.0 / n)
    else:
        raise KeyError(name)
    return p_win, p_podium


# --------------------------------------------------------------------------
# The walk-forward loop
# --------------------------------------------------------------------------
def estimate_sigma_cv(model, X: pd.DataFrame, y: pd.Series) -> float:
    """Sigma from CV inside the training split.

    Using the fold's own test residuals here would leak - the width of the
    distribution would be set by the thing we're about to score against. This
    never touches the test fold.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        preds = cross_val_predict(clone(model), X, y, cv=CV)
    return float(np.std(y.to_numpy() - preds, ddof=1))


def run_backtest(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scoreable = table[table["is_trainable"].fillna(False).astype(bool)].copy()
    models = get_models()

    fold_rows: list[dict] = []
    pred_rows: list[pd.DataFrame] = []
    importance_rows: list[pd.DataFrame] = []

    for k in TEST_ROUNDS:
        train = scoreable[scoreable["round"] < k]
        test = scoreable[(scoreable["round"] == k) & (scoreable["session_type"] == "R")]
        if test.empty:
            print(f"  round {k}: no race rows, skipping")
            continue

        X_train, y_train = split_xy(train)
        X_test, y_test = split_xy(test)
        print(f"  round {k:>2}: train {len(train):>3} rows  ->  test {len(test)} drivers")

        for name, model in models.items():
            fitted = clone(model)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fitted.fit(X_train, y_train)
                pred = np.asarray(fitted.predict(X_test), dtype="float64")

            sigma_cv = estimate_sigma_cv(model, X_train, y_train)
            sims = simulate_race(
                pred,
                sigma=sigma_cv,
                dnf_probs=test["dnf_rate_shrunk"].to_numpy(),
                n_sims=N_SIMS,
                seed=SEED,
            )
            metrics = score_probabilities(test, sims["p_win"], sims["p_podium"], pred)
            # Only reported, not used for scoring. Safe to hand to
            # predict_dutch.py since rounds 6-11 are history by round 12.
            metrics["sigma_test"] = float(np.std(y_test.to_numpy() - pred, ddof=1))
            metrics["sigma_cv"] = sigma_cv
            fold_rows.append({"model": name, "round": k, "n_drivers": len(test), **metrics})

            pred_rows.append(
                pd.DataFrame(
                    {
                        "model": name,
                        "round": k,
                        "driver_id": test["driver_id"].to_numpy(),
                        "abbreviation": test["abbreviation"].to_numpy(),
                        "predicted_position": pred,
                        "actual_position": y_test.to_numpy(),
                        "p_win": sims["p_win"].to_numpy(),
                        "p_podium": sims["p_podium"].to_numpy(),
                        "won": (y_test == 1).to_numpy(),
                        "podium": (y_test <= 3).to_numpy(),
                    }
                )
            )

            # On the held-out fold. Not impurity importance - that's in-sample
            # and biased toward high-cardinality features.
            if name != "D_grid_baseline":
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    imp = permutation_importance(
                        fitted, X_test, y_test, n_repeats=15,
                        random_state=SEED, scoring="neg_mean_absolute_error",
                    )
                importance_rows.append(
                    pd.DataFrame(
                        {"model": name, "round": k, "feature": MODEL_FEATURES,
                         "importance": imp.importances_mean}
                    )
                )

        for name in ("B1_pole_wins", "B2_champ_leader_wins", "B3_uniform"):
            p_win, p_podium = baseline_probabilities(name, test)
            metrics = score_probabilities(test, p_win, p_podium, pred_pos=None)
            metrics["sigma_test"] = np.nan
            metrics["sigma_cv"] = np.nan
            fold_rows.append({"model": name, "round": k, "n_drivers": len(test), **metrics})

    folds = pd.DataFrame(fold_rows)
    predictions = pd.concat(pred_rows, ignore_index=True)
    importances = (
        pd.concat(importance_rows, ignore_index=True)
        if importance_rows
        else pd.DataFrame(columns=["model", "round", "feature", "importance"])
    )
    return folds, predictions, importances


def summarise(folds: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "win_log_loss", "win_brier", "podium_log_loss", "podium_brier",
        "top1_hit", "spearman_rho", "mae_position", "p_actual_winner",
        "sigma_test", "sigma_cv",
    ]
    summary = folds.groupby("model")[metrics].mean()
    summary.insert(0, "n_folds", folds.groupby("model").size())
    summary["label"] = [BASELINE_LABELS.get(m, f"model {m}") for m in summary.index]
    return summary.sort_values("win_log_loss")


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------
def calibration_bins(predictions: pd.DataFrame, model_name: str, n_bins: int = 10) -> pd.DataFrame:
    """Calibration table: mean predicted probability vs observed frequency.

    Written out as numbers, not a chart, so the notebooks and the app can each
    draw it themselves. Quantile bins rather than equal-width, because the win
    probabilities all pile up near zero. Collapsed bins get dropped.
    """
    sel = predictions[predictions["model"] == model_name]
    frames = []
    for p_col, y_col, target in (("p_win", "won", "win"), ("p_podium", "podium", "podium")):
        p = sel[p_col].to_numpy(dtype="float64")
        y = sel[y_col].to_numpy(dtype="float64")
        binned = pd.qcut(p, n_bins, duplicates="drop")
        stats = (
            pd.DataFrame({"p": p, "y": y, "bin": binned})
            .groupby("bin", observed=True)
            .agg(mean_predicted=("p", "mean"), observed=("y", "mean"), n=("y", "size"))
            .reset_index(drop=True)
        )
        stats.insert(0, "target", target)
        stats.insert(1, "decile", range(1, len(stats) + 1))
        frames.append(stats)
    out = pd.concat(frames, ignore_index=True)
    out["model"] = model_name
    return out


def write_selection(summary: pd.DataFrame, best: str, sigma: float) -> None:
    baselines = ["B1_pole_wins", "B2_champ_leader_wins", "B3_uniform", GRID_BASELINE]
    row = summary.loc[best]

    beaten = {
        b: bool(row["win_log_loss"] < summary.loc[b, "win_log_loss"])
        for b in baselines
        if b in summary.index
    }
    key_baselines = ["B1_pole_wins", GRID_BASELINE]
    beats_all = all(beaten[b] for b in key_baselines if b in beaten)
    overall_best = summary["win_log_loss"].idxmin()

    lines = [
        "# Model selection",
        "",
        f"_Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')}_",
        "",
        "Selection criterion: **mean win log loss** across walk-forward folds "
        f"(rounds {TEST_ROUNDS[0]}-{TEST_ROUNDS[-1]}, {len(TEST_ROUNDS)} races). ",
        "No hyperparameter was tuned on a test fold; Ridge's alpha is chosen by "
        "cross-validation inside each training split.",
        "",
        f"## Selected: `{best}`",
        "",
        "| metric | value |",
        "| --- | --- |",
        f"| win log loss | {row['win_log_loss']:.4f} |",
        f"| win Brier | {row['win_brier']:.4f} |",
        f"| podium log loss | {row['podium_log_loss']:.4f} |",
        f"| podium Brier | {row['podium_brier']:.4f} |",
        f"| top-1 hit rate | {row['top1_hit']:.3f} |",
        f"| Spearman rho | {row['spearman_rho']:.3f} |",
        f"| MAE (positions) | {row['mae_position']:.3f} |",
        f"| mean residual sigma (test folds) | {sigma:.3f} |",
        "",
        "## Did it beat the baselines?",
        "",
    ]

    for b in baselines:
        if b in beaten:
            mark = "beats" if beaten[b] else "**does NOT beat**"
            lines.append(
                f"- {mark} `{b}` "
                f"({row['win_log_loss']:.4f} vs {summary.loc[b, 'win_log_loss']:.4f})"
            )
    lines.append("")

    if beats_all:
        lines.append(
            f"**Yes.** `{best}` has a lower mean win log loss than both B1 "
            "(pole-sitter wins) and B4 (finish = grid)."
        )
    else:
        lost_to = [b for b in key_baselines if b in beaten and not beaten[b]]
        lines += [
            "**No, and this is reported as found rather than tuned until something wins.**",
            "",
            f"`{best}` is the best of the fitted candidates ({', '.join(FITTED_CANDIDATES)}) "
            f"by win log loss, and it beats "
            f"{', '.join(f'`{b}`' for b in beaten if beaten[b]) or 'no baseline'}. "
            f"But it does not beat {', '.join(f'`{b}`' for b in lost_to)}.",
            "",
            f"The lowest win log loss of anything tested belongs to `{overall_best}` "
            f"({summary.loc[overall_best, 'win_log_loss']:.4f}). In other words: **on this "
            "dataset, assuming every driver finishes where they started is a better "
            "probabilistic predictor than any of the fitted models.**",
            "",
            "That is a believable result rather than a bug. Grid position is the single "
            "strongest signal available (Spearman rho = 0.745 in the EDA, and the highest "
            "mutual information of any feature), the other thirteen features are heavily "
            "collinear with it, and 300 rows is not enough for a fitted model to extract "
            "additional signal without adding more variance than it removes. Six test "
            "races also cannot separate methods this close together. The correct response "
            "is more data, not more tuning.",
        ]
    lines += ["", "## Full comparison", "", viz.to_markdown_table(summary.round(4), index=True), ""]
    lines += [
        "### Reading the table",
        "",
        "- B1 and B2 assert probability 1.0, so every probability is clipped to "
        f"[{EPS}, 1-{EPS}] before scoring; without that their log loss is infinite "
        "whenever they are wrong.",
        "- `spearman_rho` and `mae_position` are undefined for B1-B3: they emit "
        "probabilities but no finishing order.",
        "- `top1_hit` is undefined for B3, because a uniform distribution has no argmax.",
        "- Accuracy is not reported. At ~5% positives, predicting that nobody wins "
        "scores 95%.",
        "",
    ]
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "model_selection.md").write_text("\n".join(lines))

    with (REPORTS / "model_selection.json").open("w") as fh:
        json.dump(
            {
                "selected_model": best,
                "criterion": "mean win log loss over walk-forward folds",
                "test_rounds": TEST_ROUNDS,
                "residual_sigma": sigma,
                "beats_baselines": beats_all,
                "baseline_comparison": beaten,
                "metrics": {k: (None if pd.isna(v) else float(v))
                            for k, v in row.drop("label").items()},
                "candidates_considered": FITTED_CANDIDATES,
                "overall_best_including_baselines": str(summary["win_log_loss"].idxmin()),
            },
            fh,
            indent=2,
        )


def main() -> int:
    REPORTS.mkdir(parents=True, exist_ok=True)
    table = load_model_table()

    print("=" * 78)
    print("Walk-forward backtest — train on round < k, test on round k's race")
    print("=" * 78)
    folds, predictions, importances = run_backtest(table)

    summary = summarise(folds)
    display = summary.drop(columns=["label"]).round(4)

    print("\n" + "=" * 78)
    print("MODEL vs BASELINE — mean across folds, ranked by win log loss (lower is better)")
    print("=" * 78)
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(display.to_string())

    available = [m for m in FITTED_CANDIDATES if m in summary.index]
    best = min(available, key=lambda m: float(summary.loc[m, "win_log_loss"]))
    sigma = float(folds.loc[folds["model"] == best, "sigma_test"].mean())

    summary.drop(columns=["label"]).to_csv(REPORTS / "backtest_results.csv")
    folds.to_csv(REPORTS / "backtest_folds.csv", index=False)
    predictions.to_csv(REPORTS / "backtest_predictions.csv", index=False)
    if not importances.empty:
        (importances.groupby(["model", "feature"])["importance"].mean().reset_index()
         .to_csv(REPORTS / "permutation_importance.csv", index=False))

    calibration = calibration_bins(predictions, best)
    calibration.to_csv(REPORTS / "calibration.csv", index=False)
    write_selection(summary, best, sigma)

    print(f"\nselected model : {best}  (win log loss {summary.loc[best, 'win_log_loss']:.4f})")
    print(f"residual sigma : {sigma:.3f} positions  <- feeds src/predict_dutch.py")
    for baseline in ("B1_pole_wins", GRID_BASELINE):
        if baseline in summary.index:
            verdict = "BEATS" if summary.loc[best, "win_log_loss"] < summary.loc[
                baseline, "win_log_loss"] else "does NOT beat"
            print(f"  {verdict} {BASELINE_LABELS.get(baseline, baseline)}")

    print(f"\nwrote {(REPORTS / 'backtest_results.csv').relative_to(BASE)}")
    print(f"wrote {(REPORTS / 'model_selection.md').relative_to(BASE)}")
    print(f"wrote {(REPORTS / 'calibration.csv').relative_to(BASE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
