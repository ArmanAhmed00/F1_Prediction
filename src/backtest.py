#!/usr/bin/env python3
"""Walk-forward backtest. This is where the actual evidence comes from.

For k in 6..12: train on everything before round k, predict round k's race,
score it, move on. Nothing from round k or later is visible during training, so
each fold tests the same procedure that gets run for round 13.

Sprints are used for training (they're real sessions and the dataset is small)
but never as a test fold - the question is how well we predict a Grand Prix.

Seven races is a demonstration, not a validation. Reported as such.

MODEL SELECTION IS FROZEN. It was made once, on the rounds 6-11 folds, and it
stays made. This script now re-scores the frozen choice on one more fold and
says whether it is still the best performer - it does not act on the answer.
Re-picking the winner every week would mean each race is predicted by a
different model with a sample size of one, which demonstrates nothing. Any
change is a v2, evaluated on the whole walk-forward and run alongside v1.

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

TEST_ROUNDS = list(range(6, 13))  # 6..12 inclusive
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

# Frozen at the round 12 backtest (folds 6-11) and not up for re-selection.
# See the module docstring. Changing this is a v2, not a weekly maintenance job.
FROZEN_MODEL = "A_ridge"
FROZEN_AT = "round 12 backtest, walk-forward folds 6-11"

# The round these folds are being run in service of. Names the archived copy of
# the summary, so each week's evidence survives the next week's overwrite.
RESULTS_TAG = TEST_ROUNDS[-1] + 1

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
    """Write the selection report for the FROZEN model.

    ``best`` is the frozen choice, not an argmin recomputed today. The report
    says whether it would still win if we were choosing now, and then goes on
    using it regardless - that is the point of freezing it.
    """
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

    available = [m for m in FITTED_CANDIDATES if m in summary.index]
    best_fitted_now = min(available, key=lambda m: float(summary.loc[m, "win_log_loss"]))
    still_best = best_fitted_now == best

    lines = [
        "# Model selection",
        "",
        f"_Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')}_",
        "",
        f"**The model is frozen.** `{best}` was selected at the {FROZEN_AT} and is not "
        "re-selected here. Only the data changes between races; if the model changed too, "
        "each race would be predicted by a different model with a sample size of one, "
        "which would demonstrate nothing. A challenger is a v2: evaluated on the whole "
        "walk-forward and run alongside v1, never swapped in silently.",
        "",
        "Original selection criterion: **mean win log loss** across walk-forward folds. "
        f"Now re-scored over rounds {TEST_ROUNDS[0]}-{TEST_ROUNDS[-1]} "
        f"({len(TEST_ROUNDS)} races). "
        "No hyperparameter was tuned on a test fold; Ridge's alpha is chosen by "
        "cross-validation inside each training split.",
        "",
        f"## In use: `{best}` (frozen)",
        "",]
    if still_best:
        lines += [
            f"Still the best of the fitted candidates on the extended folds "
            f"({summary.loc[best, 'win_log_loss']:.4f} win log loss). The freeze costs "
            "nothing this week.",
            "",
        ]
    else:
        lines += [
            f"**No longer the best of the fitted candidates.** On folds "
            f"{TEST_ROUNDS[0]}-{TEST_ROUNDS[-1]}, `{best_fitted_now}` now scores "
            f"{summary.loc[best_fitted_now, 'win_log_loss']:.4f} against "
            f"`{best}`'s {summary.loc[best, 'win_log_loss']:.4f}. "
            f"**`{best}` is used anyway.** One extra fold reordering two models this "
            "close together is exactly the noise the freeze exists to absorb; swapping "
            "now would be selection on the test set. Logged in "
            "`reports/v2_backlog.md` instead.",
            "",
        ]
    lines += [
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
            f"`{best}` is the frozen model, chosen from ({', '.join(FITTED_CANDIDATES)}) "
            f"by win log loss at the {FROZEN_AT}, and it beats "
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
            "collinear with it, and a few hundred rows is not enough for a fitted model "
            f"to extract additional signal without adding more variance than it removes. "
            f"{len(TEST_ROUNDS)} test races also cannot separate methods this close "
            "together. The correct response is more data, not more tuning.",
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
                "selection_frozen": True,
                "frozen_at": FROZEN_AT,
                "still_best_fitted_candidate": bool(still_best),
                "best_fitted_candidate_on_these_folds": best_fitted_now,
                "criterion": "mean win log loss over walk-forward folds "
                             "(frozen: re-scored, not re-selected)",
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

    # FROZEN. Not an argmin - see the module docstring. The argmin is computed
    # only so the report can say whether the frozen choice would still win.
    best = FROZEN_MODEL
    if best not in summary.index:
        raise KeyError(f"frozen model {best!r} is not among the scored candidates")
    available = [m for m in FITTED_CANDIDATES if m in summary.index]
    best_fitted_now = min(available, key=lambda m: float(summary.loc[m, "win_log_loss"]))
    sigma = float(folds.loc[folds["model"] == best, "sigma_test"].mean())

    summary.drop(columns=["label"]).to_csv(REPORTS / "backtest_results.csv")
    summary.drop(columns=["label"]).to_csv(REPORTS / f"backtest_results_r{RESULTS_TAG}.csv")
    folds.to_csv(REPORTS / "backtest_folds.csv", index=False)
    predictions.to_csv(REPORTS / "backtest_predictions.csv", index=False)
    if not importances.empty:
        (importances.groupby(["model", "feature"])["importance"].mean().reset_index()
         .to_csv(REPORTS / "permutation_importance.csv", index=False))

    calibration = calibration_bins(predictions, best)
    calibration.to_csv(REPORTS / "calibration.csv", index=False)
    write_selection(summary, best, sigma)

    print(f"\nfrozen model   : {best}  (win log loss {summary.loc[best, 'win_log_loss']:.4f})")
    print(f"               frozen at the {FROZEN_AT}; NOT re-selected today")
    if best_fitted_now == best:
        print(f"               still the best fitted candidate on folds "
              f"{TEST_ROUNDS[0]}-{TEST_ROUNDS[-1]}")
    else:
        print(f"               NOTE: {best_fitted_now} now scores better "
              f"({summary.loc[best_fitted_now, 'win_log_loss']:.4f}). Using {best} "
              f"anyway - see reports/v2_backlog.md")
    print(f"residual sigma : {sigma:.3f} positions  <- feeds src/predict_race.py")
    for baseline in ("B1_pole_wins", GRID_BASELINE):
        if baseline in summary.index:
            verdict = "BEATS" if summary.loc[best, "win_log_loss"] < summary.loc[
                baseline, "win_log_loss"] else "does NOT beat"
            print(f"  {verdict} {BASELINE_LABELS.get(baseline, baseline)}")

    print(f"\nwrote {(REPORTS / 'backtest_results.csv').relative_to(BASE)}")
    print(f"wrote {(REPORTS / f'backtest_results_r{RESULTS_TAG}.csv').relative_to(BASE)}")
    print(f"wrote {(REPORTS / 'model_selection.md').relative_to(BASE)}")
    print(f"wrote {(REPORTS / 'calibration.csv').relative_to(BASE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
