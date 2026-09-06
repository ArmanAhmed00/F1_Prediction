#!/usr/bin/env python3
"""Lock in the prediction for one Grand Prix.

Takes the FROZEN model, trains it on every completed session before the target
round, predicts finishing position, then simulates to get win and podium
probabilities.

The model is not chosen here and is not chosen weekly. `reports/model_selection.json`
records a frozen choice; this script uses it and refuses to look for a better one.
Only the data changes between races. If the model changed too, three race
predictions would be three different models with a sample size of one each.

Timestamped, seeded and git-stamped so it can be regenerated exactly. Commit the
output before the race - a prediction you can still edit afterwards isn't one.

    uv run python src/predict_race.py --round 13 --label monza
    uv run python src/predict_race.py --round 14 --label madrid

Nothing below is round-specific except the contents of EXPECTED_POLE, which is a
per-round ingestion tripwire rather than a model input.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

from src.features import load_model_table  # noqa: E402
from src.io_utils import load_raw  # noqa: E402
from src.models import (  # noqa: E402
    EXCLUDED_FEATURES,
    FEATURE_DEFINITIONS,
    MODEL_FEATURES,
    get_models,
    model_hyperparameters,
    split_xy,
)
from src.simulate import DEFAULT_N_SIMS, DEFAULT_SEED, simulate_race  # noqa: E402

SEASON = 2026
PREDICTIONS = BASE / "predictions"
REPORTS = BASE / "reports"
SELECTION = REPORTS / "model_selection.json"
LOG_PATH = PREDICTIONS / "prediction_log.csv"

# Ingestion tripwire, not a feature. If FastF1 hands back a weekend whose pole
# sitter isn't who we know it to be, the round mapping or the cache is wrong and
# every number downstream is worthless - so stop rather than predict on it.
# Add a round here on the Saturday. --expect-pole overrides for a one-off.
EXPECTED_POLE: dict[int, str] = {
    13: "gasly",
}

# Below this the top pick's team is not a plausible favourite on season pace,
# which usually means grid position is carrying the prediction on its own.
PLAUSIBILITY_TEAM_RANK = 5

LOG_COLUMNS = [
    "round",
    "label",
    "generated_at_utc",
    "model_name",
    "predicted_winner",
    "p_win",
    "predicted_podium",
    "git_commit_sha",
]


def fail(message: str, code: int = 1) -> int:
    print(f"\nERROR: {message}\n", file=sys.stderr)
    return code


def git_commit_sha() -> str:
    """Current HEAD, or "" if this isn't a git checkout / git isn't installed.

    Never raises: a missing SHA degrades the provenance of the log row, it does
    not invalidate the prediction.
    """
    try:
        out = subprocess.run(
            ["git", "-C", str(BASE), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.stdout.strip() if out.returncode == 0 else ""


def load_selection() -> dict:
    if not SELECTION.exists():
        raise FileNotFoundError(
            f"{SELECTION.relative_to(BASE)} is missing - run `uv run python src/backtest.py` first"
        )
    with SELECTION.open() as fh:
        return json.load(fh)


def event_metadata(rnd: int) -> dict:
    """Event name and circuit off the ingested schedule, so nothing is hardcoded."""
    try:
        schedule = load_raw("schedule")
    except FileNotFoundError:
        return {}
    row = schedule[schedule["round"] == rnd]
    if row.empty:
        return {}
    row = row.iloc[0]
    return {
        "race": f"{SEASON} {row['EventName']}",
        "circuit": str(row["Location"]),
        "country": str(row["Country"]),
        "event_format": str(row["EventFormat"]),
    }


def team_colors_for(team_names) -> dict[str, str]:
    """Look up team colours once, here, and store them in the output.

    Deliberate: the Streamlit app should never need fastf1 or a network call
    just to colour a bar chart.
    """
    from src import viz

    return {name: viz.team_color(name) for name in sorted(set(team_names))}


def _jsonable(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return value


# --------------------------------------------------------------------------
# Pre-flight checks. Every one of these runs BEFORE anything is written.
# --------------------------------------------------------------------------
def check_pole_sitter(rnd: int, expected: str | None) -> None:
    """The ingestion tripwire.

    Reads the raw qualifying result rather than the model table, because the
    point is to check what came out of FastF1, not what survived feature
    engineering.
    """
    if expected is None:
        print(
            f"WARNING: no expected pole sitter recorded for round {rnd}. The ingestion "
            f"tripwire is NOT armed. Add round {rnd} to EXPECTED_POLE in "
            "src/predict_race.py, or pass --expect-pole.",
            file=sys.stderr,
        )
        return

    quali = load_raw("results")
    quali = quali[(quali["round"] == rnd) & (quali["session_type"] == "Q")]
    assert not quali.empty, f"round {rnd} has no qualifying results ingested"

    pole = quali.loc[quali["Position"].astype("float64").idxmin()]
    actual = str(pole["DriverId"])
    assert actual == expected, (
        f"round {rnd} pole sitter is {actual!r} ({pole['Abbreviation']}), expected "
        f"{expected!r}. The ingestion is wrong - the round mapping, the cache or the "
        "session is not what it claims to be, and everything downstream of this is "
        "worthless. Re-run: uv run python src/ingest.py --rounds "
        f"{rnd} --force"
    )
    print(f"  pole sitter          {pole['Abbreviation']} ({actual}) — as expected")


def check_features_complete(race_rows: pd.DataFrame, rnd: int) -> None:
    """Every driver has a grid slot, and nothing is NaN after imputation."""
    missing_grid = race_rows.loc[race_rows["grid_position"].isna(), "abbreviation"].tolist()
    assert not missing_grid, (
        f"round {rnd}: no grid position for {missing_grid}. Check "
        f"data/manual/grid_R{rnd}.csv."
    )

    grid = race_rows["grid_position"].astype("float64").to_numpy()
    n = grid.size
    assert sorted(int(g) for g in grid) == list(range(1, n + 1)), (
        f"round {rnd}: grid positions are not a permutation of 1..{n}: "
        f"{sorted(grid.tolist())}"
    )

    nan_features = {
        col: int(race_rows[col].isna().sum())
        for col in MODEL_FEATURES
        if race_rows[col].isna().any()
    }
    assert not nan_features, (
        f"round {rnd}: features are still NaN after imputation: {nan_features}. "
        "features.py should have made this impossible - do not predict on it."
    )
    print(f"  grid + features      complete for all {n} drivers, no NaN after imputation")


def check_probabilities(race_rows: pd.DataFrame) -> None:
    """The simulation's own invariants, re-checked at the boundary."""
    p_win = race_rows["p_win"].to_numpy(dtype="float64")
    p_podium = race_rows["p_podium"].to_numpy(dtype="float64")

    total = float(p_win.sum())
    assert abs(total - 1.0) < 1e-6, f"p_win sums to {total!r}, expected 1.0 within 1e-6"

    violations = race_rows.loc[p_win > p_podium + 1e-12, "abbreviation"].tolist()
    assert not violations, f"p_win exceeds p_podium for {violations}"
    print(f"  probabilities        p_win sums to {total:.8f}; p_win <= p_podium for all")


def plausibility_guard(race_rows: pd.DataFrame) -> str | None:
    """Warn, loudly, if the favourite's car has no business being the favourite.

    A flag, not an error. Lower team_form_pace_ewm is faster, so the rank is
    ascending. Prints and returns the warning; it never blocks - the prediction
    is locked as it stands and today is not the day to retune anything.
    """
    top = race_rows.sort_values("p_win", ascending=False).iloc[0]
    team_pace = (
        race_rows.groupby("team_name")["team_form_pace_ewm"].min().sort_values()
    )
    rank = int(team_pace.index.get_loc(top["team_name"])) + 1
    if rank <= PLAUSIBILITY_TEAM_RANK:
        print(
            f"  plausibility         top pick {top['abbreviation']} is from "
            f"{top['team_name']}, ranked #{rank} of {len(team_pace)} on season pace"
        )
        return None

    message = (
        f"Top pick is from a team ranked #{rank} on season pace. The model may be\n"
        "over-weighting grid position. This is a flag, not an error - review before\n"
        "committing, but DO NOT retune the model today."
    )
    banner = "!" * 78
    print(f"\n{banner}\nPLAUSIBILITY GUARD\n\n{message}\n{banner}\n", file=sys.stderr)
    return message.replace("\n", " ")


def append_log(row: dict) -> None:
    """One row per locked prediction. Append-only; the history is the point."""
    frame = pd.DataFrame([row], columns=LOG_COLUMNS)
    header = not LOG_PATH.exists()
    frame.to_csv(LOG_PATH, mode="a", header=header, index=False)


# --------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--round", type=int, required=True, help="round to predict")
    parser.add_argument(
        "--label", required=True, help="short slug for the filenames, e.g. monza"
    )
    parser.add_argument(
        "--expect-pole",
        default=None,
        help="driver_id expected on pole; overrides EXPECTED_POLE for this run",
    )
    parser.add_argument("--n-sims", type=int, default=DEFAULT_N_SIMS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rnd, label = args.round, args.label
    slug = f"{SEASON}_R{rnd}_{label}"
    PREDICTIONS.mkdir(parents=True, exist_ok=True)

    table = load_model_table()

    # ---- 1. Is the weekend's data actually present? -----------------------
    race_rows = table[
        (table["round"] == rnd)
        & (table["session_type"] == "R")
        & table["is_prediction"].fillna(False).astype(bool)
    ].copy()
    if race_rows.empty:
        return fail(
            f"no round {rnd} prediction rows in the model table.\n"
            f"  Either qualifying has not been ingested, or the race has already run\n"
            f"  (in which case there is nothing left to predict).\n"
            f"  Run:  uv run python src/ingest.py --rounds {rnd}\n"
            "  then: uv run python src/features.py"
        )

    if race_rows["quali_pct_off_pole_imputed"].fillna(True).astype(bool).all():
        return fail(
            f"every round {rnd} driver has an IMPUTED qualifying time, which means "
            "qualifying has not been ingested.\n"
            f"  Run:  uv run python src/ingest.py --rounds {rnd}\n"
            "  then: uv run python src/features.py"
        )

    missing_features = [c for c in MODEL_FEATURES if c not in table.columns]
    if missing_features:
        return fail(
            f"the model table is missing features {missing_features}.\n"
            "  Run:  uv run python src/features.py"
        )

    selection = load_selection()
    model_name = selection["selected_model"]
    sigma = float(selection["residual_sigma"])

    print("=" * 78)
    print(f"PRE-FLIGHT CHECKS — round {rnd} ({label})")
    print("=" * 78)

    # ---- 2. Tripwires that run before anything is computed ----------------
    expected_pole = args.expect_pole if args.expect_pole else EXPECTED_POLE.get(rnd)
    check_pole_sitter(rnd, expected_pole)
    check_features_complete(race_rows, rnd)

    # ---- 3. Train on every completed session before this round ------------
    scoreable = table[table["is_trainable"].fillna(False).astype(bool)]
    train = scoreable[scoreable["round"] < rnd]
    X_train, y_train = split_xy(train)

    model = clone(get_models()[model_name])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_train, y_train)

    # ---- 4. Predict --------------------------------------------------------
    X_race = race_rows.loc[:, MODEL_FEATURES]
    race_rows["predicted_position"] = np.asarray(model.predict(X_race), dtype="float64")

    # The grid baseline beat every fitted model in the backtest, so keep its
    # numbers alongside rather than quietly dropping them.
    baseline_model = clone(get_models()["D_grid_baseline"]).fit(X_train, y_train)
    baseline_pred = np.asarray(baseline_model.predict(X_race), dtype="float64")

    # ---- 5. Simulate -------------------------------------------------------
    sims = simulate_race(
        race_rows["predicted_position"].to_numpy(),
        sigma=sigma,
        dnf_probs=race_rows["dnf_rate_shrunk"].to_numpy(),
        n_sims=args.n_sims,
        seed=args.seed,
    )
    for col in sims.columns:
        race_rows[col] = sims[col].to_numpy()

    baseline_sims = simulate_race(
        baseline_pred,
        sigma=float(selection["metrics"].get("sigma_test") or sigma),
        dnf_probs=race_rows["dnf_rate_shrunk"].to_numpy(),
        n_sims=args.n_sims,
        seed=args.seed,
    )
    race_rows["baseline_p_win"] = baseline_sims["p_win"].to_numpy()

    # ---- 6. The remaining checks, still before any write -------------------
    check_probabilities(race_rows)
    plausibility_note = plausibility_guard(race_rows)

    ranked = race_rows.sort_values("p_win", ascending=False).reset_index(drop=True)
    colors = team_colors_for(ranked["team_name"])
    ranked["team_color"] = ranked["team_name"].map(colors)

    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    commit_sha = git_commit_sha()
    podium = ranked.head(3)
    event = event_metadata(rnd)

    # ---- 7. Write the locked JSON -----------------------------------------
    drivers = []
    for _, row in ranked.iterrows():
        drivers.append(
            {
                "driver_id": row["driver_id"],
                "abbreviation": row["abbreviation"],
                "team_name": row["team_name"],
                "team_color": row["team_color"],
                "grid_position": _jsonable(row["grid_position"]),
                "predicted_position": round(float(row["predicted_position"]), 4),
                "p_win": round(float(row["p_win"]), 6),
                "p_podium": round(float(row["p_podium"]), 6),
                "baseline_p_win": round(float(row["baseline_p_win"]), 6),
                "features": {f: _jsonable(row[f]) for f in MODEL_FEATURES},
            }
        )

    grid_file = BASE / "data" / "manual" / f"grid_R{rnd}.csv"
    caveats = [
        "The grid baseline (B4) achieved a lower win log loss than this model over "
        "the backtest races. See reports/model_selection.md.",
        f"{len(selection['test_rounds'])} backtest races is a demonstration, not a "
        "validation of calibration.",
        "The model has no circuit-specific term. It applies the same weighting to "
        "grid position at every track, regardless of how hard overtaking is there.",
    ]
    if grid_file.exists():
        caveats.insert(
            0,
            f"grid_position is the OFFICIAL starting grid, transcribed by hand into "
            f"data/manual/grid_R{rnd}.csv, with penalties applied.",
        )
    else:
        caveats.insert(
            0,
            f"grid_position for round {rnd} is QUALIFYING ORDER; grid penalties are "
            "NOT reflected.",
        )
    if plausibility_note:
        caveats.insert(0, f"PLAUSIBILITY FLAG: {plausibility_note}")

    payload = {
        "race": event.get("race", f"{SEASON} round {rnd}"),
        "season": SEASON,
        "round": rnd,
        "label": label,
        "circuit": event.get("circuit"),
        "generated_at_utc": generated_at,
        "git_commit_sha": commit_sha,
        "model": {
            "name": model_name,
            "frozen": bool(selection.get("selection_frozen", False)),
            "frozen_at": selection.get("frozen_at"),
            "still_best_fitted_candidate": selection.get("still_best_fitted_candidate"),
            "hyperparameters": model_hyperparameters(model_name),
            "selection_criterion": selection["criterion"],
            "beats_baselines": selection["beats_baselines"],
            "best_including_baselines": selection.get("overall_best_including_baselines"),
        },
        "features": {
            "used": MODEL_FEATURES,
            "n_features": len(MODEL_FEATURES),
            "definitions": FEATURE_DEFINITIONS,
            "excluded": EXCLUDED_FEATURES,
        },
        "training": {
            "rounds": sorted(int(r) for r in train["round"].unique()),
            "n_rows": int(len(train)),
            "sessions": sorted(train["session_type"].unique().tolist()),
            "note": f"races and sprints; DNS rows and the unraced round {rnd} excluded",
        },
        "backtest": {
            "test_rounds": selection["test_rounds"],
            "metrics": selection["metrics"],
            "baseline_comparison": selection["baseline_comparison"],
        },
        "simulation": {
            "residual_sigma": round(sigma, 6),
            "sigma_source": (
                "mean residual sd of the frozen model over backtest folds "
                f"{min(selection['test_rounds'])}-{max(selection['test_rounds'])}"
            ),
            "n_sims": args.n_sims,
            "seed": args.seed,
        },
        "predicted_winner": {
            "driver_id": ranked.at[0, "driver_id"],
            "abbreviation": ranked.at[0, "abbreviation"],
            "team_name": ranked.at[0, "team_name"],
            "p_win": round(float(ranked.at[0, "p_win"]), 6),
        },
        "predicted_podium": [
            {
                "position": i + 1,
                "driver_id": row["driver_id"],
                "abbreviation": row["abbreviation"],
                "team_name": row["team_name"],
                "p_win": round(float(row["p_win"]), 6),
                "p_podium": round(float(row["p_podium"]), 6),
            }
            for i, (_, row) in enumerate(podium.iterrows())
        ],
        "caveats": caveats,
        "drivers": drivers,
    }

    json_path = PREDICTIONS / f"{slug}.json"
    with json_path.open("w") as fh:
        json.dump(payload, fh, indent=2)

    # ---- 8. Flat CSV for the report ---------------------------------------
    csv_cols = (
        ["driver_id", "abbreviation", "team_name", "team_color", "grid_position",
         "predicted_position", "p_win", "p_podium", "baseline_p_win"] + MODEL_FEATURES
    )
    csv_path = PREDICTIONS / f"{slug}.csv"
    ranked.loc[:, csv_cols].to_csv(csv_path, index=False, float_format="%.6f")

    # ---- 9. Append to the running log -------------------------------------
    append_log(
        {
            "round": rnd,
            "label": label,
            "generated_at_utc": generated_at,
            "model_name": model_name,
            "predicted_winner": payload["predicted_winner"]["abbreviation"],
            "p_win": round(float(payload["predicted_winner"]["p_win"]), 6),
            "predicted_podium": " ".join(p["abbreviation"] for p in payload["predicted_podium"]),
            "git_commit_sha": commit_sha,
        }
    )

    # ---- 10. Human-readable summary ---------------------------------------
    print("\n" + "=" * 78)
    print(f"{payload['race']} — locked prediction")
    print("=" * 78)
    print(f"model            {model_name}"
          f"{' (FROZEN)' if payload['model']['frozen'] else ''}")
    print(f"trained on       rounds {min(payload['training']['rounds'])}-"
          f"{max(payload['training']['rounds'])}, {payload['training']['n_rows']} rows")
    print(f"residual sigma   {sigma:.3f} positions")
    print(f"simulations      {args.n_sims:,} (seed {args.seed})")
    print(f"generated        {generated_at}")
    print(f"git commit       {commit_sha or '(unavailable)'}")

    print(f"\n{'#':<3} {'DRIVER':<5} {'TEAM':<17} {'GRID':>4} {'PACE':>6} {'PRED':>6} "
          f"{'P(win)':>8} {'P(podium)':>10}")
    print("-" * 78)
    for i, row in ranked.head(8).iterrows():
        print(f"{i + 1:<3} {row['abbreviation']:<5} {row['team_name']:<17} "
              f"{row['grid_position']:>4.0f} {row['team_form_pace_ewm']:>6.3f} "
              f"{row['predicted_position']:>6.2f} "
              f"{row['p_win'] * 100:>7.1f}% {row['p_podium'] * 100:>9.1f}%")
    print("\nPACE = team_form_pace_ewm, the team's EWM qualifying gap to pole in %.")
    print("Lower is faster. Grid and pace side by side is the tension to watch.")

    winner = payload["predicted_winner"]
    print(f"\npredicted winner : {winner['abbreviation']} ({winner['team_name']}) "
          f"at {winner['p_win'] * 100:.1f}%")
    print("predicted podium : " + ", ".join(p["abbreviation"] for p in payload["predicted_podium"]))

    print(f"\nwrote {json_path.relative_to(BASE)}")
    print(f"wrote {csv_path.relative_to(BASE)}")
    print(f"appended to {LOG_PATH.relative_to(BASE)}")

    print("\n" + "!" * 78)
    print("COMMIT THIS PREDICTION NOW, BEFORE THE RACE STARTS:")
    print(f"    git add {json_path.relative_to(BASE)} {csv_path.relative_to(BASE)} "
          f"{LOG_PATH.relative_to(BASE)}")
    print(f'    git commit -m "Lock {payload["race"]} prediction ({generated_at})"')
    print("A prediction that can still be edited after the race is not a prediction.")
    print("!" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
