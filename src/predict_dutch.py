#!/usr/bin/env python3
"""Lock in the round 12 (Zandvoort) prediction.

Takes whichever model backtest.py picked, trains it on everything up to round 11
(races and sprints), predicts finishing position for round 12, then simulates to
get win and podium probabilities.

Timestamped and seeded so it can be regenerated exactly. Commit the output
before the race - a prediction you can still edit afterwards isn't one.

    uv run python src/predict_dutch.py
"""

from __future__ import annotations

import json
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
from src.models import (  # noqa: E402
    EXCLUDED_FEATURES,
    FEATURE_DEFINITIONS,
    MODEL_FEATURES,
    get_models,
    model_hyperparameters,
    split_xy,
)
from src.simulate import DEFAULT_N_SIMS, DEFAULT_SEED, simulate_race  # noqa: E402

RACE_ROUND = 12
RACE_LABEL = "2026 Dutch Grand Prix"
RACE_SLUG = "2026_R12_dutch"
TRAIN_THROUGH = 11

PREDICTIONS = BASE / "predictions"
REPORTS = BASE / "reports"
SELECTION = REPORTS / "model_selection.json"


def fail(message: str, code: int = 1) -> int:
    print(f"\nERROR: {message}\n", file=sys.stderr)
    return code


def load_selection() -> dict:
    if not SELECTION.exists():
        raise FileNotFoundError(
            f"{SELECTION.relative_to(BASE)} is missing - run `uv run python src/backtest.py` first"
        )
    with SELECTION.open() as fh:
        return json.load(fh)


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


def main() -> int:
    PREDICTIONS.mkdir(parents=True, exist_ok=True)

    table = load_model_table()

    # ---- 1. Is the weekend's data actually present? -----------------------
    race_rows = table[(table["round"] == RACE_ROUND) & (table["session_type"] == "R")].copy()
    if race_rows.empty:
        return fail(
            f"no round {RACE_ROUND} race rows in the model table.\n"
            "  The qualifying session may not have been ingested yet.\n"
            "  Run:  uv run python src/ingest.py --rounds 12\n"
            "  then: uv run python src/features.py"
        )

    if race_rows["quali_pct_off_pole_imputed"].fillna(True).astype(bool).all():
        return fail(
            f"every round {RACE_ROUND} driver has an IMPUTED qualifying time, which means "
            "qualifying has not been ingested.\n"
            "  Run:  uv run python src/ingest.py --rounds 12\n"
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

    # ---- 2. Train on everything up to and including round 11 --------------
    scoreable = table[table["is_trainable"].fillna(False).astype(bool)]
    train = scoreable[scoreable["round"] <= TRAIN_THROUGH]
    X_train, y_train = split_xy(train)

    model = clone(get_models()[model_name])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_train, y_train)

    # ---- 3. Predict round 12 ---------------------------------------------
    X_race = race_rows.loc[:, MODEL_FEATURES]
    race_rows["predicted_position"] = np.asarray(model.predict(X_race), dtype="float64")

    # The grid baseline beat every fitted model in the backtest, so keep its
    # numbers alongside rather than quietly dropping them.
    baseline_model = clone(get_models()["D_grid_baseline"]).fit(X_train, y_train)
    baseline_pred = np.asarray(baseline_model.predict(X_race), dtype="float64")

    # ---- 4. Simulate ------------------------------------------------------
    sims = simulate_race(
        race_rows["predicted_position"].to_numpy(),
        sigma=sigma,
        dnf_probs=race_rows["dnf_rate_shrunk"].to_numpy(),
        n_sims=DEFAULT_N_SIMS,
        seed=DEFAULT_SEED,
    )
    for col in sims.columns:
        race_rows[col] = sims[col].to_numpy()

    baseline_sims = simulate_race(
        baseline_pred,
        sigma=float(selection["metrics"].get("sigma_test") or sigma),
        dnf_probs=race_rows["dnf_rate_shrunk"].to_numpy(),
        n_sims=DEFAULT_N_SIMS,
        seed=DEFAULT_SEED,
    )
    race_rows["baseline_p_win"] = baseline_sims["p_win"].to_numpy()

    ranked = race_rows.sort_values("p_win", ascending=False).reset_index(drop=True)
    colors = team_colors_for(ranked["team_name"])
    ranked["team_color"] = ranked["team_name"].map(colors)

    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    podium = ranked.head(3)

    # ---- 5. Write the locked JSON ----------------------------------------
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

    payload = {
        "race": RACE_LABEL,
        "season": 2026,
        "round": RACE_ROUND,
        "circuit": "Zandvoort",
        "generated_at_utc": generated_at,
        "model": {
            "name": model_name,
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
            "note": "races and sprints; DNS rows and the unraced round 12 excluded",
        },
        "backtest": {
            "test_rounds": selection["test_rounds"],
            "metrics": selection["metrics"],
            "baseline_comparison": selection["baseline_comparison"],
        },
        "simulation": {
            "residual_sigma": round(sigma, 6),
            "sigma_source": "mean residual sd of the selected model over backtest folds 6-11",
            "n_sims": DEFAULT_N_SIMS,
            "seed": DEFAULT_SEED,
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
        "caveats": [
            "grid_position for round 12 is QUALIFYING ORDER; grid penalties are not "
            "reflected. Verify against the published starting grid before relying on this.",
            "The grid baseline (B4) achieved a lower win log loss than this model over "
            "the six backtest races. See reports/model_selection.md.",
            "Six backtest races is a demonstration, not a validation of calibration.",
        ],
        "drivers": drivers,
    }

    json_path = PREDICTIONS / f"{RACE_SLUG}.json"
    with json_path.open("w") as fh:
        json.dump(payload, fh, indent=2)

    # ---- 6. Flat CSV for the report --------------------------------------
    csv_cols = (
        ["driver_id", "abbreviation", "team_name", "team_color", "grid_position",
         "predicted_position", "p_win", "p_podium", "baseline_p_win"] + MODEL_FEATURES
    )
    csv_path = PREDICTIONS / f"{RACE_SLUG}.csv"
    ranked.loc[:, csv_cols].to_csv(csv_path, index=False, float_format="%.6f")

    # ---- 7. Human-readable summary ---------------------------------------
    print("=" * 78)
    print(f"{RACE_LABEL} — locked prediction")
    print("=" * 78)
    print(f"model            {model_name}")
    print(f"trained on       rounds {min(payload['training']['rounds'])}-"
          f"{max(payload['training']['rounds'])}, {payload['training']['n_rows']} rows")
    print(f"residual sigma   {sigma:.3f} positions")
    print(f"simulations      {DEFAULT_N_SIMS:,} (seed {DEFAULT_SEED})")
    print(f"generated        {generated_at}")

    print(f"\n{'#':<3} {'DRIVER':<5} {'TEAM':<17} {'GRID':>4} {'PRED':>6} "
          f"{'P(win)':>8} {'P(podium)':>10}")
    print("-" * 78)
    for i, row in ranked.head(8).iterrows():
        print(f"{i + 1:<3} {row['abbreviation']:<5} {row['team_name']:<17} "
              f"{row['grid_position']:>4.0f} {row['predicted_position']:>6.2f} "
              f"{row['p_win'] * 100:>7.1f}% {row['p_podium'] * 100:>9.1f}%")

    winner = payload["predicted_winner"]
    print(f"\npredicted winner : {winner['abbreviation']} ({winner['team_name']}) "
          f"at {winner['p_win'] * 100:.1f}%")
    print("predicted podium : " + ", ".join(p["abbreviation"] for p in payload["predicted_podium"]))

    print(f"\nwrote {json_path.relative_to(BASE)}")
    print(f"wrote {csv_path.relative_to(BASE)}")

    print("\n" + "!" * 78)
    print("COMMIT THIS PREDICTION NOW, BEFORE THE RACE STARTS:")
    print(f"    git add {json_path.relative_to(BASE)} {csv_path.relative_to(BASE)}")
    print(f'    git commit -m "Lock {RACE_LABEL} prediction ({generated_at})"')
    print("A prediction that can still be edited after the race is not a prediction.")
    print("!" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
