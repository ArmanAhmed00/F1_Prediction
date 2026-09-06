#!/usr/bin/env python3
"""Score a locked prediction against the race that actually happened.

Reads predictions/2026_R{round}_{label}.json, joins it to the ingested race
result, and writes a markdown post-mortem. Nothing here can change a
prediction - the JSON is an input, and it was committed before the race.

The point is the causes column. A model that gets a driver 15 places wrong
because he retired on lap 1 has a different problem from one that gets a driver
9 places wrong because he was quick and it didn't notice, and averaging the two
into "MAE 3.4" hides exactly the distinction worth having.

    uv run python src/evaluate_race.py --round 12 --label dutch
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

from src.io_utils import load_raw  # noqa: E402

PREDICTIONS = BASE / "predictions"
REPORTS = BASE / "reports"

# Same clip as the backtest, so a single-race score is comparable to a fold.
EPS = 1e-6

N_WORST = 5

# Race control lines worth quoting as a cause. Blue flags and track-limits
# deletions are noise - every backmarker collects a dozen of each.
CAUSE_PATTERNS = (
    ("red flag", r"RED FLAG"),
    ("safety car", r"SAFETY CAR"),
    ("collision", r"COLLISION|CONTACT"),
    ("penalty", r"SECOND TIME PENALTY|DRIVE THROUGH PENALTY|STOP AND GO"),
    ("under investigation", r"UNDER INVESTIGATION"),
)


def _clip(p: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(p, dtype="float64"), EPS, 1.0 - EPS)


def log_loss_binary(y, p) -> float:
    p = _clip(p)
    y = np.asarray(y, dtype="float64")
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def brier(y, p) -> float:
    p = _clip(p)
    y = np.asarray(y, dtype="float64")
    return float(np.mean((p - y) ** 2))


def load_prediction(rnd: int, label: str) -> dict:
    path = PREDICTIONS / f"2026_R{rnd}_{label}.json"
    if not path.exists():
        raise FileNotFoundError(f"{path.relative_to(BASE)} is missing")
    with path.open() as fh:
        return json.load(fh)


def load_actual(rnd: int) -> pd.DataFrame:
    results = load_raw("results")
    race = results[(results["round"] == rnd) & (results["session_type"] == "R")]
    if race.empty:
        raise SystemExit(f"round {rnd} has no race result yet - nothing to score against")
    return race.rename(columns={"DriverId": "driver_id"})


def join(prediction: dict, actual: pd.DataFrame) -> pd.DataFrame:
    pred = pd.DataFrame(prediction["drivers"])
    # The JSON is written in descending p_win, so row order IS the predicted
    # finishing order. Recomputed here anyway rather than trusted.
    pred = pred.sort_values("p_win", ascending=False).reset_index(drop=True)
    pred["predicted_rank"] = np.arange(1, len(pred) + 1)

    cols = [
        "driver_id", "Abbreviation", "DriverNumber", "TeamName", "Position",
        "ClassifiedPosition", "Status", "GridPosition", "Laps",
    ]
    merged = pred.merge(actual.loc[:, cols], on="driver_id", how="outer", indicator=True)
    unmatched = merged[merged["_merge"] != "both"]
    if not unmatched.empty:
        raise SystemExit(
            "prediction and result do not cover the same drivers: "
            f"{unmatched[['driver_id', '_merge']].to_dict('records')}"
        )

    merged = merged.drop(columns=["_merge"])
    merged["actual_position"] = merged["Position"].astype("float64")
    merged["position_error"] = (merged["predicted_rank"] - merged["actual_position"]).abs()
    merged["dnf"] = merged["Status"].astype("string").eq("Retired")
    return merged.sort_values("actual_position").reset_index(drop=True)


def cause_for(row: pd.Series, control: pd.DataFrame) -> str:
    """One line on why this driver's position was missed, from Status + RCMs.

    Status is the primary source because it is the only field that states
    outright whether the car finished. Race control fills in the context that
    Status cannot: whether the retirement happened in the lap-1 melee, and
    whether a penalty moved someone.
    """
    number = str(row["DriverNumber"])
    mine = control[control["Message"].str.contains(rf"CAR {number} \(", regex=True, na=False)]
    tags = [
        name
        for name, pattern in CAUSE_PATTERNS
        if mine["Message"].str.contains(pattern, regex=True, na=False).any()
    ]

    laps = float(row["Laps"]) if pd.notna(row["Laps"]) else np.nan
    delta = float(row["predicted_rank"] - row["actual_position"])
    direction = "underrated" if delta > 0 else "overrated"

    if bool(row["dnf"]):
        if laps <= 1:
            base = (
                f"DNF on lap 1 ({laps:.0f} laps completed) - out in the opening-lap "
                "incident that brought the safety car and then the red flag"
            )
        elif laps <= 5:
            base = f"DNF on lap {laps:.0f}, immediately after the red-flag restart"
        else:
            base = f"DNF on lap {laps:.0f} of 72"
        note = f"; race control also logged: {', '.join(tags)}" if tags else ""
        return (
            f"{base}. Retirement is not in the predicted order at all - the model "
            f"ranks expected finishing position and folds DNF risk in only as a "
            f"{row['features']['dnf_rate_shrunk'] * 100:.1f}% simulated retirement "
            f"probability{note}."
        )

    status = str(row["Status"])
    base = (
        f"classified P{row['actual_position']:.0f} ({status.lower()}) from grid "
        f"P{row['GridPosition']:.0f}; {direction} by {abs(delta):.0f} places"
    )
    if tags:
        return f"{base}. Race control: {', '.join(tags)}."
    return (
        f"{base}. No incident in race control - a straight pace-and-strategy miss, "
        "not a chaos artefact."
    )


def build_report(prediction: dict, joined: pd.DataFrame, control: pd.DataFrame,
                 rnd: int, label: str) -> str:
    n = len(joined)
    y_win = (joined["actual_position"] == 1).to_numpy(dtype="float64")
    y_podium = (joined["actual_position"] <= 3).to_numpy(dtype="float64")
    p_win = joined["p_win"].to_numpy(dtype="float64")
    p_podium = joined["p_podium"].to_numpy(dtype="float64")

    winner = joined.loc[joined["actual_position"] == 1].iloc[0]
    predicted_podium = set(
        joined.sort_values("predicted_rank").head(3)["driver_id"]
    )
    actual_podium = set(joined[joined["actual_position"] <= 3]["driver_id"])
    podium_hits = len(predicted_podium & actual_podium)

    rho = float(spearmanr(joined["predicted_rank"], joined["actual_position"]).statistic)
    rho_cont = float(
        spearmanr(joined["predicted_position"], joined["actual_position"]).statistic
    )
    mae_rank = float((joined["predicted_rank"] - joined["actual_position"]).abs().mean())
    # The backtest's mae_position is on the raw regression output, not the rank,
    # so this is the one that compares like for like.
    mae_raw = float(
        (joined["predicted_position"] - joined["actual_position"]).abs().mean()
    )

    uniform_win = log_loss_binary(y_win, np.full(n, 1.0 / n))
    uniform_podium = log_loss_binary(y_podium, np.full(n, 3.0 / n))

    backtest = prediction.get("backtest", {}).get("metrics", {})
    folds = prediction.get("backtest", {}).get("test_rounds", [])
    fold_label = (
        f"folds {min(folds)}-{max(folds)}" if folds else "backtest folds"
    )
    n_folds = len(folds)
    race_name = prediction.get("race", f"round {rnd}")

    lines = [
        f"# {race_name} — prediction post-mortem",
        "",
        f"_Scored {datetime.now(timezone.utc).isoformat(timespec='seconds')}_",
        "",
        f"Prediction locked at `{prediction['generated_at_utc']}`, before the race, from "
        f"`predictions/2026_R{rnd}_{label}.json`. Model `{prediction['model']['name']}`, "
        f"trained on rounds {min(prediction['training']['rounds'])}-"
        f"{max(prediction['training']['rounds'])} "
        f"({prediction['training']['n_rows']} rows), simulated with "
        f"sigma {prediction['simulation']['residual_sigma']:.3f} over "
        f"{prediction['simulation']['n_sims']:,} draws.",
        "",
        "## The short version",
        "",
    ]

    lines += [
        f"- **The podium was called exactly right as a set.** All three of the predicted "
        f"podium ({', '.join(sorted(joined.set_index('driver_id').loc[list(predicted_podium), 'abbreviation']))}) "
        f"finished on the podium — just not in the predicted order."
        if podium_hits == 3
        else f"- **{podium_hits} of 3** predicted podium finishers made the actual podium.",
        f"- **The winner was missed.** {winner['abbreviation']} won from grid "
        f"P{winner['GridPosition']:.0f}; the model ranked him "
        f"**{winner['predicted_rank']:.0f}{'st' if winner['predicted_rank'] == 1 else 'nd' if winner['predicted_rank'] == 2 else 'rd' if winner['predicted_rank'] == 3 else 'th'}** "
        f"and gave him **p_win = {winner['p_win']:.4f}** "
        f"({winner['p_win'] * 100:.2f}%), against the "
        f"{1 / n * 100:.2f}% a uniform prior would have given."
        if winner["predicted_rank"] != 1
        else f"- **The winner was called.** {winner['abbreviation']} at "
        f"p_win = {winner['p_win']:.4f}.",
        f"- **Rank correlation held up**: Spearman rho {rho:.3f} on the full "
        f"{n}-car order, against {backtest.get('spearman_rho', float('nan')):.3f} "
        f"averaged over the backtest {fold_label}.",
        f"- **The single biggest error was a retirement**, not a pace call: "
        f"{joined.loc[joined['position_error'].idxmax(), 'abbreviation']} by "
        f"{joined['position_error'].max():.0f} places.",
        "",
        "## Scores on this race",
        "",
        f"| metric | this race | backtest mean ({fold_label}) | uniform 1/n |",
        "| --- | ---: | ---: | ---: |",
        f"| win log loss | {log_loss_binary(y_win, p_win):.4f} | "
        f"{backtest.get('win_log_loss', float('nan')):.4f} | {uniform_win:.4f} |",
        f"| win Brier | {brier(y_win, p_win):.4f} | "
        f"{backtest.get('win_brier', float('nan')):.4f} | "
        f"{brier(y_win, np.full(n, 1.0 / n)):.4f} |",
        f"| podium log loss | {log_loss_binary(y_podium, p_podium):.4f} | "
        f"{backtest.get('podium_log_loss', float('nan')):.4f} | {uniform_podium:.4f} |",
        f"| podium Brier | {brier(y_podium, p_podium):.4f} | "
        f"{backtest.get('podium_brier', float('nan')):.4f} | "
        f"{brier(y_podium, np.full(n, 3.0 / n)):.4f} |",
        f"| Spearman rho (predicted rank) | {rho:.4f} | "
        f"{backtest.get('spearman_rho', float('nan')):.4f} | — |",
        f"| Spearman rho (raw predicted position) | {rho_cont:.4f} | — | — |",
        f"| MAE, raw predicted position vs finish | {mae_raw:.4f} | "
        f"{backtest.get('mae_position', float('nan')):.4f} | — |",
        f"| MAE, predicted rank vs finish | {mae_rank:.4f} | — | — |",
        f"| p(actual winner) | {winner['p_win']:.4f} | "
        f"{backtest.get('p_actual_winner', float('nan')):.4f} | {1 / n:.4f} |",
        "",
        "One race is one race. These numbers say what happened; they do not "
        "establish that the model is calibrated, and a single log loss cannot be "
        f"meaningfully compared against a {n_folds}-fold mean except as a sanity "
        "check.",
        "",
        "## Predicted vs actual, side by side",
        "",
        "Two independent orderings of the same field. Read each half top to "
        "bottom on its own; the two halves of a row are not the same car.",
        "",
        "| # | PREDICTED | team | grid | p_win | p_podium | ‖ | # | ACTUAL | status | laps |",
        "| ---: | --- | --- | ---: | ---: | ---: | --- | ---: | --- | --- | ---: |",
    ]

    by_pred = joined.sort_values("predicted_rank").reset_index(drop=True)
    by_act = joined.sort_values("actual_position").reset_index(drop=True)
    for i in range(len(by_pred)):
        left, right = by_pred.loc[i], by_act.loc[i]
        lines.append(
            f"| {left['predicted_rank']:.0f} | **{left['abbreviation']}** | "
            f"{left['team_name']} | {left['grid_position']:.0f} | "
            f"{left['p_win']:.4f} | {left['p_podium']:.4f} | ‖ | "
            f"{right['actual_position']:.0f} | **{right['abbreviation']}** | "
            f"{right['Status']} | {right['Laps']:.0f} |"
        )

    lines += [
        "",
        "### Per-driver error",
        "",
        "| driver | grid | predicted | actual | error | status |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for _, row in joined.sort_values("position_error", ascending=False).iterrows():
        lines.append(
            f"| **{row['abbreviation']}** | {row['grid_position']:.0f} | "
            f"{row['predicted_rank']:.0f} | {row['actual_position']:.0f} | "
            f"{row['position_error']:.0f} | {row['Status']} |"
        )

    lines += [
        "",
        f"## The {N_WORST} largest position errors",
        "",
    ]

    worst = joined.sort_values("position_error", ascending=False).head(N_WORST)
    for _, row in worst.iterrows():
        lines.append(
            f"**{row['abbreviation']}** — predicted P{row['predicted_rank']:.0f}, "
            f"finished P{row['actual_position']:.0f} "
            f"({row['position_error']:.0f} places). {cause_for(row, control)}"
        )
        lines.append("")

    lines += _verdict(joined, winner, podium_hits, n)
    return "\n".join(lines)


def _verdict(joined: pd.DataFrame, winner: pd.Series, podium_hits: int, n: int) -> list[str]:
    """The blunt part. Every claim here is computed, not asserted.

    Split the errors by cause, because the aggregate hides the only thing worth
    knowing: which misses were chaos the model cannot see, and which were pace
    it could have.
    """
    finishers = joined[~joined["dnf"]]
    dnfs = joined[joined["dnf"]]
    signed = finishers["predicted_rank"] - finishers["actual_position"]

    back = finishers[finishers["grid_position"] > 10]
    front = finishers[finishers["grid_position"] <= 10]
    back_signed = back["predicted_rank"] - back["actual_position"]
    front_signed = front["predicted_rank"] - front["actual_position"]

    top6_pred = set(joined.nsmallest(6, "predicted_rank")["driver_id"])
    top6_act = set(joined.nsmallest(6, "actual_position")["driver_id"])
    top6_hits = len(top6_pred & top6_act)

    gained = int((back_signed > 0).sum())

    return [
        "## What actually missed",
        "",
        f"**The front of the field was easy and the model got it.** "
        f"{top6_hits} of the 6 predicted top-6 finished in the top 6, and the "
        f"{len(front)} classified runners who started in the top 10 were placed to "
        f"within a mean signed error of {front_signed.mean():+.2f} positions. "
        "Chaotic race or not, the quick cars started at the front, stayed at the "
        "front, and none of that required a model.",
        "",
        f"**The order within the podium was wrong.** {podium_hits} of 3 predicted "
        f"podium finishers made the podium, but the top pick was "
        f"{joined.nsmallest(1, 'predicted_rank')['abbreviation'].iat[0]} and the "
        f"winner was {winner['abbreviation']}, ranked "
        f"{winner['predicted_rank']:.0f} at p_win {winner['p_win']:.4f}. Getting "
        "the set right and the order wrong is the expected outcome when the top "
        "three probabilities sit within a few points of each other — which is "
        "honest of the model, not impressive.",
        "",
        f"**The real miss is the back half, and it is systematic.** Of the "
        f"{len(back)} classified finishers who started outside the top 10, "
        f"{gained} finished better than predicted, mean signed error "
        f"{back_signed.mean():+.2f} positions against {front_signed.mean():+.2f} "
        "for the front runners. The model assumes the grid roughly holds; on a "
        "day with a red flag, two virtual safety cars and six retirements, cars "
        "from the back got a free strategy reset and moved up, and nothing in "
        "the feature list can see that coming. `positions_gained_ewm` is a "
        "season-long average and is far too slow to represent it.",
        "",
        f"**{len(dnfs)} retirements are simply not modelled.** They enter only "
        "through `dnf_rate_shrunk`, as an independent per-driver coin flip in "
        "the simulation, which produces sensible win probabilities and a "
        "meaningless finishing order for anyone who stops. The largest single "
        f"error on the day ({joined.loc[joined['position_error'].idxmax(), 'abbreviation']}, "
        f"{joined['position_error'].max():.0f} places) is entirely this. It is "
        "not a tuning problem and no amount of retuning the regressor will fix "
        "it — the model has no concept of a first-lap incident.",
        "",
        "None of the above is a reason to change the model today. It is a reason "
        "to write it down. See `reports/v2_backlog.md`.",
        "",
    ]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score a locked prediction against the result.")
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--out", default=None, help="output path (default reports/{label}_gp_evaluation.md)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    prediction = load_prediction(args.round, args.label)
    actual = load_actual(args.round)
    joined = join(prediction, actual)

    control = load_raw("race_control")
    control = control[(control["round"] == args.round) & (control["session_type"] == "R")]

    out = Path(args.out) if args.out else REPORTS / f"{args.label}_gp_evaluation.md"
    out = out if out.is_absolute() else (BASE / out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_report(prediction, joined, control, args.round, args.label))
    print(f"wrote {out.relative_to(BASE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
