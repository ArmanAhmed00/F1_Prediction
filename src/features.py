#!/usr/bin/env python3
"""Build the model table from the raw FastF1 dumps.

One row per (round, session_type, driver_id) for every classified Race and
Sprint, plus prediction rows for a Race that hasn't run yet. No scaling or
encoding here - that belongs in the model pipeline.

The rule everything follows: a feature must be computable from what was known
before that session started. Two traps worth knowing about:

  - Sprint weekends run FP1 -> SQ -> SPRINT -> Q -> Race. Qualifying is AFTER
    the sprint, so a Sprint row reading Q data is reading the future. Sprint
    rows use SQ, Race rows use Q.
  - A Sprint row can't use its own result, so sprint_* is null there and gets
    imputed like any other gap.

Rolling features are all EWM then shift(1) over the ordered session history, so
a row never sees itself. tests/test_no_leakage.py checks this by recomputing.

    uv run python src/features.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
PROCESSED = BASE / "data" / "processed"
MANUAL = BASE / "data" / "manual"
OUT_PATH = PROCESSED / "model_table.csv"

sys.path.insert(0, str(BASE))
from src.io_utils import load_raw  # noqa: E402

SEASON = 2026

# Running order within a weekend. FP2/FP3 never appear alongside SQ/S, so one
# map covers both formats:
#   conventional      FP1 FP2 FP3 Q R
#   sprint_qualifying FP1 SQ  S   Q R
SESSION_ORDER = {"FP1": 1, "FP2": 2, "FP3": 3, "SQ": 2, "S": 3, "Q": 4, "R": 5}
PRACTICE = ("FP1", "FP2", "FP3")
COMPETITIVE = ("S", "R")
# Which qualifying session sets the grid for which competitive session.
QUALI_SOURCE = {"R": "Q", "S": "SQ"}

# 3-round halflife, expressed in time rather than row count - otherwise a
# sprint weekend's two sessions would decay like two rounds instead of one.
HOURS_PER_ROUND = 24
HALFLIFE_ROUNDS = 3
# Explicit numpy unit; the pd.Timedelta keyword forms throw a DeprecationWarning
# on this pandas/numpy combination.
HALFLIFE = pd.Timedelta(np.timedelta64(HALFLIFE_ROUNDS * HOURS_PER_ROUND, "h"))
EPOCH = pd.Timestamp("2026-01-01")

QUICKLAP_THRESHOLD = 1.07  # mirrors fastf1's pick_quicklaps default
MIN_LONGRUN_LAPS = 4  # a "long run" of 2 laps tells you nothing
MIN_COMPOUND_DRIVERS = 3  # below this a compound offset is noise

# Beta-Binomial prior for DNF rate. alpha=1, beta=9 is weak (10 pseudo-races),
# recentred on the field rate so we shrink toward reality rather than 0.1.
DNF_PRIOR_ALPHA = 1.0
DNF_PRIOR_BETA = 9.0
DNF_PRIOR_STRENGTH = DNF_PRIOR_ALPHA + DNF_PRIOR_BETA

# Classification letters that mean "did not finish".
DNF_CODES = {"R", "D", "E", "N", "F"}
DNS_CODES = {"W"}

FEATURE_COLUMNS = [
    "quali_pct_off_pole",
    "grid_position",
    "quali_teammate_delta_pct",
    "fp_best_pct_off_fastest",
    "fp_longrun_pace_pct",
    "sprint_finish_position",
    "sprint_positions_gained",
    "has_sprint_data",
    "form_quali_pace_ewm",
    "form_finish_pos_ewm",
    "form_race_pace_ewm",
    "team_form_pace_ewm",
    "positions_gained_ewm",
    "dnf_rate_shrunk",
    "championship_position",
    "track_temp_mean",
    "was_wet",
]

KEY_COLUMNS = [
    "season",
    "round",
    "event_name",
    "session_type",
    "is_sprint",
    "is_sprint_weekend",
    "driver_id",
    "team_id",
    "abbreviation",
    "team_name",
]

TARGET_COLUMNS = [
    "finish_position",
    "positions_gained",
    "won",
    "podium",
    "dnf",
    "dnf_first_lap",
    "dnf_after_lap1",
    "dns",
    "is_trainable",
    "is_prediction",
]


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load_raw_frames() -> dict[str, pd.DataFrame]:
    """Every raw table, via load_raw so dtypes are the ones ingest recorded."""
    return {name: load_raw(name) for name in ("results", "laps", "weather", "schedule")}


def load_model_table() -> pd.DataFrame:
    """Read back the built model table, so consumers never hand-roll the read."""
    if not OUT_PATH.exists():
        raise FileNotFoundError(f"{OUT_PATH} is missing - run `uv run python src/features.py`")
    table = pd.read_csv(OUT_PATH)
    table["round"] = table["round"].astype("int64")
    for col in TARGET_COLUMNS + [c for c in table.columns if c.endswith("_imputed")]:
        if col in table.columns and table[col].dropna().isin([True, False]).all():
            table[col] = table[col].astype("boolean")
    return table


def _session_time(rounds: pd.Series, session_types: pd.Series) -> pd.Series:
    """A sortable timestamp per session, used to order history and decay EWMs."""
    order = session_types.map(SESSION_ORDER).astype("float64")
    hours = rounds.astype("float64") * HOURS_PER_ROUND + order
    return EPOCH + pd.to_timedelta(hours, unit="h")


# --------------------------------------------------------------------------
# Lap filters. We're working off flattened CSVs, so these reimplement what
# fastf1's pick_accurate / pick_wo_box / pick_track_status / pick_quicklaps do.
# --------------------------------------------------------------------------
def _clean_laps(laps: pd.DataFrame, session_types: tuple[str, ...]) -> pd.DataFrame:
    out = laps[laps["session_type"].isin(session_types)].copy()
    out = out[out["DriverId"].notna() & out["LapTime_s"].notna()]
    out = out[out["IsAccurate"].fillna(False).astype(bool)]  # pick_accurate()
    out = out[~out["Deleted"].fillna(False).astype(bool)]
    # pick_wo_box(): in-laps and out-laps have meaningless times.
    out = out[out["PitInTime_s"].isna() & out["PitOutTime_s"].isna()]
    # pick_track_status("1"): green flag only, no SC/VSC/yellow.
    out = out[out["TrackStatus"].astype("string") == "1"]
    return out


def _pick_quicklaps(laps: pd.DataFrame, within: list[str]) -> pd.DataFrame:
    """pick_quicklaps(): drop anything outside 107% of the fastest lap in scope.

    The scope matters more than the threshold. Against the whole session's
    fastest lap the 107% line lands on the median practice lap, because the
    session best is someone's low-fuel quali sim and race-fuel running is 5-8%
    slower. Applied session-wide it deletes most of the long runs it's meant to
    be cleaning - one round was left with a single usable driver. Per driver
    (what pick_driver().pick_quicklaps() gives you) it does the intended job:
    drops that driver's traffic, mistakes and cool-down laps.
    """
    if laps.empty:
        return laps
    fastest = laps.groupby(within)["LapTime_s"].transform("min")
    return laps[laps["LapTime_s"] < fastest * QUICKLAP_THRESHOLD]


# --------------------------------------------------------------------------
# Feature blocks
# --------------------------------------------------------------------------
def quali_block(results: pd.DataFrame) -> pd.DataFrame:
    """Pace off pole and teammate delta, per (round, quali session, driver).

    Percentage rather than seconds, so 0.8% means the same at Monaco as at Monza.
    """
    q = results[results["session_type"].isin(["Q", "SQ"])].copy()
    q["best_quali_s"] = q[["Q1_s", "Q2_s", "Q3_s"]].min(axis=1, skipna=True)

    by_session = ["round", "session_type"]
    pole = q.groupby(by_session)["best_quali_s"].transform("min")
    q["quali_pct_off_pole"] = (q["best_quali_s"] - pole) / pole * 100.0

    # Official classification where present, else rank on the best lap.
    ranked = q.groupby(by_session)["best_quali_s"].rank(method="min")
    q["quali_position"] = q["Position"].fillna(ranked)

    # Holding the car constant is the cleanest driver signal there is. Written
    # as leave-one-out so it still works if a team ever runs more than two cars.
    grp = q.groupby(by_session + ["TeamId"])["quali_pct_off_pole"]
    team_total = grp.transform("sum")
    team_count = grp.transform("count")
    teammate = (team_total - q["quali_pct_off_pole"]) / (team_count - 1).replace(0, np.nan)
    q["quali_teammate_delta_pct"] = q["quali_pct_off_pole"] - teammate

    return q[
        [
            "round",
            "session_type",
            "DriverId",
            "quali_pct_off_pole",
            "quali_position",
            "quali_teammate_delta_pct",
        ]
    ].rename(columns={"session_type": "quali_session", "DriverId": "driver_id"})


def practice_block(laps: pd.DataFrame) -> pd.DataFrame:
    """Single-lap and long-run practice pace, per (round, driver).

    Every practice session precedes every competitive session in both 2026
    formats, so these are keyed on the round alone - no ordering filter needed.
    """
    clean = _clean_laps(laps, PRACTICE)
    if clean.empty:
        return pd.DataFrame(
            columns=["round", "driver_id", "fp_best_pct_off_fastest", "fp_longrun_pace_pct"]
        )

    # ---- 1. Best clean lap in the weekend's practice, % off the field best.
    best = clean.groupby(["round", "DriverId"])["LapTime_s"].min().rename("fp_best_s").reset_index()
    fastest = best.groupby("round")["fp_best_s"].transform("min")
    best["fp_best_pct_off_fastest"] = (best["fp_best_s"] - fastest) / fastest * 100.0

    # ---- 2. Median lap of each driver's longest clean green stint = race pace.
    runs = _pick_quicklaps(clean, ["round", "session_type", "DriverId"])
    stints = (
        runs.groupby(["round", "session_type", "DriverId", "Stint"])
        .agg(
            n_laps=("LapTime_s", "size"),
            stint_median_s=("LapTime_s", "median"),
            compound=("Compound", lambda s: s.mode().iat[0] if not s.mode().empty else pd.NA),
        )
        .reset_index()
    )
    stints = stints[stints["n_laps"] >= MIN_LONGRUN_LAPS]

    if stints.empty:
        longrun = pd.DataFrame(columns=["round", "DriverId", "fp_longrun_pace_pct"])
    else:
        # Longest run wins; ties broken by the quicker median.
        stints = stints.sort_values(
            ["round", "DriverId", "n_laps", "stint_median_s"], ascending=[True, True, False, True]
        )
        longrun = stints.drop_duplicates(["round", "DriverId"]).copy()

        # Shift each stint by how far its compound sits off the round median.
        # First-order only. Compound mix is a real confound: which tyre you ran
        # is a team decision tangled up with fuel load and run plan, and a driver
        # who only did hards gets compared through an offset estimated off other
        # cars.
        overall = longrun.groupby("round")["stint_median_s"].transform("median")
        by_compound = longrun.groupby(["round", "compound"])["stint_median_s"]
        compound_median = by_compound.transform("median")
        compound_n = by_compound.transform("count")
        offset = (compound_median - overall).where(compound_n >= MIN_COMPOUND_DRIVERS, 0.0)
        longrun["adj_s"] = longrun["stint_median_s"] - offset

        reference = longrun.groupby("round")["adj_s"].transform("min")
        longrun["fp_longrun_pace_pct"] = (longrun["adj_s"] - reference) / reference * 100.0
        longrun = longrun[["round", "DriverId", "fp_longrun_pace_pct"]]

    out = best[["round", "DriverId", "fp_best_pct_off_fastest"]].merge(
        longrun, on=["round", "DriverId"], how="outer"
    )
    return out.rename(columns={"DriverId": "driver_id"})


def race_pace_block(laps: pd.DataFrame) -> pd.DataFrame:
    """Median green-flag non-pit lap per competitive session, % off session best.

    This is measured DURING the session, so it is only ever consumed shifted,
    as form_race_pace_ewm.  It is never a same-session feature.
    """
    clean = _clean_laps(laps, COMPETITIVE)
    if clean.empty:
        return pd.DataFrame(columns=["round", "session_type", "driver_id", "race_pace_pct"])
    med = (
        clean.groupby(["round", "session_type", "DriverId"])["LapTime_s"]
        .median()
        .rename("median_s")
        .reset_index()
    )
    best = med.groupby(["round", "session_type"])["median_s"].transform("min")
    med["race_pace_pct"] = (med["median_s"] - best) / best * 100.0
    return med[["round", "session_type", "DriverId", "race_pace_pct"]].rename(
        columns={"DriverId": "driver_id"}
    )


def weather_block(weather: pd.DataFrame) -> pd.DataFrame:
    """Track temp and a wet flag from the weekend sessions that already ran.

    The target session's own weather would be leakage, and doesn't exist yet for
    an unraced round anyway. Only earlier sessions in the same weekend.
    """
    w = weather.copy()
    w["order"] = w["session_type"].map(SESSION_ORDER)
    w = w[w["order"].notna()]

    rows = []
    for session_type, order in SESSION_ORDER.items():
        if session_type not in COMPETITIVE:
            continue
        prior = w[w["order"] < order]
        if prior.empty:
            continue
        agg = prior.groupby("round").agg(
            track_temp_mean=("TrackTemp", "mean"),
            was_wet=("Rainfall", lambda s: bool(s.fillna(False).astype(bool).any())),
        )
        agg = agg.reset_index()
        agg["session_type"] = session_type
        rows.append(agg)
    if not rows:
        return pd.DataFrame(columns=["round", "session_type", "track_temp_mean", "was_wet"])
    return pd.concat(rows, ignore_index=True)


# --------------------------------------------------------------------------
# Base frame and targets
# --------------------------------------------------------------------------
def base_frame(results: pd.DataFrame) -> pd.DataFrame:
    """Classified Races and Sprints, plus prediction rows for the next race.

    A round with published Qualifying but no Race gives us the entry list for
    something we want to predict but can't score yet.
    """
    comp = results[results["session_type"].isin(COMPETITIVE)].copy()
    comp["is_prediction"] = False

    have_race = set(comp.loc[comp["session_type"] == "R", "round"].unique())
    have_quali = set(results.loc[results["session_type"] == "Q", "round"].unique())
    predict_rounds = sorted(have_quali - have_race)

    if predict_rounds:
        entry = results[
            (results["session_type"] == "Q") & (results["round"].isin(predict_rounds))
        ].copy()
        entry["session_type"] = "R"
        entry["is_prediction"] = True
        # Nothing about the race itself is known yet.
        for col in ("Position", "ClassifiedPosition", "Status", "Points", "Laps", "GridPosition"):
            if col in entry.columns:
                entry[col] = np.nan if col != "ClassifiedPosition" and col != "Status" else pd.NA
        comp = pd.concat([comp, entry], ignore_index=True)

    out = pd.DataFrame(
        {
            "season": comp["season"],
            "round": comp["round"].astype("int64"),
            "event_name": comp["event_name"],
            "session_type": comp["session_type"],
            "driver_id": comp["DriverId"],
            "team_id": comp["TeamId"],
            "abbreviation": comp["Abbreviation"],
            "team_name": comp["TeamName"],
            "is_prediction": comp["is_prediction"].astype(bool),
            "grid_position": comp["GridPosition"].astype("float64"),
            "raw_position": comp["Position"].astype("float64"),
            "classified_position": comp["ClassifiedPosition"].astype("string"),
            "status": comp["Status"].astype("string"),
            "points": comp["Points"].astype("float64"),
            "laps_completed": comp["Laps"].astype("float64"),
        }
    )
    out["is_sprint"] = out["session_type"] == "S"
    sprint_rounds = set(out.loc[out["is_sprint"], "round"].unique())
    out["is_sprint_weekend"] = out["round"].isin(sprint_rounds)
    out["session_time"] = _session_time(out["round"], out["session_type"])
    return out.sort_values(["session_time", "driver_id"]).reset_index(drop=True)


def add_targets(table: pd.DataFrame) -> pd.DataFrame:
    """finish_position / won / podium / dnf, and the DNS carve-out.

    A DNS is neither a DNF nor a finish - the car never started, so it says
    nothing about reliability or pace. Those rows keep their features but get
    is_trainable=False. Kept rather than dropped so the exclusion is visible.

    On causes: the timing feed only gives Finished / Lapped / Retired / Did not
    start / Disqualified. No cause codes, so a mechanical-vs-incident split
    isn't derivable from Status. What we can do is split on distance covered.
    dnf_after_lap1 is a mixed reliability-and-incident bucket, not a clean
    mechanical flag - don't read it as one.
    """
    classified = table["classified_position"]
    status = table["status"]

    dns = status.eq("Did not start").fillna(False) | classified.isin(DNS_CODES).fillna(False)
    retired = status.eq("Retired").fillna(False) | classified.isin(DNF_CODES).fillna(False)
    dnf = retired & ~dns

    table["dns"] = dns & ~table["is_prediction"]
    table["dnf"] = dnf & ~table["is_prediction"]
    table["dnf_first_lap"] = table["dnf"] & table["laps_completed"].le(1).fillna(False)
    table["dnf_after_lap1"] = table["dnf"] & ~table["dnf_first_lap"]

    finish = table["raw_position"].where(~table["dns"])
    table["finish_position"] = finish.where(~table["is_prediction"])
    table["won"] = table["finish_position"].eq(1.0).where(table["finish_position"].notna())
    table["podium"] = table["finish_position"].le(3.0).where(table["finish_position"].notna())
    table["is_trainable"] = ~table["is_prediction"] & ~table["dns"] & table["finish_position"].notna()
    return table


# --------------------------------------------------------------------------
# Rolling form. All of these are EWM then shift(1) over the ordered session
# history, so a row can never see its own session.
# --------------------------------------------------------------------------
def _ewm_shifted(frame: pd.DataFrame, value_col: str, keys: list[str]) -> pd.Series:
    out = pd.Series(np.nan, index=frame.index, dtype="float64")
    for _, sub in frame.groupby(keys, sort=False, dropna=False):
        sub = sub.sort_values("session_time")
        ewm = (
            sub[value_col]
            .astype("float64")
            .ewm(halflife=HALFLIFE, times=sub["session_time"])
            .mean()
            .shift(1)
        )
        out.loc[sub.index] = ewm
    return out


def add_form_features(table: pd.DataFrame) -> pd.DataFrame:
    table = table.sort_values(["session_time", "driver_id"]).reset_index(drop=True)

    table["positions_gained"] = table["grid_position"] - table["finish_position"]

    table["form_quali_pace_ewm"] = _ewm_shifted(table, "quali_pct_off_pole", ["driver_id"])
    table["form_finish_pos_ewm"] = _ewm_shifted(table, "finish_position", ["driver_id"])
    table["form_race_pace_ewm"] = _ewm_shifted(table, "race_pace_pct", ["driver_id"])
    table["positions_gained_ewm"] = _ewm_shifted(table, "positions_gained", ["driver_id"])

    # Two cars per session makes team pace a steadier read on the car than
    # either driver alone. Aggregate first, or each session enters the EWM twice.
    team_hist = (
        table.groupby(["session_time", "team_id"], as_index=False)["quali_pct_off_pole"]
        .mean()
        .sort_values("session_time")
    )
    team_hist["team_form_pace_ewm"] = _ewm_shifted(team_hist, "quali_pct_off_pole", ["team_id"])
    table = table.merge(
        team_hist[["session_time", "team_id", "team_form_pace_ewm"]],
        on=["session_time", "team_id"],
        how="left",
    )
    return table


def add_reliability(table: pd.DataFrame) -> pd.DataFrame:
    """Beta-Binomial posterior DNF rate. Prior sessions only.

    Over eleven races a raw 2/11 is mostly noise, so shrink toward the field
    rate with the weight of the (1, 9) prior - ten pseudo-races. Recentred on
    the field's own rate rather than the 0.1 that (1, 9) implies, so we shrink
    toward something measured. That field rate is also prior-sessions-only.
    """
    table = table.sort_values(["session_time", "driver_id"]).reset_index(drop=True)
    scored = table["is_trainable"]
    dnf = (table["dnf"] & scored).astype("float64")
    valid = scored.astype("float64")

    prior_dnf = pd.Series(np.nan, index=table.index, dtype="float64")
    prior_n = pd.Series(np.nan, index=table.index, dtype="float64")
    for _, sub in table.groupby("driver_id", sort=False):
        sub = sub.sort_values("session_time")
        prior_dnf.loc[sub.index] = dnf.loc[sub.index].cumsum().shift(1)
        prior_n.loc[sub.index] = valid.loc[sub.index].cumsum().shift(1)

    # Field rate over everything that ran strictly before this session.
    times = np.sort(table["session_time"].unique())
    cum_dnf = dnf.groupby(table["session_time"]).sum().reindex(times).fillna(0).cumsum().shift(1)
    cum_n = valid.groupby(table["session_time"]).sum().reindex(times).fillna(0).cumsum().shift(1)
    field_rate = (cum_dnf / cum_n).fillna(DNF_PRIOR_ALPHA / DNF_PRIOR_STRENGTH)
    field = table["session_time"].map(field_rate)

    prior_dnf = prior_dnf.fillna(0.0)
    prior_n = prior_n.fillna(0.0)
    table["dnf_rate_shrunk"] = (prior_dnf + DNF_PRIOR_STRENGTH * field) / (
        prior_n + DNF_PRIOR_STRENGTH
    )
    return table


def add_championship(table: pd.DataFrame) -> pd.DataFrame:
    """Championship standing going into the session, from cumulative points."""
    table = table.sort_values(["session_time", "driver_id"]).reset_index(drop=True)
    points = table["points"].fillna(0.0)

    prior_points = pd.Series(np.nan, index=table.index, dtype="float64")
    for _, sub in table.groupby("driver_id", sort=False):
        sub = sub.sort_values("session_time")
        prior_points.loc[sub.index] = points.loc[sub.index].cumsum().shift(1)
    table["points_before"] = prior_points.fillna(0.0)

    table["championship_position"] = (
        table.groupby(["round", "session_type"])["points_before"]
        .rank(method="min", ascending=False)
        .astype("float64")
    )
    return table


def add_sprint_features(table: pd.DataFrame) -> pd.DataFrame:
    """Sprint result carried onto that weekend's Race row.

    The sprint runs first, so a Race row can use it. A Sprint row can't - that's
    its own target - so those stay null and get imputed. Same for non-sprint
    rounds; dropping them instead would cost two thirds of the training data.
    """
    sprint = table[(table["session_type"] == "S") & ~table["is_prediction"]][
        ["round", "driver_id", "finish_position", "grid_position"]
    ].copy()
    sprint = sprint.rename(
        columns={
            "finish_position": "sprint_finish_position",
            "grid_position": "sprint_grid_position",
        }
    )
    sprint["sprint_positions_gained"] = (
        sprint["sprint_grid_position"] - sprint["sprint_finish_position"]
    )
    sprint = sprint.drop(columns=["sprint_grid_position"])

    table = table.merge(sprint, on=["round", "driver_id"], how="left")
    own_sprint = table["session_type"] == "S"
    table.loc[own_sprint, ["sprint_finish_position", "sprint_positions_gained"]] = np.nan
    table["has_sprint_data"] = table["sprint_finish_position"].notna()
    return table


# --------------------------------------------------------------------------
# Starting grid for a race that hasn't run
# --------------------------------------------------------------------------
# FastF1 only exposes GridPosition inside Race results, and those don't exist
# until the race has been run. Qualifying order is the obvious stand-in and it
# is simply wrong wherever a penalty applies - at Monza 2026 four drivers moved,
# three of them by more than ten places. So an official grid, transcribed by
# hand into data/manual/grid_R{round}.csv, takes precedence over quali order.
GRID_COLUMNS = ("driver_id", "abbreviation", "quali_position", "grid_position", "penalty_note")


def manual_grid_path(rnd: int) -> Path:
    return MANUAL / f"grid_R{int(rnd)}.csv"


def _display_path(path: Path) -> str:
    """Repo-relative where possible. MANUAL is patchable, so it may not be."""
    try:
        return str(path.relative_to(BASE))
    except ValueError:
        return str(path)


def load_manual_grid(rnd: int) -> pd.DataFrame | None:
    """The hand-entered official grid for one round, or None if there isn't one.

    Raises rather than warns on a malformed file. A half-filled grid is worse
    than no grid at all: the fallback path at least announces itself, whereas a
    file with three blank rows would quietly impute them to the field median.
    """
    path = manual_grid_path(rnd)
    if not path.exists():
        return None

    grid = pd.read_csv(path)
    missing_cols = [c for c in ("driver_id", "grid_position") if c not in grid.columns]
    if missing_cols:
        raise ValueError(f"{path.name} is missing required column(s) {missing_cols}")

    grid = grid.loc[:, [c for c in GRID_COLUMNS if c in grid.columns]].copy()
    grid["driver_id"] = grid["driver_id"].astype("string").str.strip()

    blank = grid["grid_position"].isna()
    if blank.any():
        raise ValueError(
            f"{path.name}: grid_position is empty for "
            f"{grid.loc[blank, 'driver_id'].tolist()}. Fill every row in from the "
            "official starting grid before building features."
        )
    grid["grid_position"] = grid["grid_position"].astype("float64")

    dupes = grid["driver_id"][grid["driver_id"].duplicated()].tolist()
    if dupes:
        raise ValueError(f"{path.name}: duplicate driver_id rows {dupes}")
    return grid


def _assert_grid_permutation(positions: pd.Series, context: str) -> None:
    """The grid must be 1..n exactly - no duplicates, no gaps, nobody missing."""
    values = positions.to_numpy(dtype="float64")
    n = values.size
    nulls = int(np.isnan(values).sum())
    assert not nulls, f"{context}: {nulls} driver(s) have no grid position"

    expected = set(range(1, n + 1))
    actual = sorted(int(v) for v in values)
    assert (values == np.floor(values)).all(), f"{context}: non-integer grid positions {actual}"

    duplicated = sorted({p for p in actual if actual.count(p) > 1})
    assert not duplicated, f"{context}: duplicate grid positions {duplicated}"
    assert set(actual) == expected, (
        f"{context}: grid positions are not a permutation of 1..{n}. "
        f"missing {sorted(expected - set(actual))}, unexpected {sorted(set(actual) - expected)}"
    )


def apply_prediction_grid(table: pd.DataFrame) -> pd.DataFrame:
    """Fill grid_position on prediction rows, from the manual grid where present.

    Runs before imputation, so a grid that came out of the manual file is never
    flagged as imputed and never overwritten by a field median.
    """
    needs_grid = table["is_prediction"] & table["grid_position"].isna()
    for rnd in sorted(table.loc[needs_grid, "round"].unique()):
        rows = needs_grid & table["round"].eq(rnd)
        grid = load_manual_grid(int(rnd))

        if grid is None:
            banner = "!" * 78
            message = (
                f"NO OFFICIAL GRID FOR ROUND {rnd}. grid_position is falling back to "
                f"QUALIFYING ORDER.\nGrid penalties are NOT accounted for, and "
                f"grid_position is the strongest single\nfeature in the model. Write "
                f"{_display_path(manual_grid_path(int(rnd)))} from the published\n"
                "starting grid and rebuild before locking any prediction."
            )
            print(f"\n{banner}\n{message}\n{banner}\n", file=sys.stderr)
            warnings.warn(message, stacklevel=2)
            table.loc[rows, "grid_position"] = table.loc[rows, "quali_position"]
            source = "qualifying order (NO penalties applied)"
        else:
            entrants = set(table.loc[rows, "driver_id"].astype("string"))
            listed = set(grid["driver_id"])
            assert listed == entrants, (
                f"round {rnd}: {manual_grid_path(int(rnd)).name} does not match the "
                f"entry list. missing from the file: {sorted(entrants - listed)}; "
                f"not entered in the round: {sorted(listed - entrants)}"
            )
            mapping = grid.set_index("driver_id")["grid_position"]
            table.loc[rows, "grid_position"] = (
                table.loc[rows, "driver_id"].astype("string").map(mapping).to_numpy()
            )
            source = f"{manual_grid_path(int(rnd)).name}"

        _assert_grid_permutation(
            table.loc[rows, "grid_position"], f"round {rnd} grid from {source}"
        )
        print(f"round {rnd}: grid_position taken from {source}", file=sys.stderr)
    return table


# --------------------------------------------------------------------------
# Imputation
# --------------------------------------------------------------------------
def _expanding_median(table: pd.DataFrame, col: str) -> pd.Series:
    """Median of col over every session strictly earlier than this row's.

    Prior-only, so truncating to earlier rounds can't change the answer.
    """
    out = pd.Series(np.nan, index=table.index, dtype="float64")
    values = table[col].astype("float64")
    for time in np.sort(table["session_time"].unique()):
        earlier = values[table["session_time"] < time]
        if earlier.notna().any():
            out[table["session_time"] == time] = earlier.median()
    return out


def _impute(
    table: pd.DataFrame, col: str, companion: str | None = None, constant: float | None = None
) -> None:
    """Fill gaps and flag them. Rows are never dropped for a missing feature.

    Order: this round's field median (covers the rookie case), then a
    same-session stand-in that's known before the start, then the median over
    earlier sessions, then a constant. Every step reads only the current row or
    earlier sessions - that's what keeps it stable under truncation, which is
    what makes the leakage test mean anything.
    """
    missing = table[col].isna()
    if missing.any():
        filled = table[col].fillna(
            table.groupby(["round", "session_type"])[col].transform("median")
        )
        if companion is not None:
            filled = filled.fillna(table[companion])
        filled = filled.fillna(_expanding_median(table, col))
        if constant is not None:
            filled = filled.fillna(constant)
        table[col] = filled
    table[f"{col}_imputed"] = missing


def impute_features(table: pd.DataFrame) -> pd.DataFrame:
    """Order matters - a companion column has to be filled before it's used."""
    _impute(table, "quali_pct_off_pole", constant=0.0)
    _impute(table, "grid_position", companion="quali_position", constant=20.0)
    _impute(table, "quali_teammate_delta_pct", constant=0.0)
    # Quali pace stands in for practice pace - same car, same weekend.
    _impute(table, "fp_best_pct_off_fastest", companion="quali_pct_off_pole")
    _impute(table, "fp_longrun_pace_pct", companion="quali_pct_off_pole")
    # Nobody has history at round 1, so form falls back to the best same-session
    # proxy that's still known before the lights go out.
    _impute(table, "form_quali_pace_ewm", companion="quali_pct_off_pole")
    _impute(table, "form_race_pace_ewm", companion="fp_longrun_pace_pct")
    _impute(table, "team_form_pace_ewm", companion="quali_pct_off_pole")
    _impute(table, "form_finish_pos_ewm", companion="grid_position")
    _impute(table, "positions_gained_ewm", constant=0.0)
    _impute(table, "sprint_finish_position", companion="form_finish_pos_ewm")
    _impute(table, "sprint_positions_gained", constant=0.0)
    _impute(table, "dnf_rate_shrunk", constant=DNF_PRIOR_ALPHA / DNF_PRIOR_STRENGTH)
    _impute(table, "championship_position", constant=20.0)
    _impute(table, "track_temp_mean", constant=30.0)
    table["was_wet"] = table["was_wet"].fillna(False).astype(bool)
    table["was_wet_imputed"] = False
    table["has_sprint_data"] = table["has_sprint_data"].fillna(False).astype(bool)
    table["has_sprint_data_imputed"] = False
    return table


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------
def validate(table: pd.DataFrame) -> None:
    numeric = [c for c in FEATURE_COLUMNS if pd.api.types.is_numeric_dtype(table[c])]

    infinite = {c: int(np.isinf(table[c].to_numpy(dtype="float64")).sum()) for c in numeric}
    bad = {c: n for c, n in infinite.items() if n}
    assert not bad, f"infinite values in feature columns: {bad}"

    dupes = table.duplicated(["round", "session_type", "driver_id"])
    assert not dupes.any(), (
        "duplicate (round, session_type, driver_id) rows:\n"
        f"{table.loc[dupes, ['round', 'session_type', 'driver_id']]}"
    )

    # Exactly one pole-sitter per session. Per session_type, because a Race is
    # measured off Q and a Sprint off SQ.
    for (rnd, session_type), grp in table.groupby(["round", "session_type"]):
        poles = int((grp["quali_pct_off_pole"] == 0.0).sum())
        assert poles == 1, (
            f"round {rnd} {session_type}: expected exactly 1 driver at "
            f"quali_pct_off_pole == 0.0, found {poles}"
        )

    still_null = {c: int(table[c].isna().sum()) for c in FEATURE_COLUMNS}
    leftover = {c: n for c, n in still_null.items() if n}
    assert not leftover, f"features still null after imputation: {leftover}"


def missingness_report(table: pd.DataFrame) -> pd.DataFrame:
    """How much of each feature had to be imputed, per round and session."""
    flags = [f"{c}_imputed" for c in FEATURE_COLUMNS if f"{c}_imputed" in table.columns]
    report = (
        table.groupby(["round", "session_type"])[flags].mean().mul(100.0).round(1)
    )
    report.columns = [c.replace("_imputed", "") for c in report.columns]
    return report


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------
def build_features(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Raw tables in, model table out. No I/O, so tests can feed it slices."""
    results, laps = raw["results"], raw["laps"]

    for name, frame in raw.items():
        seasons = set(frame["season"].unique())
        assert seasons <= {SEASON}, f"{name} contains non-{SEASON} data: {seasons}"

    table = base_frame(results)
    table = add_targets(table)

    # Race rows read Q, Sprint rows read SQ. A Sprint row reading Q would be
    # reading a session that hasn't happened yet.
    quali = quali_block(results)
    table["quali_session"] = table["session_type"].map(QUALI_SOURCE)
    table = table.merge(quali, on=["round", "quali_session", "driver_id"], how="left")

    table = table.merge(practice_block(laps), on=["round", "driver_id"], how="left")
    table = table.merge(
        race_pace_block(laps), on=["round", "session_type", "driver_id"], how="left"
    )
    table = table.merge(weather_block(raw["weather"]), on=["round", "session_type"], how="left")

    table = add_form_features(table)
    table = add_reliability(table)
    table = add_championship(table)
    table = add_sprint_features(table)

    # Prediction rows have no GridPosition - FastF1 only publishes it with the
    # race result. Take the official grid where we have it, quali order (loudly)
    # where we don't.
    table = apply_prediction_grid(table)

    table = impute_features(table)
    validate(table)

    ordered = KEY_COLUMNS + TARGET_COLUMNS + FEATURE_COLUMNS
    ordered += [f"{c}_imputed" for c in FEATURE_COLUMNS if f"{c}_imputed" in table.columns]
    return table[ordered].sort_values(["round", "session_type", "driver_id"]).reset_index(drop=True)


def main() -> int:
    PROCESSED.mkdir(parents=True, exist_ok=True)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        table = build_features(load_raw_frames())

    table.to_csv(OUT_PATH, index=False, float_format="%.6f")

    print("=" * 78)
    print(f"{SEASON} model table")
    print("=" * 78)
    print(f"\nrows: {len(table):,}   features: {len(FEATURE_COLUMNS)}")
    counts = table.groupby(["session_type", "is_prediction"]).size()
    for (session_type, is_pred), n in counts.items():
        kind = "prediction" if is_pred else "labelled"
        print(f"  {session_type:<2} {kind:<11} {n:>4} rows")
    print(f"  trainable (excl. DNS + prediction): {int(table['is_trainable'].sum()):>4} rows")
    print(f"  DNS rows held out of training:      {int(table['dns'].sum()):>4} rows")

    print("\ntarget summary (trainable rows):")
    train = table[table["is_trainable"]]
    print(f"  finish_position  mean {train['finish_position'].mean():.2f}")
    print(f"  won              {int(train['won'].sum()):>3}")
    print(f"  podium           {int(train['podium'].sum()):>3}")
    print(f"  dnf              {int(train['dnf'].sum()):>3} "
          f"({train['dnf'].mean() * 100:.1f}%)   "
          f"first-lap {int(train['dnf_first_lap'].sum())} / "
          f"later {int(train['dnf_after_lap1'].sum())}")

    print("\nimputed share per feature per round (%):")
    report = missingness_report(table)
    with pd.option_context("display.width", 200, "display.max_columns", 50):
        print(report.to_string())

    print("\nfeatures never imputed:", [c for c in report.columns if report[c].max() == 0.0])

    for warning in caught:
        print(f"\nWARNING: {warning.message}")

    print(f"\nwrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
