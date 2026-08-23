"""Leakage tests for the feature table.

A leak doesn't announce itself. The build still runs, the missingness table
still looks fine, and the only symptom is a backtest score that's too good.
Recomputation is the only check that actually works, so every test here rebuilds
the whole table from a doctored copy of the input and demands round-k features
come out identical.

Three ways a leak can happen, and they fail differently:

  test_future_rounds_do_not_leak
      Delete everything after round k. Catches anything that peeked forward -
      a missing shift(1), an expanding stat over the full season, a median
      taken globally instead of per round.

  test_session_outcome_does_not_leak
      Keep all the rounds but scramble round k's own result and lap times.
      Catches what truncation structurally can't: a feature reading the outcome
      of the session it's meant to be predicting. Grid position and Q times are
      left alone - those are genuinely known on Saturday.

  test_sprint_row_does_not_read_qualifying
      Scramble Q on a sprint weekend. Neither test above can see this one.

Round 6 is a conventional weekend, round 9 a sprint weekend, so both session
orderings get covered.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

from src.features import FEATURE_COLUMNS, build_features, load_raw_frames  # noqa: E402

# Rounds under test: one conventional weekend, one sprint weekend.
ROUNDS = [6, 9]

# Sprint weekends, where Qualifying runs after the Sprint.
SPRINT_ROUNDS = [4, 9]

# How the session turned out. None of this is known until the flag falls, so no
# feature may depend on it. GridPosition and the Q times are left out on
# purpose - those are known on Saturday.
OUTCOME_COLUMNS = ["Position", "ClassifiedPosition", "Status", "Points", "Time_s", "Laps"]

# A real leak moves a feature by orders of magnitude. This tolerance is only
# here to absorb float summation order.
RTOL = 1e-9
ATOL = 1e-12


@pytest.fixture(scope="module")
def raw() -> dict[str, pd.DataFrame]:
    return load_raw_frames()


@pytest.fixture(scope="module")
def full(raw) -> pd.DataFrame:
    return build_features({name: frame.copy() for name, frame in raw.items()})


def _round_slice(table: pd.DataFrame, rnd: int, session_type: str | None = None) -> pd.DataFrame:
    rows = table[table["round"] == rnd]
    if session_type is not None:
        rows = rows[rows["session_type"] == session_type]
    return rows.set_index(["session_type", "driver_id"]).sort_index()


def _features_differ(expected: pd.DataFrame, actual: pd.DataFrame) -> bool:
    for col in FEATURE_COLUMNS:
        left, right = expected[col], actual[col]
        if pd.api.types.is_bool_dtype(left) or pd.api.types.is_object_dtype(left):
            if not left.equals(right):
                return True
        elif not np.allclose(
            left.to_numpy(dtype="float64"),
            right.to_numpy(dtype="float64"),
            rtol=RTOL,
            atol=ATOL,
            equal_nan=True,
        ):
            return True
    return False


def _assert_features_identical(expected: pd.DataFrame, actual: pd.DataFrame, context: str) -> None:
    assert list(expected.index) == list(actual.index), (
        f"{context}: row set changed, so the comparison would be meaningless"
    )

    mismatches = []
    for col in FEATURE_COLUMNS:
        left, right = expected[col], actual[col]
        if pd.api.types.is_bool_dtype(left) or pd.api.types.is_object_dtype(left):
            if not left.equals(right):
                differing = left.index[left != right].tolist()
                mismatches.append(f"  {col}: differs for {differing[:5]}")
            continue
        lv = left.to_numpy(dtype="float64")
        rv = right.to_numpy(dtype="float64")
        if not np.allclose(lv, rv, rtol=RTOL, atol=ATOL, equal_nan=True):
            delta = np.abs(lv - rv)
            worst = int(np.nanargmax(delta))
            mismatches.append(
                f"  {col}: max |diff| {np.nanmax(delta):.6g} "
                f"at {expected.index[worst]} ({lv[worst]!r} -> {rv[worst]!r})"
            )

    assert not mismatches, (
        f"{context}\n"
        f"{len(mismatches)} feature(s) changed, so they are reading data they "
        f"must not have:\n" + "\n".join(mismatches)
    )


@pytest.mark.parametrize("rnd", ROUNDS)
def test_future_rounds_do_not_leak(raw, full, rnd):
    """Round-k features must not move when every later round is deleted."""
    truncated = {name: frame[frame["round"] <= rnd].copy() for name, frame in raw.items()}
    rebuilt = build_features(truncated)

    expected = _round_slice(full, rnd)
    actual = _round_slice(rebuilt, rnd)

    # Otherwise this passes trivially if the round stops being built.
    assert len(expected) > 0, f"no round {rnd} rows in the full table"
    _assert_features_identical(
        expected, actual, f"round {rnd}: dropping rounds > {rnd} changed round-{rnd} features"
    )


@pytest.mark.parametrize("rnd", ROUNDS)
def test_session_outcome_does_not_leak(raw, full, rnd):
    """Round-k features must not move when round k's own race result is scrambled.

    Only the Race is doctored, never the Sprint that precedes it on a sprint
    weekend: the Sprint result is legitimately available to the Race row, so
    scrambling it would break features that are entitled to read it.
    """
    doctored = {name: frame.copy() for name, frame in raw.items()}

    results = doctored["results"]
    race = (results["round"] == rnd) & (results["session_type"] == "R")
    assert race.any(), f"round {rnd} has no Race results to scramble"
    for col in OUTCOME_COLUMNS:
        # Reversing keeps the dtype and the set of values, but guarantees
        # everyone ends up with someone else's result.
        results.loc[race, col] = results.loc[race, col].to_numpy()[::-1]

    # Race pace is measured during the race, so it can only ever be used
    # shifted onto a later round. Wrecking it here proves that.
    laps = doctored["laps"]
    race_laps = (laps["round"] == rnd) & (laps["session_type"] == "R")
    laps.loc[race_laps, "LapTime_s"] = laps.loc[race_laps, "LapTime_s"] * 1.5

    rebuilt = build_features(doctored)

    expected = _round_slice(full, rnd)
    actual = _round_slice(rebuilt, rnd)
    _assert_features_identical(
        expected,
        actual,
        f"round {rnd}: scrambling the round-{rnd} race result changed round-{rnd} features",
    )


@pytest.mark.parametrize("rnd", SPRINT_ROUNDS)
def test_sprint_row_does_not_read_qualifying(raw, full, rnd):
    """A Sprint row must be completely blind to that weekend's Qualifying.

    This is the subtlest ordering trap in the season.  A sprint weekend runs
    FP1 -> Sprint Qualifying -> SPRINT -> Qualifying -> Race, so Qualifying
    happens AFTER the Sprint.  Sourcing a Sprint row's grid or pace from Q
    rather than SQ reads a session that has not run yet, and neither of the
    tests above can see it: truncation keeps round k's Q, and scrambling the
    Race leaves Q untouched.

    So scramble Q itself.  The Sprint rows must not move; the Race rows must,
    which is what proves the scramble had teeth.
    """
    doctored = {name: frame.copy() for name, frame in raw.items()}
    results = doctored["results"]
    quali = (results["round"] == rnd) & (results["session_type"] == "Q")
    assert quali.any(), f"round {rnd} has no Qualifying results to scramble"
    for col in ["Q1_s", "Q2_s", "Q3_s", "Position"]:
        results.loc[quali, col] = results.loc[quali, col].to_numpy()[::-1]

    rebuilt = build_features(doctored)

    _assert_features_identical(
        _round_slice(full, rnd, "S"),
        _round_slice(rebuilt, rnd, "S"),
        f"round {rnd}: scrambling Qualifying changed the SPRINT features, but the "
        f"Sprint runs before Qualifying",
    )
    assert _features_differ(_round_slice(full, rnd, "R"), _round_slice(rebuilt, rnd, "R")), (
        f"round {rnd}: scrambling Qualifying left the RACE features unchanged, so the "
        "check above is vacuous"
    )


@pytest.mark.parametrize("rnd", ROUNDS)
def test_scramble_actually_changed_the_targets(raw, full, rnd):
    """If the scrambling were a no-op, the test above would prove nothing."""
    doctored = {name: frame.copy() for name, frame in raw.items()}
    results = doctored["results"]
    race = (results["round"] == rnd) & (results["session_type"] == "R")
    for col in OUTCOME_COLUMNS:
        results.loc[race, col] = results.loc[race, col].to_numpy()[::-1]

    rebuilt = build_features(doctored)
    expected = _round_slice(full, rnd)["finish_position"]
    actual = _round_slice(rebuilt, rnd)["finish_position"]
    assert not expected.equals(actual), (
        f"round {rnd}: scrambling the result left finish_position unchanged, "
        "so the leakage test above is vacuous"
    )
