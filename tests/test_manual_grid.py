"""Tests for the manual starting-grid override.

grid_position is the single strongest feature in the model, and for a race that
hasn't run FastF1 cannot supply it - GridPosition only exists inside Race
results. So it comes from a hand-transcribed file, which means the failure mode
is a typo rather than a bug: a driver left out, a position entered twice, a row
left blank. Every one of those has to stop the build rather than get imputed.

The permutation check is the whole point. A grid that is missing P7 and has two
P12s still builds a perfectly plausible-looking model table.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

from src import features  # noqa: E402
from src.features import (  # noqa: E402
    _assert_grid_permutation,
    apply_prediction_grid,
    load_manual_grid,
    manual_grid_path,
)

RACE_ROUND = 13


# --------------------------------------------------------------------------
# The real file that today's prediction depends on
# --------------------------------------------------------------------------
def test_round_13_grid_file_is_a_clean_permutation():
    grid = load_manual_grid(RACE_ROUND)
    assert grid is not None, f"{manual_grid_path(RACE_ROUND)} is missing"
    _assert_grid_permutation(grid["grid_position"], "grid_R13.csv")


def test_round_13_grid_covers_the_whole_entry_list():
    """The file has to name every entrant, not just the ones who were penalised."""
    table = features.load_model_table()
    entrants = set(
        table.loc[
            (table["round"] == RACE_ROUND) & table["is_prediction"].fillna(False),
            "driver_id",
        ]
    )
    listed = set(load_manual_grid(RACE_ROUND)["driver_id"])
    assert listed == entrants, (
        f"missing from grid_R13.csv: {sorted(entrants - listed)}; "
        f"not entered in round {RACE_ROUND}: {sorted(listed - entrants)}"
    )


def test_penalised_drivers_actually_moved():
    """Otherwise the override is in place but doing nothing, which is the bug
    it exists to prevent.

    Only demotions need a note. Everyone below a penalised driver moves *up* by
    a place or two without having done anything.
    """
    grid = load_manual_grid(RACE_ROUND)
    assert (grid["quali_position"] != grid["grid_position"]).any(), (
        "grid_R13.csv is identical to qualifying order. Either no penalties were "
        "applied, or the file was never filled in."
    )
    demoted = grid[grid["grid_position"] > grid["quali_position"]]
    unexplained = demoted.loc[demoted["penalty_note"].isna(), "driver_id"].tolist()
    assert not unexplained, (
        f"driver(s) {unexplained} start behind their qualifying position with no "
        "penalty_note. Either the note is missing or the grid is mistranscribed."
    )


# --------------------------------------------------------------------------
# Malformed files must fail loudly
# --------------------------------------------------------------------------
def _write_grid(tmp_path: Path, monkeypatch, rows: list[dict], rnd: int = 99) -> None:
    monkeypatch.setattr(features, "MANUAL", tmp_path)
    pd.DataFrame(rows).to_csv(tmp_path / f"grid_R{rnd}.csv", index=False)


def _rows(positions: list, n: int = 4) -> list[dict]:
    return [
        {
            "driver_id": f"d{i}",
            "abbreviation": f"D{i}",
            "quali_position": i + 1,
            "grid_position": positions[i],
            "penalty_note": "",
        }
        for i in range(n)
    ]


def test_missing_file_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(features, "MANUAL", tmp_path)
    assert load_manual_grid(99) is None


def test_blank_grid_position_raises(tmp_path, monkeypatch):
    """The unfilled-template case. Blanks would otherwise be imputed silently."""
    _write_grid(tmp_path, monkeypatch, _rows([1, 2, None, 4]))
    with pytest.raises(ValueError, match="grid_position is empty"):
        load_manual_grid(99)


def test_duplicate_driver_raises(tmp_path, monkeypatch):
    rows = _rows([1, 2, 3, 4])
    rows[3]["driver_id"] = "d0"
    _write_grid(tmp_path, monkeypatch, rows)
    with pytest.raises(ValueError, match="duplicate driver_id"):
        load_manual_grid(99)


def test_missing_required_column_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(features, "MANUAL", tmp_path)
    pd.DataFrame({"driver_id": ["d0"], "abbreviation": ["D0"]}).to_csv(
        tmp_path / "grid_R99.csv", index=False
    )
    with pytest.raises(ValueError, match="missing required column"):
        load_manual_grid(99)


@pytest.mark.parametrize(
    "positions, expected",
    [
        ([1, 2, 2, 4], "duplicate grid positions"),
        ([1, 2, 3, 5], "not a permutation"),
        ([0, 1, 2, 3], "not a permutation"),
    ],
)
def test_bad_permutations_are_rejected(positions, expected):
    with pytest.raises(AssertionError, match=expected):
        _assert_grid_permutation(pd.Series(positions, dtype="float64"), "test grid")


# --------------------------------------------------------------------------
# The fallback path
# --------------------------------------------------------------------------
def _prediction_table(n: int = 4) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "round": 99,
            "driver_id": [f"d{i}" for i in range(n)],
            "is_prediction": True,
            "grid_position": [float("nan")] * n,
            "quali_position": [float(i + 1) for i in range(n)],
        }
    )


def test_falls_back_to_quali_order_with_a_warning(tmp_path, monkeypatch):
    monkeypatch.setattr(features, "MANUAL", tmp_path)  # no grid_R99.csv in here
    with pytest.warns(UserWarning, match="NO OFFICIAL GRID"):
        out = apply_prediction_grid(_prediction_table())
    assert out["grid_position"].tolist() == [1.0, 2.0, 3.0, 4.0]


def test_manual_grid_beats_quali_order(tmp_path, monkeypatch):
    _write_grid(tmp_path, monkeypatch, _rows([4, 2, 3, 1]))
    out = apply_prediction_grid(_prediction_table())
    assert out["grid_position"].tolist() == [4.0, 2.0, 3.0, 1.0]


def test_entry_list_mismatch_raises(tmp_path, monkeypatch):
    """A stale grid file from the previous round must not be applied quietly."""
    rows = _rows([1, 2, 3, 4])
    rows[2]["driver_id"] = "someone_else"
    _write_grid(tmp_path, monkeypatch, rows)
    with pytest.raises(AssertionError, match="does not match the entry list"):
        apply_prediction_grid(_prediction_table())
