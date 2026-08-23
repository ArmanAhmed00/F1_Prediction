"""Typed readers for the raw 2026 FastF1 CSV dumps.

Everything downstream must go through :func:`load_raw` rather than a bare
``pd.read_csv``.  The CSVs on disk are plain text: without the dtype map written
by ``ingest.py`` pandas will happily infer ``TrackStatus`` "14" as the integer
14, or turn an all-null string column into ``float64``.  Those failures are
silent, which is why this module exists.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parents[1]
RAW = BASE / "data" / "raw"
DTYPES_PATH = RAW / "dtypes.json"

TABLES = ("results", "laps", "weather", "race_control", "schedule")

_TRUE = {"True", "true", "TRUE", "1", "1.0"}
_FALSE = {"False", "false", "FALSE", "0", "0.0"}


def dtype_map(name: str) -> dict[str, str]:
    """The column -> dtype map ingest recorded for one table."""
    if not DTYPES_PATH.exists():
        raise FileNotFoundError(
            f"{DTYPES_PATH} is missing - run `uv run python src/ingest.py` first"
        )
    with DTYPES_PATH.open() as fh:
        all_dtypes = json.load(fh)
    if name not in all_dtypes:
        raise KeyError(f"unknown raw table {name!r}; known: {sorted(all_dtypes)}")
    return all_dtypes[name]


def _to_bool(value):
    # read_csv may already have given us real bools. Check those first:
    # `True in {"True", ...}` is False, which would null the whole column.
    if isinstance(value, bool):
        return value
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return pd.NA
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    return pd.NA


def load_raw(name: str) -> pd.DataFrame:
    """Read ``data/raw/<name>.csv`` back with the dtypes recorded at ingest time.

    Timedeltas were already flattened to float seconds (``*_s``) by the ingest,
    since timedeltas do not survive a CSV round-trip.
    """
    path = RAW / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing - run `uv run python src/ingest.py` first")

    dtypes = dtype_map(name)
    date_cols = [c for c, d in dtypes.items() if d.startswith("datetime64")]
    bool_cols = [c for c, d in dtypes.items() if d in ("bool", "boolean")]
    # Bools and dates get fixed after the read. Everything else is pinned now,
    # which is what keeps TrackStatus and DriverNumber as strings.
    read_dtypes: dict = {
        c: d for c, d in dtypes.items() if c not in date_cols and c not in bool_cols
    }

    df = pd.read_csv(path, dtype=read_dtypes, low_memory=False)

    for col in date_cols:
        if col not in df.columns:
            continue
        tz_aware = "," in dtypes[col]
        df[col] = pd.to_datetime(df[col], errors="coerce", utc=tz_aware)

    for col in bool_cols:
        if col not in df.columns:
            continue
        mapped = df[col].astype("object").map(_to_bool).astype("boolean")
        df[col] = mapped.astype("bool") if dtypes[col] == "bool" else mapped

    return df


def load_all() -> dict[str, pd.DataFrame]:
    """Every raw table that exists on disk, keyed by name."""
    return {t: load_raw(t) for t in TABLES if (RAW / f"{t}.csv").exists()}
