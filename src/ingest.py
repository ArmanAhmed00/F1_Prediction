#!/usr/bin/env python3
"""Pull every 2026 session out of FastF1 and dump it to CSV.

Download and store only - no features, no targets. 2026 only, too: the
regulation reset makes earlier seasons useless for pace and mixing them in
would be a leakage risk.

    uv run python src/ingest.py [--rounds 1-12] [--force]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# --------------------------------------------------------------------------
# Derived from this file's location, never hardcoded.
# --------------------------------------------------------------------------
BASE = Path(__file__).resolve().parents[1]
CACHE = BASE / "cache"
RAW = BASE / "data" / "raw"
PARTS = RAW / "_parts"

SEASON = 2026

# Which sessions exist for each EventFormat. Read off the schedule rather than
# hardcoding which rounds are sprint weekends.
FORMAT_SESSIONS: dict[str, tuple[str, ...]] = {
    "conventional": ("FP1", "FP2", "FP3", "Q", "R"),
    "sprint_qualifying": ("FP1", "SQ", "S", "Q", "R"),
    # Old sprint layouts. Only here so an unexpected format degrades gracefully
    # instead of silently dropping sessions.
    "sprint_shootout": ("FP1", "SQ", "FP2", "S", "Q", "R"),
    "sprint": ("FP1", "Q", "FP2", "S", "R"),
}

# session.results is only meaningful for classified sessions.
RESULT_SESSIONS = frozenset({"R", "S", "Q", "SQ"})
# Race control messages are only collected for the two races of a weekend.
RCM_SESSIONS = frozenset({"R", "S"})

TABLES = ("results", "laps", "weather", "race_control", "schedule")

# These have to stay strings. TrackStatus is concatenated codes - "1" is
# all-clear, "14" means the lap saw green AND a safety car - so casting it to int
# destroys it. Driver and team numbers are identifiers, not quantities.
FORCE_STR_COLS = (
    "TrackStatus",
    "DriverNumber",
    "RacingNumber",
    "DriverId",
    "TeamId",
)

RETRY_DELAYS = (2, 4, 8)  # seconds; 1 initial attempt + 3 retries

log = logging.getLogger("ingest")


# --------------------------------------------------------------------------
# Frame hygiene
# --------------------------------------------------------------------------
def flatten_timedeltas(df: pd.DataFrame) -> pd.DataFrame:
    """Every timedelta64 column -> float seconds, renamed with an _s suffix.

    Timedeltas don't survive a CSV round-trip and fail silently on the way back,
    which is the worst kind of failure. Column order is preserved.
    """
    df = df.copy()
    renames: dict[str, str] = {}
    for col in df.columns:
        if pd.api.types.is_timedelta64_dtype(df[col]):
            df[col] = df[col].dt.total_seconds()
            renames[col] = f"{col}_s"
    return df.rename(columns=renames)


def _clean_str(value):
    """Keep identifiers as strings, but turn FastF1's blanks into NA.

    FastF1 gives "" rather than NaN for DriverId/TeamId when its driver-info
    source is empty - every SQ session and some FP1s. A blank string is a null
    join key in disguise, so make it an honest null.
    """
    if isinstance(value, str):
        stripped = value.strip()
        return stripped if stripped else pd.NA
    return value if pd.isna(value) else str(value)


def force_str_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in FORCE_STR_COLS:
        if col in df.columns:
            df[col] = df[col].map(_clean_str)
    return df


def add_metadata(df: pd.DataFrame, rnd: int, event_name: str, session_type: str) -> pd.DataFrame:
    """Stamp season / round / event_name / session_type onto every row."""
    df = df.reset_index(drop=True)
    for pos, (name, value) in enumerate(
        (
            ("season", SEASON),
            ("round", int(rnd)),
            ("event_name", str(event_name)),
            ("session_type", str(session_type)),
        )
    ):
        if name in df.columns:
            df = df.drop(columns=[name])
        df.insert(pos, name, value)
    return df


def prepare(df: pd.DataFrame, rnd: int, event_name: str, session_type: str) -> pd.DataFrame:
    df = flatten_timedeltas(pd.DataFrame(df))
    df = force_str_columns(df)
    return add_metadata(df, rnd, event_name, session_type)


# --------------------------------------------------------------------------
# Part files (resumability)
# --------------------------------------------------------------------------
# One directory per round - _parts/round_07/7_Q_laps.csv - so a weekend's files
# stay together. The round stays in the filename as well, so a part copied out of
# its folder still says what it is.
def round_dir(rnd: int) -> Path:
    return PARTS / f"round_{int(rnd):02d}"


def part_path(rnd: int, session_type: str, table: str) -> Path:
    return round_dir(rnd) / f"{rnd}_{session_type}_{table}.csv"


def migrate_flat_parts() -> int:
    """Move parts from the old flat layout into their round folder.

    Without this the resume check sees an empty _parts/ and re-downloads the
    whole season, which runs straight into FastF1's 500-calls/hour limit.
    """
    moved = 0
    for path in list(PARTS.glob("*_*")):
        if path.is_dir():
            continue
        rnd, _, rest = path.name.partition("_")
        if not rnd.isdigit() or not rest:
            continue
        if int(rnd) == 0:
            # Legacy whole-season schedule part; the schedule is per-round now.
            path.unlink()
            continue
        target = round_dir(int(rnd)) / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        path.replace(target)
        moved += 1
    if moved:
        log.info("moved %s part file(s) into per-round folders", moved)
    return moved


def write_part(df: pd.DataFrame, path: Path) -> None:
    """Write a part, plus a sidecar recording its dtypes.

    The sidecar holds the in-memory dtypes. Without it, reading the parts back
    for the final concat re-infers them, and an all-null string column in one
    part comes back as float64 and poisons the whole concatenation.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".csv.tmp")
    df.to_csv(tmp, index=False, float_format="%.6f")
    tmp.replace(path)
    sidecar = path.with_suffix(".dtypes.json")
    with sidecar.open("w") as fh:
        json.dump({c: str(d) for c, d in df.dtypes.items()}, fh, indent=2, sort_keys=True)


def _merge_dtype(observed: list[str]) -> str:
    """Reconcile one column's dtype across parts."""
    seen = set(observed)
    if len(seen) == 1:
        return observed[0]
    if all(d.startswith("datetime64") for d in seen):
        return sorted(seen)[-1]
    if seen <= {"bool", "boolean", "object"}:
        # A part where the column was all-null records "object". Still boolean,
        # it just needs the nullable dtype to hold the gaps.
        return "boolean" if seen & {"bool", "boolean"} else "object"
    if all(d.startswith(("int", "uint", "float")) for d in seen):
        return "float64"
    return "object"


def _cast(series: pd.Series, dtype: str) -> pd.Series:
    """Cast a string column read off disk back to its recorded dtype."""
    if dtype.startswith("datetime64"):
        return pd.to_datetime(series, errors="coerce", utc=("," in dtype))
    if dtype in ("bool", "boolean"):
        mapped = series.map({"True": True, "False": False}).astype("boolean")
        return mapped.astype("bool") if dtype == "bool" and mapped.notna().all() else mapped
    if dtype.startswith(("int", "uint", "float")):
        numeric = pd.to_numeric(series, errors="coerce")
        try:
            return numeric.astype(dtype)
        except (ValueError, TypeError):
            return numeric.astype("float64")
    return series


def concat_parts(table: str) -> pd.DataFrame:
    """Read every part for one table and stitch them into the final frame."""
    paths = sorted(PARTS.glob(f"round_*/*_{table}.csv"), key=lambda p: _part_sort_key(p, table))
    if not paths:
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    observed: dict[str, list[str]] = {}
    for path in paths:
        # Read as text, so no two parts can disagree about a dtype.
        frame = pd.read_csv(path, dtype=str, low_memory=False)
        frames.append(frame)
        sidecar = path.with_suffix(".dtypes.json")
        if sidecar.exists():
            with sidecar.open() as fh:
                for col, dtype in json.load(fh).items():
                    observed.setdefault(col, []).append(dtype)

    combined = pd.concat(frames, ignore_index=True, sort=False)
    for col in combined.columns:
        dtype = _merge_dtype(observed[col]) if col in observed else "object"
        if dtype == "object":
            continue
        combined[col] = _cast(combined[col], dtype)
    return force_str_columns(combined)


def backfill_driver_keys(tables: dict[str, pd.DataFrame]) -> dict[str, int]:
    """Fill DriverId/TeamId gaps using the season-wide car-number mapping.

    Some sessions come back with no driver ids at all, which leaves them
    unjoinable on the only stable key. Car numbers are fixed for the season, so
    a number seen with an id anywhere resolves it everywhere. Returns how many
    rows per table still have no DriverId afterwards.
    """
    sources = [
        tables[name].loc[:, [c for c in ("round", "DriverNumber", "DriverId", "TeamId") if c in tables[name].columns]]
        for name in ("results", "laps")
        if name in tables and {"DriverNumber", "DriverId"} <= set(tables[name].columns)
    ]
    if not sources:
        return {}
    src = pd.concat(sources, ignore_index=True)

    known = src.dropna(subset=["DriverId"])
    driver_map = known.drop_duplicates("DriverNumber").set_index("DriverNumber")["DriverId"]
    clashes = known.groupby("DriverNumber")["DriverId"].nunique()
    for number in clashes[clashes > 1].index:
        log.warning(
            "car number %s maps to several DriverIds this season; backfilling with %r",
            number,
            driver_map[number],
        )

    team_known = src.dropna(subset=["TeamId"]) if "TeamId" in src.columns else src.iloc[:0]
    # Per round first, so a mid-season seat change is respected.
    team_by_round = (
        team_known.drop_duplicates(["round", "DriverNumber"]).set_index(["round", "DriverNumber"])["TeamId"]
        if not team_known.empty and "round" in team_known.columns
        else None
    )
    team_by_season = (
        team_known.drop_duplicates("DriverNumber").set_index("DriverNumber")["TeamId"]
        if not team_known.empty
        else None
    )

    residual: dict[str, int] = {}
    for name in ("results", "laps"):
        df = tables.get(name)
        if df is None or df.empty or "DriverNumber" not in df.columns:
            continue
        if "DriverId" in df.columns:
            df["DriverId"] = df["DriverId"].fillna(df["DriverNumber"].map(driver_map))
        if "TeamId" in df.columns:
            if team_by_round is not None and "round" in df.columns:
                keys = pd.MultiIndex.from_arrays([df["round"], df["DriverNumber"]])
                lookup = pd.Series(team_by_round.reindex(keys).to_numpy(), index=df.index)
                df["TeamId"] = df["TeamId"].fillna(lookup)
            if team_by_season is not None:
                df["TeamId"] = df["TeamId"].fillna(df["DriverNumber"].map(team_by_season))
        if "DriverId" in df.columns:
            still_null = int(df["DriverId"].isna().sum())
            if still_null:
                residual[name] = still_null
    return residual


def _part_sort_key(path: Path, table: str) -> tuple[int, str]:
    stem = path.name[: -len(f"_{table}.csv")]
    rnd, _, session_type = stem.partition("_")
    try:
        return (int(rnd), session_type)
    except ValueError:
        return (10**6, stem)


# --------------------------------------------------------------------------
# Session download
# --------------------------------------------------------------------------
def _frame_or_none(session, attr: str):
    """FastF1 raises if a frame was never populated. Treat that as 'not there'."""
    try:
        frame = getattr(session, attr)
    except Exception as exc:  # noqa: BLE001 - fastf1 raises several types here
        log.debug("%s unavailable: %s", attr, exc)
        return None
    return None if frame is None else pd.DataFrame(frame)


def _driver_key_map(results: pd.DataFrame | None) -> pd.DataFrame | None:
    """DriverNumber -> DriverId/TeamId, the keys everything downstream joins on.

    session.laps only carries Abbreviation and TeamName, which are for display.
    Keying on driver name strings is exactly what we don't want downstream.
    """
    if results is None or results.empty:
        return None
    needed = {"DriverNumber", "DriverId", "TeamId"}
    if not needed.issubset(results.columns):
        return None
    keys = results.loc[:, ["DriverNumber", "DriverId", "TeamId"]].copy()
    keys["DriverNumber"] = keys["DriverNumber"].astype(str)
    return keys.drop_duplicates(subset="DriverNumber")


def _attach_driver_keys(laps: pd.DataFrame, keys: pd.DataFrame | None) -> pd.DataFrame:
    if keys is None or laps.empty or "DriverNumber" not in laps.columns:
        return laps
    laps = laps.copy()
    laps["DriverNumber"] = laps["DriverNumber"].astype(str)
    merged = laps.merge(keys, on="DriverNumber", how="left")
    # Put the keys next to the number they came from.
    cols = [c for c in merged.columns if c not in ("DriverId", "TeamId")]
    at = cols.index("DriverNumber") + 1
    return merged.loc[:, cols[:at] + ["DriverId", "TeamId"] + cols[at:]]


def load_session(fastf1_mod, rnd: int, session_type: str):
    """Load one session with exponential backoff. Raises if it never succeeds."""
    last_exc: Exception | None = None
    for attempt in range(len(RETRY_DELAYS) + 1):
        try:
            session = fastf1_mod.get_session(SEASON, rnd, session_type)
            session.load(laps=True, telemetry=False, weather=True, messages=True)
            return session
        except Exception as exc:  # noqa: BLE001 - fastf1/network raise broadly
            last_exc = exc
            if attempt < len(RETRY_DELAYS):
                delay = RETRY_DELAYS[attempt]
                tqdm.write(
                    f"  round {rnd} {session_type}: attempt {attempt + 1} failed "
                    f"({type(exc).__name__}: {exc}); retrying in {delay}s"
                )
                time.sleep(delay)
    raise last_exc  # type: ignore[misc]


def ingest_session(fastf1_mod, rnd: int, event_name: str, session_type: str, force: bool):
    """Download one session and write its parts.

    Returns ``("done"|"cached"|"skipped"|"failed", detail)``.
    """
    wanted = ["laps", "weather"]
    if session_type in RESULT_SESSIONS:
        wanted.append("results")
    if session_type in RCM_SESSIONS:
        wanted.append("race_control")

    if not force and all(part_path(rnd, session_type, t).exists() for t in wanted):
        return "cached", ""

    try:
        session = load_session(fastf1_mod, rnd, session_type)
    except Exception as exc:  # noqa: BLE001
        return "failed", f"{type(exc).__name__}: {exc}"

    results = _frame_or_none(session, "results")
    laps = _frame_or_none(session, "laps")

    # A session that hasn't happened yet still gives you a Session object -
    # FastF1 just warns and loads nothing. Catch that here and skip.
    has_laps = laps is not None and not laps.empty
    has_results = results is not None and not results.empty
    if not (has_laps or has_results):
        return "skipped", "no data published yet (session has not taken place?)"

    frames: dict[str, pd.DataFrame | None] = {}
    if "results" in wanted:
        frames["results"] = results
    if "laps" in wanted:
        frames["laps"] = _attach_driver_keys(laps, _driver_key_map(results)) if has_laps else laps
    if "weather" in wanted:
        frames["weather"] = _frame_or_none(session, "weather_data")
    if "race_control" in wanted:
        frames["race_control"] = _frame_or_none(session, "race_control_messages")

    missing: list[str] = []
    for table, frame in frames.items():
        if frame is None:
            missing.append(table)
            continue
        write_part(prepare(frame, rnd, event_name, session_type), part_path(rnd, session_type, table))

    detail = f"no {', '.join(missing)} data" if missing else ""
    return "done", detail


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def parse_rounds(spec: str) -> list[int]:
    """Parse "1-12", "3", or "1-4,7,9-11" into a sorted list of round numbers."""
    rounds: set[int] = set()
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            lo, _, hi = chunk.partition("-")
            start, end = int(lo), int(hi)
            if start > end:
                raise argparse.ArgumentTypeError(f"empty round range {chunk!r}")
            rounds.update(range(start, end + 1))
        else:
            rounds.add(int(chunk))
    if not rounds:
        raise argparse.ArgumentTypeError("no rounds selected")
    return sorted(rounds)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"Ingest {SEASON} F1 session data via FastF1.")
    parser.add_argument(
        "--rounds",
        type=parse_rounds,
        default="1-12",
        help="rounds to ingest, e.g. 1-12 or 1-4,9 (default: 1-12)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-download sessions even if their part files already exist "
        "(the FastF1 HTTP cache is kept, so this stays fast)",
    )
    args = parser.parse_args(argv)
    if isinstance(args.rounds, str):  # default value never went through the type
        args.rounds = parse_rounds(args.rounds)
    return args


def sessions_for(event_format: str, rnd: int) -> tuple[str, ...]:
    try:
        return FORMAT_SESSIONS[event_format]
    except KeyError:
        log.warning(
            "round %s: unknown EventFormat %r, falling back to a conventional weekend",
            rnd,
            event_format,
        )
        return FORMAT_SESSIONS["conventional"]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stderr
    )

    for directory in (CACHE, RAW, PARTS):
        directory.mkdir(parents=True, exist_ok=True)
    migrate_flat_parts()

    # Cache has to be enabled before any other FastF1 call. An existing cache
    # makes re-runs near-instant - don't delete it.
    import fastf1

    fastf1.Cache.enable_cache(str(CACHE))
    fastf1.set_log_level("WARNING")

    schedule = fastf1.get_event_schedule(SEASON, include_testing=False)
    schedule = schedule[schedule["RoundNumber"].isin(args.rounds)].sort_values("RoundNumber")
    if schedule.empty:
        log.error("no %s events match rounds %s", SEASON, args.rounds)
        return 1

    # The schedule is per-event, so session_type gets the sentinel "EVENT".
    # One part per round, so ingesting a subset of rounds never shrinks
    # schedule.csv below what earlier runs already collected.
    sched_out = flatten_timedeltas(pd.DataFrame(schedule).reset_index(drop=True))
    sched_out = force_str_columns(sched_out)
    sched_out.insert(0, "season", SEASON)
    sched_out.insert(1, "round", sched_out["RoundNumber"].astype("int64"))
    sched_out.insert(2, "event_name", sched_out["EventName"].astype(str))
    sched_out.insert(3, "session_type", "EVENT")
    for rnd, rows in sched_out.groupby("round", sort=True):
        write_part(rows.reset_index(drop=True), part_path(int(rnd), "EVENT", "schedule"))

    tasks: list[tuple[int, str, str]] = []
    for _, event in schedule.iterrows():
        rnd = int(event["RoundNumber"])
        name = str(event["EventName"])
        for code in sessions_for(str(event["EventFormat"]), rnd):
            tasks.append((rnd, name, code))

    outcomes: dict[str, list[str]] = {"done": [], "cached": [], "skipped": [], "failed": []}
    notes: list[str] = []

    # disable=None turns the bar off when stdout is not a terminal, so piped
    # or logged runs do not fill up with redraw spam.
    bar = tqdm(tasks, desc="sessions", unit="session", disable=None)
    for rnd, name, code in bar:
        bar.set_postfix_str(f"R{rnd:02d} {code}")
        status, detail = ingest_session(fastf1, rnd, name, code, args.force)
        label = f"R{rnd:02d} {code} ({name})"
        outcomes[status].append(label)
        if status == "skipped":
            tqdm.write(f"WARNING  skipping {label}: {detail}")
        elif status == "failed":
            tqdm.write(f"ERROR    failed {label}: {detail}")
        elif detail:
            tqdm.write(f"WARNING  {label}: {detail}")
        if detail and status == "done":
            notes.append(f"{label}: {detail}")
    bar.close()

    # ------------------------------------------------------------------
    # Concatenate every part on disk - including parts from earlier runs -
    # into the five final tables, and record their dtypes.
    # ------------------------------------------------------------------
    frames: dict[str, pd.DataFrame] = {}
    for table in TABLES:
        frame = concat_parts(table)
        if frame.empty and not frame.columns.size:
            log.warning("no parts found for %s; skipping %s.csv", table, table)
            continue
        frames[table] = frame

    residual = backfill_driver_keys(frames)

    dtypes: dict[str, dict[str, str]] = {}
    row_counts: dict[str, int] = {}
    for table, frame in frames.items():
        frame.to_csv(RAW / f"{table}.csv", index=False, float_format="%.6f")
        dtypes[table] = {str(c): str(d) for c, d in frame.dtypes.items()}
        row_counts[table] = len(frame)

        leftover = [c for c, d in frame.dtypes.items() if str(d).startswith("timedelta")]
        if leftover:
            log.error("%s.csv still has timedelta columns: %s", table, leftover)

    with (RAW / "dtypes.json").open("w") as fh:
        json.dump(dtypes, fh, indent=2, sort_keys=True)

    _summary(row_counts, outcomes, notes, residual)
    return 1 if outcomes["failed"] else 0


def _summary(row_counts, outcomes, notes, residual) -> None:
    print(f"\n{'=' * 68}\n{SEASON} ingest summary\n{'=' * 68}")

    print("\nrows per file:")
    for table in TABLES:
        if table in row_counts:
            print(f"  {table + '.csv':<20} {row_counts[table]:>8,} rows")
        else:
            print(f"  {table + '.csv':<20} {'not written':>8}")

    covered = sorted(
        {
            int(p.name.split("_", 1)[0])
            for p in PARTS.glob("round_*/*_laps.csv")
            if p.name.split("_", 1)[0].isdigit()
        }
    )
    print(f"\nrounds covered (laps on disk): {covered if covered else 'none'}")

    print(
        f"\nsessions: {len(outcomes['done'])} downloaded, "
        f"{len(outcomes['cached'])} already cached, "
        f"{len(outcomes['skipped'])} skipped, "
        f"{len(outcomes['failed'])} failed"
    )
    for title, key in (("skipped", "skipped"), ("failed", "failed")):
        if outcomes[key]:
            print(f"\n{title}:")
            for label in outcomes[key]:
                print(f"  - {label}")
    if notes:
        print("\npartial sessions:")
        for note in notes:
            print(f"  - {note}")

    if residual:
        print("\nrows with no DriverId even after season-wide backfill:")
        for table, count in residual.items():
            print(f"  - {table}.csv: {count:,} (drivers that appear in no classified session)")

    print(f"\nwrote {RAW}")
    print("read it back with:  from src.io_utils import load_raw; load_raw('laps')")


if __name__ == "__main__":
    raise SystemExit(main())
