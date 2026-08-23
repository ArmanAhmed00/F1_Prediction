"""Plot styling for the 2026 project: official team colours, consistent output.

Charts are never written to disk as images. Every figure is drawn from data at
render time by whichever consumer needs it, so the repository carries the
numbers rather than pixels.

Charts should read like broadcast graphics rather than matplotlib defaults, so
colour comes from fastf1.plotting's official 2026 constants rather than from
anything hand-picked here.  Those lookups need a loaded Session for their
driver/team roster, which `reference_session()` provides and caches.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
CACHE = BASE / "cache"

# Fallback palette for anything that is not a team or driver.
INK = "#16181d"
MUTED = "#6b7280"
GRID = "#d8dbe0"
ACCENT = "#0072b2"
WARN = "#d55e00"
_SEQ = ["#0072b2", "#d55e00", "#009e73", "#cc79a7", "#56b4e9", "#e69f00", "#7f3f98"]

_SESSION = None


def apply_style() -> None:
    """Shared matplotlib defaults so every chart looks the same."""
    mpl.rcParams.update(
        {
            "figure.dpi": 110,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.titlepad": 10,
            "axes.labelsize": 11,
            "axes.labelcolor": INK,
            "axes.edgecolor": GRID,
            "axes.linewidth": 1.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "grid.alpha": 0.7,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "lines.linewidth": 2.0,
            "lines.markersize": 6,
            "text.color": INK,
            "axes.prop_cycle": mpl.cycler(color=_SEQ),
        }
    )


def reference_session(rnd: int = 12, session_type: str = "Q"):
    """A loaded Session - fastf1's colour lookups need one for the roster.

    Results only; laps and telemetry aren't needed just to get a colour.
    """
    global _SESSION
    if _SESSION is None:
        import fastf1

        fastf1.Cache.enable_cache(str(CACHE))
        fastf1.set_log_level("ERROR")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            session = fastf1.get_session(2026, rnd, session_type)
            session.load(laps=False, telemetry=False, weather=False, messages=False)
        _SESSION = session
    return _SESSION


def team_color(team_name: str, fallback: str = MUTED) -> str:
    """Official 2026 constructor colour, by team name."""
    import fastf1.plotting as f1plt

    try:
        return f1plt.get_team_color(str(team_name), session=reference_session())
    except (ValueError, KeyError):
        return fallback


def team_colors(team_names) -> list[str]:
    return [team_color(name) for name in team_names]


def driver_style(abbreviation: str, keys=("color", "linestyle")) -> dict:
    """Colour plus a teammate-distinguishing linestyle, per fastf1's scheme."""
    import fastf1.plotting as f1plt

    try:
        return f1plt.get_driver_style(
            str(abbreviation), style=list(keys), session=reference_session()
        )
    except (ValueError, KeyError):
        return {"color": MUTED, "linestyle": "solid"}


def sorted_driver_legend(ax, **kwargs):
    """Legend ordered by team, then driver."""
    import fastf1.plotting as f1plt

    try:
        return f1plt.add_sorted_driver_legend(ax, reference_session(), **kwargs)
    except Exception:
        return ax.legend(**kwargs)


def annotate(ax, text: str, loc: str = "upper left", **kwargs) -> None:
    """Small boxed note on an axes - correlations, sample sizes, that sort of thing."""
    positions = {
        "upper left": (0.02, 0.98, "left", "top"),
        "upper right": (0.98, 0.98, "right", "top"),
        "lower left": (0.02, 0.02, "left", "bottom"),
        "lower right": (0.98, 0.02, "right", "bottom"),
    }
    x, y, ha, va = positions[loc]
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=10,
        color=INK,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "edgecolor": GRID, "alpha": 0.92},
        **kwargs,
    )


def to_markdown_table(frame: pd.DataFrame, index: bool = False) -> str:
    """DataFrame -> markdown table.

    Done by hand instead of DataFrame.to_markdown() so we don't need tabulate,
    which is an optional pandas extra and often missing.
    """
    df = frame.reset_index() if index else frame
    columns = [str(c) for c in df.columns]
    body = [
        "| " + " | ".join("" if pd.isna(v) else str(v) for v in record) + " |"
        for record in df.itertuples(index=False, name=None)
    ]
    return "\n".join(
        [
            "| " + " | ".join(columns) + " |",
            "| " + " | ".join("---" for _ in columns) + " |",
            *body,
        ]
    )


def team_order(table: pd.DataFrame, by: str = "quali_pct_off_pole") -> list[str]:
    """Teams fastest-first, so every chart orders them the same way."""
    ranked = table.groupby("team_name")[by].median().sort_values()
    return list(ranked.index)


__all__ = [
    "ACCENT",
    "GRID",
    "INK",
    "MUTED",
    "WARN",
    "annotate",
    "apply_style",
    "driver_style",
    "reference_session",
    "sorted_driver_legend",
    "team_color",
    "team_colors",
    "team_order",
    "to_markdown_table",
]
