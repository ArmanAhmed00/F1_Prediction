"""Shared rendering for the per-race pages.

One race page is one locked prediction. The pages are discovered from
``predictions/2026_R*_*.json`` rather than listed by hand, so the week Madrid is
predicted its page appears without anyone editing this file - the same property
``predict_race.py`` has.

Everything a page shows comes from local files. No network calls, and fastf1 is
never imported; team colours were resolved and saved by ``predict_race.py``
precisely so this module never needs it.

Not a Streamlit page itself - ``streamlit_app.py`` builds the navigation and
calls in here.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

from src.simulate import simulate_race  # noqa: E402

PREDICTIONS = BASE / "predictions"
REPORTS = BASE / "reports"
BACKTEST_CSV = REPORTS / "backtest_results.csv"
SELECTION_JSON = REPORTS / "model_selection.json"
IMPORTANCE_CSV = REPORTS / "permutation_importance.csv"
CALIBRATION_CSV = REPORTS / "calibration.csv"
LOG_CSV = PREDICTIONS / "prediction_log.csv"
RAW_RESULTS = BASE / "data" / "raw" / "results.csv"

# predictions/2026_R13_monza.json -> round 13, label "monza"
PREDICTION_PATTERN = re.compile(r"^(?P<season>\d{4})_R(?P<round>\d+)_(?P<label>.+)$")

# Cosmetic only. An unknown label still gets a page, just a generic flag.
ICONS = {"dutch": "🇳🇱", "monza": "🇮🇹", "madrid": "🇪🇸"}

PLOTLY_LAYOUT = dict(
    template="plotly_white",
    font=dict(size=13),
    margin=dict(l=10, r=10, t=50, b=10),
    hoverlabel=dict(font_size=13),
)


@dataclass(frozen=True)
class RaceSpec:
    """Everything needed to render one race page, resolved from disk."""

    round: int
    label: str
    race: str
    circuit: str
    json_path: Path
    csv_path: Path

    @property
    def icon(self) -> str:
        return ICONS.get(self.label, "🏁")

    @property
    def short_title(self) -> str:
        return self.race.replace("2026 ", "").replace("Grand Prix", "GP")

    @property
    def pre_registration(self) -> Path | None:
        """reports/{label}_pre_registration.md, if one was written."""
        path = REPORTS / f"{self.label}_pre_registration.md"
        return path if path.exists() else None

    @property
    def evaluation(self) -> Path | None:
        """reports/{label}_gp_evaluation.md, written after the race."""
        path = REPORTS / f"{self.label}_gp_evaluation.md"
        return path if path.exists() else None


# --------------------------------------------------------------------------
# Loading. Every read is cached and every miss is a named warning, never a
# stack trace in front of an audience.
# --------------------------------------------------------------------------
def missing(path: Path, script: str) -> None:
    st.warning(
        f"**Missing file:** `{path.relative_to(BASE)}`\n\n"
        f"Generate it by running:\n\n```\nuv run python {script}\n```"
    )


@st.cache_data
def load_json(path_str: str):
    path = Path(path_str)
    return json.loads(path.read_text()) if path.exists() else None


@st.cache_data
def load_csv(path_str: str, index_col=None):
    path = Path(path_str)
    return pd.read_csv(path, index_col=index_col) if path.exists() else None


@st.cache_data
def load_text(path_str: str) -> str | None:
    path = Path(path_str)
    return path.read_text() if path.exists() else None


@st.cache_data
def discover_races() -> list[dict]:
    """Every locked prediction on disk, oldest round first.

    Returns plain dicts because cache_data has to be able to hash the result;
    the caller rebuilds them into RaceSpec.
    """
    found: list[dict] = []
    for json_path in sorted(PREDICTIONS.glob("*.json")):
        match = PREDICTION_PATTERN.match(json_path.stem)
        if not match:
            continue
        csv_path = json_path.with_suffix(".csv")
        if not csv_path.exists():
            continue
        try:
            payload = json.loads(json_path.read_text())
        except json.JSONDecodeError:
            continue
        found.append(
            {
                "round": int(payload.get("round", match.group("round"))),
                "label": str(payload.get("label", match.group("label"))),
                "race": str(payload.get("race", f"Round {match.group('round')}")),
                "circuit": str(payload.get("circuit") or "—"),
                "json_path": str(json_path),
                "csv_path": str(csv_path),
            }
        )
    return sorted(found, key=lambda r: r["round"])


def race_specs() -> list[RaceSpec]:
    return [
        RaceSpec(
            round=r["round"],
            label=r["label"],
            race=r["race"],
            circuit=r["circuit"],
            json_path=Path(r["json_path"]),
            csv_path=Path(r["csv_path"]),
        )
        for r in discover_races()
    ]


@st.cache_data
def race_has_run(rnd: int) -> bool:
    """True once a classified race for this round exists in the raw results."""
    if not RAW_RESULTS.exists():
        return False
    cols = pd.read_csv(RAW_RESULTS, nrows=0).columns
    if not {"round", "session_type"}.issubset(cols):
        return False
    res = pd.read_csv(RAW_RESULTS, usecols=["round", "session_type"])
    return not res[(res["round"] == rnd) & (res["session_type"] == "R")].empty


@st.cache_data
def load_actual_results(rnd: int) -> pd.DataFrame:
    res = pd.read_csv(RAW_RESULTS)
    race = res[(res["round"] == rnd) & (res["session_type"] == "R")]
    return race[["DriverId", "Abbreviation", "Position", "ClassifiedPosition",
                 "Status", "GridPosition", "Laps"]].copy()


# --------------------------------------------------------------------------
# Charts and tables
# --------------------------------------------------------------------------
def probability_bars(frame: pd.DataFrame, column: str, title: str, xlabel: str) -> go.Figure:
    """Horizontal bars, sorted, in team colours, labelled as percentages."""
    data = frame.sort_values(column, ascending=True)
    fig = go.Figure(
        go.Bar(
            x=data[column] * 100,
            y=data["abbreviation"],
            orientation="h",
            marker=dict(color=data["team_color"], line=dict(color="white", width=1)),
            text=[f"{v * 100:.1f}%" for v in data[column]],
            textposition="outside",
            customdata=data[["team_name", "grid_position"]],
            hovertemplate=(
                "<b>%{y}</b><br>%{customdata[0]}<br>"
                "Grid P%{customdata[1]:.0f}<br>" + xlabel + ": %{x:.2f}%<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title=xlabel,
        yaxis_title="",
        height=max(420, 26 * len(data)),
        showlegend=False,
        **PLOTLY_LAYOUT,
    )
    fig.update_xaxes(range=[0, float(data[column].max() * 100) * 1.18])
    return fig


def grid_versus_pace(frame: pd.DataFrame, race_name: str) -> go.Figure:
    """Grid slot against team season pace - the tension the model has to resolve.

    Lower is faster on both axes, so anything far from the diagonal is a car
    starting well out of position relative to how quick it has been all year.
    """
    data = frame.copy()
    pace_rank = data["team_form_pace_ewm"].rank(method="dense")
    fig = go.Figure(
        go.Scatter(
            x=data["grid_position"],
            y=data["team_form_pace_ewm"],
            mode="markers+text",
            text=data["abbreviation"],
            textposition="top center",
            textfont=dict(size=10),
            marker=dict(
                size=data["p_win"] * 260 + 9,
                color=data["team_color"],
                line=dict(color="#444", width=1),
            ),
            customdata=data[["team_name", "p_win", "p_podium"]].assign(rank=pace_rank),
            hovertemplate=(
                "<b>%{text}</b><br>%{customdata[0]}<br>"
                "Grid P%{x:.0f}<br>team pace %{y:.3f}%<br>"
                "P(win) %{customdata[1]:.1%}<br>P(podium) %{customdata[2]:.1%}"
                "<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title=f"Grid position vs season pace — {race_name}",
        xaxis_title="grid position (1 = pole)",
        yaxis_title="team_form_pace_ewm (% off pole, lower = faster)",
        height=520,
        showlegend=False,
        **PLOTLY_LAYOUT,
    )
    return fig


def percent_table(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame[["abbreviation", "team_name", "grid_position",
                 "predicted_position", "p_win", "p_podium"]].copy()
    out["grid_position"] = out["grid_position"].astype(int)
    out["predicted_position"] = out["predicted_position"].round(2)
    out["p_win"] = (out["p_win"] * 100).map("{:.1f}%".format)
    out["p_podium"] = (out["p_podium"] * 100).map("{:.1f}%".format)
    out.columns = ["Driver", "Team", "Grid", "Predicted position", "P(win)", "P(podium)"]
    out.index = range(1, len(out) + 1)
    return out


# --------------------------------------------------------------------------
# Tab bodies
# --------------------------------------------------------------------------
def _tab_prediction(spec: RaceSpec, prediction: dict, pred_table: pd.DataFrame) -> None:
    winner = prediction["predicted_winner"]
    podium = prediction["predicted_podium"]
    # Row order is the predicted order (the CSV is written sorted by p_win),
    # so the pole sitter's rank is just their position in it.
    pole = pred_table.loc[pred_table["grid_position"].idxmin()]
    pole_rank = pred_table["driver_id"].tolist().index(pole["driver_id"]) + 1

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Predicted winner", winner["abbreviation"], winner["team_name"])
    c2.metric("Win probability", f"{winner['p_win'] * 100:.1f}%",
              help=f"Share of {prediction['simulation']['n_sims']:,} simulated races "
                   "this driver finished first.")
    c3.metric("Predicted podium", " · ".join(p["abbreviation"] for p in podium),
              f"{podium[0]['p_podium'] * 100:.0f}% / {podium[1]['p_podium'] * 100:.0f}%"
              f" / {podium[2]['p_podium'] * 100:.0f}% podium chance")
    c4.metric("Pole sitter", pole["abbreviation"],
              f"model ranks P{pole_rank} · {pole['p_win'] * 100:.1f}% win",
              delta_color="off",
              help="Where the model puts the driver starting from pole. A large gap "
                   "means the model is disagreeing with the grid.")

    st.info(
        f"No driver is above {pred_table['p_win'].max() * 100:.0f}%. That is the model "
        "being honest, not vague: its out-of-sample error is "
        f"{prediction['simulation']['residual_sigma']:.1f} finishing positions, so the "
        "simulation spreads the outcomes accordingly."
    )

    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            probability_bars(pred_table, "p_win", "Probability of winning", "P(win)"),
            width="stretch", key=f"{spec.label}_pwin",
        )
    with right:
        st.plotly_chart(
            probability_bars(pred_table, "p_podium", "Probability of a podium", "P(podium)"),
            width="stretch", key=f"{spec.label}_ppodium",
        )

    if "team_form_pace_ewm" in pred_table.columns:
        st.subheader("Grid against pace")
        st.plotly_chart(
            grid_versus_pace(pred_table, spec.short_title),
            width="stretch", key=f"{spec.label}_gridpace",
        )
        st.caption(
            "Bubble size is win probability. A car in the bottom-right is quick all "
            "season but starting badly; top-left is the opposite. The model has no "
            "circuit term, so it resolves this tension the same way at every track — "
            "which is exactly the thing worth arguing about at Monza."
        )

    st.subheader("Full ranked prediction")
    st.dataframe(percent_table(pred_table), width="stretch", height=560)

    if prediction.get("caveats"):
        with st.expander("Caveats attached to this prediction", expanded=False):
            for caveat in prediction["caveats"]:
                st.markdown(f"- {caveat}")


def _tab_simulator(spec: RaceSpec, prediction: dict, pred_table: pd.DataFrame) -> None:
    st.subheader("How certainty changes the answer")
    fitted_sigma = float(prediction["simulation"]["residual_sigma"])

    controls, chart = st.columns([1, 2.6])
    with controls:
        sigma = st.slider(
            "sigma — race-day uncertainty (finishing positions)",
            min_value=0.5, max_value=6.0, value=float(round(fitted_sigma, 2)), step=0.1,
            help="The standard deviation of the model's out-of-sample error.",
            key=f"{spec.label}_sigma",
        )
        n_sims = st.slider("number of simulations", 1_000, 50_000, 20_000, step=1_000,
                           key=f"{spec.label}_nsims")
        st.metric("Fitted sigma (from backtest)", f"{fitted_sigma:.2f}")
        if abs(sigma - fitted_sigma) < 0.05:
            st.success("You are at the fitted value — this is the locked prediction.")
        elif sigma < fitted_sigma:
            st.info("Below the fitted value: **more confident than the evidence supports.**")
        else:
            st.info("Above the fitted value: more cautious than the backtest requires.")

    sims = simulate_race(
        pred_table["predicted_position"].to_numpy(),
        sigma=sigma,
        dnf_probs=pred_table["dnf_rate_shrunk"].to_numpy(),
        n_sims=int(n_sims),
        seed=42,
    )
    live = pred_table.copy()
    live["p_win"] = sims["p_win"].to_numpy()
    live["p_podium"] = sims["p_podium"].to_numpy()

    with chart:
        st.plotly_chart(
            probability_bars(
                live, "p_win",
                f"Win probability at sigma = {sigma:.2f} ({int(n_sims):,} simulations)",
                "P(win)",
            ),
            width="stretch", key=f"{spec.label}_sim",
        )

    n_folds = len(prediction.get("backtest", {}).get("test_rounds", []))
    st.caption(
        "**What you are looking at.** Sigma is how wrong the model expects to be. Drag it "
        "down and the field separates — the favourite's probability climbs and the model "
        "looks decisive. Drag it up and everything flattens toward a uniform 1-in-22, "
        "because if you cannot predict finishing position to better than six places, you "
        "cannot tell anyone much about who wins. The fitted value, "
        f"**{fitted_sigma:.2f} positions**, is not a choice — it is measured from how far "
        f"off this model actually was across {n_folds} held-out races. That is why the "
        "locked prediction tops out near "
        f"{prediction['predicted_winner']['p_win'] * 100:.0f}% rather than at something "
        "more dramatic."
    )

    top_now = live.nlargest(5, "p_win")[["abbreviation", "team_name", "p_win", "p_podium"]].copy()
    top_now["p_win"] = (top_now["p_win"] * 100).map("{:.1f}%".format)
    top_now["p_podium"] = (top_now["p_podium"] * 100).map("{:.1f}%".format)
    top_now.columns = ["Driver", "Team", "P(win)", "P(podium)"]
    st.dataframe(top_now, width="stretch", hide_index=True)


def _tab_result(spec: RaceSpec, prediction: dict, pred_table: pd.DataFrame) -> None:
    import numpy as np
    from scipy.stats import spearmanr

    actual = load_actual_results(spec.round)
    merged = pred_table.merge(actual, left_on="driver_id", right_on="DriverId", how="left")
    merged = merged[merged["Position"].notna()].copy()
    merged["actual_position"] = merged["Position"].astype(float)
    merged["error"] = merged["predicted_position"] - merged["actual_position"]

    eps = 1e-6
    y_win = (merged["actual_position"] == 1).to_numpy(float)
    y_pod = (merged["actual_position"] <= 3).to_numpy(float)
    pw = np.clip(merged["p_win"].to_numpy(), eps, 1 - eps)
    pp = np.clip(merged["p_podium"].to_numpy(), eps, 1 - eps)

    winner_row = merged.loc[merged["actual_position"] == 1].iloc[0]
    predicted_winner = pred_table.iloc[0]["abbreviation"]
    predicted_podium = set(pred_table.head(3)["abbreviation"])
    actual_podium = set(merged.nsmallest(3, "actual_position")["abbreviation"])

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Predicted winner", predicted_winner,
              "correct" if predicted_winner == winner_row["abbreviation"] else
              f"actual: {winner_row['abbreviation']}")
    m2.metric("Podium hits", f"{len(predicted_podium & actual_podium)} / 3")
    m3.metric("P assigned to actual winner", f"{winner_row['p_win'] * 100:.1f}%")
    m4.metric("Spearman rho",
              f"{spearmanr(merged['predicted_position'], merged['actual_position']).statistic:.3f}")

    scores = pd.DataFrame({
        "metric": ["win log loss", "win Brier", "podium log loss", "podium Brier",
                   "MAE (positions)"],
        "value": [
            float(-np.mean(y_win * np.log(pw) + (1 - y_win) * np.log(1 - pw))),
            float(np.mean((pw - y_win) ** 2)),
            float(-np.mean(y_pod * np.log(pp) + (1 - y_pod) * np.log(1 - pp))),
            float(np.mean((pp - y_pod) ** 2)),
            float(merged["error"].abs().mean()),
        ],
    })
    st.subheader("Scores on this single race")
    st.dataframe(scores.style.format({"value": "{:.4f}"}), width="stretch", hide_index=True)
    n_folds = len(prediction.get("backtest", {}).get("test_rounds", []))
    st.caption(
        "One race cannot validate a probabilistic model. A 17% favourite winning is "
        "not confirmation, and losing is not refutation — the backtest over "
        f"{n_folds} races is the evidence; this is the demonstration."
    )

    st.subheader("Predicted vs actual")
    side = merged.sort_values("actual_position")[
        ["abbreviation", "team_name", "grid_position", "predicted_position",
         "actual_position", "p_win", "p_podium", "Status"]].copy()
    side["p_win"] = (side["p_win"] * 100).map("{:.1f}%".format)
    side["p_podium"] = (side["p_podium"] * 100).map("{:.1f}%".format)
    side["predicted_position"] = side["predicted_position"].round(2)
    side["actual_position"] = side["actual_position"].astype(int)
    side["grid_position"] = side["grid_position"].astype(int)
    side.columns = ["Driver", "Team", "Grid", "Predicted", "Actual",
                    "P(win)", "P(podium)", "Status"]
    st.dataframe(side, width="stretch", hide_index=True, height=560)

    st.subheader("Biggest misses")
    worst = merged.reindex(merged["error"].abs().sort_values(ascending=False).index).head(5)
    for _, row in worst.iterrows():
        status = str(row.get("Status", ""))
        if status in ("Retired", "Disqualified"):
            why = (f"retired after {int(row.get('Laps') or 0)} laps "
                   f"({status.lower()}) — the model carries a retirement probability "
                   "but cannot know who")
        elif row["error"] > 0:
            why = "finished ahead of prediction — strategy, or inherited places from retirements"
        else:
            why = "finished behind prediction — traffic, strategy, or an incident"
        st.markdown(
            f"- **{row['abbreviation']}** predicted P{row['predicted_position']:.1f}, "
            f"finished P{row['actual_position']:.0f} "
            f"(error {row['error']:+.1f}) — {why}"
        )


def _tab_document(path: Path, blurb: str) -> None:
    text = load_text(str(path))
    if text is None:
        st.warning(f"`{path.relative_to(BASE)}` is missing.")
        return
    st.caption(blurb)
    st.markdown(text)


# --------------------------------------------------------------------------
# The page
# --------------------------------------------------------------------------
def render_race(spec: RaceSpec) -> None:
    prediction = load_json(str(spec.json_path))
    pred_table = load_csv(str(spec.csv_path))

    if prediction is None or pred_table is None:
        st.title(f"{spec.icon} {spec.race}")
        missing(spec.json_path,
                f"src/predict_race.py --round {spec.round} --label {spec.label}")
        return

    has_run = race_has_run(spec.round)

    st.title(f"{spec.icon} {prediction['race']}")
    caption = (
        f"**{prediction.get('circuit', '—')}** · Round {prediction['round']} · "
        f"Model `{prediction['model']['name']}`"
        f"{' (frozen)' if prediction['model'].get('frozen') else ''} · "
        f"Locked **{prediction['generated_at_utc']}** · "
        f"{prediction['simulation']['n_sims']:,} simulations · "
        f"sigma {prediction['simulation']['residual_sigma']:.2f} positions"
    )
    sha = prediction.get("git_commit_sha")
    if sha:
        caption += f" · commit `{sha[:10]}`"
    st.caption(caption)

    if has_run:
        st.success(
            f"This race has been run. The prediction below was locked at "
            f"`{prediction['generated_at_utc']}` and has not been touched since — "
            "see the **Result** tab for how it did."
        )
    else:
        st.warning(
            "This race has **not** been run yet. Everything below is a locked, "
            "committed forecast."
        )

    # Tabs are assembled rather than fixed: a race with no pre-registration and
    # no result does not get empty tabs for them.
    panes: list[tuple[str, callable]] = [
        ("Prediction", lambda: _tab_prediction(spec, prediction, pred_table)),
    ]
    if spec.pre_registration is not None:
        panes.append((
            "Pre-registration",
            lambda: _tab_document(
                spec.pre_registration,
                "Written and committed before the race, so the expectation is on "
                "record independently of the outcome.",
            ),
        ))
    if has_run:
        panes.append(("Result", lambda: _tab_result(spec, prediction, pred_table)))
    if spec.evaluation is not None:
        panes.append((
            "Post-mortem",
            lambda: _tab_document(
                spec.evaluation,
                "Generated by `src/evaluate_race.py`. The prediction JSON is an input "
                "here and cannot be changed by it.",
            ),
        ))
    panes.append(("Simulator", lambda: _tab_simulator(spec, prediction, pred_table)))

    tabs = st.tabs([name for name, _ in panes])
    for tab, (_, body) in zip(tabs, panes):
        with tab:
            body()
