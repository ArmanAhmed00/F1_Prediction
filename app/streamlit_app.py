"""Streamlit demo for the 2026 F1 predictions — one page per race.

No network calls, and it never imports fastf1. Everything comes from local
files - predictions/*.json, predictions/*.csv, reports/* and
data/processed/model_table.csv. Charts are drawn by plotly from those numbers,
not loaded as images. Team colours come from the team_color column that
predict_race.py already resolved and saved.

That's deliberate. A live API call failing during a presentation isn't
recoverable and there's no reason to risk it.

Only src.simulate gets imported (numpy and pandas), because the simulator tab
re-runs the simulation live. Nothing is fitted here.

Structure: race pages are discovered from the prediction files, not listed, so
the week Madrid is predicted its page appears on its own. The method pages
(How it works, Model evidence) are shared, because the model is shared - it is
frozen across all three races, which is the whole point of the exercise.

    uv run streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

from app.race_view import (  # noqa: E402
    BACKTEST_CSV,
    CALIBRATION_CSV,
    IMPORTANCE_CSV,
    LOG_CSV,
    PLOTLY_LAYOUT,
    SELECTION_JSON,
    RaceSpec,
    load_csv,
    load_json,
    missing,
    race_has_run,
    race_specs,
    render_race,
)

# Fail at startup, not halfway through a render, if someone adds an import that
# breaks the no-network rule.
assert "fastf1" not in sys.modules, "streamlit_app must never import fastf1"

st.set_page_config(page_title="F1 2026 — Race Predictions", page_icon="🏁", layout="wide")

REPORTS = BASE / "reports"
V2_BACKLOG = REPORTS / "v2_backlog.md"

SPECS = race_specs()


# --------------------------------------------------------------------------
# Overview
# --------------------------------------------------------------------------
def page_overview() -> None:
    st.title("🏁 F1 2026 — locked race predictions")
    st.caption(
        "One frozen model, one page per race. Every prediction was written to disk "
        "and committed before its race started."
    )

    if not SPECS:
        st.warning(
            "No predictions found in `predictions/`. Generate one with:\n\n"
            "```\nuv run python src/predict_race.py --round 13 --label monza\n```"
        )
        return

    selection = load_json(str(SELECTION_JSON))

    st.subheader("The experiment")
    if selection and selection.get("selection_frozen"):
        rounds = selection.get("test_rounds", [])
        fold_range = f"{min(rounds)}–{max(rounds)}" if rounds else "—"
        still = selection.get("still_best_fitted_candidate")
        st.success(
            f"**The model is frozen.** `{selection['selected_model']}` was selected at "
            f"the {selection.get('frozen_at', 'round 12 backtest')} and is **not** "
            "re-selected between races. Only the data changes: each week one more "
            "completed race enters training.\n\n"
            f"It is currently re-scored over walk-forward folds {fold_range}, where it "
            + ("**remains** the best of the fitted candidates."
               if still else
               "is **no longer** the best fitted candidate (`"
               f"{selection.get('best_fitted_candidate_on_these_folds')}` now scores "
               "better) — and is used anyway, because swapping now would be selection "
               "on the test set.")
        )
    st.markdown(
        "If the model were re-tuned between races, the three race predictions would be "
        "three different models with a sample size of one each, which would demonstrate "
        "nothing. Anything believed to be wrong with it goes in "
        "`reports/v2_backlog.md` and becomes an explicitly labelled v2, evaluated on the "
        "full walk-forward and run **alongside** v1 — never swapped in silently."
    )

    st.subheader("Races")
    cards = st.columns(len(SPECS))
    for col, spec in zip(cards, SPECS):
        prediction = load_json(str(spec.json_path))
        if prediction is None:
            continue
        winner = prediction["predicted_winner"]
        done = race_has_run(spec.round)
        with col:
            st.markdown(f"### {spec.icon} {spec.short_title}")
            st.caption(
                f"Round {spec.round} · {spec.circuit} · "
                + ("**raced**" if done else "**not yet raced**")
            )
            st.metric(
                "Predicted winner",
                winner["abbreviation"],
                f"{winner['p_win'] * 100:.1f}% · {winner['team_name']}",
            )
            st.caption(
                "Podium: "
                + " · ".join(p["abbreviation"] for p in prediction["predicted_podium"])
            )

    st.subheader("Prediction log")
    log = load_csv(str(LOG_CSV))
    if log is None:
        st.info(
            "`predictions/prediction_log.csv` does not exist yet. It is appended to by "
            "`src/predict_race.py` — one row per locked prediction. Predictions made "
            "before the log existed are not in it."
        )
    else:
        shown = log.copy()
        if "p_win" in shown.columns:
            shown["p_win"] = (shown["p_win"] * 100).map("{:.1f}%".format)
        if "git_commit_sha" in shown.columns:
            shown["git_commit_sha"] = shown["git_commit_sha"].fillna("").astype(str).str[:10]
        st.dataframe(shown, width="stretch", hide_index=True)
        st.caption(
            "Append-only. The commit SHA is the repository state the prediction was "
            "generated from, so any row can be regenerated exactly. Predictions locked "
            "before the log existed are absent from it — the JSON on disk is the record."
        )

    if V2_BACKLOG.exists():
        with st.expander("What is believed to be wrong with the model (v2 backlog)"):
            st.markdown(V2_BACKLOG.read_text())


# --------------------------------------------------------------------------
# How it works — about the method, not about any one race
# --------------------------------------------------------------------------
def page_how_it_works() -> None:
    st.title("How it works")

    latest = SPECS[-1] if SPECS else None
    prediction = load_json(str(latest.json_path)) if latest else None

    st.markdown(
        """
### Two steps, and the second one is the interesting one

**Step 1 — predict where each car finishes.** The model is a regression. It takes
14 numbers about a driver and their weekend (how fast they qualified, how they
have been going recently, how reliable the car is) and predicts a finishing
position: 1st, 4th, 12th.

**Step 2 — simulate the race 20,000 times.** Each driver's predicted position is
nudged up or down by a random amount, a dice roll decides whether their car
breaks, and the field is sorted. Do that 20,000 times and count: if a driver comes
first in 3,460 of them, their probability of winning is 17.3%.

### Why not just train a model to predict "who wins"?

Because there have only been a couple of dozen winners all season across a few
hundred driver-sessions. Fitting 14 features to that many positive examples
memorises those races instead of learning anything. Predicting *finishing
position* uses every single row: a driver coming ninth is as informative as a
driver winning.

### The property this buys you

Because win and podium chances are counted from the *same* simulated races,
**a driver's chance of winning can never exceed their chance of a podium** — any
race they win is a race they finished in the top three. The win probabilities also
add up to exactly 100%, because exactly one car wins each simulation.

Two separately trained models would guarantee neither, and would happily tell you
a driver has a 30% chance of winning and a 20% chance of a podium — which is
impossible.

### Where the starting grid comes from

FastF1 only publishes `GridPosition` inside *race* results, which do not exist
before a race. Qualifying order is the obvious stand-in and it is simply wrong
wherever a penalty applies — at Monza four drivers moved, three of them by more
than ten places. So the official grid is transcribed by hand into
`data/manual/grid_R{round}.csv`, and the feature build asserts it is a
permutation of 1..n before it will use it. Without that file the build falls back
to qualifying order and says so, loudly.
        """
    )

    if prediction is None:
        return

    st.subheader("The 14 features")
    definitions = prediction["features"]["definitions"]
    st.dataframe(
        pd.DataFrame({"Feature": list(definitions), "Definition": list(definitions.values())}),
        width="stretch", hide_index=True, height=530,
    )
    excluded = prediction["features"].get("excluded", {})
    if excluded:
        st.caption("Excluded: " + "; ".join(f"`{k}` — {v}" for k, v in excluded.items()))

    st.subheader("Permutation importance")
    importance = load_csv(str(IMPORTANCE_CSV))
    if importance is None:
        missing(IMPORTANCE_CSV, "src/backtest.py")
        return

    sel = importance[importance["model"] == prediction["model"]["name"]].sort_values("importance")
    fig = go.Figure(
        go.Bar(
            x=sel["importance"], y=sel["feature"], orientation="h",
            marker=dict(color=["#0072b2" if v > 0 else "#9aa0a6" for v in sel["importance"]]),
            hovertemplate="<b>%{y}</b><br>+%{x:.3f} MAE when shuffled<extra></extra>",
        )
    )
    fig.update_layout(
        title="How much worse the model gets when each feature is shuffled",
        xaxis_title="increase in mean absolute error (positions)",
        height=520, showlegend=False, **PLOTLY_LAYOUT,
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Measured out-of-sample on each walk-forward test fold. Impurity-based "
        "importance is deliberately not used: it is computed in-sample and is biased "
        "toward high-cardinality features. Bars at or below zero are features that "
        "are not earning their place."
    )


# --------------------------------------------------------------------------
# Model evidence — the backtest, shared across every race page
# --------------------------------------------------------------------------
def page_model_evidence() -> None:
    st.title("Model evidence")

    backtest = load_csv(str(BACKTEST_CSV), index_col=0)
    selection = load_json(str(SELECTION_JSON))
    calibration = load_csv(str(CALIBRATION_CSV))

    if backtest is None or selection is None:
        missing(BACKTEST_CSV, "src/backtest.py")
        return

    selected = selection["selected_model"]
    rounds = selection.get("test_rounds", [])
    fold_range = f"{min(rounds)}–{max(rounds)}" if rounds else "—"
    n_folds = len(rounds)

    if selection["beats_baselines"]:
        st.success(
            f"**`{selected}` beat every baseline on win log loss** across the "
            f"{n_folds} walk-forward races."
        )
    else:
        grid_ll = float(backtest.loc["D_grid_baseline", "win_log_loss"])
        sel_ll = float(backtest.loc[selected, "win_log_loss"])
        st.warning(
            f"**`{selected}` did not beat every baseline.** It beats B1, B2 and B3, "
            f"but B4 — assume every driver finishes where they started — scores "
            f"{grid_ll:.4f} against `{selected}`'s {sel_ll:.4f} on win log loss. "
            "This is reported as found; nothing was tuned to reverse it."
        )

    st.subheader(f"Walk-forward backtest, rounds {fold_range}")
    st.caption(
        "Train on everything before round k, predict round k's race, score it, move on. "
        "Sprints are used for training but never as a test fold. The model is re-scored "
        "here as folds are added — it is **not** re-selected."
    )
    display = backtest.drop(columns=[c for c in ["label"] if c in backtest.columns])
    st.dataframe(
        display.style.format("{:.4f}", na_rep="—")
        .apply(lambda row: ["background-color: #fff3cd" if row.name == selected else ""
                            for _ in row], axis=1),
        width="stretch",
    )
    st.caption(
        "Highlighted row is the frozen model in use. Lower is better for log loss and "
        "Brier; higher is better for top-1 hit rate and Spearman rho. All "
        "probabilities are clipped to [1e-6, 1−1e-6] before scoring, because B1 and "
        "B2 assert probability 1.0 and would otherwise score infinite log loss when "
        "wrong. Accuracy is not shown: at ~5% positives, predicting nobody wins "
        "scores 95%."
    )

    # Each locked prediction was generated against the backtest as it stood that
    # week. Saying so stops the current table being read as evidence the older
    # predictions never had.
    stale = []
    for spec in SPECS:
        payload = load_json(str(spec.json_path))
        if payload and payload.get("backtest", {}).get("test_rounds", rounds) != rounds:
            stale.append((spec, payload))
    if stale:
        lines = [
            f"- **{spec.short_title}** was locked against folds "
            f"{min(p['backtest']['test_rounds'])}–{max(p['backtest']['test_rounds'])} "
            f"({len(p['backtest']['test_rounds'])} races), sigma "
            f"{p['simulation']['residual_sigma']:.3f}"
            for spec, p in stale
        ]
        st.info(
            "The table above is the backtest **as it stands now**. Earlier predictions "
            "were locked against fewer folds and did not have access to these numbers:\n\n"
            + "\n".join(lines)
        )

    st.subheader("Calibration")
    if calibration is None:
        missing(CALIBRATION_CSV, "src/backtest.py")
        return

    cols = st.columns(2)
    for col, target, title in zip(cols, ["win", "podium"],
                                  ["Win probability", "Podium probability"]):
        d = calibration[calibration["target"] == target]
        top = float(max(d["mean_predicted"].max(), d["observed"].max())) * 1.15 + 0.02
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[0, top], y=[0, top], mode="lines", name="perfect calibration",
            line=dict(dash="dot", color="#9aa0a6", width=1.5), hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=d["mean_predicted"], y=d["observed"], mode="lines+markers+text",
            name=selected, line=dict(color="#0072b2", width=2.5),
            marker=dict(size=9), text=[f"n={int(v)}" for v in d["n"]],
            textposition="bottom right", textfont=dict(size=10, color="#6b7280"),
            hovertemplate="predicted %{x:.3f}<br>observed %{y:.3f}<extra></extra>",
        ))
        fig.update_layout(
            title=title, xaxis_title="mean predicted probability",
            yaxis_title="observed frequency", height=430,
            xaxis=dict(range=[0, top]), yaxis=dict(range=[0, top]),
            showlegend=True, **PLOTLY_LAYOUT,
        )
        col.plotly_chart(fig, width="stretch")
    st.caption(
        "Drawn live from `reports/calibration.csv`. Points below the diagonal "
        f"indicate overconfidence. With {n_folds} races the bins are thin — the n "
        "labels show how thin — so this is a check that nothing is grossly "
        "miscalibrated, not proof of calibration."
    )


# --------------------------------------------------------------------------
# Navigation. One page per race, discovered from disk.
# --------------------------------------------------------------------------
def _race_page(spec: RaceSpec):
    """Bind the spec now; st.Page wants a zero-argument callable."""

    def page() -> None:
        render_race(spec)

    page.__name__ = f"page_race_{spec.label}"
    return page


race_pages = [
    st.Page(
        _race_page(spec),
        title=spec.short_title,
        icon=spec.icon,
        url_path=f"race-{spec.label}",
    )
    for spec in SPECS
]

pages: dict[str, list] = {
    "Season": [st.Page(page_overview, title="Overview", icon="🏁", default=True)],
}
if race_pages:
    pages["Races"] = race_pages
pages["Method"] = [
    st.Page(page_how_it_works, title="How it works", icon="📘", url_path="how-it-works"),
    st.Page(page_model_evidence, title="Model evidence", icon="📊", url_path="evidence"),
]

st.navigation(pages).run()
