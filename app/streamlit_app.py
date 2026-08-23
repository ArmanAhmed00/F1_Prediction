"""Streamlit demo for the Dutch GP prediction.

No network calls, and it never imports fastf1. Everything comes from local
files - predictions/*.json, predictions/*.csv, reports/*.csv and
data/processed/model_table.csv. Charts are drawn by plotly from those numbers,
not loaded as images. Team colours come from the team_color column that
predict_dutch.py already resolved and saved.

That's deliberate. A live API call failing during a presentation isn't
recoverable and there's no reason to risk it.

Only src.simulate gets imported (numpy and pandas), because tab 4 re-runs the
simulation live. Nothing is fitted here.

    uv run streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

from src.simulate import simulate_race  # noqa: E402

# Fail at startup, not halfway through a render, if someone adds an import that
# breaks the no-network rule.
assert "fastf1" not in sys.modules, "streamlit_app must never import fastf1"

PREDICTIONS = BASE / "predictions"
REPORTS = BASE / "reports"
PRED_JSON = PREDICTIONS / "2026_R12_dutch.json"
PRED_CSV = PREDICTIONS / "2026_R12_dutch.csv"
BACKTEST_CSV = REPORTS / "backtest_results.csv"
SELECTION_JSON = REPORTS / "model_selection.json"
IMPORTANCE_CSV = REPORTS / "permutation_importance.csv"
CALIBRATION_CSV = REPORTS / "calibration.csv"
MODEL_TABLE = BASE / "data" / "processed" / "model_table.csv"
RAW_RESULTS = BASE / "data" / "raw" / "results.csv"

st.set_page_config(
    page_title="F1 2026 — Dutch GP Prediction",
    page_icon="🏁",
    layout="wide",
)

PLOTLY_LAYOUT = dict(
    template="plotly_white",
    font=dict(size=13),
    margin=dict(l=10, r=10, t=50, b=10),
    hoverlabel=dict(font_size=13),
)


# --------------------------------------------------------------------------
# Every read is cached. A missing file gets a warning naming the script that
# produces it, never a stack trace in front of an audience.
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
def race_has_run() -> bool:
    """True once a classified round 12 race exists in the raw results."""
    if not RAW_RESULTS.exists():
        return False
    cols = pd.read_csv(RAW_RESULTS, nrows=0).columns
    if not {"round", "session_type"}.issubset(cols):
        return False
    res = pd.read_csv(RAW_RESULTS, usecols=["round", "session_type"])
    return not res[(res["round"] == 12) & (res["session_type"] == "R")].empty


@st.cache_data
def load_actual_results():
    res = pd.read_csv(RAW_RESULTS)
    race = res[(res["round"] == 12) & (res["session_type"] == "R")]
    return race[["DriverId", "Abbreviation", "Position", "ClassifiedPosition",
                 "Status", "GridPosition", "Laps"]].copy()


prediction = load_json(str(PRED_JSON))
pred_table = load_csv(str(PRED_CSV))
backtest = load_csv(str(BACKTEST_CSV), index_col=0)
selection = load_json(str(SELECTION_JSON))
importance = load_csv(str(IMPORTANCE_CSV))
calibration = load_csv(str(CALIBRATION_CSV))

if prediction is None or pred_table is None:
    st.title("🏁 F1 2026 — Dutch Grand Prix")
    missing(PRED_JSON, "src/predict_dutch.py")
    st.stop()


# --------------------------------------------------------------------------
# Charts
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
# Header
# --------------------------------------------------------------------------
st.title("🏁 " + prediction["race"])
st.caption(
    f"**{prediction['circuit']}** · Round {prediction['round']} · "
    f"Model `{prediction['model']['name']}` · "
    f"Locked **{prediction['generated_at_utc']}** · "
    f"{prediction['simulation']['n_sims']:,} simulations · "
    f"sigma {prediction['simulation']['residual_sigma']:.2f} positions"
)

RACE_DONE = race_has_run()
tab_names = ["Prediction", "How it works", "Model evidence", "Simulator"]
if RACE_DONE:
    tab_names.append("Result")
tabs = st.tabs(tab_names)


# --------------------------------------------------------------------------
# Tab 1 — Prediction
# --------------------------------------------------------------------------
with tabs[0]:
    winner = prediction["predicted_winner"]
    podium = prediction["predicted_podium"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted winner", winner["abbreviation"], winner["team_name"])
    c2.metric("Win probability", f"{winner['p_win'] * 100:.1f}%",
              help="Share of 20,000 simulated races this driver finished first.")
    c3.metric("Predicted podium", " · ".join(p["abbreviation"] for p in podium),
              f"{podium[0]['p_podium'] * 100:.0f}% / {podium[1]['p_podium'] * 100:.0f}%"
              f" / {podium[2]['p_podium'] * 100:.0f}% podium chance")

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
            width="stretch",
        )
    with right:
        st.plotly_chart(
            probability_bars(pred_table, "p_podium", "Probability of a podium", "P(podium)"),
            width="stretch",
        )

    st.subheader("Full ranked prediction")
    st.dataframe(percent_table(pred_table), width="stretch", height=560)

    with st.expander("Caveats attached to this prediction"):
        for caveat in prediction.get("caveats", []):
            st.markdown(f"- {caveat}")


# --------------------------------------------------------------------------
# Tab 2 — How it works
# --------------------------------------------------------------------------
with tabs[1]:
    st.header("How it works")
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

Because there have only been **16 winners all season** across 343 driver-sessions.
Fitting 14 features to 16 examples memorises those 16 races instead of learning
anything. Predicting *finishing position* uses every single row: a driver coming
ninth is as informative as a driver winning.

### The property this buys you

Because win and podium chances are counted from the *same* simulated races,
**a driver's chance of winning can never exceed their chance of a podium** — any
race they win is a race they finished in the top three. The win probabilities also
add up to exactly 100%, because exactly one car wins each simulation.

Two separately trained models would guarantee neither, and would happily tell you
a driver has a 30% chance of winning and a 20% chance of a podium — which is
impossible.
        """
    )

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
    if importance is None:
        missing(IMPORTANCE_CSV, "src/backtest.py")
    else:
        sel = importance[importance["model"] == prediction["model"]["name"]]
        sel = sel.sort_values("importance")
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
# Tab 3 — Model evidence
# --------------------------------------------------------------------------
with tabs[2]:
    st.header("Model evidence")
    if backtest is None or selection is None:
        missing(BACKTEST_CSV, "src/backtest.py")
    else:
        selected = selection["selected_model"]
        beat = selection["beats_baselines"]

        if beat:
            st.success(
                f"**`{selected}` beat every baseline on win log loss** across the six "
                "walk-forward races."
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

        st.subheader("Walk-forward backtest, rounds 6–11")
        display = backtest.drop(columns=[c for c in ["label"] if c in backtest.columns])
        st.dataframe(
            display.style.format("{:.4f}", na_rep="—")
            .apply(lambda row: ["background-color: #fff3cd" if row.name == selected else ""
                                for _ in row], axis=1),
            width="stretch",
        )
        st.caption(
            "Highlighted row is the selected model. Lower is better for log loss and "
            "Brier; higher is better for top-1 hit rate and Spearman rho. All "
            "probabilities are clipped to [1e-6, 1−1e-6] before scoring, because B1 and "
            "B2 assert probability 1.0 and would otherwise score infinite log loss when "
            "wrong. Accuracy is not shown: at ~5% positives, predicting nobody wins "
            "scores 95%."
        )

        st.subheader("Calibration")
        if calibration is None:
            missing(CALIBRATION_CSV, "src/backtest.py")
        else:
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
                "indicate overconfidence. With six races the bins are thin — the n "
                "labels show how thin — so this is a check that nothing is grossly "
                "miscalibrated, not proof of calibration."
            )


# --------------------------------------------------------------------------
# Tab 4 — Simulator
# --------------------------------------------------------------------------
with tabs[3]:
    st.header("Simulator — how certainty changes the answer")
    fitted_sigma = float(prediction["simulation"]["residual_sigma"])

    controls, chart = st.columns([1, 2.6])
    with controls:
        sigma = st.slider(
            "sigma — race-day uncertainty (finishing positions)",
            min_value=0.5, max_value=6.0, value=float(round(fitted_sigma, 2)), step=0.1,
            help="The standard deviation of the model's out-of-sample error.",
        )
        n_sims = st.slider("number of simulations", 1_000, 50_000, 20_000, step=1_000)
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
            width="stretch",
        )

    st.caption(
        "**What you are looking at.** Sigma is how wrong the model expects to be. Drag it "
        "down and the field separates — the favourite's probability climbs and the model "
        "looks decisive. Drag it up and everything flattens toward a uniform 1-in-20, "
        "because if you cannot predict finishing position to better than six places, you "
        "cannot tell anyone much about who wins. The fitted value, "
        f"**{fitted_sigma:.2f} positions**, is not a choice — it is measured from how far "
        "off this model actually was across six held-out races. That is why the locked "
        "prediction tops out near "
        f"{prediction['predicted_winner']['p_win'] * 100:.0f}% rather than at something "
        "more dramatic."
    )

    top_now = live.nlargest(5, "p_win")[["abbreviation", "team_name", "p_win", "p_podium"]].copy()
    top_now["p_win"] = (top_now["p_win"] * 100).map("{:.1f}%".format)
    top_now["p_podium"] = (top_now["p_podium"] * 100).map("{:.1f}%".format)
    top_now.columns = ["Driver", "Team", "P(win)", "P(podium)"]
    st.dataframe(top_now, width="stretch", hide_index=True)


# --------------------------------------------------------------------------
# Tab 5 — Result (only once the race has been run)
# --------------------------------------------------------------------------
if RACE_DONE:
    with tabs[4]:
        st.header("Result")
        import numpy as np
        from scipy.stats import spearmanr

        actual = load_actual_results()
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
        st.dataframe(scores.style.format({"value": "{:.4f}"}),
                     width="stretch", hide_index=True)
        st.caption(
            "One race cannot validate a probabilistic model. A 17% favourite winning is "
            "not confirmation, and losing is not refutation — the backtest over six races "
            "is the evidence; this is the demonstration."
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
