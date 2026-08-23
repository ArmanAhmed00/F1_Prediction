# 2026 Dutch Grand Prix — Prediction Report

_Generated 2026-08-23T12:10:35+00:00 · state: PRE-RACE_

Model: `A_ridge` · Locked: 2026-08-23T12:09:31+00:00 ·
Backtest: walk-forward rounds 6-11


---

## Executive summary

For the **2026 Dutch Grand Prix** at Zandvoort, the model predicts
**ANT (Mercedes)** to win, with a probability of
**17.3%**. The predicted podium is
**ANT, NOR, RUS**, with podium probabilities of
ANT 43.0%, NOR 39.9%, RUS 38.4%.

The prediction comes from a **A_ridge** regression on
finishing position, trained on rounds
1-11
(321 driver-sessions), with win and podium
probabilities derived from **20,000** Monte-Carlo
simulations using a fitted residual sigma of
**4.62 positions**.

The forecast was locked at **2026-08-23T12:09:31+00:00**, before the race,
and committed to version control at that time.

No driver exceeds a 17% chance
of winning. That flatness is the honest output of a model whose out-of-sample error
is 4.6 positions, not hedging: over six backtest races it identified the winner
50% of the time and assigned the eventual winner an average
probability of 19.6%.

---

## Data summary

| Item | Value |
| --- | --- |
| Season | 2026 only — see methodology |
| Rounds covered | 1-12 |
| Sessions used | Races and Sprints (classified); practice and qualifying feed features |
| Model table rows | 374 (343 scoreable, 22 to predict) |
| Training rows | 321 (rounds 1-11) |
| Drivers / teams | 23 / 11 |
| Features | 14 |
| Excluded features | has_sprint_data, track_temp_mean, was_wet |

### Features

| Feature | Definition |
| --- | --- |
| `quali_pct_off_pole` | Qualifying gap to pole, as a percentage of the pole lap. |
| `grid_position` | Starting position on the grid (1 = pole). |
| `quali_teammate_delta_pct` | Driver's qualifying gap minus their teammate's, same session. |
| `fp_best_pct_off_fastest` | Best clean practice lap, as a percentage off the fastest in the field. |
| `fp_longrun_pace_pct` | Median lap of the driver's longest clean practice stint, % off the field best. |
| `sprint_finish_position` | Finishing position in that weekend's sprint (imputed when there was none). |
| `sprint_positions_gained` | Sprint grid minus sprint finish. |
| `form_quali_pace_ewm` | Exponentially weighted mean of past qualifying gaps (halflife 3 rounds). |
| `form_finish_pos_ewm` | Exponentially weighted mean of past finishing positions. |
| `form_race_pace_ewm` | Exponentially weighted mean of past clean-air race pace, % off session best. |
| `team_form_pace_ewm` | Team-level exponentially weighted qualifying pace — two cars per estimate. |
| `positions_gained_ewm` | Exponentially weighted mean of past (grid − finish); racecraft and strategy. |
| `dnf_rate_shrunk` | Beta-Binomial posterior retirement probability, shrunk toward the field rate. |
| `championship_position` | Championship standing going into the session, from cumulative points. |

---

## Methodology

### Why only 2026 data

2026 is a complete technical reset: new chassis and aerodynamic regulations
arrived alongside a new power-unit formula, and the grid gained two constructors.
A qualifying gap or a race pace figure from 2025 describes a car that no longer
exists under rules that no longer apply. Pooling earlier seasons would not add
signal, it would add confidently wrong signal — and a backtest would not expose
that, because the leakage is conceptual rather than temporal. The cost is sample
size: eleven races and five sprints, 343 scoreable driver-sessions.

### Why the target is finishing position, not "won"

The dataset contains 16 winners against 343 observations. Training a binary
classifier on that, with 14 features, fits noise and produces probabilities that
cannot be trusted. Predicting **finishing position** instead makes every row
informative: a driver coming ninth teaches the model as much as a driver winning.

Win and podium probabilities are then recovered from those positions by Monte-Carlo
simulation: each driver's predicted position is perturbed by Normal(0, sigma) noise,
a retirement is drawn from their shrunk DNF probability, the field is ranked, and
the exercise is repeated 20,000 times. P(win) is the share of simulations a driver
ranks first; P(podium) the share they rank in the top three.

### Why simulation guarantees P(win) <= P(podium)

Both quantities are counted from the *same* simulated races. Any simulation in
which a driver finishes first is also a simulation in which they finish top three,
so the win count can never exceed the podium count. The win probabilities sum to
exactly 1.0 for the same reason: precisely one car is ranked first in each
simulated race. Two independently fitted classifiers guarantee neither property and
will cheerfully output a driver with a 30% chance of winning and 20% of a podium.

### Walk-forward backtesting, and why three live races prove nothing

The model is validated by walking forward through the season: train on every round
before round k, predict round k's race, score, advance. Nothing from round k or
later is visible at training time, so each fold tests the exact procedure used for
the live prediction. Six folds (rounds 6-11) are available.

Six races — and certainly the one live race this project culminates in — is a
**demonstration, not an evaluation**. A probabilistic forecast can only be judged
over many events: a model that says 17% and turns out right has not been proven
correct, and one that says 17% and turns out wrong has not been proven wrong. The
walk-forward backtest is the evidence; the Dutch Grand Prix is the demonstration.

---

## Backtest results

Walk-forward over rounds 6-11, six races. All probabilities clipped to [1e-6, 1-1e-6] before scoring, because B1 and B2 assert probability 1.0 and would otherwise produce infinite log loss.

| model | n_folds | win_log_loss | win_brier | podium_log_loss | podium_brier | top1_hit | spearman_rho | mae_position | p_actual_winner | sigma_test | sigma_cv |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D_grid_baseline | 6 | 0.0982 | 0.0298 | 0.2227 | 0.0714 | 0.6667 | 0.6701 | 3.2879 | 0.2596 | 5.0454 | 4.6764 |
| A_ridge | 6 | 0.1142 | 0.0338 | 0.2418 | 0.0774 | 0.5 | 0.6725 | 3.403 | 0.1957 | 4.6238 | 4.3937 |
| B_random_forest | 6 | 0.1186 | 0.0349 | 0.2542 | 0.0826 | 0.3333 | 0.6836 | 3.3088 | 0.188 | 4.6092 | 4.4899 |
| C_gradient_boosting | 6 | 0.1449 | 0.0396 | 0.2988 | 0.0997 | 0.1667 | 0.6657 | 3.513 | 0.1529 | 4.833 | 4.8031 |
| B3_uniform | 6 | 0.1849 | 0.0434 | 0.3983 | 0.1178 |  |  |  | 0.0455 |  |  |
| B1_pole_wins | 6 | 0.4187 | 0.0303 | 1.6746 | 0.1212 | 0.6667 |  |  | 0.6667 |  |  |
| B2_champ_leader_wins | 6 | 0.8373 | 0.0606 | 1.8839 | 0.1364 | 0.3333 |  |  | 0.3333 |  |  |

**The selected model (`A_ridge`) did not beat all baselines.** It beats B1, B2 and B3 on win log loss, but B4 — assume every driver finishes where they started — scores 0.0982 against `A_ridge`'s 0.1142. This is reported as found; no tuning was done to reverse it.

### Calibration of the selected model

Mean predicted probability against observed frequency, by decile, across the six walk-forward races. Perfect calibration would put `observed` equal to `mean_predicted` on every row. The `n` column shows how few drivers land in each bin, which is why this is a sanity check rather than evidence of calibration.

| target | decile | mean_predicted | observed | n |
| --- | --- | --- | --- | --- |
| win | 1 | 0.0 | 0.0 | 16 |
| win | 2 | 0.0003 | 0.0 | 11 |
| win | 3 | 0.0014 | 0.0 | 14 |
| win | 4 | 0.003 | 0.0 | 12 |
| win | 5 | 0.0063 | 0.0 | 13 |
| win | 6 | 0.0147 | 0.0 | 13 |
| win | 7 | 0.0353 | 0.0 | 13 |
| win | 8 | 0.071 | 0.0 | 13 |
| win | 9 | 0.126 | 0.0769 | 13 |
| win | 10 | 0.1891 | 0.3571 | 14 |
| podium | 1 | 0.0004 | 0.0 | 14 |
| podium | 2 | 0.003 | 0.0 | 13 |
| podium | 3 | 0.011 | 0.0 | 13 |
| podium | 4 | 0.0201 | 0.0 | 13 |
| podium | 5 | 0.0379 | 0.0 | 13 |
| podium | 6 | 0.0755 | 0.0769 | 13 |
| podium | 7 | 0.1447 | 0.0 | 13 |
| podium | 8 | 0.2408 | 0.2308 | 13 |
| podium | 9 | 0.3553 | 0.5385 | 13 |
| podium | 10 | 0.4604 | 0.5 | 14 |

---

## The locked prediction

| Driver | Team | Grid | Predicted pos. | P(win) % | P(podium) % |
| --- | --- | --- | --- | --- | --- |
| ANT | Mercedes | 3 | 4.54 | 17.3 | 43.0 |
| NOR | McLaren | 1 | 4.69 | 15.3 | 39.9 |
| RUS | Mercedes | 2 | 4.8 | 14.7 | 38.4 |
| HAM | Ferrari | 5 | 5.21 | 14.3 | 38.6 |
| LEC | Ferrari | 6 | 5.95 | 10.2 | 30.5 |
| PIA | McLaren | 4 | 6.34 | 9.4 | 29.1 |
| VER | Red Bull Racing | 7 | 6.95 | 6.7 | 22.5 |
| LAW | Red Bull Racing | 8 | 8.47 | 3.9 | 15.9 |
| LIN | Racing Bulls | 10 | 10.19 | 2.0 | 9.1 |
| BOR | Audi | 9 | 10.06 | 1.9 | 9.5 |
| GAS | Alpine | 11 | 10.26 | 1.6 | 8.5 |
| COL | Alpine | 14 | 12.33 | 0.8 | 4.0 |
| HUL | Audi | 13 | 12.01 | 0.6 | 3.4 |
| TSU | Racing Bulls | 12 | 13.2 | 0.4 | 2.4 |
| OCO | Haas F1 Team | 15 | 14.15 | 0.2 | 1.7 |
| ALB | Williams | 16 | 14.35 | 0.2 | 1.5 |
| BEA | Haas F1 Team | 20 | 15.71 | 0.1 | 0.7 |
| ALO | Aston Martin | 18 | 16.28 | 0.1 | 0.4 |
| SAI | Williams | 17 | 15.79 | 0.1 | 0.6 |
| STR | Aston Martin | 19 | 17.23 | 0.0 | 0.2 |
| BOT | Cadillac | 21 | 18.73 | 0.0 | 0.1 |
| PER | Cadillac | 22 | 18.34 | 0.0 | 0.1 |

---

## Post-race evaluation

_Race not yet run - re-run this notebook after 13:00 UTC on 23 August to populate the evaluation section._

---

## Limitations

1. **Eleven races of training data.** 343 scoreable driver-sessions. This is a
   small-data problem, and it is the binding constraint on everything else.
2. **A single season, with no possibility of extension.** The 2026 regulation reset
   makes every earlier season invalid for pace. More data can only arrive by waiting
   for more races.
3. **The fitted model did not beat the grid baseline.** Over six walk-forward races,
   "assume everyone finishes where they started" had a lower win log loss than any
   fitted candidate. The honest reading is that this dataset is too small to
   demonstrate that a learned model improves on a well-chosen heuristic.
4. **No tyre-strategy modelling.** Stint plans, compound choice, undercut and
   overcut opportunities and pit-window dynamics are entirely absent, despite being
   among the largest determinants of finishing position.
5. **Weather is handled only as a flag,** and that flag carries no measurable
   signal (zero mutual information with finishing position), so it was dropped from
   the model. There is no wet-weather pace model, and race-day weather is not
   forecast at all — the dataset holds observations only.
6. **No safety-car or incident modelling.** Both reorder races and neither is
   represented.
7. **One race cannot validate a probabilistic model.** A 17% favourite winning is
   neither confirmation nor refutation. Calibration can only be assessed over many
   events, and six backtest races is already too few to establish it.
8. **Grid position for round 12 is qualifying order, not the official grid.**
   Penalties are not reflected; this must be verified manually before the race.
9. **Features are heavily collinear** — 33 of 136 pairs above |r| = 0.70 — so
   individual feature importances should not be read as causal contributions.
10. **`sprint_finish_position` is imputed on 76% of rows** from `form_finish_pos_ewm`,
    so its apparent contribution is largely borrowed from that feature.
