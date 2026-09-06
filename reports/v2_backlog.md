# v2 backlog

Things believed to be wrong with the model. **None of them are acted on between
races.**

The rule: v1 (`A_ridge`, frozen at the round 12 backtest on folds 6-11) predicts
every race from Zandvoort to Madrid unchanged. Only the data moves. Anything in
this file is a candidate for an explicitly labelled v2, which gets evaluated on
the full walk-forward backtest and then run **alongside** v1 at Madrid, not
instead of it. Two models, both logged, both scored.

The reason is not conservatism. If the model is re-tuned every week, the three
race predictions are three different models with a sample size of one each, and
no amount of good intent recovers a comparison from that.

Ordered by expected value, not by ease.

---

## 1. No circuit-specific overtaking term

**Status:** open. Identified at round 13 (Monza), pre-registered in
`reports/monza_pre_registration.md` §1.

`grid_position` is the strongest feature in the model and it carries exactly one
fitted weight, applied identically at Monaco and Monza. Overtaking difficulty
varies enormously across the calendar; the model treats every circuit as average.

Expected effect: overconfident in the pole sitter at easy-overtaking circuits,
underconfident at hard ones. Symmetric and probably large.

**Candidate fix:** a per-circuit overtaking index — passes per race, or the
historical variance of (grid − finish) at that venue — interacted with
`grid_position`. The 2026-only constraint bites here: with one season there is at
most one prior observation per circuit, so the index has to come from either an
external source or a physical proxy (lap length, DRS zone count, average speed)
rather than from the data itself. Prefer the physical proxy; it is honest about
being a proxy.

**Blocker:** needs a circuit metadata table that does not currently exist.

## 2. Retirements are not modelled, only sampled

**Status:** open. Quantified at round 12, `reports/dutch_gp_evaluation.md`.

DNFs enter only via `dnf_rate_shrunk`, as an independent per-driver coin flip in
the simulation. This produces defensible *win* probabilities and a meaningless
*finishing order* for anyone who stops. At Zandvoort the single largest error of
the race — Verstappen, 15 places — was entirely this, and no amount of retuning
the regressor touches it.

Worse, the independence assumption is wrong in exactly the cases that matter. A
first-lap multi-car incident retires several cars at once; the simulation treats
those as independent events and therefore understates the probability of a
high-attrition race.

**Candidate fix:** correlate the retirement draws with a per-race shared factor
(a "chaos" latent), fitted on the observed race-level DNF counts. Cheap to
implement, and it widens the tails where they should be wide.

**Do not** try to predict *who* retires. Twelve races of data cannot support it.

## 3. Form features are too slow to see an upgrade package

**Status:** open. Identified at round 13, pre-registered in
`reports/monza_pre_registration.md` §3.

Every form feature is an EWM with a three-round halflife. After a genuine step
change in car performance the model takes roughly three races to catch up, and
underrates the car throughout. It cannot distinguish "this team got quicker" from
"this team got lucky", because both look like one good weekend against a slow
mean.

Gasly's Monza pole is the test case: one feature (`quali_pct_off_pole` = 0.000)
says fastest in Italy, five say seventh-quickest car and a tenth-place finisher.

**Candidate fix:** the cheap version is a shorter-halflife companion to each form
feature, letting the model learn how much weight to put on recent versus
season-long. The expensive version is a changepoint or a two-timescale
state-space model, which ~370 rows will not support.

**Watch for:** a shorter halflife raises variance. Test it on the walk-forward,
not on intuition.

## 4. No engine-supplier feature

**Status:** open. Identified at round 13.

Customer teams inherit power unit quality from their supplier, and at
power-sensitive circuits that matters more than at others. Alpine and Mercedes
are unrelated categories to the current model, despite Alpine running Mercedes
power. With four suppliers and eleven teams, a supplier-level pace aggregate is
better-estimated than a team-level one — more cars per estimate — and would let
Monza-like circuits borrow strength across the supplier's customers.

**Candidate fix:** a `supplier_form_pace_ewm`, built exactly like
`team_form_pace_ewm` but grouped by power unit supplier. Requires a
team → supplier mapping, which FastF1 does not expose; it would have to be a
small hand-maintained table alongside `data/manual/`.

## 5. Back-of-grid recovery is systematically underrated

**Status:** open. Quantified at round 12, `reports/dutch_gp_evaluation.md`.

At Zandvoort, of the seven classified finishers who started outside the top 10,
six finished better than predicted — mean signed error +3.71 positions, against
−0.56 for the front runners. The model assumes the grid roughly holds.
`positions_gained_ewm` exists to capture racecraft but is a season-long average
and far too slow to represent a specific race's strategy resets.

This partly overlaps item 1: a circuit overtaking term would absorb some of it.
Whether anything is left over after that is an empirical question and should be
tested in that order — item 1 first, then re-measure this.

## 6. The fitted model still does not beat the grid baseline

**Status:** open, and the honest headline result. Seven walk-forward folds.

| model | win log loss |
| --- | ---: |
| `D_grid_baseline` (finish = grid) | 0.0976 |
| `A_ridge` (frozen, in use) | 0.1157 |
| `B_random_forest` | 0.1162 |
| `C_gradient_boosting` | 0.1392 |
| `B3_uniform` | 0.1849 |

Assuming every driver finishes where they started remains a better probabilistic
predictor than any fitted model tried. This is believable rather than a bug —
grid position has by far the highest mutual information of any feature, the other
thirteen are heavily collinear with it, and ~370 rows is not enough for a fitted
model to extract additional signal without adding more variance than it removes.

**The correct response is more signal, not more tuning.** Items 1-4 all add
information the grid does not already contain. Hyperparameter search does not,
and any apparent gain from it over seven folds is noise.

**Guard against:** reading item 6 as a reason to ship the baseline. The baseline
cannot express uncertainty about anything, so it will be catastrophically
overconfident the first time a front-row car retires. Its log loss advantage over
seven races is real but thin.

---

## Not doing

- **Re-selecting the model on each new backtest.** The whole point of the freeze.
  Rounds 6-12 currently still rank `A_ridge` first among the fitted candidates,
  but if that flipped it would be noise at this sample size, not a signal.
- **Pre-2026 seasons.** The regulation reset makes earlier pace data actively
  misleading, and mixing it in is a leakage risk, not just a bias risk.
- **Predicting which specific driver retires.** Not supported by the data volume.
- **Reacting to a surprising result.** If Gasly wins from pole at Monza, that
  changes nothing in this file. A single race cannot promote or demote any item
  here.
