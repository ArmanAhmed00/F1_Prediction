# Pre-registration — 2026 Italian Grand Prix (round 13, Monza)

**Written and committed before the race.** Race start 13:00 UTC, 6 September 2026.

The purpose of this document is to put the expectation on record independently of
the outcome. An explanation offered after the result is worth very little; the
same explanation offered before it is evidence. Everything below is written from
the feature table and the published grid, with no knowledge of the race.

The model is frozen (`A_ridge`, selected at the round 12 backtest on folds 6-11).
Nothing in this document is a reason to change it today. Where it identifies a
defect, that defect goes in `reports/v2_backlog.md` and is addressed as a labelled
v2 at Madrid, run alongside v1.

---

## 1. The core expectation: the model will be overconfident in the pole sitter

Monza is the easiest overtaking circuit of the season. Two DRS zones on the
longest full-throttle stretch on the calendar, three heavy braking zones that
reward a slipstream, and a lap short enough that a pace deficit converts into a
passing opportunity within a few laps. Zandvoort, two weeks ago, is at the other
end of that scale: a narrow, banked, single-line circuit where track position is
close to decisive.

**Grid position should therefore matter substantially LESS at Monza than it did
at Zandvoort.**

The model cannot know this. There is no circuit-specific term anywhere in the
14-feature list — no overtaking-difficulty index, no track-length or DRS-zone
count, nothing that distinguishes Monaco from Monza. `grid_position` enters with
one weight, fitted across all twelve completed rounds, and that single weight is
applied identically at the hardest and easiest overtaking tracks of the year.

**Registered prediction: the model will place too much weight on grid position
today, and will therefore be overconfident in the pole sitter and underconfident
in fast cars starting out of position.**

This is a structural claim about the model, not a claim about who wins. It is
correct whether or not the pole sitter goes on to win.

## 2. Monza is the most power-sensitive circuit of the year

Under the 2026 power unit regulations — the near-50/50 split between internal
combustion and electrical deployment, and the active-aerodynamics rules that make
the low-drag configuration cheaper to run than it used to be — Monza is the
circuit where power unit quality converts most directly into lap time. Roughly
80% of the lap is spent at full throttle. Energy management over the lap matters
more here than anywhere else on the calendar.

None of this is visible to the model either. The features measure *observed pace*
(qualifying gaps, practice long runs, race pace), which absorbs power unit
quality only through the average of the twelve circuits already raced. A car whose
advantage is specifically a power unit advantage will be systematically
underrated here, because its season-long form is diluted by twelve tracks where
that advantage counted for less.

## 3. The specific blind spot: Gasly, Alpine, and an upgrade the features cannot see

Pierre Gasly took pole by 0.060s over Russell — his first, and Alpine's first in
this regulation era. The model's view of that car is this:

| feature | Gasly | field context |
| --- | ---: | --- |
| `grid_position` | 1 | pole |
| `quali_pct_off_pole` | 0.000 | fastest in the session |
| `team_form_pace_ewm` | 2.122 | **7th of 11 teams** on season pace |
| `form_quali_pace_ewm` | 1.889 | ~1.9% off pole on a season average |
| `form_finish_pos_ewm` | 10.24 | a midfield runner |
| `championship_position` | 9 | ninth |

The tension is on that one row. One feature says he is the fastest man in Italy;
five say he drives the seventh-quickest car and finishes tenth. **This is the most
internally contradictory driver-row the model has been handed all season.**

Two reasons to think the single-session evidence is closer to the truth than the
season-long evidence, both of which the feature list is structurally blind to:

1. **Alpine runs Mercedes power.** Mercedes is ranked #1 of 11 on
   `team_form_pace_ewm` (0.236). At the most power-sensitive circuit of the year,
   a customer team with the best power unit on the grid is in a materially
   better position than its season-long average suggests. The model has no
   engine-supplier feature; Alpine and Mercedes are unrelated categories to it.

2. **Alpine brought a significant update package to this weekend.** The features
   cannot see upgrade packages at all. Every form feature is an exponentially
   weighted mean with a three-round halflife, which means that after a genuine
   step change in car performance the model needs roughly three races to catch
   up, and spends all three underrating the car. A team that gets quicker on a
   Friday looks, to this model, exactly like a team that got lucky.

**Registered prediction: Gasly's true pace this weekend is better than his
season-long form features represent, and the model's estimate of him is a
compromise between a correct single-session signal and five stale ones.**

Note the direction of the two errors. §1 says the model over-weights his grid
slot; §3 says it under-weights his car. They partially cancel, which means the
number the model emits for Gasly may look reasonable while being reasonable for
the wrong reasons. That is worth saying out loud in advance, because "the
prediction looked sensible" is not evidence that the model works.

## 4. The other side of the same coin: the penalised drivers

Four drivers were moved from their qualifying positions. Three of them are quick
cars sent to the back for power unit component changes:

| driver | qualified | starts | team pace rank | championship |
| --- | ---: | ---: | ---: | ---: |
| Antonelli | 7 | **20** | #1 (Mercedes) | **1st** |
| Lawson | 14 | **21** | #4 (Red Bull) | 8th |
| Albon | 18 | **22** | #9 (Williams) | 16th |
| Piastri | 3 | **6** | #2 (McLaren) | 7th |

Antonelli is the championship leader in the fastest car on the grid, starting
20th, at the easiest overtaking circuit of the season. The model will rank him
low, because `grid_position` dominates and it has no way to know that 20th at
Monza is a far less costly place to start than 20th at Zandvoort.

**Registered prediction: the model will underrate Antonelli, Lawson and Piastri's
recovery today.** This is the same failure the round 12 post-mortem already
identified and quantified: at Zandvoort, of the seven classified finishers who
started outside the top 10, six finished better than predicted, with a mean
signed error of +3.71 positions against −0.56 for the front runners. Monza should
make that bias worse, not better, because overtaking is easier here.

## 5. What would falsify this

Stated plainly, so it cannot be reinterpreted afterwards:

- **§1 is supported** if the pole sitter's realised finishing position is worse
  than his p_win implies, and more generally if the model's Spearman rho against
  the actual order is lower today than its backtest mean of 0.682 — i.e. the grid
  told it less than usual.
- **§1 is falsified** if the race finishes close to grid order and the model's
  rank correlation comes in at or above its backtest mean.
- **§3 is supported** if Gasly finishes ahead of where his season-long features
  (10th on `form_finish_pos_ewm`) would put him — concretely, a podium or a win.
- **§3 is falsified** if Gasly drops back to the midfield, which would mean pole
  was a single-lap, low-fuel artefact and the season-long features were right.
- **§4 is supported** if Antonelli, Lawson and Piastri collectively gain more
  places than the model's predicted positions imply.
- **§4 is falsified** if the back-of-grid starters stay at the back.

Note that §1 and §3 can both be supported at once, and that this is the most
likely outcome: the model can simultaneously over-weight the grid *as a
mechanism* and land near the right answer *for Gasly specifically*, because his
grid slot and his true pace happen to point the same way this weekend. A correct
prediction is not, on its own, evidence that the reasoning behind it was correct.

## 6. What this document does not claim

- It does not predict a winner. That is the model's job, and the model's output
  is locked separately in `predictions/2026_R13_monza.json`.
- It does not claim the model is bad. It beats a uniform prior comfortably and
  beats the pole-sitter-wins baseline. It does **not** beat the "everyone
  finishes where they started" baseline, which remains the honest headline
  result after seven walk-forward folds.
- It does not justify changing anything today. Every defect named here is a v2
  item. Changing the model between races would make each race prediction a
  different model with a sample size of one, and the whole three-race exercise
  would demonstrate nothing.

---

_Sources for the starting grid and penalties: formula1.com official starting
grid, corroborated against motorsport.com. Two secondary outlets (crash.net,
racingnews365) order Albon and Lawson the other way round at P21/P22; the
official classification is used. This affects two back-of-grid cars and no
podium candidate._

---

# Addendum — written after the model ran, still before the race

Appended at 2026-09-06T10:51Z, roughly two hours before the start. Everything
above this line was written before `predict_race.py` was run and **has not been
edited**. This section records what the model actually produced, because one of
the registered claims was falsified immediately, by the model's own output,
without needing the race at all. That is worth recording as such rather than
quietly rewriting the prediction it came from.

## §1's stated consequence is wrong

§1 registered: *"the model will place too much weight on grid position today, and
will therefore be overconfident in the pole sitter."*

It is not. The model ranks Gasly **4th**, at p_win 12.1%:

| | model (`A_ridge`) | grid baseline (B4) |
| --- | ---: | ---: |
| GAS (pole) | **12.1%** | **29.0%** |
| RUS (P2) | 18.4% | 20.8% |
| HAM (P4) | 16.7% | 11.9% |
| LEC (P3) | 12.6% | 15.6% |

The fitted model **discounts** the pole sitter heavily against the grid baseline
— 12.1% against 29.0%, less than half. Its top pick is Russell from P2, in the
car ranked #1 of 11 on season pace. The plausibility guard did not fire, and on
the guard's own terms it was right not to.

So the derived consequence in §1 was wrong. The premise was not: the model still
has no circuit term and still applies one grid weighting everywhere. What the
premise does not license is the step from "uniform grid weight" to "overconfident
in the pole sitter", because `grid_position` is one of fourteen features and for
Gasly the other five form features pull hard in the opposite direction —
`team_form_pace_ewm` 2.122 (7th of 11), `form_finish_pos_ewm` 10.24. The ridge
blends them, and the blend lands well below the grid slot.

That was a reasoning error on my part, and it was in the registered document
before the race, which is exactly where an error of that kind belongs.

## What this does to the other registered claims

- **§3 (Gasly underrated by his season features) is now sharper, not weaker.**
  The model has him 4th on a track where he starts 1st. If Gasly wins or
  podiums, §3 is supported and the model's stale form features are demonstrably
  the reason it missed him. If he drops to the midfield, §3 is falsified and the
  season-long features were right to discount him. The test is cleaner than it
  was when §3 was written, because the model has now committed to disagreeing
  with the grid.
- **§4 (penalised drivers underrated) is partly pre-falsified too.** The model
  puts Antonelli **11th** from a P20 start — a predicted nine-place recovery,
  against the grid baseline's 0.000% win probability and last-place ordering. It
  is not ignoring him. Lawson (P21 → predicted 16th) and Piastri (P6 → predicted
  6th) get less. §4 should be judged on the realised numbers, but the direction
  of the claim is weaker than stated.
- **The correct version of §1**, which I did not register in time and which is
  therefore worth less: the model has no way to know that Monza rewards a fast
  car starting out of position *more* than an average circuit does. Whether its
  uniform blend happens to be too grid-heavy or too form-heavy today is not
  something it can reason about, and getting the answer approximately right here
  would be luck rather than skill.

Nothing here changes the locked prediction, which stands exactly as generated.
The v2 items in `reports/v2_backlog.md` are unchanged: item 1 (no circuit
overtaking term) is still the right diagnosis, even though the consequence I
predicted from it did not follow.
