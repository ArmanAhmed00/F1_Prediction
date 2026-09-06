# 2026 Dutch Grand Prix — prediction post-mortem

_Scored 2026-09-06T10:47:39+00:00_

Prediction locked at `2026-08-23T12:09:31+00:00`, before the race, from `predictions/2026_R12_dutch.json`. Model `A_ridge`, trained on rounds 1-11 (321 rows), simulated with sigma 4.624 over 20,000 draws.

## The short version

- **The podium was called exactly right as a set.** All three of the predicted podium (ANT, NOR, RUS) finished on the podium — just not in the predicted order.
- **The winner was missed.** NOR won from grid P1; the model ranked him **2nd** and gave him **p_win = 0.1531** (15.31%), against the 4.55% a uniform prior would have given.
- **Rank correlation held up**: Spearman rho 0.737 on the full 22-car order, against 0.673 averaged over the backtest folds 6-11.
- **The single biggest error was a retirement**, not a pace call: VER by 15 places.

## Scores on this race

| metric | this race | backtest mean (folds 6-11) | uniform 1/n |
| --- | ---: | ---: | ---: |
| win log loss | 0.1263 | 0.1142 | 0.1849 |
| win Brier | 0.0371 | 0.0338 | 0.0434 |
| podium log loss | 0.2174 | 0.2418 | 0.3983 |
| podium Brier | 0.0681 | 0.0774 | 0.1178 |
| Spearman rho (predicted rank) | 0.7369 | 0.6725 | — |
| Spearman rho (raw predicted position) | 0.7380 | — | — |
| MAE, raw predicted position vs finish | 3.0476 | 3.4030 | — |
| MAE, predicted rank vs finish | 3.0000 | — | — |
| p(actual winner) | 0.1531 | 0.1957 | 0.0455 |

One race is one race. These numbers say what happened; they do not establish that the model is calibrated, and a single log loss cannot be meaningfully compared against a 6-fold mean except as a sanity check.

## Predicted vs actual, side by side

Two independent orderings of the same field. Read each half top to bottom on its own; the two halves of a row are not the same car.

| # | PREDICTED | team | grid | p_win | p_podium | ‖ | # | ACTUAL | status | laps |
| ---: | --- | --- | ---: | ---: | ---: | --- | ---: | --- | --- | ---: |
| 1 | **ANT** | Mercedes | 3 | 0.1732 | 0.4298 | ‖ | 1 | **NOR** | Finished | 72 |
| 2 | **NOR** | McLaren | 1 | 0.1531 | 0.3986 | ‖ | 2 | **ANT** | Finished | 72 |
| 3 | **RUS** | Mercedes | 2 | 0.1474 | 0.3835 | ‖ | 3 | **RUS** | Finished | 72 |
| 4 | **HAM** | Ferrari | 5 | 0.1434 | 0.3863 | ‖ | 4 | **HAM** | Finished | 72 |
| 5 | **LEC** | Ferrari | 6 | 0.1017 | 0.3050 | ‖ | 5 | **LEC** | Finished | 72 |
| 6 | **PIA** | McLaren | 4 | 0.0940 | 0.2908 | ‖ | 6 | **PIA** | Finished | 72 |
| 7 | **VER** | Red Bull Racing | 7 | 0.0674 | 0.2245 | ‖ | 7 | **LAW** | Finished | 72 |
| 8 | **LAW** | Red Bull Racing | 8 | 0.0393 | 0.1595 | ‖ | 8 | **HUL** | Lapped | 71 |
| 9 | **LIN** | Racing Bulls | 10 | 0.0200 | 0.0909 | ‖ | 9 | **ALO** | Lapped | 71 |
| 10 | **BOR** | Audi | 9 | 0.0186 | 0.0946 | ‖ | 10 | **GAS** | Lapped | 71 |
| 11 | **GAS** | Alpine | 11 | 0.0164 | 0.0845 | ‖ | 11 | **TSU** | Lapped | 71 |
| 12 | **COL** | Alpine | 14 | 0.0078 | 0.0398 | ‖ | 12 | **LIN** | Lapped | 71 |
| 13 | **HUL** | Audi | 13 | 0.0062 | 0.0345 | ‖ | 13 | **BOR** | Lapped | 71 |
| 14 | **TSU** | Racing Bulls | 12 | 0.0040 | 0.0239 | ‖ | 14 | **COL** | Lapped | 70 |
| 15 | **OCO** | Haas F1 Team | 15 | 0.0023 | 0.0169 | ‖ | 15 | **PER** | Lapped | 70 |
| 16 | **ALB** | Williams | 16 | 0.0022 | 0.0150 | ‖ | 16 | **SAI** | Lapped | 70 |
| 17 | **BEA** | Haas F1 Team | 20 | 0.0010 | 0.0066 | ‖ | 17 | **ALB** | Retired | 66 |
| 18 | **ALO** | Aston Martin | 18 | 0.0008 | 0.0045 | ‖ | 18 | **BOT** | Retired | 61 |
| 19 | **SAI** | Williams | 17 | 0.0006 | 0.0065 | ‖ | 19 | **OCO** | Retired | 52 |
| 20 | **STR** | Aston Martin | 19 | 0.0003 | 0.0022 | ‖ | 20 | **STR** | Retired | 45 |
| 21 | **BOT** | Cadillac | 21 | 0.0001 | 0.0009 | ‖ | 21 | **BEA** | Retired | 2 |
| 22 | **PER** | Cadillac | 22 | 0.0001 | 0.0011 | ‖ | 22 | **VER** | Retired | 0 |

### Per-driver error

| driver | grid | predicted | actual | error | status |
| --- | ---: | ---: | ---: | ---: | --- |
| **VER** | 7 | 7 | 22 | 15 | Retired |
| **ALO** | 18 | 18 | 9 | 9 | Lapped |
| **PER** | 22 | 22 | 15 | 7 | Lapped |
| **HUL** | 13 | 13 | 8 | 5 | Lapped |
| **BEA** | 20 | 17 | 21 | 4 | Retired |
| **OCO** | 15 | 15 | 19 | 4 | Retired |
| **TSU** | 12 | 14 | 11 | 3 | Lapped |
| **BOT** | 21 | 21 | 18 | 3 | Retired |
| **SAI** | 17 | 19 | 16 | 3 | Lapped |
| **BOR** | 9 | 10 | 13 | 3 | Lapped |
| **LIN** | 10 | 9 | 12 | 3 | Lapped |
| **COL** | 14 | 12 | 14 | 2 | Lapped |
| **ANT** | 3 | 1 | 2 | 1 | Finished |
| **GAS** | 11 | 11 | 10 | 1 | Lapped |
| **LAW** | 8 | 8 | 7 | 1 | Finished |
| **ALB** | 16 | 16 | 17 | 1 | Retired |
| **NOR** | 1 | 2 | 1 | 1 | Finished |
| **PIA** | 4 | 6 | 6 | 0 | Finished |
| **LEC** | 6 | 5 | 5 | 0 | Finished |
| **HAM** | 5 | 4 | 4 | 0 | Finished |
| **STR** | 19 | 20 | 20 | 0 | Retired |
| **RUS** | 2 | 3 | 3 | 0 | Finished |

## The 5 largest position errors

**VER** — predicted P7, finished P22 (15 places). DNF on lap 1 (0 laps completed) - out in the opening-lap incident that brought the safety car and then the red flag. Retirement is not in the predicted order at all - the model ranks expected finishing position and folds DNF risk in only as a 17.4% simulated retirement probability.

**ALO** — predicted P18, finished P9 (9 places). classified P9 (lapped) from grid P18; underrated by 9 places. No incident in race control - a straight pace-and-strategy miss, not a chaos artefact.

**PER** — predicted P22, finished P15 (7 places). classified P15 (lapped) from grid P22; underrated by 7 places. No incident in race control - a straight pace-and-strategy miss, not a chaos artefact.

**HUL** — predicted P13, finished P8 (5 places). classified P8 (lapped) from grid P13; underrated by 5 places. No incident in race control - a straight pace-and-strategy miss, not a chaos artefact.

**BEA** — predicted P17, finished P21 (4 places). DNF on lap 2, immediately after the red-flag restart. Retirement is not in the predicted order at all - the model ranks expected finishing position and folds DNF risk in only as a 17.4% simulated retirement probability.

## What actually missed

**The front of the field was easy and the model got it.** 6 of the 6 predicted top-6 finished in the top 6, and the 9 classified runners who started in the top 10 were placed to within a mean signed error of -0.56 positions. Chaotic race or not, the quick cars started at the front, stayed at the front, and none of that required a model.

**The order within the podium was wrong.** 3 of 3 predicted podium finishers made the podium, but the top pick was ANT and the winner was NOR, ranked 2 at p_win 0.1531. Getting the set right and the order wrong is the expected outcome when the top three probabilities sit within a few points of each other — which is honest of the model, not impressive.

**The real miss is the back half, and it is systematic.** Of the 7 classified finishers who started outside the top 10, 6 finished better than predicted, mean signed error +3.71 positions against -0.56 for the front runners. The model assumes the grid roughly holds; on a day with a red flag, two virtual safety cars and six retirements, cars from the back got a free strategy reset and moved up, and nothing in the feature list can see that coming. `positions_gained_ewm` is a season-long average and is far too slow to represent it.

**6 retirements are simply not modelled.** They enter only through `dnf_rate_shrunk`, as an independent per-driver coin flip in the simulation, which produces sensible win probabilities and a meaningless finishing order for anyone who stops. The largest single error on the day (VER, 15 places) is entirely this. It is not a tuning problem and no amount of retuning the regressor will fix it — the model has no concept of a first-lap incident.

None of the above is a reason to change the model today. It is a reason to write it down. See `reports/v2_backlog.md`.
