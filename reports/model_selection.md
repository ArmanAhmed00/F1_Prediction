# Model selection

_Generated 2026-09-06T10:43:50+00:00_

**The model is frozen.** `A_ridge` was selected at the round 12 backtest, walk-forward folds 6-11 and is not re-selected here. Only the data changes between races; if the model changed too, each race would be predicted by a different model with a sample size of one, which would demonstrate nothing. A challenger is a v2: evaluated on the whole walk-forward and run alongside v1, never swapped in silently.

Original selection criterion: **mean win log loss** across walk-forward folds. Now re-scored over rounds 6-12 (7 races). No hyperparameter was tuned on a test fold; Ridge's alpha is chosen by cross-validation inside each training split.

## In use: `A_ridge` (frozen)

Still the best of the fitted candidates on the extended folds (0.1157 win log loss). The freeze costs nothing this week.

| metric | value |
| --- | --- |
| win log loss | 0.1157 |
| win Brier | 0.0343 |
| podium log loss | 0.2373 |
| podium Brier | 0.0757 |
| top-1 hit rate | 0.429 |
| Spearman rho | 0.682 |
| MAE (positions) | 3.352 |
| mean residual sigma (test folds) | 4.601 |

## Did it beat the baselines?

- beats `B1_pole_wins` (0.1157 vs 0.3588)
- beats `B2_champ_leader_wins` (0.1157 vs 0.8971)
- beats `B3_uniform` (0.1157 vs 0.1849)
- **does NOT beat** `D_grid_baseline` (0.1157 vs 0.0976)

**No, and this is reported as found rather than tuned until something wins.**

`A_ridge` is the frozen model, chosen from (A_ridge, B_random_forest, C_gradient_boosting) by win log loss at the round 12 backtest, walk-forward folds 6-11, and it beats `B1_pole_wins`, `B2_champ_leader_wins`, `B3_uniform`. But it does not beat `D_grid_baseline`.

The lowest win log loss of anything tested belongs to `D_grid_baseline` (0.0976). In other words: **on this dataset, assuming every driver finishes where they started is a better probabilistic predictor than any of the fitted models.**

That is a believable result rather than a bug. Grid position is the single strongest signal available (Spearman rho = 0.745 in the EDA, and the highest mutual information of any feature), the other thirteen features are heavily collinear with it, and a few hundred rows is not enough for a fitted model to extract additional signal without adding more variance than it removes. 7 test races also cannot separate methods this close together. The correct response is more data, not more tuning.

## Full comparison

| model | n_folds | win_log_loss | win_brier | podium_log_loss | podium_brier | top1_hit | spearman_rho | mae_position | p_actual_winner | sigma_test | sigma_cv | label |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D_grid_baseline | 7 | 0.0976 | 0.0295 | 0.2148 | 0.0681 | 0.7143 | 0.6817 | 3.2208 | 0.2618 | 4.9785 | 4.6723 | B4  predicted position = grid position |
| A_ridge | 7 | 0.1157 | 0.0343 | 0.2373 | 0.0757 | 0.4286 | 0.6819 | 3.3522 | 0.1905 | 4.601 | 4.3747 | model A_ridge |
| B_random_forest | 7 | 0.1162 | 0.0342 | 0.2446 | 0.0789 | 0.4286 | 0.6838 | 3.2547 | 0.1955 | 4.6243 | 4.4645 | model B_random_forest |
| C_gradient_boosting | 7 | 0.1392 | 0.0388 | 0.2789 | 0.0926 | 0.2857 | 0.688 | 3.3136 | 0.164 | 4.7058 | 4.733 | model C_gradient_boosting |
| B3_uniform | 7 | 0.1849 | 0.0434 | 0.3983 | 0.1178 |  |  |  | 0.0455 |  |  | B3  uniform 1/n |
| B1_pole_wins | 7 | 0.3588 | 0.026 | 1.4354 | 0.1039 | 0.7143 |  |  | 0.7143 |  |  | B1  pole-sitter wins, p=1.0 |
| B2_champ_leader_wins | 7 | 0.8971 | 0.0649 | 1.7942 | 0.1299 | 0.2857 |  |  | 0.2857 |  |  | B2  championship leader wins, p=1.0 |

### Reading the table

- B1 and B2 assert probability 1.0, so every probability is clipped to [1e-06, 1-1e-06] before scoring; without that their log loss is infinite whenever they are wrong.
- `spearman_rho` and `mae_position` are undefined for B1-B3: they emit probabilities but no finishing order.
- `top1_hit` is undefined for B3, because a uniform distribution has no argmax.
- Accuracy is not reported. At ~5% positives, predicting that nobody wins scores 95%.
