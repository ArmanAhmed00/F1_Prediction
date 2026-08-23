# Model selection

_Generated 2026-08-23T19:08:28+00:00_

Selection criterion: **mean win log loss** across walk-forward folds (rounds 6-11, 6 races). 
No hyperparameter was tuned on a test fold; Ridge's alpha is chosen by cross-validation inside each training split.

## Selected: `A_ridge`

| metric | value |
| --- | --- |
| win log loss | 0.1142 |
| win Brier | 0.0338 |
| podium log loss | 0.2418 |
| podium Brier | 0.0774 |
| top-1 hit rate | 0.500 |
| Spearman rho | 0.673 |
| MAE (positions) | 3.403 |
| mean residual sigma (test folds) | 4.624 |

## Did it beat the baselines?

- beats `B1_pole_wins` (0.1142 vs 0.4187)
- beats `B2_champ_leader_wins` (0.1142 vs 0.8373)
- beats `B3_uniform` (0.1142 vs 0.1849)
- **does NOT beat** `D_grid_baseline` (0.1142 vs 0.0982)

**No, and this is reported as found rather than tuned until something wins.**

`A_ridge` is the best of the fitted candidates (A_ridge, B_random_forest, C_gradient_boosting) by win log loss, and it beats `B1_pole_wins`, `B2_champ_leader_wins`, `B3_uniform`. But it does not beat `D_grid_baseline`.

The lowest win log loss of anything tested belongs to `D_grid_baseline` (0.0982). In other words: **on this dataset, assuming every driver finishes where they started is a better probabilistic predictor than any of the fitted models.**

That is a believable result rather than a bug. Grid position is the single strongest signal available (Spearman rho = 0.745 in the EDA, and the highest mutual information of any feature), the other thirteen features are heavily collinear with it, and 300 rows is not enough for a fitted model to extract additional signal without adding more variance than it removes. Six test races also cannot separate methods this close together. The correct response is more data, not more tuning.

## Full comparison

| model | n_folds | win_log_loss | win_brier | podium_log_loss | podium_brier | top1_hit | spearman_rho | mae_position | p_actual_winner | sigma_test | sigma_cv | label |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D_grid_baseline | 6 | 0.0982 | 0.0298 | 0.2227 | 0.0714 | 0.6667 | 0.6701 | 3.2879 | 0.2596 | 5.0454 | 4.6764 | B4  predicted position = grid position |
| A_ridge | 6 | 0.1142 | 0.0338 | 0.2418 | 0.0774 | 0.5 | 0.6725 | 3.403 | 0.1957 | 4.6238 | 4.3937 | model A_ridge |
| B_random_forest | 6 | 0.1186 | 0.0349 | 0.2542 | 0.0826 | 0.3333 | 0.6836 | 3.3088 | 0.188 | 4.6092 | 4.4899 | model B_random_forest |
| C_gradient_boosting | 6 | 0.1449 | 0.0396 | 0.2988 | 0.0997 | 0.1667 | 0.6657 | 3.513 | 0.1529 | 4.833 | 4.8031 | model C_gradient_boosting |
| B3_uniform | 6 | 0.1849 | 0.0434 | 0.3983 | 0.1178 |  |  |  | 0.0455 |  |  | B3  uniform 1/n |
| B1_pole_wins | 6 | 0.4187 | 0.0303 | 1.6746 | 0.1212 | 0.6667 |  |  | 0.6667 |  |  | B1  pole-sitter wins, p=1.0 |
| B2_champ_leader_wins | 6 | 0.8373 | 0.0606 | 1.8839 | 0.1364 | 0.3333 |  |  | 0.3333 |  |  | B2  championship leader wins, p=1.0 |

### Reading the table

- B1 and B2 assert probability 1.0, so every probability is clipped to [1e-06, 1-1e-06] before scoring; without that their log loss is infinite whenever they are wrong.
- `spearman_rho` and `mae_position` are undefined for B1-B3: they emit probabilities but no finishing order.
- `top1_hit` is undefined for B3, because a uniform distribution has no argmax.
- Accuracy is not reported. At ~5% positives, predicting that nobody wins scores 95%.
