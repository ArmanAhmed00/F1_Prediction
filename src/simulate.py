"""Win and podium probabilities, by simulating the race from predicted positions.

Counting both from the same simulated races is what keeps them consistent:
p_win <= p_podium always holds, and p_win sums to 1 because exactly one car wins
each race. Two separate classifiers give you neither guarantee.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DNF_SCORE = 99.0  # sorts behind every finisher
DEFAULT_N_SIMS = 20_000
DEFAULT_SEED = 42


def simulate_race(
    pred_positions,
    sigma: float,
    dnf_probs,
    n_sims: int = DEFAULT_N_SIMS,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Run n_sims races and count how often each driver wins or podiums.

    sigma is the spread of the race-day noise in positions. Take it from the
    backtest residuals rather than guessing - it sets how confident the output
    looks. dnf_probs is the shrunk retirement rate per driver.

    Seeded, because a locked prediction has to be reproducible.
    """
    pred = np.asarray(pred_positions, dtype="float64").ravel()
    dnf = np.asarray(dnf_probs, dtype="float64").ravel()
    n_drivers = pred.size

    if dnf.size != n_drivers:
        raise ValueError(f"dnf_probs has {dnf.size} entries, expected {n_drivers}")
    if not np.isfinite(pred).all():
        raise ValueError("pred_positions contains non-finite values")
    if not np.isfinite(dnf).all():
        raise ValueError("dnf_probs contains non-finite values")
    if sigma <= 0 or not np.isfinite(sigma):
        raise ValueError(f"sigma must be finite and positive, got {sigma!r}")
    if ((dnf < 0) | (dnf > 1)).any():
        raise ValueError("dnf_probs must lie in [0, 1]")
    if n_drivers < 3:
        raise ValueError("need at least 3 drivers to define a podium")

    rng = np.random.default_rng(seed)

    # One draw for pace noise, one for retirements, both over the whole
    # n_sims x n_drivers grid. Looping 20k races in Python is not worth it.
    scores = pred[None, :] + rng.normal(0.0, sigma, size=(n_sims, n_drivers))
    retired = rng.random(size=(n_sims, n_drivers)) < dnf[None, :]
    scores = np.where(retired, DNF_SCORE, scores)

    # argsort twice turns scores into ranks; rank 1 wins.
    order = np.argsort(scores, axis=1, kind="stable")
    ranks = np.empty_like(order)
    np.put_along_axis(ranks, order, np.arange(1, n_drivers + 1)[None, :], axis=1)

    p_win = (ranks == 1).mean(axis=0)
    p_podium = (ranks <= 3).mean(axis=0)

    # Ties break deterministically, so there is exactly one winner per race.
    total = float(p_win.sum())
    assert abs(total - 1.0) < 1e-6, f"p_win sums to {total!r}, expected 1.0"
    assert (p_win <= p_podium + 1e-12).all(), "p_win exceeds p_podium for some driver"
    assert (p_podium <= 1.0 + 1e-12).all(), "p_podium exceeds 1"

    return pd.DataFrame(
        {
            "p_win": p_win,
            "p_podium": p_podium,
            "p_dnf_simulated": retired.mean(axis=0),
            "mean_sim_position": np.where(retired, np.nan, ranks).mean(axis=0),
        }
    )


def simulate_frame(
    frame: pd.DataFrame,
    pred_col: str = "predicted_position",
    dnf_col: str = "dnf_rate_shrunk",
    sigma: float = 3.0,
    n_sims: int = DEFAULT_N_SIMS,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """simulate_race for a driver frame, with the result columns joined back on."""
    sims = simulate_race(
        frame[pred_col].to_numpy(),
        sigma=sigma,
        dnf_probs=frame[dnf_col].to_numpy(),
        n_sims=n_sims,
        seed=seed,
    )
    out = frame.reset_index(drop=True).copy()
    for col in sims.columns:
        out[col] = sims[col].to_numpy()
    return out


__all__ = ["DEFAULT_N_SIMS", "DEFAULT_SEED", "DNF_SCORE", "simulate_frame", "simulate_race"]
