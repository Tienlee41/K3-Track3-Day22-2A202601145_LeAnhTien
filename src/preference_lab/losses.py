from __future__ import annotations

from typing import cast

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _as_matching_arrays(*values: ArrayLike) -> tuple[NDArray[np.float64], ...]:
    arrays = tuple(np.asarray(value, dtype=np.float64) for value in values)
    if not arrays or arrays[0].size == 0:
        raise ValueError("loss inputs must not be empty")
    if any(array.shape != arrays[0].shape for array in arrays[1:]):
        raise ValueError("all loss inputs must have the same shape")
    if any(not np.all(np.isfinite(array)) for array in arrays):
        raise ValueError("loss inputs must contain only finite values")
    return arrays


def _log_one_minus_exp(log_probabilities: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute log(1 - exp(x)) stably for log probabilities x <= 0."""
    clipped = np.minimum(log_probabilities, -np.finfo(np.float64).eps)
    cutoff = -np.log(2.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return cast(
            NDArray[np.float64],
            np.where(
                clipped < cutoff,
                np.log1p(-np.exp(clipped)),
                np.log(-np.expm1(clipped)),
            ),
        )


def dpo_loss(
    policy_chosen_logps: ArrayLike,
    policy_rejected_logps: ArrayLike,
    ref_chosen_logps: ArrayLike,
    ref_rejected_logps: ArrayLike,
    beta: float,
) -> float:
    """Compute the mean, numerically stable DPO loss for a batch."""
    if not np.isfinite(beta) or beta <= 0.0:
        raise ValueError("beta must be a positive finite number")

    policy_chosen, policy_rejected, ref_chosen, ref_rejected = _as_matching_arrays(
        policy_chosen_logps,
        policy_rejected_logps,
        ref_chosen_logps,
        ref_rejected_logps,
    )
    preference_logits = beta * ((policy_chosen - policy_rejected) - (ref_chosen - ref_rejected))
    # -log(sigmoid(x)) == log(1 + exp(-x)); logaddexp avoids overflow.
    return float(np.mean(np.logaddexp(0.0, -preference_logits)))


def orpo_loss(
    sft_nll: ArrayLike,
    chosen_logps: ArrayLike,
    rejected_logps: ArrayLike,
    lambda_orpo: float,
) -> float:
    """Compute mean SFT NLL plus the ORPO odds-ratio preference penalty."""
    if not np.isfinite(lambda_orpo) or lambda_orpo < 0.0:
        raise ValueError("lambda_orpo must be a non-negative finite number")

    nll, chosen, rejected = _as_matching_arrays(sft_nll, chosen_logps, rejected_logps)
    if np.any(chosen > 0.0) or np.any(rejected > 0.0):
        raise ValueError("chosen_logps and rejected_logps must be <= 0")

    chosen_log_odds = chosen - _log_one_minus_exp(chosen)
    rejected_log_odds = rejected - _log_one_minus_exp(rejected)
    log_odds_ratio = chosen_log_odds - rejected_log_odds
    preference_penalty = np.logaddexp(0.0, -log_odds_ratio)
    return float(np.mean(nll + lambda_orpo * preference_penalty))
