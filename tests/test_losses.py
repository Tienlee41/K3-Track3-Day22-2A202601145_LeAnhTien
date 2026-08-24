import numpy as np
import pytest

from preference_lab.losses import dpo_loss, orpo_loss


def test_dpo_loss_matches_expected_value() -> None:
    loss = dpo_loss(
        np.array([-0.5]),
        np.array([-1.5]),
        np.array([-0.6]),
        np.array([-1.0]),
        beta=0.1,
    )
    expected = np.logaddexp(0.0, -0.1 * ((-0.5 + 1.5) - (-0.6 + 1.0)))
    assert loss == pytest.approx(float(expected))


def test_dpo_loss_is_stable_for_extreme_margin() -> None:
    loss = dpo_loss([-10_000.0], [0.0], [0.0], [0.0], beta=1.0)
    assert np.isfinite(loss)
    assert loss == pytest.approx(10_000.0)


def test_orpo_loss_matches_odds_ratio_objective() -> None:
    chosen = -0.5
    rejected = -1.5
    chosen_log_odds = chosen - np.log1p(-np.exp(chosen))
    rejected_log_odds = rejected - np.log1p(-np.exp(rejected))
    expected = 1.0 + 0.1 * np.logaddexp(0.0, -(chosen_log_odds - rejected_log_odds))

    loss = orpo_loss([1.0], [chosen], [rejected], lambda_orpo=0.1)
    assert loss == pytest.approx(float(expected))


@pytest.mark.parametrize("name", ["shape", "parameter", "positive_logp"])
def test_losses_validate_inputs(name: str) -> None:
    if name == "shape":
        with pytest.raises(ValueError, match="same shape"):
            dpo_loss([-1.0], [-2.0, -3.0], [-1.0], [-2.0], beta=0.1)
    elif name == "parameter":
        with pytest.raises(ValueError, match="beta"):
            dpo_loss([-1.0], [-2.0], [-1.0], [-2.0], beta=0.0)
    else:
        with pytest.raises(ValueError, match="must be <= 0"):
            orpo_loss([1.0], [0.1], [-1.0], lambda_orpo=0.1)
