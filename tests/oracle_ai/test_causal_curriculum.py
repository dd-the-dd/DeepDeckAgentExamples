from __future__ import annotations

import pytest

from oracle_ai.causal.curriculum import (
    CausalCurriculumConfig,
    causal_confidence,
    causal_maturity,
    curriculum_weights,
)


def test_causal_maturity_uses_held_out_loss_bounds() -> None:
    assert causal_maturity(1.5, warmup_loss=1.0, target_loss=0.1) == 0.0
    assert causal_maturity(0.05, warmup_loss=1.0, target_loss=0.1) == 1.0
    midpoint = causal_maturity(0.55, warmup_loss=1.0, target_loss=0.1)
    assert midpoint == pytest.approx(0.5)


def test_local_prediction_error_reduces_strategic_confidence() -> None:
    good = causal_confidence(0.0, temperature=0.25)
    bad = causal_confidence(1.0, temperature=0.25)
    assert good == 1.0
    assert bad < 0.02


def test_strategy_is_gated_until_causal_model_is_mature() -> None:
    config = CausalCurriculumConfig(
        validation_warmup_loss=1.0,
        validation_target_loss=0.1,
        confidence_temperature=0.25,
        strategy_floor=0.05,
        exploration_floor=0.05,
    )
    immature = curriculum_weights(1.0, 0.0, config)
    mature = curriculum_weights(0.1, 0.0, config)
    locally_wrong = curriculum_weights(0.1, 1.0, config)

    assert immature.causal == 1.0
    assert immature.strategy == pytest.approx(0.05)
    assert immature.exploration == 1.0
    assert mature.strategy == 1.0
    assert mature.exploration == pytest.approx(0.05)
    assert locally_wrong.strategy < 0.08


def test_invalid_curriculum_bounds_are_rejected() -> None:
    with pytest.raises(ValueError):
        CausalCurriculumConfig(
            validation_warmup_loss=0.1,
            validation_target_loss=0.1,
        )
