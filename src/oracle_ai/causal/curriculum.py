from __future__ import annotations

import math
from dataclasses import dataclass


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


@dataclass(frozen=True)
class CausalCurriculumConfig:
    """Controls the transition from causal understanding to strategic RL.

    The causal loss remains fully weighted.  Only strategic learning is gated by
    global validation maturity and local prediction confidence.
    """

    validation_warmup_loss: float = 1.0
    validation_target_loss: float = 0.05
    confidence_temperature: float = 0.25
    strategy_floor: float = 0.05
    exploration_floor: float = 0.05
    exploration_power: float = 1.0

    def __post_init__(self) -> None:
        if self.validation_target_loss < 0.0:
            raise ValueError("validation_target_loss must be nonnegative")
        if self.validation_warmup_loss <= self.validation_target_loss:
            raise ValueError(
                "validation_warmup_loss must be greater than validation_target_loss"
            )
        if self.confidence_temperature <= 0.0:
            raise ValueError("confidence_temperature must be positive")
        if not 0.0 <= self.strategy_floor <= 1.0:
            raise ValueError("strategy_floor must be between zero and one")
        if not 0.0 <= self.exploration_floor <= 1.0:
            raise ValueError("exploration_floor must be between zero and one")
        if self.exploration_power <= 0.0:
            raise ValueError("exploration_power must be positive")


@dataclass(frozen=True)
class CausalCurriculumWeights:
    causal: float
    strategy: float
    exploration: float
    maturity: float
    local_confidence: float


def causal_maturity(
    validation_loss: float,
    *,
    warmup_loss: float,
    target_loss: float,
) -> float:
    """Map held-out causal loss to [0, 1] without using training loss.

    A validation loss at or above ``warmup_loss`` means the world model is still
    immature.  A loss at or below ``target_loss`` means the strategic learner may
    use the causal representation at full strength.
    """

    if warmup_loss <= target_loss:
        raise ValueError("warmup_loss must be greater than target_loss")
    if validation_loss <= target_loss:
        return 1.0
    if validation_loss >= warmup_loss:
        return 0.0
    return _clamp01((warmup_loss - validation_loss) / (warmup_loss - target_loss))


def causal_confidence(
    prediction_error: float,
    *,
    temperature: float,
) -> float:
    """Convert normalized local transition error into a strategic-learning gate."""

    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    error = max(0.0, float(prediction_error))
    return _clamp01(math.exp(-error / temperature))


def curriculum_weights(
    validation_loss: float,
    prediction_error: float,
    config: CausalCurriculumConfig,
) -> CausalCurriculumWeights:
    """Return weights for one transition.

    ``causal`` intentionally remains 1.0.  Winning/value/policy losses should be
    multiplied by ``strategy``.  Exploration should use ``exploration`` only as a
    behavior-policy bonus; it must not alter terminal rewards.
    """

    maturity = causal_maturity(
        validation_loss,
        warmup_loss=config.validation_warmup_loss,
        target_loss=config.validation_target_loss,
    )
    confidence = causal_confidence(
        prediction_error,
        temperature=config.confidence_temperature,
    )
    strategic_signal = (maturity**2) * confidence
    strategy = config.strategy_floor + (1.0 - config.strategy_floor) * strategic_signal
    exploration = config.exploration_floor + (
        1.0 - config.exploration_floor
    ) * ((1.0 - maturity) ** config.exploration_power)
    return CausalCurriculumWeights(
        causal=1.0,
        strategy=_clamp01(strategy),
        exploration=_clamp01(exploration),
        maturity=maturity,
        local_confidence=confidence,
    )
