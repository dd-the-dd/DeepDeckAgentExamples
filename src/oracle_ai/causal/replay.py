from __future__ import annotations


def _nonnegative(value: float) -> float:
    return max(0.0, float(value))


def causal_replay_priority(
    prediction_error: float,
    actual_impact: float,
    novelty: float,
    *,
    impact_floor: float = 0.05,
) -> float:
    """Prioritize transitions that are surprising, meaningful and underexplored."""

    if impact_floor < 0.0:
        raise ValueError("impact_floor must be nonnegative")
    error = _nonnegative(prediction_error)
    impact = max(impact_floor, _nonnegative(actual_impact))
    novelty_score = _nonnegative(novelty)
    return error * impact * novelty_score


def exploration_bonus(
    uncertainty: float,
    predicted_impact: float,
    novelty: float,
    curriculum_exploration_weight: float,
) -> float:
    """Behavior-policy bonus for useful causal exploration.

    This bonus must not be written into the episode reward.  It is only meant to
    bias action sampling toward uncertain actions that are expected to change the
    game and exercise an underrepresented causal pattern.
    """

    return (
        _nonnegative(uncertainty)
        * _nonnegative(predicted_impact)
        * _nonnegative(novelty)
        * _nonnegative(curriculum_exploration_weight)
    )
