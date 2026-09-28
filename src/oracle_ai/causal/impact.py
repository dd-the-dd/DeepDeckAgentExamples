from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def _number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _count(value: Any) -> int:
    return len(value) if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) else 0


def _mana_count(player: Mapping[str, Any]) -> float:
    mana_pool = player.get("manaPool")
    if not isinstance(mana_pool, Sequence) or isinstance(mana_pool, (str, bytes)):
        return 0.0
    total = 0.0
    for entry in mana_pool:
        if isinstance(entry, Mapping):
            total += max(1.0, _number(entry.get("amount", 1.0)))
        else:
            total += 1.0
    return total


def _counter_total(player: Mapping[str, Any]) -> float:
    counters = player.get("counters")
    if not isinstance(counters, Mapping):
        return 0.0
    return sum(max(0.0, _number(value)) for value in counters.values())


def state_impact_signature(state: Mapping[str, Any]) -> tuple[float, ...]:
    """Return a deliberately coarse net-impact signature.

    It ignores turn/event sequence counters so a tap/untap loop that returns to
    the same material state has zero net impact.  Counterspells, zone movement,
    damage, card use, mana changes and permanent changes remain visible.
    """

    values: list[float] = []
    players = state.get("players")
    if isinstance(players, Sequence) and not isinstance(players, (str, bytes)):
        ordered = sorted(
            (player for player in players if isinstance(player, Mapping)),
            key=lambda player: str(player.get("id", "")),
        )
        for player in ordered:
            values.extend(
                (
                    _number(player.get("life")),
                    float(_count(player.get("hand"))),
                    float(_count(player.get("library"))),
                    float(_count(player.get("graveyard"))),
                    float(_count(player.get("exile"))),
                    float(_count(player.get("battlefield"))),
                    _mana_count(player),
                    _counter_total(player),
                )
            )
    values.append(float(_count(state.get("stack"))))
    return tuple(values)


def causal_impact_score(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> float:
    """Measure net material impact without assigning strategic value.

    A result of zero means the tracked material state returned to the same place;
    it does *not* mean passing priority or preserving hidden information is bad.
    The score is intended for ghost-loop detection and exploration filtering only.
    """

    left = state_impact_signature(before)
    right = state_impact_signature(after)
    size = max(len(left), len(right))
    padded_left = (*left, *((0.0,) * (size - len(left))))
    padded_right = (*right, *((0.0,) * (size - len(right))))
    return sum(abs(a - b) for a, b in zip(padded_left, padded_right, strict=True))
