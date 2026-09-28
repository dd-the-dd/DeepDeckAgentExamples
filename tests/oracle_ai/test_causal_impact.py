from __future__ import annotations

from oracle_ai.causal.impact import causal_impact_score
from oracle_ai.causal.replay import causal_replay_priority, exploration_bonus


def _state(*, life: int = 20, hand: int = 3, graveyard: int = 0, mana: int = 0) -> dict:
    return {
        "players": [
            {
                "id": "p1",
                "life": life,
                "hand": [{} for _ in range(hand)],
                "library": [{} for _ in range(40)],
                "graveyard": [{} for _ in range(graveyard)],
                "exile": [],
                "battlefield": [],
                "manaPool": [{} for _ in range(mana)],
                "counters": {},
            }
        ],
        "stack": [],
    }


def test_materially_identical_state_has_no_net_impact() -> None:
    state = _state()
    assert causal_impact_score(state, state) == 0.0


def test_counterspell_like_card_and_stack_changes_have_impact() -> None:
    before = _state(hand=4)
    before["stack"] = [{"id": "spell-a"}, {"id": "counter"}]
    after = _state(hand=3, graveyard=2)
    assert causal_impact_score(before, after) > 0.0


def test_damage_and_mill_are_material_impact() -> None:
    before = _state(life=20, graveyard=0)
    after = _state(life=17, graveyard=13)
    assert causal_impact_score(before, after) == 16.0


def test_replay_priority_prefers_surprising_impactful_novel_events() -> None:
    useful = causal_replay_priority(0.8, 4.0, 1.0)
    mastered = causal_replay_priority(0.05, 4.0, 1.0)
    ghost = causal_replay_priority(0.8, 0.0, 1.0)
    assert useful > mastered
    assert useful > ghost


def test_exploration_bonus_requires_all_three_terms() -> None:
    assert exploration_bonus(1.0, 1.0, 1.0, 1.0) == 1.0
    assert exploration_bonus(1.0, 0.0, 1.0, 1.0) == 0.0
    assert exploration_bonus(0.0, 1.0, 1.0, 1.0) == 0.0
