from __future__ import annotations

import pytest
from deepdeck_agent import Decision, Game

from deepdeck_examples import ForgeReferenceAgent, ForgeReferenceProfile


def card(
    instance_id: str,
    name: str,
    type_line: str,
    *,
    controller: str = "p1",
    mana_cost: str = "",
    power: int = 0,
    toughness: int = 0,
    rules: list | None = None,
    tapped: bool = False,
) -> dict:
    return {
        "instanceId": instance_id,
        "owner": controller,
        "controller": controller,
        "definition": {
            "id": name.casefold().replace(" ", "-"),
            "name": name,
            "typeLine": type_line,
            "manaCost": mana_cost,
            "power": str(power),
            "toughness": str(toughness),
            "rules": rules or [],
        },
        "tapped": tapped,
        "counters": {},
    }


def game(
    *,
    hand: list[dict] | None = None,
    battlefield: list[dict] | None = None,
    opponent_battlefield: list[dict] | None = None,
    life: int = 20,
    opponent_life: int = 20,
    active_player: int = 0,
    stack: list[dict] | None = None,
) -> Game:
    return Game(
        {
            "turnNumber": 4,
            "activePlayer": active_player,
            "step": "precombatMain",
            "players": [
                {
                    "id": "p1",
                    "life": life,
                    "hand": hand or [],
                    "battlefield": battlefield or [],
                    "library": [],
                    "graveyard": [],
                    "exile": [],
                    "sideboard": [],
                    "commandZone": [],
                    "manaPool": [],
                },
                {
                    "id": "p2",
                    "life": opponent_life,
                    "hand": [],
                    "battlefield": opponent_battlefield or [],
                    "library": [],
                    "graveyard": [],
                    "exile": [],
                    "sideboard": [],
                    "commandZone": [],
                    "manaPool": [],
                },
            ],
            "stack": stack or [],
        },
        "p1",
    )


def decision(kind: str, options: list[dict], current_game: Game) -> Decision:
    return Decision("request", "p1", {"kind": kind, "options": options}, current_game)


@pytest.mark.asyncio
async def test_keeps_a_functional_opening_hand_and_mulligans_one_land() -> None:
    keep_hand = [card(f"land-{index}", "Island", "Basic Land") for index in range(3)]
    keep_hand.append(card("spell", "Bear", "Creature", mana_cost="{1}{G}", power=2, toughness=2))
    options = [{"id": "keep", "kind": "keepHand"}, {"id": "mull", "kind": "takeMulligan"}]
    agent = ForgeReferenceAgent()

    kept = await agent.make_decision(decision("mulligan", options, game(hand=keep_hand)))
    mulliganed = await agent.make_decision(
        decision("mulligan", options, game(hand=keep_hand[2:]))
    )

    assert kept.action_id == "keep"
    assert mulliganed.action_id == "mull"


@pytest.mark.asyncio
async def test_plays_a_land_before_casting_a_spell() -> None:
    land = card("land", "Forest", "Basic Land")
    bear = card("bear", "Bear", "Creature", mana_cost="{1}{G}", power=2, toughness=2)
    choice = decision(
        "priority",
        [
            {"id": "cast", "kind": "castSpell", "cardInstanceId": "bear"},
            {"id": "land", "kind": "playLand", "cardInstanceId": "land"},
            {"id": "pass", "kind": "passPriority"},
        ],
        game(hand=[land, bear]),
    )

    assert (await ForgeReferenceAgent().make_decision(choice)).action_id == "land"


@pytest.mark.asyncio
async def test_removal_prefers_the_most_valuable_opposing_target() -> None:
    removal = card(
        "removal",
        "Clean Removal",
        "Instant",
        rules=[{"effects": [{"kind": "destroyPermanent"}]}],
    )
    small = card("small", "Small", "Creature", controller="p2", power=1, toughness=1)
    large = card("large", "Large", "Creature", controller="p2", power=6, toughness=6)
    choice = decision(
        "priority",
        [
            {
                "id": "small-target",
                "kind": "castSpell",
                "cardInstanceId": "removal",
                "targets": {"target": {"kind": "permanent", "instanceId": "small"}},
            },
            {
                "id": "large-target",
                "kind": "castSpell",
                "cardInstanceId": "removal",
                "targets": {"target": {"kind": "permanent", "instanceId": "large"}},
            },
            {"id": "pass", "kind": "passPriority"},
        ],
        game(hand=[removal], opponent_battlefield=[small, large]),
    )

    assert (await ForgeReferenceAgent().make_decision(choice)).action_id == "large-target"


@pytest.mark.asyncio
async def test_aggressive_profile_attacks_where_cautious_profile_stops() -> None:
    attacker = card("attacker", "Attacker", "Creature", power=3, toughness=2)
    blocker = card("blocker", "Blocker", "Creature", controller="p2", power=2, toughness=3)
    options = [
        {
            "id": "attack",
            "kind": "declareAttacker",
            "attackerId": "attacker",
            "targets": {"defender": {"kind": "player", "playerId": "p2"}},
        },
        {"id": "finish", "kind": "finishAttackers"},
    ]
    choice = decision(
        "attackers",
        options,
        game(battlefield=[attacker], opponent_battlefield=[blocker]),
    )

    cautious = ForgeReferenceAgent(ForgeReferenceProfile.CAUTIOUS)
    aggressive = ForgeReferenceAgent(ForgeReferenceProfile.AGGRESSIVE)

    assert (await cautious.make_decision(choice)).action_id == "finish"
    assert (await aggressive.make_decision(choice)).action_id == "attack"


@pytest.mark.asyncio
async def test_blocks_lethal_damage_even_with_an_unfavorable_trade() -> None:
    blocker = card("blocker", "Blocker", "Creature", power=1, toughness=1)
    attacker = card("attacker", "Attacker", "Creature", controller="p2", power=5, toughness=5)
    choice = decision(
        "blockers",
        [
            {
                "id": "block",
                "kind": "declareBlocker",
                "attackerId": "attacker",
                "blockerId": "blocker",
            },
            {"id": "finish", "kind": "finishBlockers"},
        ],
        game(battlefield=[blocker], opponent_battlefield=[attacker], life=5),
    )

    assert (await ForgeReferenceAgent().make_decision(choice)).action_id == "block"


@pytest.mark.asyncio
async def test_discards_excess_land_before_a_spell() -> None:
    lands = [card(f"land-{index}", "Mountain", "Basic Land") for index in range(5)]
    spell = card("spell", "Dragon", "Creature", mana_cost="{4}{R}", power=5, toughness=5)
    choice = decision(
        "discard",
        [
            {"id": "discard-land", "kind": "discard", "cardInstanceId": "land-0"},
            {"id": "discard-spell", "kind": "discard", "cardInstanceId": "spell"},
        ],
        game(hand=[*lands, spell]),
    )

    assert (await ForgeReferenceAgent().make_decision(choice)).action_id == "discard-land"
