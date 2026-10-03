import pytest

from oracle_ai.decision_choices import expand_policy_actions


def test_number_selection_expands_only_inside_the_policy_boundary() -> None:
    actions = expand_policy_actions({
        "choice": {
            "kind": "numberSelection",
            "decisionId": "loopIterations",
            "minimum": 0,
            "maximum": 2,
        },
        "options": [{
            "id": "choose-number:loopIterations",
            "kind": "chooseResolution",
            "playerId": "player-1",
        }],
    })

    assert [action["_numberValue"] for action in actions] == [0, 1, 2]
    assert {action["_engineActionId"] for action in actions} == {
        "choose-number:loopIterations"
    }
    assert [action["decisions"]["loopIterations"] for action in actions] == [0, 1, 2]


def test_number_selection_rejects_an_unbounded_policy_expansion() -> None:
    with pytest.raises(ValueError, match="too large"):
        expand_policy_actions({
            "choice": {
                "kind": "numberSelection",
                "minimum": 0,
                "maximum": 1_001,
            },
            "options": [{"id": "choose-number"}],
        })


def test_sideboarding_expands_grouped_swaps_with_exact_main_deck_ids() -> None:
    main = [
        "player-1:main-a:0",
        "player-1:main-a:1",
        "player-1:main-b:0",
        "player-1:main-b:1",
    ]
    sideboard = ["player-1:side-c:0", "player-1:side-c:1"]
    actions = expand_policy_actions(
        {
            "kind": "sideboarding",
            "choice": {
                "kind": "cardSelection",
                "decisionId": "sideboard:configure:cards",
                "candidateCardInstanceIds": main + sideboard,
            },
            "options": [
                {
                    "id": "confirm-sideboard",
                    "label": "Confirm",
                    "decisions": {"initialMainDeckIds": main},
                }
            ],
        },
        {"main-a": "Main A", "main-b": "Main B", "side-c": "Side C"},
    )

    assert len(actions) == 3
    assert actions[0]["_cardInstanceIds"] == main
    assert actions[1]["label"] == "Sideboard 2 Side C for Main A"
    assert set(actions[1]["_cardInstanceIds"]) == {
        "player-1:main-b:0",
        "player-1:main-b:1",
        "player-1:side-c:0",
        "player-1:side-c:1",
    }
    assert actions[1]["decisions"]["initialMainDeckIds"] == main


def test_sideboarding_exposes_complete_known_matchup_plans() -> None:
    main = [f"player-1:main-{index // 2}:{index}" for index in range(6)]
    sideboard = [
        "player-1:target-a:0",
        "player-1:target-a:1",
        "player-1:target-b:0",
    ]
    decision = {
        "kind": "sideboarding",
        "choice": {
            "kind": "cardSelection",
            "decisionId": "sideboard:configure:cards",
            "candidateCardInstanceIds": main + sideboard,
        },
        "options": [
            {
                "id": "confirm-sideboard",
                "decisions": {"initialMainDeckIds": main},
            }
        ],
    }

    actions = expand_policy_actions(
        decision,
        {
            "main-0": "Main 0",
            "main-1": "Main 1",
            "main-2": "Main 2",
            "target-a": "Target A",
            "target-b": "Target B",
        },
        ("Target A", "Target B"),
        ("Main 1",),
    )
    known_plan = next(
        action
        for action in actions
        if str(action.get("id", "")).startswith("confirm-sideboard:known-plan")
    )

    assert len(known_plan["_cardInstanceIds"]) == len(main)
    assert set(sideboard) <= set(known_plan["_cardInstanceIds"])
    assert "player-1:main-1:2" not in known_plan["_cardInstanceIds"]
    assert "player-1:main-1:3" not in known_plan["_cardInstanceIds"]
    assert "Main 1" in known_plan["label"]
    assert known_plan["label"].startswith("Known matchup plan: bring in 3 cards")
    known_plans = [action for action in actions if ":known-plan:" in str(action.get("id", ""))]
    assert len({tuple(sorted(action["_cardInstanceIds"])) for action in known_plans}) == len(
        known_plans
    )
