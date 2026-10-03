from __future__ import annotations

import json
import random
from collections import Counter

from oracle_ai.training.core import DecisionStep
from oracle_ai.training.curriculum import AdaptiveCurriculumSampler, curriculum_catalog
from oracle_ai.training.environments import Matchup, RustSessionEnvironment
from oracle_ai.training.league import RandomTrainingMatchupSampler
from oracle_ai.training.rl_v13 import _prune_rl_checkpoints


def deck(name: str) -> list[dict[str, object]]:
    return [
        {
            "id": f"{name}-{index}",
            "name": f"{name} card {index}",
            "typeLine": "Basic Land — Island" if index < 20 else "Sorcery",
            "manaCost": "" if index < 20 else "{3}{U}",
            "rules": [],
            "isSideboard": False,
            "isCommander": False,
            "isToken": False,
            "isGamePiece": False,
            "sourceSessionId": name,
        }
        for index in range(60)
    ]


def sampler(*scenario_ids: str) -> AdaptiveCurriculumSampler:
    decks = {
        "Deck A": deck("a"),
        "Deck B": deck("b"),
        "Dimir tempo": deck("dimir"),
        "Radkos reanimator": deck("rakdos"),
        "Mono-Green Lands": deck("lands"),
        "Sneak and Show": deck("sneak"),
        "Mono-White Initiative": deck("initiative"),
        "Izzet Delver": deck("delver"),
    }
    base = RandomTrainingMatchupSampler(
        {"formats": ["legacy"], "playerCounts": [2], "maxTurns": 80}, decks
    )
    return AdaptiveCurriculumSampler(
        {"scenarioIds": list(scenario_ids), "adaptive": True}, decks, base
    )


def test_catalog_covers_requested_focused_training_families() -> None:
    scenarios = curriculum_catalog()["scenarios"]
    families = {scenario["family"] for scenario in scenarios}

    assert {
        "opening-hand",
        "free-cast",
        "predictable-opponent",
        "combo",
        "known-combo",
        "sideboard",
    } <= families
    assert len(scenarios) == 21
    assert all(1 <= scenario["difficulty"] <= 5 for scenario in scenarios)


def test_reanimator_hand_clinic_uses_exact_candidate_pool_and_dense_roles() -> None:
    catalog = curriculum_catalog()["scenarios"]
    scenario = next(item for item in catalog if item["id"] == "opening-hand-reanimator-clinic")
    reanimator = deck("rakdos")
    for card, name in zip(reanimator, scenario["opening_hand_candidate_cards"], strict=False):
        card["name"] = name
    decks = {"Radkos reanimator": reanimator, "Deck B": deck("b")}
    base = RandomTrainingMatchupSampler(
        {"formats": ["legacy"], "playerCounts": [2], "maxTurns": 80}, decks
    )
    focused = AdaptiveCurriculumSampler(
        {"scenarioIds": [scenario["id"]], "adaptive": True}, decks, base
    )

    matchup = focused.build_scenario(scenario["id"], 7)
    learner_cards = matchup.setup["players"][0]["cards"]

    assert Counter(card["name"] for card in learner_cards) == Counter(
        scenario["opening_hand_candidate_cards"]
    )
    assert matchup.anchor_opening_hand_pool_size == 20
    assert len(matchup.opening_hand_target_roles) == 4


def test_opening_hand_roles_measure_available_and_selected_components() -> None:
    matchup = Matchup(
        id="hand-clinic",
        setup={"players": []},
        learner_player_id="player-1",
        opponent_player_id="player-2",
        opening_hand_target_roles=(("Swamp",), ("Faithless Looting",), ("Reanimate",)),
    )
    environment = RustSessionEnvironment("http://engine.test", {matchup.id: matchup})
    environment.current_matchup = matchup
    actions = [
        {
            "decisions": {
                "openingHandCandidate": {"definition": {"name": name}}
            }
        }
        for name in ("Swamp", "Faithless Looting", "Griselbrand")
    ]
    environment._initialize_opening_hand_roles(DecisionStep({}, actions, 0.0, False))
    environment.opening_hand_selected_card_names = ["Swamp", "Faithless Looting"]

    assert environment.opening_hand_roles_available == 2
    assert environment._opening_hand_role_progress() == 2
    environment.close()


def test_fast_win_damage_progress_is_dense_and_capped() -> None:
    matchup = Matchup(
        id="damage-clinic",
        setup={
            "players": [
                {"id": "player-1", "startingLife": 20},
                {"id": "player-2", "startingLife": 20},
            ]
        },
        learner_player_id="player-1",
        opponent_player_id="player-2",
        objective="fast-win",
        training_anchor_player_ids=("player-2",),
    )
    environment = RustSessionEnvironment("http://engine.test", {matchup.id: matchup})
    environment.current_matchup = matchup

    assert environment._damage_progress(
        {"state": {"players": [{"id": "player-2", "life": 15}]}}
    ) == 0.25
    assert environment._damage_progress(
        {"state": {"players": [{"id": "player-2", "life": -3}]}}
    ) == 1.0
    environment.close()


def test_zero_cost_scenario_changes_only_nonland_mana_costs() -> None:
    matchup = sampler("zero-cost-sequencing").sample(random.Random(7))
    cards = matchup.setup["players"][0]["cards"]

    assert matchup.scenario_id == "zero-cost-sequencing"
    assert matchup.anchor_opening_hand_pool_size == 30
    assert {card["manaCost"] for card in cards if "land" not in str(card["typeLine"]).lower()} == {"{0}"}
    assert {card["manaCost"] for card in cards if "land" in str(card["typeLine"]).lower()} == {""}


def test_progressive_scenarios_increase_search_space_and_tighten_clock() -> None:
    catalog = {scenario["id"]: scenario for scenario in curriculum_catalog()["scenarios"]}

    assert [
        catalog[scenario_id]["opening_hand_pool_size"]
        for scenario_id in (
            "opening-hand-sprint-easy",
            "opening-hand-sprint-medium",
            "opening-hand-sprint-hard",
        )
    ] == [20, 40, 60]
    assert [
        catalog[scenario_id]["target_round"]
        for scenario_id in (
            "zero-cost-sequencing-easy",
            "zero-cost-sequencing",
            "zero-cost-sequencing-hard",
        )
    ] == [6, 4, 3]
    assert catalog["combo-rehearsal-guided"]["difficulty"] == 2
    assert catalog["combo-rehearsal-expert"]["difficulty"] == 5


def test_miracle_opponent_is_non_neural_seeded_pressure_deck() -> None:
    matchup = sampler("miracle-pressure-hard").sample(random.Random(11))
    opponent_cards = matchup.setup["players"][1]["cards"]

    assert matchup.training_anchor_player_ids == ("player-2",)
    assert matchup.target_round == 4
    assert {card["manaCost"] for card in opponent_cards} == {
        "{1}{W}",
        "{3}{R}",
        "{5}{U}",
        "{7}{B}",
    }
    assert {
        card["rules"][0]["effects"][0]["kind"] for card in opponent_cards
    } == {"gainLife", "dealDamageToEachOpponent", "drawCards", "discardCards"}
    assert all(
        any(rule.get("ability", {}).get("kind") == "miracle" for rule in card["rules"])
        for card in opponent_cards
    )


def test_sideboard_scenario_uses_distinct_known_decks_and_short_match() -> None:
    matchup = sampler("known-sideboard-match").sample(random.Random(3))

    assert matchup.game_mode == "legacy"
    assert matchup.max_turns == 20
    assert matchup.deck_names == ("Dimir tempo", "Radkos reanimator")
    assert "predictable opponent" in matchup.setup["players"][1]["name"]
    assert matchup.sideboard_target_card_names == (
        "Duress",
        "Force of Negation",
        "Sheoldred's Edict",
    )
    assert matchup.sideboard_cut_card_names == (
        "Snuff Out",
        "Kaito, Bane of Nightmares",
        "Fatal Push",
    )


def test_sideboard_metric_counts_cards_brought_in_not_final_deck_size() -> None:
    action = {
        "decisions": {
            "initialMainDeckIds": ["main-1", "main-2", "main-3"],
            "sideboard:0:player-1:configure:cards": [
                "main-1",
                "main-2",
                "side-1",
            ],
        }
    }

    assert RustSessionEnvironment._sideboard_cards_brought_in(action) == 1


def test_sideboard_mastery_counts_only_matchup_target_cards() -> None:
    matchup = sampler("known-sideboard-match").sample(random.Random(3))
    matchup.setup["players"][0]["cards"][0].update(
        {"id": "duress-id", "name": "Duress", "isSideboard": True}
    )
    matchup.setup["players"][0]["cards"][1].update(
        {"id": "snuff-id", "name": "Snuff Out", "isSideboard": False}
    )
    environment = RustSessionEnvironment("http://engine.test", {matchup.id: matchup})
    environment.current_matchup = matchup
    environment.sideboard_cut_cards_expected = 1
    action = {
        "decisions": {
            "initialMainDeckIds": ["player-1:snuff-id:0"],
            "sideboard:0:player-1:configure:cards": ["player-1:duress-id:0"],
        }
    }

    assert environment._sideboard_target_cards_brought_in(action) == 1
    assert environment._sideboard_cut_cards_taken_out(action) == 1
    environment.close()


def test_adaptive_sampler_increases_weight_for_failed_scenario() -> None:
    focused = sampler("opening-hand-sprint-easy", "combo-rehearsal")
    initial = focused.stats()
    for _ in range(8):
        focused.observe("combo-rehearsal", False)
        focused.observe("opening-hand-sprint-easy", True)
    updated = focused.stats()

    assert updated["combo-rehearsal"]["weight"] > initial["combo-rehearsal"]["weight"]
    assert updated["combo-rehearsal"]["weight"] > updated["opening-hand-sprint-easy"]["weight"]


def test_adaptive_sampler_uses_partial_combo_mastery() -> None:
    focused = sampler("combo-rehearsal", "opening-hand-sprint-easy")
    for _ in range(6):
        focused.observe("combo-rehearsal", False, mastery=0.75)
        focused.observe("opening-hand-sprint-easy", False, mastery=0.0)

    stats = focused.stats()

    assert stats["combo-rehearsal"]["masteryRate"] == 0.75
    assert stats["combo-rehearsal"]["weight"] < stats["opening-hand-sprint-easy"]["weight"]


def test_adaptive_sampler_prioritizes_underexposed_failed_scenarios() -> None:
    focused = sampler("known-sideboard-match", "opening-hand-sprint-easy")
    for _ in range(100):
        focused.observe("opening-hand-sprint-easy", False)

    stats = focused.stats()

    assert stats["known-sideboard-match"]["weight"] > stats["opening-hand-sprint-easy"]["weight"]


def test_known_combo_requires_its_enabling_line_before_the_payoff() -> None:
    focused = sampler("combo-aluren-acererak")
    scenario = focused.scenarios[0]
    learner_cards = deck("aluren")
    for index, name in enumerate(scenario.fixed_hand):
        learner_cards[index]["name"] = name
    focused.decks["Sultai Aluren"] = learner_cards
    matchup = focused.sample(random.Random(5))
    environment = RustSessionEnvironment(
        "http://engine.test",
        {matchup.id: matchup},
    )
    environment.current_matchup = matchup
    environment.learner_player_id = matchup.learner_player_id
    view = {
        "state": {
            "players": [
                {
                    "id": matchup.learner_player_id,
                    "battlefield": [{"name": "Acererak the Archlich"}],
                }
            ],
            "events": [],
        },
        "decision": {
            "id": "priority:1",
            "kind": "priority",
            "playerId": matchup.learner_player_id,
            "options": [],
        },
    }

    bypass = environment._to_step(view)
    environment.objective_action_labels = [
        "Cast Aluren from hand",
        "Cast Acererak the Archlich from hand",
    ]
    environment.objective_milestones_completed = (
        environment._objective_sequence_progress()
    )
    completed = environment._to_step(view)

    assert not bypass.done
    assert completed.done
    assert completed.reward == 1.0
    environment.close()


def test_rl_checkpoint_retention_keeps_latest_complete_versions(tmp_path) -> None:
    root = tmp_path / "rl-checkpoints"
    for update in (5, 10, 15, 20):
        checkpoint = root / f"update-{update}"
        checkpoint.mkdir(parents=True)
        (checkpoint / "rl-model.pt").write_bytes(b"checkpoint")
    incomplete = root / "update-25"
    incomplete.mkdir()

    _prune_rl_checkpoints(root, keep=2)

    assert sorted(path.name for path in root.iterdir()) == ["update-15", "update-20", "update-25"]


def test_adaptive_sampler_restores_scenario_weakness_history(tmp_path) -> None:
    focused = sampler("opening-hand-sprint-easy", "combo-rehearsal")
    history = tmp_path / "games.jsonl"
    history.write_text(
        "\n".join(
            json.dumps(
                    {
                        "scenarioId": "combo-rehearsal",
                        "scenarioVersion": 2,
                        "objective": "combo-win",
                        "objectiveDamageProgress": 0.0,
                        "learnerPlayerId": "player-1",
                    "outcome": {},
                    "rewardsByPlayer": {"player-1": -1.0},
                }
            )
            for _ in range(4)
        ),
        encoding="utf-8",
    )

    focused.load_history(history)

    assert focused.stats()["combo-rehearsal"]["games"] == 4
    assert focused.stats()["combo-rehearsal"]["wins"] == 0


def test_adaptive_sampler_ignores_results_from_an_older_scenario_revision(
    tmp_path,
) -> None:
    focused = sampler("combo-show-and-tell-emrakul")
    history = tmp_path / "games.jsonl"
    history.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "scenarioId": "combo-show-and-tell-emrakul",
                        "scenarioVersion": 1,
                        "learnerPlayerId": "player-1",
                        "rewardsByPlayer": {"player-1": 1.0},
                    }
                ),
                json.dumps(
                    {
                        "scenarioId": "combo-show-and-tell-emrakul",
                        "scenarioVersion": 2,
                        "learnerPlayerId": "player-1",
                        "rewardsByPlayer": {"player-1": -1.0},
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )

    focused.load_history(history)

    assert focused.stats()["combo-show-and-tell-emrakul"]["games"] == 1
    assert focused.stats()["combo-show-and-tell-emrakul"]["wins"] == 0
