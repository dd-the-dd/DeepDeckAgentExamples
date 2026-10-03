import json
import math
import random
import threading
from collections import Counter
from copy import deepcopy
from pathlib import Path

import httpx
import pytest
import torch

from oracle_ai.architectures import upgrade_model
from oracle_ai.checkpoints import load_checkpoint, save_checkpoint
from oracle_ai.encoding import HashingObservationEncoder
from oracle_ai.encoding_v2 import (
    NUMERIC_FEATURE_INDEX,
    StructuredObservationEncoder,
    StructuredTokens,
    TokenType,
    normalize_nonnegative,
)
from oracle_ai.encoding_v3 import (
    CyclicStructuredObservationEncoder,
    cyclic_player_coordinates,
)
from oracle_ai.encoding_v4 import OracleStructuredObservationEncoder
from oracle_ai.encoding_v6 import (
    PlanningObservationEncoder,
    TokenTypeV6,
)
from oracle_ai.encoding_v7 import (
    StrategicPlanningObservationEncoder,
    TokenTypeV7,
)
from oracle_ai.encoding_v11 import AlphaStarObservationEncoder, TokenTypeV11
from oracle_ai.model import MagicTransformerActorCritic, ModelConfig
from oracle_ai.model_v2 import MagicTransformerActorCriticV2, ModelConfigV2
from oracle_ai.model_v3 import MagicTransformerActorCriticV3, ModelConfigV3
from oracle_ai.model_v4 import MagicTransformerActorCriticV4, ModelConfigV4
from oracle_ai.model_v5 import MagicTransformerActorCriticV5, ModelConfigV5
from oracle_ai.model_v6 import (
    ContextualSemanticEncoder,
    MagicTransformerActorCriticV6,
    ModelConfigV6,
)
from oracle_ai.model_v7 import MagicTransformerActorCriticV7, ModelConfigV7
from oracle_ai.model_v9 import MagicTransformerActorCriticV9, ModelConfigV9
from oracle_ai.model_v10 import MagicTransformerActorCriticV10, ModelConfigV10
from oracle_ai.model_v11 import MagicTransformerActorCriticV11, ModelConfigV11
from oracle_ai.model_v12 import MagicTransformerActorCriticV12, ModelConfigV12
from oracle_ai.training.behavior import (
    build_decision_trace,
    dominated_action_indices,
    summarize_decision_traces,
)
from oracle_ai.training.core import (
    DecisionStep,
    PackedStructuredTokens,
    PackedTensor,
    PPOConfig,
    PPOLearner,
    SelfPlayJob,
    Transition,
)
from oracle_ai.training.environments import (
    Matchup,
    RustSelfPlayEnvironment,
    RustSessionEnvironment,
    TinySelfPlayEnvironment,
)
from oracle_ai.training.evaluation import (
    PolicyService,
    analytics_pilot_for_champion,
    summarize_evaluation,
)
from oracle_ai.training.future_features import (
    FUTURE_FEATURE_NAMES,
    future_feature_targets,
)
from oracle_ai.training.gameplay_metrics import (
    round_number,
    summarize_gameplay_metrics,
    update_hourly_gameplay_metrics,
)
from oracle_ai.training.league import (
    EvaluationBenchmarkOpponent,
    LeagueState,
    LeagueTrainer,
    RandomTrainingMatchupSampler,
    _anchor_matchup,
    _apply_promotion_result,
    _deck_catalog,
    _deck_catalog_revision,
    _engine_api_headers,
    _evaluation_benchmark_opponents,
    _evaluation_scenarios,
    _evaluation_seed_map,
    _initialize_ground_truth_checkpoint,
    _prune_checkpoints,
    _restore_league_state,
    _resume_counter,
    _rollout_batch_size,
    _set_additional_episode_limit,
    _training_control_state,
    _training_episode_limit,
    _training_matchups,
    _write_learning_curve,
)
from oracle_ai.training.plackett_luce import (
    AnchorChallenge,
    PlackettLuceRating,
    TrainingLeaderboard,
    anchor_challenges,
    anchor_participant_id,
    hypothetical_first_place_deltas,
    rank_gradient_rewards,
)
from oracle_ai.training.seeds import UniqueSeedStream


def test_engine_api_headers_are_forwarded_to_protected_validation(monkeypatch) -> None:
    monkeypatch.delenv("MTG_ENGINE_API_KEY", raising=False)
    assert _engine_api_headers() is None
    monkeypatch.setenv("MTG_ENGINE_API_KEY", " protected-engine-key ")
    assert _engine_api_headers() == {"x-mtg-api-key": "protected-engine-key"}


def test_rollout_batch_stops_on_exact_evaluation_and_episode_boundaries() -> None:
    assert (
        _rollout_batch_size(
            63,
            4,
            evaluation_cadences=(10, 100),
            episode_limit=None,
        )
        == 4
    )
    assert (
        _rollout_batch_size(
            67,
            4,
            evaluation_cadences=(10, 100),
            episode_limit=None,
        )
        == 3
    )
    assert (
        _rollout_batch_size(
            99,
            4,
            evaluation_cadences=(10, 100),
            episode_limit=None,
        )
        == 1
    )
    assert (
        _rollout_batch_size(
            12,
            4,
            evaluation_cadences=(10, 100),
            episode_limit=14,
        )
        == 2
    )


@pytest.mark.parametrize("player_count", [2, 3, 4])
def test_training_round_counts_one_table_rotation(player_count: int) -> None:
    setup = {"players": [{"id": f"player-{index}"} for index in range(player_count)]}

    assert round_number(setup, 1) == 1
    assert round_number(setup, player_count) == 1
    assert round_number(setup, player_count + 1) == 2
    assert round_number(setup, player_count * 5) == 5


def test_meta_session_deck_source_is_discovered_without_a_static_catalog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    class Result:
        returncode = 0
        stderr = ""
        stdout = json.dumps({"New Meta Session": [{"name": "New card"}]})

    monkeypatch.setattr("oracle_ai.training.league.shutil.which", lambda _: "node")

    def run(command: list[str], **_: object) -> Result:
        calls.append(command)
        return Result()

    monkeypatch.setattr("oracle_ai.training.league.subprocess.run", run)

    catalog = _deck_catalog({"deckSource": "database"})

    assert list(catalog) == ["New Meta Session"]
    assert calls == [
        [
            "node",
            str(Path("scripts/build-ai-deck-catalog.mjs")),
            "--stdout",
            "--meta-only",
        ]
    ]


def test_meta_legacy_deck_source_filters_meta_sessions_by_creator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    class Result:
        returncode = 0
        stderr = ""
        stdout = json.dumps({"Oli Legacy": [{"name": "Legacy card"}]})

    monkeypatch.setattr("oracle_ai.training.league.shutil.which", lambda _: "node")

    def run(command: list[str], **_: object) -> Result:
        calls.append(command)
        return Result()

    monkeypatch.setattr("oracle_ai.training.league.subprocess.run", run)

    catalog = _deck_catalog(
        {
            "deckSource": "database",
            "metaLegacyDeckSelection": {
                "enabled": True,
                "creators": ["Meta legacy", "oli"],
            },
        }
    )

    assert list(catalog) == ["Oli Legacy"]
    assert calls[0][-2:] == ["--creator=Meta legacy", "--creator=oli"]


def test_running_trainer_reloads_only_when_the_meta_deck_catalog_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sessions: list[dict[str, object]] = []

    def write_sessions(*, include_new_meta: bool, play_revision: int) -> None:
        sessions[:] = [
            {
                "id": "meta-old",
                "name": "Old Meta",
                "creator": "tester",
                "isMetaDeck": True,
                "updatedAt": "meta-1",
            },
            {
                "id": "free-play",
                "name": "Current game",
                "creator": "tester",
                "isMetaDeck": False,
                "updatedAt": f"play-{play_revision}",
            },
        ]
        if include_new_meta:
            sessions.append(
                {
                    "id": "meta-new",
                    "name": "Landlord's dream (Landfall)",
                    "creator": "tester",
                    "isMetaDeck": True,
                    "updatedAt": "meta-1",
                }
            )

    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {"data": deepcopy(sessions)}

    monkeypatch.setattr(
        "oracle_ai.training.league.httpx.get",
        lambda *_args, **_kwargs: Response(),
    )

    write_sessions(include_new_meta=False, play_revision=1)
    config = {
        "deckSource": "database",
        "trainingScenarioRandomizer": {
            "formats": ["free"],
            "playerCounts": [2],
        },
    }
    old_decks = {"Old Meta": [{"name": "Old card"}]}
    trainer = LeagueTrainer.__new__(LeagueTrainer)
    trainer.config = config
    trainer.training_randomizer_config = deepcopy(config["trainingScenarioRandomizer"])
    trainer.training_matchup_sampler = RandomTrainingMatchupSampler(
        trainer.training_randomizer_config,
        old_decks,
    )
    template = trainer.training_matchup_sampler.template()
    trainer.training_matchups = {template.id: template}
    live_environment_matchups = trainer.training_matchups
    trainer.training_deck_names = tuple(old_decks)
    trainer.deck_catalog_revision = _deck_catalog_revision(config)

    write_sessions(include_new_meta=False, play_revision=2)
    assert _deck_catalog_revision(config) == trainer.deck_catalog_revision

    new_decks = {
        **old_decks,
        "Landlord's dream (Landfall)": [{"name": "Landfall card"}],
    }
    monkeypatch.setattr(
        "oracle_ai.training.league._deck_catalog",
        lambda _config: new_decks,
    )
    write_sessions(include_new_meta=True, play_revision=3)
    trainer._refresh_training_decks_if_changed()

    assert trainer.training_deck_names == tuple(new_decks)
    assert trainer.training_matchup_sampler is not None
    assert trainer.training_matchup_sampler.deck_names == list(new_decks)
    assert trainer.training_matchups is live_environment_matchups
    assert trainer.deck_catalog_revision == _deck_catalog_revision(config)


def test_learner_resource_plan_resizes_training_workers_between_batches(
    tmp_path: Path,
) -> None:
    resource_path = tmp_path / "learner-resources.json"
    resource_path.write_text(
        json.dumps({"trainingMatches": 3, "gpuMemoryMb": 4096}),
        encoding="utf-8",
    )
    trainer = LeagueTrainer.__new__(LeagueTrainer)
    trainer.resource_plan_path = resource_path
    trainer.parallel_game_workers = 1
    trainer.rollout_batch_games = 1

    class Environment:
        progress_callback = None

    trainer.environments = [Environment()]
    trainer.environment = trainer.environments[0]
    trainer.device = torch.device("cpu")
    trainer.gpu_memory_limit_mb = 0
    trainer._new_training_environment = Environment
    trainer._observe_training_view = lambda worker_index, view: None

    trainer._refresh_learner_resources()

    assert trainer.parallel_game_workers == 3
    assert trainer.rollout_batch_games == 3
    assert len(trainer.environments) == 3
    assert all(environment.progress_callback is not None for environment in trainer.environments)
    assert [environment.progress_callback.args for environment in trainer.environments] == [
        (0,),
        (1,),
        (2,),
    ]
    assert trainer.gpu_memory_limit_mb == 4096


def test_gameplay_metrics_summarize_early_game_and_suicidal_attacks() -> None:
    setup = {
        "startingPlayer": 1,
        "players": [
            {"id": "player-1"},
            {"id": "player-2"},
            {"id": "player-3"},
        ],
    }
    state = {
        "turnNumber": 18,
        "events": [
            {
                "kind": "openingHandFinalized",
                "playerId": "player-1",
                "detail": {"handSize": 6, "landCount": 3},
            },
            {
                "kind": "spellCast",
                "playerId": "player-1",
                "turnNumber": 15,
                "detail": {"manaValue": 4},
            },
            {
                "kind": "spellCast",
                "playerId": "player-1",
                "turnNumber": 18,
                "detail": {"manaValue": 9},
            },
            {"kind": "permanentEnteredBattlefield", "playerId": "player-1"},
            {"kind": "attackerDeclared", "playerId": "player-1"},
            {
                "kind": "attackResolved",
                "playerId": "player-1",
                "detail": {"suicidal": True},
            },
        ],
    }

    summary = summarize_gameplay_metrics(state, setup)

    assert summary["playerSamples"] == 3
    assert summary["samples"]["startingHandCards"] == 1
    assert summary["sums"] == {
        "startingHandCards": 6.0,
        "startingHandLands": 3.0,
        "manaValueFirstFiveTurns": 4.0,
        "permanentsExisted": 1.0,
        "attacks": 1.0,
        "suicidalAttacks": 1.0,
        "gameRounds": 6.0,
    }
    assert summary["samples"]["gameRounds"] == 1


def test_gameplay_metrics_are_averaged_per_active_training_hour() -> None:
    episode = {
        "playerSamples": 2,
        "sums": {
            "startingHandCards": 13,
            "startingHandLands": 5,
            "manaValueFirstFiveTurns": 18,
            "permanentsExisted": 12,
            "attacks": 8,
            "suicidalAttacks": 2,
            "gameRounds": 8,
        },
        "samples": {
            "startingHandCards": 2,
            "startingHandLands": 2,
            "manaValueFirstFiveTurns": 2,
            "permanentsExisted": 2,
            "attacks": 2,
            "suicidalAttacks": 2,
            "gameRounds": 1,
        },
    }

    payload = update_hourly_gameplay_metrics(None, episode, 3)
    payload = update_hourly_gameplay_metrics(payload, episode, 3)

    hour = payload["hours"][0]
    assert hour["trainingHour"] == 3
    assert hour["games"] == 2
    assert hour["playerSamples"] == 4
    assert hour["averages"]["startingHandCards"] == 6.5
    assert hour["averages"]["manaValueFirstFiveTurns"] == 9.0
    assert hour["averages"]["attacks"] == 4.0
    assert hour["averages"]["gameRounds"] == 8.0
    assert hour["suicidalAttackRate"] == 0.25


def test_game_round_average_resets_legacy_turn_metrics() -> None:
    payload = {
        "schemaVersion": "oracle-ai/training-gameplay-hourly/v1",
        "hours": [{"trainingHour": 3, "games": 4, "sums": {}, "samples": {}}],
    }
    episode = {
        "playerSamples": 0,
        "sums": {"gameRounds": 6},
        "samples": {"gameRounds": 1},
    }

    updated = update_hourly_gameplay_metrics(payload, episode, 3)
    hour = updated["hours"][0]

    assert hour["games"] == 1
    assert hour["samples"]["gameRounds"] == 1
    assert hour["averages"]["gameRounds"] == 6.0


def test_transformer_scores_variable_legal_actions() -> None:
    model = MagicTransformerActorCritic(
        ModelConfig(feature_dim=32, d_model=32, layers=1, heads=4, feedforward_dim=64)
    )
    encoder = HashingObservationEncoder(feature_dim=32)
    encoded = encoder.encode(
        {"turn": 1, "life": 20},
        [{"id": "a"}, {"id": "b"}, {"id": "c"}],
    )
    logits, value = model(encoded.state_tokens, encoded.action_tokens)
    assert logits.shape == (1, 3)
    assert value.shape == (1,)


def test_v2_nonnegative_normalization_uses_requested_negative_domain() -> None:
    assert normalize_nonnegative(-4) == -1.0
    assert normalize_nonnegative(0) == -1.0
    assert normalize_nonnegative(20) == -0.5
    assert normalize_nonnegative(60) == -0.25
    assert -0.001 < normalize_nonnegative(1_000_000) < 0.0


def test_scenario_matrices_cover_eight_decks_and_player_counts() -> None:
    deck_names = [
        "Aang",
        "Katara",
        "Toph",
        "Omnath",
        "Alexios",
        "Mobilize",
        "Sek'tuar",
        "4c Control",
    ]
    decks = {name: [{"name": name}] for name in deck_names}
    config = {
        "trainingScenarioMatrix": {
            "decks": deck_names,
            "playerCounts": [2, 3, 4],
            "freeMulligans": [0, 1, 2],
        },
        "evaluation": {
            "scenarioMatrix": {
                "decks": deck_names,
                "playerCounts": [2, 3, 4],
                "freeMulligans": [0, 1, 2],
            }
        },
    }

    training = _training_matchups(config, decks)
    evaluations = _evaluation_scenarios(config, decks)

    assert len(training) == (28 + 56 + 70) * 3
    assert {len(matchup.setup["players"]) for matchup in training.values()} == {2, 3, 4}
    assert {matchup.free_mulligans for matchup in training.values()} == {0, 1, 2}
    assert len(evaluations) == len(deck_names) * 3
    assert {scenario.candidate_deck for scenario in evaluations.values()} == set(deck_names)
    assert {len(scenario.matchup.setup["players"]) for scenario in evaluations.values()} == {
        2,
        3,
        4,
    }


def test_training_randomizer_samples_formats_rules_and_matchups_per_episode() -> None:
    def deck(name: str, *, commander: bool, size: int) -> list[dict]:
        return [
            {
                "name": f"{name} card {index}",
                "isCommander": commander and index == 0,
                "typeLine": "Legendary Creature" if index == 0 else "Artifact",
            }
            for index in range(size)
        ]

    decks = {
        "Aang": deck("Aang", commander=True, size=100),
        "Omnath": deck("Omnath", commander=True, size=100),
        "Mobilize": deck("Mobilize", commander=True, size=113),
        "Control": deck("Control", commander=False, size=75),
    }
    sampler = RandomTrainingMatchupSampler(
        {
            "formats": ["free", "commander"],
            "decks": list(decks),
            "playerCounts": [2, 3, 4],
            "maxTurns": 80,
            "free": {
                "startingLifeRange": [20, 40],
                "freeMulliganRange": [0, 2],
            },
        },
        decks,
    )
    randomizer = random.Random(37)
    samples = [sampler.sample(randomizer) for _ in range(200)]

    assert {matchup.game_mode for matchup in samples} == {"free", "commander"}
    assert {len(matchup.setup["players"]) for matchup in samples} == {2, 3, 4}
    assert any(len(set(matchup.deck_names)) < len(matchup.deck_names) for matchup in samples)
    free_games = [matchup for matchup in samples if matchup.game_mode == "free"]
    commander_games = [matchup for matchup in samples if matchup.game_mode == "commander"]
    assert {matchup.free_mulligans for matchup in free_games} == {0, 1, 2}
    assert all(20 <= matchup.setup["players"][0]["startingLife"] <= 40 for matchup in free_games)
    assert any("Control" in matchup.deck_names for matchup in free_games)
    assert all(matchup.free_mulligans == 1 for matchup in commander_games)
    assert all(
        {player["startingLife"] for player in matchup.setup["players"]} == {40}
        for matchup in commander_games
    )
    assert all(set(matchup.deck_names) <= {"Aang", "Omnath"} for matchup in commander_games)


def test_training_randomizer_builds_two_player_legacy_games_from_legal_decks() -> None:
    legacy = [
        {
            "id": f"legacy-{index}",
            "name": f"Legacy card {index}",
            "typeLine": "Artifact",
        }
        for index in range(60)
    ]
    commander = [
        {
            "id": f"commander-{index}",
            "name": f"Commander card {index}",
            "isCommander": index == 0,
            "typeLine": "Legendary Creature" if index == 0 else "Artifact",
        }
        for index in range(100)
    ]
    sampler = RandomTrainingMatchupSampler(
        {
            "formats": ["legacy"],
            "decks": ["Legacy", "Commander"],
            "playerCounts": [2, 3, 4],
        },
        {"Legacy": legacy, "Commander": commander},
    )

    samples = [sampler.sample(random.Random(seed)) for seed in range(10)]

    assert {matchup.game_mode for matchup in samples} == {"legacy"}
    assert {len(matchup.setup["players"]) for matchup in samples} == {2}
    assert {matchup.deck_names for matchup in samples} == {("Legacy", "Legacy")}
    assert all(matchup.setup["openingHandSize"] == 7 for matchup in samples)
    assert all(matchup.free_mulligans == 0 for matchup in samples)
    assert all(matchup.max_mulligans is None for matchup in samples)
    assert all(
        {player["startingLife"] for player in matchup.setup["players"]} == {20}
        for matchup in samples
    )


def test_legacy_candidate_ignores_tokens_that_only_have_a_token_type_line() -> None:
    legacy = [
        {
            "id": f"legacy-{index}",
            "name": f"Legacy card {index}",
            "typeLine": "Artifact",
        }
        for index in range(60)
    ]
    legacy.extend(
        {
            "id": f"token-{index}",
            "name": "Zombie",
            "typeLine": "Token Creature — Zombie",
            "isSideboard": True,
        }
        for index in range(16)
    )

    sampler = RandomTrainingMatchupSampler(
        {"formats": ["legacy"], "decks": ["Legacy"], "playerCounts": [2]},
        {"Legacy": legacy},
    )

    matchup = sampler.sample(random.Random(7))
    assert matchup.deck_names == ("Legacy", "Legacy")
    assert [seat["plackettLuceDelta"] for seat in sampler.last_selection["seats"]] == [
        {
            "win": pytest.approx(0.5),
            "draw": pytest.approx(0.0),
            "loss": pytest.approx(-0.5),
        },
        {
            "win": pytest.approx(0.5),
            "draw": pytest.approx(0.0),
            "loss": pytest.approx(-0.5),
        },
    ]


def test_training_randomizer_keeps_rollout_batch_structurally_homogeneous() -> None:
    decks = {
        name: [
            {
                "id": f"{name}-{index}",
                "name": f"{name} card {index}",
                "isCommander": index == 0,
                "typeLine": "Legendary Creature" if index == 0 else "Artifact",
            }
            for index in range(100)
        ]
        for name in ["Aang", "Omnath", "Yenna"]
    }
    sampler = RandomTrainingMatchupSampler(
        {
            "formats": ["free", "commander", "training", "training2"],
            "decks": list(decks),
            "playerCounts": [2, 3, 4],
            "free": {
                "startingLifeRange": [20, 40],
                "freeMulliganRange": [0, 2],
            },
        },
        decks,
    )
    randomizer = random.Random(101)

    batches = []
    for _ in range(24):
        profile = sampler.sample_batch_profile(randomizer)
        batch = [sampler.sample(randomizer, profile=profile) for _ in range(4)]
        batches.append(batch)

    assert all(len({game.game_mode for game in batch}) == 1 for batch in batches)
    assert all(len({len(game.setup["players"]) for game in batch}) == 1 for batch in batches)
    assert all(
        len({game.setup["players"][0]["startingLife"] for game in batch}) == 1 for batch in batches
    )
    assert all(len({game.free_mulligans for game in batch}) == 1 for batch in batches)
    assert {batch[0].game_mode for batch in batches} == {
        "free",
        "commander",
        "training",
        "training2",
    }
    assert {len(batch[0].setup["players"]) for batch in batches} == {2, 3, 4}


def test_training_randomizer_builds_only_simplified_training_games() -> None:
    decks = {
        name: [
            {
                "id": f"{name}-{index}",
                "name": f"{name} card {index}",
                "isCommander": index == 0,
                "typeLine": "Artifact",
            }
            for index in range(100)
        ]
        for name in ["Aang", "Omnath", "Yenna"]
    }
    sampler = RandomTrainingMatchupSampler(
        {
            "formats": ["training"],
            "decks": list(decks),
            "playerCounts": [2, 3, 4],
            "training": {"freeMulligans": 3, "maxMulligans": 3},
        },
        decks,
    )
    samples = [sampler.sample(random.Random(seed)) for seed in range(30)]

    assert {matchup.game_mode for matchup in samples} == {"training"}
    assert {len(matchup.setup["players"]) for matchup in samples} == {2, 3, 4}
    assert all(matchup.setup["openingHandSize"] == 5 for matchup in samples)
    assert all(matchup.free_mulligans == 3 for matchup in samples)
    assert all(matchup.max_mulligans == 3 for matchup in samples)
    assert all(
        {player["startingLife"] for player in matchup.setup["players"]} == {5}
        for matchup in samples
    )


def test_training_randomizer_builds_training2_commander_games() -> None:
    decks = {
        name: [
            {
                "id": f"{name}-{index}",
                "name": f"{name} card {index}",
                "isCommander": index == 0,
                "typeLine": "Legendary Creature" if index == 0 else "Artifact",
            }
            for index in range(100)
        ]
        for name in ["Aang", "Omnath", "Yenna"]
    }
    sampler = RandomTrainingMatchupSampler(
        {
            "formats": ["training2"],
            "decks": list(decks),
            "playerCounts": [2, 3, 4],
        },
        decks,
    )
    samples = [sampler.sample(random.Random(seed)) for seed in range(30)]

    assert {matchup.game_mode for matchup in samples} == {"training2"}
    assert {len(matchup.setup["players"]) for matchup in samples} == {2, 3, 4}
    assert all(matchup.setup["openingHandSize"] == 6 for matchup in samples)
    assert all(matchup.free_mulligans == 1 for matchup in samples)
    assert all(matchup.max_mulligans == 3 for matchup in samples)
    assert all(
        {player["startingLife"] for player in matchup.setup["players"]} == {10}
        for matchup in samples
    )
    assert all(
        sum(bool(card.get("isCommander")) for card in player["cards"]) == 1
        for matchup in samples
        for player in matchup.setup["players"]
    )


def test_training_randomizer_round_robin_guarantees_every_configured_format() -> None:
    decks = {
        name: [
            {
                "id": f"{name}-{index}",
                "name": f"{name} card {index}",
                "isCommander": index == 0,
                "typeLine": "Legendary Creature" if index == 0 else "Artifact",
            }
            for index in range(100)
        ]
        for name in ["Aang", "Omnath", "Yenna"]
    }
    sampler = RandomTrainingMatchupSampler(
        {
            "formats": ["training", "training2", "free", "commander"],
            "formatSampling": "roundRobin",
            "decks": list(decks),
            "playerCounts": [2, 3, 4],
        },
        decks,
    )

    formats = [sampler.sample(random.Random(37)).game_mode for _ in range(8)]

    assert formats == [
        "training",
        "training2",
        "free",
        "commander",
        "training",
        "training2",
        "free",
        "commander",
    ]


def test_training_matchmaking_favors_close_plackett_luce_opponents() -> None:
    decks = {
        name: [{"name": f"{name} card", "typeLine": "Artifact"}]
        for name in ["Anchor", "Near", "Far"]
    }
    sampler = RandomTrainingMatchupSampler(
        {
            "formats": ["free"],
            "decks": list(decks),
            "playerCounts": [2],
            "matchmaking": {
                "enabled": True,
                "randomFloor": 0.20,
                "ratingScale": 1,
                "underplayedStrength": 0.35,
                "matchPrior": 10,
            },
        },
        decks,
    )
    stats = {
        "Anchor": {"mu": 25.0, "sigma": 8.0, "ordinal": 1.0, "games": 100},
        "Near": {"mu": 25.1, "sigma": 8.0, "ordinal": 1.1, "games": 100},
        "Far": {"mu": 35.0, "sigma": 8.0, "ordinal": 11.0, "games": 100},
    }
    randomizer = random.Random(71)
    opponents = Counter(
        matchup.deck_names[1]
        for matchup in (
            sampler.sample(
                randomizer,
                lambda _participant, _decks: stats,
                participant_ids=["v11", "v10"],
            )
            for _ in range(6000)
        )
        if matchup.deck_names[0] == "Anchor"
    )

    assert opponents["Near"] > opponents["Far"] * 2
    assert opponents["Far"] > 0
    assert sampler.last_selection["ratingSystem"] == "plackett-luce"
    assert sampler.last_selection["ratingSource"] == ("training-model-deck-leaderboard")
    assert sampler.last_selection["seats"][1]["participantId"] == "v10"
    assert "elo" not in sampler.last_selection["seats"][1]
    for seat in sampler.last_selection["seats"]:
        assert seat["plackettLuceDelta"]["win"] > 0.0
        assert seat["plackettLuceDelta"]["loss"] < 0.0


def test_training_matchmaking_slightly_favors_underplayed_decks() -> None:
    decks = {
        name: [{"name": f"{name} card", "typeLine": "Artifact"}]
        for name in ["Well sampled", "Underplayed"]
    }
    sampler = RandomTrainingMatchupSampler(
        {
            "formats": ["free"],
            "decks": list(decks),
            "playerCounts": [2],
            "matchmaking": {
                "enabled": True,
                "randomFloor": 0.20,
                "ratingScale": 10,
                "underplayedStrength": 0.35,
                "matchPrior": 10,
            },
        },
        decks,
    )
    stats = {
        "Well sampled": {"ordinal": 1.0, "games": 400},
        "Underplayed": {"ordinal": 1.0, "games": 5},
    }
    randomizer = random.Random(73)
    appearances = Counter(
        deck_name
        for _ in range(3000)
        for deck_name in sampler.sample(
            randomizer,
            lambda _participant, _decks: stats,
            participant_ids=["v11", "v11"],
        ).deck_names
    )

    assert appearances["Underplayed"] > appearances["Well sampled"]
    assert appearances["Well sampled"] > 0


def test_anchor_matchmaking_uses_the_model_deck_plackett_luce_strength(
    tmp_path: Path,
) -> None:
    leaderboard = TrainingLeaderboard(
        tmp_path / "training-leaderboard.json",
        {
            "v11": "V11",
            "anchor-m01-n020-p2": "Anchor M1/N20/P2",
            "anchor-m10-n020-p2": "Anchor M10/N20/P2",
        },
    )

    def set_rating(
        participant_id: str,
        deck_name: str,
        rating: PlackettLuceRating,
    ) -> None:
        entry_id = leaderboard.deck_participants(
            {"player": participant_id},
            {"player": deck_name},
        )["player"]
        leaderboard.deck_ratings[entry_id] = rating

    set_rating("v11", "Omnath", PlackettLuceRating(mu=25.0, sigma=8.0, games=100))
    set_rating(
        "anchor-m01-n020-p2",
        "Anchor",
        PlackettLuceRating(mu=25.1, sigma=8.0, games=100),
    )
    set_rating(
        "anchor-m10-n020-p2",
        "Anchor",
        PlackettLuceRating(mu=35.0, sigma=8.0, games=100),
    )
    sampler = RandomTrainingMatchupSampler(
        {
            "formats": ["free"],
            "decks": ["Omnath"],
            "playerCounts": [2],
            "matchmaking": {
                "enabled": True,
                "randomFloor": 0.20,
                "ratingScale": 1,
                "underplayedStrength": 0.35,
                "matchPrior": 10,
            },
        },
        {"Omnath": [{"name": "Card", "typeLine": "Artifact"}]},
    )
    trainer = LeagueTrainer.__new__(LeagueTrainer)
    trainer.training_leaderboard = leaderboard
    trainer.training_matchup_sampler = sampler
    trainer.anchor_deadline_rounds = (1, 10)
    trainer.anchor_opening_hand_pool_sizes = (20,)
    trainer.anchor_player_counts = (2,)
    trainer.matchup_randomizer = random.Random(79)

    selections = Counter(trainer._sample_anchor_challenge("Omnath", 2)[:2] for _ in range(3000))

    assert selections[(1, 20)] > selections[(10, 20)] * 2
    assert selections[(10, 20)] > 0
    _, _, detail = trainer._sample_anchor_challenge("Omnath", 2)
    assert detail["target"]["participantId"] == "v11"
    assert detail["target"]["deck"] == "Omnath"
    assert detail["playerCount"] == 2


def test_evaluation_matrix_covers_free_and_commander_formats() -> None:
    def deck(name: str, *, commander: bool) -> list[dict]:
        return [
            {
                "name": f"{name} card {index}",
                "isCommander": commander and index == 0,
                "typeLine": "Legendary Creature" if index == 0 else "Artifact",
            }
            for index in range(100 if commander else 60)
        ]

    decks = {
        "Aang": deck("Aang", commander=True),
        "Omnath": deck("Omnath", commander=True),
        "Control": deck("Control", commander=False),
    }
    scenarios = _evaluation_scenarios(
        {
            "evaluation": {
                "scenarioMatrix": {
                    "formats": ["free", "commander"],
                    "decks": list(decks),
                    "playerCounts": [2, 4],
                    "freeMulligans": [0, 2],
                }
            }
        },
        decks,
    )

    free = [scenario for scenario in scenarios.values() if scenario.matchup.game_mode == "free"]
    commander = [
        scenario for scenario in scenarios.values() if scenario.matchup.game_mode == "commander"
    ]
    assert len(free) == 6
    assert len(commander) == 4
    assert {scenario.candidate_deck for scenario in commander} == {"Aang", "Omnath"}
    assert all(scenario.matchup.free_mulligans == 1 for scenario in commander)
    assert all(
        {player["startingLife"] for player in scenario.matchup.setup["players"]} == {40}
        for scenario in commander
    )
    assert {len(scenario.matchup.setup["players"]) for scenario in commander} == {2, 4}


def test_evaluation_matrix_builds_simplified_training_scenarios() -> None:
    decks = {
        name: [{"name": f"{name} card {index}"} for index in range(60)]
        for name in ["Aang", "Omnath"]
    }
    scenarios = _evaluation_scenarios(
        {
            "evaluation": {
                "scenarioMatrix": {
                    "formats": ["training"],
                    "decks": list(decks),
                    "playerCounts": [2],
                    "trainingFreeMulligans": 3,
                    "trainingMaxMulligans": 3,
                }
            }
        },
        decks,
    )

    assert len(scenarios) == 2
    assert all(scenario.matchup.game_mode == "training" for scenario in scenarios.values())
    assert all(scenario.matchup.setup["openingHandSize"] == 5 for scenario in scenarios.values())
    assert all(scenario.matchup.max_mulligans == 3 for scenario in scenarios.values())
    assert all(
        {player["startingLife"] for player in scenario.matchup.setup["players"]} == {5}
        for scenario in scenarios.values()
    )


def test_evaluation_matrix_builds_training2_commander_scenarios() -> None:
    decks = {
        name: [
            {
                "name": f"{name} card {index}",
                "isCommander": index == 0,
                "typeLine": "Legendary Creature" if index == 0 else "Artifact",
            }
            for index in range(100)
        ]
        for name in ["Aang", "Omnath"]
    }
    scenarios = _evaluation_scenarios(
        {
            "evaluation": {
                "scenarioMatrix": {
                    "formats": ["training2"],
                    "decks": list(decks),
                    "playerCounts": [2],
                }
            }
        },
        decks,
    )

    assert len(scenarios) == 2
    assert all(scenario.matchup.game_mode == "training2" for scenario in scenarios.values())
    assert all(scenario.matchup.setup["openingHandSize"] == 6 for scenario in scenarios.values())
    assert all(scenario.matchup.free_mulligans == 1 for scenario in scenarios.values())
    assert all(scenario.matchup.max_mulligans == 3 for scenario in scenarios.values())
    assert all(
        {player["startingLife"] for player in scenario.matchup.setup["players"]} == {10}
        for scenario in scenarios.values()
    )


def _structured_test_card(
    instance_id: str,
    name: str,
    *,
    rules: list[dict] | None = None,
    mana_cost: str = "{2}{G}",
    power: str | None = "3",
    toughness: str | None = "4",
) -> dict:
    return {
        "instanceId": instance_id,
        "owner": "owner",
        "controller": "owner",
        "tapped": False,
        "summoningSick": False,
        "damageMarked": 0,
        "powerModifier": 0,
        "toughnessModifier": 0,
        "counters": {},
        "definition": {
            "id": f"definition:{name}",
            "name": name,
            "typeLine": "Creature — Test",
            "manaCost": mana_cost,
            "power": power,
            "toughness": toughness,
            "rules": rules or [{"kind": "keywordAbility", "ability": "flying"}],
        },
    }


def _structured_test_state() -> dict:
    opponent_secret = _structured_test_card("secret:1", "Opponent Secret")
    active_permanent = _structured_test_card("permanent:active", "Active Permanent")
    previous_permanent = _structured_test_card("permanent:previous", "Previous Permanent")
    acting_hand = _structured_test_card(
        "hand:acting",
        "Acting Hand Card",
        rules=[
            {"kind": "spellAbility", "effects": "draw two cards"},
            {"kind": "keywordAbility", "ability": "kicker"},
        ],
    )
    return {
        "schemaVersion": "mtg-game/v1",
        "status": "inProgress",
        "turnNumber": 5,
        "activePlayer": 1,
        "priorityPlayer": 2,
        "step": "draw",
        "players": [
            {
                "id": "player-1",
                "life": 20,
                "library": [opponent_secret],
                "hand": [opponent_secret],
                "battlefield": [previous_permanent],
                "graveyard": [],
                "exile": [],
                "sideboard": [],
                "manaPool": [],
                "landPlaysRemaining": 1,
                "maxHandSize": 7,
            },
            {
                "id": "player-2",
                "life": 18,
                "library": [],
                "hand": [opponent_secret],
                "battlefield": [active_permanent],
                "graveyard": [],
                "exile": [],
                "sideboard": [],
                "manaPool": [{"symbol": "G"}],
                "landPlaysRemaining": 1,
                "maxHandSize": 7,
            },
            {
                "id": "player-3",
                "life": 12,
                "library": [],
                "hand": [acting_hand],
                "battlefield": [],
                "graveyard": [],
                "exile": [],
                "sideboard": [],
                "manaPool": [{"symbol": "U"}, {"symbol": "C"}],
                "landPlaysRemaining": 0,
                "maxHandSize": 7,
            },
        ],
        "stack": [],
        "combat": {"attackers": [], "blockers": []},
        "events": [],
        "permissions": [],
        "ruleModifiers": [],
        "_decisionContext": {"playerId": "player-3", "kind": "priority"},
    }


def test_v2_encoder_uses_structured_visible_tokens_and_active_player_positions() -> None:
    encoder = StructuredObservationEncoder(
        word_vocab_size=256,
        max_words=16,
        max_relative_players=4,
    )
    state = _structured_test_state()
    encoded = encoder.encode(
        state,
        [{"id": "pass:1", "kind": "passPriority", "playerId": "player-3"}],
    )
    token_types = encoded.state_tokens.token_types

    player_mask = token_types.eq(int(TokenType.PLAYER_STATS))
    assert encoded.state_tokens.relative_players[player_mask].tolist() == [2, 0, 1]
    assert token_types.eq(int(TokenType.CARD_STATS)).sum().item() == 1
    assert token_types.eq(int(TokenType.PERMANENT_STATS)).sum().item() == 2
    assert token_types.eq(int(TokenType.ORACLE_TEXT)).sum().item() == 4
    assert token_types.eq(int(TokenType.GAME_PHASE)).sum().item() == 1

    phase_index = token_types.eq(int(TokenType.GAME_PHASE)).nonzero()[0].item()
    phase_numeric = encoded.state_tokens.numeric[phase_index]
    expected_angle = 2.0 * math.pi * 2 / 10
    assert phase_numeric[NUMERIC_FEATURE_INDEX["phase_sin"]].item() == pytest.approx(
        math.sin(expected_angle)
    )
    assert phase_numeric[NUMERIC_FEATURE_INDEX["phase_cos"]].item() == pytest.approx(
        math.cos(expected_angle)
    )

    hidden_variant = deepcopy(state)
    hidden_variant["players"][0]["hand"][0]["definition"]["name"] = "Different Secret"
    hidden_variant["players"][0]["library"][0]["definition"]["name"] = "Different Library"
    encoded_variant = encoder.encode(
        hidden_variant,
        [{"id": "pass:2", "kind": "passPriority", "playerId": "player-3"}],
    )
    assert torch.equal(encoded.state_tokens.numeric, encoded_variant.state_tokens.numeric)
    assert torch.equal(encoded.state_tokens.word_ids, encoded_variant.state_tokens.word_ids)

    known_state = deepcopy(state)
    known_card = _structured_test_card("known:opponent", "Known Counterspell")
    known_card["flags"] = {"knownToViewer": True}
    known_state["players"][0]["hand"].append(known_card)
    encoded_known = encoder.encode(
        known_state,
        [{"id": "pass:3", "kind": "passPriority", "playerId": "player-3"}],
    )
    renamed_known_state = deepcopy(known_state)
    renamed_known_state["players"][0]["hand"][-1]["definition"]["name"] = "Known Lightning Bolt"
    encoded_renamed_known = encoder.encode(
        renamed_known_state,
        [{"id": "pass:4", "kind": "passPriority", "playerId": "player-3"}],
    )
    assert not torch.equal(
        encoded_known.state_tokens.word_ids,
        encoded_renamed_known.state_tokens.word_ids,
    )


def test_v2_encoder_resolves_casting_decision_card_ids_to_card_semantics() -> None:
    encoder = StructuredObservationEncoder(
        word_vocab_size=4096,
        max_words=32,
        max_relative_players=4,
    )
    state = _structured_test_state()
    state["players"][2]["hand"].extend(
        [
            _structured_test_card("hand:grief", "Grief", mana_cost="{2}{B}{B}"),
            _structured_test_card(
                "hand:thoughtseize",
                "Thoughtseize",
                mana_cost="{B}",
                power=None,
                toughness=None,
            ),
        ]
    )
    actions = [
        {
            "id": "cast:unmask:grief",
            "kind": "castSpell",
            "label": "Cast Unmask using the alternative cost",
            "playerId": "player-3",
            "cardInstanceId": "hand:acting",
            "decisions": {
                "useAlternativeCost": True,
                "alternativeExileCard": "hand:grief",
            },
        },
        {
            "id": "cast:unmask:thoughtseize",
            "kind": "castSpell",
            "label": "Cast Unmask using the alternative cost",
            "playerId": "player-3",
            "cardInstanceId": "hand:acting",
            "decisions": {
                "useAlternativeCost": True,
                "alternativeExileCard": "hand:thoughtseize",
            },
        },
    ]

    grief = state["players"][2]["hand"][1]
    thoughtseize = state["players"][2]["hand"][2]
    entities = {
        "hand:grief": (grief, 0),
        "hand:thoughtseize": (thoughtseize, 0),
    }
    grief_words = list(encoder._action_decision_entity_words(actions[0], entities))
    thoughtseize_words = list(encoder._action_decision_entity_words(actions[1], entities))
    assert len(grief_words) == 1
    assert grief_words[0].startswith("decision alternativeExileCard Grief ")
    assert grief_words[0].endswith("{2}{B}{B}")
    assert len(thoughtseize_words) == 1
    assert thoughtseize_words[0].startswith("decision alternativeExileCard Thoughtseize ")
    assert thoughtseize_words[0].endswith("{B}")

    encoded = encoder.encode(state, actions)
    assert not torch.equal(
        encoded.action_tokens.word_ids[0],
        encoded.action_tokens.word_ids[1],
    )

    target_actions = [
        {
            "id": "target:grief",
            "kind": "chooseTarget",
            "label": "Choose target card",
            "playerId": "player-3",
            "targets": {
                "targetCard": {"card": {"instanceId": "hand:grief"}},
            },
        },
        {
            "id": "target:thoughtseize",
            "kind": "chooseTarget",
            "label": "Choose target card",
            "playerId": "player-3",
            "targets": {
                "targetCard": {"card": {"instanceId": "hand:thoughtseize"}},
            },
        },
    ]
    target_encoded = encoder.encode(state, target_actions)
    assert "target targetCard Grief" in target_encoded.action_token_labels[0]
    assert "target targetCard Thoughtseize" in target_encoded.action_token_labels[1]
    assert not torch.equal(
        target_encoded.action_tokens.word_ids[0],
        target_encoded.action_tokens.word_ids[1],
    )


def test_v2_transformer_scores_actions_and_packs_structured_tokens() -> None:
    config = ModelConfigV2(
        d_model=32,
        layers=1,
        heads=4,
        feedforward_dim=64,
        word_vocab_size=256,
        max_words=16,
        max_relative_players=4,
    )
    model = MagicTransformerActorCriticV2(config)
    encoder = StructuredObservationEncoder(
        word_vocab_size=config.word_vocab_size,
        max_words=config.max_words,
        max_relative_players=config.max_relative_players,
    )
    encoded = encoder.encode(
        _structured_test_state(),
        [
            {"id": "pass", "kind": "passPriority", "playerId": "player-3"},
            {"id": "cast", "kind": "castSpell", "playerId": "player-3"},
        ],
    )

    logits, value = model(encoded.state_tokens, encoded.action_tokens)
    assert logits.shape == (1, 2)
    assert value.shape == (1,)

    packed = PackedStructuredTokens.pack(encoded.state_tokens)
    restored = packed.unpack(torch.device("cpu"))
    assert torch.allclose(restored.numeric, encoded.state_tokens.numeric, atol=1e-3)
    assert torch.equal(restored.word_ids, encoded.state_tokens.word_ids)
    assert torch.equal(restored.relative_players, encoded.state_tokens.relative_players)
    assert torch.equal(restored.token_types, encoded.state_tokens.token_types)


def test_v3_encoder_anchors_cyclic_positions_on_deciding_player() -> None:
    encoder = CyclicStructuredObservationEncoder(
        word_vocab_size=256,
        max_words=16,
        max_relative_players=4,
    )
    state = _structured_test_state()
    encoded = encoder.encode(
        state,
        [{"id": "pass", "kind": "passPriority", "playerId": "player-3"}],
    )
    player_mask = encoded.state_tokens.token_types.eq(int(TokenType.PLAYER_STATS))

    assert encoded.state_tokens.relative_players[player_mask].tolist() == [280, 560, 0]
    assert cyclic_player_coordinates(0) == pytest.approx((0.0, 1.0))
    assert cyclic_player_coordinates(420) == pytest.approx((0.0, -1.0))

    four_players = [{"id": f"player-{index}"} for index in range(1, 5)]
    assert encoder._decision_player_positions(
        {},
        four_players,
        "player-1",
    ) == {
        "player-1": 0,
        "player-2": 210,
        "player-3": 420,
        "player-4": 630,
    }


def test_v3_two_player_checkpoint_can_encode_four_engine_seats() -> None:
    encoder = CyclicStructuredObservationEncoder(
        word_vocab_size=256,
        max_words=16,
        max_relative_players=2,
    )
    players = [{"id": f"player-{index}"} for index in range(1, 5)]

    assert encoder._decision_player_positions({}, players, "player-2") == {
        "player-1": 630,
        "player-2": 0,
        "player-3": 210,
        "player-4": 420,
    }


def test_v3_transformer_and_checkpoint_support_cyclic_positions(
    tmp_path: Path,
) -> None:
    config = ModelConfigV3(
        d_model=32,
        layers=1,
        heads=4,
        feedforward_dim=64,
        word_vocab_size=256,
        max_words=16,
        max_relative_players=4,
    )
    model = MagicTransformerActorCriticV3(config)
    encoder = CyclicStructuredObservationEncoder(
        word_vocab_size=config.word_vocab_size,
        max_words=config.max_words,
        max_relative_players=config.max_relative_players,
        player_angle_steps=config.player_angle_steps,
    )
    encoded = encoder.encode(
        _structured_test_state(),
        [
            {"id": "pass", "kind": "passPriority", "playerId": "player-3"},
            {"id": "cast", "kind": "castSpell", "playerId": "player-3"},
        ],
    )

    logits, value = model(encoded.state_tokens, encoded.action_tokens)
    assert logits.shape == (1, 2)
    assert value.shape == (1,)

    packed = PackedStructuredTokens.pack(encoded.state_tokens)
    restored_tokens = packed.unpack(torch.device("cpu"))
    assert torch.equal(
        restored_tokens.relative_players,
        encoded.state_tokens.relative_players,
    )

    optimizer = torch.optim.AdamW(model.parameters())
    checkpoint = tmp_path / "checkpoint-v3"
    save_checkpoint(checkpoint, model, optimizer, 12, ["smoke-v3"])
    restored_model, payload = load_checkpoint(checkpoint, torch.device("cpu"))

    assert isinstance(restored_model, MagicTransformerActorCriticV3)
    assert restored_model.config == config
    assert payload["training_step"] == 12


def test_v4_encoder_uses_oracle_models_instead_of_card_names() -> None:
    encoder = OracleStructuredObservationEncoder(
        word_vocab_size=4096,
        max_words=32,
        max_relative_players=4,
    )
    state = _structured_test_state()
    action = {
        "id": "cast:acting",
        "kind": "castSpell",
        "label": "Cast Acting Hand Card",
        "playerId": "player-3",
        "cardInstanceId": "hand:acting",
        "decisions": {"mode": "default"},
    }
    encoded = encoder.encode(state, [action])

    renamed_state = deepcopy(state)
    for player in renamed_state["players"]:
        for zone in ("hand", "battlefield"):
            for card in player[zone]:
                card["definition"]["name"] = "Unseen Card Identity"
    renamed_action = {**action, "label": "Cast Unseen Card Identity"}
    renamed = encoder.encode(renamed_state, [renamed_action])

    assert torch.equal(encoded.state_tokens.word_ids, renamed.state_tokens.word_ids)
    assert torch.equal(encoded.action_tokens.word_ids, renamed.action_tokens.word_ids)

    changed_rules_state = deepcopy(state)
    changed_rules_state["players"][2]["hand"][0]["definition"]["rules"] = [
        {"kind": "spellAbility", "effects": "destroy target creature"},
        {"kind": "keywordAbility", "ability": "flash"},
    ]
    changed_rules = encoder.encode(changed_rules_state, [action])

    assert not torch.equal(
        encoded.state_tokens.word_ids,
        changed_rules.state_tokens.word_ids,
    )
    assert not torch.equal(
        encoded.action_tokens.word_ids,
        changed_rules.action_tokens.word_ids,
    )


def test_v3_checkpoint_weights_upgrade_to_v4_without_changing_v3() -> None:
    config_v3 = ModelConfigV3(
        d_model=32,
        layers=1,
        heads=4,
        feedforward_dim=64,
        word_vocab_size=128,
        max_words=8,
        max_relative_players=4,
    )
    source = MagicTransformerActorCriticV3(config_v3)
    with torch.no_grad():
        source.numeric_projection.weight.fill_(0.375)
    original = {name: value.detach().clone() for name, value in source.state_dict().items()}
    target = MagicTransformerActorCriticV4(ModelConfigV4(**source.export_config()))

    upgraded = upgrade_model(source, target)

    assert isinstance(upgraded, MagicTransformerActorCriticV4)
    assert all(torch.equal(original[name], value) for name, value in upgraded.state_dict().items())
    assert all(torch.equal(original[name], value) for name, value in source.state_dict().items())


def test_v5_conditions_each_legal_action_on_encoded_state() -> None:
    config = ModelConfigV5(
        d_model=32,
        layers=1,
        heads=4,
        feedforward_dim=64,
        word_vocab_size=128,
        max_words=8,
        max_relative_players=4,
        action_layers=2,
    )
    model = MagicTransformerActorCriticV5(config)
    encoder = OracleStructuredObservationEncoder(
        word_vocab_size=config.word_vocab_size,
        max_words=config.max_words,
        max_relative_players=config.max_relative_players,
    )
    encoded = encoder.encode(
        _structured_test_state(),
        [
            {"id": "pass", "kind": "passPriority", "playerId": "player-3"},
            {
                "id": "cast",
                "kind": "castSpell",
                "playerId": "player-3",
                "cardInstanceId": "hand:acting",
            },
        ],
    )

    analysis = model.analyze(encoded.state_tokens, encoded.action_tokens)

    assert analysis["logits"].shape == (1, 2)
    assert analysis["value"].shape == (1,)
    assert analysis["attention"].shape == (
        1,
        config.heads,
        2,
        encoded.state_tokens.numeric.shape[0] + 1,
    )
    assert torch.allclose(
        analysis["attention"].sum(dim=-1),
        torch.ones((1, config.heads, 2)),
        atol=1e-5,
    )


def test_v6_encodes_public_graveyards_and_recent_events_without_hashing() -> None:
    state = _structured_test_state()
    state["players"][0]["graveyard"] = [
        _structured_test_card(
            "grave:known",
            "Ignored Card Name",
            rules=[{"kind": "triggeredAbility", "event": "dies"}],
        )
    ]
    state["events"] = [
        {
            "kind": "cardMoved",
            "playerId": "player-1",
            "from": "battlefield",
            "to": "graveyard",
        }
    ]
    encoder = PlanningObservationEncoder(max_words=48, max_relative_players=4)

    encoded = encoder.encode(
        state,
        [{"id": "pass", "kind": "passPriority", "playerId": "player-3"}],
    )

    token_types = encoded.state_tokens.token_types
    assert token_types.eq(int(TokenTypeV6.GRAVEYARD_CARD)).sum().item() == 1
    assert token_types.eq(int(TokenTypeV6.GAME_EVENT)).sum().item() == 1
    assert any("graveyard card" in label for label in encoded.state_token_labels)
    assert any("game event" in label for label in encoded.state_token_labels)
    assert encoded.state_tokens.word_ids.max().item() < encoder.word_vocab_size


def test_v10_encodes_mode_mulligan_configuration_exile_and_command_zone() -> None:
    state = _structured_test_state()
    state["gameMode"] = "commander"
    state["_decisionContext"] = {
        "id": "mulligan:player-3:0",
        "playerId": "player-3",
        "kind": "mulligan",
        "gameMode": "commander",
        "mulliganEnabled": True,
        "openingHandSize": 7,
        "freeMulligans": 1,
        "maxMulligans": 3,
        "mulligansTaken": 0,
        "freeMulligansRemaining": 1,
        "paidMulligansTaken": 0,
        "mulligansRemaining": 3,
    }
    state["players"][0]["exile"] = [
        _structured_test_card(
            "exile:public",
            "Ignored Exiled Name",
            rules=[{"kind": "spellAbility", "effect": "draw a card"}],
        )
    ]
    state["players"][1]["commandZone"] = [
        _structured_test_card(
            "command:public",
            "Ignored Commander Name",
            rules=[{"kind": "triggeredAbility", "event": "landfall"}],
        )
    ]
    encoder = StrategicPlanningObservationEncoder(
        max_words=64,
        max_relative_players=4,
        max_state_tokens=256,
    )

    encoded = encoder.encode(
        state,
        [
            {
                "id": "cast-exiled",
                "kind": "castSpell",
                "playerId": "player-3",
                "cardInstanceId": "exile:public",
            }
        ],
    )

    configuration = " ".join(
        label for label in encoded.state_token_labels if label.startswith("game_configuration:")
    )
    assert "game mode commander" in configuration
    assert "mulligan rules enabled 1 openinghand 7 free 1 maximum 3" in configuration
    assert "mulligan progress taken 0 freeremaining 1 paid 0 remaining 3" in configuration
    assert any("exile card" in label for label in encoded.state_token_labels)
    assert any("command zone card" in label for label in encoded.state_token_labels)
    assert "draw a card" in encoded.action_token_labels[0]


def test_v11_separates_pregame_knowledge_from_state_differences() -> None:
    state = _structured_test_state()
    state["_decisionContext"] = {
        "playerId": "player-3",
        "kind": "priority",
    }
    commander = _structured_test_card("commander:1", "Public Commander")
    commander["definition"]["isCommander"] = True
    deck_card = _structured_test_card("deck:1", "Known Deck Card")
    state["_pregameDeck"] = [deck_card["definition"]]
    state["_pregameCommanders"] = [{"playerId": "player-1", "card": commander["definition"]}]
    previous = deepcopy(state)
    previous.pop("_pregameDeck")
    previous.pop("_pregameCommanders")
    previous["players"][2]["life"] = state["players"][2]["life"] + 3
    previous["events"] = []
    state["events"] = [{"sequence": 4, "kind": "damageDealt", "playerId": "player-3", "amount": 3}]
    state["_previousObservation"] = previous
    encoder = AlphaStarObservationEncoder(
        max_words=64,
        max_relative_players=4,
        max_state_tokens=256,
    )

    encoded = encoder.encode(
        state,
        [{"id": "pass", "kind": "passPriority", "playerId": "player-3"}],
    )

    token_types = encoded.state_tokens.token_types
    assert token_types.eq(int(TokenTypeV11.PREGAME_DECK_CARD)).sum().item() == 1
    assert token_types.eq(int(TokenTypeV11.COMMANDER)).sum().item() == 1
    assert token_types.eq(int(TokenTypeV11.STATE_DELTA)).sum().item() >= 1
    assert token_types.eq(int(TokenTypeV11.DECISION_EVENT)).sum().item() == 1
    assert any("life decreased by 3" in label for label in encoded.state_token_labels)


def test_v11_alpha_star_core_predicts_every_relative_player() -> None:
    config = ModelConfigV11(
        d_model=32,
        layers=1,
        heads=4,
        feedforward_dim=64,
        action_layers=1,
        plan_layers=1,
        difference_layers=1,
        max_words=32,
        max_relative_players=4,
        semantic_dim=16,
        semantic_layers=1,
        semantic_heads=4,
        future_horizons=2,
        future_player_slots=4,
        future_feature_dim=len(FUTURE_FEATURE_NAMES),
        multiplayer_value_slots=4,
    )
    model = MagicTransformerActorCriticV11(config)
    encoder = AlphaStarObservationEncoder(
        max_words=config.max_words,
        max_relative_players=config.max_relative_players,
        max_state_tokens=256,
    )
    state = _structured_test_state()
    state["_decisionContext"] = {"playerId": "player-3", "kind": "priority"}
    previous = deepcopy(state)
    previous["players"][2]["life"] += 2
    state["_previousObservation"] = previous
    encoded = encoder.encode(
        state,
        [
            {"id": "cast", "kind": "castSpell", "playerId": "player-3"},
            {"id": "pass", "kind": "passPriority", "playerId": "player-3"},
        ],
    )

    analysis = model.analyze(encoded.state_tokens, encoded.action_tokens)

    assert analysis["logits"].shape == (1, 2)
    assert analysis["player_values"].shape == (1, 4)
    assert analysis["strategic_plan"].shape == (1, config.d_model)
    assert analysis["difference_token_count"].item() >= 1


def test_v12_uses_one_antisymmetric_two_player_value() -> None:
    config = ModelConfigV12(
        d_model=32,
        layers=1,
        heads=4,
        feedforward_dim=64,
        action_layers=1,
        plan_layers=1,
        difference_layers=1,
        max_words=32,
        max_relative_players=2,
        semantic_dim=16,
        semantic_layers=1,
        semantic_heads=4,
        future_horizons=2,
        future_player_slots=2,
        future_feature_dim=len(FUTURE_FEATURE_NAMES),
        multiplayer_value_slots=2,
    )
    model = MagicTransformerActorCriticV12(config)
    encoder = AlphaStarObservationEncoder(
        max_words=config.max_words,
        max_relative_players=config.max_relative_players,
        max_state_tokens=256,
    )
    state = _structured_test_state()
    state["players"] = state["players"][:2]
    state["_decisionContext"] = {"playerId": "player-1", "kind": "priority"}
    encoded = encoder.encode(
        state,
        [
            {"id": "cast", "kind": "castSpell", "playerId": "player-1"},
            {"id": "pass", "kind": "passPriority", "playerId": "player-1"},
        ],
    )

    analysis = model.analyze(encoded.state_tokens, encoded.action_tokens)

    assert analysis["player_values"].shape == (1, 2)
    assert analysis["player_values"][0, 0] == pytest.approx(-analysis["player_values"][0, 1])


def test_anchor_matchup_exposes_opening_hand_challenge_and_deadline() -> None:
    matchup = Matchup(
        id="base",
        setup={
            "openingHandSize": 7,
            "players": [
                {"id": "learner", "name": "Learner", "cards": [], "startingLife": 40},
                {"id": "opponent", "name": "Opponent", "cards": [], "startingLife": 40},
            ],
        },
        learner_player_id="learner",
        opponent_player_id="opponent",
        deck_names=("Omnath", "Other"),
        deck_session_ids=("omnath-session", "other-session"),
    )

    anchor = _anchor_matchup(matchup, deadline_round=9, opening_hand_pool_size=60)

    assert anchor.anchor_deadline_round == 9
    assert anchor.anchor_opening_hand_pool_size == 60
    assert anchor.training_anchor_player_ids == ("opponent",)
    assert anchor.punching_bag_player_ids == ()
    assert not anchor.mulligan_enabled
    assert anchor.max_mulligans == 0
    assert len(anchor.setup["players"][1]["cards"]) == 99


def test_plackett_luce_rank_reward_is_zero_sum_and_strength_adjusted() -> None:
    ratings = {
        "underdog": PlackettLuceRating(mu=15.0),
        "favorite": PlackettLuceRating(mu=35.0),
        "middle": PlackettLuceRating(mu=25.0),
    }

    rewards = rank_gradient_rewards(
        ["underdog", "middle", "favorite"],
        ratings,
    )

    assert sum(rewards.values()) == pytest.approx(0.0)
    assert rewards["underdog"] > 0.0
    assert rewards["favorite"] < 0.0


def test_plackett_luce_outcome_preview_makes_draw_strength_adjusted() -> None:
    previews = hypothetical_first_place_deltas(
        [PlackettLuceRating(mu=35.0), PlackettLuceRating(mu=15.0)]
    )

    assert previews[0]["win"] > 0.0
    assert previews[0]["draw"] < 0.0
    assert previews[0]["loss"] < 0.0
    assert previews[1]["win"] > 0.0
    assert previews[1]["draw"] > 0.0
    assert previews[1]["loss"] < 0.0
    assert previews[0]["draw"] + previews[1]["draw"] == pytest.approx(0.0)


def test_training_leaderboard_scores_a_complete_draw_without_seat_order_bias(
    tmp_path: Path,
) -> None:
    leaderboard = TrainingLeaderboard(
        tmp_path / "training-leaderboard.json",
        {"favorite": "Favorite", "underdog": "Underdog"},
    )
    leaderboard.ratings["favorite"] = PlackettLuceRating(mu=35.0)
    leaderboard.ratings["underdog"] = PlackettLuceRating(mu=15.0)

    leaderboard.update(
        {"player-1": "favorite", "player-2": "underdog"},
        {
            "players": [
                {"id": "player-1", "hasLost": False},
                {"id": "player-2", "hasLost": False},
            ],
            "outcome": None,
        },
    )

    assert leaderboard.ratings["favorite"].mu < 35.0
    assert leaderboard.ratings["underdog"].mu > 15.0


def test_anchor_challenge_order_uses_round_pool_then_player_count() -> None:
    challenges = [
        AnchorChallenge(2, 20, 4),
        AnchorChallenge(1, 100, 2),
        AnchorChallenge(1, 20, 2),
        AnchorChallenge(1, 20, 4),
        AnchorChallenge(1, 20, 3),
    ]

    assert [
        challenge.participant_id
        for challenge in sorted(
            challenges,
            key=lambda challenge: challenge.ranking_key,
        )
    ] == [
        "anchor-m01-n020-p4",
        "anchor-m01-n020-p3",
        "anchor-m01-n020-p2",
        "anchor-m01-n100-p2",
        "anchor-m02-n020-p4",
    ]
    assert anchor_participant_id(7, 60, 3) == "anchor-m07-n060-p3"
    assert AnchorChallenge(7, 60, 4).opponent_count == 3
    assert "contre 3 ancres en équipe" in AnchorChallenge(7, 60, 4).label


def test_anchor_calibration_records_games_and_orders_every_challenge(
    tmp_path: Path,
) -> None:
    challenges = anchor_challenges((1, 2, 3), (20, 40), (2, 3, 4))
    labels = {challenge.participant_id: challenge.label for challenge in challenges}
    leaderboard = TrainingLeaderboard(
        tmp_path / "training-leaderboard.json",
        {"v11": "V11", **labels},
    )

    report = leaderboard.calibrate_anchors(challenges, games=900, seed=83)

    assert report["games"] == 900
    assert report["challengeCount"] == 18
    assert report["minimumGamesPerAnchor"] > 0
    assert report["maximumGamesPerAnchor"] - report["minimumGamesPerAnchor"] <= 3
    assert (
        sum(
            row["wins"]
            for row in leaderboard.payload()["participants"]
            if row["id"].startswith("anchor-")
        )
        == 900
    )
    anchor_rows = {
        row["participantId"]: row
        for row in leaderboard.payload()["deckParticipants"]
        if row["participantId"].startswith("anchor-")
    }
    expected = sorted(challenges, key=lambda challenge: challenge.ranking_key)
    actual = sorted(
        challenges,
        key=lambda challenge: -anchor_rows[challenge.participant_id]["ordinal"],
    )
    assert actual == expected
    restored = TrainingLeaderboard(
        tmp_path / "training-leaderboard.json",
        {"v11": "V11", **labels},
    )
    assert restored.anchor_calibration == leaderboard.anchor_calibration


def test_training_leaderboard_persists_model_deck_combinations(
    tmp_path: Path,
) -> None:
    path = tmp_path / "training-leaderboard.json"
    labels = {
        "v11": "V11 (AlphaStar)",
        "anchor-m03-n060": "Anchor M3/N60",
    }
    leaderboard = TrainingLeaderboard(path, labels)

    leaderboard.update(
        {
            "player-1": "v11",
            "player-2": "anchor-m03-n060",
        },
        {},
        ordered_players=["player-1", "player-2"],
        deck_by_player={
            "player-1": "Omnath, Locus of Creation",
            "player-2": "Anchor",
        },
    )

    rows = leaderboard.payload()["deckParticipants"]
    assert [row["deckName"] for row in rows] == [
        "Omnath, Locus of Creation",
        "Anchor",
    ]
    assert rows[0]["participantId"] == "v11"
    assert rows[0]["mu"] > 25.0
    assert rows[0]["wins"] == 1
    assert rows[1]["mu"] < 25.0
    assert rows[1]["wins"] == 0

    restored = TrainingLeaderboard(path, labels)
    assert restored.payload()["deckParticipants"] == rows


def test_training_leaderboard_retries_a_transient_windows_replace_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "training-leaderboard.json"
    leaderboard = TrainingLeaderboard(path, {"v12": "V12"})
    original_replace = Path.replace
    attempts = 0

    def replace_with_transient_lock(source: Path, target: Path) -> Path:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise PermissionError("simulated Windows file lock")
        return original_replace(source, target)

    monkeypatch.setattr(Path, "replace", replace_with_transient_lock)

    leaderboard.save()

    assert attempts == 3
    assert json.loads(path.read_text(encoding="utf-8"))["ratingSystem"] == "plackett-luce"
    assert not list(tmp_path.glob(".*.tmp"))


def test_training_leaderboard_compares_decks_during_same_model_self_play(
    tmp_path: Path,
) -> None:
    leaderboard = TrainingLeaderboard(
        tmp_path / "training-leaderboard.json",
        {"v11": "V11 (AlphaStar)"},
    )

    leaderboard.update(
        {"player-1": "v11", "player-2": "v11"},
        {},
        ordered_players=["player-1", "player-2"],
        deck_by_player={"player-1": "Omnath", "player-2": "Tatyova"},
    )

    assert leaderboard.payload()["participants"][0]["games"] == 0
    deck_rows = leaderboard.payload()["deckParticipants"]
    assert len(deck_rows) == 2
    assert all(row["games"] == 1 for row in deck_rows)
    assert deck_rows[0]["deckName"] == "Omnath"
    assert deck_rows[0]["wins"] == 1
    assert deck_rows[1]["wins"] == 0


def test_anchor_team_counts_as_one_plackett_luce_opponent(tmp_path: Path) -> None:
    leaderboard = TrainingLeaderboard(
        tmp_path / "training-leaderboard.json",
        {"v11": "V11", "anchor-m05-n100-p4": "Anchor team"},
    )
    leaderboard.update(
        {
            "player-1": "v11",
            "player-2": "anchor-m05-n100-p4",
            "player-3": "anchor-m05-n100-p4",
            "player-4": "anchor-m05-n100-p4",
        },
        {},
        ordered_players=["player-2", "player-3", "player-4", "player-1"],
        deck_by_player={
            "player-1": "Omnath",
            "player-2": "Anchor",
            "player-3": "Anchor",
            "player-4": "Anchor",
        },
    )

    assert leaderboard.ratings["anchor-m05-n100-p4"].games == 1
    assert leaderboard.ratings["anchor-m05-n100-p4"].wins == 1
    anchor_entry = leaderboard._deck_entry_id("anchor-m05-n100-p4", "Anchor")
    assert leaderboard.deck_ratings[anchor_entry].games == 1
    assert leaderboard.deck_ratings[anchor_entry].wins == 1


def test_training_leaderboard_v4_backfills_wins_from_training_history(
    tmp_path: Path,
) -> None:
    path = tmp_path / "training-leaderboard.json"
    path.write_text(
        json.dumps(
            {
                "schemaVersion": "oracle-ai-training-leaderboard/v3",
                "participants": [
                    {"id": "v11", "mu": 26, "sigma": 7, "games": 2},
                    {"id": "v10", "mu": 24, "sigma": 7, "games": 2},
                ],
                "deckParticipants": [
                    {
                        "participantId": "v11",
                        "deckName": "Omnath",
                        "mu": 26,
                        "sigma": 7,
                        "games": 2,
                    },
                    {
                        "participantId": "v10",
                        "deckName": "Tatyova",
                        "mu": 24,
                        "sigma": 7,
                        "games": 2,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "training.jsonl").write_text(
        json.dumps(
            {
                "participantsByPlayer": {"player-1": "v11", "player-2": "v10"},
                "decks": ["Omnath", "Tatyova"],
                "outcome": {"winner": "player-1"},
                "gameStatus": "completed",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    leaderboard = TrainingLeaderboard(path, {"v10": "V10", "v11": "V11"})

    assert leaderboard.payload()["schemaVersion"] == "oracle-ai-training-leaderboard/v4"
    assert leaderboard.ratings["v11"].wins == 1
    assert leaderboard.ratings["v10"].wins == 0
    deck_rows = {
        (row["participantId"], row["deckName"]): row
        for row in leaderboard.payload()["deckParticipants"]
    }
    assert deck_rows[("v11", "Omnath")]["wins"] == 1
    assert deck_rows[("v10", "Tatyova")]["wins"] == 0


def test_v6_combines_transformer_action_values_with_puct_visits() -> None:
    config = ModelConfigV6(
        d_model=32,
        layers=1,
        heads=4,
        feedforward_dim=64,
        action_layers=1,
        max_words=32,
        max_relative_players=4,
        semantic_dim=16,
        semantic_layers=1,
        semantic_heads=4,
        root_search_simulations=16,
    )
    model = MagicTransformerActorCriticV6(config)
    encoder = PlanningObservationEncoder(
        max_words=config.max_words,
        max_relative_players=config.max_relative_players,
    )
    encoded = encoder.encode(
        _structured_test_state(),
        [
            {"id": "land", "kind": "playLand", "playerId": "player-3"},
            {"id": "pass", "kind": "passPriority", "playerId": "player-3"},
        ],
    )

    analysis = model.analyze(encoded.state_tokens, encoded.action_tokens)
    improved = model.improve_policy(
        analysis["logits"],
        analysis["action_values"],
    )

    assert analysis["action_values"].shape == (1, 2)
    assert analysis["attention_layers"].shape[0] == config.action_layers
    assert analysis["action_activation_norms"].shape == (2, 1, 2)
    assert torch.isclose(improved.sum(), torch.tensor(1.0))
    assert torch.allclose(
        improved * config.root_search_simulations,
        (improved * config.root_search_simulations).round(),
    )

    explored = model.improve_policy(
        analysis["logits"],
        analysis["action_values"],
        masked_action_indices=(1,),
        add_exploration_noise=True,
    )
    assert torch.isclose(explored.sum(), torch.tensor(1.0))
    assert explored[1].item() == 0.0


def test_v6_semantic_encoder_handles_fully_masked_word_tokens() -> None:
    config = ModelConfigV6(
        d_model=32,
        layers=1,
        heads=4,
        feedforward_dim=64,
        action_layers=1,
        max_words=8,
        max_relative_players=4,
        semantic_dim=16,
        semantic_layers=1,
        semantic_heads=4,
    )
    encoder = ContextualSemanticEncoder(config)
    token_ids = torch.zeros((3, config.max_words), dtype=torch.long)

    encoder.train()
    training_output = encoder(token_ids)
    encoder.eval()
    evaluation_output = encoder(token_ids)

    assert torch.isfinite(training_output).all()
    assert torch.isfinite(evaluation_output).all()
    assert torch.count_nonzero(training_output).item() == 0
    assert torch.count_nonzero(evaluation_output).item() == 0


def test_v7_encodes_known_deck_as_name_free_permutation_stable_tokens() -> None:
    state = _structured_test_state()
    first = _structured_test_card(
        "deck:1",
        "Memorized Name One",
        mana_cost="{1}{G}",
        rules=[{"kind": "triggeredAbility", "event": "landfall", "effect": "draw"}],
    )
    second = _structured_test_card(
        "deck:2",
        "Memorized Name Two",
        mana_cost="{U}",
        rules=[{"kind": "spellAbility", "effect": "counter target spell"}],
    )
    state["_knownDeck"] = [first, second]
    encoder = StrategicPlanningObservationEncoder(
        max_words=64,
        max_relative_players=4,
        max_state_tokens=256,
    )
    actions = [{"id": "pass", "kind": "passPriority", "playerId": "player-3"}]

    encoded = encoder.encode(state, actions)
    reversed_state = deepcopy(state)
    reversed_state["_knownDeck"] = [second, first]
    reversed_encoded = encoder.encode(reversed_state, actions)

    deck_mask = encoded.state_tokens.token_types.eq(int(TokenTypeV7.DECK_CARD))
    assert deck_mask.sum().item() == 2
    assert torch.equal(encoded.state_tokens.word_ids, reversed_encoded.state_tokens.word_ids)
    deck_labels = [label for label in encoded.state_token_labels if label.startswith("deck_card:")]
    assert any("landfall" in label for label in deck_labels)
    assert all("Memorized Name" not in label for label in deck_labels)


def test_v7_strategic_plan_conditions_policy_value_and_diagnostics() -> None:
    config = ModelConfigV7(
        d_model=32,
        layers=1,
        heads=4,
        feedforward_dim=64,
        action_layers=1,
        plan_layers=1,
        max_words=32,
        max_relative_players=4,
        semantic_dim=16,
        semantic_layers=1,
        semantic_heads=4,
    )
    model = MagicTransformerActorCriticV7(config)
    encoder = StrategicPlanningObservationEncoder(
        max_words=config.max_words,
        max_relative_players=config.max_relative_players,
        max_deck_cards=config.max_deck_cards,
    )
    state = _structured_test_state()
    state["_knownDeck"] = [
        _structured_test_card(
            "deck:strategic",
            "Ignored Strategic Card",
            rules=[{"kind": "spellAbility", "effect": "create creature token"}],
        )
    ]
    encoded = encoder.encode(
        state,
        [
            {"id": "cast", "kind": "castSpell", "playerId": "player-3"},
            {"id": "pass", "kind": "passPriority", "playerId": "player-3"},
        ],
    )

    logits, value, action_values = model.evaluate_actions(
        encoded.state_tokens,
        encoded.action_tokens,
    )
    (logits.sum() + value.sum()).backward()
    analysis = model.analyze(encoded.state_tokens, encoded.action_tokens)

    assert logits.shape == (1, 2)
    assert value.shape == (1,)
    assert action_values.shape == (1, 2)
    assert analysis["strategic_plan"].shape == (1, config.d_model)
    assert analysis["deck_attention"].shape[-1] == 1
    assert model.plan_initializer[1].weight.grad is not None


def test_v9_predicts_action_consequences_and_carries_a_gated_plan() -> None:
    config = ModelConfigV9(
        d_model=32,
        layers=1,
        heads=4,
        feedforward_dim=64,
        action_layers=1,
        plan_layers=1,
        max_words=32,
        max_relative_players=4,
        semantic_dim=16,
        semantic_layers=1,
        semantic_heads=4,
        future_player_slots=4,
        future_feature_dim=len(FUTURE_FEATURE_NAMES),
    )
    model = MagicTransformerActorCriticV9(config)
    encoder = StrategicPlanningObservationEncoder(
        max_words=config.max_words,
        max_relative_players=config.max_relative_players,
        max_deck_cards=config.max_deck_cards,
    )
    state = _structured_test_state()
    state["_knownDeck"] = [
        _structured_test_card(
            "deck:v9",
            "Ignored V9 Card",
            rules=[{"kind": "spellAbility", "effect": "draw a card"}],
        )
    ]
    encoded = encoder.encode(
        state,
        [
            {"id": "cast", "kind": "castSpell", "playerId": "player-3"},
            {"id": "pass", "kind": "passPriority", "playerId": "player-3"},
        ],
    )

    first = model.evaluate_actions_with_memory(
        encoded.state_tokens,
        encoded.action_tokens,
    )
    second = model.evaluate_actions_with_memory(
        encoded.state_tokens,
        encoded.action_tokens,
        first["strategic_plan"].detach(),
    )
    loss = (
        first["logits"].sum()
        + first["value"].sum()
        + first["future_mean"].sum()
        + first["belief_prediction"].sum()
    )
    loss.backward()

    assert first["logits"].shape == (1, 2)
    assert first["future_mean"].shape == (
        1,
        2,
        config.future_horizons,
        config.future_player_slots,
        len(FUTURE_FEATURE_NAMES),
    )
    assert first["future_log_variance"].shape == first["future_mean"].shape
    assert first["belief_prediction"].shape == (1, 4, len(FUTURE_FEATURE_NAMES))
    assert first["plan_gate"].min() >= 0.0
    assert first["plan_gate"].max() <= 1.0
    assert first["future_mean"].max() <= 0.0
    assert first["future_mean"].min() >= -1.0
    assert not torch.equal(first["strategic_plan"], second["strategic_plan"])
    assert model.consequence_head[-1].weight.grad is not None
    assert not callable(model.improve_policy)


def test_v9_future_targets_are_relative_and_include_event_flows() -> None:
    current = _structured_test_state()
    future = deepcopy(current)
    future["players"][2]["life"] = 15
    future["players"][2]["hand"].append(_structured_test_card("hand:drawn", "Drawn Card"))
    future["events"] = [
        {
            "sequence": 1,
            "turnNumber": 5,
            "step": "draw",
            "kind": "cardDrawn",
            "playerId": "player-3",
            "detail": {},
        }
    ]

    targets, mask = future_feature_targets(
        current,
        [future],
        "player-3",
        player_slots=4,
        horizons=4,
    )

    assert targets.shape == (4, 4, len(FUTURE_FEATURE_NAMES))
    assert mask[:, :3].all()
    assert not mask[:, 3].any()
    assert targets[0, 0, FUTURE_FEATURE_NAMES.index("life")].item() == pytest.approx(
        normalize_nonnegative(15)
    )
    assert targets[0, 0, FUTURE_FEATURE_NAMES.index("retained_hand_count")].item() == pytest.approx(
        normalize_nonnegative(1)
    )
    assert targets[0, 0, FUTURE_FEATURE_NAMES.index("new_hand_count")].item() == pytest.approx(
        normalize_nonnegative(1)
    )
    assert targets[0, 0, FUTURE_FEATURE_NAMES.index("total_hand_count")].item() == pytest.approx(
        normalize_nonnegative(2)
    )
    assert targets[0, 0, FUTURE_FEATURE_NAMES.index("cards_drawn")].item() == pytest.approx(
        normalize_nonnegative(1)
    )


def test_v9_ppo_trains_future_belief_and_plan_auxiliary_heads() -> None:
    config = ModelConfigV9(
        d_model=32,
        layers=1,
        heads=4,
        feedforward_dim=64,
        action_layers=1,
        plan_layers=1,
        max_words=32,
        max_relative_players=4,
        semantic_dim=16,
        semantic_layers=1,
        semantic_heads=4,
        future_feature_dim=len(FUTURE_FEATURE_NAMES),
    )
    model = MagicTransformerActorCriticV9(config)
    learner = PPOLearner(
        model,
        StrategicPlanningObservationEncoder(
            max_words=config.max_words,
            max_relative_players=config.max_relative_players,
            max_deck_cards=config.max_deck_cards,
        ),
        PPOConfig(
            epochs=1,
            minibatch_size=1,
            future_prediction_coefficient=0.1,
            belief_coefficient=0.025,
            plan_coefficient=0.01,
        ),
        torch.device("cpu"),
    )
    initial = _structured_test_state()
    terminal = deepcopy(initial)
    terminal["status"] = "completed"
    terminal["outcome"] = {
        "winner": "player-3",
        "losers": ["player-1", "player-2"],
    }

    class OneDecisionV9Environment:
        def reset(self, matchup_id: str, seed: int, seat_swap: bool) -> DecisionStep:
            return DecisionStep(
                initial,
                [
                    {"id": "cast", "kind": "castSpell", "playerId": "player-3"},
                    {"id": "pass", "kind": "passPriority", "playerId": "player-3"},
                ],
                0.0,
                False,
                "player-3",
            )

        def step(self, action_index: int) -> DecisionStep:
            return DecisionStep(
                terminal,
                [],
                0.0,
                True,
                rewards_by_player={"player-3": 1.0},
            )

    trajectory, _ = learner.collect_self_play_episode(
        OneDecisionV9Environment(),
        "v9-auxiliary",
        7,
    )
    consequence_before = model.consequence_head[-1].weight.detach().clone()
    metrics = learner.update(trajectory)

    assert trajectory[0].future_targets is not None
    assert trajectory[0].plan_target is not None
    assert math.isfinite(metrics["future_prediction_loss"])
    assert math.isfinite(metrics["belief_loss"])
    assert math.isfinite(metrics["plan_loss"])
    assert not torch.equal(consequence_before, model.consequence_head[-1].weight)


def test_v10_quantizes_contextual_events_and_keeps_concrete_actions() -> None:
    config = ModelConfigV10(
        d_model=32,
        layers=1,
        heads=4,
        feedforward_dim=64,
        action_layers=1,
        plan_layers=1,
        max_words=32,
        max_relative_players=4,
        semantic_dim=16,
        semantic_layers=1,
        semantic_heads=4,
        future_feature_dim=len(FUTURE_FEATURE_NAMES),
        event_codebook_size=8,
        event_latent_dim=8,
        root_search_simulations=4,
    )
    model = MagicTransformerActorCriticV10(config)
    encoder = StrategicPlanningObservationEncoder(
        max_words=config.max_words,
        max_relative_players=config.max_relative_players,
        max_deck_cards=config.max_deck_cards,
    )
    encoded = encoder.encode(
        _structured_test_state(),
        [
            {"id": "cast", "kind": "castSpell", "playerId": "player-3"},
            {"id": "pass", "kind": "passPriority", "playerId": "player-3"},
        ],
    )

    result = model.evaluate_actions_with_memory(
        encoded.state_tokens,
        encoded.action_tokens,
    )
    improved = model.improve_policy(
        result["logits"],
        result["action_values"],
        event_code_indices=result["event_code_indices"],
    )

    assert result["logits"].shape == (1, 2)
    assert result["player_values"].shape == (1, 4)
    assert result["event_code_indices"].shape == (1, 2)
    assert result["event_code_indices"].min() >= 0
    assert result["event_code_indices"].max() < config.event_codebook_size
    assert result["event_reconstructed_future"].shape == (
        1,
        2,
        config.future_horizons,
        config.future_player_slots,
        len(FUTURE_FEATURE_NAMES),
    )
    assert improved.shape == (2,)
    assert improved.sum().item() == pytest.approx(1.0)


def test_v10_prediction_losses_are_gradient_isolated_from_choice_tensors() -> None:
    config = ModelConfigV10(
        d_model=32,
        layers=1,
        heads=4,
        feedforward_dim=64,
        action_layers=1,
        plan_layers=1,
        max_words=32,
        max_relative_players=4,
        semantic_dim=16,
        semantic_layers=1,
        semantic_heads=4,
        future_feature_dim=len(FUTURE_FEATURE_NAMES),
        event_codebook_size=8,
        event_latent_dim=8,
        root_search_simulations=2,
    )
    model = MagicTransformerActorCriticV10(config)
    encoder = StrategicPlanningObservationEncoder(
        max_words=config.max_words,
        max_relative_players=config.max_relative_players,
        max_deck_cards=config.max_deck_cards,
    )
    encoded = encoder.encode(
        _structured_test_state(),
        [
            {"id": "cast", "kind": "castSpell", "playerId": "player-3"},
            {"id": "pass", "kind": "passPriority", "playerId": "player-3"},
        ],
    )

    def has_gradient(module: torch.nn.Module) -> bool:
        return any(
            parameter.grad is not None and bool(parameter.grad.abs().sum() > 0)
            for parameter in module.parameters()
        )

    prediction = model.evaluate_actions_with_memory(
        encoded.state_tokens,
        encoded.action_tokens,
    )
    (prediction["future_mean"].sum() + prediction["belief_prediction"].sum()).backward()
    assert has_gradient(model.consequence_head)
    assert has_gradient(model.belief_prediction_head)
    assert not has_gradient(model.policy_key)
    assert not has_gradient(model.future_encoder)

    model.zero_grad(set_to_none=True)
    choice = model.evaluate_actions_with_memory(
        encoded.state_tokens,
        encoded.action_tokens,
    )
    (choice["logits"].sum() + choice["action_values"].sum()).backward()
    assert has_gradient(model.policy_key)
    assert not has_gradient(model.consequence_head)
    assert not has_gradient(model.event_encoder)

    model.zero_grad(set_to_none=True)
    event_prediction = model.evaluate_actions_with_memory(
        encoded.state_tokens,
        encoded.action_tokens,
    )
    event_prediction["event_reconstructed_future"].sum().backward()
    assert has_gradient(model.event_encoder)
    assert has_gradient(model.event_decoder)
    assert not has_gradient(model.policy_key)
    assert not has_gradient(model.consequence_head)


def test_v10_ppo_trains_vq_event_and_multiplayer_value_heads() -> None:
    config = ModelConfigV10(
        d_model=32,
        layers=1,
        heads=4,
        feedforward_dim=64,
        action_layers=1,
        plan_layers=1,
        max_words=32,
        max_relative_players=4,
        semantic_dim=16,
        semantic_layers=1,
        semantic_heads=4,
        future_feature_dim=len(FUTURE_FEATURE_NAMES),
        event_codebook_size=8,
        event_latent_dim=8,
        root_search_simulations=2,
    )
    model = MagicTransformerActorCriticV10(config)
    learner = PPOLearner(
        model,
        StrategicPlanningObservationEncoder(
            max_words=config.max_words,
            max_relative_players=config.max_relative_players,
            max_deck_cards=config.max_deck_cards,
        ),
        PPOConfig(
            epochs=1,
            minibatch_size=1,
            multiplayer_value_coefficient=0.2,
            event_reconstruction_coefficient=0.1,
            event_codebook_coefficient=0.05,
            event_commitment_coefficient=0.0125,
            latent_value_coefficient=0.2,
        ),
        torch.device("cpu"),
    )
    initial = _structured_test_state()
    terminal = deepcopy(initial)
    terminal["status"] = "completed"
    terminal["outcome"] = {
        "winner": "player-3",
        "losers": ["player-1", "player-2"],
    }

    class OneDecisionV10Environment:
        def reset(self, matchup_id: str, seed: int, seat_swap: bool) -> DecisionStep:
            return DecisionStep(
                initial,
                [
                    {"id": "cast", "kind": "castSpell", "playerId": "player-3"},
                    {"id": "pass", "kind": "passPriority", "playerId": "player-3"},
                ],
                0.0,
                False,
                "player-3",
            )

        def step(self, action_index: int) -> DecisionStep:
            return DecisionStep(
                terminal,
                [],
                0.0,
                True,
                rewards_by_player={
                    "player-1": -0.5,
                    "player-2": -0.5,
                    "player-3": 1.0,
                },
            )

    trajectory, _ = learner.collect_self_play_episode(
        OneDecisionV10Environment(),
        "v10-events",
        10,
    )
    codebook_before = model.event_codebook.weight.detach().clone()
    decoder_before = model.event_decoder[-1].weight.detach().clone()
    multiplayer_before = model.multiplayer_value_head[-2].weight.detach().clone()
    metrics = learner.update(trajectory)

    assert trajectory[0].player_value_targets is not None
    assert trajectory[0].value_player_ids == (
        "player-3",
        "player-1",
        "player-2",
    )
    assert math.isfinite(metrics["event_reconstruction_loss"])
    assert math.isfinite(metrics["event_codebook_loss"])
    assert math.isfinite(metrics["event_commitment_loss"])
    assert math.isfinite(metrics["latent_value_loss"])
    assert math.isfinite(metrics["multiplayer_value_loss"])
    assert not torch.equal(codebook_before, model.event_codebook.weight)
    assert not torch.equal(decoder_before, model.event_decoder[-1].weight)
    assert not torch.equal(
        multiplayer_before,
        model.multiplayer_value_head[-2].weight,
    )


def test_v9_checkpoint_can_upgrade_to_v10_without_losing_shared_weights() -> None:
    shared = dict(
        d_model=32,
        layers=1,
        heads=4,
        feedforward_dim=64,
        action_layers=1,
        plan_layers=1,
        max_words=32,
        max_relative_players=4,
        semantic_dim=16,
        semantic_layers=1,
        semantic_heads=4,
        future_feature_dim=len(FUTURE_FEATURE_NAMES),
    )
    source = MagicTransformerActorCriticV9(ModelConfigV9(**shared))
    target = MagicTransformerActorCriticV10(
        ModelConfigV10(
            **shared,
            event_codebook_size=8,
            event_latent_dim=8,
        )
    )

    upgraded = upgrade_model(source, target)

    assert isinstance(upgraded, MagicTransformerActorCriticV10)
    assert torch.equal(
        upgraded.state_marker,
        source.state_marker,
    )


def test_v10_checkpoint_expands_hand_features_without_losing_existing_outputs() -> None:
    shared = dict(
        d_model=32,
        layers=1,
        heads=4,
        feedforward_dim=64,
        action_layers=1,
        plan_layers=1,
        max_words=32,
        max_relative_players=4,
        semantic_dim=16,
        semantic_layers=1,
        semantic_heads=4,
        future_horizons=2,
        future_player_slots=2,
        event_codebook_size=8,
        event_latent_dim=8,
    )
    source = MagicTransformerActorCriticV10(ModelConfigV10(**shared, future_feature_dim=21))
    target = MagicTransformerActorCriticV10(
        ModelConfigV10(**shared, future_feature_dim=len(FUTURE_FEATURE_NAMES))
    )
    with torch.no_grad():
        source.consequence_head[-1].weight.copy_(
            torch.arange(
                source.consequence_head[-1].weight.numel(),
                dtype=torch.float32,
            ).reshape_as(source.consequence_head[-1].weight)
        )
        source.event_decoder[-1].bias.copy_(
            torch.arange(source.event_decoder[-1].bias.numel(), dtype=torch.float32)
        )

    upgraded = upgrade_model(source, target)
    old_features = 21
    new_features = len(FUTURE_FEATURE_NAMES)
    old_hand_mean = (0 * old_features + 1) * 2
    new_total_mean = (0 * new_features + FUTURE_FEATURE_NAMES.index("total_hand_count")) * 2
    old_library_event = 0 * old_features + 2
    new_library_event = 0 * new_features + FUTURE_FEATURE_NAMES.index("library_count")

    assert upgraded.config.future_feature_dim == 23
    assert torch.equal(
        upgraded.consequence_head[-1].weight[new_total_mean],
        source.consequence_head[-1].weight[old_hand_mean],
    )
    assert upgraded.event_decoder[-1].bias[new_library_event].item() == pytest.approx(
        source.event_decoder[-1].bias[old_library_event].item()
    )
    assert torch.equal(upgraded.event_codebook.weight, source.event_codebook.weight)


def test_v6_can_freeze_its_untrained_weights_as_initial_ground_truth(
    tmp_path: Path,
) -> None:
    model = MagicTransformerActorCriticV6(
        ModelConfigV6(
            d_model=32,
            layers=1,
            heads=4,
            feedforward_dim=64,
            action_layers=1,
            max_words=32,
            max_relative_players=4,
            semantic_dim=16,
            semantic_layers=1,
            semantic_heads=4,
        )
    )
    checkpoint = tmp_path / "champions" / "ia-gt-0"

    _initialize_ground_truth_checkpoint(
        checkpoint,
        model,
        PPOConfig(),
        ["smoke-v6"],
        {"initialGroundTruthMode": "learner-step-zero"},
    )

    frozen, payload = load_checkpoint(checkpoint, torch.device("cpu"))
    assert isinstance(frozen, MagicTransformerActorCriticV6)
    assert payload["training_step"] == 0
    assert all(
        torch.equal(weights, frozen.state_dict()[name])
        for name, weights in model.state_dict().items()
    )


def test_training_control_defaults_to_running_and_accepts_pause(tmp_path: Path) -> None:
    control = tmp_path / "training-control.json"
    assert _training_control_state(control) == "running"

    control.write_text(json.dumps({"desiredState": "paused"}), encoding="utf-8")
    assert _training_control_state(control) == "paused"

    control.write_text(json.dumps({"desiredState": "invalid"}), encoding="utf-8")
    assert _training_control_state(control) == "running"


def test_v5_checkpoint_weights_upgrade_to_v6_with_new_semantic_and_value_heads() -> None:
    source = MagicTransformerActorCriticV5(
        ModelConfigV5(
            d_model=32,
            layers=1,
            heads=4,
            feedforward_dim=64,
            word_vocab_size=128,
            max_words=8,
            max_relative_players=4,
            action_layers=1,
        )
    )
    with torch.no_grad():
        source.numeric_projection.weight.fill_(0.4375)
    target = MagicTransformerActorCriticV6(
        ModelConfigV6(
            d_model=32,
            layers=1,
            heads=4,
            feedforward_dim=64,
            max_words=32,
            max_relative_players=4,
            action_layers=1,
            semantic_dim=16,
            semantic_layers=1,
            semantic_heads=4,
        )
    )

    upgraded = upgrade_model(source, target)

    assert isinstance(upgraded, MagicTransformerActorCriticV6)
    assert torch.all(upgraded.numeric_projection.weight.eq(0.4375))
    assert upgraded.token_type_embedding.weight.shape[0] == len(TokenTypeV6)


def test_v4_checkpoint_weights_upgrade_to_v5_without_changing_v4() -> None:
    config_v4 = ModelConfigV4(
        d_model=32,
        layers=1,
        heads=4,
        feedforward_dim=64,
        word_vocab_size=128,
        max_words=8,
        max_relative_players=4,
    )
    source = MagicTransformerActorCriticV4(config_v4)
    with torch.no_grad():
        source.numeric_projection.weight.fill_(0.625)
    original = {name: value.detach().clone() for name, value in source.state_dict().items()}
    target = MagicTransformerActorCriticV5(ModelConfigV5(**source.export_config(), action_layers=2))

    upgraded = upgrade_model(source, target)

    encoder = OracleStructuredObservationEncoder(
        word_vocab_size=config_v4.word_vocab_size,
        max_words=config_v4.max_words,
        max_relative_players=config_v4.max_relative_players,
    )
    encoded = encoder.encode(
        _structured_test_state(),
        [
            {"id": "pass", "kind": "passPriority", "playerId": "player-3"},
            {"id": "cast", "kind": "castSpell", "playerId": "player-3"},
        ],
    )
    source.eval()
    upgraded.eval()
    source_logits, source_value = source(encoded.state_tokens, encoded.action_tokens)
    upgraded_logits, upgraded_value = upgraded(
        encoded.state_tokens,
        encoded.action_tokens,
    )

    assert isinstance(upgraded, MagicTransformerActorCriticV5)
    assert torch.equal(source_logits, upgraded_logits)
    assert torch.equal(source_value, upgraded_value)
    assert all(torch.equal(original[name], upgraded.state_dict()[name]) for name in original)
    assert all(torch.equal(original[name], source.state_dict()[name]) for name in original)


def test_behavior_summary_flags_mulligan_to_zero() -> None:
    state = {
        "turnNumber": 0,
        "players": [{"id": "player-1", "hand": [{}] * 7}],
        "_decisionContext": {
            "id": "mulligan:player-1:6",
            "playerId": "player-1",
            "kind": "mulligan",
            "freeMulligans": 0,
        },
    }
    actions = [
        {"id": "keep", "kind": "keepHand"},
        {"id": "take", "kind": "takeMulligan"},
    ]

    trace = build_decision_trace(
        state,
        actions,
        1,
        confidence=0.8,
        entropy=0.2,
    )
    summary = summarize_decision_traces([trace])

    assert trace.projected_hand_size == 0
    assert dominated_action_indices(state, actions) == (1,)
    assert "mulliganToZero" in trace.anomalies
    assert summary["mulligan"]["toZero"] == 1
    assert summary["mulligan"]["criticalToOneOrLess"] == 1


def test_dominated_actions_prevent_missing_a_legal_land_for_the_turn() -> None:
    state = {
        "step": "postcombatMain",
        "activePlayer": 1,
        "players": [{"id": "opponent"}, {"id": "agent"}],
        "_decisionContext": {
            "id": "priority:2:1:0",
            "playerId": "agent",
            "kind": "priority",
        },
    }
    actions = [
        {"id": "land-a", "kind": "playLand"},
        {"id": "land-b", "kind": "playLand"},
        {"id": "spell", "kind": "castSpell"},
        {"id": "pass", "kind": "passPriority"},
    ]

    assert dominated_action_indices(state, actions) == (3,)


def test_dominated_actions_allow_passing_before_combat_or_without_a_land() -> None:
    state = {
        "step": "precombatMain",
        "activePlayer": "agent",
        "players": [{"id": "agent"}, {"id": "opponent"}],
        "_decisionContext": {
            "id": "priority:3:0:0",
            "playerId": "agent",
            "kind": "priority",
        },
    }
    actions = [
        {"id": "land", "kind": "playLand"},
        {"id": "pass", "kind": "passPriority"},
    ]

    assert dominated_action_indices(state, actions) == ()
    state["step"] = "postcombatMain"
    assert dominated_action_indices(state, actions[:0] + actions[1:]) == ()


def test_observation_dropout_masks_structured_state_only_during_training() -> None:
    config = ModelConfigV2(
        d_model=32,
        layers=1,
        heads=4,
        feedforward_dim=64,
        word_vocab_size=128,
        max_words=8,
        max_relative_players=4,
    )
    learner = PPOLearner(
        MagicTransformerActorCriticV2(config),
        StructuredObservationEncoder(
            word_vocab_size=config.word_vocab_size,
            max_words=config.max_words,
            max_relative_players=config.max_relative_players,
        ),
        PPOConfig(
            observation_token_dropout=1.0,
            observation_word_dropout=1.0,
            observation_numeric_dropout=1.0,
        ),
        torch.device("cpu"),
    )
    encoded = learner.encoder.encode(
        _structured_test_state(),
        [{"id": "pass", "kind": "passPriority", "playerId": "player-3"}],
    )

    masked = learner._apply_observation_dropout(encoded.state_tokens)

    assert isinstance(masked, StructuredTokens)
    assert masked.token_types.tolist() == [
        int(TokenType.GAME_CONFIGURATION),
        int(TokenType.GAME_PHASE),
        int(TokenType.GAME_CONFIGURATION),
        int(TokenType.GAME_CONFIGURATION),
    ]
    assert torch.count_nonzero(masked.word_ids).item() == 0
    assert torch.all(masked.numeric[masked.numeric_mask].eq(-1.0))

    learner.model.eval()
    unmasked = learner._apply_observation_dropout(encoded.state_tokens)
    assert unmasked is encoded.state_tokens


def test_self_play_search_defers_training_observation_dropout_until_update() -> None:
    config = ModelConfigV2(
        d_model=32,
        layers=1,
        heads=4,
        feedforward_dim=64,
        word_vocab_size=128,
        max_words=8,
        max_relative_players=4,
    )
    learner = PPOLearner(
        MagicTransformerActorCriticV2(config),
        StructuredObservationEncoder(
            word_vocab_size=config.word_vocab_size,
            max_words=config.max_words,
            max_relative_players=config.max_relative_players,
        ),
        PPOConfig(
            observation_token_dropout=1.0,
            observation_word_dropout=1.0,
            observation_numeric_dropout=1.0,
        ),
        torch.device("cpu"),
    )
    observed: dict[str, object] = {}

    class OneStepEnvironment:
        def reset(self, matchup_id: str, seed: int, seat_swap: bool) -> DecisionStep:
            return DecisionStep(
                state=_structured_test_state(),
                actions=[{"id": "pass", "kind": "passPriority", "playerId": "player-3"}],
                reward=0.0,
                done=False,
                player_id="player-3",
            )

        def step(self, action_index: int) -> DecisionStep:
            return DecisionStep(
                state={},
                actions=[],
                reward=0.0,
                done=True,
                rewards_by_player={"player-3": 0.0},
            )

    def select_action(
        state_tokens: torch.Tensor | StructuredTokens,
        action_tokens: torch.Tensor | StructuredTokens,
        state: dict,
        actions: list[dict],
        *,
        add_exploration_noise: bool = False,
        previous_plan: torch.Tensor | None = None,
    ) -> tuple[
        int,
        float,
        float,
        float,
        float,
        tuple[float, ...],
        dict[str, torch.Tensor],
    ]:
        observed["training"] = learner.model.training
        observed["token_types"] = state_tokens.token_types.tolist()
        observed["exploration"] = add_exploration_noise
        return 0, 0.0, 0.0, 1.0, 0.0, (1.0,), {}

    learner._select_action = select_action
    trajectory, terminal = learner.collect_self_play_episode(
        OneStepEnvironment(),
        matchup_id="one-step",
        seed=1,
    )

    assert observed == {
        "training": False,
        "exploration": True,
        "token_types": learner.encoder.encode(
            _structured_test_state(),
            [{"id": "pass", "kind": "passPriority", "playerId": "player-3"}],
        ).state_tokens.token_types.tolist(),
    }
    assert learner.model.training
    assert len(trajectory) == 1
    assert trajectory[0].apply_observation_dropout
    assert terminal.done

    original_dropout = learner._apply_observation_dropout

    def track_dropout(
        tokens: torch.Tensor | StructuredTokens,
        *,
        enabled: bool | None = None,
    ) -> torch.Tensor | StructuredTokens:
        observed["update_dropout"] = enabled
        return original_dropout(tokens, enabled=enabled)

    learner._apply_observation_dropout = track_dropout
    learner.update(trajectory)

    assert observed["update_dropout"] is True


def test_v2_checkpoint_weights_upgrade_to_v3_without_changing_v2() -> None:
    config_v2 = ModelConfigV2(
        d_model=32,
        layers=1,
        heads=4,
        feedforward_dim=64,
        word_vocab_size=128,
        max_words=8,
        max_relative_players=4,
    )
    source = MagicTransformerActorCriticV2(config_v2)
    with torch.no_grad():
        source.numeric_projection.weight.fill_(0.25)
        source.relative_player_embedding.weight.copy_(
            torch.arange(5 * 32, dtype=torch.float32).reshape(5, 32)
        )
    original_relative_players = source.relative_player_embedding.weight.detach().clone()
    target = MagicTransformerActorCriticV3(ModelConfigV3(**source.export_config()))

    upgraded = upgrade_model(source, target)

    assert isinstance(upgraded, MagicTransformerActorCriticV3)
    assert torch.all(upgraded.numeric_projection.weight.eq(0.25))
    assert torch.equal(
        upgraded.no_player_embedding,
        original_relative_players[-1:],
    )
    assert torch.equal(
        source.relative_player_embedding.weight,
        original_relative_players,
    )


def test_packed_tensor_round_trip_compresses_sparse_observations() -> None:
    tensor = torch.zeros((512, 256), dtype=torch.float32)
    tensor[0, 0] = 0.75
    tensor[128, 64] = -0.5

    packed = PackedTensor.pack(tensor)
    restored = packed.unpack(torch.device("cpu"))

    assert len(packed.payload) < tensor.numel()
    assert torch.allclose(restored, tensor, atol=1e-3)


def test_ppo_updates_and_checkpoint_round_trip(tmp_path: Path) -> None:
    torch.manual_seed(7)
    config = ModelConfig(feature_dim=32, d_model=32, layers=1, heads=4, feedforward_dim=64)
    model = MagicTransformerActorCritic(config)
    learner = PPOLearner(
        model,
        HashingObservationEncoder(feature_dim=32),
        PPOConfig(epochs=1),
        torch.device("cpu"),
    )
    environment = TinySelfPlayEnvironment(horizon=4)
    trajectory = learner.collect_episode(environment, "smoke", seed=11, seat_swap=False)
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    metrics = learner.update(trajectory)
    assert metrics["loss"] == metrics["loss"]
    assert any(not torch.equal(before[name], value) for name, value in model.state_dict().items())

    checkpoint = tmp_path / "checkpoint"
    save_checkpoint(checkpoint, model, learner.optimizer, learner.training_step, ["smoke"])
    restored, payload = load_checkpoint(checkpoint, torch.device("cpu"))
    assert payload["training_step"] == learner.training_step
    assert "optimizer" in payload
    assert restored.config == model.config

    playable_checkpoint = tmp_path / "playable-checkpoint"
    save_checkpoint(
        playable_checkpoint,
        model,
        learner.optimizer,
        learner.training_step,
        ["smoke"],
        include_optimizer=False,
    )
    playable_model, playable_payload = load_checkpoint(playable_checkpoint, torch.device("cpu"))
    assert "optimizer" not in playable_payload
    assert playable_model.config == model.config
    assert (playable_checkpoint / "checkpoint.pt").stat().st_size < (
        checkpoint / "checkpoint.pt"
    ).stat().st_size


def test_v2_ppo_and_checkpoint_round_trip(tmp_path: Path) -> None:
    torch.manual_seed(13)
    config = ModelConfigV2(
        d_model=32,
        layers=1,
        heads=4,
        feedforward_dim=64,
        word_vocab_size=128,
        max_words=8,
        max_relative_players=4,
    )
    model = MagicTransformerActorCriticV2(config)
    learner = PPOLearner(
        model,
        StructuredObservationEncoder(
            word_vocab_size=config.word_vocab_size,
            max_words=config.max_words,
            max_relative_players=config.max_relative_players,
        ),
        PPOConfig(epochs=1, minibatch_size=2),
        torch.device("cpu"),
    )
    trajectory = learner.collect_episode(
        TinySelfPlayEnvironment(horizon=2),
        "smoke-v2",
        seed=17,
        seat_swap=False,
    )
    metrics = learner.update(trajectory)
    assert metrics["loss"] == metrics["loss"]

    checkpoint = tmp_path / "checkpoint-v2"
    save_checkpoint(
        checkpoint,
        model,
        learner.optimizer,
        learner.training_step,
        ["smoke-v2"],
    )
    restored, payload = load_checkpoint(checkpoint, torch.device("cpu"))

    assert isinstance(restored, MagicTransformerActorCriticV2)
    assert restored.config == model.config
    assert payload["training_step"] == learner.training_step


def test_loader_keeps_legacy_v1_manifest_compatible(tmp_path: Path) -> None:
    model = MagicTransformerActorCritic(
        ModelConfig(feature_dim=16, d_model=16, layers=1, heads=4, feedforward_dim=32)
    )
    optimizer = torch.optim.AdamW(model.parameters())
    checkpoint = tmp_path / "legacy-v1"
    save_checkpoint(checkpoint, model, optimizer, 4, ["legacy"])
    manifest_path = checkpoint / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("model_family")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    restored, payload = load_checkpoint(checkpoint, torch.device("cpu"))

    assert isinstance(restored, MagicTransformerActorCritic)
    assert payload["training_step"] == 4


def test_rust_environment_removes_a_completed_session() -> None:
    requests: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append((request.method, request.url.path))
        if request.method == "POST" and request.url.path == "/game/sessions":
            payload = json.loads(request.content)
            assert payload["mulliganEnabled"] is True
            assert payload["freeMulligans"] == 2
            assert payload["maxMulligans"] == 3
            assert payload["waitTimeoutMs"] == 29_000
            return httpx.Response(
                200,
                json={
                    "sessionId": "session:1",
                    "revision": 1,
                    "state": {
                        "gameMode": "commander",
                        "winnerIds": [],
                        "players": [
                            {
                                "id": "learner",
                                "library": [
                                    {
                                        "instanceId": "selected:1",
                                        "definition": {"name": "Selected Card"},
                                    }
                                ],
                                "hand": [],
                            }
                        ],
                    },
                    "decision": {
                        "id": "mulligan:learner:1",
                        "kind": "mulligan",
                        "playerId": "learner",
                        "options": [{"id": "pass"}],
                    },
                },
            )
        if request.method == "POST" and request.url.path.endswith("/actions"):
            return httpx.Response(
                200,
                json={
                    "sessionId": "session:1",
                    "revision": 2,
                    "state": {
                        "outcome": {
                            "winner": "learner",
                            "losers": ["opponent"],
                            "reason": "lifeTotal",
                            "turnNumber": 8,
                        }
                    },
                },
            )
        if request.method == "DELETE" and request.url.path == "/game/sessions/session:1":
            return httpx.Response(200, json={"removed": True})
        return httpx.Response(404)

    matchup = Matchup(
        id="cleanup",
        setup={"openingHandSize": 8, "players": []},
        learner_player_id="learner",
        opponent_player_id="opponent",
        mulligan_enabled=True,
        free_mulligans=2,
        max_mulligans=3,
        game_mode="commander",
    )
    environment = RustSessionEnvironment(
        "http://engine.test",
        {matchup.id: matchup},
    )
    environment.client.close()
    environment.client = httpx.Client(
        base_url="http://engine.test",
        transport=httpx.MockTransport(handler),
    )

    initial = environment.reset(matchup.id, seed=7, seat_swap=False)
    terminal = environment.step(0)

    assert not initial.done
    assert initial.state["_knownDeck"] == [{"name": "Selected Card"}]
    assert initial.state["_decisionContext"] == {
        "id": "mulligan:learner:1",
        "playerId": "learner",
        "kind": "mulligan",
        "gameMode": "commander",
        "mulliganEnabled": True,
        "openingHandSize": 8,
        "freeMulligans": 2,
        "maxMulligans": 3,
        "mulligansTaken": 1,
        "freeMulligansRemaining": 1,
        "paidMulligansTaken": 0,
        "mulligansRemaining": 2,
    }
    assert terminal.done
    assert terminal.reward == 1.0
    assert environment.session_id is None
    assert ("DELETE", "/game/sessions/session:1") in requests
    environment.close()


def test_rust_environment_reports_rejected_legal_action() -> None:
    requests: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append((request.method, request.url.path))
        if request.method == "POST" and request.url.path == "/game/sessions":
            return httpx.Response(
                200,
                json={
                    "sessionId": "session:2",
                    "revision": 4,
                    "state": {},
                    "decision": {
                        "id": "decision:2",
                        "kind": "priority",
                        "playerId": "learner",
                        "options": [{"id": "cast:1", "kind": "castSpell"}],
                    },
                },
            )
        if request.method == "POST" and request.url.path.endswith("/actions"):
            return httpx.Response(400, json={"error": "action became invalid"})
        if request.method == "DELETE":
            return httpx.Response(200, json={"removed": True})
        return httpx.Response(404)

    matchup = Matchup(
        id="rejected",
        setup={"players": []},
        learner_player_id="learner",
        opponent_player_id="opponent",
    )
    environment = RustSessionEnvironment("http://engine.test", {matchup.id: matchup})
    environment.client.close()
    environment.client = httpx.Client(
        base_url="http://engine.test",
        transport=httpx.MockTransport(handler),
    )
    environment.reset(matchup.id, seed=7, seat_swap=False)

    with pytest.raises(RuntimeError, match="decisionKind=priority.*actionKind=castSpell"):
        environment.step(0)

    assert environment.session_id is None
    assert ("DELETE", "/game/sessions/session:2") in requests
    environment.close()


def test_rust_environment_submits_the_policy_number_as_one_engine_choice() -> None:
    submitted: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/game/sessions":
            return httpx.Response(
                200,
                json={
                    "sessionId": "session:number",
                    "revision": 1,
                    "state": {},
                    "decision": {
                        "id": "loop-iterations:1",
                        "kind": "resolutionChoice",
                        "playerId": "learner",
                        "choice": {
                            "kind": "numberSelection",
                            "decisionId": "loopIterations",
                            "minimum": 0,
                            "maximum": 2,
                        },
                        "options": [
                            {
                                "id": "choose-number:loopIterations",
                                "kind": "chooseResolution",
                            }
                        ],
                    },
                },
            )
        if request.method == "POST" and request.url.path.endswith("/actions"):
            submitted.update(json.loads(request.content))
            return httpx.Response(
                200,
                json={
                    "sessionId": "session:number",
                    "revision": 2,
                    "state": {"outcome": {"winner": "learner", "losers": []}},
                },
            )
        if request.method == "DELETE":
            return httpx.Response(200, json={"removed": True})
        return httpx.Response(404)

    matchup = Matchup(
        id="number",
        setup={"players": []},
        learner_player_id="learner",
        opponent_player_id="opponent",
    )
    environment = RustSessionEnvironment("http://engine.test", {matchup.id: matchup})
    environment.client.close()
    environment.client = httpx.Client(
        base_url="http://engine.test",
        transport=httpx.MockTransport(handler),
    )

    initial = environment.reset(matchup.id, seed=7, seat_swap=False)
    assert [action["_numberValue"] for action in initial.actions] == [0, 1, 2]
    environment.step(2)

    assert submitted == {
        "revision": 1,
        "decisionId": "loop-iterations:1",
        "actionId": "choose-number:loopIterations",
        "numberValue": 2,
    }
    environment.close()


def test_rust_environment_submits_card_name_from_the_selected_legal_action() -> None:
    submitted: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/game/sessions":
            return httpx.Response(
                200,
                json={
                    "sessionId": "session:card-name",
                    "revision": 1,
                    "state": {},
                    "decision": {
                        "id": "name:1",
                        "kind": "replacementChoice",
                        "playerId": "learner",
                        "choice": {
                            "kind": "cardNameSelection",
                            "decisionId": "chosenCardName",
                            "suggestions": ["Brainstorm", "Force of Will"],
                        },
                        "options": [
                            {
                                "id": "choose:name:0",
                                "kind": "chooseResolution",
                                "decisions": {"chosenCardName": ["Brainstorm"]},
                            },
                            {
                                "id": "choose:name:1",
                                "kind": "chooseResolution",
                                "decisions": {"chosenCardName": ["Force of Will"]},
                            },
                        ],
                    },
                },
            )
        if request.method == "POST" and request.url.path.endswith("/actions"):
            submitted.update(json.loads(request.content))
            return httpx.Response(
                200,
                json={
                    "sessionId": "session:card-name",
                    "revision": 2,
                    "state": {"outcome": {"winner": "learner", "losers": []}},
                },
            )
        if request.method == "DELETE":
            return httpx.Response(200, json={"removed": True})
        return httpx.Response(404)

    matchup = Matchup(
        id="card-name",
        setup={"players": []},
        learner_player_id="learner",
        opponent_player_id="opponent",
    )
    environment = RustSessionEnvironment("http://engine.test", {matchup.id: matchup})
    environment.client.close()
    environment.client = httpx.Client(
        base_url="http://engine.test",
        transport=httpx.MockTransport(handler),
    )

    environment.reset(matchup.id, seed=7, seat_swap=False)
    environment.step(1)

    assert submitted["actionId"] == "choose:name:1"
    assert submitted["cardName"] == "Force of Will"
    environment.close()


def test_rust_environment_preserves_a_long_configured_wait_timeout() -> None:
    matchup = Matchup(
        id="long-timeout",
        setup={"players": []},
        learner_player_id="learner",
        opponent_player_id="opponent",
    )

    environment = RustSessionEnvironment(
        "http://engine.test",
        {matchup.id: matchup},
        timeout_seconds=300,
    )

    assert environment.wait_timeout_ms == 299_000
    environment.client.close()


def test_self_play_environment_controls_every_player_and_reports_rewards() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/game/sessions":
            payload = json.loads(request.content)
            assert payload["gameMode"] == "commander"
            assert payload["humanPlayerIds"] == ["player-1", "player-2"]
            assert payload["combatDeclarationRevisionPlayerIds"] == []
            assert payload["waitTimeoutMs"] == 29_000
            assert payload["analyticsPilotByPlayerId"] == {
                "player-1": "ia-v6-in-training",
                "player-2": "ia-v6-in-training",
            }
            assert payload["analyticsContextId"] == "training:ia-v6-in-training"
            assert payload["analyticsDeckSessionByPlayerId"] == {
                "player-1": "session-a",
                "player-2": "session-b",
            }
            return httpx.Response(
                200,
                json={
                    "sessionId": "session:self-play",
                    "revision": 1,
                    "state": {},
                    "decision": {
                        "id": "decision:self-play",
                        "kind": "priority",
                        "playerId": "player-2",
                        "options": [{"id": "pass", "kind": "passPriority"}],
                    },
                },
            )
        if request.method == "POST" and request.url.path.endswith("/actions"):
            return httpx.Response(
                200,
                json={
                    "sessionId": "session:self-play",
                    "revision": 2,
                    "state": {
                        "turnNumber": 9,
                        "status": "completed",
                        "outcome": {
                            "winner": "player-2",
                            "losers": ["player-1"],
                            "reason": "lifeTotal",
                        },
                    },
                },
            )
        if request.method == "DELETE":
            return httpx.Response(200, json={"removed": True})
        return httpx.Response(404)

    matchup = Matchup(
        id="self-play",
        setup={"players": [{"id": "player-1"}, {"id": "player-2"}]},
        learner_player_id="player-1",
        opponent_player_id="player-2",
        game_mode="commander",
        deck_session_ids=("session-a", "session-b"),
    )
    environment = RustSelfPlayEnvironment(
        "http://engine.test",
        {matchup.id: matchup},
        learner_pilot_id="ia-v6-in-training",
    )
    environment.client.close()
    environment.client = httpx.Client(
        base_url="http://engine.test",
        transport=httpx.MockTransport(handler),
    )

    initial = environment.reset(matchup.id, seed=17, seat_swap=False)
    terminal = environment.step(0)

    assert initial.player_id == "player-2"
    assert initial.state["_decisionContext"]["playerId"] == "player-2"
    assert terminal.rewards_by_player == {"player-1": -1.0, "player-2": 1.0}
    environment.close()


def test_compact_replay_view_preserves_pixi_session_and_stack_shape() -> None:
    compact = RustSelfPlayEnvironment._compact_replay_view(
        {
            "schemaVersion": "mtg-game-session/v1",
            "revision": 4,
            "state": {
                "players": [],
                "stack": [
                    {
                        "id": "stack:1",
                        "controller": "player-1",
                        "card": {
                            "instanceId": "spell:1",
                            "controller": "player-1",
                            "owner": "player-1",
                            "definition": {
                                "id": "ponder",
                                "name": "Ponder",
                                "manaCost": "{U}",
                                "typeLine": "Sorcery",
                            },
                        },
                    }
                ],
            },
        }
    )

    assert compact["schemaVersion"] == "mtg-game-session/v1"
    assert compact["state"]["stack"] == [
        {
            "id": "stack:1",
            "controller": "player-1",
            "card": {
                "instanceId": "spell:1",
                "name": "Ponder",
                "cardId": "ponder",
                "controllerId": "player-1",
                "ownerId": "player-1",
                "manaCost": "{U}",
                "typeLine": "Sorcery",
            },
        }
    ]


def test_self_play_rewards_and_advantages_remain_player_relative() -> None:
    model = MagicTransformerActorCritic(
        ModelConfig(feature_dim=8, d_model=8, layers=1, heads=2, feedforward_dim=16)
    )
    learner = PPOLearner(
        model,
        HashingObservationEncoder(feature_dim=8),
        PPOConfig(gamma=1.0, gae_lambda=1.0, epochs=1),
        torch.device("cpu"),
    )
    token = torch.zeros((1, 8))
    trajectory = [
        Transition(token, token, 0, 0.0, 0.0, 0.0, False, player_id="player-1"),
        Transition(token, token, 0, 0.0, 0.0, 0.0, False, player_id="player-2"),
        Transition(token, token, 0, 0.0, 0.0, 0.0, False, player_id="player-1"),
        Transition(token, token, 0, 0.0, 0.0, 0.0, False, player_id="player-2"),
    ]

    learner._apply_terminal_self_play_rewards(
        trajectory,
        {"player-1": 1.0, "player-2": -1.0},
    )
    advantages, returns = learner._advantages(trajectory)

    assert [transition.reward for transition in trajectory] == [0.0, 0.0, 1.0, -1.0]
    assert [transition.done for transition in trajectory] == [False, False, True, True]
    assert torch.allclose(returns, torch.tensor([1.0, -1.0, 1.0, -1.0]))
    assert torch.allclose(advantages, torch.tensor([1.0, -1.0, 1.0, -1.0]))


def test_self_play_batch_collects_games_concurrently_and_updates_once() -> None:
    model = MagicTransformerActorCritic(
        ModelConfig(feature_dim=8, d_model=8, layers=1, heads=2, feedforward_dim=16)
    )
    learner = PPOLearner(
        model,
        HashingObservationEncoder(feature_dim=8),
        PPOConfig(epochs=1, minibatch_size=2),
        torch.device("cpu"),
    )
    barrier = threading.Barrier(2)

    class OneDecisionEnvironment:
        def reset(self, matchup_id: str, seed: int, seat_swap: bool) -> DecisionStep:
            barrier.wait(timeout=2.0)
            return DecisionStep(
                {"turn": 0, "seed": seed},
                [{"id": "left"}, {"id": "right"}],
                0.0,
                False,
                "player-1",
            )

        def step(self, action_index: int) -> DecisionStep:
            return DecisionStep(
                {"turn": 1, "status": "completed"},
                [],
                0.0,
                True,
                rewards_by_player={"player-1": 1.0},
            )

    jobs = [
        SelfPlayJob(OneDecisionEnvironment(), f"parallel-{index}", 100 + index)
        for index in range(2)
    ]

    results = learner.collect_self_play_batch(jobs, max_workers=2)
    combined = [transition for result in results for transition in result.trajectory]
    metrics = learner.update(combined)

    assert len(results) == 2
    assert all(result.error is None for result in results)
    assert all(result.terminal is not None for result in results)
    assert [len(result.trajectory) for result in results] == [1, 1]
    assert learner.training_step == 2
    assert math.isfinite(metrics["loss"])


def test_centered_multiplayer_rewards_do_not_punish_shared_policy_three_times() -> None:
    matchup = Matchup(
        id="four-player",
        setup={"players": [{"id": f"player-{index}"} for index in range(1, 5)]},
        learner_player_id="player-1",
        opponent_player_id="player-2",
    )
    environment = RustSelfPlayEnvironment(
        "http://engine.test",
        {matchup.id: matchup},
        multiplayer_reward_mode="centeredWinner",
    )
    environment.current_matchup = matchup

    terminal = environment._to_step(
        {
            "state": {
                "outcome": {
                    "winner": "player-1",
                    "losers": ["player-2", "player-3", "player-4"],
                }
            }
        }
    )

    assert terminal.rewards_by_player == {
        "player-1": 1.0,
        "player-2": pytest.approx(-1.0 / 3.0),
        "player-3": pytest.approx(-1.0 / 3.0),
        "player-4": pytest.approx(-1.0 / 3.0),
    }
    assert sum(terminal.rewards_by_player.values()) == pytest.approx(0.0)
    environment.close()


def test_v12_rewards_each_legacy_game_and_the_match_as_zero_sum() -> None:
    matchup = Matchup(
        id="legacy-match",
        setup={"players": [{"id": "player-1"}, {"id": "player-2"}]},
        learner_player_id="player-1",
        opponent_player_id="player-2",
        game_mode="legacy",
    )
    environment = RustSelfPlayEnvironment(
        "http://engine.test",
        {matchup.id: matchup},
        multiplayer_reward_mode="alphaStarTwoPlayer",
        legacy_game_win_reward=0.25,
        legacy_match_win_reward=1.0,
    )
    environment.current_matchup = matchup

    game_one = environment._to_step(
        {
            "decision": {
                "id": "sideboard",
                "kind": "sideboard",
                "playerId": "player-1",
                "actions": [{"id": "done", "kind": "done"}],
            },
            "state": {},
            "matchState": {
                "phase": "sideboarding",
                "winsByPlayerId": {"player-1": 1, "player-2": 0},
            },
        }
    )
    match_end = environment._to_step(
        {
            "state": {"status": "completed"},
            "matchState": {
                "phase": "complete",
                "winsByPlayerId": {"player-1": 2, "player-2": 0},
                "winnerPlayerId": "player-1",
            },
        }
    )

    assert game_one.rewards_by_player == {"player-1": 0.25, "player-2": -0.25}
    assert match_end.rewards_by_player == {"player-1": 1.25, "player-2": -1.25}
    environment.close()


def test_v12_plackett_luce_scaling_keeps_mirror_seat_rewards_distinct() -> None:
    matchup = Matchup(
        id="legacy-mirror",
        setup={"players": [{"id": "player-1"}, {"id": "player-2"}]},
        learner_player_id="player-1",
        opponent_player_id="player-2",
        game_mode="legacy",
        deck_names=("Mirror deck", "Mirror deck"),
    )
    environment = RustSelfPlayEnvironment(
        "http://engine.test",
        {matchup.id: matchup},
        multiplayer_reward_mode="alphaStarTwoPlayer",
        legacy_game_win_reward=0.25,
        legacy_match_win_reward=1.0,
        scale_rewards_by_plackett_luce=True,
    )
    environment.current_matchup = matchup
    environment.plackett_luce_participant_by_player_id = {
        "player-1": '["v12","Mirror deck"]',
        "player-2": '["v12","Mirror deck"]',
    }
    shared_rating = PlackettLuceRating(mu=37.0)
    environment.plackett_luce_ratings_by_player_id = {
        "player-1": shared_rating,
        "player-2": shared_rating,
    }

    game_one = environment._to_step(
        {
            "decision": {
                "id": "sideboard",
                "kind": "sideboard",
                "playerId": "player-1",
                "actions": [{"id": "done", "kind": "done"}],
            },
            "state": {},
            "matchState": {
                "phase": "sideboarding",
                "winsByPlayerId": {"player-1": 1, "player-2": 0},
            },
        }
    )
    match_end = environment._to_step(
        {
            "state": {"status": "completed"},
            "matchState": {
                "phase": "complete",
                "winsByPlayerId": {"player-1": 2, "player-2": 0},
                "winnerPlayerId": "player-1",
            },
        }
    )

    assert game_one.rewards_by_player == {
        "player-1": pytest.approx(0.125),
        "player-2": pytest.approx(-0.125),
    }
    assert match_end.rewards_by_player == {
        "player-1": pytest.approx(0.625),
        "player-2": pytest.approx(-0.625),
    }
    environment.close()


def test_plackett_luce_reward_compares_mixed_pilots_instead_of_duplicate_seats() -> None:
    matchup = Matchup(
        id="mixed-pilots",
        setup={"players": [{"id": f"player-{index}"} for index in range(1, 5)]},
        learner_player_id="player-1",
        opponent_player_id="player-2",
    )
    environment = RustSelfPlayEnvironment(
        "http://engine.test",
        {matchup.id: matchup},
        multiplayer_reward_mode="plackettLuce",
    )
    environment.current_matchup = matchup
    environment.participant_by_player_id = {
        "player-1": "v11",
        "player-2": "v10",
        "player-3": "v10",
        "player-4": "v10",
    }

    terminal = environment._to_step(
        {
            "state": {
                "status": "completed",
                "players": [
                    {"id": "player-1", "hasLost": False},
                    {"id": "player-2", "hasLost": True},
                    {"id": "player-3", "hasLost": True},
                    {"id": "player-4", "hasLost": True},
                ],
                "outcome": {
                    "winner": "player-1",
                    "losers": ["player-2", "player-3", "player-4"],
                },
            }
        }
    )

    assert terminal.rewards_by_player == {
        "player-1": pytest.approx(0.5),
        "player-2": pytest.approx(-0.5),
        "player-3": pytest.approx(-0.5),
        "player-4": pytest.approx(-0.5),
    }
    environment.close()


def test_plackett_luce_reward_uses_model_deck_strength_for_an_upset() -> None:
    matchup = Matchup(
        id="model-deck-upset",
        setup={"players": [{"id": "player-1"}, {"id": "player-2"}]},
        learner_player_id="player-1",
        opponent_player_id="player-2",
    )
    environment = RustSelfPlayEnvironment(
        "http://engine.test",
        {matchup.id: matchup},
        multiplayer_reward_mode="plackettLuce",
    )
    environment.current_matchup = matchup
    environment.participant_by_player_id = {
        "player-1": "v11",
        "player-2": "v11",
    }
    environment.plackett_luce_participant_by_player_id = {
        "player-1": '["v11","Weak deck"]',
        "player-2": '["v11","Strong deck"]',
    }
    environment.plackett_luce_ratings_by_player_id = {
        "player-1": PlackettLuceRating(mu=15.0),
        "player-2": PlackettLuceRating(mu=35.0),
    }

    terminal = environment._to_step(
        {
            "state": {
                "status": "completed",
                "players": [
                    {"id": "player-1", "hasLost": False},
                    {"id": "player-2", "hasLost": True},
                ],
                "outcome": {
                    "winner": "player-1",
                    "losers": ["player-2"],
                },
            }
        }
    )

    assert terminal.rewards_by_player["player-1"] > 0.9
    assert terminal.rewards_by_player["player-2"] < -0.9
    environment.close()


def test_self_play_penalizes_every_player_when_terminal_state_has_no_winner() -> None:
    matchup = Matchup(
        id="turn-limit",
        setup={"players": [{"id": "player-1"}, {"id": "player-2"}]},
        learner_player_id="player-1",
        opponent_player_id="player-2",
    )
    environment = RustSelfPlayEnvironment(
        "http://engine.test",
        {matchup.id: matchup},
        multiplayer_reward_mode="centeredWinner",
        no_winner_reward=-0.25,
    )
    environment.current_matchup = matchup

    terminal = environment._to_step(
        {
            "state": {
                "status": "turnLimitReached",
                "outcome": None,
            }
        }
    )

    assert terminal.rewards_by_player == {
        "player-1": -0.25,
        "player-2": -0.25,
    }
    environment.close()


def test_training_seed_stream_excludes_fixed_evaluation_seeds() -> None:
    scenarios = {"a": object(), "b": object()}
    first = _evaluation_seed_map(scenarios, seed=91, games_per_scenario=3)
    second = _evaluation_seed_map(scenarios, seed=91, games_per_scenario=3)
    fixed_seeds = {seed for seeds in first.values() for seed in seeds}
    training_seeds = UniqueSeedStream(91, excluded=fixed_seeds).take(100)

    assert first == second
    assert len(fixed_seeds) == 6
    assert len(set(training_seeds)) == 100
    assert fixed_seeds.isdisjoint(training_seeds)


def test_v8_step_zero_uses_a_new_pilot_without_renaming_future_champions() -> None:
    assert analytics_pilot_for_champion("ia-gt-0", "ia-v8-s0") == "ia-v8-s0"
    assert analytics_pilot_for_champion("ia-gt-1", "ia-v8-s0") == "ia-gt-1"


def test_continuous_training_has_no_episode_limit() -> None:
    assert _training_episode_limit({"continuous": True, "episodes": 1}) is None
    assert _training_episode_limit({"episodes": 12}) == 12
    with pytest.raises(ValueError, match="episodes must be positive"):
        _training_episode_limit({"episodes": 0})


def test_additional_episode_limit_is_relative_to_resumed_state() -> None:
    config = {"continuous": True}

    _set_additional_episode_limit(config, completed_episodes=284, additional_episodes=2)

    assert config["continuous"] is False
    assert config["episodes"] == 286


def test_additional_episode_limit_must_be_positive() -> None:
    with pytest.raises(ValueError, match="additional episodes must be positive"):
        _set_additional_episode_limit({}, completed_episodes=10, additional_episodes=0)


def test_resume_counter_prefers_persisted_consumed_seed_count(tmp_path: Path) -> None:
    (tmp_path / "league-state.json").write_text(
        json.dumps({"trainingSeedSkip": 161}),
        encoding="utf-8",
    )

    assert (
        _resume_counter(
            {
                "resumeLeagueState": True,
                "trainingSeedSkip": 141,
            },
            tmp_path,
            "trainingSeedSkip",
        )
        == 161
    )


def test_evaluation_summary_includes_rounds_to_win() -> None:
    games = [
        {
            "scenarioId": "a",
            "result": "candidateWin",
            "turnNumber": 12,
            "roundNumber": 4,
            "gameSeconds": 1.0,
        },
        {
            "scenarioId": "a",
            "result": "candidateWin",
            "turnNumber": 18,
            "roundNumber": 6,
            "gameSeconds": 2.0,
        },
    ]

    summary = summarize_evaluation(games, [])

    assert summary["perfect"] is True
    assert summary["candidateWinRate"] == 1.0
    assert summary["meanRoundsToCandidateWin"] == 5
    assert summary["medianRoundsToCandidateWin"] == 5
    assert summary["minRoundsToCandidateWin"] == 4
    assert summary["maxRoundsToCandidateWin"] == 6


def test_policy_service_reuses_matching_external_service(
    monkeypatch,
    tmp_path: Path,
) -> None:
    class Response:
        status_code = 200

        def __init__(self, payload: dict[str, object]) -> None:
            self.payload = payload

        def json(self) -> dict[str, object]:
            return self.payload

        def raise_for_status(self) -> None:
            return None

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    monkeypatch.setattr(
        httpx,
        "get",
        lambda url, **_: Response(
            {
                "checkpointPath": str(checkpoint.resolve()),
                "status": "ok",
                "model": "ia-in-training",
                "trainingStep": 90531,
            }
        ),
    )
    monkeypatch.setattr(
        httpx,
        "post",
        lambda url, **_: Response(
            {
                "status": "ok",
                "model": "ia-in-training",
                "trainingStep": 90532,
            }
        ),
    )
    service = PolicyService(
        port=8791,
        model_name="ia-in-training",
        log_dir=tmp_path,
        device="cpu",
        checkpoint=checkpoint,
    )

    assert service.start()["trainingStep"] == 90531
    assert service.process is None
    assert service.reload()["trainingStep"] == 90532


def test_evaluation_benchmark_opponents_are_fixed_and_periodic(tmp_path: Path) -> None:
    checkpoint = tmp_path / "v8"
    checkpoint.mkdir()
    (checkpoint / "manifest.json").write_text("{}", encoding="utf-8")

    opponents = _evaluation_benchmark_opponents(
        {
            "evaluation": {
                "championPort": 8793,
                "candidatePort": 8792,
                "benchmarkOpponents": [
                    {
                        "id": "ia-v8-step-18587",
                        "checkpoint": str(checkpoint),
                        "port": 8794,
                        "device": "cpu",
                        "everyPeriods": 2,
                    }
                ],
            }
        }
    )

    assert opponents == [
        EvaluationBenchmarkOpponent(
            id="ia-v8-step-18587",
            checkpoint=checkpoint.resolve(),
            port=8794,
            device="cpu",
            every_periods=2,
        )
    ]


def test_policy_service_rejects_an_external_service_from_another_checkpoint(
    monkeypatch,
    tmp_path: Path,
) -> None:
    class Response:
        status_code = 200

        def json(self) -> dict[str, object]:
            return {
                "checkpointPath": str((tmp_path / "old-run").resolve()),
                "model": "ia-in-training",
                "trainingStep": 2458,
            }

    checkpoint = tmp_path / "current-run"
    checkpoint.mkdir()
    monkeypatch.setattr(httpx, "get", lambda url, **_: Response())
    service = PolicyService(
        port=8791,
        model_name="ia-in-training",
        log_dir=tmp_path,
        device="cpu",
        checkpoint=checkpoint,
    )

    with pytest.raises(RuntimeError, match="different model checkpoint"):
        service.start()


def test_learning_curve_writes_csv_and_svg(tmp_path: Path) -> None:
    evaluations = [
        {
            "period": 1,
            "candidateTrainingStep": 50,
            "opponentVersion": "ia-gt-0",
            "summary": {
                "candidateWinRate": 0.5,
                "meanRoundsToCandidateWin": 7.0,
            },
            "perfectStreakAfter": 0,
            "promotionCountAfter": 0,
        }
    ]

    _write_learning_curve(evaluations, tmp_path)

    assert "candidate_win_rate" in (tmp_path / "learning-curve.csv").read_text()
    assert "ia-gt-0" in (tmp_path / "learning-curve.svg").read_text()


def test_checkpoint_retention_keeps_latest_steps(tmp_path: Path) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint_root.mkdir()
    for step in (10, 30, 20, 50, 40):
        (checkpoint_root / f"step-{step}").mkdir()
    (checkpoint_root / "notes").mkdir()

    _prune_checkpoints(checkpoint_root, keep=2)

    assert sorted(path.name for path in checkpoint_root.iterdir()) == [
        "notes",
        "step-40",
        "step-50",
    ]


def test_champion_promotes_after_three_consecutive_perfect_periods(
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "checkpoint.pt").write_text("weights")
    state = LeagueState()

    for training_step in (100, 200):
        opponent, promotion = _apply_promotion_result(
            state,
            True,
            3,
            candidate,
            tmp_path,
            training_step,
        )
        assert opponent == "ia-gt-0"
        assert promotion is None

    opponent, promotion = _apply_promotion_result(
        state,
        True,
        3,
        candidate,
        tmp_path,
        300,
    )

    assert opponent == "ia-gt-0"
    assert promotion is not None
    assert promotion["to"] == "ia-gt-1"
    assert Path(promotion["checkpoint"]).name == "ia-gt-1-step-300"
    assert state.promotion_count == 1
    assert state.perfect_evaluation_periods == 3
    assert state.perfect_streak == 0
    assert Path(state.champion_checkpoint).is_dir()


def test_league_resume_reconciles_state_with_completed_logs(tmp_path: Path) -> None:
    champion = tmp_path / "champions" / "ia-gt-0"
    champion.mkdir(parents=True)
    (tmp_path / "league-state.json").write_text(
        json.dumps(
            {
                "champion_version": 0,
                "champion_checkpoint": str(champion),
                "completed_episodes": 4,
                "attempted_episodes": 5,
                "perfect_streak": 0,
                "perfect_evaluation_periods": 0,
                "promotion_count": 0,
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "training.jsonl").write_text(
        json.dumps({"episode": 5, "attempt": 6, "trainingStep": 1700}) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "evaluations.jsonl").write_text(
        json.dumps({"period": 1, "summary": {"candidateWinRate": 0.5}}) + "\n",
        encoding="utf-8",
    )

    state, evaluations = _restore_league_state(tmp_path, champion)

    assert state.completed_episodes == 5
    assert state.attempted_episodes == 6
    assert state.champion_checkpoint == str(champion)
    assert [evaluation["period"] for evaluation in evaluations] == [1]


def test_league_resume_rebases_champion_after_run_directory_moves(
    tmp_path: Path,
) -> None:
    output = tmp_path / "moved-run"
    champion = output / "champions" / "ia-gt-0"
    champion.mkdir(parents=True)
    (output / "league-state.json").write_text(
        json.dumps(
            {
                "champion_version": 0,
                "champion_checkpoint": str(tmp_path / "old-worktree" / "champions" / "ia-gt-0"),
            }
        ),
        encoding="utf-8",
    )

    state, _ = _restore_league_state(output, tmp_path / "fallback")

    assert state.champion_checkpoint == str(champion.resolve())
