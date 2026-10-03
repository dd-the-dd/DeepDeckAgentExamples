from __future__ import annotations

import json
from types import SimpleNamespace

import httpx
import torch

from oracle_ai.encoding_v13 import GraphObservationEncoderV13
from oracle_ai.model_v13 import (
    GraphBeliefWorldModelV13,
    ModelConfigV13,
    PolicyBatchV13,
    v13_losses,
)
from oracle_ai.training import rl_v13
from oracle_ai.training.contracts import v13_training_contract
from oracle_ai.training.core import DecisionStep
from oracle_ai.training.rl_v13 import _write_json_atomic, train_rl
from oracle_ai.training.world_model import WorldModelTrainingConfig, synthetic_batch, train


def test_v13_distinguishes_engine_connection_failures_from_bad_games() -> None:
    assert rl_v13._engine_connection_failure(httpx.ConnectError("offline"))  # noqa: SLF001
    assert rl_v13._engine_connection_failure(httpx.ReadError("restarting"))  # noqa: SLF001
    assert not rl_v13._engine_connection_failure(RuntimeError("bad game"))  # noqa: SLF001


def test_v13_state_update_does_not_kill_training_on_persistent_windows_lock(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "v13-training-state.json"
    path.write_text('{"status": "running"}\n', encoding="utf-8")

    def locked_replace(_source, _target):
        raise PermissionError("simulated dashboard file lock")

    monkeypatch.setattr(type(path), "replace", locked_replace)
    monkeypatch.setattr(rl_v13.time, "sleep", lambda _seconds: None)

    assert not _write_json_atomic(path, {"status": "collecting"})
    assert json.loads(path.read_text(encoding="utf-8"))["status"] == "running"
    assert not list(tmp_path.glob("*.pending"))


def test_v13_publishes_agent_sdk_training_contract() -> None:
    payload = v13_training_contract("v13-test", "V13 Test").to_dict()

    assert [phase["id"] for phase in payload["phases"]] == [
        "world-model",
        "reinforcement-learning",
        "engine-evaluation",
    ]
    rl_metrics = {
        metric["key"] for metric in payload["phases"][1]["metrics"]
    }
    assert {"policy_loss", "value_loss", "entropy", "approx_kl"} <= rl_metrics
    evaluation_metrics = {
        metric["key"] for metric in payload["phases"][2]["metrics"]
    }
    assert {
        "win_rate",
        "lower_95",
        "curriculum_success_rate",
        "curriculum_mastery",
    } <= evaluation_metrics


def test_v13_world_model_reports_every_objective() -> None:
    config = ModelConfigV13(node_feature_dim=12, latent_dim=32, stochastic_dim=8,
                            action_dim=12, heads=4, graph_layers=1)
    training = WorldModelTrainingConfig(batch_size=3, nodes=7)
    batch = synthetic_batch(config, training, torch.device("cpu"))
    model = GraphBeliefWorldModelV13(config)

    losses = v13_losses(batch, model(batch))

    assert set(losses) == {
        "reconstruction_loss", "dynamics_loss", "kl_loss", "belief_loss",
        "opponent_loss", "search_loss", "value_loss",
    }
    assert all(loss.ndim == 0 and torch.isfinite(loss) for loss in losses.values())


def test_v13_graph_encoder_is_permutation_invariant_after_pooling() -> None:
    config = ModelConfigV13(node_feature_dim=8, latent_dim=16, stochastic_dim=4,
                            action_dim=8, heads=4, graph_layers=1)
    training = WorldModelTrainingConfig(batch_size=2, nodes=6)
    batch = synthetic_batch(config, training, torch.device("cpu"))
    model = GraphBeliefWorldModelV13(config).eval()
    permutation = torch.tensor([3, 1, 5, 0, 4, 2])
    permuted = type(batch)(
        batch.node_features[:, permutation], batch.node_types[:, permutation],
        batch.owners[:, permutation], batch.node_mask[:, permutation], batch.action_features,
        batch.next_node_features[:, permutation], batch.hidden_labels,
        batch.opponent_action, batch.search_policy, batch.outcome,
    )

    with torch.no_grad():
        original = model(batch)["pooled"]
        reordered = model(permuted)["pooled"]

    assert torch.allclose(original, reordered, atol=1e-5)


def test_v13_encoder_does_not_leak_opponent_hidden_identity() -> None:
    encoder = GraphObservationEncoderV13()
    state = {
        "players": [
            {"id": "us", "life": 20, "hand": [{"instanceId": "ours", "manaValue": 1}]},
            {"id": "them", "life": 20, "hand": [{"instanceId": "secret", "manaValue": 7}]},
        ]
    }
    changed = {
        "players": [state["players"][0], {**state["players"][1], "hand": [{"instanceId": "other", "manaValue": 1}]}]
    }

    first = encoder.encode(state, "us")
    second = encoder.encode(changed, "us")

    assert first.node_keys == second.node_keys
    assert torch.equal(first.node_features, second.node_features)


def test_v13_oracle_text_uses_masked_variable_length_tokens() -> None:
    encoder = GraphObservationEncoderV13()
    short_text = "Draw a card."
    long_text = " ".join([f"ability{index}" for index in range(180)]) + " tailmarker"
    state = {
        "players": [
            {
                "id": "us",
                "life": 20,
                "hand": [
                    {
                        "instanceId": "short",
                        "definition": {
                            "name": "Short spell",
                            "typeLine": "Instant",
                            "manaCost": "{U}",
                            "oracleText": short_text,
                        },
                    },
                    {
                        "instanceId": "long",
                        "definition": {
                            "name": "Long spell",
                            "typeLine": "Sorcery",
                            "manaCost": "{3}{U}",
                            "oracleText": long_text,
                        },
                    },
                ],
            }
        ]
    }

    graph = encoder.encode(state, "us")
    short_row = graph.node_keys.index("card:short")
    long_row = graph.node_keys.index("card:long")
    expected_tail = encoder._oracle_tokens(  # noqa: SLF001
        {"typeLine": "", "manaCost": "", "oracleText": "tailmarker"}
    )[-1]

    assert int(graph.text_mask[short_row].sum()) < int(graph.text_mask[long_row].sum())
    assert int(graph.text_mask[long_row].sum()) == encoder.max_oracle_tokens
    assert int(graph.text_tokens[long_row][graph.text_mask[long_row]][-1]) == expected_tail
    assert not graph.text_mask[graph.node_keys.index("player:us")].any()


def test_v13_structured_sequence_has_explicit_context_and_per_ability_tokens() -> None:
    encoder = GraphObservationEncoderV13()
    state = {
        "players": [
            {
                "id": "owner",
                "hand": [
                    {
                        "instanceId": "card",
                        "controllerId": "controller",
                        "definition": {
                            "oracleText": "Discard a card: Draw two cards.",
                            "rules": [
                                {"trigger": "activated", "cost": "discard", "effect": "draw"},
                                {"trigger": "when enters", "effect": "gain life"},
                            ],
                        },
                    }
                ],
            },
            {"id": "controller"},
        ]
    }

    observation = encoder.encode(state, "owner")
    card_row = observation.node_keys.index("card:card")
    ability_rows = [
        index for index, key in enumerate(observation.node_keys) if key.startswith("ability:card:card:")
    ]

    assert observation.edges.numel() == 0
    assert observation.owners[card_row].item() == 0
    assert observation.controllers[card_row].item() == 1
    assert observation.zones[card_row].item() == 2
    assert observation.positions[card_row].item() == 1
    assert len(ability_rows) == 2
    assert all(observation.text_mask[index].any() for index in ability_rows)


def test_v13_semantic_actions_ignore_volatile_engine_ids() -> None:
    encoder = GraphObservationEncoderV13()
    state = {
        "players": [
            {
                "id": "us",
                "hand": [
                    {
                        "instanceId": "spell-instance",
                        "definition": {
                            "typeLine": "Instant",
                            "manaCost": "{U}",
                            "colors": ["U"],
                            "oracleText": "Draw a card.",
                        },
                    }
                ],
            }
        ]
    }
    actions = [
        {
            "id": "cast:volatile-one",
            "kind": "castSpell",
            "cardInstanceId": "spell-instance",
            "paymentSources": ["land-instance-one"],
        },
        {
            "id": "cast:volatile-two",
            "kind": "castSpell",
            "cardInstanceId": "spell-instance",
            "paymentSources": ["land-instance-two"],
        },
    ]

    encoded = encoder.encode_actions(actions, state, "us")

    assert torch.equal(encoded[0], encoded[1])
    assert encoded[0, 3] == 1
    assert encoded[0, 11] == 1


def test_v13_text_padding_is_ignored_by_the_model() -> None:
    config = ModelConfigV13(
        node_feature_dim=8,
        latent_dim=16,
        stochastic_dim=4,
        action_dim=8,
        heads=4,
        graph_layers=1,
        text_max_tokens=8,
    )
    model = GraphBeliefWorldModelV13(config).eval()
    features = torch.zeros(1, 1, 8)
    node_types = torch.full((1, 1), 2, dtype=torch.long)
    owners = torch.zeros(1, 1, dtype=torch.long)
    node_mask = torch.ones(1, 1, dtype=torch.bool)

    with torch.no_grad():
        short = model.encode_graph(
            features,
            node_types,
            owners,
            node_mask,
            text_tokens=torch.tensor([[[12, 34]]]),
            text_mask=torch.tensor([[[True, True]]]),
        )
        padded = model.encode_graph(
            features,
            node_types,
            owners,
            node_mask,
            text_tokens=torch.tensor([[[12, 34, 777, 888]]]),
            text_mask=torch.tensor([[[True, True, False, False]]]),
        )

    assert torch.allclose(short, padded, atol=1e-6)


def test_v13_training_resumes_latest_checkpoint(tmp_path) -> None:
    config = tmp_path / "v13.yaml"
    config.write_text(
        """
model:
  node_feature_dim: 8
  latent_dim: 16
  heads: 4
  graph_layers: 1
  stochastic_dim: 4
  action_dim: 8
training:
  steps: 1
  batch_size: 2
  nodes: 4
  checkpoint_every: 1
  device: cpu
  resume: true
""".strip(),
        encoding="utf-8",
    )
    output = tmp_path / "run"
    train(config, output)
    contents = config.read_text(encoding="utf-8").replace("steps: 1", "steps: 2")
    config.write_text(contents, encoding="utf-8")

    train(config, output)

    records = [json.loads(line) for line in (output / "v13-metrics.jsonl").read_text().splitlines()]
    assert [record["step"] for record in records] == [1, 2]
    assert (output / "checkpoints" / "step-2" / "world-model.pt").is_file()


def test_v13_rl_skips_an_incomplete_latest_checkpoint(tmp_path) -> None:
    config = ModelConfigV13(
        node_feature_dim=8,
        latent_dim=16,
        heads=4,
        graph_layers=1,
        stochastic_dim=4,
        action_dim=8,
    )
    model = GraphBeliefWorldModelV13(config)
    valid = tmp_path / "rl-checkpoints" / "update-4" / "rl-model.pt"
    valid.parent.mkdir(parents=True)
    torch.save(
        {
            "model": model.state_dict(),
            "model_config": vars(config),
            "observation_schema": model.observation_schema,
            "completed_episodes": 12,
            "update": 4,
        },
        valid,
    )
    incomplete = tmp_path / "rl-checkpoints" / "update-5" / "rl-model.pt"
    incomplete.parent.mkdir(parents=True)
    incomplete.write_bytes(b"interrupted checkpoint")

    _, completed_episodes, update, payload = rl_v13._load_model(  # noqa: SLF001
        tmp_path, config, torch.device("cpu")
    )

    assert completed_episodes == 12
    assert update == 4
    assert payload is not None


def test_v13_policy_masks_padded_legal_actions() -> None:
    config = ModelConfigV13(
        node_feature_dim=8, latent_dim=16, stochastic_dim=4,
        action_dim=8, heads=4, graph_layers=1,
    )
    model = GraphBeliefWorldModelV13(config)
    batch = PolicyBatchV13(
        node_features=torch.randn(2, 3, 8),
        node_types=torch.zeros(2, 3, dtype=torch.long),
        owners=torch.zeros(2, 3, dtype=torch.long),
        node_mask=torch.ones(2, 3, dtype=torch.bool),
        legal_action_features=torch.randn(2, 4, 8),
        legal_action_mask=torch.tensor([[True, True, False, False], [True] * 4]),
    )

    logits, values = model.policy_value(batch)

    assert logits.shape == (2, 4)
    assert values.shape == (2,)
    assert logits[0, 2] < -1e20


def test_v13_rl_consumes_world_checkpoint_and_reports_ppo_metrics(tmp_path) -> None:
    config = tmp_path / "v13.yaml"
    config.write_text(
        """
model:
  node_feature_dim: 8
  latent_dim: 16
  heads: 4
  graph_layers: 1
  stochastic_dim: 4
  action_dim: 8
training:
  steps: 1
  batch_size: 2
  nodes: 4
  checkpoint_every: 1
  device: cpu
rl:
  episodes: 8
  rollout_episodes: 4
  horizon: 2
  epochs: 1
  minibatch_size: 8
  checkpoint_every: 1
  device: cpu
""".strip(),
        encoding="utf-8",
    )
    output = tmp_path / "run"
    train(config, output)

    train_rl(config, output)

    records = [
        json.loads(line)
        for line in (output / "v13-rl-metrics.jsonl").read_text().splitlines()
    ]
    state = json.loads((output / "v13-training-state.json").read_text())
    assert len(records) == 2
    assert records[-1]["data_source"] == "tiny-self-play-v1"
    assert records[-1]["rolloutBatch"]["completedGames"] == 4
    assert {"policy_loss", "value_loss", "entropy", "approx_kl"} <= records[-1].keys()
    assert state["trainingPhase"] == "reinforcement-learning"
    assert state["completed_episodes"] == 8
    assert (output / "rl-checkpoints" / "update-2" / "rl-model.pt").is_file()


def test_v13_rl_retries_a_failed_environment_game(tmp_path, monkeypatch) -> None:
    config = tmp_path / "v13-retry.yaml"
    config.write_text(
        """
model:
  node_feature_dim: 8
  latent_dim: 16
  heads: 4
  graph_layers: 1
  stochastic_dim: 4
  action_dim: 8
training:
  steps: 1
  batch_size: 2
  nodes: 4
  checkpoint_every: 1
  device: cpu
rl:
  episodes: 2
  rollout_episodes: 2
  horizon: 2
  epochs: 1
  minibatch_size: 8
  checkpoint_every: 1
  max_consecutive_errors: 2
  device: cpu
""".strip(),
        encoding="utf-8",
    )
    output = tmp_path / "run"
    train(config, output)
    original_environment = rl_v13.TinySelfPlayEnvironment

    class FlakyEnvironment(original_environment):
        failed = False

        def step(self, action_index):
            if not self.failed:
                self.failed = True
                raise RuntimeError("temporary Engine-style failure")
            return super().step(action_index)

    monkeypatch.setattr(rl_v13, "TinySelfPlayEnvironment", FlakyEnvironment)

    train_rl(config, output)

    record = json.loads((output / "v13-rl-metrics.jsonl").read_text().splitlines()[-1])
    assert record["rolloutBatch"] == {
        "requestedGames": 3,
        "completedGames": 2,
        "failedGames": 1,
        "totalDecisions": 4,
        "collectionWallSeconds": record["rolloutBatch"]["collectionWallSeconds"],
        "trainingWallSeconds": record["rolloutBatch"]["trainingWallSeconds"],
    }


def test_v13_rl_waits_for_engine_reconnect_without_exhausting_game_errors(
    tmp_path, monkeypatch
) -> None:
    config = tmp_path / "v13-reconnect.yaml"
    config.write_text(
        """
model:
  node_feature_dim: 8
  latent_dim: 16
  heads: 4
  graph_layers: 1
  stochastic_dim: 4
  action_dim: 8
training:
  steps: 1
  batch_size: 2
  nodes: 4
  checkpoint_every: 1
  device: cpu
rl:
  episodes: 2
  rollout_episodes: 2
  horizon: 2
  epochs: 1
  minibatch_size: 8
  checkpoint_every: 1
  max_consecutive_errors: 2
  device: cpu
""".strip(),
        encoding="utf-8",
    )
    output = tmp_path / "run"
    train(config, output)
    original_environment = rl_v13.TinySelfPlayEnvironment

    class ReconnectingEnvironment(original_environment):
        connection_failures = 0

        def reset(self, matchup_id, seed, seat_swap):
            if self.connection_failures < 3:
                self.connection_failures += 1
                raise httpx.ConnectError("Engine restarting")
            return super().reset(matchup_id, seed, seat_swap)

    delays: list[float] = []
    monkeypatch.setattr(rl_v13, "TinySelfPlayEnvironment", ReconnectingEnvironment)
    monkeypatch.setattr(rl_v13.time, "sleep", delays.append)

    train_rl(config, output)

    record = json.loads((output / "v13-rl-metrics.jsonl").read_text().splitlines()[-1])
    assert record["rolloutBatch"]["completedGames"] == 2
    assert record["rolloutBatch"]["failedGames"] == 3
    assert record["rolloutBatch"]["requestedGames"] == 5
    assert delays == [0.25, 0.5, 1.0]


def test_v13_engine_evaluation_records_failures_without_stopping_training() -> None:
    class Model:
        def eval(self):
            return self

        def train(self):
            return self

    class Sampler:
        def sample(self, _randomizer):
            return SimpleNamespace(id="match", deck_names=("Deck A", "Deck B"))

    class FailingEnvironment:
        def __init__(self):
            self.matchups = {}

        def reset(self, _matchup_id, _seed, _seat_swap):
            raise RuntimeError("temporary Engine failure")

    evaluation = rl_v13._run_engine_evaluation(  # noqa: SLF001
        model=Model(),
        baseline=Model(),
        encoder=SimpleNamespace(),
        environment=FailingEnvironment(),
        sampler=Sampler(),
        device=torch.device("cpu"),
        seed=123,
        games=2,
        training_step=9,
        completed_episodes=12,
    )

    assert evaluation["summary"]["completedGames"] == 0
    assert evaluation["summary"]["failedGames"] == 5
    assert len(evaluation["failures"]) == 5
    assert evaluation["opponentVersion"] == "v13-engine-baseline"


def test_v13_curriculum_evaluation_records_each_failed_skill() -> None:
    class Model:
        def eval(self):
            return self

        def train(self):
            return self

    scenario = SimpleNamespace(id="combo-test", revision=2, family="known-combo")

    class Sampler:
        scenarios = [scenario]

        def build_scenario(self, scenario_id, _seed):
            assert scenario_id == scenario.id
            return SimpleNamespace(id="match", learner_player_id="player-1")

    class FailingEnvironment:
        def __init__(self):
            self.matchups = {}

        def reset(self, _matchup_id, _seed, _seat_swap):
            raise RuntimeError("temporary curriculum failure")

    evaluation = rl_v13._run_curriculum_evaluation(  # noqa: SLF001
        model=Model(),
        encoder=SimpleNamespace(),
        environment=FailingEnvironment(),
        sampler=Sampler(),
        device=torch.device("cpu"),
        seed=123,
        training_step=9,
        completed_episodes=12,
    )

    assert evaluation["summary"]["completedScenarios"] == 0
    assert evaluation["summary"]["failedScenarios"] == 1
    assert evaluation["summary"]["engineErrors"] == 3
    assert evaluation["failures"][0]["attempts"] == 3
    assert evaluation["failures"][0]["scenarioId"] == "combo-test"


def test_v13_curriculum_evaluation_recovers_transient_engine_failure() -> None:
    class Model:
        def eval(self):
            return self

        def train(self):
            return self

    scenario = SimpleNamespace(id="fast-test", revision=3, family="opening-hand")

    class Sampler:
        scenarios = [scenario]

        def build_scenario(self, scenario_id, _seed):
            assert scenario_id == scenario.id
            return SimpleNamespace(
                id="match",
                learner_player_id="player-1",
                success_action_sequence=(),
                opening_hand_target_roles=(),
                objective="fast-win",
                training_anchor_player_ids=("player-2",),
            )

    class RecoveringEnvironment:
        def __init__(self):
            self.matchups = {}
            self.resets = 0
            self.current_view = {"state": {"turnNumber": 3}}
            self.opening_hand_roles_available = 0
            self.objective_milestones_completed = 0
            self.objective_damage_progress = 0.4
            self.sideboard_target_cards_selected = 0
            self.sideboard_target_cards_available = 0
            self.sideboard_cut_cards_selected = 0
            self.sideboard_cut_cards_expected = 0
            self.sideboard_cards_selected = 0

        def reset(self, _matchup_id, _seed, _seat_swap):
            self.resets += 1
            if self.resets == 1:
                raise RuntimeError("temporary curriculum failure")
            return DecisionStep({}, [], -1.0, True, "player-1")

    environment = RecoveringEnvironment()
    evaluation = rl_v13._run_curriculum_evaluation(  # noqa: SLF001
        model=Model(),
        encoder=SimpleNamespace(),
        environment=environment,
        sampler=Sampler(),
        device=torch.device("cpu"),
        seed=123,
        training_step=9,
        completed_episodes=12,
    )

    assert evaluation["summary"]["completedScenarios"] == 1
    assert evaluation["summary"]["failedScenarios"] == 0
    assert evaluation["summary"]["recoveredScenarios"] == 1
    assert evaluation["summary"]["engineErrors"] == 1
    assert evaluation["scenarios"][0]["attempt"] == 2
    assert evaluation["scenarios"][0]["mastery"] == 0.4
