from __future__ import annotations

from pathlib import Path

import pytest
import torch
from deepdeck_agent import ComputeMode, Decision, Event, Game

from deepdeck_examples.v15_agent import load_v15_agent
from oracle_ai.model_v15 import ModelConfigV15, TypedLatentWorldModelV15


def _checkpoint(path: Path) -> Path:
    config = ModelConfigV15(
        node_feature_dim=16,
        latent_dim=24,
        tactical_dim=16,
        strategic_dim=12,
        heads=4,
        graph_layers=1,
        text_max_tokens=16,
    )
    model = TypedLatentWorldModelV15(config)
    path.mkdir(parents=True)
    torch.save(
        {
            "model": model.state_dict(),
            "model_config": vars(config),
            "observation_schema": model.observation_schema,
        },
        path / "v15-model.pt",
    )
    return path


def _game(*, stack: list[dict] | None = None, step: str = "precombatMain") -> Game:
    return Game(
        {
            "sessionId": "game:v15",
            "turnNumber": 3,
            "step": step,
            "players": [
                {"id": "us", "life": 20, "hand": [], "battlefield": []},
                {"id": "them", "life": 20, "hand": [], "battlefield": []},
            ],
            "stack": stack or [],
        },
        "us",
    )


@pytest.mark.asyncio
async def test_v15_serves_only_an_engine_legal_action_and_records_plan(tmp_path: Path) -> None:
    agent = load_v15_agent(_checkpoint(tmp_path / "checkpoint"))
    decision = Decision(
        "decision-1",
        "us",
        {
            "kind": "priority",
            "options": [
                {"id": "play-land", "kind": "playLand", "cardInstanceId": "land"},
                {"id": "pass", "kind": "passPriority"},
            ],
        },
        _game(),
    )

    response = await agent.make_decision(decision)

    assert response.action_id in {"play-land", "pass"}
    assert agent.last_inference["planId"].startswith("v15:")
    assert agent.last_inference["computeMode"] in {"tactical", "strategic"}


@pytest.mark.asyncio
async def test_v15_advances_plan_when_a_counterspell_event_arrives(tmp_path: Path) -> None:
    agent = load_v15_agent(_checkpoint(tmp_path / "checkpoint"))
    game = _game(stack=[{"id": "our-spell"}, {"id": "force"}])
    await agent.on_game_start(game, [])
    await agent.on_event(
        Event(4, "spellCountered", 3, "precombatMain", "them", "force", {}, {}), game
    )
    decision = Decision(
        "decision-2",
        "us",
        {"kind": "priority", "options": [{"id": "pass", "kind": "passPriority"}]},
        game,
    )

    response = await agent.make_decision(decision)

    assert response.action_id == "pass"
    assert agent.last_inference["nodeId"] == "respond"
    assert agent.last_inference["computeMode"] == ComputeMode.REFLEX.value


def test_v15_allocates_strategic_compute_to_complex_combat(tmp_path: Path) -> None:
    agent = load_v15_agent(_checkpoint(tmp_path / "checkpoint"))
    actions = [
        {"id": f"attack-{index}", "kind": "declareAttacker", "attackerId": str(index)}
        for index in range(6)
    ]
    decision = Decision(
        "decision-3",
        "us",
        {"kind": "attackers", "options": actions},
        _game(step="declareAttackers"),
    )

    budget = agent.compute_budget(decision)

    assert budget.mode is ComputeMode.STRATEGIC
    assert budget.budget_ms >= 1_500


def test_v15_preserves_clock_when_time_is_low(tmp_path: Path) -> None:
    agent = load_v15_agent(_checkpoint(tmp_path / "checkpoint"))
    decision = Decision(
        "decision-4",
        "us",
        {
            "kind": "attackers",
            "clock": {"remainingMs": 12_000, "incrementMs": 1_000},
            "options": [
                {"id": f"attack-{index}", "kind": "declareAttacker"}
                for index in range(6)
            ],
        },
        _game(step="declareAttackers"),
    )

    budget = agent.compute_budget(decision)

    assert budget.mode is ComputeMode.TACTICAL
    assert budget.remaining_ms == 12_000
