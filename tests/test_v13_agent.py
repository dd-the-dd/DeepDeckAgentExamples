from __future__ import annotations

from pathlib import Path

import pytest
import torch
from deepdeck_agent import Decision, Game

from deepdeck_examples.v13_agent import load_v13_agent
from oracle_ai.model_v13 import GraphBeliefWorldModelV13, ModelConfigV13


def _checkpoint(path: Path) -> Path:
    config = ModelConfigV13(latent_dim=16, heads=4, graph_layers=1, stochastic_dim=4)
    model = GraphBeliefWorldModelV13(config)
    path.mkdir(parents=True)
    torch.save(
        {
            "model": model.state_dict(),
            "model_config": vars(config),
            "observation_schema": model.observation_schema,
        },
        path / "rl-model.pt",
    )
    return path


@pytest.mark.asyncio
async def test_v13_checkpoint_serves_a_legal_sdk_action(tmp_path: Path) -> None:
    agent = load_v13_agent(_checkpoint(tmp_path / "update-1"))
    game = Game(
        {
            "turn": 1,
            "players": [
                {"id": "player-1", "life": 20},
                {"id": "player-2", "life": 20},
            ],
        },
        "player-2",
    )
    decision = Decision(
        "decision-1",
        "player-2",
        {"kind": "priority", "options": [{"id": "pass", "kind": "passPriority"}]},
        game,
    )

    response = await agent.make_decision(decision)

    assert response.action_id == "pass"


@pytest.mark.asyncio
async def test_v13_sdk_adapter_submits_the_selected_number(tmp_path: Path) -> None:
    agent = load_v13_agent(_checkpoint(tmp_path / "update-2"))
    game = Game({"players": [{"id": "player-1", "life": 20}]}, "player-1")
    decision = Decision(
        "decision-2",
        "player-1",
        {
            "kind": "resolutionChoice",
            "options": [{"id": "choose-x", "kind": "chooseNumber"}],
            "choice": {"kind": "numberSelection", "minimum": 2, "maximum": 2},
        },
        game,
    )

    response = await agent.make_decision(decision)

    assert response.action_id == "choose-x"
    assert response.number_value == 2


@pytest.mark.asyncio
async def test_v13_uses_an_active_legal_fallback_when_policy_only_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent = load_v13_agent(_checkpoint(tmp_path / "update-3"))
    land = {
        "instanceId": "land",
        "owner": "player-1",
        "controller": "player-1",
        "definition": {
            "id": "island",
            "name": "Island",
            "typeLine": "Basic Land â€” Island",
            "manaCost": "",
            "rules": [],
        },
        "tapped": False,
        "counters": {},
    }
    game = Game(
        {
            "turnNumber": 1,
            "activePlayer": 0,
            "step": "precombatMain",
            "players": [
                {
                    "id": "player-1",
                    "life": 20,
                    "hand": [land],
                    "battlefield": [],
                    "library": [],
                    "graveyard": [],
                    "exile": [],
                    "sideboard": [],
                    "commandZone": [],
                    "manaPool": [],
                },
                {
                    "id": "player-2",
                    "life": 20,
                    "hand": [],
                    "battlefield": [],
                    "library": [],
                    "graveyard": [],
                    "exile": [],
                    "sideboard": [],
                    "commandZone": [],
                    "manaPool": [],
                },
            ],
            "stack": [],
        },
        "player-1",
    )
    decision = Decision(
        "decision-3",
        "player-1",
        {
            "kind": "priority",
            "options": [
                {"id": "play-land", "kind": "playLand", "cardInstanceId": "land"},
                {"id": "pass", "kind": "passPriority"},
            ],
        },
        game,
    )
    monkeypatch.setattr(agent, "_select_model_action", lambda _decision, actions: actions[1])

    response = await agent.make_decision(decision)

    assert response.action_id == "play-land"


@pytest.mark.asyncio
async def test_v13_does_not_replace_pass_with_a_standalone_mana_action(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent = load_v13_agent(_checkpoint(tmp_path / "update-4"))
    land = {
        "instanceId": "land",
        "owner": "player-1",
        "controller": "player-1",
        "definition": {
            "id": "island",
            "name": "Island",
            "typeLine": "Basic Land â€” Island",
            "manaCost": "",
            "rules": [],
        },
        "tapped": False,
        "counters": {},
    }
    game = Game(
        {
            "turnNumber": 1,
            "activePlayer": 0,
            "step": "precombatMain",
            "players": [
                {"id": "player-1", "life": 20, "hand": [], "battlefield": [land]},
                {"id": "player-2", "life": 20, "hand": [], "battlefield": []},
            ],
            "stack": [],
        },
        "player-1",
    )
    decision = Decision(
        "decision-4",
        "player-1",
        {
            "kind": "priority",
            "options": [
                {
                    "id": "tap-land",
                    "kind": "activateAbility",
                    "cardInstanceId": "land",
                    "decisions": {"manaAbility": True},
                },
                {"id": "pass", "kind": "passPriority"},
            ],
        },
        game,
    )
    monkeypatch.setattr(agent, "_select_model_action", lambda _decision, actions: actions[1])

    response = await agent.make_decision(decision)

    assert response.action_id == "pass"
