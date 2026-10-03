from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from oracle_ai.agents.protocol import (
    AgentAuthor,
    AgentCompatibility,
    AgentManifest,
    DeckSelection,
    GameEndedRequest,
    GameSharing,
    ObservationStream,
    TimeoutCategory,
)
from oracle_ai.agents.state import ObservationReplica, apply_merge_patch


def manifest() -> AgentManifest:
    return AgentManifest(
        agent_id="org.example.agent",
        name="Example agent",
        version="1.0.0",
        authors=[AgentAuthor(name="Example Author")],
        compatibility=AgentCompatibility(
            game_modes=["legacy"],
            decks=DeckSelection(selection="all"),
            time_controls=[TimeoutCategory.STANDARD],
            observation_streams=[ObservationStream.FULL, ObservationStream.DELTA],
            game_sharing=[GameSharing.PRIVATE],
        ),
    )


def test_manifest_is_typed_and_serializes_the_rust_contract() -> None:
    payload = manifest().model_dump(by_alias=True, mode="json")
    assert payload["schemaVersion"] == "agent-manifest/v1"
    assert payload["compatibility"]["decks"] == {"selection": "all", "deckIds": []}
    assert payload["compatibility"]["observationStreams"] == [
        "full-observation-stream",
        "delta-event-stream",
    ]


def test_maximal_manifest_matches_the_shared_agent_contract_fixture() -> None:
    fixture = Path(__file__).parents[1] / "fixtures" / "agent_manifest_v1_maximal.json"
    payload = json.loads(fixture.read_text(encoding="utf-8"))

    serialized = AgentManifest.model_validate(payload).model_dump(
        by_alias=True,
        mode="json",
        exclude_none=True,
    )

    assert serialized == payload


def test_allow_list_requires_a_deck() -> None:
    with pytest.raises(ValidationError):
        DeckSelection(selection="allow-list")


def test_delta_replica_rejects_a_sequence_gap() -> None:
    replica = ObservationReplica()
    replica.replace(4, {"life": 20, "nested": {"old": True}})
    assert replica.apply(5, 4, {"life": 18, "nested": {"old": None}}) == {
        "life": 18,
        "nested": {},
    }
    with pytest.raises(ValueError, match="expected sequence 5"):
        replica.apply(7, 6, {"life": 10})


def test_merge_patch_does_not_mutate_the_source() -> None:
    source = {"cards": [1], "stats": {"life": 20}}
    result = apply_merge_patch(source, {"cards": [1, 2], "stats": {"life": 19}})
    assert result == {"cards": [1, 2], "stats": {"life": 19}}
    assert source == {"cards": [1], "stats": {"life": 20}}


@pytest.mark.asyncio
async def test_agent_client_releases_finished_game_observations() -> None:
    from oracle_ai.agents.base import MagicAgent
    from oracle_ai.agents.client import AgentClient

    client = AgentClient(
        "ws://localhost",
        manifest(),
        MagicAgent(),
        observation_stream=ObservationStream.FULL,
        timeout_category=TimeoutCategory.STANDARD,
    )
    client._replicas["game-1:player-1"] = ObservationReplica()

    await client._dispatch(
        {
            "type": "gameEnded",
            "contextId": "game-1:player-1",
            "outcome": {"winner": "player-1"},
        }
    )

    assert client._replicas == {}


@pytest.mark.asyncio
async def test_v11_v12_adapter_expands_numeric_decisions() -> None:
    from oracle_ai.agents.model_agent import OracleModelAgent
    from oracle_ai.agents.protocol import DecisionRequest, StartingSituationRequest

    class Model:
        model_family = "structured-v12"

    class Runtime:
        model = Model()

        def choose_with_model(self, state, actions, deterministic, request_id, **kwargs):
            assert state["_decisionContext"]["playerId"] == "p1"
            assert state["_pregameDeck"][0]["id"] == "known-card"
            assert [action["_numberValue"] for action in actions] == [1, 2, 3]
            return 2, None, "test-v12"

    adapter = OracleModelAgent(Runtime())
    await adapter.analyze_starting_situation(
        StartingSituationRequest.model_validate(
            {
                "type": "startingSituationRequested",
                "schemaVersion": "mtg-agent/v1",
                "requestId": "start",
                "contextId": "game:p1",
                "deadlineUnixMs": 1,
                "targetDurationMs": 5,
                "analysisDurationMs": 25,
                "observation": {"players": [{"id": "p1", "library": []}]},
                "knownDeck": [{"id": "known-card", "name": "Known Card", "typeLine": "Land"}],
            }
        )
    )
    response = await adapter.make_decision(
        DecisionRequest.model_validate(
            {
                "type": "decisionRequested",
                "schemaVersion": "mtg-agent/v1",
                "observationSchemaVersion": "player-observation/v1",
                "requestId": "d1",
                "contextId": "game:p1",
                "playerId": "p1",
                "deadlineUnixMs": 1,
                "observationUpdate": {
                    "kind": "fullObservation",
                    "sequence": 1,
                    "observation": {"players": [{"id": "p1"}]},
                },
                "decision": {
                    "id": "d1",
                    "kind": "resolutionChoice",
                    "playerId": "p1",
                    "choice": {
                        "kind": "numberSelection",
                        "decisionId": "amount",
                        "minimum": 1,
                        "maximum": 3,
                    },
                    "options": [
                        {
                            "id": "choose",
                            "kind": "chooseResolution",
                            "playerId": "p1",
                            "label": "Choose",
                        }
                    ],
                },
                "observation": {"players": [{"id": "p1"}]},
            }
        )
    )
    assert response.action_id == "choose"
    assert response.number_value == 3
    await adapter.game_ended(
        GameEndedRequest.model_validate(
            {
                "type": "gameEnded",
                "contextId": "game:p1",
                "outcome": {"winner": "p1"},
            }
        )
    )
    assert adapter._pregame_by_context == {}
    assert adapter._known_deck_by_context == {}
