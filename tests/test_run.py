from __future__ import annotations

import asyncio
from argparse import Namespace

import pytest
from deepdeck_agent import AgentConfig

from deepdeck_examples.run import (
    _league_match_snapshot,
    _matchmaking_entries,
    _serve_local,
    parser,
)


def test_same_demo_command_exposes_local_and_public_targets() -> None:
    local = parser().parse_args(["alexios", "--target", "local"])
    public = parser().parse_args(["alexios", "--target", "ddl"])
    assert local.target == "local"
    assert public.target == "ddl"


def test_v11_and_v12_are_available_but_untrained_weights_require_opt_in() -> None:
    v11 = parser().parse_args(["v11", "--target", "local", "--allow-untrained"])
    v12 = parser().parse_args(["v12", "--target", "ddl", "--checkpoint", "checkpoint"])
    assert v11.allow_untrained is True
    assert v12.checkpoint == "checkpoint"


def test_v13_is_available_with_an_rl_checkpoint() -> None:
    v13 = parser().parse_args(["v13", "--target", "local", "--checkpoint", "checkpoint"])

    assert v13.example == "v13"
    assert v13.checkpoint == "checkpoint"


def test_forge_reference_profiles_are_available() -> None:
    for profile in ("cautious", "balanced", "aggressive"):
        arguments = parser().parse_args([f"forge-{profile}", "--target", "local"])
        assert arguments.example == f"forge-{profile}"


def test_public_runner_accepts_a_pool_for_concurrent_league_seats() -> None:
    arguments = parser().parse_args(
        [
            "v12",
            "--target",
            "ddl",
            "--deck-version-id",
            "deck-a",
            "--additional-deck-version-id",
            "deck-b",
            "--matchmaking-concurrency",
            "2",
        ]
    )

    assert arguments.deck_version_id == "deck-a"
    assert arguments.additional_deck_version_id == ["deck-b"]
    assert arguments.matchmaking_concurrency == 2

    entries = _matchmaking_entries(arguments)
    assert [entry.deck_version_ids for entry in entries] == [
        ("deck-a", "deck-b"),
        ("deck-a", "deck-b"),
    ]
    assert [entry.client_seat_id for entry in entries] == [
        "learner:seat-1",
        "learner:seat-2",
    ]


def test_league_match_snapshot_exposes_live_game_telemetry() -> None:
    snapshot = _league_match_snapshot(
        "match-1",
        {
            "games": [
                {
                    "id": "game-1",
                    "number": 2,
                    "status": "running",
                    "turnCount": 7,
                    "startedAt": "2026-09-03T04:56:32+00:00",
                }
            ],
            "summary": {
                "status": "running",
                "participants": [{"deck": "Deck A"}, {"deck": "Deck B"}],
            },
        },
        "https://staging.deepdeckleague.com/api/v1",
    )

    assert snapshot["gameId"] == "game-1"
    assert snapshot["decks"] == ["Deck A", "Deck B"]
    assert snapshot["players"] == 2
    assert snapshot["turnNumber"] == 7
    assert snapshot["roundNumber"] == 2
    assert snapshot["startedAtUnixMs"] == 1_788_411_392_000
    assert snapshot["watchUrl"] == (
        "https://staging.deepdeckleague.com/matches?tab=games&match=match-1&replay=1&seat=0"
    )


@pytest.mark.asyncio
async def test_local_target_can_create_a_game_after_agent_registration() -> None:
    class FakeRunner:
        def __init__(self) -> None:
            self.config = AgentConfig(
                agent_id="example",
                name="Example",
                version="1",
                author="Tester",
                formats=("commander",),
            )
            self.started = asyncio.Event()
            self.game = None

        async def serve(self) -> None:
            await asyncio.Event().wait()

        async def wait_until_connected(self, *, timeout: float) -> str:
            assert timeout == 30
            return "agent:example"

        async def start_local_game(self, game):
            self.game = game
            self.started.set()
            return {"session": {"id": "local-game"}}

    runner = FakeRunner()
    arguments = Namespace(
        start_local_game=True,
        local_deck_session_id="our-deck",
        local_opponent_deck_session_id="their-deck",
        local_opponent_controller="ai-random",
        local_format="commander",
        local_max_turns=80,
        seed=7,
    )
    task = asyncio.create_task(_serve_local(runner, arguments))  # type: ignore[arg-type]
    await asyncio.wait_for(runner.started.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert runner.game is not None
    payload = runner.game.payload()
    assert payload["gameMode"] == "commander"
    assert payload["aiControllerByPlayerId"] == {
        "example-agent": "agent:example",
        "local-opponent": "ai-random",
    }
