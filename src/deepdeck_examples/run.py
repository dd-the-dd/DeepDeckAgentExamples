from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
from deepdeck_agent import (
    Agent,
    AgentConfig,
    AgentRunner,
    LocalGame,
    LocalPlayer,
    MatchmakingEntry,
    PlaySpeed,
    ServerTarget,
)
from dotenv import find_dotenv, load_dotenv

from .alexios import AlexiosAgent
from .configuration import (
    alexios_config,
    deep_learning_config,
    forge_reference_config,
    random_config,
)
from .forge_reference import ForgeReferenceProfile, build_forge_reference_agent
from .oracle_checkpoint_agent import OracleCheckpointAgent, is_oracle_checkpoint
from .random_baseline import build_random_agent


def _speed(value: str) -> PlaySpeed:
    try:
        return PlaySpeed(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("speed must be 100ms, 1s, or 10s") from error


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Run a public Deep Deck example agent.")
    result.add_argument(
        "example",
        nargs="?",
        choices=(
            "random",
            "alexios",
            "forge-cautious",
            "forge-balanced",
            "forge-aggressive",
            "v11",
            "v12",
            "v13",
            "v15",
        ),
        default=os.getenv("DEEPDECK_EXAMPLE", "alexios"),
    )
    result.add_argument(
        "--target",
        choices=("local", "ddl"),
        default=os.getenv("DEEPDECK_TARGET", "local"),
    )
    result.add_argument(
        "--speed",
        type=_speed,
        default=_speed(os.getenv("DEEPDECK_SPEED", "1s")),
    )
    result.add_argument(
        "--engine-url",
        default=os.getenv("MTG_ENGINE_URL", "http://127.0.0.1:8787"),
    )
    result.add_argument("--seed", type=int, default=1)
    result.add_argument(
        "--checkpoint",
        default=os.getenv("DEEPDECK_CHECKPOINT"),
        help=(
            "V11/V12 checkpoint directory, V13 directory containing rl-model.pt, "
            "or V15 directory containing v15-model.pt."
        ),
    )
    result.add_argument(
        "--device",
        default=os.getenv("DEEPDECK_DEVICE", "cpu"),
        help="PyTorch device used by V11/V12/V13/V15, for example cpu or cuda.",
    )
    result.add_argument(
        "--allow-untrained",
        action="store_true",
        help="Allow random V11/V12 weights for protocol and connectivity testing only.",
    )
    result.add_argument(
        "--competition-version-id",
        default=os.getenv("DEEPDECK_COMPETITION_VERSION_ID"),
    )
    result.add_argument(
        "--deck-version-id",
        default=os.getenv("DEEPDECK_DECK_VERSION_ID"),
    )
    result.add_argument(
        "--additional-deck-version-id",
        action="append",
        default=[],
        help="Additional deck the agent is ready to play; the League selects per match.",
    )
    result.add_argument(
        "--matchmaking-concurrency",
        type=int,
        default=1,
        help="Number of simultaneous League tickets served by this one agent connection.",
    )
    result.add_argument(
        "--once",
        action="store_true",
        help="Play one public match instead of rejoining matchmaking continuously.",
    )
    result.add_argument(
        "--start-local-game",
        action="store_true",
        help="After connecting locally, create one game against a configured controller.",
    )
    result.add_argument(
        "--local-deck-session-id",
        default=os.getenv("DEEPDECK_LOCAL_DECK_SESSION_ID"),
    )
    result.add_argument(
        "--local-opponent-deck-session-id",
        default=os.getenv("DEEPDECK_LOCAL_OPPONENT_DECK_SESSION_ID"),
    )
    result.add_argument(
        "--local-opponent-controller",
        default=os.getenv("DEEPDECK_LOCAL_OPPONENT_CONTROLLER", "ai-random"),
    )
    result.add_argument(
        "--local-game-setup",
        help="Path to an inline local GameSetup prepared by DeepDeckLearner.",
    )
    result.add_argument(
        "--local-format",
        choices=("legacy", "commander"),
        default=os.getenv("DEEPDECK_LOCAL_FORMAT"),
    )
    result.add_argument(
        "--local-max-turns",
        type=int,
        default=int(os.getenv("DEEPDECK_LOCAL_MAX_TURNS", "200")),
    )
    result.add_argument("--log-level", default="INFO")
    return result


def _agent_and_config(arguments: argparse.Namespace) -> tuple[Agent, AgentConfig]:
    if arguments.example == "random":
        return build_random_agent(arguments.seed), random_config()
    if arguments.example == "alexios":
        return AlexiosAgent(), alexios_config()
    if arguments.example.startswith("forge-"):
        profile = ForgeReferenceProfile(arguments.example.removeprefix("forge-"))
        return build_forge_reference_agent(profile), forge_reference_config(profile.value)
    if arguments.example == "v13":
        if not arguments.checkpoint:
            raise SystemExit("V13 requires --checkpoint")
        try:
            from .v13_agent import load_v13_agent

            v13_agent = load_v13_agent(arguments.checkpoint, device=arguments.device)
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            raise SystemExit(str(error)) from error
        return v13_agent, deep_learning_config("v13")
    if arguments.example == "v15":
        if not arguments.checkpoint:
            raise SystemExit("V15 requires --checkpoint")
        try:
            from .v15_agent import load_v15_agent

            v15_agent = load_v15_agent(arguments.checkpoint, device=arguments.device)
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            raise SystemExit(str(error)) from error
        return v15_agent, deep_learning_config("v15")
    if is_oracle_checkpoint(arguments.checkpoint):
        return (
            OracleCheckpointAgent(arguments.checkpoint, device=arguments.device),
            deep_learning_config(arguments.example),
        )
    try:
        from .deep_learning import build_deep_learning_agent
    except ModuleNotFoundError as error:
        if error.name != "torch":
            raise
        raise SystemExit(
            'V11/V12 requires the optional dependency: pip install -e ".[deep-learning]"'
        ) from error
    try:
        agent = build_deep_learning_agent(
            arguments.example,
            checkpoint=arguments.checkpoint,
            device=arguments.device,
            allow_untrained=arguments.allow_untrained,
            seed=arguments.seed,
        )
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from error
    return agent, deep_learning_config(arguments.example)


async def _serve_local(runner: AgentRunner, arguments: argparse.Namespace) -> None:
    connection = asyncio.create_task(runner.serve())
    try:
        if arguments.start_local_game:
            local_game_setup = getattr(arguments, "local_game_setup", None)
            missing = (
                []
                if local_game_setup
                else [
                    name
                    for name, value in (
                        ("DEEPDECK_LOCAL_DECK_SESSION_ID", arguments.local_deck_session_id),
                        (
                            "DEEPDECK_LOCAL_OPPONENT_DECK_SESSION_ID",
                            arguments.local_opponent_deck_session_id,
                        ),
                    )
                    if not value
                ]
            )
            if missing:
                raise SystemExit(f"Missing local game configuration: {', '.join(missing)}")
            controller_id = await runner.wait_until_connected(timeout=30)
            game_format = arguments.local_format or runner.config.formats[0]
            if game_format not in runner.config.formats:
                supported = ", ".join(runner.config.formats)
                raise SystemExit(
                    f"{runner.config.name} does not declare {game_format}; supported: {supported}"
                )
            starting_life = 40 if game_format == "commander" else 20
            if local_game_setup:
                payload = json.loads(Path(local_game_setup).read_text(encoding="utf-8"))
                payload["aiControllerByPlayerId"] = {"local-agent": controller_id}
                headers = (
                    {"x-mtg-api-key": runner.target.engine_api_key}
                    if runner.target.engine_api_key
                    else None
                )
                async with httpx.AsyncClient(timeout=30) as client:
                    response = await client.post(
                        f"{arguments.engine_url.rstrip('/')}/game/sessions",
                        headers=headers,
                        json=payload,
                    )
                    if response.is_error:
                        raise RuntimeError(
                            f"Engine rejected the local game ({response.status_code}): "
                            f"{response.text}"
                        )
                    bootstrap = response.json()
            else:
                bootstrap = await runner.start_local_game(
                    LocalGame(
                        format=game_format,
                        seed=arguments.seed,
                        max_turns=arguments.local_max_turns,
                        free_mulligans=1 if game_format == "commander" else 0,
                        players=(
                            LocalPlayer(
                                player_id="example-agent",
                                deck_session_id=arguments.local_deck_session_id,
                                controller_id=controller_id,
                                name=runner.config.name,
                                starting_life=starting_life,
                            ),
                            LocalPlayer(
                                player_id="local-opponent",
                                deck_session_id=arguments.local_opponent_deck_session_id,
                                controller_id=arguments.local_opponent_controller,
                                name="Local opponent",
                                starting_life=starting_life,
                            ),
                        ),
                    )
                )
            session = bootstrap.get("session", bootstrap)
            session_id = (
                session.get("id") or session.get("sessionId") if isinstance(session, dict) else None
            )
            if session_id:
                print(
                    "DEEPDECK_PLAYTEST_SESSION "
                    + json.dumps({"sessionId": session_id, "engineUrl": arguments.engine_url}),
                    flush=True,
                )
            logging.getLogger(__name__).info("local game started: %s", session_id or bootstrap)
        await connection
    finally:
        connection.cancel()
        await asyncio.gather(connection, return_exceptions=True)


async def run(arguments: argparse.Namespace) -> None:
    agent, config = _agent_and_config(arguments)
    target = (
        ServerTarget.local(arguments.engine_url)
        if arguments.target == "local"
        else ServerTarget.deepdeckleague()
    )
    runner = AgentRunner(agent=agent, config=config, target=target, speed=arguments.speed)
    logging.getLogger(__name__).info(
        "connecting %s to %s as %s",
        config.name,
        target.agent_url,
        config.agent_id,
    )
    if arguments.target == "local":
        await _serve_local(runner, arguments)
        return
    matchmaking_ids = {
        "DEEPDECK_COMPETITION_VERSION_ID": arguments.competition_version_id,
        "DEEPDECK_DECK_VERSION_ID": arguments.deck_version_id,
    }
    missing = [name for name, value in matchmaking_ids.items() if not value]
    if missing:
        raise SystemExit(f"Missing public matchmaking configuration: {', '.join(missing)}")
    entries = _matchmaking_entries(arguments)
    connection = asyncio.create_task(runner.serve())
    active_tickets: dict[int, str] = {}
    match_watchers: dict[str, asyncio.Task[str]] = {}

    async def watch_match(match_id: str) -> str:
        while True:
            match = await runner.match(match_id)
            snapshot = _league_match_snapshot(match_id, match, runner.target.platform_url)
            print("DEEPDECK_LEAGUE_MATCH " + json.dumps(snapshot), flush=True)
            status = str(snapshot["status"])
            if status in {"complete", "failed", "cancelled"}:
                return status
            await asyncio.sleep(5.0)

    async def serve_seat(seat: int, entry: MatchmakingEntry) -> None:
        logger = logging.getLogger(__name__)
        while True:
            await runner.wait_until_connected(timeout=30)
            ticket = await (
                runner.join_matchmaking(entry)
                if arguments.once
                else runner.join_matchmaking_with_retry(entry)
            )
            ticket_id = str(ticket.get("id", ""))
            if ticket_id:
                active_tickets[seat] = ticket_id
            logger.info(
                "League seat %s queued ticket %s with %s eligible decks",
                seat + 1,
                ticket.get("id"),
                len(entry.deck_version_ids),
            )
            match_id = await runner._wait_for_match_id(ticket, 5.0)  # noqa: SLF001
            if match_id is None:
                return
            logger.info("League seat %s matched game %s", seat + 1, match_id)
            watcher = match_watchers.get(match_id)
            if watcher is None:
                watcher = asyncio.create_task(watch_match(match_id))
                match_watchers[match_id] = watcher
            try:
                await watcher
            finally:
                if match_watchers.get(match_id) is watcher:
                    match_watchers.pop(match_id, None)
            active_tickets.pop(seat, None)
            if arguments.once:
                return
            await asyncio.sleep(2.0)

    try:
        await asyncio.sleep(0)
        await runner.wait_until_connected(timeout=30)
        await asyncio.gather(*(serve_seat(index, entry) for index, entry in enumerate(entries)))
    finally:
        await asyncio.gather(
            *(runner.cancel_matchmaking_ticket(ticket_id) for ticket_id in active_tickets.values()),
            return_exceptions=True,
        )
        connection.cancel()
        await asyncio.gather(connection, return_exceptions=True)


def _unix_ms(value: Any) -> int | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp() * 1000)
    except ValueError:
        return None


def _league_match_snapshot(
    match_id: str,
    match: dict[str, Any],
    platform_url: str | None,
) -> dict[str, Any]:
    summary = match.get("summary")
    if not isinstance(summary, dict):
        summary = match
    participants = summary.get("participants", [])
    if not isinstance(participants, list):
        participants = []
    games = match.get("games", [])
    if not isinstance(games, list):
        games = []
    current_game = next(
        (game for game in games if isinstance(game, dict) and game.get("status") == "running"),
        next((game for game in reversed(games) if isinstance(game, dict)), {}),
    )
    status = str(summary.get("status", "scheduled"))
    frontend_url = (platform_url or "https://staging.deepdeckleague.com/api/v1").removesuffix(
        "/api/v1"
    )
    return {
        "matchId": match_id,
        "gameId": current_game.get("id"),
        "status": status,
        "decks": [
            participant.get("deck") for participant in participants if isinstance(participant, dict)
        ],
        "players": len(participants),
        "turnNumber": current_game.get("turnCount"),
        "roundNumber": current_game.get("number"),
        "decisions": 0,
        "startedAtUnixMs": _unix_ms(current_game.get("startedAt") or summary.get("startedAt")),
        "updatedAtUnixMs": int(datetime.now().timestamp() * 1000),
        "watchUrl": f"{frontend_url}/matches?tab=games&match={match_id}&replay=1&seat=0",
    }


def _matchmaking_entries(arguments: argparse.Namespace) -> list[MatchmakingEntry]:
    concurrency = max(1, min(32, int(arguments.matchmaking_concurrency)))
    deck_ids = tuple(
        dict.fromkeys([arguments.deck_version_id, *arguments.additional_deck_version_id])
    )
    return [
        MatchmakingEntry(
            competition_version_id=arguments.competition_version_id,
            client_seat_id=f"learner:seat-{index + 1}",
            deck_version_ids=deck_ids,
        )
        for index in range(concurrency)
    ]


def main() -> None:
    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path)
    arguments = parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, arguments.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    asyncio.run(run(arguments))


if __name__ == "__main__":
    main()
