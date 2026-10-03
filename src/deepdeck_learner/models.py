from __future__ import annotations

import json
import threading
import time
from contextlib import suppress
from pathlib import Path
from typing import Any

METRIC_WINDOWS = frozenset({"50", "200", "1000", "5000", "all"})
MAX_METRIC_POINTS = 500
_DIRECTORY_SIZE_CACHE_SECONDS = 300.0
_directory_size_cache: dict[Path, tuple[float, int]] = {}
_directory_size_lock = threading.Lock()


def _directory_size(path: Path) -> int:
    if not path.is_dir():
        return 0
    resolved = path.resolve()
    now = time.monotonic()
    cached = _directory_size_cache.get(resolved)
    if cached and now - cached[0] < _DIRECTORY_SIZE_CACHE_SECONDS:
        return cached[1]
    # Model discovery is polled by the UI. Serialize cache misses so concurrent
    # refreshes cannot repeatedly walk a run containing tens of thousands of replays.
    with _directory_size_lock:
        cached = _directory_size_cache.get(resolved)
        if cached and now - cached[0] < _DIRECTORY_SIZE_CACHE_SECONDS:
            return cached[1]
        total = 0
        for candidate in resolved.rglob("*"):
            try:
                if candidate.is_file():
                    total += candidate.stat().st_size
            except OSError:
                continue
        _directory_size_cache[resolved] = (time.monotonic(), total)
        return total


def _training_state(run: Path) -> dict[str, Any]:
    try:
        value = json.loads((run / "league-state.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        value = None
        for name in ("v15-training-state.json", "v13-training-state.json"):
            try:
                value = json.loads((run / name).read_text(encoding="utf-8"))
                break
            except (OSError, ValueError):
                continue
        if value is None:
            return {}
    if not isinstance(value, dict):
        return {}
    if "step" in value and "trainingStep" not in value:
        return {
            "phase": "world-model",
            "desiredState": value.get("status", "stopped"),
            "completedGames": 0,
            "trainingStep": int(value.get("step", 0) or 0),
            "parallelGames": 0,
            "activeGames": 0,
            "updatedAtUnixMs": None,
        }
    return {
        "phase": value.get("trainingPhase"),
        "desiredState": value.get("desiredState"),
        "completedGames": int(value.get("completed_episodes", 0) or 0),
        "trainingStep": int(value.get("trainingStep", 0) or 0),
        "parallelGames": int(value.get("parallelGameWorkers", 0) or 0),
        "activeGames": len(value.get("activeAttempts", []) or []),
        "updatedAtUnixMs": value.get("updatedAtUnixMs"),
    }


def _model_decks(root: Path, run: Path, metadata: dict[str, Any]) -> list[dict[str, Any]]:
    configured = metadata.get("decks", [])
    if isinstance(configured, list) and configured:
        return [deck for deck in configured if isinstance(deck, dict)]
    try:
        resolved = json.loads((run / "resolved-config.json").read_text(encoding="utf-8"))
        selected_ids = resolved.get("learnerSettings", {}).get("selectedDeckVersionIds", [])
    except (AttributeError, OSError, ValueError):
        selected_ids = []
    decks: list[dict[str, Any]] = []
    for version_id in selected_ids if isinstance(selected_ids, list) else []:
        try:
            deck = json.loads(
                (root / ".deepdeck" / "decks" / f"{version_id}.json").read_text(encoding="utf-8")
            )
        except (OSError, ValueError):
            continue
        if not isinstance(deck, dict):
            continue
        decks.append(
            {
                "id": str(deck.get("id", version_id)),
                "name": str(deck.get("name", "Local training deck")),
                "creator": deck.get("creator"),
                "version": int(deck.get("version") or 1),
                "format": str(deck.get("format", metadata.get("format", "legacy"))),
                "colors": deck.get("colors", []),
                "playableCardCount": int(
                    deck.get("playableCardCount", deck.get("cardCount", 0)) or 0
                ),
            }
        )
    return decks


def _v13_runtime_compatible(run: Path) -> bool:
    """Reject known pre-V13.2 runs before presenting them as playable."""
    config_path = run / "resolved-v13-config.json"
    if not config_path.is_file():
        # Older test fixtures and externally-produced checkpoints may not publish
        # the resolved config. Their loader remains the final compatibility check.
        return True
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        model = config.get("model", {})
    except (AttributeError, OSError, ValueError):
        return False
    return bool(
        isinstance(model, dict)
        and model.get("max_sequence_positions")
        and model.get("text_vocabulary_size")
    )


def _runtime_required_files(metadata: dict[str, Any]) -> list[str]:
    runtime = metadata.get("agentSdkRuntime")
    if isinstance(runtime, dict):
        required = runtime.get("requiredFiles")
        if isinstance(required, list) and all(isinstance(value, str) for value in required):
            return [value for value in required if value]
    return (
        ["rl-model.pt"]
        if str(metadata.get("architecture", "")).casefold() == "v13"
        else ["manifest.json", "checkpoint.pt"]
    )


def _runtime_ready(metadata: dict[str, Any], checkpoint: Path) -> bool:
    """Check the files declared by an Agent SDK runtime, without knowing its model."""
    required = _runtime_required_files(metadata)
    return bool(required) and all(
        not Path(value).is_absolute()
        and ".." not in Path(value).parts
        and (checkpoint / value).is_file()
        for value in required
    )


def local_models(root: Path, jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return user-owned models created by local deck-pool training runs."""
    statuses = {
        str(Path(str(job["artifact_path"])).resolve()): str(job["status"])
        for job in jobs
        if job.get("artifact_path")
    }
    models: list[dict[str, Any]] = []
    runs = root / ".deepdeck" / "runs"
    if not runs.is_dir():
        return models
    for metadata_path in runs.glob("*/local-model.json"):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(metadata, dict) or metadata.get("schemaVersion") != "local-model/v1":
            continue
        run_path = metadata_path.parent.resolve()
        checkpoint = Path(str(metadata.get("checkpointPath", "")))
        architecture = str(metadata.get("architecture", "")).casefold()
        checkpoint_ready = _runtime_ready(metadata, checkpoint)
        # Compatibility with old V13 metadata. A runtime descriptor is the
        # authority for new agents; the Learner must not inspect their model schema.
        if architecture == "v13" and not isinstance(metadata.get("agentSdkRuntime"), dict):
            checkpoint_ready = checkpoint_ready and _v13_runtime_compatible(metadata_path.parent)
        ready = bool(metadata.get("reservePlaytest")) and checkpoint_ready
        models.append(
            {
                **metadata,
                "runPath": str(run_path),
                "checkpointPath": str(checkpoint),
                "status": statuses.get(str(run_path), "stopped"),
                "ready": ready,
                "decks": _model_decks(root, metadata_path.parent, metadata),
                "diskBytes": _directory_size(run_path),
                "weightsBytes": _directory_size(checkpoint),
                "trainingState": _training_state(run_path),
            }
        )
    return sorted(models, key=lambda model: str(model.get("createdAt", "")), reverse=True)


def deck_statistics(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    runs = root / ".deepdeck" / "runs"
    if not runs.is_dir():
        return rows
    for metadata_path in runs.glob("*/local-model.json"):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            leaderboard = json.loads(
                (metadata_path.parent / "training-leaderboard.json").read_text(encoding="utf-8")
            )
        except (OSError, ValueError):
            leaderboard = {"deckParticipants": []}
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
        if not isinstance(metadata, dict):
            continue
        participants = (
            leaderboard.get("deckParticipants", []) if isinstance(leaderboard, dict) else []
        )
        if str(metadata.get("architecture", "")).casefold() == "v13":
            games: list[dict[str, Any]] = []
            try:
                lines = (
                    (metadata_path.parent / "v13-engine-games.jsonl")
                    .read_text(encoding="utf-8", errors="replace")
                    .splitlines()
                )
            except OSError:
                lines = []
            for line in lines:
                try:
                    game = json.loads(line)
                except ValueError:
                    continue
                if isinstance(game, dict):
                    games.append(game)
            for deck in _model_decks(root, metadata_path.parent, metadata):
                deck_name = str(deck.get("name", ""))
                wins = losses = draws = appearances = 0
                for game in games:
                    by_player = game.get("deckByPlayer", {})
                    if not isinstance(by_player, dict):
                        continue
                    winner = (game.get("outcome") or {}).get("winner")
                    for player_id, played_deck in by_player.items():
                        if not str(played_deck).startswith(deck_name):
                            continue
                        appearances += 1
                        if not winner:
                            draws += 1
                        elif str(winner) == str(player_id):
                            wins += 1
                        else:
                            losses += 1
                rows.append(
                    {
                        "modelId": metadata.get("id"),
                        "modelName": metadata.get("name"),
                        "architecture": metadata.get("architecture"),
                        "deckVersionId": str(deck.get("id", "")),
                        "deckName": deck_name,
                        "format": deck.get("format", metadata.get("format")),
                        "ratingSystem": "engine-self-play",
                        "mu": 0.0,
                        "sigma": 0.0,
                        "ordinal": 0.0,
                        "rank": None,
                        "matches": appearances,
                        "gameWins": wins,
                        "gameLosses": losses,
                        "draws": draws,
                        "winRate": wins / (wins + losses) if wins + losses else None,
                    }
                )
            continue
        for deck in _model_decks(root, metadata_path.parent, metadata):
            if not isinstance(deck, dict):
                continue
            version_id = str(deck.get("id", ""))
            candidate = next(
                (
                    item
                    for item in participants
                    if isinstance(item, dict)
                    and item.get("participantId") == metadata.get("id")
                    and (
                        version_id[:8] in str(item.get("deckName", ""))
                        or str(item.get("deckName", "")).startswith(str(deck.get("name", "")))
                    )
                ),
                {},
            )
            game_wins = int(candidate.get("gameWins", 0))
            game_losses = int(candidate.get("gameLosses", 0))
            decided_games = game_wins + game_losses
            rows.append(
                {
                    "modelId": metadata.get("id"),
                    "modelName": metadata.get("name"),
                    "architecture": metadata.get("architecture"),
                    "deckVersionId": version_id,
                    "deckName": deck.get("name"),
                    "format": deck.get("format", metadata.get("format")),
                    "ratingSystem": "plackett-luce",
                    "mu": float(candidate.get("mu", 25.0)),
                    "sigma": float(candidate.get("sigma", 25.0 / 3.0)),
                    "ordinal": float(candidate.get("ordinal", 0.0)),
                    "rank": candidate.get("rank"),
                    "matches": int(candidate.get("games", 0)),
                    "gameWins": game_wins,
                    "gameLosses": game_losses,
                    "winRate": game_wins / decided_games if decided_games else None,
                }
            )
    return sorted(rows, key=lambda row: (str(row["modelName"]), -float(row["ordinal"])))


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return ordered[index]


def _downsample(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(records) <= MAX_METRIC_POINTS:
        return records
    indices = {
        round(index * (len(records) - 1) / (MAX_METRIC_POINTS - 1))
        for index in range(MAX_METRIC_POINTS)
    }
    return [records[index] for index in sorted(indices)]


def _window_summary(records: list[dict[str, Any]], *, is_v13_world_model: bool) -> dict[str, Any]:
    if is_v13_world_model:
        steps = [int(record.get("step", 0) or 0) for record in records]
        covered_steps = max(steps) - min(steps) + 1 if steps else 0
        elapsed = [
            float(record["elapsed_seconds"])
            for record in records
            if isinstance(record.get("elapsed_seconds"), (int, float))
        ]
        return {
            "sampleType": "synthetic-batches",
            "gameMetricsAvailable": False,
            "engineGameMetrics": False,
            "generatedSamples": covered_steps,
            "usedSamples": covered_steps,
            "failedSamples": 0,
            "utilizationRate": 1.0 if records else None,
            "coveredTrainingSteps": covered_steps,
            "generatedDecisions": 0,
            "usedDecisions": 0,
            "averageGameSeconds": None,
            "p50GameSeconds": None,
            "p95GameSeconds": None,
            "simulationSeconds": None,
            "trainingSeconds": max(elapsed) - min(elapsed) if len(elapsed) > 1 else None,
            "collectionWallSeconds": None,
            "gamesPerHour": None,
        }

    durations: list[float] = []
    for record in records:
        raw_durations = record.get("gameDurationsSeconds")
        if isinstance(raw_durations, list):
            durations.extend(
                float(value) for value in raw_durations if isinstance(value, (int, float))
            )
        elif isinstance(record.get("gameDurationSeconds"), (int, float)):
            durations.append(float(record["gameDurationSeconds"]))
    batches: dict[int, dict[str, Any]] = {}
    for index, record in enumerate(records):
        batch = record.get("rolloutBatch")
        if not isinstance(batch, dict):
            continue
        key = int(record.get("trainingStep", index) or index)
        batches.setdefault(key, batch)
    generated = len(records)
    used = sum(1 for record in records if isinstance(record.get("ppo"), dict))
    failed = 0
    attempted = generated
    used_decisions = sum(int(record.get("decisions", 0) or 0) for record in records)
    collection_wall = sum(
        float(batch.get("collectionWallSeconds", 0.0) or 0.0) for batch in batches.values()
    )
    training_seconds = sum(
        float(batch.get("trainingWallSeconds", batch.get("ppoWallSeconds", 0.0)) or 0.0)
        for batch in batches.values()
    )
    if batches:
        attempted = sum(int(batch.get("requestedGames", 0) or 0) for batch in batches.values())
        generated = sum(int(batch.get("completedGames", 0) or 0) for batch in batches.values())
        failed = sum(int(batch.get("failedGames", 0) or 0) for batch in batches.values())
        used = generated
        used_decisions = sum(int(batch.get("totalDecisions", 0) or 0) for batch in batches.values())
    simulation_seconds = sum(durations)
    latest_source = str(records[-1].get("data_source", "")) if records else ""
    is_in_process_rl = latest_source.startswith("tiny-self-play")
    return {
        "sampleType": "rl-episodes" if is_in_process_rl else "games",
        "gameMetricsAvailable": True,
        "engineGameMetrics": not is_in_process_rl,
        "attemptedSamples": attempted,
        "generatedSamples": generated,
        "usedSamples": used,
        "failedSamples": failed,
        "utilizationRate": used / generated if generated else None,
        "coveredTrainingSteps": 0,
        "generatedDecisions": sum(int(record.get("decisions", 0) or 0) for record in records),
        "usedDecisions": used_decisions,
        "averageGameSeconds": sum(durations) / len(durations) if durations else None,
        "p50GameSeconds": _percentile(durations, 0.50),
        "p95GameSeconds": _percentile(durations, 0.95),
        "simulationSeconds": simulation_seconds,
        "trainingSeconds": training_seconds,
        "collectionWallSeconds": collection_wall,
        "gamesPerHour": generated / collection_wall * 3600.0 if collection_wall else None,
    }


def training_statistics(root: Path, metric_window: str = "200") -> list[dict[str, Any]]:
    if metric_window not in METRIC_WINDOWS:
        raise ValueError(f"unsupported metric window: {metric_window}")
    rows: list[dict[str, Any]] = []
    runs = root / ".deepdeck" / "runs"
    if not runs.is_dir():
        return rows
    for metadata_path in runs.glob("*/local-model.json"):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(metadata, dict):
            continue
        run = metadata_path.parent
        state: dict[str, Any] = {}
        try:
            loaded_state = json.loads((run / "league-state.json").read_text(encoding="utf-8"))
            if isinstance(loaded_state, dict):
                state = loaded_state
        except (OSError, ValueError):
            pass
        records: list[dict[str, Any]] = []
        training_path = run / "training.jsonl"
        v13_training_path = run / "v13-metrics.jsonl"
        v13_rl_training_path = run / "v13-rl-metrics.jsonl"
        architecture = str(metadata.get("architecture", "")).casefold()
        is_v13 = architecture == "v13"
        is_v15 = architecture == "v15"
        if is_v13:
            training_path = (
                v13_rl_training_path if v13_rl_training_path.is_file() else v13_training_path
            )
        elif is_v15:
            training_path = run / "v15-metrics.jsonl"
        is_v13_world_model = is_v13 and training_path == v13_training_path
        is_latent_world_model = is_v13_world_model or is_v15
        if training_path.is_file():
            try:
                lines = training_path.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                lines = []
            for line in lines:
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                if isinstance(record, dict):
                    records.append(record)
            if is_v13 and not is_v13_world_model:
                unique_records: dict[tuple[int, int], dict[str, Any]] = {}
                for record in records:
                    key = (
                        int(record.get("trainingStep", 0) or 0),
                        int(record.get("episode", 0) or 0),
                    )
                    unique_records[key] = record
                records = sorted(
                    unique_records.values(),
                    key=lambda record: (
                        int(record.get("trainingStep", 0) or 0),
                        int(record.get("episode", 0) or 0),
                    ),
                )
            if metric_window != "all":
                records = records[-int(metric_window) :]
        production_records = records
        if is_v13 and records:
            latest_data_source = str(records[-1].get("data_source", ""))
            if latest_data_source:
                same_source_records = [
                    record
                    for record in records
                    if str(record.get("data_source", "")) == latest_data_source
                ]
                if same_source_records:
                    production_records = same_source_records
        durations: list[float] = []
        for record in production_records:
            raw_durations = record.get("gameDurationsSeconds")
            if isinstance(raw_durations, list):
                durations.extend(
                    float(value) for value in raw_durations if isinstance(value, (int, float))
                )
            elif isinstance(record.get("gameDurationSeconds"), (int, float)):
                durations.append(float(record["gameDurationSeconds"]))
        latest_metrics: list[dict[str, Any]] = []
        plotted_records = _downsample(records)
        for record in plotted_records:
            raw_ppo = record.get("ppo")
            ppo: dict[str, Any] = raw_ppo if isinstance(raw_ppo, dict) else {}
            source = record if is_v13 or is_v15 else ppo
            losses = {
                key: value
                for key, value in source.items()
                if (key == "loss" or key.endswith("_loss")) and isinstance(value, (int, float))
            }
            latest_metrics.append(
                {
                    "episode": int(record.get("episode", 0) or 0),
                    "trainingStep": int(record.get("trainingStep", record.get("step", 0)) or 0),
                    "loss": source.get("loss"),
                    "policyLoss": source.get("policy_loss"),
                    "valueLoss": source.get("value_loss"),
                    "entropy": source.get("entropy"),
                    "episodeReward": record.get("episode_reward"),
                    "approxKl": source.get("approx_kl"),
                    "clipFraction": source.get("clip_fraction"),
                    "dataSource": record.get("data_source"),
                    "losses": losses,
                    "gameDurationSeconds": record.get("gameDurationSeconds"),
                }
            )
        if is_v13:
            try:
                v13_state = json.loads(
                    (run / "v13-training-state.json").read_text(encoding="utf-8")
                )
                if isinstance(v13_state, dict):
                    state = v13_state
            except (OSError, ValueError):
                pass
        elif is_v15:
            try:
                v15_state = json.loads(
                    (run / "v15-training-state.json").read_text(encoding="utf-8")
                )
                if isinstance(v15_state, dict):
                    state = v15_state
            except (OSError, ValueError):
                pass
        engine_game_count = 0
        if is_v13:
            with suppress(OSError):
                engine_game_count = sum(
                    bool(line.strip())
                    for line in (run / "v13-engine-games.jsonl")
                    .read_text(encoding="utf-8", errors="replace")
                    .splitlines()
                )
        active_attempts = state.get("activeAttempts", [])
        rows.append(
            {
                "modelId": metadata.get("id"),
                "modelName": metadata.get("name"),
                "architecture": metadata.get("architecture"),
                "format": metadata.get("format"),
                "completedGames": (
                    engine_game_count
                    if engine_game_count
                    else int(state.get("completed_episodes", 0) or 0)
                ),
                "totalRLEpisodes": int(state.get("completed_episodes", 0) or 0),
                "engineCompletedGames": engine_game_count,
                "trainingStep": int(state.get("trainingStep", state.get("step", 0)) or 0),
                "parallelGames": int(state.get("parallelGameWorkers", 0) or 0),
                "activeGames": len(active_attempts) if isinstance(active_attempts, list) else 0,
                "phase": state.get(
                    "trainingPhase", "world-model" if is_v13 or is_v15 else "not-started"
                ),
                "desiredState": state.get("desiredState", state.get("status", "stopped")),
                "trainingElapsedSeconds": float(state.get("trainingElapsedSeconds", 0) or 0),
                "simulationSeconds": float(state.get("gameSimulationSeconds", 0) or 0),
                "modelTrainingSeconds": float(state.get("modelTrainingSeconds", 0) or 0),
                "averageGameSeconds": (sum(durations) / len(durations) if durations else None),
                "metricWindow": metric_window,
                "metricRecordCount": len(records),
                "metricPointsReturned": len(latest_metrics),
                "windowSummary": _window_summary(
                    production_records, is_v13_world_model=is_latent_world_model
                ),
                "latestMetrics": latest_metrics,
                "updatedAtUnixMs": state.get("updatedAtUnixMs"),
            }
        )
    return sorted(rows, key=lambda row: str(row.get("modelName", "")))
