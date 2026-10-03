from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import random
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx
import torch
import yaml
from deepdeck_agent import TrainingBridge
from torch.distributions import Categorical

from oracle_ai.encoding_v13 import GraphObservation, GraphObservationEncoderV13
from oracle_ai.model_v13 import GraphBeliefWorldModelV13, ModelConfigV13, PolicyBatchV13
from oracle_ai.training.contracts import v13_training_contract
from oracle_ai.training.core import DecisionStep
from oracle_ai.training.curriculum import AdaptiveCurriculumSampler
from oracle_ai.training.environments import (
    RustSelfPlayEnvironment,
    RustSessionEnvironment,
    TinySelfPlayEnvironment,
)
from oracle_ai.training.league import RandomTrainingMatchupSampler, _deck_catalog


@dataclass(frozen=True)
class V13RLTrainingConfig:
    episodes: int = 10_000
    rollout_episodes: int = 16
    horizon: int = 8
    learning_rate: float = 1e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    epochs: int = 4
    minibatch_size: int = 64
    max_grad_norm: float = 1.0
    target_kl: float = 0.02
    checkpoint_every: int = 25
    max_checkpoints: int = 3
    seed: int = 13013
    device: str = "auto"
    environment: str = "tiny"
    evaluation_every_updates: int = 10
    evaluation_games: int = 4
    max_consecutive_errors: int = 10
    max_episode_decisions: int = 512
    greedy_actions: bool = False
    staging_evaluation_every_updates: int = 100
    staging_platform_url: str = "https://staging.deepdeckleague.com/api"
    staging_format: str = "legacy"


@dataclass
class _Transition:
    graph: GraphObservation
    action_features: torch.Tensor
    action_index: int
    old_log_probability: float
    value: float
    reward: float
    done: bool
    recurrent_state: torch.Tensor | None = None
    player_id: str = "learner"
    advantage: float = 0.0
    return_: float = 0.0


def _append(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as destination:
        destination.write(json.dumps(value, sort_keys=True) + "\n")
        destination.flush()


def _write_json_atomic(path: Path, value: dict[str, Any]) -> bool:
    pending = path.with_name(f".{path.name}.{os.getpid()}.pending")
    pending.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for _ in range(20):
        try:
            pending.replace(path)
            return True
        except PermissionError:
            time.sleep(0.05)
    pending.unlink(missing_ok=True)
    return False


def _report_with_retry(method: Any, **kwargs: Any) -> None:
    """Tolerate short-lived Windows locks from dashboard readers."""

    for attempt in range(20):
        try:
            method(**kwargs)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.05)


def _launch_staging_evaluation(
    *,
    checkpoint: Path,
    output: Path,
    update: int,
    config: V13RLTrainingConfig,
) -> dict[str, Any]:
    """Launch one isolated public exhibition without blocking PPO training."""

    event: dict[str, Any] = {
        "event": "staging-evaluation",
        "update": update,
        "checkpoint": str(checkpoint),
        "status": "skipped",
        "timestampUnixMs": int(time.time() * 1000),
    }
    if not os.getenv("DEEPDECK_API_KEY", "").strip():
        event["reason"] = "DEEPDECK_API_KEY is not available to the training process"
        _append(output / "v13-staging-evaluations.jsonl", event)
        return event

    worker_dir = output / "staging-evaluations"
    worker_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = worker_dir / "tmp"
    temp_dir.mkdir(exist_ok=True)
    result_path = worker_dir / f"update-{update}.json"
    log_path = worker_dir / f"update-{update}.log"
    version = f"v13-update-{update}"
    environment = os.environ.copy()
    environment.update(
        {
            "DEEPDECK_AGENT_ID": f"com.deepdeckleague.training.v13.update-{update}",
            "DEEPDECK_AGENT_VERSION": version,
            "DEEPDECK_PLATFORM_URL": f"{config.staging_platform_url.rstrip('/')}/v1",
            "TEMP": str(temp_dir),
            "TMP": str(temp_dir),
        }
    )
    command = [
        sys.executable,
        "-m",
        "oracle_ai.training.staging_v13",
        "--checkpoint",
        str(checkpoint),
        "--platform-url",
        config.staging_platform_url,
        "--format",
        config.staging_format,
        "--version",
        version,
        "--result",
        str(result_path),
    ]
    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    with log_path.open("ab") as log:
        process = subprocess.Popen(
            command,
            cwd=Path.cwd(),
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            creationflags=creationflags,
            start_new_session=os.name != "nt",
        )
    event.update({"status": "launched", "processId": process.pid, "version": version})
    _append(output / "v13-staging-evaluations.jsonl", event)
    return event


def _latest_checkpoint(root: Path, pattern: str) -> Path | None:
    candidates: list[tuple[int, Path]] = []
    for path in root.glob(pattern):
        try:
            number = int(path.parent.name.rsplit("-", 1)[-1])
        except ValueError:
            continue
        candidates.append((number, path))
    return max(candidates, default=(0, None), key=lambda item: item[0])[1]


def _latest_readable_checkpoint(
    root: Path, pattern: str, device: torch.device
) -> tuple[Path | None, dict[str, Any] | None]:
    """Load the newest complete checkpoint, skipping interrupted writes safely."""

    candidates: list[tuple[int, Path]] = []
    for path in root.glob(pattern):
        try:
            number = int(path.parent.name.rsplit("-", 1)[-1])
        except ValueError:
            continue
        candidates.append((number, path))
    for _, path in sorted(candidates, reverse=True):
        try:
            payload = torch.load(path, map_location=device, weights_only=False)
        except (EOFError, OSError, RuntimeError, pickle.UnpicklingError) as error:
            print(
                json.dumps(
                    {
                        "event": "checkpoint-skipped",
                        "checkpoint": str(path),
                        "reason": str(error),
                    }
                ),
                flush=True,
            )
            continue
        if isinstance(payload, dict):
            return path, payload
    return None, None


def _prune_rl_checkpoints(root: Path, keep: int) -> None:
    """Keep only the newest complete, resumable RL checkpoints."""
    retention = max(1, int(keep))
    candidates: list[tuple[int, Path]] = []
    for path in root.glob("update-*") if root.is_dir() else ():
        try:
            update = int(path.name.removeprefix("update-"))
        except ValueError:
            continue
        if path.is_dir() and (path / "rl-model.pt").is_file():
            candidates.append((update, path))
    for _, expired in sorted(candidates, reverse=True)[retention:]:
        shutil.rmtree(expired)


def _single_batch(
    graph: GraphObservation,
    actions: torch.Tensor,
    device: torch.device,
    recurrent_state: torch.Tensor | None = None,
) -> PolicyBatchV13:
    edge_index = (
        torch.cat(
            (torch.zeros((1, graph.edges.shape[1]), dtype=torch.long), graph.edges), dim=0
        ).to(device)
        if graph.edges.numel()
        else torch.empty((3, 0), dtype=torch.long, device=device)
    )
    return PolicyBatchV13(
        node_features=graph.node_features.unsqueeze(0).to(device),
        node_types=graph.node_types.unsqueeze(0).to(device),
        owners=graph.owners.unsqueeze(0).to(device),
        node_mask=torch.ones((1, graph.node_features.shape[0]), dtype=torch.bool, device=device),
        legal_action_features=actions.unsqueeze(0).to(device),
        legal_action_mask=torch.ones((1, actions.shape[0]), dtype=torch.bool, device=device),
        edge_index=edge_index,
        edge_types=graph.edge_types.to(device),
        recurrent_state=recurrent_state,
        text_tokens=graph.text_tokens.unsqueeze(0).to(device),
        text_mask=graph.text_mask.unsqueeze(0).to(device),
        controllers=graph.controllers.unsqueeze(0).to(device),
        zones=graph.zones.unsqueeze(0).to(device),
        positions=graph.positions.unsqueeze(0).to(device),
    )


def _minibatch(
    transitions: list[_Transition], indices: list[int], device: torch.device
) -> tuple[PolicyBatchV13, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    items = [transitions[index] for index in indices]
    batch_size = len(items)
    feature_dim = items[0].graph.node_features.shape[-1]
    action_dim = items[0].action_features.shape[-1]
    max_nodes = max(item.graph.node_features.shape[0] for item in items)
    max_actions = max(item.action_features.shape[0] for item in items)
    max_text = max(item.graph.text_tokens.shape[1] for item in items)
    node_features = torch.zeros((batch_size, max_nodes, feature_dim), device=device)
    node_types = torch.zeros((batch_size, max_nodes), dtype=torch.long, device=device)
    owners = torch.zeros((batch_size, max_nodes), dtype=torch.long, device=device)
    controllers = torch.zeros((batch_size, max_nodes), dtype=torch.long, device=device)
    zones = torch.zeros((batch_size, max_nodes), dtype=torch.long, device=device)
    positions = torch.zeros((batch_size, max_nodes), dtype=torch.long, device=device)
    node_mask = torch.zeros((batch_size, max_nodes), dtype=torch.bool, device=device)
    action_features = torch.zeros((batch_size, max_actions, action_dim), device=device)
    action_mask = torch.zeros((batch_size, max_actions), dtype=torch.bool, device=device)
    text_tokens = torch.zeros(
        (batch_size, max_nodes, max_text), dtype=torch.long, device=device
    )
    text_mask = torch.zeros(
        (batch_size, max_nodes, max_text), dtype=torch.bool, device=device
    )
    edge_indices: list[torch.Tensor] = []
    edge_types: list[torch.Tensor] = []
    memory_example = next(
        (item.recurrent_state for item in items if item.recurrent_state is not None), None
    )
    recurrent_states = (
        torch.zeros((batch_size, memory_example.numel()), device=device)
        if memory_example is not None
        else None
    )
    for row, item in enumerate(items):
        nodes = item.graph.node_features.shape[0]
        actions = item.action_features.shape[0]
        node_features[row, :nodes] = item.graph.node_features.to(device)
        node_types[row, :nodes] = item.graph.node_types.to(device)
        owners[row, :nodes] = item.graph.owners.to(device)
        controllers[row, :nodes] = item.graph.controllers.to(device)
        zones[row, :nodes] = item.graph.zones.to(device)
        positions[row, :nodes] = item.graph.positions.to(device)
        node_mask[row, :nodes] = True
        action_features[row, :actions] = item.action_features.to(device)
        action_mask[row, :actions] = True
        text_width = item.graph.text_tokens.shape[1]
        text_tokens[row, :nodes, :text_width] = item.graph.text_tokens.to(device)
        text_mask[row, :nodes, :text_width] = item.graph.text_mask.to(device)
        if item.graph.edges.numel():
            edge_indices.append(
                torch.cat(
                    (
                        torch.full((1, item.graph.edges.shape[1]), row, dtype=torch.long),
                        item.graph.edges,
                    ),
                    dim=0,
                )
            )
            edge_types.append(item.graph.edge_types)
        if recurrent_states is not None and item.recurrent_state is not None:
            recurrent_states[row] = item.recurrent_state.reshape(-1).to(device)
    packed_edges = (
        torch.cat(edge_indices, dim=1).to(device)
        if edge_indices
        else torch.empty((3, 0), dtype=torch.long, device=device)
    )
    packed_types = (
        torch.cat(edge_types).to(device)
        if edge_types
        else torch.empty((0,), dtype=torch.long, device=device)
    )
    batch = PolicyBatchV13(
        node_features=node_features,
        node_types=node_types,
        owners=owners,
        node_mask=node_mask,
        legal_action_features=action_features,
        legal_action_mask=action_mask,
        edge_index=packed_edges,
        edge_types=packed_types,
        recurrent_state=recurrent_states,
        text_tokens=text_tokens,
        text_mask=text_mask,
        controllers=controllers,
        zones=zones,
        positions=positions,
    )
    return (
        batch,
        torch.tensor([item.action_index for item in items], dtype=torch.long, device=device),
        torch.tensor([item.old_log_probability for item in items], device=device),
        torch.tensor([item.advantage for item in items], device=device),
        torch.tensor([item.return_ for item in items], device=device),
    )


def _advantages(trajectory: list[_Transition], config: V13RLTrainingConfig) -> None:
    advantage = 0.0
    next_value = 0.0
    for transition in reversed(trajectory):
        continuation = 0.0 if transition.done else 1.0
        delta = transition.reward + config.gamma * next_value * continuation - transition.value
        advantage = delta + config.gamma * config.gae_lambda * continuation * advantage
        transition.advantage = advantage
        transition.return_ = advantage + transition.value
        next_value = transition.value


def _self_play_advantages(trajectory: list[_Transition], config: V13RLTrainingConfig) -> None:
    by_player: dict[str, list[_Transition]] = defaultdict(list)
    for transition in trajectory:
        by_player[transition.player_id].append(transition)
    for player_trajectory in by_player.values():
        if player_trajectory:
            player_trajectory[-1].done = True
        _advantages(player_trajectory, config)


def _save_replay(
    output: Path,
    environment: RustSelfPlayEnvironment,
    record: dict[str, Any],
    limit: int,
) -> None:
    if limit <= 0 or not environment.replay_frames:
        return
    replay_dir = output / "replays"
    replay_dir.mkdir(parents=True, exist_ok=True)
    replay_id = f"episode-{int(record['episode']):08d}"
    target = replay_dir / f"{replay_id}.json"
    saved_target = output / "saved-replays" / f"{replay_id}.json"
    if target.exists() or saved_target.exists():
        replay_id = f"{replay_id}-{int(record['completedAtUnixMs'])}"
        target = replay_dir / f"{replay_id}.json"
    _write_json_atomic(
        target,
        {
            "schemaVersion": "deepdeck-replay/v1",
            "id": replay_id,
            "createdAtUnixMs": record["completedAtUnixMs"],
            "metadata": record,
            "frames": environment.replay_frames,
        },
    )
    paths = sorted(replay_dir.glob("episode-*.json"), key=lambda path: path.name)
    excess = max(0, len(paths) - limit)
    for expired in paths:
        if excess <= 0:
            break
        lease_dir = replay_dir / ".leases" / expired.stem
        protected = False
        for lease in lease_dir.glob("*.lease") if lease_dir.is_dir() else ():
            try:
                protected = int(lease.read_text(encoding="utf-8").strip()) > int(
                    time.time()
                )
            except (OSError, ValueError):
                continue
            if protected:
                break
        if protected:
            continue
        expired.unlink(missing_ok=True)
        excess -= 1


def _terminal_result(
    environment: RustSessionEnvironment,
    matchup: Any,
    learner_reward: float | None = None,
) -> tuple[dict[str, Any], dict[str, float]]:
    state = dict((environment.current_view or {}).get("state", {}))
    outcome = state.get("outcome") or {}
    winner = outcome.get("winner")
    players = [str(player["id"]) for player in matchup.setup.get("players", [])]
    losers = set(outcome.get("losers", []))
    rewards = {
        player_id: 1.0 if player_id == winner else (-1.0 if player_id in losers else 0.0)
        for player_id in players
    }
    if learner_reward is not None:
        rewards[matchup.learner_player_id] = float(learner_reward)
        if learner_reward < 0 and not winner and matchup.opponent_player_id in rewards:
            rewards[matchup.opponent_player_id] = 1.0
    return state, rewards


def _engine_connection_failure(error: BaseException) -> bool:
    """Return whether an Engine failure is transport infrastructure, not a bad game."""

    return isinstance(error, httpx.NetworkError)


def _run_engine_evaluation(
    *,
    model: GraphBeliefWorldModelV13,
    baseline: GraphBeliefWorldModelV13,
    encoder: GraphObservationEncoderV13,
    environment: RustSelfPlayEnvironment,
    sampler: RandomTrainingMatchupSampler,
    device: torch.device,
    seed: int,
    games: int,
    training_step: int,
    completed_episodes: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    model.eval()
    baseline.eval()
    for pair_index in range((games + 1) // 2):
        seats = min(2, games - len(results))
        for retry in range(5):
            game_seed = seed + pair_index + retry * 1_000_003
            matchup = sampler.sample(random.Random(game_seed))
            pair_results: list[dict[str, Any]] = []
            pair_failed = False
            for seat in range(seats):
                matchup_id = f"v13-evaluation-{pair_index}-{retry}-{seat}"
                current_matchup = type(matchup)(**{**vars(matchup), "id": matchup_id})
                environment.matchups[matchup_id] = current_matchup
                candidate_player = f"player-{seat + 1}"
                decisions = 0
                memories: dict[str, torch.Tensor] = {}
                try:
                    step = environment.reset(matchup_id, game_seed, False)
                    while not step.done:
                        if decisions >= 512:
                            raise RuntimeError(
                                "Engine evaluation exceeded 512 policy decisions"
                            )
                        observer_id = step.player_id or "player-1"
                        graph = encoder.encode(step.state, observer_id)
                        actions = encoder.encode_actions(step.actions, step.state, observer_id)
                        policy = model if observer_id == candidate_player else baseline
                        with torch.no_grad():
                            logits, _, memory = policy.policy_value_with_memory(
                                _single_batch(
                                    graph, actions, device, memories.get(observer_id)
                                )
                            )
                        memories[observer_id] = memory.detach()
                        step = environment.step(int(logits.argmax(dim=-1).item()))
                        decisions += 1
                    state = dict((environment.current_view or {}).get("state", {}))
                    outcome = state.get("outcome") or {}
                    winner = outcome.get("winner")
                    pair_results.append(
                        {
                            "seed": game_seed,
                            "candidateSeat": seat,
                            "candidatePlayerId": candidate_player,
                            "candidateDeck": current_matchup.deck_names[seat],
                            "opponentDeck": current_matchup.deck_names[1 - seat],
                            "result": (
                                "candidateWin"
                                if winner == candidate_player
                                else "baselineWin"
                                if winner
                                else "draw"
                            ),
                            "winner": winner,
                            "turnNumber": state.get("turnNumber"),
                            "decisions": decisions,
                        }
                    )
                except (RuntimeError, httpx.HTTPError) as error:
                    failures.append(
                        {
                            "seed": game_seed,
                            "pair": pair_index,
                            "seat": seat,
                            "retry": retry + 1,
                            "error": f"{type(error).__name__}: {error}",
                        }
                    )
                    pair_failed = True
                    break
                finally:
                    environment.matchups.pop(matchup_id, None)
            if not pair_failed:
                results.extend(pair_results)
                break
    candidate_wins = sum(result["result"] == "candidateWin" for result in results)
    baseline_wins = sum(result["result"] == "baselineWin" for result in results)
    games_played = len(results)
    proportion = candidate_wins / games_played if games_played else 0.0
    z = 1.959963984540054
    denominator = 1 + z * z / games_played if games_played else 1.0
    centre = proportion + z * z / (2 * games_played) if games_played else 0.0
    margin = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / games_played
            + z * z / (4 * games_played * games_played)
        )
        if games_played
        else 0.0
    )
    model.train()
    return {
        "schemaVersion": "v13-engine-evaluation/v1",
        "period": training_step,
        "candidateTrainingStep": training_step,
        "completedEpisodes": completed_episodes,
        "opponentVersion": "v13-engine-baseline",
        "fixedSeeds": [result["seed"] for result in results],
        "summary": {
            "expectedGames": games,
            "completedGames": len(results),
            "candidateWins": candidate_wins,
            "baselineWins": baseline_wins,
            "draws": len(results) - candidate_wins - baseline_wins,
            "failedGames": len(failures),
            "candidateWinRate": proportion,
            "lower95": (centre - margin) / denominator if games_played else 0.0,
            "upper95": (centre + margin) / denominator if games_played else 0.0,
        },
        "games": results,
        "failures": failures,
        "evaluationSeconds": time.perf_counter() - started,
    }


def _run_curriculum_evaluation(
    *,
    model: GraphBeliefWorldModelV13,
    encoder: GraphObservationEncoderV13,
    environment: RustSessionEnvironment,
    sampler: AdaptiveCurriculumSampler,
    device: torch.device,
    seed: int,
    training_step: int,
    completed_episodes: int,
) -> dict[str, Any]:
    """Evaluate every selected skill once on stable seeds without updating PPO."""

    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    model.eval()
    for index, scenario in enumerate(sampler.scenarios):
        base_game_seed = seed + index * 10_007
        attempt_errors: list[dict[str, Any]] = []
        for attempt in range(3):
            attempt_started = time.perf_counter()
            game_seed = base_game_seed + attempt * 1_000_003
            print(
                json.dumps(
                    {
                        "event": "curriculum-evaluation-scenario-start",
                        "trainingStep": training_step,
                        "scenarioId": scenario.id,
                        "scenarioIndex": index + 1,
                        "scenarioCount": len(sampler.scenarios),
                        "attempt": attempt + 1,
                        "maxAttempts": 3,
                        "seed": game_seed,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            matchup_id = (
                f"curriculum-evaluation:{scenario.id}:{training_step}:attempt-{attempt + 1}"
            )
            matchup = sampler.build_scenario(scenario.id, game_seed)
            matchup = type(matchup)(**{**vars(matchup), "id": matchup_id})
            environment.matchups[matchup_id] = matchup
            decisions = 0
            memories: dict[str, torch.Tensor] = {}
            try:
                step = environment.reset(matchup_id, game_seed, False)
                while not step.done:
                    if decisions >= 256:
                        raise RuntimeError(
                            "curriculum evaluation exceeded 256 policy decisions"
                        )
                    graph = encoder.encode(
                        step.state,
                        step.player_id or matchup.learner_player_id,
                    )
                    observer_id = step.player_id or matchup.learner_player_id
                    actions = encoder.encode_actions(step.actions, step.state, observer_id)
                    with torch.no_grad():
                        logits, _, memory = model.policy_value_with_memory(
                            _single_batch(graph, actions, device, memories.get(observer_id))
                        )
                    memories[observer_id] = memory.detach()
                    step = environment.step(int(logits.argmax(dim=-1).item()))
                    decisions += 1
                milestone_total = max(
                    len(matchup.success_action_sequence),
                    environment.opening_hand_roles_available,
                )
                sideboard_in_mastery = (
                    environment.sideboard_target_cards_selected
                    / environment.sideboard_target_cards_available
                    if environment.sideboard_target_cards_available
                    else None
                )
                sideboard_out_mastery = (
                    environment.sideboard_cut_cards_selected
                    / environment.sideboard_cut_cards_expected
                    if environment.sideboard_cut_cards_expected
                    else None
                )
                mastery = (
                    (sideboard_in_mastery + sideboard_out_mastery) / 2
                    if sideboard_in_mastery is not None
                    and sideboard_out_mastery is not None
                    else sideboard_in_mastery
                    if sideboard_in_mastery is not None
                    else environment.objective_milestones_completed / milestone_total
                    if milestone_total
                    else environment.objective_damage_progress
                    if matchup.objective in {"fast-win", "combo-win"}
                    else float(step.reward > 0)
                )
                state = dict((environment.current_view or {}).get("state", {}))
                results.append(
                    {
                        "scenarioId": scenario.id,
                        "scenarioVersion": scenario.revision,
                        "family": scenario.family,
                        "seed": game_seed,
                        "attempt": attempt + 1,
                        "recoveredErrors": attempt_errors,
                        "success": bool(step.reward > 0),
                        "mastery": mastery,
                        "milestonesCompleted": environment.objective_milestones_completed,
                        "milestonesTotal": milestone_total,
                        "openingHandRolesCompleted": (
                            environment.objective_milestones_completed
                            if matchup.opening_hand_target_roles
                            else 0
                        ),
                        "openingHandRolesAvailable": environment.opening_hand_roles_available,
                        "objectiveDamageProgress": (
                            environment.objective_damage_progress
                            if matchup.objective in {"fast-win", "combo-win"}
                            and matchup.training_anchor_player_ids
                            else None
                        ),
                        "sideboardCardsSelected": environment.sideboard_cards_selected,
                        "sideboardTargetCardsSelected": (
                            environment.sideboard_target_cards_selected
                        ),
                        "sideboardTargetCardsAvailable": (
                            environment.sideboard_target_cards_available
                        ),
                        "sideboardCutCardsSelected": environment.sideboard_cut_cards_selected,
                        "sideboardCutCardsExpected": environment.sideboard_cut_cards_expected,
                        "turnNumber": state.get("turnNumber"),
                        "decisions": decisions,
                    }
                )
                print(
                    json.dumps(
                        {
                            "event": "curriculum-evaluation-scenario-finished",
                            "trainingStep": training_step,
                            "scenarioId": scenario.id,
                            "scenarioIndex": index + 1,
                            "scenarioCount": len(sampler.scenarios),
                            "attempt": attempt + 1,
                            "success": bool(step.reward > 0),
                            "mastery": mastery,
                            "decisions": decisions,
                            "seconds": time.perf_counter() - attempt_started,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                break
            except (RuntimeError, httpx.HTTPError) as error:
                error_record = {
                    "attempt": attempt + 1,
                    "seed": game_seed,
                    "error": f"{type(error).__name__}: {error}",
                }
                attempt_errors.append(error_record)
                print(
                    json.dumps(
                        {
                            "event": "curriculum-evaluation-scenario-retry"
                            if attempt < 2
                            else "curriculum-evaluation-scenario-failed",
                            "trainingStep": training_step,
                            "scenarioId": scenario.id,
                            "scenarioIndex": index + 1,
                            "scenarioCount": len(sampler.scenarios),
                            "attempt": attempt + 1,
                            "maxAttempts": 3,
                            "seed": game_seed,
                            "error": error_record["error"],
                            "seconds": time.perf_counter() - attempt_started,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                if attempt == 2:
                    failures.append(
                        {
                            "scenarioId": scenario.id,
                            "scenarioVersion": scenario.revision,
                            "seed": base_game_seed,
                            "attempts": len(attempt_errors),
                            "errors": attempt_errors,
                        }
                    )
            finally:
                environment.matchups.pop(matchup_id, None)
    model.train()
    completed = len(results)
    return {
        "schemaVersion": "v13-curriculum-evaluation/v1",
        "trainingStep": training_step,
        "completedEpisodes": completed_episodes,
        "fixedSeeds": True,
        "summary": {
            "expectedScenarios": len(sampler.scenarios),
            "completedScenarios": completed,
            "successes": sum(bool(result["success"]) for result in results),
            "successRate": (
                sum(bool(result["success"]) for result in results) / completed
                if completed
                else 0.0
            ),
            "meanMastery": (
                sum(float(result["mastery"]) for result in results) / completed
                if completed
                else 0.0
            ),
            "failedScenarios": len(failures),
            "recoveredScenarios": sum(int(result.get("attempt", 1)) > 1 for result in results),
            "engineErrors": sum(
                len(result.get("recoveredErrors", [])) for result in results
            )
            + sum(len(failure.get("errors", [])) for failure in failures),
        },
        "scenarios": results,
        "failures": failures,
        "evaluationSeconds": time.perf_counter() - started,
    }


def _load_model(
    output: Path, model_config: ModelConfigV13, device: torch.device
) -> tuple[GraphBeliefWorldModelV13, int, int, dict[str, Any] | None]:
    model = GraphBeliefWorldModelV13(model_config).to(device)
    rl_checkpoint, payload = _latest_readable_checkpoint(
        output / "rl-checkpoints", "update-*/rl-model.pt", device
    )
    checkpoint = rl_checkpoint
    if checkpoint is None:
        checkpoint, payload = _latest_readable_checkpoint(
            output / "checkpoints", "step-*/world-model.pt", device
        )
    completed_episodes = 0
    update = 0
    if checkpoint is not None and payload is not None:
        saved_config = payload.get("model_config")
        if saved_config != asdict(model_config):
            raise ValueError("V13 checkpoint model dimensions do not match the RL configuration")
        if payload.get("observation_schema") != model.observation_schema:
            raise ValueError("V13 checkpoint observation schema is not compatible with V13.1")
        missing, unexpected = model.load_state_dict(payload["model"], strict=False)
        allowed_missing = {
            name for name in missing if name.startswith(("rl_policy_head.", "rl_value_head."))
        }
        if set(missing) != allowed_missing or unexpected:
            raise ValueError(
                f"Incompatible V13 checkpoint (missing={missing}, unexpected={unexpected})"
            )
        if rl_checkpoint is not None:
            completed_episodes = int(payload.get("completed_episodes", 0))
            update = int(payload.get("update", 0))
        print(
            json.dumps(
                {
                    "event": "rl-resumed" if rl_checkpoint else "world-model-loaded",
                    "checkpoint": str(checkpoint),
                    "update": update,
                    "completed_episodes": completed_episodes,
                    "new_rl_parameters": sorted(allowed_missing),
                }
            ),
            flush=True,
        )
    else:
        raise FileNotFoundError("Train a V13 world-model checkpoint before starting RL")
    return model, completed_episodes, update, payload if rl_checkpoint is not None else None


def train_rl(config_path: Path, output_override: Path | None = None) -> Path:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    model_config = ModelConfigV13(**raw.get("model", {}))
    config = V13RLTrainingConfig(**raw.get("rl", {}))
    output = (
        output_override
        if output_override is not None
        else Path(raw.get("outputDir", ".deepdeck/runs/v13-world-model"))
    ).resolve()
    output.mkdir(parents=True, exist_ok=True)
    learner_settings = raw.get("learnerSettings") or {}
    agent_id = str(learner_settings.get("modelId", output.name))
    display_name = str(learner_settings.get("modelName", raw.get("name", agent_id)))
    training_bridge = TrainingBridge(
        output,
        v13_training_contract(agent_id, display_name),
    )
    control_revision = 0
    saved_game_limit = int(raw.get("savedGameLimit", 20))
    trace_training_actions = bool(raw.get("traceTrainingActions", False))

    def next_control_action() -> str | None:
        nonlocal control_revision, saved_game_limit
        while True:
            control = training_bridge.requested_control()
            revision = int((control or {}).get("revision", 0) or 0)
            if revision <= control_revision:
                return None
            control_revision = revision
            parameters = control.get("parameters", {}) if control else {}
            if isinstance(parameters, dict) and "savedGameLimit" in parameters:
                saved_game_limit = max(0, min(500, int(parameters["savedGameLimit"])))
            action = str((control or {}).get("action", ""))
            if action != "pause":
                return action
            _report_with_retry(
                training_bridge.report_state,
                phase="reinforcement-learning",
                status="paused",
                step=update,
                control_revision=control_revision,
                details={"completedEpisodes": completed_episodes},
            )
            while True:
                time.sleep(1.0)
                resumed = training_bridge.requested_control()
                resumed_revision = int((resumed or {}).get("revision", 0) or 0)
                if resumed_revision <= control_revision:
                    continue
                control_revision = resumed_revision
                return str((resumed or {}).get("action", ""))
    device_name = config.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    model, completed_episodes, update, resume_payload = _load_model(
        output, model_config, device
    )
    _prune_rl_checkpoints(output / "rl-checkpoints", config.max_checkpoints)
    curriculum_config = raw.get("trainingCurriculum", {})
    curriculum_enabled = bool(
        isinstance(curriculum_config, dict) and curriculum_config.get("enabled", False)
    )
    data_source = (
        "engine-focused-curriculum-v1"
        if config.environment == "engine" and curriculum_enabled
        else "engine-self-play-legacy-v1"
        if config.environment == "engine"
        else "tiny-self-play-v1"
    )
    if config.environment not in {"tiny", "engine"}:
        raise ValueError("rl.environment must be tiny or engine")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    resume_source = str((resume_payload or {}).get("data_source", "tiny-self-play-v1"))
    if (
        resume_payload is not None
        and "optimizer" in resume_payload
        and resume_source == data_source
    ):
        optimizer.load_state_dict(resume_payload["optimizer"])
        for group in optimizer.param_groups:
            group["lr"] = config.learning_rate
    encoder = GraphObservationEncoderV13(
        feature_dim=model_config.node_feature_dim,
        action_feature_dim=model_config.action_dim,
    )
    matchup_sampler: RandomTrainingMatchupSampler | AdaptiveCurriculumSampler | None = None
    evaluation_sampler: RandomTrainingMatchupSampler | None = None
    evaluation_environment: RustSelfPlayEnvironment | None = None
    curriculum_evaluation_sampler: AdaptiveCurriculumSampler | None = None
    curriculum_evaluation_environment: RustSessionEnvironment | None = None
    if config.environment == "engine":
        decks = _deck_catalog(raw)
        if not decks:
            raise ValueError("Engine self-play requires a non-empty deckCatalog")
        randomizer_config = raw.get("trainingScenarioRandomizer") or {
            "formats": ["legacy"],
            "playerCounts": [2],
            "maxTurns": 80,
        }
        evaluation_sampler = RandomTrainingMatchupSampler(randomizer_config, decks)
        engine_url = str(raw.get("engineUrl", "http://127.0.0.1:8787"))
        engine_timeout = float(raw.get("engineTimeoutSeconds", 300))
        evaluation_sampler.validate_commander_decks(engine_url, engine_timeout)
        matchup_sampler = (
            AdaptiveCurriculumSampler(curriculum_config, decks, evaluation_sampler)
            if curriculum_enabled
            else evaluation_sampler
        )
        if isinstance(matchup_sampler, AdaptiveCurriculumSampler):
            matchup_sampler.load_history(output / "v13-engine-games.jsonl")
        template = matchup_sampler.template()
        environment: RustSelfPlayEnvironment | RustSessionEnvironment | TinySelfPlayEnvironment = (
            RustSessionEnvironment(
                engine_url,
                {template.id: template},
                engine_timeout,
                str(raw.get("analyticsPilotId", "v13-engine-self-play")),
            )
            if curriculum_enabled
            else RustSelfPlayEnvironment(
                engine_url,
                {template.id: template},
                engine_timeout,
                "winnerLoser",
                str(raw.get("analyticsPilotId", "v13-engine-self-play")),
            )
        )
        environment.capture_replay = saved_game_limit > 0
        evaluation_template = evaluation_sampler.template()
        evaluation_environment = (
            RustSelfPlayEnvironment(
                engine_url,
                {evaluation_template.id: evaluation_template},
                engine_timeout,
                "winnerLoser",
                str(raw.get("analyticsPilotId", "v13-engine-self-play")),
            )
            if curriculum_enabled
            else environment
        )
        if curriculum_enabled:
            curriculum_evaluation_sampler = AdaptiveCurriculumSampler(
                {**curriculum_config, "adaptive": False},
                decks,
                evaluation_sampler,
            )
            curriculum_evaluation_template = curriculum_evaluation_sampler.template()
            curriculum_evaluation_environment = RustSessionEnvironment(
                engine_url,
                {
                    curriculum_evaluation_template.id: curriculum_evaluation_template,
                },
                engine_timeout,
                str(raw.get("analyticsPilotId", "v13-engine-self-play"))
                + "-curriculum-evaluation",
            )
    else:
        environment = TinySelfPlayEnvironment(horizon=config.horizon)
    baseline: GraphBeliefWorldModelV13 | None = None
    baseline_path = output / "v13-engine-baseline.pt"
    if config.environment == "engine":
        baseline = deepcopy(model).eval()
        if baseline_path.is_file():
            baseline_payload = torch.load(
                baseline_path, map_location=device, weights_only=False
            )
            if baseline_payload.get("observation_schema") != model.observation_schema:
                raise ValueError("V13 baseline observation schema is not compatible with V13.1")
            baseline.load_state_dict(baseline_payload["model"])
        else:
            torch.save(
                {
                    "model": baseline.state_dict(),
                    "model_config": asdict(model_config),
                    "observation_schema": model.observation_schema,
                },
                baseline_path,
            )
    metrics_path = output / "v13-rl-metrics.jsonl"
    state_path = output / "v13-training-state.json"
    metadata_path = output / "local-model.json"
    games_path = output / "v13-engine-games.jsonl"
    evaluations_path = output / "v13-engine-evaluations.jsonl"
    curriculum_evaluations_path = output / "v13-curriculum-evaluations.jsonl"
    live_started = [0]
    live_decisions = [0]
    last_live_write = [0.0]

    def report_engine_progress(view: dict[str, Any]) -> None:
        now = time.monotonic()
        live_decisions[0] += 1
        if now - last_live_write[0] < 0.75:
            return
        last_live_write[0] = now
        state = view.get("state", {})
        matchup = getattr(environment, "current_matchup", None)
        players_state = [
            {
                "id": player.get("id"),
                "name": player.get("name"),
                "life": player.get("life"),
                "hasLost": player.get("hasLost", False),
                "handCount": len(player.get("hand", [])),
                "battlefieldCount": len(player.get("battlefield", [])),
            }
            for player in state.get("players", [])
            if isinstance(player, dict)
        ]
        turn_number = int(state.get("turnNumber", 0) or 0)
        _write_json_atomic(
            state_path,
            {
                "status": "running",
                "desiredState": "running",
                "processId": os.getpid(),
                "trainingPhase": "collecting-engine-games",
                "trainingStep": update,
                "completed_episodes": completed_episodes,
                "activeAttempts": [
                    {
                        "sessionId": getattr(environment, "session_id", None),
                        "worker": 1,
                        "status": "collecting",
                        "opponentMode": (
                            "predictable-builtin"
                            if getattr(matchup, "scenario_id", None)
                            else "self"
                        ),
                        "gameMode": getattr(matchup, "game_mode", "legacy"),
                        "decks": list(getattr(matchup, "deck_names", ())),
                        "players": len(players_state),
                        "playersState": players_state,
                        "turnNumber": turn_number,
                        "roundNumber": (turn_number + 1) // 2 if turn_number else None,
                        "decisions": live_decisions[0],
                        "startedAtUnixMs": live_started[0],
                        "updatedAtUnixMs": int(time.time() * 1000),
                    }
                ],
                "updatedAtUnixMs": int(time.time() * 1000),
            },
        )

    if isinstance(environment, RustSessionEnvironment):
        environment.progress_callback = report_engine_progress
    started = time.perf_counter()
    total_collection_seconds = 0.0
    total_training_seconds = 0.0
    latest: dict[str, Any] = {
        "trainingStep": update,
        "completed_episodes": completed_episodes,
        "data_source": data_source,
    }
    consecutive_errors = 0
    consecutive_connection_errors = 0
    total_failed_games = 0
    stopped_by_control = False

    while completed_episodes < config.episodes:
        requested_action = next_control_action()
        if requested_action == "stop":
            stopped_by_control = True
            break
        force_checkpoint = requested_action == "checkpoint"
        force_evaluation = requested_action == "evaluate"
        rollout_target = min(
            config.rollout_episodes, config.episodes - completed_episodes
        )
        collection_started = time.perf_counter()
        transitions: list[_Transition] = []
        episode_rewards: list[float] = []
        episode_lengths: list[int] = []
        episode_durations: list[float] = []
        rollout_failed_games = 0
        for offset in range(rollout_target):
            episode_number = completed_episodes + offset
            while True:
                episode_started = time.perf_counter()
                episode_seed = (
                    config.seed + episode_number + total_failed_games * 1_000_003
                )
                matchup = None
                matchup_id = "v13-rl"
                if matchup_sampler is not None:
                    matchup = matchup_sampler.sample(random.Random(episode_seed))
                    matchup_id = matchup.id
                    environment.matchups[matchup_id] = matchup
                live_started[0] = int(time.time() * 1000)
                live_decisions[0] = 0
                trajectory: list[_Transition] = []
                memories: dict[str, torch.Tensor] = {}
                try:
                    step = environment.reset(matchup_id, episode_seed, False)
                    print(
                        json.dumps(
                            {
                                "event": "rl-episode-start",
                                "episode": episode_number + 1,
                                "seed": episode_seed,
                                "scenarioId": getattr(matchup, "scenario_id", None),
                            }
                        ),
                        flush=True,
                    )
                    while not step.done:
                        if not step.actions:
                            raise RuntimeError(
                                "RL environment returned a live state without legal actions"
                            )
                        observer_id = step.player_id or "learner"
                        graph = encoder.encode(step.state, observer_id)
                        action_features = encoder.encode_actions(
                            step.actions, step.state, observer_id
                        )
                        previous_memory = memories.get(observer_id)
                        batch = _single_batch(
                            graph, action_features, device, previous_memory
                        )
                        with torch.no_grad():
                            logits, value, next_memory = model.policy_value_with_memory(batch)
                            distribution = Categorical(logits=logits)
                            action = (
                                logits.argmax(dim=-1)
                                if config.greedy_actions
                                else distribution.sample()
                            )
                            log_probability = distribution.log_prob(action)
                        memories[observer_id] = next_memory.detach()
                        if trace_training_actions:
                            selected_action = step.actions[int(action.item())]
                            print(
                                json.dumps(
                                    {
                                        "event": "rl-action",
                                        "episode": episode_number + 1,
                                        "decision": len(trajectory) + 1,
                                        "decisionKind": (
                                            (environment.current_view or {})
                                            .get("decision", {})
                                            .get("kind")
                                        ),
                                        "actionId": selected_action.get("id"),
                                        "actionLabel": selected_action.get("label"),
                                    }
                                ),
                                flush=True,
                            )
                        next_step = environment.step(int(action.item()))
                        if (
                            not next_step.done
                            and len(trajectory) + 1 >= config.max_episode_decisions
                        ):
                            if isinstance(environment, RustSessionEnvironment):
                                environment._remove_session()
                            print(
                                json.dumps(
                                    {
                                        "event": "rl-fragment-truncated",
                                        "episode": episode_number + 1,
                                        "decisions": len(trajectory) + 1,
                                    }
                                ),
                                flush=True,
                            )
                            next_step = DecisionStep(
                                state=next_step.state,
                                actions=[],
                                reward=float(next_step.reward) - 0.25,
                                done=True,
                                player_id=next_step.player_id,
                                rewards_by_player=next_step.rewards_by_player,
                            )
                        trajectory.append(
                            _Transition(
                                graph=graph,
                                action_features=action_features,
                                action_index=int(action.item()),
                                old_log_probability=float(log_probability.item()),
                                value=float(value.item()),
                                reward=float(next_step.reward),
                                done=next_step.done,
                                recurrent_state=(
                                    previous_memory.detach().cpu()
                                    if previous_memory is not None
                                    else torch.zeros(model.config.latent_dim)
                                ),
                                player_id=observer_id,
                            )
                        )
                        if next_step.rewards_by_player:
                            for player_id, player_reward in (
                                next_step.rewards_by_player.items()
                            ):
                                previous = next(
                                    (
                                        item
                                        for item in reversed(trajectory)
                                        if item.player_id == player_id
                                    ),
                                    None,
                                )
                                if previous is not None:
                                    previous.reward += float(player_reward)
                        step = next_step
                except (RuntimeError, httpx.HTTPError) as error:
                    rollout_failed_games += 1
                    total_failed_games += 1
                    connection_failure = _engine_connection_failure(error)
                    if connection_failure:
                        consecutive_connection_errors += 1
                    else:
                        consecutive_errors += 1
                        consecutive_connection_errors = 0
                    retry_delay = (
                        min(
                            5.0,
                            0.25
                            * 2
                            ** min(consecutive_connection_errors - 1, 4),
                        )
                        if connection_failure
                        else 0.0
                    )
                    failure = {
                        "episode": episode_number + 1,
                        "seed": episode_seed,
                        "attempt": total_failed_games,
                        "consecutiveErrors": consecutive_errors,
                        "consecutiveConnectionErrors": consecutive_connection_errors,
                        "maxConsecutiveErrors": config.max_consecutive_errors,
                        "category": (
                            "engine-connection" if connection_failure else "engine-game"
                        ),
                        "retryDelaySeconds": retry_delay,
                        "decks": list(matchup.deck_names) if matchup is not None else [],
                        "error": f"{type(error).__name__}: {error}",
                    }
                    print(json.dumps({"trainingError": failure}), flush=True)
                    _write_json_atomic(
                        state_path,
                        {
                            "status": "running",
                            "desiredState": "running",
                            "trainingPhase": "recovering-engine-game",
                            "trainingStep": update,
                            "completed_episodes": completed_episodes,
                            "failedGames": total_failed_games,
                            "latestError": failure,
                            "activeAttempts": [],
                            "updatedAtUnixMs": int(time.time() * 1000),
                            "processId": os.getpid(),
                        },
                    )
                    if connection_failure:
                        time.sleep(retry_delay)
                        continue
                    if consecutive_errors >= config.max_consecutive_errors:
                        raise RuntimeError(
                            "V13 Engine self-play exceeded "
                            f"{config.max_consecutive_errors} consecutive failed games"
                        ) from error
                    continue
                consecutive_errors = 0
                consecutive_connection_errors = 0
                break
            if matchup_sampler is not None and not curriculum_enabled:
                _self_play_advantages(trajectory, config)
            else:
                _advantages(trajectory, config)
            transitions.extend(trajectory)
            player_rewards = step.rewards_by_player or {"learner": float(step.reward)}
            episode_rewards.append(
                float(
                    player_rewards.get(
                        "player-1",
                        next(iter(player_rewards.values()), float(step.reward)),
                    )
                )
            )
            episode_lengths.append(len(trajectory))
            episode_seconds = time.perf_counter() - episode_started
            episode_durations.append(episode_seconds)
            print(
                json.dumps(
                    {
                        "event": "rl-episode-collected",
                        "episode": episode_number + 1,
                        "decisions": len(trajectory),
                        "seconds": episode_seconds,
                    }
                ),
                flush=True,
            )
            if matchup is not None and isinstance(environment, RustSessionEnvironment):
                state, terminal_rewards = _terminal_result(
                    environment,
                    matchup,
                    learner_reward=float(step.reward) if curriculum_enabled else None,
                )
                outcome = state.get("outcome") or {}
                player_ids = [
                    str(player["id"]) for player in matchup.setup.get("players", [])
                ]
                deck_by_player = {
                    player_id: matchup.deck_names[index]
                    for index, player_id in enumerate(player_ids)
                }
                record = {
                    "episode": episode_number + 1,
                    "trainingStep": update + 1,
                    "seed": episode_seed,
                    "matchupId": matchup.id,
                    "gameMode": matchup.game_mode,
                    "players": len(player_ids),
                    "decks": list(matchup.deck_names),
                    "deckByPlayer": deck_by_player,
                    "decisions": len(trajectory),
                    "rewardsByPlayer": terminal_rewards,
                    "turnNumber": state.get("turnNumber"),
                    "gameStatus": (
                        "objectiveComplete"
                        if environment.objective_completed
                        else state.get("status")
                    ),
                    "outcome": outcome,
                    "gameDurationSeconds": episode_seconds,
                    "completedAtUnixMs": int(time.time() * 1000),
                    "data_source": data_source,
                    "scenarioId": matchup.scenario_id,
                    "scenarioFamily": matchup.scenario_family,
                    "scenarioVersion": matchup.scenario_version,
                    "objective": matchup.objective,
                    "difficulty": matchup.difficulty,
                    "learnerPlayerId": matchup.learner_player_id,
                    "objectiveCompleted": environment.objective_completed,
                    "objectiveMilestonesCompleted": (
                        environment.objective_milestones_completed
                    ),
                    "objectiveMilestonesTotal": max(
                        len(matchup.success_action_sequence),
                        environment.opening_hand_roles_available,
                    ),
                    "openingHandRolesCompleted": environment.objective_milestones_completed
                    if matchup.opening_hand_target_roles
                    else 0,
                    "openingHandRolesAvailable": environment.opening_hand_roles_available,
                    "objectiveDamageProgress": (
                        environment.objective_damage_progress
                        if matchup.objective in {"fast-win", "combo-win"}
                        and matchup.training_anchor_player_ids
                        else None
                    ),
                    "sideboardCardsSelected": environment.sideboard_cards_selected,
                    "sideboardTargetCardsSelected": (
                        environment.sideboard_target_cards_selected
                    ),
                    "sideboardTargetCardsAvailable": (
                        environment.sideboard_target_cards_available
                    ),
                    "sideboardCutCardsSelected": environment.sideboard_cut_cards_selected,
                    "sideboardCutCardsExpected": environment.sideboard_cut_cards_expected,
                }
                if isinstance(matchup_sampler, AdaptiveCurriculumSampler):
                    scenario_success = float(step.reward) > 0
                    milestone_total = max(
                        len(matchup.success_action_sequence),
                        environment.opening_hand_roles_available,
                    )
                    sideboard_in_mastery = (
                        environment.sideboard_target_cards_selected
                        / environment.sideboard_target_cards_available
                        if environment.sideboard_target_cards_available
                        else None
                    )
                    sideboard_out_mastery = (
                        environment.sideboard_cut_cards_selected
                        / environment.sideboard_cut_cards_expected
                        if environment.sideboard_cut_cards_expected
                        else None
                    )
                    scenario_mastery = (
                        (sideboard_in_mastery + sideboard_out_mastery) / 2
                        if sideboard_in_mastery is not None
                        and sideboard_out_mastery is not None
                        else sideboard_in_mastery
                        if sideboard_in_mastery is not None
                        else environment.objective_milestones_completed / milestone_total
                        if milestone_total
                        else environment.objective_damage_progress
                        if matchup.objective in {"fast-win", "combo-win"}
                        else float(scenario_success)
                    )
                    matchup_sampler.observe(
                        str(matchup.scenario_id),
                        scenario_success,
                        scenario_mastery,
                    )
                    record["curriculum"] = matchup_sampler.stats()
                _append(games_path, record)
                _save_replay(
                    output,
                    environment,
                    record,
                    saved_game_limit,
                )
        collection_seconds = time.perf_counter() - collection_started
        total_collection_seconds += collection_seconds
        print(
            json.dumps(
                {
                    "event": "rl-update-start",
                    "update": update + 1,
                    "transitions": len(transitions),
                }
            ),
            flush=True,
        )

        advantages = torch.tensor([item.advantage for item in transitions])
        advantage_mean = float(advantages.mean())
        advantage_std = float(advantages.std(unbiased=False))
        denominator = max(advantage_std, 1e-8)
        for item in transitions:
            item.advantage = (item.advantage - advantage_mean) / denominator

        training_started = time.perf_counter()
        totals = {
            "loss": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "clip_fraction": 0.0,
            "gradient_norm": 0.0,
        }
        minibatches = 0
        stopped_for_kl = False
        for _ in range(config.epochs):
            indices = torch.randperm(len(transitions)).tolist()
            for start in range(0, len(indices), config.minibatch_size):
                selected = indices[start : start + config.minibatch_size]
                batch, action_indices, old_log_probabilities, batch_advantages, returns = (
                    _minibatch(transitions, selected, device)
                )
                logits, values = model.policy_value(batch)
                distribution = Categorical(logits=logits)
                log_probabilities = distribution.log_prob(action_indices)
                ratio = (log_probabilities - old_log_probabilities).exp()
                unclipped = ratio * batch_advantages
                clipped = ratio.clamp(1.0 - config.clip_ratio, 1.0 + config.clip_ratio) * batch_advantages
                policy_loss = -torch.minimum(unclipped, clipped).mean()
                value_loss = torch.nn.functional.mse_loss(values, returns)
                entropy = distribution.entropy().mean()
                loss = (
                    policy_loss
                    + config.value_coefficient * value_loss
                    - config.entropy_coefficient * entropy
                )
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("non-finite V13 PPO loss")
                approx_kl = (old_log_probabilities - log_probabilities).mean()
                if (
                    config.target_kl > 0
                    and float(approx_kl.detach().cpu()) > config.target_kl
                ):
                    stopped_for_kl = True
                    break
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.max_grad_norm
                )
                optimizer.step()
                measurements = {
                    "loss": loss,
                    "policy_loss": policy_loss,
                    "value_loss": value_loss,
                    "entropy": entropy,
                    "approx_kl": approx_kl,
                    "clip_fraction": ((ratio - 1.0).abs() > config.clip_ratio).float().mean(),
                    "gradient_norm": gradient_norm,
                }
                for name, value in measurements.items():
                    totals[name] += float(value.detach().cpu())
                minibatches += 1
            if stopped_for_kl:
                break
        training_seconds = time.perf_counter() - training_started
        total_training_seconds += training_seconds
        print(
            json.dumps(
                {
                    "event": "rl-update-finished",
                    "update": update + 1,
                    "seconds": training_seconds,
                }
            ),
            flush=True,
        )
        update += 1
        completed_episodes += rollout_target
        means = {name: value / max(1, minibatches) for name, value in totals.items()}
        mean_reward = sum(episode_rewards) / len(episode_rewards)
        elapsed = time.perf_counter() - started
        ppo = {
            **means,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        latest = {
            "episode": completed_episodes,
            "trainingStep": update,
            **means,
            "episode_reward": mean_reward,
            "minimum_episode_reward": min(episode_rewards),
            "maximum_episode_reward": max(episode_rewards),
            "average_episode_length": sum(episode_lengths) / len(episode_lengths),
            "gameDurationSeconds": sum(episode_durations) / len(episode_durations),
            "gameDurationsSeconds": episode_durations,
            "decisions": len(transitions),
            "completed_episodes": completed_episodes,
            "elapsed_seconds": elapsed,
            "data_source": data_source,
            "engineGameMetrics": config.environment == "engine",
            "training_phase": "reinforcement-learning",
            "optimizer_minibatches": minibatches,
            "stopped_for_target_kl": stopped_for_kl,
            "ppo": ppo,
            "rolloutBatch": {
                "requestedGames": rollout_target + rollout_failed_games,
                "completedGames": rollout_target,
                "failedGames": rollout_failed_games,
                "totalDecisions": len(transitions),
                "collectionWallSeconds": collection_seconds,
                "trainingWallSeconds": training_seconds,
            },
        }
        _append(metrics_path, latest)
        _report_with_retry(
            training_bridge.report_metrics,
            phase="reinforcement-learning",
            step=update,
            metrics={
                key: float(latest[key])
                for key in (
                    "loss",
                    "policy_loss",
                    "value_loss",
                    "entropy",
                    "approx_kl",
                    "episode_reward",
                )
            },
            context={
                "completedEpisodes": completed_episodes,
                "dataSource": data_source,
                "rolloutBatch": latest["rolloutBatch"],
            },
        )
        evaluation: dict[str, Any] | None = None
        if (
            baseline is not None
            and matchup_sampler is not None
            and (
                force_evaluation
                or (
                    config.evaluation_every_updates > 0
                    and update % config.evaluation_every_updates == 0
                )
            )
        ):
            assert evaluation_environment is not None
            assert evaluation_sampler is not None
            evaluation = _run_engine_evaluation(
                model=model,
                baseline=baseline,
                encoder=encoder,
                environment=evaluation_environment,
                sampler=evaluation_sampler,
                device=device,
                seed=config.seed + 10_000_000 + update * 10_000,
                games=max(2, config.evaluation_games),
                training_step=update,
                completed_episodes=completed_episodes,
            )
            _append(evaluations_path, evaluation)
            latest["engineEvaluation"] = evaluation["summary"]
            _report_with_retry(
                training_bridge.report_metrics,
                phase="engine-evaluation",
                step=update,
                metrics={
                    "win_rate": float(evaluation["summary"]["candidateWinRate"]),
                    "lower_95": float(evaluation["summary"]["lower95"]),
                },
                context={
                    "games": evaluation["summary"]["completedGames"],
                    "opponent": evaluation["opponentVersion"],
                    "fixedSeeds": evaluation["fixedSeeds"],
                },
            )
            if (
                curriculum_evaluation_environment is not None
                and curriculum_evaluation_sampler is not None
            ):
                curriculum_evaluation = _run_curriculum_evaluation(
                    model=model,
                    encoder=encoder,
                    environment=curriculum_evaluation_environment,
                    sampler=curriculum_evaluation_sampler,
                    device=device,
                    seed=config.seed + 20_000_000 + update * 10_000,
                    training_step=update,
                    completed_episodes=completed_episodes,
                )
                _append(curriculum_evaluations_path, curriculum_evaluation)
                latest["curriculumEvaluation"] = curriculum_evaluation["summary"]
                _report_with_retry(
                    training_bridge.report_metrics,
                    phase="engine-evaluation",
                    step=update,
                    metrics={
                        "curriculum_success_rate": float(
                            curriculum_evaluation["summary"]["successRate"]
                        ),
                        "curriculum_mastery": float(
                            curriculum_evaluation["summary"]["meanMastery"]
                        ),
                    },
                    context={
                        "scenarios": curriculum_evaluation["summary"][
                            "completedScenarios"
                        ],
                        "fixedSeeds": True,
                    },
                )
        state = {
            "status": "running",
            "desiredState": "running",
            "trainingPhase": "reinforcement-learning",
            "trainingStep": update,
            "completed_episodes": completed_episodes,
            "trainingElapsedSeconds": elapsed,
            "gameSimulationSeconds": total_collection_seconds,
            "modelTrainingSeconds": total_training_seconds,
            "updatedAtUnixMs": int(time.time() * 1000),
            "processId": os.getpid(),
            "activeAttempts": [],
            **latest,
        }
        _write_json_atomic(state_path, state)
        _report_with_retry(
            training_bridge.report_state,
            phase="reinforcement-learning",
            status="running",
            step=update,
            control_revision=control_revision,
            details={
                "completedEpisodes": completed_episodes,
                "activeGames": 0,
                "dataSource": data_source,
            },
        )
        print(json.dumps(latest, sort_keys=True), flush=True)

        staging_due = (
            config.staging_evaluation_every_updates > 0
            and update % config.staging_evaluation_every_updates == 0
        )
        if (
            force_checkpoint
            or update % config.checkpoint_every == 0
            or staging_due
            or completed_episodes == config.episodes
        ):
            checkpoint = output / "rl-checkpoints" / f"update-{update}"
            checkpoint.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "model_config": asdict(model_config),
                    "observation_schema": model.observation_schema,
                    "rl_config": asdict(config),
                    "update": update,
                    "completed_episodes": completed_episodes,
                    "data_source": data_source,
                },
                checkpoint / "rl-model.pt",
            )
            _prune_rl_checkpoints(output / "rl-checkpoints", config.max_checkpoints)
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                metadata = {}
            if isinstance(metadata, dict):
                metadata["checkpointPath"] = str(checkpoint)
                metadata["description"] = (
                    "V13 structured world-model agent fine-tuned with on-policy PPO "
                    + (
                        "on authoritative Legacy Engine games."
                        if config.environment == "engine"
                        else "self-play."
                    )
                )
                _write_json_atomic(metadata_path, metadata)
            if staging_due:
                staging_event = _launch_staging_evaluation(
                    checkpoint=checkpoint,
                    output=output,
                    update=update,
                    config=config,
                )
                latest["stagingEvaluation"] = staging_event

    _write_json_atomic(
        state_path,
        {
            "status": "stopped" if stopped_by_control else "completed",
            "desiredState": "stopped",
            "trainingPhase": "reinforcement-learning",
            "trainingStep": update,
            "completed_episodes": completed_episodes,
            "trainingElapsedSeconds": time.perf_counter() - started,
            "gameSimulationSeconds": total_collection_seconds,
            "modelTrainingSeconds": total_training_seconds,
            "updatedAtUnixMs": int(time.time() * 1000),
            "processId": os.getpid(),
            "activeAttempts": [],
            **latest,
        },
    )
    _report_with_retry(
        training_bridge.report_state,
        phase="reinforcement-learning",
        status="idle" if stopped_by_control else "completed",
        step=update,
        control_revision=control_revision,
        details={"completedEpisodes": completed_episodes, "dataSource": data_source},
    )
    if isinstance(environment, RustSessionEnvironment):
        environment.close()
    if evaluation_environment is not None and evaluation_environment is not environment:
        evaluation_environment.close()
    if curriculum_evaluation_environment is not None:
        curriculum_evaluation_environment.close()
    return output


def evaluate_rl(config_path: Path, output_override: Path | None = None) -> Path:
    """Run a bounded Engine evaluation without starting another training rollout."""

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    model_config = ModelConfigV13(**raw.get("model", {}))
    config = V13RLTrainingConfig(**raw.get("rl", {}))
    if config.environment != "engine":
        raise ValueError("V13 standalone evaluation requires rl.environment=engine")
    output = (
        output_override
        if output_override is not None
        else Path(raw.get("outputDir", ".deepdeck/runs/v13-world-model"))
    ).resolve()
    device_name = config.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    model, completed_episodes, update, _ = _load_model(output, model_config, device)
    model.eval()
    baseline_path = output / "v13-engine-baseline.pt"
    if not baseline_path.is_file():
        raise FileNotFoundError("Run V13 Engine self-play once to create its frozen baseline")
    baseline = GraphBeliefWorldModelV13(model_config).to(device).eval()
    baseline.load_state_dict(
        torch.load(baseline_path, map_location=device, weights_only=False)["model"]
    )
    decks = _deck_catalog(raw)
    if not decks:
        raise ValueError("Engine evaluation requires a non-empty deckCatalog")
    randomizer_config = raw.get("trainingScenarioRandomizer") or {
        "formats": ["legacy"],
        "playerCounts": [2],
        "maxTurns": 80,
    }
    sampler = RandomTrainingMatchupSampler(randomizer_config, decks)
    engine_url = str(raw.get("engineUrl", "http://127.0.0.1:8787"))
    engine_timeout = float(raw.get("engineTimeoutSeconds", 300))
    sampler.validate_commander_decks(engine_url, engine_timeout)
    template = sampler.template()
    environment = RustSelfPlayEnvironment(
        engine_url,
        {template.id: template},
        engine_timeout,
        "winnerLoser",
        str(raw.get("analyticsPilotId", "v13-engine-evaluation")),
    )
    encoder = GraphObservationEncoderV13(
        feature_dim=model_config.node_feature_dim,
        action_feature_dim=model_config.action_dim,
    )
    learner_settings = raw.get("learnerSettings") or {}
    agent_id = str(learner_settings.get("modelId", output.name))
    display_name = str(learner_settings.get("modelName", raw.get("name", agent_id)))
    training_bridge = TrainingBridge(
        output,
        v13_training_contract(agent_id, display_name),
    )
    _report_with_retry(
        training_bridge.report_state,
        phase="engine-evaluation",
        status="running",
        step=update,
        details={"games": max(2, config.evaluation_games)},
    )
    try:
        evaluation = _run_engine_evaluation(
            model=model,
            baseline=baseline,
            encoder=encoder,
            environment=environment,
            sampler=sampler,
            device=device,
            seed=config.seed + 10_000_000 + update * 10_000,
            games=max(2, config.evaluation_games),
            training_step=update,
            completed_episodes=completed_episodes,
        )
        _append(output / "v13-engine-evaluations.jsonl", evaluation)
        _report_with_retry(
            training_bridge.report_metrics,
            phase="engine-evaluation",
            step=update,
            metrics={
                "win_rate": float(evaluation["summary"]["candidateWinRate"]),
                "lower_95": float(evaluation["summary"]["lower95"]),
            },
            context={
                "games": evaluation["summary"]["completedGames"],
                "opponent": evaluation["opponentVersion"],
                "fixedSeeds": evaluation["fixedSeeds"],
            },
        )
        _report_with_retry(
            training_bridge.report_state,
            phase="engine-evaluation",
            status="completed",
            step=update,
            details=evaluation["summary"],
        )
        print(json.dumps(evaluation, sort_keys=True), flush=True)
    finally:
        environment.close()
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune V13 with on-policy PPO")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    if args.evaluate_only:
        evaluate_rl(args.config, args.output)
    else:
        train_rl(args.config, args.output)


if __name__ == "__main__":
    main()
