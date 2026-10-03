from __future__ import annotations

import argparse
import json
import random
import time
from contextlib import suppress
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from deepdeck_agent import TrainingBridge

from oracle_ai.model_v13 import GraphBatch, GraphBeliefWorldModelV13, ModelConfigV13, v13_losses
from oracle_ai.training.contracts import v13_training_contract


@dataclass(frozen=True)
class WorldModelTrainingConfig:
    steps: int = 1000
    batch_size: int = 16
    nodes: int = 32
    learning_rate: float = 3e-4
    checkpoint_every: int = 100
    log_every: int = 1
    seed: int = 13
    device: str = "auto"
    resume: bool = True


LOSS_WEIGHTS = {
    "reconstruction_loss": 1.0,
    "dynamics_loss": 1.0,
    "kl_loss": 0.01,
    "belief_loss": 0.25,
    "opponent_loss": 0.25,
    "search_loss": 0.25,
    "value_loss": 0.5,
}


def synthetic_batch(
    config: ModelConfigV13, training: WorldModelTrainingConfig, device: torch.device
) -> GraphBatch:
    b, n = training.batch_size, training.nodes
    features = torch.randn(b, n, config.node_feature_dim, device=device)
    node_types = torch.randint(config.node_type_count, (b, n), device=device)
    owners = torch.randint(config.player_slots + 1, (b, n), device=device)
    mask = torch.rand(b, n, device=device).gt(0.12)
    mask[:, 0] = True
    actions = torch.randn(b, config.action_dim, device=device)
    action_effect = actions[:, : config.node_feature_dim].unsqueeze(1) * 0.08
    if action_effect.shape[-1] < config.node_feature_dim:
        action_effect = torch.nn.functional.pad(
            action_effect, (0, config.node_feature_dim - action_effect.shape[-1])
        )
    next_features = features + action_effect + torch.randn_like(features) * 0.02
    hidden = torch.stack(
        [(features[:, :, slot].mean(1) > 0).float() for slot in range(config.player_slots)], 1
    )
    opponent_action = node_types[:, 0].remainder(config.opponent_actions)
    search_policy = torch.nn.functional.one_hot(
        (opponent_action + 1).remainder(config.opponent_actions), config.opponent_actions
    ).float()
    outcome = (next_features.mean((1, 2)).mul(10).long() + 1).clamp(0, 2)
    return GraphBatch(
        features, node_types, owners, mask, actions, next_features, hidden,
        opponent_action, search_policy, outcome,
    )


def _append(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as destination:
        destination.write(json.dumps(value, sort_keys=True) + "\n")
        destination.flush()


def _write_json_atomic(path: Path, value: dict[str, Any]) -> bool:
    pending = path.with_suffix(".pending")
    pending.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for _ in range(10):
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


def _latest_checkpoint(output: Path) -> Path | None:
    candidates: list[tuple[int, Path]] = []
    for path in (output / "checkpoints").glob("step-*/world-model.pt"):
        try:
            step = int(path.parent.name.removeprefix("step-"))
        except ValueError:
            continue
        candidates.append((step, path))
    return max(candidates, default=(0, None), key=lambda item: item[0])[1]


def _update_checkpoint_metadata(metadata_path: Path, checkpoint: Path) -> None:
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    if not isinstance(metadata, dict):
        return
    metadata["checkpointPath"] = str(checkpoint)
    _write_json_atomic(metadata_path, metadata)


def train(config_path: Path, output_override: Path | None = None) -> Path:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    model_config = ModelConfigV13(**raw.get("model", {}))
    training = WorldModelTrainingConfig(**raw.get("training", {}))
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

    def next_control_action(current_step: int) -> str | None:
        nonlocal control_revision
        while True:
            control = training_bridge.requested_control()
            revision = int((control or {}).get("revision", 0) or 0)
            if revision <= control_revision:
                return None
            control_revision = revision
            action = str((control or {}).get("action", ""))
            if action != "pause":
                return action
            _report_with_retry(
                training_bridge.report_state,
                phase="world-model",
                status="paused",
                step=current_step,
                control_revision=control_revision,
            )
            while True:
                time.sleep(1.0)
                resumed = training_bridge.requested_control()
                resumed_revision = int((resumed or {}).get("revision", 0) or 0)
                if resumed_revision <= control_revision:
                    continue
                control_revision = resumed_revision
                return str((resumed or {}).get("action", ""))
    random.seed(training.seed)
    device_name = training.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    model = GraphBeliefWorldModelV13(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training.learning_rate)
    weights = {**LOSS_WEIGHTS, **raw.get("lossWeights", {})}
    metrics_path = output / "v13-metrics.jsonl"
    state_path = output / "v13-training-state.json"
    metadata_path = output / "local-model.json"
    if not metadata_path.exists():
        metadata_path.write_text(
            json.dumps(
                {
                    "id": output.name,
                    "name": str(raw.get("name", "V13 structured world model")),
                    "architecture": "v13",
                    "format": str(raw.get("format", "legacy")),
                    "description": "V13 structured Transformer world-model pretraining",
                    "createdAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "decks": [],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    (output / "resolved-v13-config.json").write_text(
        json.dumps(raw, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    start_step = 0
    if training.resume:
        checkpoint_path = _latest_checkpoint(output)
        if checkpoint_path is not None:
            checkpoint_data = torch.load(checkpoint_path, map_location=device, weights_only=False)
            if checkpoint_data.get("model_config") != asdict(model_config):
                raise ValueError("V13 resume checkpoint model dimensions do not match")
            if checkpoint_data.get("observation_schema") != model.observation_schema:
                raise ValueError("V13 checkpoint observation schema is not compatible with V13.2")
            model.load_state_dict(checkpoint_data["model"], strict=False)
            with suppress(ValueError):
                optimizer.load_state_dict(checkpoint_data["optimizer"])
            start_step = int(checkpoint_data.get("step", 0))
            for group in optimizer.param_groups:
                group["lr"] = training.learning_rate
            print(
                json.dumps({"event": "resumed", "step": start_step, "checkpoint": str(checkpoint_path)}),
                flush=True,
            )
    torch.manual_seed(training.seed + start_step)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(training.seed + start_step)
    previous_elapsed = 0.0
    try:
        previous_state = json.loads(state_path.read_text(encoding="utf-8"))
        previous_elapsed = float(previous_state.get("elapsed_seconds", 0.0))
    except (OSError, ValueError, TypeError, AttributeError):
        pass
    started = time.time()
    record: dict[str, Any] = {
        "step": start_step,
        "loss": None,
        "elapsed_seconds": previous_elapsed,
        "data_source": "synthetic-pretraining",
    }
    stopped_by_control = False
    for step in range(start_step + 1, training.steps + 1):
        requested_action = next_control_action(step - 1)
        if requested_action == "stop":
            stopped_by_control = True
            break
        force_checkpoint = requested_action == "checkpoint"
        batch = synthetic_batch(model_config, training, device)
        losses = v13_losses(batch, model(batch))
        total = sum(float(weights[name]) * loss for name, loss in losses.items())
        if not bool(torch.isfinite(total)):
            raise FloatingPointError(f"non-finite V13 loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not bool(torch.isfinite(gradient_norm)):
            raise FloatingPointError(f"non-finite V13 gradient at step {step}")
        optimizer.step()
        record = {
            "step": step,
            "loss": float(total.detach().cpu()),
            **{name: float(loss.detach().cpu()) for name, loss in losses.items()},
            "gradient_norm": float(gradient_norm.detach().cpu()),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "elapsed_seconds": previous_elapsed + time.time() - started,
            "data_source": "synthetic-pretraining",
        }
        if step % training.log_every == 0 or step == training.steps:
            _append(metrics_path, record)
            _report_with_retry(
                training_bridge.report_metrics,
                phase="world-model",
                step=step,
                metrics={name: float(record[name]) for name in LOSS_WEIGHTS},
                context={"dataSource": "synthetic-pretraining"},
            )
            print(json.dumps(record, sort_keys=True), flush=True)
        _write_json_atomic(state_path, {"status": "running", **record})
        _report_with_retry(
            training_bridge.report_state,
            phase="world-model",
            status="running",
            step=step,
            control_revision=control_revision,
            details={"dataSource": "synthetic-pretraining"},
        )
        if force_checkpoint or step % training.checkpoint_every == 0 or step == training.steps:
            checkpoint = output / "checkpoints" / f"step-{step}"
            checkpoint.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "model_config": asdict(model_config),
                 "observation_schema": model.observation_schema, "step": step},
                checkpoint / "world-model.pt",
            )
            _update_checkpoint_metadata(metadata_path, checkpoint)
    _write_json_atomic(
        state_path,
        {"status": "stopped" if stopped_by_control else "completed", **record},
    )
    _report_with_retry(
        training_bridge.report_state,
        phase="world-model",
        status="idle" if stopped_by_control else "completed",
        step=int(record["step"]),
        control_revision=control_revision,
        details={"dataSource": "synthetic-pretraining"},
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the V13 structured world model")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    train(args.config, args.output)


if __name__ == "__main__":
    main()
