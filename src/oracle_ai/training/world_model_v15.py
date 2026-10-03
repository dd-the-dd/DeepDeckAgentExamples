from __future__ import annotations

import argparse
import glob
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import torch
import yaml

from oracle_ai.encoding_v15 import GraphObservationEncoderV15
from oracle_ai.model_v15 import (
    ModelConfigV15,
    TransitionBatchV15,
    TransitionKind,
    TypedLatentWorldModelV15,
    v15_losses,
)


@dataclass(frozen=True)
class V15TrainingConfig:
    steps: int = 100
    batch_size: int = 8
    nodes: int = 24
    learning_rate: float = 3e-4
    checkpoint_every: int = 25
    seed: int = 15
    replay_paths: tuple[str, ...] = ()
    max_replay_files: int = 64
    max_transitions: int = 20_000
    synthetic_smoke: bool = False


LOSS_WEIGHTS = {
    "latent_transition_loss": 1.0,
    "invariant_loss": 0.35,
    "effect_loss": 0.5,
    "legality_loss": 0.25,
    "timing_loss": 0.25,
    "uncertainty_loss": 0.1,
    "sampled_reconstruction_loss": 0.2,
}


def synthetic_transition_batch(
    config: ModelConfigV15,
    training: V15TrainingConfig,
    device: torch.device,
) -> TransitionBatchV15:
    batch_size, nodes = training.batch_size, training.nodes
    features = torch.randn(batch_size, nodes, config.node_feature_dim, device=device)
    actions = torch.randn(batch_size, config.action_dim, device=device)
    kinds = torch.randint(config.transition_kinds, (batch_size,), device=device)
    effect = torch.zeros(batch_size, config.effect_dim, device=device)
    width = min(config.effect_dim, config.node_feature_dim, config.action_dim)
    effect[:, :width] = actions[:, :width] * 0.08
    next_features = features.clone()
    next_features[:, 0, :width] += effect[:, :width]
    invariants = torch.zeros(batch_size, config.invariant_dim, device=device)
    invariant_width = min(config.invariant_dim, config.node_feature_dim)
    invariants[:, :invariant_width] = next_features[:, :, :invariant_width].mean(1)
    reconstruction_mask = torch.rand(batch_size, nodes, device=device) < float(
        config.reconstruction_sample_rate
    )
    # Always sample at least one detailed token per batch so the expensive
    # decoder stays calibrated without running over every state token.
    reconstruction_mask[:, 0] = True
    return TransitionBatchV15(
        node_features=features,
        node_types=torch.randint(config.node_type_count, (batch_size, nodes), device=device),
        owners=torch.randint(config.player_slots + 1, (batch_size, nodes), device=device),
        node_mask=torch.ones(batch_size, nodes, dtype=torch.bool, device=device),
        action_features=actions,
        next_node_features=next_features,
        transition_kind=kinds,
        legal_label=(actions[:, 0] > 0).float(),
        timing_label=torch.randint(config.timing_classes, (batch_size,), device=device),
        effect_target=effect,
        invariant_target=invariants,
        controllers=torch.randint(config.player_slots + 1, (batch_size, nodes), device=device),
        zones=torch.randint(config.zone_count + 1, (batch_size, nodes), device=device),
        positions=torch.arange(nodes, device=device).unsqueeze(0).expand(batch_size, -1),
        reconstruction_mask=reconstruction_mask,
    )


def _transition_kind(
    state: dict[str, Any], action: dict[str, Any], next_state: dict[str, Any]
) -> int:
    before_stack = len(state.get("stack", []))
    after_stack = len(next_state.get("stack", []))
    kind = str(action.get("kind", "")).casefold()
    if action.get("paymentSources") or action.get("manaPayment"):
        return int(TransitionKind.COST)
    if after_stack > before_stack:
        return int(TransitionKind.ANNOUNCEMENT)
    if before_stack and after_stack < before_stack:
        return int(TransitionKind.RESOLUTION)
    if kind in {"passpriority", "castspell", "activateability", "counterspell"}:
        return int(TransitionKind.RESPONSE)
    return int(TransitionKind.RULES_CLOSURE)


def _timing_label(state: dict[str, Any], classes: int) -> int:
    phase = str(state.get("step", state.get("phase", ""))).casefold()
    phases = (
        "untap",
        "upkeep",
        "draw",
        "precombatmain",
        "begincombat",
        "declareattackers",
        "declareblockers",
        "combatdamage",
        "endcombat",
        "postcombatmain",
        "end",
        "cleanup",
    )
    return next((index for index, value in enumerate(phases) if value in phase), 0) % classes


def _aligned_transition(
    encoder: GraphObservationEncoderV15,
    state: dict[str, Any],
    action: dict[str, Any],
    next_state: dict[str, Any],
    observer_id: str,
    config: ModelConfigV15,
) -> TransitionBatchV15:
    before = encoder.encode(state, observer_id)
    after = encoder.encode(next_state, observer_id)
    keys = tuple(dict.fromkeys((*before.node_keys, *after.node_keys)))
    before_rows = {key: index for index, key in enumerate(before.node_keys)}
    after_rows = {key: index for index, key in enumerate(after.node_keys)}
    nodes = len(keys)
    features = torch.zeros((nodes, config.node_feature_dim), dtype=torch.float32)
    next_features = torch.zeros_like(features)
    node_types = torch.zeros(nodes, dtype=torch.long)
    owners = torch.zeros(nodes, dtype=torch.long)
    controllers = torch.zeros(nodes, dtype=torch.long)
    zones = torch.zeros(nodes, dtype=torch.long)
    positions = torch.zeros(nodes, dtype=torch.long)
    text_width = max(before.text_tokens.shape[1], after.text_tokens.shape[1])
    text_tokens = torch.zeros((nodes, text_width), dtype=torch.long)
    text_mask = torch.zeros((nodes, text_width), dtype=torch.bool)
    for row, key in enumerate(keys):
        source_row = before_rows.get(key)
        metadata = before
        if source_row is not None:
            features[row] = before.node_features[source_row]
            width = before.text_tokens.shape[1]
            text_tokens[row, :width] = before.text_tokens[source_row]
            text_mask[row, :width] = before.text_mask[source_row]
        else:
            source_row = after_rows[key]
            metadata = after
        target_row = after_rows.get(key)
        if target_row is not None:
            next_features[row] = after.node_features[target_row]
        node_types[row] = metadata.node_types[source_row]
        owners[row] = metadata.owners[source_row]
        controllers[row] = metadata.controllers[source_row]
        zones[row] = metadata.zones[source_row]
        positions[row] = metadata.positions[source_row]
    action_features = encoder.encode_actions([action], state, observer_id)
    effect = (next_features - features).mean(0)[: config.effect_dim]
    if effect.shape[0] < config.effect_dim:
        effect = torch.nn.functional.pad(effect, (0, config.effect_dim - effect.shape[0]))
    invariant = next_features.mean(0)[: config.invariant_dim]
    if invariant.shape[0] < config.invariant_dim:
        invariant = torch.nn.functional.pad(
            invariant, (0, config.invariant_dim - invariant.shape[0])
        )
    reconstruction_mask = torch.rand(nodes) < float(config.reconstruction_sample_rate)
    reconstruction_mask[0] = True
    return TransitionBatchV15(
        node_features=features.unsqueeze(0),
        node_types=node_types.unsqueeze(0),
        owners=owners.unsqueeze(0),
        node_mask=torch.ones((1, nodes), dtype=torch.bool),
        action_features=action_features,
        next_node_features=next_features.unsqueeze(0),
        transition_kind=torch.tensor([_transition_kind(state, action, next_state)]),
        legal_label=torch.ones(1),
        timing_label=torch.tensor([_timing_label(state, config.timing_classes)]),
        effect_target=effect.unsqueeze(0),
        invariant_target=invariant.unsqueeze(0),
        text_tokens=text_tokens.unsqueeze(0),
        text_mask=text_mask.unsqueeze(0),
        controllers=controllers.unsqueeze(0),
        zones=zones.unsqueeze(0),
        positions=positions.unsqueeze(0),
        reconstruction_mask=reconstruction_mask.unsqueeze(0),
    )


def load_replay_transitions(
    paths: tuple[str, ...] | list[str],
    config: ModelConfigV15,
    maximum: int,
    max_files: int | None = None,
) -> list[TransitionBatchV15]:
    encoder = GraphObservationEncoderV15(config.node_feature_dim, config.action_dim)
    files: list[Path] = []
    for pattern in paths:
        files.extend(Path(item) for item in glob.glob(str(pattern), recursive=True))
    files = sorted({path.resolve() for path in files if path.is_file()})
    if max_files is not None:
        files = files[-max_files:]
    transitions: list[TransitionBatchV15] = []
    for path in files:
        raw = json.loads(path.read_text(encoding="utf-8"))
        frames = raw.get("frames", []) if isinstance(raw, dict) else []
        for frame, following in zip(frames, frames[1:]):
            if not isinstance(frame, dict) or not isinstance(following, dict):
                continue
            state, next_state, action = (
                frame.get("state"),
                following.get("state"),
                frame.get("selectedAction"),
            )
            decision = frame.get("decision", {})
            observer_id = str(decision.get("playerId", "")) if isinstance(decision, dict) else ""
            if (
                not all(isinstance(item, dict) for item in (state, next_state, action))
                or not observer_id
            ):
                continue
            transitions.append(
                _aligned_transition(
                    encoder,
                    cast(dict[str, Any], state),
                    cast(dict[str, Any], action),
                    cast(dict[str, Any], next_state),
                    observer_id,
                    config,
                )
            )
            if len(transitions) >= maximum:
                return transitions
    return transitions


def _collate(samples: list[TransitionBatchV15], device: torch.device) -> TransitionBatchV15:
    width = max(sample.node_features.shape[1] for sample in samples)
    text_width = max(
        sample.text_tokens.shape[2] for sample in samples if sample.text_tokens is not None
    )

    def pad_nodes(value: torch.Tensor, fill: int | float = 0) -> torch.Tensor:
        node_pad = width - value.shape[1]
        if value.ndim == 3:
            return torch.nn.functional.pad(value, (0, 0, 0, node_pad), value=fill)
        return torch.nn.functional.pad(value, (0, node_pad), value=fill)

    names = (
        "node_features",
        "next_node_features",
        "node_types",
        "owners",
        "node_mask",
        "controllers",
        "zones",
        "positions",
        "reconstruction_mask",
    )
    values: dict[str, Any] = {}
    for name in names:
        tensors = [getattr(sample, name) for sample in samples]
        values[name] = torch.cat([pad_nodes(tensor) for tensor in tensors], dim=0)
    for name in ("text_tokens", "text_mask"):
        tensors = [getattr(sample, name) for sample in samples]
        values[name] = torch.cat(
            [
                torch.nn.functional.pad(
                    tensor,
                    (0, text_width - tensor.shape[2], 0, width - tensor.shape[1]),
                )
                for tensor in tensors
            ],
            dim=0,
        )
    for name in (
        "action_features",
        "transition_kind",
        "legal_label",
        "timing_label",
        "effect_target",
        "invariant_target",
    ):
        values[name] = torch.cat([getattr(sample, name) for sample in samples], dim=0)
    return TransitionBatchV15(**values).to(device)


def _load_config(path: Path) -> tuple[ModelConfigV15, V15TrainingConfig, Path]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("V15 configuration must be an object")
    model_raw = raw.get("model", {})
    training_raw = raw.get("training", {})
    if not isinstance(model_raw, dict) or not isinstance(training_raw, dict):
        raise ValueError("V15 model and training sections must be objects")
    output = Path(str(raw.get("outputDir", ".deepdeck/runs/v15-typed-latent")))
    return ModelConfigV15(**model_raw), V15TrainingConfig(**training_raw), output


def train(config_path: str | Path, output_override: str | Path | None = None) -> Path:
    model_config, training, configured_output = _load_config(Path(config_path))
    output = Path(output_override) if output_override is not None else configured_output
    output.mkdir(parents=True, exist_ok=True)
    random.seed(training.seed)
    torch.manual_seed(training.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TypedLatentWorldModelV15(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training.learning_rate)
    transitions = (
        []
        if training.synthetic_smoke
        else load_replay_transitions(
            training.replay_paths,
            model_config,
            training.max_transitions,
            training.max_replay_files,
        )
    )
    if not training.synthetic_smoke and not transitions:
        raise ValueError(
            "V15 requires Engine replay transitions; use synthetic_smoke only for tests"
        )
    metrics_path = output / "v15-metrics.jsonl"
    state_path = output / "v15-training-state.json"
    started = time.monotonic()
    for step in range(1, training.steps + 1):
        if training.synthetic_smoke:
            batch = synthetic_transition_batch(model_config, training, device)
        else:
            batch = _collate(random.choices(transitions, k=training.batch_size), device)
        losses = v15_losses(batch, model(batch))
        total = sum(LOSS_WEIGHTS[name] * loss for name, loss in losses.items())
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        record = {
            "step": step,
            "loss": float(total.detach().cpu()),
            **{name: float(loss.detach().cpu()) for name, loss in losses.items()},
            "elapsed_seconds": time.monotonic() - started,
            "reconstructionSampleRate": model_config.reconstruction_sample_rate,
            "dataSource": "synthetic-smoke" if training.synthetic_smoke else "engine-replay",
            "availableTransitions": len(transitions),
        }
        with metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
        state_path.write_text(
            json.dumps(
                {
                    "status": "running" if step < training.steps else "stopped",
                    "trainingPhase": "typed-latent-world-model",
                    "trainingStep": step,
                    "completed_episodes": 0,
                    "parallelGameWorkers": 0,
                    "activeAttempts": [],
                    "trainingElapsedSeconds": time.monotonic() - started,
                    "updatedAtUnixMs": int(time.time() * 1000),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        if step % training.checkpoint_every == 0 or step == training.steps:
            checkpoint = output / "checkpoints" / f"step-{step}"
            checkpoint.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model": model.state_dict(),
                    "model_config": asdict(model_config),
                    "observation_schema": model.observation_schema,
                    "training_step": step,
                },
                checkpoint / "v15-model.pt",
            )
    resolved = {
        "schemaVersion": "v15-training/v1",
        "model": asdict(model_config),
        "training": asdict(training),
        "outputDir": str(output.resolve()),
        "dataSource": "synthetic-smoke" if training.synthetic_smoke else "engine-replay",
        "availableTransitions": len(transitions),
    }
    (output / "resolved-v15-config.json").write_text(
        json.dumps(resolved, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output / "checkpoints" / f"step-{training.steps}"


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Train the V15 typed latent planner")
    result.add_argument("--config", required=True)
    result.add_argument("--output")
    return result


def main() -> None:
    arguments = parser().parse_args()
    train(arguments.config, arguments.output)


if __name__ == "__main__":
    main()
