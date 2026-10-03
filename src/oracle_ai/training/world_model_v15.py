from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import yaml

from oracle_ai.model_v15 import (
    ModelConfigV15,
    TransitionBatchV15,
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
    metrics_path = output / "v15-metrics.jsonl"
    state_path = output / "v15-training-state.json"
    started = time.monotonic()
    for step in range(1, training.steps + 1):
        batch = synthetic_transition_batch(model_config, training, device)
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
