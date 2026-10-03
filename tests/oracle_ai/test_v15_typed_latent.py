from __future__ import annotations

import torch

from oracle_ai.model_v15 import (
    ModelConfigV15,
    TransitionKind,
    TypedLatentWorldModelV15,
    v15_losses,
    v15_multistep_loss,
)
from oracle_ai.training.world_model_v15 import (
    V15TrainingConfig,
    synthetic_transition_batch,
    train,
)


def _config() -> ModelConfigV15:
    return ModelConfigV15(
        node_feature_dim=12,
        latent_dim=24,
        tactical_dim=16,
        strategic_dim=12,
        heads=4,
        graph_layers=1,
        action_dim=32,
        effect_dim=12,
        invariant_dim=8,
        text_max_tokens=16,
    )


def test_v15_trains_typed_transition_heads_with_sampled_reconstruction() -> None:
    config = _config()
    training = V15TrainingConfig(batch_size=3, nodes=7, steps=1)
    batch = synthetic_transition_batch(config, training, torch.device("cpu"))
    model = TypedLatentWorldModelV15(config)

    output = model(batch)
    losses = v15_losses(batch, output)

    assert output["predicted_next"].shape == (3, config.latent_dim)
    assert set(losses) == {
        "latent_transition_loss",
        "invariant_loss",
        "effect_loss",
        "legality_loss",
        "timing_loss",
        "uncertainty_loss",
        "sampled_reconstruction_loss",
    }
    assert batch.reconstruction_mask is not None
    assert int(batch.reconstruction_mask.sum()) < batch.node_mask.numel()
    assert all(loss.ndim == 0 and torch.isfinite(loss) for loss in losses.values())


def test_v15_composes_cost_announcement_response_and_resolution_deltas() -> None:
    config = _config()
    model = TypedLatentWorldModelV15(config)
    initial = torch.randn(2, config.latent_dim)
    actions = torch.randn(2, 4, config.latent_dim)
    kinds = torch.tensor(
        [
            [
                TransitionKind.COST,
                TransitionKind.ANNOUNCEMENT,
                TransitionKind.RESPONSE,
                TransitionKind.RESOLUTION,
            ],
            [
                TransitionKind.COST,
                TransitionKind.ANNOUNCEMENT,
                TransitionKind.RESOLUTION,
                TransitionKind.RULES_CLOSURE,
            ],
        ]
    )
    with torch.no_grad():
        target, uncertainty = model.compose_transitions(initial, actions, kinds)

    loss = v15_multistep_loss(model, initial, actions, kinds, target)

    assert target.shape == initial.shape
    assert uncertainty.shape == (2,)
    assert torch.isfinite(loss)
    assert float(loss) > 0.0  # calibrated uncertainty remains a trained objective


def test_v15_uses_different_operators_for_cost_and_resolution() -> None:
    config = _config()
    model = TypedLatentWorldModelV15(config).eval()
    state = torch.randn(2, config.latent_dim)
    action = torch.randn(2, config.latent_dim)

    with torch.no_grad():
        cost, _, _ = model.apply_transition(
            state, action, torch.full((2,), int(TransitionKind.COST))
        )
        resolution, _, _ = model.apply_transition(
            state, action, torch.full((2,), int(TransitionKind.RESOLUTION))
        )

    assert not torch.allclose(cost, resolution)


def test_v15_trainer_writes_a_playable_checkpoint_and_metrics(tmp_path) -> None:
    config = tmp_path / "v15.yaml"
    output = tmp_path / "run"
    config.write_text(
        f"""outputDir: {output.as_posix()}
model:
  node_feature_dim: 8
  latent_dim: 16
  tactical_dim: 12
  strategic_dim: 8
  heads: 4
  graph_layers: 1
  action_dim: 32
  effect_dim: 8
  invariant_dim: 4
  text_max_tokens: 8
training:
  steps: 1
  batch_size: 2
  nodes: 4
  checkpoint_every: 1
  seed: 15
""",
        encoding="utf-8",
    )

    checkpoint = train(config)

    assert (checkpoint / "v15-model.pt").is_file()
    assert (output / "v15-metrics.jsonl").is_file()
    assert '"trainingPhase": "typed-latent-world-model"' in (
        output / "v15-training-state.json"
    ).read_text(encoding="utf-8")
