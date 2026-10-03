from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import IntEnum

import torch
from torch import nn
from torch.nn import functional as F

from oracle_ai.model_v13 import GraphBeliefWorldModelV13, ModelConfigV13


class TransitionKind(IntEnum):
    COST = 0
    ANNOUNCEMENT = 1
    RESPONSE = 2
    RESOLUTION = 3
    RULES_CLOSURE = 4


@dataclass(frozen=True)
class ModelConfigV15:
    node_feature_dim: int = 32
    latent_dim: int = 128
    tactical_dim: int = 96
    strategic_dim: int = 64
    node_type_count: int = 16
    player_slots: int = 2
    zone_count: int = 8
    heads: int = 4
    graph_layers: int = 2
    action_dim: int = 32
    target_dim: int = 32
    timing_classes: int = 12
    effect_dim: int = 32
    invariant_dim: int = 16
    transition_kinds: int = len(TransitionKind)
    text_vocabulary_size: int = 8192
    text_max_tokens: int = 128
    max_sequence_positions: int = 1024
    reconstruction_sample_rate: float = 0.05
    max_plan_nodes: int = 8
    max_plan_branches: int = 2


@dataclass(frozen=True)
class LatentStateV15:
    detailed: torch.Tensor
    tactical: torch.Tensor
    strategic: torch.Tensor
    uncertainty: torch.Tensor


@dataclass
class TransitionBatchV15:
    node_features: torch.Tensor
    node_types: torch.Tensor
    owners: torch.Tensor
    node_mask: torch.Tensor
    action_features: torch.Tensor
    next_node_features: torch.Tensor
    transition_kind: torch.Tensor
    legal_label: torch.Tensor
    timing_label: torch.Tensor
    effect_target: torch.Tensor
    invariant_target: torch.Tensor
    edge_index: torch.Tensor | None = None
    edge_types: torch.Tensor | None = None
    text_tokens: torch.Tensor | None = None
    text_mask: torch.Tensor | None = None
    controllers: torch.Tensor | None = None
    zones: torch.Tensor | None = None
    positions: torch.Tensor | None = None
    reconstruction_mask: torch.Tensor | None = None

    def to(self, device: torch.device) -> TransitionBatchV15:
        return TransitionBatchV15(
            **{
                name: value.to(device) if isinstance(value, torch.Tensor) else value
                for name, value in vars(self).items()
            }
        )


@dataclass
class PolicyBatchV15:
    node_features: torch.Tensor
    node_types: torch.Tensor
    owners: torch.Tensor
    node_mask: torch.Tensor
    legal_action_features: torch.Tensor
    legal_action_mask: torch.Tensor
    edge_index: torch.Tensor | None = None
    edge_types: torch.Tensor | None = None
    text_tokens: torch.Tensor | None = None
    text_mask: torch.Tensor | None = None
    controllers: torch.Tensor | None = None
    zones: torch.Tensor | None = None
    positions: torch.Tensor | None = None


class TypedLatentWorldModelV15(nn.Module):  # type: ignore[misc]
    """Multi-level latent world model with composable typed transitions."""

    model_family = "typed-latent-planner-v15"
    observation_schema = "typed-latent-observation/v15"

    def __init__(self, config: ModelConfigV15) -> None:
        super().__init__()
        self.config = config
        d = config.latent_dim
        backbone_config = ModelConfigV13(
            node_feature_dim=config.node_feature_dim,
            latent_dim=d,
            node_type_count=config.node_type_count,
            player_slots=config.player_slots,
            zone_count=config.zone_count,
            heads=config.heads,
            graph_layers=config.graph_layers,
            action_dim=24,
            text_vocabulary_size=config.text_vocabulary_size,
            text_max_tokens=config.text_max_tokens,
            max_sequence_positions=config.max_sequence_positions,
        )
        self.state_backbone = GraphBeliefWorldModelV13(backbone_config)
        self.tactical_projection = nn.Sequential(
            nn.Linear(d, config.tactical_dim), nn.GELU(), nn.LayerNorm(config.tactical_dim)
        )
        self.strategic_projection = nn.Sequential(
            nn.Linear(config.tactical_dim, config.strategic_dim),
            nn.GELU(),
            nn.LayerNorm(config.strategic_dim),
        )
        self.action_kind_encoder = nn.Linear(9, d)
        self.action_effect_encoder = nn.Linear(11, d)
        self.cost_encoder = nn.Linear(3, d)
        self.target_encoder = nn.Linear(config.action_dim - 22, d)
        self.timing_encoder = nn.Linear(4, d)
        self.action_norm = nn.LayerNorm(d)
        self.transition_embedding = nn.Embedding(config.transition_kinds, d)
        self.transition_operators = nn.ModuleList(
            nn.Sequential(nn.Linear(d * 3, d), nn.GELU(), nn.Linear(d, d), nn.LayerNorm(d))
            for _ in range(config.transition_kinds)
        )
        self.delta_gate = nn.Sequential(nn.Linear(d * 2, d), nn.Sigmoid())
        self.state_update_norm = nn.LayerNorm(d)
        self.uncertainty_head = nn.Sequential(nn.Linear(d * 2, d), nn.GELU(), nn.Linear(d, 1))
        self.state_decoder = nn.Linear(d, config.node_feature_dim)
        self.invariant_head = nn.Linear(d, config.invariant_dim)
        self.effect_head = nn.Linear(d, config.effect_dim)
        self.legality_head = nn.Sequential(nn.Linear(d * 2, d), nn.GELU(), nn.Linear(d, 1))
        self.timing_head = nn.Sequential(
            nn.Linear(d * 2, d), nn.GELU(), nn.Linear(d, config.timing_classes)
        )
        self.policy_head = nn.Sequential(nn.Linear(d * 2, d), nn.GELU(), nn.Linear(d, 1))
        self.value_head = nn.Sequential(
            nn.Linear(config.strategic_dim, config.strategic_dim),
            nn.GELU(),
            nn.Linear(config.strategic_dim, 1),
        )
        self.event_relevance_head = nn.Sequential(
            nn.Linear(config.strategic_dim + config.effect_dim, config.strategic_dim),
            nn.GELU(),
            nn.Linear(config.strategic_dim, 3),
        )

    def export_config(self) -> dict[str, object]:
        return {"architecture": self.model_family, **asdict(self.config)}

    def _encoded_nodes(
        self,
        node_features: torch.Tensor,
        node_types: torch.Tensor,
        owners: torch.Tensor,
        node_mask: torch.Tensor,
        batch: TransitionBatchV15 | PolicyBatchV15,
    ) -> torch.Tensor:
        return self.state_backbone.encode_graph(
            node_features,
            node_types,
            owners,
            node_mask,
            batch.edge_index,
            batch.edge_types,
            batch.text_tokens,
            batch.text_mask,
            batch.controllers,
            batch.zones,
            batch.positions,
        )

    def encode_state(
        self,
        node_features: torch.Tensor,
        node_types: torch.Tensor,
        owners: torch.Tensor,
        node_mask: torch.Tensor,
        batch: TransitionBatchV15 | PolicyBatchV15,
    ) -> tuple[LatentStateV15, torch.Tensor]:
        nodes = self._encoded_nodes(node_features, node_types, owners, node_mask, batch)
        weights = node_mask.unsqueeze(-1).to(nodes.dtype)
        detailed = (nodes * weights).sum(1) / weights.sum(1).clamp_min(1.0)
        tactical = self.tactical_projection(detailed)
        strategic = self.strategic_projection(tactical)
        uncertainty = torch.zeros(detailed.shape[0], device=detailed.device)
        return LatentStateV15(detailed, tactical, strategic, uncertainty), nodes

    def encode_action(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[-1] != self.config.action_dim:
            raise ValueError("V15 action feature width does not match its configuration")
        kind = features[..., :9]
        effect = features[..., 9:20]
        cost = features[..., 20:23]
        timing = torch.stack(
            (features[..., 0], features[..., 4], features[..., 5], features[..., 29]), dim=-1
        )
        target = torch.cat((features[..., 23:32], features[..., 24:25]), dim=-1)
        return self.action_norm(
            self.action_kind_encoder(kind)
            + self.action_effect_encoder(effect)
            + self.cost_encoder(cost)
            + self.target_encoder(target)
            + self.timing_encoder(timing)
        )

    def apply_transition(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        transition_kind: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        kinds = transition_kind.long().clamp(0, self.config.transition_kinds - 1)
        kind_embedding = self.transition_embedding(kinds)
        operator_input = torch.cat((state, action, kind_embedding), dim=-1)
        stacked = torch.stack(
            [operator(operator_input) for operator in self.transition_operators], dim=1
        )
        row = torch.arange(state.shape[0], device=state.device)
        delta = stacked[row, kinds]
        gate = self.delta_gate(torch.cat((state, delta), dim=-1))
        next_state = self.state_update_norm(state + gate * delta)
        uncertainty = F.softplus(self.uncertainty_head(torch.cat((state, delta), dim=-1))).squeeze(
            -1
        )
        return next_state, delta, uncertainty

    def compose_transitions(
        self,
        state: torch.Tensor,
        actions: torch.Tensor,
        transition_kinds: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        uncertainty = torch.zeros(state.shape[0], device=state.device)
        current = state
        for index in range(actions.shape[1]):
            current, _, step_uncertainty = self.apply_transition(
                current, actions[:, index], transition_kinds[:, index]
            )
            uncertainty = uncertainty + step_uncertainty
        return current, uncertainty

    def policy_value(
        self, batch: PolicyBatchV15
    ) -> tuple[torch.Tensor, torch.Tensor, LatentStateV15]:
        state, _ = self.encode_state(
            batch.node_features, batch.node_types, batch.owners, batch.node_mask, batch
        )
        actions = self.encode_action(batch.legal_action_features)
        expanded = state.detailed.unsqueeze(1).expand_as(actions)
        logits = self.policy_head(torch.cat((expanded, actions), dim=-1)).squeeze(-1)
        logits = logits.masked_fill(~batch.legal_action_mask, torch.finfo(logits.dtype).min)
        value = self.value_head(state.strategic).squeeze(-1)
        return logits, value, state

    def score_imagined_actions(
        self, state: LatentStateV15, action_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        actions = self.encode_action(action_features)
        detailed = state.detailed.unsqueeze(1).expand_as(actions)
        feasibility = torch.sigmoid(
            self.legality_head(torch.cat((detailed, actions), dim=-1)).squeeze(-1)
        )
        batch, count, width = actions.shape
        imagined, _, uncertainty = self.apply_transition(
            detailed.reshape(batch * count, width),
            actions.reshape(batch * count, width),
            torch.full(
                (batch * count,),
                int(TransitionKind.RESOLUTION),
                dtype=torch.long,
                device=actions.device,
            ),
        )
        tactical = self.tactical_projection(imagined)
        strategic = self.strategic_projection(tactical)
        values = self.value_head(strategic).reshape(batch, count)
        return values * feasibility, uncertainty.reshape(batch, count)

    def forward(self, batch: TransitionBatchV15) -> dict[str, torch.Tensor]:
        state, nodes = self.encode_state(
            batch.node_features, batch.node_types, batch.owners, batch.node_mask, batch
        )
        next_state, _ = self.encode_state(
            batch.next_node_features,
            batch.node_types,
            batch.owners,
            batch.node_mask,
            batch,
        )
        action = self.encode_action(batch.action_features)
        predicted, delta, uncertainty = self.apply_transition(
            state.detailed, action, batch.transition_kind
        )
        pair = torch.cat((state.detailed, action), dim=-1)
        return {
            "state": state.detailed,
            "predicted_next": predicted,
            "next_target": next_state.detailed.detach(),
            "delta": delta,
            "uncertainty": uncertainty,
            "reconstruction": self.state_decoder(nodes),
            "invariants": self.invariant_head(predicted),
            "effect": self.effect_head(delta),
            "legality_logits": self.legality_head(pair).squeeze(-1),
            "timing_logits": self.timing_head(pair),
        }


def v15_losses(
    batch: TransitionBatchV15, output: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    latent = F.smooth_l1_loss(output["predicted_next"], output["next_target"])
    invariants = F.mse_loss(output["invariants"], batch.invariant_target)
    effect = F.smooth_l1_loss(output["effect"], batch.effect_target)
    legality = F.binary_cross_entropy_with_logits(output["legality_logits"], batch.legal_label)
    timing = F.cross_entropy(output["timing_logits"], batch.timing_label)
    calibration = F.mse_loss(
        output["uncertainty"],
        (output["predicted_next"].detach() - output["next_target"]).pow(2).mean(-1),
    )
    mask = batch.reconstruction_mask
    if mask is None:
        mask = batch.node_mask & (torch.rand_like(batch.node_mask, dtype=torch.float32) < 0.05)
    expanded = mask.unsqueeze(-1).expand_as(batch.node_features)
    reconstruction = (
        F.mse_loss(output["reconstruction"][expanded], batch.node_features[expanded])
        if expanded.any()
        else output["reconstruction"].sum() * 0.0
    )
    return {
        "latent_transition_loss": latent,
        "invariant_loss": invariants,
        "effect_loss": effect,
        "legality_loss": legality,
        "timing_loss": timing,
        "uncertainty_loss": calibration,
        "sampled_reconstruction_loss": reconstruction,
    }


def v15_multistep_loss(
    model: TypedLatentWorldModelV15,
    initial_state: torch.Tensor,
    action_sequence: torch.Tensor,
    transition_sequence: torch.Tensor,
    target_state: torch.Tensor,
) -> torch.Tensor:
    """Verify that a composed sequence of latent deltas reconstructs a future anchor."""
    predicted, uncertainty = model.compose_transitions(
        initial_state, action_sequence, transition_sequence
    )
    horizon = max(1, action_sequence.shape[1])
    return F.smooth_l1_loss(predicted, target_state.detach()) + uncertainty.mean() / horizon
