from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelConfigV13:
    """Structured Transformer state encoder with V13 world-model heads."""

    node_feature_dim: int = 32
    latent_dim: int = 128
    node_type_count: int = 16
    player_slots: int = 2
    zone_count: int = 8
    heads: int = 4
    graph_layers: int = 2
    stochastic_dim: int = 32
    action_dim: int = 24
    opponent_actions: int = 8
    text_vocabulary_size: int = 8192
    text_max_tokens: int = 128
    max_sequence_positions: int = 1024


@dataclass
class GraphBatch:
    node_features: torch.Tensor
    node_types: torch.Tensor
    owners: torch.Tensor
    node_mask: torch.Tensor
    action_features: torch.Tensor
    next_node_features: torch.Tensor
    hidden_labels: torch.Tensor
    opponent_action: torch.Tensor
    search_policy: torch.Tensor
    outcome: torch.Tensor
    edge_index: torch.Tensor | None = None
    edge_types: torch.Tensor | None = None
    text_tokens: torch.Tensor | None = None
    text_mask: torch.Tensor | None = None
    controllers: torch.Tensor | None = None
    zones: torch.Tensor | None = None
    positions: torch.Tensor | None = None

    def to(self, device: torch.device) -> GraphBatch:
        return GraphBatch(
            **{
                name: value.to(device) if isinstance(value, torch.Tensor) else value
                for name, value in vars(self).items()
            }
        )


@dataclass
class PolicyBatchV13:
    """Observer-safe structured state plus legal actions for one decision."""

    node_features: torch.Tensor
    node_types: torch.Tensor
    owners: torch.Tensor
    node_mask: torch.Tensor
    legal_action_features: torch.Tensor
    legal_action_mask: torch.Tensor
    edge_index: torch.Tensor | None = None
    edge_types: torch.Tensor | None = None
    recurrent_state: torch.Tensor | None = None
    text_tokens: torch.Tensor | None = None
    text_mask: torch.Tensor | None = None
    controllers: torch.Tensor | None = None
    zones: torch.Tensor | None = None
    positions: torch.Tensor | None = None

    def to(self, device: torch.device) -> PolicyBatchV13:
        return PolicyBatchV13(
            **{
                name: value.to(device) if isinstance(value, torch.Tensor) else value
                for name, value in vars(self).items()
            }
        )


class GraphGRUCell(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.message = nn.Linear(dimension, dimension, bias=False)
        self.cell = nn.GRUCell(dimension * 2, dimension)

    def forward(
        self, nodes: torch.Tensor, memory: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        weights = mask.unsqueeze(-1).to(nodes.dtype)
        global_message = self.message((nodes * weights).sum(1) / weights.sum(1).clamp_min(1.0))
        messages = global_message.unsqueeze(1).expand_as(nodes)
        updated = self.cell(
            torch.cat((nodes, messages), -1).flatten(0, 1), memory.flatten(0, 1)
        ).view_as(memory)
        return torch.where(mask.unsqueeze(-1), updated, memory)


class GraphBeliefWorldModelV13(nn.Module):
    """Structured Transformer, recurrent latent dynamics and prediction heads.

    V13.2 restores V12's ordered-token principle while retaining the world
    model, opponent belief and persistent recurrent memory introduced in V13.
    The class name remains stable so existing learner integrations keep working.
    """

    model_family = "structured-world-v13"
    observation_schema = "structured-observation/v13.2"

    def __init__(self, config: ModelConfigV13) -> None:
        super().__init__()
        self.config = config
        d = config.latent_dim
        self.node_projection = nn.Linear(config.node_feature_dim, d)
        self.node_type_embedding = nn.Embedding(config.node_type_count, d)
        self.owner_embedding = nn.Embedding(config.player_slots + 1, d)
        self.controller_embedding = nn.Embedding(config.player_slots + 1, d)
        self.zone_embedding = nn.Embedding(config.zone_count + 1, d)
        self.sequence_position_embedding = nn.Embedding(config.max_sequence_positions, d)
        self.text_embedding = nn.Embedding(config.text_vocabulary_size, d, padding_idx=0)
        self.text_position_embedding = nn.Embedding(config.text_max_tokens, d)
        text_layer = nn.TransformerEncoderLayer(
            d, config.heads, d * 2, batch_first=True, norm_first=True, activation="gelu"
        )
        self.text_encoder = nn.TransformerEncoder(text_layer, 1)
        layer = nn.TransformerEncoderLayer(
            d, config.heads, d * 4, batch_first=True, norm_first=True, activation="gelu"
        )
        self.sequence_encoder = nn.TransformerEncoder(layer, config.graph_layers)
        self.graph_memory = GraphGRUCell(d)
        self.policy_memory = nn.GRUCell(d, d)
        # The 24 semantic features are deliberately encoded as independent
        # trigger/timing, cost and effect streams before they are fused.
        self.action_trigger_projection = nn.Linear(9, d)
        self.action_cost_projection = nn.Linear(3, d)
        self.action_effect_projection = nn.Linear(12, d)
        self.action_compatibility_projection = nn.Linear(config.action_dim, d)
        self.action_norm = nn.LayerNorm(d)
        self.afterstate = nn.Sequential(nn.Linear(d * 2, d), nn.GELU(), nn.LayerNorm(d))
        self.prior = nn.Linear(d, config.stochastic_dim * 2)
        self.posterior = nn.Linear(d * 2, config.stochastic_dim * 2)
        self.next_latent = nn.Sequential(
            nn.Linear(d + config.stochastic_dim, d), nn.GELU(), nn.LayerNorm(d)
        )
        self.decoder = nn.Linear(d, config.node_feature_dim)
        self.belief_head = nn.Linear(d, config.player_slots)
        self.opponent_head = nn.Linear(d, config.opponent_actions)
        self.search_head = nn.Linear(d, config.opponent_actions)
        self.value_head = nn.Linear(d, 3)
        self.rl_policy_head = nn.Sequential(
            nn.Linear(d * 2, d), nn.GELU(), nn.Linear(d, 1)
        )
        self.rl_value_head = nn.Sequential(
            nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1)
        )

    def export_config(self) -> dict[str, object]:
        return {"architecture": self.model_family, **asdict(self.config)}

    def encode_graph(
        self,
        node_features: torch.Tensor,
        node_types: torch.Tensor,
        owners: torch.Tensor,
        node_mask: torch.Tensor,
        edge_index: torch.Tensor | None = None,
        edge_types: torch.Tensor | None = None,
        text_tokens: torch.Tensor | None = None,
        text_mask: torch.Tensor | None = None,
        controllers: torch.Tensor | None = None,
        zones: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        controllers = owners if controllers is None else controllers
        zones = torch.zeros_like(owners) if zones is None else zones
        if positions is None:
            positions = torch.zeros_like(owners)
        nodes = (
            self.node_projection(node_features)
            + self.node_type_embedding(node_types)
            + self.owner_embedding(owners.clamp(0, self.config.player_slots))
            + self.controller_embedding(controllers.clamp(0, self.config.player_slots))
            + self.zone_embedding(zones.clamp(0, self.config.zone_count))
            + self.sequence_position_embedding(
                positions.clamp(0, self.config.max_sequence_positions - 1)
            )
        )
        if text_tokens is not None and text_mask is not None and text_mask.any():
            batch_size, node_count, text_width = text_tokens.shape
            flat_tokens = text_tokens.reshape(batch_size * node_count, text_width)
            flat_mask = text_mask.reshape(batch_size * node_count, text_width)
            has_text = flat_mask.any(-1)
            selected_tokens = flat_tokens[has_text]
            selected_mask = flat_mask[has_text]
            positions = torch.arange(text_width, device=nodes.device).unsqueeze(0)
            embedded_text = self.text_embedding(selected_tokens) + self.text_position_embedding(
                positions
            )
            encoded_text = self.text_encoder(
                embedded_text, src_key_padding_mask=~selected_mask
            )
            weights = selected_mask.unsqueeze(-1).to(encoded_text.dtype)
            pooled_text = (encoded_text * weights).sum(1) / weights.sum(1).clamp_min(1.0)
            flat_node_text = torch.zeros(
                (batch_size * node_count, nodes.shape[-1]),
                dtype=nodes.dtype,
                device=nodes.device,
            )
            flat_node_text[has_text] = pooled_text
            nodes = nodes + flat_node_text.view(batch_size, node_count, -1)
        # edge_index/edge_types are accepted only for loading older call sites;
        # V13.2 intentionally relies on Transformer attention over this sequence.
        return self.sequence_encoder(nodes, src_key_padding_mask=~node_mask)

    def encode_actions(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[-1] != self.config.action_dim:
            raise ValueError("action feature width does not match the model configuration")
        if self.config.action_dim != 24:
            return self.action_norm(self.action_compatibility_projection(features))
        trigger = features[..., :9]
        cost = features[..., 20:23]
        effect = torch.cat((features[..., 9:20], features[..., 23:24]), dim=-1)
        return self.action_norm(
            self.action_trigger_projection(trigger)
            + self.action_cost_projection(cost)
            + self.action_effect_projection(effect)
        )

    def encode(self, batch: GraphBatch, features: torch.Tensor | None = None) -> torch.Tensor:
        raw = batch.node_features if features is None else features
        return self.encode_graph(
            raw,
            batch.node_types,
            batch.owners,
            batch.node_mask,
            batch.edge_index,
            batch.edge_types,
            batch.text_tokens,
            batch.text_mask,
            batch.controllers,
            batch.zones,
            batch.positions,
        )

    def pooled_state(
        self,
        node_features: torch.Tensor,
        node_types: torch.Tensor,
        owners: torch.Tensor,
        node_mask: torch.Tensor,
        edge_index: torch.Tensor | None = None,
        edge_types: torch.Tensor | None = None,
        text_tokens: torch.Tensor | None = None,
        text_mask: torch.Tensor | None = None,
        controllers: torch.Tensor | None = None,
        zones: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        encoded = self.encode_graph(
            node_features,
            node_types,
            owners,
            node_mask,
            edge_index,
            edge_types,
            text_tokens,
            text_mask,
            controllers,
            zones,
            positions,
        )
        memory = self.graph_memory(encoded, torch.zeros_like(encoded), node_mask)
        weights = node_mask.unsqueeze(-1).to(encoded.dtype)
        return (memory * weights).sum(1) / weights.sum(1).clamp_min(1.0)

    def policy_value_with_memory(
        self, batch: PolicyBatchV13
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pooled = self.pooled_state(
            batch.node_features,
            batch.node_types,
            batch.owners,
            batch.node_mask,
            batch.edge_index,
            batch.edge_types,
            batch.text_tokens,
            batch.text_mask,
            batch.controllers,
            batch.zones,
            batch.positions,
        )
        previous = batch.recurrent_state
        if previous is None:
            previous = torch.zeros_like(pooled)
        remembered = self.policy_memory(pooled, previous)
        actions = self.encode_actions(batch.legal_action_features)
        state = remembered.unsqueeze(1).expand(-1, actions.shape[1], -1)
        logits = self.rl_policy_head(torch.cat((state, actions), dim=-1)).squeeze(-1)
        logits = logits.masked_fill(~batch.legal_action_mask, torch.finfo(logits.dtype).min)
        value = self.rl_value_head(remembered).squeeze(-1)
        return logits, value, remembered

    def policy_value(self, batch: PolicyBatchV13) -> tuple[torch.Tensor, torch.Tensor]:
        """Return masked logits over legal actions and a recurrent-state value."""

        logits, value, _ = self.policy_value_with_memory(batch)
        return logits, value

    @staticmethod
    def _normal(parameters: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_variance = parameters.chunk(2, dim=-1)
        return mean, log_variance.clamp(-8.0, 4.0)

    def forward(self, batch: GraphBatch) -> dict[str, torch.Tensor]:
        encoded = self.encode(batch)
        memory = self.graph_memory(encoded, torch.zeros_like(encoded), batch.node_mask)
        weights = batch.node_mask.unsqueeze(-1).to(encoded.dtype)
        pooled = (memory * weights).sum(1) / weights.sum(1).clamp_min(1.0)
        action = self.encode_actions(batch.action_features)
        afterstate = self.afterstate(torch.cat((pooled, action), dim=-1))
        next_encoded = self.encode(batch, batch.next_node_features)
        next_pooled = (next_encoded * weights).sum(1) / weights.sum(1).clamp_min(1.0)
        prior_mean, prior_log_variance = self._normal(self.prior(afterstate))
        posterior_mean, posterior_log_variance = self._normal(
            self.posterior(torch.cat((afterstate, next_pooled), dim=-1))
        )
        stochastic = posterior_mean
        predicted_next = self.next_latent(torch.cat((afterstate, stochastic), dim=-1))
        return {
            "encoded": encoded,
            "pooled": pooled,
            "next_target": next_pooled.detach(),
            "predicted_next": predicted_next,
            "reconstruction": self.decoder(encoded),
            "prior_mean": prior_mean,
            "prior_log_variance": prior_log_variance,
            "posterior_mean": posterior_mean,
            "posterior_log_variance": posterior_log_variance,
            "belief_logits": self.belief_head(pooled),
            "opponent_logits": self.opponent_head(afterstate),
            "search_logits": self.search_head(afterstate),
            "value_logits": self.value_head(afterstate),
        }


def v13_losses(batch: GraphBatch, output: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    mask = batch.node_mask.unsqueeze(-1).expand_as(batch.node_features)
    reconstruction = F.mse_loss(output["reconstruction"][mask], batch.node_features[mask])
    dynamics = F.mse_loss(output["predicted_next"], output["next_target"])
    prior_mean, prior_logvar = output["prior_mean"], output["prior_log_variance"]
    post_mean, post_logvar = output["posterior_mean"], output["posterior_log_variance"]
    kl = 0.5 * (
        prior_logvar - post_logvar
        + (post_logvar.exp() + (post_mean - prior_mean).pow(2)) / prior_logvar.exp()
        - 1.0
    ).mean()
    belief = F.binary_cross_entropy_with_logits(output["belief_logits"], batch.hidden_labels)
    opponent = F.cross_entropy(output["opponent_logits"], batch.opponent_action)
    search = -(batch.search_policy * F.log_softmax(output["search_logits"], -1)).sum(-1).mean()
    value = F.cross_entropy(output["value_logits"], batch.outcome)
    return {
        "reconstruction_loss": reconstruction,
        "dynamics_loss": dynamics,
        "kl_loss": kl,
        "belief_loss": belief,
        "opponent_loss": opponent,
        "search_loss": search,
        "value_loss": value,
    }
