from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any, cast

import torch
from deepdeck_agent import Agent, Decision, DecisionResponse, Game

from oracle_ai.decision_choices import expand_policy_actions
from oracle_ai.encoding_v13 import GraphObservationEncoderV13
from oracle_ai.model_v13 import (
    GraphBeliefWorldModelV13,
    ModelConfigV13,
    PolicyBatchV13,
)

from .forge_reference import ForgeReferenceAgent, ForgeReferenceProfile

LOGGER = logging.getLogger(__name__)
_PASSIVE_ACTION_KINDS = {"passPriority", "finishAttackers", "finishBlockers"}


class V13Agent(Agent):
    """Serve a trained V13 structured world-model policy through DeepDeckAgent."""

    def __init__(
        self,
        model: GraphBeliefWorldModelV13,
        *,
        device: torch.device | str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.encoder = GraphObservationEncoderV13(
            feature_dim=model.config.node_feature_dim,
            action_feature_dim=model.config.action_dim,
        )
        # A trained policy remains the primary decision maker. The transparent
        # reference policy only prevents an under-trained checkpoint from passing
        # every useful priority or combat decision in a human playtest.
        self.playtest_fallback = ForgeReferenceAgent(ForgeReferenceProfile.BALANCED)
        self._recurrent_states: dict[str, torch.Tensor] = {}

    async def on_game_start(self, game: Game, known_deck: list[dict[str, object]]) -> None:
        self._recurrent_states.clear()

    async def on_game_end(self, outcome: dict[str, object]) -> None:
        self._recurrent_states.clear()

    @staticmethod
    def _memory_key(decision: Decision) -> str:
        game_id = decision.game.raw.get("sessionId", decision.game.raw.get("gameId", "game"))
        return f"{game_id}:{decision.player_id}"

    def _select_model_action(
        self, decision: Decision, actions: list[dict[str, Any]]
    ) -> dict[str, Any]:
        graph = self.encoder.encode(copy.deepcopy(decision.game.raw), decision.player_id)
        action_features = self.encoder.encode_actions(
            actions, decision.game.raw, decision.player_id
        )
        edge_index = (
            torch.cat(
                (
                    torch.zeros((1, graph.edges.shape[1]), dtype=torch.long),
                    graph.edges,
                ),
                dim=0,
            )
            if graph.edges.numel()
            else torch.empty((3, 0), dtype=torch.long)
        )
        memory_key = self._memory_key(decision)
        batch = PolicyBatchV13(
            node_features=graph.node_features.unsqueeze(0).to(self.device),
            node_types=graph.node_types.unsqueeze(0).to(self.device),
            owners=graph.owners.unsqueeze(0).to(self.device),
            node_mask=torch.ones(
                (1, graph.node_features.shape[0]), dtype=torch.bool, device=self.device
            ),
            legal_action_features=action_features.unsqueeze(0).to(self.device),
            legal_action_mask=torch.ones(
                (1, action_features.shape[0]), dtype=torch.bool, device=self.device
            ),
            edge_index=edge_index.to(self.device),
            edge_types=graph.edge_types.to(self.device),
            recurrent_state=self._recurrent_states.get(memory_key),
            text_tokens=graph.text_tokens.unsqueeze(0).to(self.device),
            text_mask=graph.text_mask.unsqueeze(0).to(self.device),
            controllers=graph.controllers.unsqueeze(0).to(self.device),
            zones=graph.zones.unsqueeze(0).to(self.device),
            positions=graph.positions.unsqueeze(0).to(self.device),
        )
        with torch.no_grad():
            logits, _, memory = self.model.policy_value_with_memory(batch)
        self._recurrent_states[memory_key] = memory.detach()
        return actions[int(logits.argmax(dim=-1).item())]

    async def make_decision(self, decision: Decision) -> DecisionResponse:
        choice_kind = str((decision.choice or {}).get("kind", ""))
        if choice_kind in {"cardSelection", "cardOrder", "cardNameSelection"}:
            return await super().make_decision(decision)

        actions = expand_policy_actions(decision.raw)
        if not actions:
            return await super().make_decision(decision)

        selected = self._select_model_action(decision, actions)
        if str(selected.get("kind", "")) in _PASSIVE_ACTION_KINDS:
            fallback = await self.playtest_fallback.make_decision(decision)
            fallback_action = next(
                (action for action in actions if action.get("id") == fallback.action_id),
                None,
            )
            is_standalone_mana_action = bool(
                fallback_action
                and (fallback_action.get("decisions") or {}).get("manaAbility") is True
            )
            if (
                fallback.action_id
                and fallback.action_id != selected.get("id")
                and not is_standalone_mana_action
            ):
                LOGGER.info(
                    "V13 anti-pass fallback selected %s instead of %s",
                    fallback.action_id,
                    selected.get("id"),
                )
                return fallback
        return DecisionResponse(
            action_id=str(selected.get("_engineActionId", selected["id"])),
            number_value=cast(int | None, selected.get("_numberValue")),
        )


def load_v13_agent(
    checkpoint: str | Path,
    *,
    device: torch.device | str = "cpu",
) -> V13Agent:
    source = Path(checkpoint)
    checkpoint_file = source / "rl-model.pt"
    if not checkpoint_file.is_file():
        raise ValueError(f"V13 checkpoint is missing {checkpoint_file.name}")
    payload = torch.load(checkpoint_file, map_location=device, weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("V13 checkpoint payload must be an object")
    raw_config = payload.get("model_config")
    state = payload.get("model")
    if not isinstance(raw_config, dict) or not isinstance(state, dict):
        raise ValueError("V13 checkpoint is missing its model configuration or weights")
    model = GraphBeliefWorldModelV13(ModelConfigV13(**cast(dict[str, Any], raw_config)))
    if payload.get("observation_schema") != model.observation_schema:
        raise ValueError("Train a V13.2 checkpoint before using the structured Oracle encoder")
    model.load_state_dict(state)
    return V13Agent(model, device=device)
