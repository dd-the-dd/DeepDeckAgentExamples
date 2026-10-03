from __future__ import annotations

import copy
import uuid
from pathlib import Path
from typing import Any, cast

import torch
from deepdeck_agent import (
    ActionIntent,
    ActivePlan,
    Agent,
    ComputeBudget,
    ComputeMode,
    Decision,
    DecisionResponse,
    EffectPrediction,
    FeasibilityPrediction,
    PlanBranch,
    PlanGraph,
    PlannedAgent,
    PlanningContext,
    PlanNode,
    PlanRevision,
    PlanRevisionKind,
    TimingIntent,
)

from oracle_ai.encoding_v15 import GraphObservationEncoderV15
from oracle_ai.model_v15 import ModelConfigV15, PolicyBatchV15, TypedLatentWorldModelV15

_PLAN_INVALIDATING_EVENTS = frozenset(
    {
        "spellcountered",
        "permanentdestroyed",
        "carddrawn",
        "spellcast",
        "lifechanged",
        "attackersdeclared",
        "blockersdeclared",
    }
)


class V15Agent(PlannedAgent):
    """Typed-latent anytime planner exposed through the regular Agent SDK."""

    def __init__(
        self,
        model: TypedLatentWorldModelV15,
        *,
        device: torch.device | str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.encoder = GraphObservationEncoderV15(
            feature_dim=model.config.node_feature_dim,
            action_feature_dim=model.config.action_dim,
        )
        self.last_inference: dict[str, Any] = {}
        super().__init__(self)

    def compute_budget(self, decision: Decision) -> ComputeBudget:
        base = super().compute_budget(decision)
        state = decision.game.raw
        stack = state.get("stack", [])
        step = str(state.get("step", "")).casefold()
        irreversible = any(
            action.kind.casefold()
            in {"castspell", "declareattacker", "declareblocker", "activateability"}
            for action in decision.actions
        )
        importance = min(
            1.0,
            base.importance
            + (0.3 if isinstance(stack, list) and stack else 0.0)
            + (0.25 if "combat" in step or "block" in step else 0.0)
            + (0.15 if irreversible else 0.0),
        )
        if len(decision.actions) <= 1:
            return ComputeBudget(ComputeMode.REFLEX, 20, base.remaining_ms, importance)
        if importance >= 0.65 and (base.remaining_ms is None or base.remaining_ms > 30_000):
            available = base.remaining_ms or 120_000
            return ComputeBudget(
                ComputeMode.STRATEGIC,
                min(12_000, max(1_500, available // 24)),
                base.remaining_ms,
                importance,
            )
        return ComputeBudget(ComputeMode.TACTICAL, 600, base.remaining_ms, importance)

    async def create_plan(self, context: PlanningContext) -> PlanGraph:
        state = context.game.raw
        stack = state.get("stack", [])
        objective = (
            "resolve-current-interaction" if isinstance(stack, list) and stack else "develop"
        )
        nodes = (
            PlanNode(
                "develop",
                "Improve resources while preserving interaction",
                ActionIntent("playLand", timing=TimingIntent("main-phase")),
                EffectPrediction({"resourceDevelopment": 1}, confidence=0.5),
                FeasibilityPrediction(0.5),
                (
                    PlanBranch("opponent interacts", "respond", ("spellCast",), 0.3),
                    PlanBranch("combat begins", "combat", ("attackersDeclared",), 0.3),
                ),
                "respond",
            ),
            PlanNode(
                "respond",
                "Protect the current position or disrupt the opponent",
                ActionIntent("castSpell", timing=TimingIntent("priority")),
                feasibility=FeasibilityPrediction(0.5),
                fallback_node_id="develop",
            ),
            PlanNode(
                "combat",
                "Select the highest-value combat line",
                ActionIntent("declareAttackers", timing=TimingIntent("declare-attackers")),
                feasibility=FeasibilityPrediction(0.5),
                fallback_node_id="develop",
            ),
        )
        root = "respond" if objective == "resolve-current-interaction" else "develop"
        return PlanGraph(f"v15:{uuid.uuid4().hex[:12]}", root, nodes, objective)

    async def review_plan(self, context: PlanningContext, plan: ActivePlan) -> PlanRevision:
        important = [
            event for event in context.events if event.kind.casefold() in _PLAN_INVALIDATING_EVENTS
        ]
        if plan.invalidated_reason or len(important) >= 2:
            return PlanRevision(
                PlanRevisionKind.REPLAN,
                plan=await self.create_plan(context),
                reason=plan.invalidated_reason or "multiple strategically relevant events",
            )
        if important:
            event = important[-1]
            next_node = (
                "combat"
                if event.kind.casefold() in {"attackersdeclared", "blockersdeclared"}
                else "respond"
            )
            return PlanRevision(
                PlanRevisionKind.ADVANCE,
                next_node_id=next_node,
                reason=f"branch selected by {event.kind}",
            )
        return PlanRevision(PlanRevisionKind.KEEP)

    def _policy_batch(self, decision: Decision) -> tuple[PolicyBatchV15, list[dict[str, Any]]]:
        actions = [action.raw for action in decision.actions]
        graph = self.encoder.encode(copy.deepcopy(decision.game.raw), decision.player_id)
        action_features = self.encoder.encode_actions(
            actions, decision.game.raw, decision.player_id
        )
        edge_index = (
            torch.cat((torch.zeros((1, graph.edges.shape[1]), dtype=torch.long), graph.edges), 0)
            if graph.edges.numel()
            else torch.empty((3, 0), dtype=torch.long)
        )
        batch = PolicyBatchV15(
            node_features=graph.node_features.unsqueeze(0).to(self.device),
            node_types=graph.node_types.unsqueeze(0).to(self.device),
            owners=graph.owners.unsqueeze(0).to(self.device),
            node_mask=torch.ones(
                (1, graph.node_features.shape[0]), dtype=torch.bool, device=self.device
            ),
            legal_action_features=action_features.unsqueeze(0).to(self.device),
            legal_action_mask=torch.ones((1, len(actions)), dtype=torch.bool, device=self.device),
            edge_index=edge_index.to(self.device),
            edge_types=graph.edge_types.to(self.device),
            text_tokens=graph.text_tokens.unsqueeze(0).to(self.device),
            text_mask=graph.text_mask.unsqueeze(0).to(self.device),
            controllers=graph.controllers.unsqueeze(0).to(self.device),
            zones=graph.zones.unsqueeze(0).to(self.device),
            positions=graph.positions.unsqueeze(0).to(self.device),
        )
        return batch, actions

    async def select_intent(
        self, decision: Decision, plan: ActivePlan, budget: ComputeBudget
    ) -> ActionIntent:
        batch, actions = self._policy_batch(decision)
        with torch.no_grad():
            logits, value, state = self.model.policy_value(batch)
            scores = logits
            uncertainty = torch.zeros_like(logits)
            if budget.mode is ComputeMode.STRATEGIC:
                imagined, uncertainty = self.model.score_imagined_actions(
                    state, batch.legal_action_features
                )
                scores = logits + imagined - 0.1 * uncertainty
            index = int(scores.argmax(dim=-1).item())
        selected = actions[index]
        self.last_inference = {
            "planId": plan.graph.plan_id,
            "planRevision": plan.revision,
            "nodeId": plan.current_node_id,
            "computeMode": budget.mode.value,
            "budgetMs": budget.budget_ms,
            "stateValue": float(value.item()),
            "uncertainty": float(uncertainty[0, index].item()),
            "actionId": selected.get("id"),
        }
        source = selected.get("cardInstanceId", selected.get("sourceId"))
        return ActionIntent(
            kind=str(selected.get("kind", "")),
            source_id=str(source) if source else None,
            engine_action_id=str(selected.get("id", "")),
            parameters={"planNodeId": plan.current_node_id},
            timing=TimingIntent("now"),
        )

    async def make_decision(self, decision: Decision) -> DecisionResponse:
        choice_kind = str((decision.choice or {}).get("kind", ""))
        if choice_kind in {"cardSelection", "cardOrder", "cardNameSelection", "numberSelection"}:
            return await Agent.make_decision(self, decision)
        return await super().make_decision(decision)


def load_v15_agent(
    checkpoint: str | Path,
    *,
    device: torch.device | str = "cpu",
) -> V15Agent:
    checkpoint_file = Path(checkpoint) / "v15-model.pt"
    if not checkpoint_file.is_file():
        raise ValueError(f"V15 checkpoint is missing {checkpoint_file.name}")
    payload = torch.load(checkpoint_file, map_location=device, weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("V15 checkpoint payload must be an object")
    raw_config = payload.get("model_config")
    state = payload.get("model")
    if not isinstance(raw_config, dict) or not isinstance(state, dict):
        raise ValueError("V15 checkpoint is missing its model configuration or weights")
    model = TypedLatentWorldModelV15(ModelConfigV15(**cast(dict[str, Any], raw_config)))
    if payload.get("observation_schema") != model.observation_schema:
        raise ValueError("V15 checkpoint uses an incompatible observation schema")
    model.load_state_dict(state)
    return V15Agent(model, device=device)
