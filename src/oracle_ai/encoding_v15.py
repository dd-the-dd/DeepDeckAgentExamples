from __future__ import annotations

from typing import Any

import torch

from oracle_ai.encoding_v13 import GraphObservationEncoderV13


class GraphObservationEncoderV15(GraphObservationEncoderV13):
    """Observer-safe V15 state plus typed action/target/timing features."""

    action_feature_dim = 32
    schema_version = "typed-latent-observation/v15"
    action_schema_version = "typed-latent-action/v15"

    def __init__(self, feature_dim: int = 32, action_feature_dim: int = 32) -> None:
        if action_feature_dim < 32:
            raise ValueError("V15 requires at least 32 action features")
        super().__init__(feature_dim=feature_dim, action_feature_dim=24)
        self.action_feature_dim = action_feature_dim

    @staticmethod
    def _target_count(action: dict[str, Any]) -> int:
        targets = action.get("targets", {})
        if isinstance(targets, dict):
            return len(targets)
        values = action.get("targetIds", [])
        return len(values) if isinstance(values, list) else 0

    def encode_actions(
        self,
        actions: list[dict[str, Any]],
        state: dict[str, Any] | None = None,
        observer_id: str = "",
    ) -> torch.Tensor:
        # V13 consults the instance width when allocating its compatibility
        # tensor. V15 retains exactly its first 24 semantic channels and owns
        # the remaining typed target/timing channels below.
        base = super().encode_actions(actions, state, observer_id)[:, :24]
        extras = torch.zeros((len(actions), self.action_feature_dim - 24), dtype=torch.float32)
        for row, action in enumerate(actions):
            kind = str(action.get("kind", "")).casefold()
            choice = action.get("choice", action.get("decisions", {}))
            extras[row, 0] = min(1.0, self._target_count(action) / 4.0)
            extras[row, 1] = float(bool(action.get("cardInstanceId")))
            extras[row, 2] = float(bool(action.get("paymentSources")))
            extras[row, 3] = float(isinstance(choice, dict) and bool(choice))
            extras[row, 4] = float(kind in {"castspell", "playland", "declareattacker"})
            extras[row, 5] = float(kind in {"passpriority", "finishattackers", "finishblockers"})
            extras[row, 6] = float(kind in {"activateability", "castspell"})
            extras[row, 7] = float(kind in {"counterspell", "choosecounter", "redirect"})
        return torch.cat((base, extras), dim=-1)
