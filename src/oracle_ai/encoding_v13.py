from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import torch


class GraphNodeType(IntEnum):
    PLAYER = 0
    ZONE = 1
    CARD = 2
    STACK_OBJECT = 3
    DECISION = 4
    ABILITY = 5


class GraphRelation(IntEnum):
    IN_ZONE = 0
    OWNED_BY = 1
    CONTROLLED_BY = 2
    STACK_ORDER = 3
    TARGETS = 4


@dataclass(frozen=True)
class GraphObservation:
    node_features: torch.Tensor
    node_types: torch.Tensor
    owners: torch.Tensor
    controllers: torch.Tensor
    zones: torch.Tensor
    positions: torch.Tensor
    edges: torch.Tensor
    edge_types: torch.Tensor
    node_keys: tuple[str, ...]
    text_tokens: torch.Tensor
    text_mask: torch.Tensor


class GraphObservationEncoderV13:
    """Project observer-authorized state into an ordered structured sequence.

    Hidden zones contribute counts through their zone node. Card identities are
    emitted only for the observer, public zones, or explicitly revealed cards.
    The historic class name is retained for API compatibility, but V13.2 no
    longer builds message-passing edges: a Transformer receives explicit token
    type, owner, controller, zone and within-zone position channels.
    """

    feature_dim = 32
    action_feature_dim = 24
    text_vocabulary_size = 8192
    max_oracle_tokens = 128
    schema_version = "structured-observation/v13.2"
    action_schema_version = "decomposed-action/v13.2"
    zones = ("library", "hand", "battlefield", "graveyard", "exile", "commandZone")
    public_zones = frozenset(("battlefield", "graveyard", "exile", "commandZone"))

    def __init__(self, feature_dim: int = 32, action_feature_dim: int = 24) -> None:
        self.feature_dim = feature_dim
        self.action_feature_dim = action_feature_dim

    @staticmethod
    def _number(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _text(value: Any) -> str:
        return json.dumps(value, sort_keys=True, default=str).casefold()

    def _oracle_tokens(self, definition: dict[str, Any]) -> list[int]:
        oracle = definition.get("oracleText", definition.get("text"))
        if not isinstance(oracle, str) or not oracle.strip():
            oracle = json.dumps(
                definition.get("rules", []), sort_keys=True, separators=(",", ":"), default=str
            )
        document = " ".join(
            (
                str(definition.get("typeLine", "")),
                str(definition.get("manaCost", "")),
                oracle,
            )
        ).casefold()
        pieces = re.findall(r"\{[^}]+\}|[a-z0-9]+|[^\w\s]", document)
        token_ids = [
            1
            + int.from_bytes(
                hashlib.blake2b(piece.encode("utf-8"), digest_size=4).digest(), "little"
            )
            % (self.text_vocabulary_size - 1)
            for piece in pieces
        ]
        if len(token_ids) <= self.max_oracle_tokens:
            return token_ids
        head = self.max_oracle_tokens * 3 // 4
        return token_ids[:head] + token_ids[-(self.max_oracle_tokens - head) :]

    def _text_tokens(self, text: str) -> list[int]:
        pieces = re.findall(r"\{[^}]+\}|[a-z0-9]+|[^\w\s]", text.casefold())
        token_ids = [
            1
            + int.from_bytes(
                hashlib.blake2b(piece.encode("utf-8"), digest_size=4).digest(), "little"
            )
            % (self.text_vocabulary_size - 1)
            for piece in pieces
        ]
        if len(token_ids) <= self.max_oracle_tokens:
            return token_ids
        head = self.max_oracle_tokens * 3 // 4
        return token_ids[:head] + token_ids[-(self.max_oracle_tokens - head) :]

    def _abilities(self, definition: dict[str, Any]) -> list[str]:
        rules = definition.get("rules")
        if isinstance(rules, list) and rules:
            return [self._text(rule) for rule in rules]
        oracle = definition.get("oracleText", definition.get("text"))
        if not isinstance(oracle, str):
            return []
        return [line.strip() for line in oracle.splitlines() if line.strip()]

    def _ability_features(self, ability: str) -> list[float]:
        text = ability.casefold()
        # Trigger/timing, non-mana costs, then produced effects. These labels
        # give the Oracle encoder useful auxiliary structure without replacing
        # the full variable-length text.
        return [
            float("when " in text or "whenever " in text or "at the " in text),
            float(":" in text),
            float("instant" in text or "any time" in text),
            float("sorcery" in text or "only during" in text),
            float("{" in text),
            float("tap" in text),
            float("sacrifice" in text),
            float("discard" in text),
            float("pay " in text and "life" in text),
            float("exile" in text and ("cost" in text or ":" in text)),
            float("draw" in text),
            float("damage" in text),
            float("destroy" in text),
            float("exile" in text),
            float("counter target" in text),
            float("add {" in text or "add one mana" in text),
            float("search" in text),
            float("create" in text and "token" in text),
            float("gain" in text and "life" in text),
            float("counter" in text and "counter target" not in text),
        ]

    @staticmethod
    def _seat(players: list[dict[str, Any]], player_id: Any) -> int:
        return next(
            (index for index, player in enumerate(players) if str(player.get("id")) == str(player_id)),
            len(players),
        )

    def _card_features(self, card: dict[str, Any], position: int) -> list[float]:
        definition = card.get("definition")
        definition = definition if isinstance(definition, dict) else card
        type_line = str(definition.get("typeLine", "")).casefold()
        mana_cost = str(definition.get("manaCost", "")).upper()
        colors = definition.get("colors", [])
        color_text = "".join(str(value).upper() for value in colors) + mana_cost
        rules_text = self._text(definition.get("rules", []))
        counters = card.get("counters", {})
        counter_count = (
            sum(self._number(value) for value in counters.values())
            if isinstance(counters, dict)
            else self._number(counters)
        )
        effects = (
            ("addmana", "mana"),
            ("draw", "draw"),
            ("destroy", "destroy"),
            ("exile", "exile"),
            ("dealdamage", "damage"),
            ("counterstack", "counter"),
            ("graveyard", "graveyard"),
            ("search", "search"),
        )
        return [
            self._number(definition.get("manaValue")),
            self._number(card.get("power", definition.get("power"))),
            self._number(card.get("toughness", definition.get("toughness"))),
            counter_count,
            float(bool(card.get("tapped"))),
            float(position),
            float(bool(card.get("summoningSick"))),
            self._number(card.get("damageMarked")),
            *(float(color in color_text) for color in "WUBRG"),
            *(float(card_type in type_line) for card_type in (
                "land", "creature", "artifact", "enchantment", "instant",
                "sorcery", "planeswalker", "battle",
            )),
            float("legendary" in type_line),
            float(bool(card.get("token", card.get("isToken", False)))),
            *(float(any(term in rules_text for term in terms)) for terms in effects),
        ]

    def _decision_features(
        self, state: dict[str, Any], players: list[dict[str, Any]], observer_id: str
    ) -> list[float]:
        active = state.get("activePlayer")
        priority = state.get("priorityPlayer")
        observer_seat = self._seat(players, observer_id)
        active_seat = self._seat(players, active) if isinstance(active, str) else int(active or 0)
        priority_seat = (
            self._seat(players, priority) if isinstance(priority, str) else int(priority or 0)
        )
        steps = (
            "untap", "upkeep", "draw", "precombatmain", "begincombat", "declareattackers",
            "declareblockers", "combatdamage", "endcombat", "postcombatmain", "end", "cleanup",
        )
        step = str(state.get("step", "")).replace("-", "").casefold()
        return [
            self._number(state.get("turnNumber", state.get("turn"))),
            self._number(state.get("roundNumber", state.get("round"))),
            float(active_seat == observer_seat),
            float(priority_seat == observer_seat),
            float(len(state.get("stack", []))),
            *(float(step == candidate) for candidate in steps),
            float(len(players)),
        ]

    def encode(self, state: dict[str, Any], observer_id: str) -> GraphObservation:
        features: list[list[float]] = []
        types: list[int] = []
        owners: list[int] = []
        controllers: list[int] = []
        zone_ids: list[int] = []
        positions: list[int] = []
        keys: list[str] = []
        edges: list[tuple[int, int]] = []
        relations: list[int] = []
        texts: list[list[int]] = []

        def add(
            key: str,
            node_type: GraphNodeType,
            owner: int,
            values: list[float],
            text_tokens: list[int] | None = None,
            *,
            controller: int | None = None,
            zone: int = 0,
            position: int = 0,
        ) -> int:
            index = len(features)
            features.append((values + [0.0] * self.feature_dim)[: self.feature_dim])
            types.append(int(node_type))
            owners.append(owner)
            controllers.append(owner if controller is None else controller)
            zone_ids.append(zone)
            positions.append(position)
            keys.append(key)
            texts.append(text_tokens or [])
            return index

        def connect(source: int, target: int, relation: GraphRelation) -> None:
            # Kept as a no-op while older encoder call sites still describe
            # relationships. V13.2 represents them through explicit channels.
            return None

        players = [item for item in state.get("players", []) if isinstance(item, dict)]
        decision_values = self._decision_features(state, players, observer_id)
        add("decision:state", GraphNodeType.DECISION, 0, decision_values)

        player_nodes: dict[str, int] = {}
        for seat, player in enumerate(players):
            player_id = str(player.get("id", seat))
            player_nodes[player_id] = add(
                f"player:{player_id}", GraphNodeType.PLAYER, seat,
                [
                    self._number(player.get("life")),
                    self._number(player.get("poisonCounters")),
                    float(player_id == observer_id),
                    self._number(player.get("landPlaysRemaining")),
                    float(len(player.get("manaPool", []))),
                ],
            )
        for seat, player in enumerate(players):
            player_id = str(player.get("id", seat))
            for zone_number, zone in enumerate(self.zones):
                cards = player.get(zone, [])
                cards = cards if isinstance(cards, list) else []
                zone_node = add(
                    f"zone:{player_id}:{zone}", GraphNodeType.ZONE, seat,
                    [
                        float(len(cards)),
                        *(float(zone_number == index) for index in range(len(self.zones))),
                        float(zone in self.public_zones),
                        float(player_id == observer_id),
                    ],
                    zone=zone_number + 1,
                )
                connect(zone_node, player_nodes[player_id], GraphRelation.OWNED_BY)
                for position, card in enumerate(cards):
                    if not isinstance(card, dict):
                        continue
                    visible = (
                        player_id == observer_id
                        or zone in self.public_zones
                        or bool(card.get("revealed", card.get("isRevealed", False)))
                    )
                    if not visible:
                        continue
                    card_node = add(
                        f"card:{card.get('instanceId', f'{player_id}:{zone}:{position}')}",
                        GraphNodeType.CARD,
                        seat,
                        self._card_features(card, position),
                        self._oracle_tokens(
                            card.get("definition")
                            if isinstance(card.get("definition"), dict)
                            else card
                        ),
                        controller=self._seat(
                            players,
                            card.get("controller", card.get("controllerId", player_id)),
                        ),
                        zone=zone_number + 1,
                        position=position + 1,
                    )
                    connect(card_node, zone_node, GraphRelation.IN_ZONE)
                    connect(card_node, player_nodes[player_id], GraphRelation.OWNED_BY)
                    controller = str(card.get("controller", card.get("controllerId", player_id)))
                    if controller in player_nodes:
                        connect(card_node, player_nodes[controller], GraphRelation.CONTROLLED_BY)
                    definition = (
                        card.get("definition")
                        if isinstance(card.get("definition"), dict)
                        else card
                    )
                    for ability_index, ability in enumerate(self._abilities(definition)):
                        add(
                            f"ability:{keys[card_node]}:{ability_index}",
                            GraphNodeType.ABILITY,
                            seat,
                            self._ability_features(ability),
                            self._text_tokens(ability),
                            controller=self._seat(players, controller),
                            zone=zone_number + 1,
                            position=position + 1,
                        )
        previous_stack: int | None = None
        for position, item in enumerate(state.get("stack", [])):
            if not isinstance(item, dict):
                continue
            controller = str(item.get("controllerId", item.get("playerId", "")))
            seat = next(
                (index for index, player in enumerate(players) if str(player.get("id")) == controller),
                len(players),
            )
            stack_node = add(
                f"stack:{item.get('id', position)}", GraphNodeType.STACK_OBJECT, seat,
                [
                    float(position),
                    self._number(item.get("manaValue")),
                    float(controller == observer_id),
                    float(str(item.get("kind", "")).casefold() == "spell"),
                ],
                controller=seat,
                zone=len(self.zones) + 1,
                position=position + 1,
            )
            if previous_stack is not None:
                connect(previous_stack, stack_node, GraphRelation.STACK_ORDER)
            previous_stack = stack_node
        edge_tensor = (
            torch.tensor(edges, dtype=torch.long).t().contiguous()
            if edges else torch.empty((2, 0), dtype=torch.long)
        )
        text_width = max(1, max((len(tokens) for tokens in texts), default=0))
        text_tensor = torch.zeros((len(texts), text_width), dtype=torch.long)
        text_mask = torch.zeros((len(texts), text_width), dtype=torch.bool)
        for row, tokens in enumerate(texts):
            if tokens:
                text_tensor[row, : len(tokens)] = torch.tensor(tokens, dtype=torch.long)
                text_mask[row, : len(tokens)] = True
        return GraphObservation(
            node_features=torch.tensor(features, dtype=torch.float32),
            node_types=torch.tensor(types, dtype=torch.long),
            owners=torch.tensor(owners, dtype=torch.long),
            controllers=torch.tensor(controllers, dtype=torch.long),
            zones=torch.tensor(zone_ids, dtype=torch.long),
            positions=torch.tensor(positions, dtype=torch.long),
            edges=edge_tensor,
            edge_types=torch.tensor(relations, dtype=torch.long),
            node_keys=tuple(keys),
            text_tokens=text_tensor,
            text_mask=text_mask,
        )

    def encode_actions(
        self,
        actions: list[dict[str, Any]],
        state: dict[str, Any] | None = None,
        observer_id: str = "",
    ) -> torch.Tensor:
        """Encode stable action semantics rather than volatile instance identifiers."""

        state = state or {}
        players = [item for item in state.get("players", []) if isinstance(item, dict)]
        cards: dict[str, dict[str, Any]] = {}
        for player in players:
            for zone in self.zones:
                for card in player.get(zone, []) if isinstance(player.get(zone, []), list) else []:
                    if isinstance(card, dict) and card.get("instanceId"):
                        cards[str(card["instanceId"])] = card
        kind_slots = {
            "passpriority": 1,
            "finishattackers": 1,
            "finishblockers": 1,
            "playland": 2,
            "castspell": 3,
            "activateability": 4,
            "specialaction": 5,
            "declareattacker": 6,
            "declareblocker": 7,
        }
        encoded: list[list[float]] = []
        for action in actions:
            values = [0.0] * self.action_feature_dim
            def set_value(index: int, value: float) -> None:
                if index < len(values):
                    values[index] = value

            set_value(0, 1.0)
            kind = str(action.get("kind", "")).casefold()
            set_value(kind_slots.get(kind, min(8, self.action_feature_dim - 1)), 1.0)
            source = cards.get(str(action.get("cardInstanceId", "")), {})
            source_features = self._card_features(source, 0) if source else [0.0] * 32
            set_value(9, source_features[0])
            for offset, color_value in enumerate(source_features[8:13], start=10):
                set_value(offset, color_value)
            set_value(15, float(any(source_features[13:21])))
            targets = action.get("targets", {})
            targets = targets if isinstance(targets, dict) else {}
            for target in targets.values():
                if not isinstance(target, dict):
                    continue
                target_player = str(target.get("playerId", ""))
                set_value(16, float(bool(target_player) and target_player != observer_id))
                set_value(17, float(target_player == observer_id))
                set_value(18, float(target.get("kind") == "permanent" or bool(target.get("instanceId"))))
                set_value(19, float(target.get("kind") == "stackObject" or bool(target.get("stackId"))))
            set_value(20, float(len(action.get("paymentSources", []))))
            decisions = action.get("decisions", {})
            decisions = decisions if isinstance(decisions, dict) else {}
            set_value(21, float(bool(decisions.get("manaAbility"))))
            payments = decisions.get("manaPayment", [])
            if isinstance(payments, list):
                set_value(22, sum(self._number(item.get("lifePaid")) for item in payments if isinstance(item, dict)))
            set_value(23, self._number(action.get("numberValue", action.get("amount"))))
            encoded.append(values)
        if not encoded:
            return torch.empty((0, self.action_feature_dim), dtype=torch.float32)
        return torch.tensor(encoded, dtype=torch.float32)


# Public name for new code. The legacy name above remains import-compatible
# with already configured learner jobs.
StructuredObservationEncoderV13 = GraphObservationEncoderV13
StructuredObservationV13 = GraphObservation
