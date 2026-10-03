from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from deepdeck_agent import Action, Agent, Card, Decision, DecisionResult


class ForgeReferenceProfile(str, Enum):
    """Clean-room policy profiles for the Forge-inspired reference agent."""

    CAUTIOUS = "cautious"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"


@dataclass(frozen=True)
class _Profile:
    attack_margin: int
    trade_bonus: int
    instant_threshold: int


_PROFILES = {
    ForgeReferenceProfile.CAUTIOUS: _Profile(
        attack_margin=2,
        trade_bonus=-15,
        instant_threshold=145,
    ),
    ForgeReferenceProfile.BALANCED: _Profile(
        attack_margin=0,
        trade_bonus=10,
        instant_threshold=115,
    ),
    ForgeReferenceProfile.AGGRESSIVE: _Profile(
        attack_margin=-2,
        trade_bonus=35,
        instant_threshold=90,
    ),
}

_PERMANENT_TYPES = ("Artifact", "Battle", "Creature", "Enchantment", "Planeswalker")
_REMOVAL_EFFECTS = (
    "destroyPermanent",
    "exilePermanent",
    "moveToGraveyard",
    "moveToExile",
    "returnToHand",
    "dealDamage",
)
_VALUE_EFFECTS = (
    "addCounter",
    "createToken",
    "drawCards",
    "gainLife",
    "searchLibrary",
)


def _mana_value(card: Card | None) -> int:
    if card is None:
        return 0
    result = 0
    for symbol in re.findall(r"\{([^}]+)\}", card.mana_cost):
        if symbol.isdigit():
            result += int(symbol)
        elif symbol.upper() not in {"X", "Y", "Z"}:
            result += 1
    return result


def _is_permanent(card: Card | None) -> bool:
    return card is not None and any(card.is_type(card_type) for card_type in _PERMANENT_TYPES)


def _card_value(card: Card | None) -> int:
    if card is None:
        return 0
    value = 35 + 8 * _mana_value(card)
    if card.is_type("Land"):
        value = 55
    if card.is_type("Creature"):
        value += 10 * max(card.power, 0) + 6 * max(card.toughness, 0)
    if card.is_type("Planeswalker"):
        value += 35
    if card.rules_contain(*_VALUE_EFFECTS):
        value += 25
    if card.rules_contain(*_REMOVAL_EFFECTS):
        value += 30
    return value


def _opponent_target_value(decision: Decision, action: Action) -> int:
    opponents = {player.id for player in decision.game.opponents}
    return sum(
        _card_value(permanent)
        for permanent in decision.target_permanents(action)
        if permanent.controller in opponents
    )


def _friendly_target_value(decision: Decision, action: Action) -> int:
    return sum(
        _card_value(permanent)
        for permanent in decision.target_permanents(action)
        if permanent.controller == decision.player_id
    )


class ForgeReferenceAgent(Agent):
    """A transparent, clean-room heuristic agent inspired by Forge's AI shape.

    No Forge source code is copied. DeepDeck's engine remains responsible for
    rules and legality; this agent only ranks the exact legal actions it receives.
    """

    def __init__(self, profile: ForgeReferenceProfile | str = ForgeReferenceProfile.BALANCED):
        self.profile = ForgeReferenceProfile(profile)
        self._settings = _PROFILES[self.profile]

    async def choose_mulligan(self, decision: Decision) -> DecisionResult:
        hand = decision.game.me.hand
        land_count = sum(card.is_type("Land") for card in hand)
        nonlands = [card for card in hand if not card.is_type("Land")]
        castable_early = any(_mana_value(card) <= 3 for card in nonlands)
        keep = 2 <= land_count <= 5 and (castable_early or not nonlands)
        preferred = "keepHand" if keep else "takeMulligan"
        return decision.first(preferred) or self._safe_default(decision)

    async def choose_mulligan_bottom(self, decision: Decision) -> DecisionResult:
        candidates = decision.actions_of("bottomCard")
        if not candidates:
            return self._safe_default(decision)
        lands = sum(card.is_type("Land") for card in decision.game.me.hand)

        def bottom_score(action: Action) -> tuple[int, int, str]:
            card = decision.card_for(action)
            excess_land = int(card is not None and card.is_type("Land") and lands > 3)
            return (excess_land, _mana_value(card), action.id)

        return max(candidates, key=bottom_score)

    def _priority_score(self, decision: Decision, action: Action) -> int:
        if action.kind == "playLand":
            return 10_000
        if action.kind not in {"castSpell", "activateAbility", "specialAction"}:
            return -10_000

        source = decision.card_for(action)
        score = _card_value(source)
        opponent_value = _opponent_target_value(decision, action)
        friendly_value = _friendly_target_value(decision, action)
        is_removal = source is not None and source.rules_contain(*_REMOVAL_EFFECTS)

        if is_removal:
            score += opponent_value - 2 * friendly_value
        else:
            score += friendly_value // 3
            score -= opponent_value // 4
        score -= 3 * len(set(action.payment_sources))

        if decision.game.stack and action.target_stack_ids:
            score += 100
        if _is_permanent(source) and decision.game.is_my_turn:
            score += 30
        if not decision.game.is_my_turn and source is not None and not source.is_type("Instant"):
            score -= 45
        return score

    async def choose_priority(self, decision: Decision) -> DecisionResult:
        candidates = [
            action
            for action in decision.actions
            if action.kind in {"playLand", "castSpell", "activateAbility", "specialAction"}
        ]
        if not candidates:
            return decision.pass_action or self._safe_default(decision)
        best = max(
            candidates,
            key=lambda action: (self._priority_score(decision, action), action.id),
        )
        threshold = 1 if decision.game.is_my_turn else self._settings.instant_threshold
        if self._priority_score(decision, best) < threshold:
            return decision.pass_action or best
        return best

    def _attack_score(self, decision: Decision, action: Action) -> int:
        attacker = decision.game.permanent(action.attacker_id)
        if attacker is None:
            return -10_000
        target_life = min(
            (
                player.life
                for player_id in action.target_player_ids
                if (player := decision.game.player(player_id)) is not None
            ),
            default=10**6,
        )
        blockers = [
            card
            for opponent in decision.game.opponents
            for card in opponent.battlefield
            if card.is_type("Creature") and not card.tapped
        ]
        if target_life <= attacker.power:
            return 10_000
        if not blockers or attacker.rules_contain("cantBeBlocked", "flying", "menace"):
            return 200 + attacker.power
        strongest_power = max((blocker.power for blocker in blockers), default=0)
        weakest_toughness = min((blocker.toughness for blocker in blockers), default=0)
        survives = attacker.toughness > strongest_power
        trades = attacker.power >= weakest_toughness
        return (
            attacker.power
            + (60 if survives else 0)
            + (self._settings.trade_bonus if trades else -40)
            - self._settings.attack_margin * 10
        )

    async def choose_attackers(self, decision: Decision) -> DecisionResult:
        attacks = decision.actions_of("declareAttacker")
        if not attacks:
            return decision.first("finishAttackers") or self._safe_default(decision)
        best = max(attacks, key=lambda action: (self._attack_score(decision, action), action.id))
        if self._attack_score(decision, best) <= 0:
            return decision.first("finishAttackers") or best
        return best

    async def choose_blockers(self, decision: Decision) -> DecisionResult:
        blocks = decision.actions_of("declareBlocker")
        if not blocks:
            return decision.first("finishBlockers") or self._safe_default(decision)

        def block_score(action: Action) -> tuple[int, int, int, str]:
            blocker = decision.game.permanent(action.blocker_id)
            attacker = decision.game.permanent(action.attacker_id)
            if blocker is None or attacker is None:
                return (-10_000, 0, 0, action.id)
            kills = blocker.power >= attacker.toughness
            survives = blocker.toughness > attacker.power
            prevents_lethal = attacker.power >= decision.game.me.life
            value = (
                (500 if prevents_lethal else 0)
                + (120 if kills else 0)
                + (80 if survives else 0)
                + attacker.power * 8
                - blocker.power * (3 if survives else 7)
            )
            return (value, int(kills), int(survives), action.id)

        best = max(blocks, key=block_score)
        if block_score(best)[0] <= 0:
            return decision.first("finishBlockers") or best
        return best

    async def choose_discard(self, decision: Decision) -> DecisionResult:
        discards = decision.actions_of("discard")
        if not discards:
            return self._safe_default(decision)
        lands = sum(card.is_type("Land") for card in decision.game.me.hand)

        def discard_value(action: Action) -> tuple[int, str]:
            card = decision.card_for(action)
            value = _card_value(card)
            if card is not None and card.is_type("Land") and lands > 4:
                value -= 45
            return (value, action.id)

        return min(discards, key=discard_value)


def build_forge_reference_agent(
    profile: ForgeReferenceProfile | str = ForgeReferenceProfile.BALANCED,
) -> ForgeReferenceAgent:
    return ForgeReferenceAgent(profile)
