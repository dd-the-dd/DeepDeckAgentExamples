from __future__ import annotations

import json
import math
import random
from collections import Counter
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from oracle_ai.training.environments import Matchup
from oracle_ai.training.league import RandomTrainingMatchupSampler


@dataclass(frozen=True)
class CurriculumScenario:
    id: str
    family: str
    label: str
    description: str
    difficulty: int
    objective: str
    revision: int = 1
    target_round: int | None = None
    opening_hand_pool_size: int | None = None
    opening_hand_candidate_cards: tuple[str, ...] = ()
    opening_hand_roles: tuple[tuple[str, ...], ...] = ()
    free_casts: bool = False
    miracle_pressure: int = 0
    legacy_match: bool = False
    deck_hint: str | None = None
    opponent_deck_hint: str | None = None
    fixed_hand: tuple[str, ...] = ()
    success_cards: tuple[str, ...] = ()
    success_zones: tuple[str, ...] = ("battlefield",)
    success_action_sequence: tuple[str, ...] = ()
    sideboard_target_cards: tuple[str, ...] = ()
    sideboard_cut_cards: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()

    def public(self) -> dict[str, Any]:
        value = asdict(self)
        value["tags"] = list(self.tags)
        return value


BUILT_IN_CURRICULUM: tuple[CurriculumScenario, ...] = (
    CurriculumScenario(
        id="opening-hand-sprint-easy",
        family="opening-hand",
        label="Opening hand sprint · easy",
        revision=2,
        description="Choose seven cards from twenty and defeat a passive anchor before round 8.",
        difficulty=1,
        objective="fast-win",
        target_round=8,
        opening_hand_pool_size=20,
        tags=("opening-hand", "lethal", "planning"),
    ),
    CurriculumScenario(
        id="opening-hand-sprint-hard",
        family="opening-hand",
        label="Opening hand sprint · hard",
        revision=2,
        description="Choose seven cards from sixty and assemble a win before round 4.",
        difficulty=4,
        objective="fast-win",
        target_round=4,
        opening_hand_pool_size=60,
        tags=("opening-hand", "lethal", "search"),
    ),
    CurriculumScenario(
        id="opening-hand-sprint-medium",
        family="opening-hand",
        label="Opening hand sprint · medium",
        revision=1,
        description="Choose seven cards from forty and defeat a passive anchor before round 6.",
        difficulty=2,
        objective="fast-win",
        target_round=6,
        opening_hand_pool_size=40,
        tags=("opening-hand", "lethal", "planning"),
    ),
    CurriculumScenario(
        id="opening-hand-reanimator-clinic",
        family="opening-hand",
        label="Opening hand clinic · Reanimator",
        revision=2,
        description=(
            "Build a functional seven-card Reanimator hand from a controlled twenty-card "
            "pool, then convert it into a fast win."
        ),
        difficulty=2,
        objective="fast-win",
        target_round=3,
        opening_hand_pool_size=20,
        deck_hint="Radkos reanimator",
        opening_hand_candidate_cards=(
            "Swamp",
            "Swamp",
            "Swamp",
            "Badlands",
            "Badlands",
            "Lotus Petal",
            "Lotus Petal",
            "Dark Ritual",
            "Faithless Looting",
            "Faithless Looting",
            "Thoughtseize",
            "Unmask",
            "Griselbrand",
            "Griselbrand",
            "Atraxa, Grand Unifier",
            "Archon of Cruelty",
            "Reanimate",
            "Reanimate",
            "Animate Dead",
            "Shallow Grave",
        ),
        opening_hand_roles=(
            ("Swamp", "Badlands", "Lotus Petal", "Dark Ritual"),
            ("Faithless Looting", "Thoughtseize", "Unmask"),
            ("Griselbrand", "Atraxa, Grand Unifier", "Archon of Cruelty"),
            ("Reanimate", "Animate Dead", "Shallow Grave"),
        ),
        tags=("opening-hand", "reanimator", "dense-reward", "planning"),
    ),
    CurriculumScenario(
        id="zero-cost-sequencing",
        family="free-cast",
        label="Zero-cost sequencing",
        revision=2,
        description="Every nonland card in the learner deck costs zero; win quickly by ordering actions correctly.",
        difficulty=2,
        objective="fast-win",
        target_round=4,
        opening_hand_pool_size=30,
        free_casts=True,
        tags=("free-cast", "sequencing", "stack"),
    ),
    CurriculumScenario(
        id="zero-cost-sequencing-easy",
        family="free-cast",
        label="Zero-cost sequencing · easy",
        revision=1,
        description="Sequence free nonland cards from a small pool and win before round 6.",
        difficulty=1,
        objective="fast-win",
        target_round=6,
        opening_hand_pool_size=20,
        free_casts=True,
        tags=("free-cast", "sequencing", "foundation"),
    ),
    CurriculumScenario(
        id="zero-cost-sequencing-hard",
        family="free-cast",
        label="Zero-cost sequencing · hard",
        revision=1,
        description="Find a winning order among many free actions before round 3.",
        difficulty=4,
        objective="fast-win",
        target_round=3,
        opening_hand_pool_size=50,
        free_casts=True,
        tags=("free-cast", "sequencing", "stack", "search"),
    ),
    CurriculumScenario(
        id="combo-rehearsal",
        family="combo",
        label="Combo rehearsal",
        revision=2,
        description="A large opening-hand pool and free spells isolate combo discovery and execution.",
        difficulty=3,
        objective="combo-win",
        target_round=3,
        opening_hand_pool_size=60,
        free_casts=True,
        tags=("combo", "opening-hand", "lethal"),
    ),
    CurriculumScenario(
        id="combo-rehearsal-guided",
        family="combo",
        label="Combo rehearsal · guided",
        revision=1,
        description="Discover a combo from a thirty-card pool with a forgiving five-round clock.",
        difficulty=2,
        objective="combo-win",
        target_round=5,
        opening_hand_pool_size=30,
        free_casts=True,
        tags=("combo", "opening-hand", "foundation"),
    ),
    CurriculumScenario(
        id="combo-rehearsal-expert",
        family="combo",
        label="Combo rehearsal · expert",
        revision=1,
        description="Discover and execute a combo from sixty cards before round 2.",
        difficulty=5,
        objective="combo-win",
        target_round=2,
        opening_hand_pool_size=60,
        free_casts=True,
        tags=("combo", "opening-hand", "lethal", "search", "expert"),
    ),
    CurriculumScenario(
        id="miracle-pressure-easy",
        family="predictable-opponent",
        revision=3,
        label="Miracle clock · easy",
        description="A seeded non-neural opponent top-decks free Miracle spells that gain small amounts of life.",
        difficulty=1,
        objective="fast-win",
        target_round=8,
        opening_hand_pool_size=20,
        miracle_pressure=1,
        tags=("miracle", "deterministic", "clock"),
    ),
    CurriculumScenario(
        id="miracle-pressure-medium",
        family="predictable-opponent",
        revision=3,
        label="Miracle clock · medium",
        description="The predictable Miracle opponent gains life and applies one damage per top-deck.",
        difficulty=3,
        objective="fast-win",
        target_round=6,
        opening_hand_pool_size=30,
        miracle_pressure=2,
        tags=("miracle", "deterministic", "pressure"),
    ),
    CurriculumScenario(
        id="miracle-pressure-hard",
        family="predictable-opponent",
        revision=3,
        label="Miracle clock · hard",
        description="A short clock against free Miracle spells that gain life and damage the learner.",
        difficulty=5,
        objective="fast-win",
        target_round=4,
        opening_hand_pool_size=40,
        miracle_pressure=3,
        tags=("miracle", "deterministic", "pressure", "hard"),
    ),
    CurriculumScenario(
        id="known-sideboard-match",
        family="sideboard",
        label="Known sideboard · Dimir vs Reanimator",
        description="Practice a reproducible Dimir Tempo sideboard plan against Rakdos Reanimator.",
        difficulty=4,
        objective="match-win",
        revision=6,
        legacy_match=True,
        deck_hint="Dimir tempo",
        opponent_deck_hint="Radkos reanimator",
        sideboard_target_cards=("Duress", "Force of Negation", "Sheoldred's Edict"),
        sideboard_cut_cards=("Snuff Out", "Kaito, Bane of Nightmares", "Fatal Push"),
        tags=("legacy", "sideboard", "best-of-three"),
    ),
    CurriculumScenario(
        id="sideboard-lands-vs-sneak-show",
        family="sideboard",
        label="Known sideboard · Lands vs Sneak and Show",
        description="Learn which Lands tools matter against a fast Show and Tell combo deck.",
        difficulty=5,
        objective="match-win",
        revision=5,
        legacy_match=True,
        deck_hint="Mono-Green Lands",
        opponent_deck_hint="Sneak and Show",
        sideboard_target_cards=("Disruptor Flute", "Pithing Needle", "Choke"),
        sideboard_cut_cards=(
            "Maze of Ith",
            "The Tabernacle at Pendrell Vale",
            "Grafdigger's Cage",
            "Bojuka Bog",
            "Ghost Quarter",
        ),
        tags=("legacy", "sideboard", "combo-matchup"),
    ),
    CurriculumScenario(
        id="sideboard-initiative-vs-delver",
        family="sideboard",
        label="Known sideboard · Initiative vs Delver",
        description="Practice a fair post-board plan against a fixed Izzet Delver opponent.",
        difficulty=3,
        objective="match-win",
        revision=5,
        legacy_match=True,
        deck_hint="Mono-White Initiative",
        opponent_deck_hint="Izzet Delver",
        sideboard_target_cards=("Trinisphere", "Disruptor Flute"),
        sideboard_cut_cards=("Sheltered by Ghosts", "Anointed Peacekeeper"),
        tags=("legacy", "sideboard", "tempo-matchup"),
    ),
    CurriculumScenario(
        id="combo-reanimate-griselbrand",
        family="known-combo",
        revision=2,
        label="Known combo · Reanimate Griselbrand",
        description="Sequence discard and reanimation from a fixed Rakdos opening hand.",
        difficulty=2,
        objective="combo-trigger",
        target_round=2,
        free_casts=True,
        deck_hint="Radkos reanimator",
        fixed_hand=(
            "Faithless Looting",
            "Griselbrand",
            "Reanimate",
            "Dark Ritual",
            "Swamp",
            "Lotus Petal",
            "Thoughtseize",
        ),
        success_cards=("Griselbrand",),
        success_action_sequence=(
            "Cast Faithless Looting",
            "Discard Griselbrand",
            "Cast Reanimate",
        ),
        tags=("known-combo", "reanimator", "graveyard"),
    ),
    CurriculumScenario(
        id="combo-show-and-tell-emrakul",
        family="known-combo",
        revision=2,
        label="Known combo · Show and Tell",
        description="Deploy Emrakul from a fixed Sneak and Show opening hand.",
        difficulty=2,
        objective="combo-trigger",
        target_round=2,
        free_casts=True,
        deck_hint="Sneak and Show",
        fixed_hand=(
            "Show and Tell",
            "Emrakul, the Aeons Torn",
            "Omniscience",
            "Ancient Tomb",
            "Lotus Petal",
            "Ponder",
            "Force of Will",
        ),
        success_cards=("Emrakul, the Aeons Torn",),
        success_action_sequence=(
            "Cast Show and Tell",
            "Choose Emrakul, the Aeons Torn",
        ),
        tags=("known-combo", "show-and-tell", "cheat-into-play"),
    ),
    CurriculumScenario(
        id="combo-aluren-acererak",
        family="known-combo",
        revision=2,
        label="Known combo · Aluren and Acererak",
        description="Cast Aluren and start the Acererak dungeon loop from a fixed hand.",
        difficulty=4,
        objective="combo-trigger",
        target_round=3,
        free_casts=True,
        deck_hint="Sultai Aluren",
        fixed_hand=(
            "Aluren",
            "Acererak the Archlich",
            "Ancient Tomb",
            "Lotus Petal",
            "Brainstorm",
            "Ponder",
            "Force of Will",
        ),
        success_cards=("Acererak the Archlich",),
        success_zones=("battlefield", "graveyard", "exile"),
        success_action_sequence=("Cast Aluren", "Cast Acererak the Archlich"),
        tags=("known-combo", "aluren", "loop"),
    ),
    CurriculumScenario(
        id="combo-dark-depths",
        family="known-combo",
        revision=2,
        label="Known combo · Dark Depths",
        description="Assemble Dark Depths and Thespian's Stage to create Marit Lage.",
        difficulty=3,
        objective="combo-trigger",
        target_round=4,
        free_casts=True,
        deck_hint="Mono-Green Lands",
        fixed_hand=(
            "Dark Depths",
            "Thespian's Stage",
            "Crop Rotation",
            "Mox Diamond",
            "Forest",
            "Exploration",
            "Life from the Loam",
        ),
        success_cards=("Marit Lage",),
        success_action_sequence=(
            "Play Dark Depths as land",
            "Play Thespian's Stage as land",
            "Activate Thespian's Stage",
        ),
        tags=("known-combo", "lands", "marit-lage"),
    ),
    CurriculumScenario(
        id="combo-isochron-reversal",
        family="known-combo",
        revision=2,
        label="Known combo · Isochron Reversal",
        description="Imprint Dramatic Reversal, generate mana, then convert it into Brain Freeze.",
        difficulty=5,
        objective="combo-trigger",
        target_round=4,
        free_casts=True,
        deck_hint="Mill dd",
        fixed_hand=(
            "Isochron Scepter",
            "Dramatic Reversal",
            "Grim Monolith",
            "Mox Opal",
            "Brain Freeze",
            "Lotus Petal",
            "Ancient Tomb",
        ),
        success_cards=("Brain Freeze",),
        success_zones=("graveyard", "exile"),
        success_action_sequence=(
            "Cast Isochron Scepter",
            "Choose Dramatic Reversal",
            "Activate Isochron Scepter",
            "Cast Brain Freeze",
        ),
        tags=("known-combo", "isochron-scepter", "storm"),
    ),
)


def curriculum_catalog() -> dict[str, Any]:
    return {
        "schemaVersion": "deepdeck-training-curriculum/v1",
        "scenarios": [scenario.public() for scenario in BUILT_IN_CURRICULUM],
    }


def _passive_anchor_cards() -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    for index in range(60):
        cards.append(
            {
                "id": f"curriculum-anchor-{index % 3}",
                "name": f"Curriculum Anchor {index % 3 + 1}",
                "typeLine": ("Artifact", "Creature — Construct", "Enchantment")[index % 3],
                "manaCost": "{9}",
                "power": "0" if index % 3 == 1 else None,
                "toughness": "1" if index % 3 == 1 else None,
                "rules": [],
                "isCommander": False,
                "isToken": False,
                "isGamePiece": False,
                "isSideboard": False,
            }
        )
    return cards


def _miracle_rule() -> dict[str, Any]:
    return {
        "kind": "keywordAbility",
        "source": {"kind": "self"},
        "ability": {
            "kind": "miracle",
            "cost": {"kind": "payMana", "manaCost": "{0}"},
        },
    }


def _miracle_cards(pressure: int) -> list[dict[str, Any]]:
    controller = {"kind": "controllerOf", "object": {"kind": "self"}}
    opponents = {"kind": "opponentsOf", "player": controller}
    variants = (
        (
            "Mercy",
            "{1}{W}",
            [
                {
                    "kind": "gainLife",
                    "player": controller,
                    "amount": {"kind": "integer", "value": pressure + 1},
                }
            ],
        ),
        (
            "Bolt",
            "{3}{R}",
            [
                {
                    "kind": "dealDamageToEachOpponent",
                    "source": {"kind": "self"},
                    "amount": {"kind": "integer", "value": pressure},
                }
            ],
        ),
        (
            "Insight",
            "{5}{U}",
            [
                {
                    "kind": "drawCards",
                    "player": controller,
                    "count": {"kind": "integer", "value": min(2, pressure)},
                }
            ],
        ),
        (
            "Duress",
            "{7}{B}",
            [
                {
                    "kind": "discardCards",
                    "player": opponents,
                    "count": {"kind": "integer", "value": 1},
                }
            ],
        ),
    )
    cards: list[dict[str, Any]] = []
    for index in range(60):
        variant, mana_cost, effects = variants[index % len(variants)]
        cards.append(
            {
                "id": f"curriculum-miracle-{pressure}-{index % 4}",
                "name": f"Miracle {variant} · level {pressure}",
                "typeLine": "Sorcery",
                "manaCost": mana_cost,
                "power": None,
                "toughness": None,
                "rules": [
                    {
                        "kind": "spellAbility",
                        "source": {"kind": "self"},
                        "effects": deepcopy(effects),
                    },
                    _miracle_rule(),
                ],
                "isCommander": False,
                "isToken": False,
                "isGamePiece": False,
                "isSideboard": False,
            }
        )
    return cards


def _free_cast_deck(cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = deepcopy(cards)
    for card in result:
        type_line = str(card.get("typeLine", "")).casefold()
        if "land" not in type_line and not card.get("isGamePiece") and not card.get("isToken"):
            card["manaCost"] = "{0}"
    return result


def _legacy_training_deck(cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep imported decks usable when their source has an oversized sideboard."""
    result: list[dict[str, Any]] = []
    sideboard_cards = 0
    for card in deepcopy(cards):
        type_line = str(card.get("typeLine", "")).casefold()
        is_auxiliary = (
            bool(card.get("isToken"))
            or bool(card.get("isGamePiece"))
            or "token" in type_line
        )
        if bool(card.get("isSideboard")) and not is_auxiliary:
            sideboard_cards += 1
            if sideboard_cards > 15:
                continue
        result.append(card)
    return result


def _deck_session_id(cards: list[dict[str, Any]]) -> str:
    return next(
        (
            str(card.get("sourceSessionId", ""))
            for card in cards
            if str(card.get("sourceSessionId", ""))
        ),
        "",
    )


class AdaptiveCurriculumSampler:
    """Select focused Engine games, emphasizing scenarios the learner fails."""

    def __init__(
        self,
        config: dict[str, Any],
        decks: dict[str, list[dict[str, Any]]],
        base_sampler: RandomTrainingMatchupSampler,
    ) -> None:
        requested = [str(value) for value in config.get("scenarioIds", [])]
        known = {scenario.id: scenario for scenario in BUILT_IN_CURRICULUM}
        unknown = sorted(set(requested) - set(known))
        if unknown:
            raise ValueError("unknown curriculum scenarios: " + ", ".join(unknown))
        self.scenarios = (
            [known[value] for value in requested] if requested else list(known.values())
        )
        if not self.scenarios:
            raise ValueError("training curriculum requires at least one scenario")
        self.decks = decks
        self.base_sampler = base_sampler
        self.adaptive = bool(config.get("adaptive", True))
        self.minimum_weight = max(0.05, float(config.get("minimumWeight", 0.25)))
        self.results: dict[str, dict[str, int | float]] = {
            scenario.id: {"games": 0, "wins": 0, "masterySum": 0.0} for scenario in self.scenarios
        }
        self.last_scenario: CurriculumScenario | None = None

    def template(self) -> Matchup:
        return self._build(self.scenarios[0], random.Random(0))

    def build_scenario(self, scenario_id: str, seed: int) -> Matchup:
        scenario = next(
            (scenario for scenario in self.scenarios if scenario.id == scenario_id),
            None,
        )
        if scenario is None:
            raise ValueError(f"unknown selected curriculum scenario: {scenario_id}")
        return self._build(scenario, random.Random(seed))

    def _weights(self) -> list[float]:
        weights: list[float] = []
        for scenario in self.scenarios:
            stats = self.results[scenario.id]
            failure_rate = (stats["games"] - stats["masterySum"] + 1) / (stats["games"] + 2)
            if self.adaptive:
                exposure_bonus = 0.5 / math.sqrt(float(stats["games"]) + 1.0)
                difficulty_factor = 1.0 + 0.1 * max(0, scenario.difficulty - 1)
                adaptive_weight = failure_rate * difficulty_factor + exposure_bonus
            else:
                adaptive_weight = 1.0
            weights.append(max(self.minimum_weight, adaptive_weight))
        return weights

    def sample(self, randomizer: random.Random) -> Matchup:
        weights = self._weights()
        if self.last_scenario is not None and len(self.scenarios) > 1:
            previous_index = self.scenarios.index(self.last_scenario)
            weights[previous_index] *= 0.25
        scenario = randomizer.choices(self.scenarios, weights=weights, k=1)[0]
        self.last_scenario = scenario
        return self._build(scenario, randomizer)

    def observe(
        self,
        scenario_id: str,
        won: bool,
        mastery: float | None = None,
    ) -> None:
        if scenario_id not in self.results:
            return
        self.results[scenario_id]["games"] += 1
        self.results[scenario_id]["wins"] += int(won)
        self.results[scenario_id]["masterySum"] += max(
            0.0,
            min(1.0, float(won) if mastery is None else float(mastery)),
        )

    def load_history(self, path: Path) -> None:
        """Restore adaptive success counts from completed Engine game records."""
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return
        for line in lines:
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if not isinstance(record, dict):
                continue
            scenario_id = str(record.get("scenarioId", ""))
            if scenario_id not in self.results:
                continue
            current_revision = next(
                scenario.revision for scenario in self.scenarios if scenario.id == scenario_id
            )
            if int(record.get("scenarioVersion", 1) or 1) != current_revision:
                continue
            outcome = record.get("outcome", {})
            winner = outcome.get("winner") if isinstance(outcome, dict) else None
            learner_player_id = str(record.get("learnerPlayerId", "player-1"))
            rewards = record.get("rewardsByPlayer", {})
            won = winner == learner_player_id or (
                isinstance(rewards, dict) and float(rewards.get(learner_player_id, 0) or 0) > 0
            )
            milestone_total = int(record.get("objectiveMilestonesTotal", 0) or 0)
            sideboard_target_total = int(
                record.get("sideboardTargetCardsAvailable", 0) or 0
            )
            sideboard_cut_total = int(record.get("sideboardCutCardsExpected", 0) or 0)
            sideboard_in_mastery = (
                int(record.get("sideboardTargetCardsSelected", 0) or 0)
                / sideboard_target_total
                if sideboard_target_total > 0
                else None
            )
            sideboard_out_mastery = (
                int(record.get("sideboardCutCardsSelected", 0) or 0)
                / sideboard_cut_total
                if sideboard_cut_total > 0
                else None
            )
            mastery = (
                (sideboard_in_mastery + sideboard_out_mastery) / 2
                if sideboard_in_mastery is not None and sideboard_out_mastery is not None
                else sideboard_in_mastery
                if sideboard_in_mastery is not None
                else int(record.get("objectiveMilestonesCompleted", 0) or 0)
                / milestone_total
                if milestone_total > 0
                else float(record.get("objectiveDamageProgress", 0) or 0)
                if str(record.get("objective", "")) in {"fast-win", "combo-win"}
                else float(won)
            )
            self.observe(scenario_id, won, mastery)

    def stats(self) -> dict[str, dict[str, int | float]]:
        return {
            scenario.id: {
                **self.results[scenario.id],
                "masteryRate": (
                    self.results[scenario.id]["masterySum"] / self.results[scenario.id]["games"]
                    if self.results[scenario.id]["games"]
                    else 0.0
                ),
                "weight": self._weights()[index],
            }
            for index, scenario in enumerate(self.scenarios)
        }

    def _build(self, scenario: CurriculumScenario, randomizer: random.Random) -> Matchup:
        base = self.base_sampler.sample(randomizer)
        players = deepcopy(base.setup["players"][:2])
        learner_deck_name = base.deck_names[0]
        if scenario.deck_hint:
            learner_deck_name = next(
                (name for name in self.decks if scenario.deck_hint.casefold() in name.casefold()),
                "",
            )
            if not learner_deck_name:
                raise ValueError(
                    f"curriculum scenario {scenario.id} requires a deck matching {scenario.deck_hint}"
                )
        learner_cards = self.decks[learner_deck_name]
        if scenario.opening_hand_candidate_cards:
            remaining = Counter(scenario.opening_hand_candidate_cards)
            selected_cards: list[dict[str, Any]] = []
            for card in learner_cards:
                card_name = str(card.get("name", ""))
                if bool(card.get("isSideboard")) or remaining[card_name] <= 0:
                    continue
                selected_cards.append(card)
                remaining[card_name] -= 1
            missing = [
                card_name
                for card_name, count in remaining.items()
                for _ in range(max(0, count))
            ]
            if missing:
                raise ValueError(
                    f"curriculum scenario {scenario.id} is missing candidate cards: "
                    + ", ".join(missing)
                )
            learner_cards = selected_cards
        if scenario.free_casts:
            learner_cards = _free_cast_deck(learner_cards)
        players[0]["cards"] = learner_cards
        players[0]["name"] = f"{learner_deck_name} · learner"

        fixed_definition_ids: tuple[str, ...] = ()
        if scenario.fixed_hand:
            selected_ids: list[str] = []
            for card_name in scenario.fixed_hand:
                definition = next(
                    (
                        card
                        for card in learner_cards
                        if str(card.get("name", "")) == card_name
                        and not bool(card.get("isSideboard"))
                    ),
                    None,
                )
                if definition is None:
                    raise ValueError(f"curriculum scenario {scenario.id} is missing {card_name}")
                selected_ids.append(str(definition["id"]))
            fixed_definition_ids = tuple(selected_ids)

        if scenario.legacy_match:
            learner_cards = _legacy_training_deck(learner_cards)
            players[0]["cards"] = learner_cards
            if scenario.opponent_deck_hint:
                opponent_name = next(
                    (
                        name
                        for name in self.decks
                        if scenario.opponent_deck_hint.casefold() in name.casefold()
                        and name != learner_deck_name
                    ),
                    "",
                )
                if not opponent_name:
                    raise ValueError(
                        f"curriculum scenario {scenario.id} requires an opponent deck "
                        f"matching {scenario.opponent_deck_hint}"
                    )
            else:
                opponents = [name for name in self.decks if name != learner_deck_name]
                opponent_name = randomizer.choice(opponents) if opponents else base.deck_names[1]
            players[1]["cards"] = _legacy_training_deck(self.decks[opponent_name])
            players[1]["name"] = f"{opponent_name} · predictable opponent"
            return Matchup(
                id=f"curriculum:{scenario.id}",
                setup={
                    **deepcopy(base.setup),
                    "players": players,
                    "startingPlayer": randomizer.randrange(2),
                },
                learner_player_id=str(players[0]["id"]),
                opponent_player_id=str(players[1]["id"]),
                max_turns=min(base.max_turns, 20),
                mulligan_enabled=True,
                free_mulligans=0,
                max_mulligans=None,
                game_mode="legacy",
                deck_names=(learner_deck_name, opponent_name),
                deck_session_ids=(
                    _deck_session_id(self.decks[learner_deck_name]),
                    _deck_session_id(self.decks[opponent_name]),
                ),
                scenario_id=scenario.id,
                scenario_family=scenario.family,
                scenario_version=scenario.revision,
                objective=scenario.objective,
                difficulty=scenario.difficulty,
                start_with_sideboarding=True,
                sideboard_target_card_names=scenario.sideboard_target_cards,
                sideboard_cut_card_names=scenario.sideboard_cut_cards,
            )

        opponent_cards = (
            _miracle_cards(scenario.miracle_pressure)
            if scenario.miracle_pressure
            else _passive_anchor_cards()
        )
        players[1].update(
            {
                "name": (
                    f"Miracle Anchor · level {scenario.miracle_pressure}"
                    if scenario.miracle_pressure
                    else "Passive training anchor"
                ),
                "startingLife": 20,
                "cards": opponent_cards,
            }
        )
        target_round = scenario.target_round or 8
        return Matchup(
            id=f"curriculum:{scenario.id}",
            setup={"openingHandSize": 7, "startingPlayer": 0, "players": players},
            learner_player_id=str(players[0]["id"]),
            opponent_player_id=str(players[1]["id"]),
            max_turns=max(2, target_round * 2),
            mulligan_enabled=False,
            free_mulligans=0,
            max_mulligans=0,
            game_mode="free",
            deck_names=(
                learner_deck_name,
                "Miracle Anchor" if scenario.miracle_pressure else "Passive Anchor",
            ),
            deck_session_ids=(base.deck_session_ids[0], ""),
            training_anchor_player_ids=(str(players[1]["id"]),),
            anchor_deadline_round=target_round,
            anchor_opening_hand_pool_size=scenario.opening_hand_pool_size,
            scenario_id=scenario.id,
            scenario_family=scenario.family,
            scenario_version=scenario.revision,
            objective=scenario.objective,
            difficulty=scenario.difficulty,
            target_round=target_round,
            fixed_opening_hand_definition_ids=fixed_definition_ids,
            success_card_names=scenario.success_cards,
            success_zones=scenario.success_zones,
            success_action_sequence=scenario.success_action_sequence,
            opening_hand_target_roles=scenario.opening_hand_roles,
        )
