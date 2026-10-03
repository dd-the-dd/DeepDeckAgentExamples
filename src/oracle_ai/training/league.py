from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import html
import json
import math
import os
import random
import re
import shutil
import subprocess
import threading
import time
from collections import Counter
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from functools import partial
from itertools import combinations
from pathlib import Path
from typing import Any

import httpx
import torch
import yaml

from oracle_ai.architectures import build_model, encoder_for_model, upgrade_model
from oracle_ai.checkpoints import load_checkpoint, save_checkpoint
from oracle_ai.ground_truth import (
    evaluate_ground_truth_service,
    load_ground_truth_scenarios,
)
from oracle_ai.training.behavior import summarize_decision_traces
from oracle_ai.training.core import PPOConfig, PPOLearner, SelfPlayJob
from oracle_ai.training.environments import Matchup, RustSelfPlayEnvironment
from oracle_ai.training.evaluation import (
    EvaluationRunner,
    EvaluationScenario,
    PolicyHttpClient,
    PolicyService,
    summarize_evaluation,
)
from oracle_ai.training.gameplay_metrics import (
    round_number,
    summarize_gameplay_metrics,
    update_hourly_gameplay_metrics,
)
from oracle_ai.training.matchmaking import plackett_luce_matchmaking_weight
from oracle_ai.training.plackett_luce import (
    PlackettLuceRating,
    TrainingLeaderboard,
    anchor_challenges,
    anchor_participant_id,
    hypothetical_first_place_deltas,
)
from oracle_ai.training.seeds import UniqueSeedStream


def _punching_bag_cards() -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    specifications = (
        ("artifact", "Artifact", None, None),
        ("creature", "Creature — Construct", "0", "1"),
        ("enchantment", "Enchantment", None, None),
    )
    for index in range(99):
        suffix, type_line, power, toughness = specifications[index % len(specifications)]
        cards.append(
            {
                "id": f"training-anchor-{suffix}",
                "name": f"Training Anchor {suffix.title()}",
                "typeLine": type_line,
                "manaCost": "{0}",
                "power": power,
                "toughness": toughness,
                "isCommander": False,
                "isToken": False,
                "isGamePiece": False,
                "isSideboard": False,
                "rules": [],
            }
        )
    return cards


def _anchor_matchup(
    matchup: Matchup,
    deadline_round: int,
    opening_hand_pool_size: int,
) -> Matchup:
    players = deepcopy(matchup.setup.get("players", []))
    if len(players) < 2:
        raise ValueError("anchor training requires at least two players")
    anchor_ids: list[str] = []
    for player in players[1:]:
        player["name"] = "Ancre déterministe"
        player["cards"] = _punching_bag_cards()
        anchor_ids.append(str(player["id"]))
    setup = {
        **deepcopy(matchup.setup),
        "openingHandSize": 7,
        "players": players,
    }
    return replace(
        matchup,
        setup=setup,
        max_turns=max(2, deadline_round * len(players)),
        mulligan_enabled=False,
        free_mulligans=0,
        max_mulligans=0,
        game_mode="free",
        deck_names=(matchup.deck_names[0], *("Anchor" for _ in players[1:])),
        deck_session_ids=(
            matchup.deck_session_ids[0] if matchup.deck_session_ids else "",
            *("" for _ in players[1:]),
        ),
        punching_bag_player_ids=(),
        training_anchor_player_ids=tuple(anchor_ids),
        anchor_deadline_round=deadline_round,
        anchor_opening_hand_pool_size=opening_hand_pool_size,
    )


@dataclass
class LeagueState:
    champion_version: int = 0
    champion_checkpoint: str | None = None
    champion_training_step: int = 0
    perfect_streak: int = 0
    perfect_evaluation_periods: int = 0
    promotion_count: int = 0
    completed_episodes: int = 0
    attempted_episodes: int = 0
    paused: bool = False

    @property
    def champion_name(self) -> str:
        return f"ia-gt-{self.champion_version}"


@dataclass(frozen=True)
class EvaluationBenchmarkOpponent:
    id: str
    checkpoint: Path
    port: int
    device: str
    every_periods: int


@dataclass(frozen=True)
class TrainingBatchProfile:
    """Structural game settings shared by every rollout in one batch."""

    game_format: str
    player_count: int
    starting_life: int
    free_mulligans: int
    max_mulligans: int | None


def _evaluation_benchmark_opponents(
    config: dict[str, Any],
) -> list[EvaluationBenchmarkOpponent]:
    evaluation = config.get("evaluation", {})
    opponents: list[EvaluationBenchmarkOpponent] = []
    used_ports = {
        int(evaluation.get("championPort", 8790)),
        int(evaluation.get("candidatePort", 8791)),
    }
    for item in evaluation.get("benchmarkOpponents", []):
        opponent_id = str(item.get("id", "")).strip()
        checkpoint = Path(str(item.get("checkpoint", ""))).resolve()
        port = int(item.get("port", 8794))
        every_periods = int(item.get("everyPeriods", 1))
        if not opponent_id:
            raise ValueError("evaluation benchmark opponent id cannot be empty")
        if not (checkpoint / "manifest.json").is_file():
            raise ValueError(
                f"evaluation benchmark {opponent_id} has no checkpoint manifest at {checkpoint}"
            )
        if port in used_ports:
            raise ValueError(f"evaluation benchmark port {port} is already in use")
        if every_periods <= 0:
            raise ValueError("evaluation benchmark everyPeriods must be positive")
        used_ports.add(port)
        opponents.append(
            EvaluationBenchmarkOpponent(
                id=opponent_id,
                checkpoint=checkpoint,
                port=port,
                device=str(item.get("device", evaluation.get("championDevice", "cpu"))),
                every_periods=every_periods,
            )
        )
    return opponents


def _initialize_ground_truth_checkpoint(
    checkpoint: Path,
    model: Any,
    ppo_config: PPOConfig,
    matchup_ids: list[str],
    config: dict[str, Any],
) -> None:
    mode = str(config.get("initialGroundTruthMode", "legacy-v1"))
    source = config.get("initialGroundTruthCheckpoint")
    if mode not in {"legacy-v1", "learner-step-zero"}:
        raise ValueError(f"unsupported initialGroundTruthMode: {mode}")
    if source and mode == "learner-step-zero":
        raise ValueError(
            "initialGroundTruthCheckpoint cannot be combined with "
            "initialGroundTruthMode=learner-step-zero"
        )
    if source:
        shutil.copytree(Path(source), checkpoint)
        return
    if (
        mode == "learner-step-zero"
        or getattr(
            model,
            "model_family",
            "hashing-v1",
        )
        == "hashing-v1"
    ):
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=ppo_config.learning_rate,
        )
        save_checkpoint(
            checkpoint,
            model,
            optimizer,
            0,
            matchup_ids,
        )
        return
    raise ValueError(
        "a non-V1 learner requires initialGroundTruthCheckpoint or "
        "initialGroundTruthMode=learner-step-zero"
    )


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True))
        stream.write("\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        for attempt in range(5):
            try:
                temporary.replace(path)
                return
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(0.05 * (attempt + 1))
    finally:
        temporary.unlink(missing_ok=True)


def _plackett_luce_matchmaking_weight(
    stat: dict[str, float | int | None],
    *,
    target_ordinal: float | None,
    minimum_games: int,
    random_floor: float,
    rating_scale: float,
    underplayed_strength: float,
    game_prior: float,
) -> float:
    return plackett_luce_matchmaking_weight(
        stat,
        target_ordinal=target_ordinal,
        minimum_games=minimum_games,
        random_floor=random_floor,
        rating_scale=rating_scale,
        underplayed_strength=underplayed_strength,
        game_prior=game_prior,
    )


def _prune_checkpoints(checkpoint_root: Path, keep: int) -> None:
    if keep <= 0:
        raise ValueError("maxCheckpoints must be positive")
    checkpoint_root = checkpoint_root.resolve()
    candidates: list[tuple[int, Path]] = []
    for path in checkpoint_root.iterdir():
        match = re.fullmatch(r"step-(\d+)", path.name)
        if path.is_dir() and match is not None:
            candidates.append((int(match.group(1)), path.resolve()))
    for _, stale in sorted(candidates)[:-keep]:
        if stale.parent != checkpoint_root:
            raise ValueError(f"refusing to prune checkpoint outside {checkpoint_root}")
        shutil.rmtree(stale)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _resume_counter(
    config: dict[str, Any],
    output: Path,
    key: str,
    default: int = 0,
) -> int:
    value = int(config.get(key, default))
    state_path = output / "league-state.json"
    if bool(config.get("resumeLeagueState", False)) and state_path.is_file():
        payload = json.loads(state_path.read_text(encoding="utf-8"))
        value = max(value, int(payload.get(key, value)))
    if value < 0:
        raise ValueError(f"{key} cannot be negative")
    return value


def _restore_league_state(
    output: Path,
    fallback_champion_checkpoint: Path,
) -> tuple[LeagueState, list[dict[str, Any]]]:
    state_path = output / "league-state.json"
    if not state_path.is_file():
        raise ValueError("resumeLeagueState requires an existing league-state.json")
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    champion_version = int(payload.get("champion_version", 0))
    stored_champion = payload.get("champion_checkpoint")
    champion_checkpoint = fallback_champion_checkpoint.resolve()
    if stored_champion:
        stored_path = Path(stored_champion).resolve()
        if stored_path.is_relative_to(output.resolve()) and stored_path.is_dir():
            champion_checkpoint = stored_path
    local_champions = [
        output / "champions" / f"ia-gt-{champion_version}",
        *sorted(
            (output / "champions").glob(f"ia-gt-{champion_version}-step-*"),
            reverse=True,
        ),
    ]
    local_champion = next(
        (candidate.resolve() for candidate in local_champions if candidate.is_dir()),
        None,
    )
    if local_champion is not None:
        champion_checkpoint = local_champion
    state = LeagueState(
        champion_version=champion_version,
        champion_checkpoint=str(champion_checkpoint),
        champion_training_step=int(payload.get("champion_training_step", 0)),
        perfect_streak=int(payload.get("perfect_streak", 0)),
        perfect_evaluation_periods=int(payload.get("perfect_evaluation_periods", 0)),
        promotion_count=int(payload.get("promotion_count", 0)),
        completed_episodes=int(payload.get("completed_episodes", 0)),
        attempted_episodes=int(payload.get("attempted_episodes", 0)),
    )
    training_records = _read_jsonl(output / "training.jsonl")
    error_records = _read_jsonl(output / "training-errors.jsonl")
    state.completed_episodes = max(
        [state.completed_episodes] + [int(record.get("episode", 0)) for record in training_records]
    )
    state.attempted_episodes = max(
        [state.attempted_episodes]
        + [int(record.get("attempt", 0)) for record in [*training_records, *error_records]]
    )
    return state, _read_jsonl(output / "evaluations.jsonl")


def _deck_catalog(config: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    source = config.get("deckSource")
    if source:
        if source != "database":
            raise ValueError(f"unsupported deck source: {source}")
        node = shutil.which("node")
        if node is None:
            raise ValueError("database deck loading requires Node.js")
        builder = Path(
            config.get(
                "deckSessionBuilder",
                "scripts/build-ai-deck-catalog.mjs",
            )
        )
        command = [node, str(builder), "--stdout", "--meta-only"]
        meta_legacy = config.get("metaLegacyDeckSelection", {})
        if bool(meta_legacy.get("enabled", False)):
            creators = [
                str(creator).strip()
                for creator in meta_legacy.get("creators", [])
                if str(creator).strip()
            ]
            if not creators:
                raise ValueError("metaLegacyDeckSelection requires at least one creator")
            command.extend(f"--creator={creator}" for creator in creators)
        result = subprocess.run(
            command,
            capture_output=True,
            check=False,
            text=True,
        )
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip()
            raise ValueError(f"failed to load database deck sessions: {detail}")
        catalog = json.loads(result.stdout)
        if not isinstance(catalog, dict) or not catalog:
            raise ValueError("database deck sessions must contain at least one deck")
        return catalog
    path = config.get("deckCatalog")
    if not path:
        return {}
    catalog = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(catalog, dict):
        raise ValueError("deck catalog must contain a mapping of deck names to card lists")
    return catalog


def _deck_catalog_revision(config: dict[str, Any]) -> str | None:
    """Return a stable revision of the PostgreSQL meta decks used by training."""
    if config.get("deckSource") != "database":
        return None
    api_url = str(
        config.get("deckApiUrl")
        or os.environ.get("DDL_PLATFORM_API_URL")
        or os.environ.get("VITE_DDL_API_URL")
        or "http://127.0.0.1:8790/v1"
    ).rstrip("/")
    try:
        response = httpx.get(
            f"{api_url}/local/deck-sessions",
            timeout=float(config.get("deckApiTimeoutSeconds", 30)),
        )
        response.raise_for_status()
        sessions = response.json().get("data", [])
    except (httpx.HTTPError, ValueError, AttributeError):
        return None
    meta_legacy = config.get("metaLegacyDeckSelection", {})
    selected_creators = (
        {
            str(creator).strip().casefold()
            for creator in meta_legacy.get("creators", [])
            if str(creator).strip()
        }
        if bool(meta_legacy.get("enabled", False))
        else set()
    )
    meta_decks = [
        {
            "id": session.get("id"),
            "name": session.get("name"),
            "creator": session.get("creator"),
            "updatedAt": session.get("updatedAt"),
        }
        for session in sessions
        if session.get("isMetaDeck") is True
        and (
            not selected_creators
            or str(session.get("creator", "")).strip().casefold() in selected_creators
        )
    ]
    meta_decks.sort(key=lambda session: (str(session["id"]), str(session["name"])))
    encoded = json.dumps(
        meta_decks,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _setup(item: dict[str, Any], decks: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    if "setup" in item:
        return item["setup"]
    deck_names = item.get("decks", [])
    if not deck_names:
        raise ValueError(f"matchup {item['id']} has neither setup nor decks")
    unknown = [deck_name for deck_name in deck_names if deck_name not in decks]
    if unknown:
        raise ValueError(f"matchup {item['id']} references unknown decks: {unknown}")
    return {
        "openingHandSize": int(item.get("openingHandSize", 7)),
        "startingPlayer": int(item.get("startingPlayer", 0)),
        "players": [
            {
                "id": f"player-{index + 1}",
                "name": f"{deck_name} #{index + 1}",
                "startingLife": int(item.get("startingLife", 20)),
                "cards": decks[deck_name],
            }
            for index, deck_name in enumerate(deck_names)
        ],
    }


def _deck_session_id(cards: list[dict[str, Any]]) -> str:
    return next(
        (
            str(card.get("sourceSessionId", "")).strip()
            for card in cards
            if str(card.get("sourceSessionId", "")).strip()
        ),
        "",
    )


def _matchup(
    item: dict[str, Any],
    decks: dict[str, list[dict[str, Any]]],
) -> Matchup:
    setup = _setup(item, decks)
    players = setup.get("players", [])
    if not players:
        raise ValueError(f"matchup {item['id']} has no players")
    deck_names = tuple(str(name) for name in item.get("decks", []))
    return Matchup(
        id=item["id"],
        setup=setup,
        learner_player_id=players[0]["id"],
        opponent_player_id=players[1]["id"] if len(players) > 1 else players[0]["id"],
        max_turns=int(item.get("maxTurns", 200)),
        mulligan_enabled=bool(item.get("mulliganEnabled", True)),
        free_mulligans=int(item.get("freeMulligans", 0)),
        max_mulligans=(int(item["maxMulligans"]) if item.get("maxMulligans") is not None else None),
        game_mode=str(item.get("gameMode", "free")),
        deck_names=deck_names,
        deck_session_ids=tuple(_deck_session_id(decks[name]) for name in deck_names),
    )


def _training_matchups(
    config: dict[str, Any],
    decks: dict[str, list[dict[str, Any]]],
) -> dict[str, Matchup]:
    items = [
        *config.get("trainingMatchups", []),
        *_training_matrix_items(config.get("trainingScenarioMatrix"), decks),
    ]
    matchups = [_matchup(item, decks) for item in items]
    if not matchups:
        raise ValueError("league training requires at least one training matchup")
    return {matchup.id: matchup for matchup in matchups}


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")


def _matrix_deck_names(
    matrix: dict[str, Any],
    decks: dict[str, list[dict[str, Any]]],
) -> list[str]:
    names = list(matrix.get("decks") or decks)
    unknown = [name for name in names if name not in decks]
    if unknown:
        raise ValueError(f"scenario matrix references unknown decks: {unknown}")
    return names


def _commander_deck_is_legal(cards: list[dict[str, Any]]) -> bool:
    deck_cards = [
        card
        for card in cards
        if not bool(card.get("isSideboard")) and not _is_auxiliary_game_piece(card)
    ]
    if len(deck_cards) != 100 or sum(bool(card.get("isCommander")) for card in deck_cards) != 1:
        return False
    non_basic_names = [
        str(card.get("name", "")).strip().casefold()
        for card in deck_cards
        if "basic land" not in str(card.get("typeLine", "")).casefold()
        and "basic snow land" not in str(card.get("typeLine", "")).casefold()
    ]
    return len(non_basic_names) == len(set(non_basic_names))


def _legacy_deck_is_candidate(cards: list[dict[str, Any]]) -> bool:
    main_deck = [
        card
        for card in cards
        if not bool(card.get("isSideboard")) and not _is_auxiliary_game_piece(card)
    ]
    sideboard = [
        card
        for card in cards
        if bool(card.get("isSideboard")) and not _is_auxiliary_game_piece(card)
    ]
    return (
        len(main_deck) >= 60
        and len(sideboard) <= 15
        and not any(bool(card.get("isCommander")) for card in cards)
    )


def _is_auxiliary_game_piece(card: dict[str, Any]) -> bool:
    type_line = str(card.get("typeLine", "")).strip().casefold()
    return (
        bool(card.get("isGamePiece"))
        or bool(card.get("isToken"))
        or type_line.startswith(("token ", "emblem", "dungeon"))
    )


def _retained_deck_size(game_mode: str) -> int | None:
    return {"training": 20, "training2": 40}.get(game_mode)


def _engine_api_headers() -> dict[str, str] | None:
    api_key = os.getenv("MTG_ENGINE_API_KEY", "").strip()
    return {"x-mtg-api-key": api_key} if api_key else None


class RandomTrainingMatchupSampler:
    """Samples one independent, fully randomized game setup per episode."""

    matchup_id = "training-random"

    def __init__(
        self,
        config: dict[str, Any],
        decks: dict[str, list[dict[str, Any]]],
    ) -> None:
        self.decks = decks
        self.deck_names = _matrix_deck_names(config, decks)
        self.formats = [str(value) for value in config.get("formats", ["free"])]
        if not self.formats or any(
            game_format not in {"free", "legacy", "commander", "training", "training2"}
            for game_format in self.formats
        ):
            raise ValueError(
                "training scenario randomizer formats must contain free, legacy, commander, training, or training2"
            )
        self.format_sampling = str(config.get("formatSampling", "random"))
        if self.format_sampling not in {"random", "roundRobin"}:
            raise ValueError(
                "training scenario randomizer formatSampling must be random or roundRobin"
            )
        self.format_sample_index = 0
        self.player_counts = [int(value) for value in config.get("playerCounts", [2, 3, 4])]
        if not self.player_counts or any(
            player_count < 2 or player_count > 4 for player_count in self.player_counts
        ):
            raise ValueError("training scenario randomizer player counts must be 2 to 4")
        free_config = config.get("free", {})
        self.free_life_range = self._integer_range(
            free_config.get("startingLifeRange", [20, 40]),
            "free starting life",
            minimum=1,
        )
        self.free_mulligan_range = self._integer_range(
            free_config.get("freeMulliganRange", [0, 2]),
            "free mulligans",
            minimum=0,
        )
        training_config = config.get("training", {})
        self.training_free_mulligans = int(training_config.get("freeMulligans", 3))
        self.training_max_mulligans = int(training_config.get("maxMulligans", 3))
        if self.training_free_mulligans < 0:
            raise ValueError("training free mulligans cannot be negative")
        if self.training_max_mulligans < 0:
            raise ValueError("training max mulligans cannot be negative")
        self.max_turns = int(config.get("maxTurns", 80))
        if self.max_turns <= 0:
            raise ValueError("training scenario randomizer maxTurns must be positive")
        matchmaking = config.get("matchmaking", {})
        self.matchmaking_enabled = bool(matchmaking.get("enabled", False))
        self.matchmaking_random_floor = float(matchmaking.get("randomFloor", 0.20))
        self.matchmaking_rating_scale = float(
            matchmaking.get("ratingScale", matchmaking.get("eloScale", 10.0))
        )
        self.matchmaking_underplayed_strength = float(matchmaking.get("underplayedStrength", 0.35))
        self.matchmaking_match_prior = float(matchmaking.get("matchPrior", 10.0))
        self.matchmaking_plackett_luce_beta = float(matchmaking.get("plackettLuceBeta", 25.0 / 6.0))
        self.matchmaking_plackett_luce_learning_rate = float(
            matchmaking.get("plackettLuceLearningRate", 1.0)
        )
        if not 0.0 <= self.matchmaking_random_floor <= 1.0:
            raise ValueError("matchmaking randomFloor must be between 0 and 1")
        if self.matchmaking_rating_scale <= 0.0:
            raise ValueError("matchmaking ratingScale must be positive")
        if self.matchmaking_underplayed_strength < 0.0:
            raise ValueError("matchmaking underplayedStrength cannot be negative")
        if self.matchmaking_match_prior <= 0.0:
            raise ValueError("matchmaking matchPrior must be positive")
        if self.matchmaking_plackett_luce_beta <= 0.0:
            raise ValueError("matchmaking plackettLuceBeta must be positive")
        if self.matchmaking_plackett_luce_learning_rate <= 0.0:
            raise ValueError("matchmaking plackettLuceLearningRate must be positive")
        self.commander_deck_names = [
            name for name in self.deck_names if _commander_deck_is_legal(decks[name])
        ]
        self.training2_deck_names = list(self.commander_deck_names)
        self.legacy_deck_names = [
            name for name in self.deck_names if _legacy_deck_is_candidate(decks[name])
        ]
        if {"commander", "training2"}.intersection(self.formats) and not self.commander_deck_names:
            raise ValueError(
                "Commander and Training 2 require a 100-card deck with exactly one commander"
            )
        self.rejected_commander_decks: dict[str, list[dict[str, Any]]] = {}
        self.rejected_training2_decks: dict[str, list[dict[str, Any]]] = {}
        self.rejected_legacy_decks: dict[str, list[dict[str, Any]]] = {}
        self.last_selection: dict[str, Any] | None = None

    @staticmethod
    def _integer_range(
        values: Any,
        label: str,
        *,
        minimum: int,
    ) -> tuple[int, int]:
        if not isinstance(values, (list, tuple)) or len(values) != 2:
            raise ValueError(f"{label} range must contain a minimum and maximum")
        lower, upper = (int(value) for value in values)
        if lower < minimum or upper < lower:
            raise ValueError(f"{label} range is invalid")
        return lower, upper

    def _build_matchup(
        self,
        game_format: str,
        lineup: list[str],
        starting_player: int,
        starting_life: int,
        free_mulligans: int,
        max_mulligans: int | None,
    ) -> Matchup:
        players = [
            {
                "id": f"player-{index + 1}",
                "name": f"{deck_name} #{index + 1}",
                "startingLife": starting_life,
                "cards": self.decks[deck_name],
            }
            for index, deck_name in enumerate(lineup)
        ]
        return Matchup(
            id=self.matchup_id,
            setup={
                "openingHandSize": {
                    "training": 5,
                    "training2": 6,
                }.get(game_format, 7),
                "startingPlayer": starting_player,
                "players": players,
            },
            learner_player_id=players[0]["id"],
            opponent_player_id=players[1]["id"],
            max_turns=self.max_turns,
            mulligan_enabled=True,
            free_mulligans=free_mulligans,
            max_mulligans=max_mulligans,
            game_mode=game_format,
            deck_names=tuple(lineup),
            deck_session_ids=tuple(_deck_session_id(self.decks[deck_name]) for deck_name in lineup),
        )

    def _eligible_decks(self, game_format: str) -> list[str]:
        if game_format == "legacy":
            return self.legacy_deck_names
        if game_format == "commander":
            return self.commander_deck_names
        if game_format == "training2":
            return self.training2_deck_names
        return self.deck_names

    def template(self) -> Matchup:
        game_format = self.formats[0]
        eligible_decks = self._eligible_decks(game_format)
        player_count = 2 if game_format == "legacy" else self.player_counts[0]
        lineup = [eligible_decks[index % len(eligible_decks)] for index in range(player_count)]
        if game_format == "commander":
            starting_life = 40
            free_mulligans = 1
            max_mulligans = None
        elif game_format == "legacy":
            starting_life = 20
            free_mulligans = 0
            max_mulligans = None
        elif game_format == "training":
            starting_life = 5
            free_mulligans = self.training_free_mulligans
            max_mulligans = self.training_max_mulligans
        elif game_format == "training2":
            starting_life = 10
            free_mulligans = 1
            max_mulligans = 3
        else:
            starting_life = self.free_life_range[0]
            free_mulligans = self.free_mulligan_range[0]
            max_mulligans = None
        return self._build_matchup(
            game_format,
            lineup,
            starting_player=0,
            starting_life=starting_life,
            free_mulligans=free_mulligans,
            max_mulligans=max_mulligans,
        )

    def validate_commander_decks(
        self,
        engine_url: str,
        timeout_seconds: float,
    ) -> None:
        formats = [
            game_format
            for game_format in ("legacy", "commander", "training2")
            if game_format in self.formats
        ]
        if not formats:
            return
        with httpx.Client(
            base_url=engine_url.rstrip("/"),
            timeout=timeout_seconds,
            headers=_engine_api_headers(),
        ) as client:
            for game_format in formats:
                candidates = self._eligible_decks(game_format)
                validated: list[str] = []
                rejected: dict[str, list[dict[str, Any]]] = {}
                starting_life = 40 if game_format == "commander" else 10
                if game_format == "legacy":
                    starting_life = 20
                max_mulligans = None if game_format in {"legacy", "commander"} else 3
                for deck_name in candidates:
                    setup = self._build_matchup(
                        game_format,
                        [deck_name, deck_name],
                        starting_player=0,
                        starting_life=starting_life,
                        free_mulligans=0 if game_format == "legacy" else 1,
                        max_mulligans=max_mulligans,
                    ).setup
                    response = client.post(
                        "/game/setups/validate",
                        json={"setup": setup, "gameMode": game_format},
                    )
                    response.raise_for_status()
                    payload = response.json()
                    if payload.get("valid") is True:
                        validated.append(deck_name)
                    else:
                        rejected[deck_name] = list(payload.get("violations", []))
                if game_format == "legacy":
                    self.legacy_deck_names = validated
                    self.rejected_legacy_decks = rejected
                elif game_format == "commander":
                    self.commander_deck_names = validated
                    self.rejected_commander_decks = rejected
                else:
                    self.training2_deck_names = validated
                    self.rejected_training2_decks = rejected
                if not validated:
                    raise ValueError(
                        f"{game_format} training has no deck accepted by the Rust game rules"
                    )

    def _deck_weight(
        self,
        stat: dict[str, float | int | None],
        target_ordinal: float | None,
        minimum_games: int,
    ) -> float:
        return _plackett_luce_matchmaking_weight(
            stat,
            target_ordinal=target_ordinal,
            minimum_games=minimum_games,
            random_floor=self.matchmaking_random_floor,
            rating_scale=self.matchmaking_rating_scale,
            underplayed_strength=self.matchmaking_underplayed_strength,
            game_prior=self.matchmaking_match_prior,
        )

    def sample_batch_profile(
        self,
        randomizer: random.Random,
    ) -> TrainingBatchProfile:
        """Choose game-length-sensitive settings once for a rollout batch."""
        if self.format_sampling == "roundRobin":
            game_format = self.formats[self.format_sample_index % len(self.formats)]
            self.format_sample_index += 1
        else:
            game_format = randomizer.choice(self.formats)
        player_count = 2 if game_format == "legacy" else randomizer.choice(self.player_counts)
        if game_format == "commander":
            starting_life = 40
            free_mulligans = 1
            max_mulligans = None
        elif game_format == "legacy":
            starting_life = 20
            free_mulligans = 0
            max_mulligans = None
        elif game_format == "training":
            starting_life = 5
            free_mulligans = self.training_free_mulligans
            max_mulligans = self.training_max_mulligans
        elif game_format == "training2":
            starting_life = 10
            free_mulligans = 1
            max_mulligans = 3
        else:
            starting_life = randomizer.randint(*self.free_life_range)
            free_mulligans = randomizer.randint(*self.free_mulligan_range)
            max_mulligans = None
        return TrainingBatchProfile(
            game_format=game_format,
            player_count=player_count,
            starting_life=starting_life,
            free_mulligans=free_mulligans,
            max_mulligans=max_mulligans,
        )

    def sample(
        self,
        randomizer: random.Random,
        stats_provider: Any | None = None,
        *,
        profile: TrainingBatchProfile | None = None,
        participant_ids: list[str] | tuple[str, ...] | None = None,
    ) -> Matchup:
        profile = profile or self.sample_batch_profile(randomizer)
        game_format = profile.game_format
        player_count = profile.player_count
        eligible_decks = self._eligible_decks(game_format)
        participants = list(participant_ids or ("v11" for _ in range(player_count)))
        if len(participants) != player_count:
            raise ValueError("participant_ids must match the sampled player count")
        lineup: list[str] = []
        selection_details: list[dict[str, Any]] = []
        learner_ordinal: float | None = None
        ratings_available = False
        for seat in range(player_count):
            participant_id = participants[seat]
            stats = (
                stats_provider(participant_id, eligible_decks)
                if self.matchmaking_enabled and stats_provider is not None
                else {}
            )
            normalized_stats = {
                str(name).casefold(): dict(values) for name, values in stats.items()
            }
            minimum_games = min(
                (
                    max(
                        0,
                        int(
                            normalized_stats.get(deck_name.casefold(), {}).get(
                                "games",
                                0,
                            )
                            or 0
                        ),
                    )
                    for deck_name in eligible_decks
                ),
                default=0,
            )
            weighted = [
                self._deck_weight(
                    normalized_stats.get(deck_name.casefold(), {}),
                    learner_ordinal if seat > 0 else None,
                    minimum_games,
                )
                for deck_name in eligible_decks
            ]
            deck_name = randomizer.choices(eligible_decks, weights=weighted, k=1)[0]
            deck_index = eligible_decks.index(deck_name)
            weight = weighted[deck_index]
            stat = normalized_stats.get(deck_name.casefold(), {})
            ordinal = float(stat.get("ordinal", 0.0) or 0.0)
            mu = float(stat.get("mu", 25.0) or 25.0)
            sigma = float(stat.get("sigma", 25.0 / 3.0) or (25.0 / 3.0))
            games = max(0, int(stat.get("games", 0) or 0))
            rank = int(stat["rank"]) if stat.get("rank") is not None else None
            ratings_available = ratings_available or games > 0
            if seat == 0:
                learner_ordinal = ordinal
            lineup.append(deck_name)
            selection_details.append(
                {
                    "seat": seat + 1,
                    "deck": deck_name,
                    "participantId": participant_id,
                    "mu": mu,
                    "sigma": sigma,
                    "ordinal": ordinal,
                    "rank": rank,
                    "games": games,
                    "weight": weight,
                }
            )
        previews = hypothetical_first_place_deltas(
            [
                PlackettLuceRating(
                    mu=float(detail["mu"]),
                    sigma=float(detail["sigma"]),
                    games=int(detail["games"]),
                )
                for detail in selection_details
            ],
            beta=self.matchmaking_plackett_luce_beta,
            learning_rate=self.matchmaking_plackett_luce_learning_rate,
        )
        for detail, preview in zip(selection_details, previews):
            detail["plackettLuceDelta"] = preview
        self.last_selection = {
            "enabled": self.matchmaking_enabled,
            "ratingSystem": "plackett-luce",
            "ratingSource": "training-model-deck-leaderboard",
            "ratingsAvailable": ratings_available,
            "randomFloor": self.matchmaking_random_floor,
            "ratingScale": self.matchmaking_rating_scale,
            "underplayedStrength": self.matchmaking_underplayed_strength,
            "seats": selection_details,
        }
        return self._build_matchup(
            game_format,
            lineup,
            starting_player=randomizer.randrange(player_count),
            starting_life=profile.starting_life,
            free_mulligans=profile.free_mulligans,
            max_mulligans=profile.max_mulligans,
        )


def _training_matrix_items(
    matrix: dict[str, Any] | None,
    decks: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    if not matrix:
        return []
    deck_names = _matrix_deck_names(matrix, decks)
    player_counts = [int(value) for value in matrix.get("playerCounts", [2, 3, 4])]
    free_mulligans = [int(value) for value in matrix.get("freeMulligans", [0])]
    items: list[dict[str, Any]] = []
    for player_count in player_counts:
        if player_count < 2 or player_count > 4:
            raise ValueError("training scenario matrix player counts must be 2 to 4")
        for combination_index, lineup in enumerate(combinations(deck_names, player_count)):
            for mulligan_index, free_mulligan_count in enumerate(free_mulligans):
                items.append(
                    {
                        "id": (
                            f"matrix-{player_count}p-m{free_mulligan_count}-"
                            + "-".join(_slug(name) for name in lineup)
                        ),
                        "decks": list(lineup),
                        "startingPlayer": (combination_index + mulligan_index) % player_count,
                        "freeMulligans": free_mulligan_count,
                        "maxTurns": int(matrix.get("maxTurns", 80)),
                        "startingLife": int(matrix.get("startingLife", 20)),
                    }
                )
    return items


def _evaluation_scenarios(
    config: dict[str, Any],
    decks: dict[str, list[dict[str, Any]]],
) -> dict[str, EvaluationScenario]:
    scenarios: dict[str, EvaluationScenario] = {}
    evaluation = config.get("evaluation", {})
    items = [
        *evaluation.get("scenarios", []),
        *_evaluation_matrix_items(evaluation.get("scenarioMatrix"), decks),
    ]
    for item in items:
        matchup = _matchup(item, decks)
        candidate_player_id = item["candidatePlayerId"]
        player_ids = {player["id"] for player in matchup.setup["players"]}
        if candidate_player_id not in player_ids:
            raise ValueError(
                f"evaluation scenario {matchup.id} has unknown candidate {candidate_player_id}"
            )
        scenarios[matchup.id] = EvaluationScenario(
            matchup=matchup,
            candidate_player_id=candidate_player_id,
            candidate_deck=item["candidateDeck"],
        )
    if not scenarios:
        raise ValueError("league training requires at least one evaluation scenario")
    return scenarios


def _evaluation_matrix_items(
    matrix: dict[str, Any] | None,
    decks: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    if not matrix:
        return []
    deck_names = _matrix_deck_names(matrix, decks)
    formats = [str(value) for value in matrix.get("formats", ["free"])]
    if not formats or any(
        game_format not in {"free", "legacy", "commander", "training", "training2"}
        for game_format in formats
    ):
        raise ValueError(
            "evaluation scenario matrix formats must contain free, legacy, commander, training, or training2"
        )
    player_counts = [int(value) for value in matrix.get("playerCounts", [2])]
    free_mulligans = [int(value) for value in matrix.get("freeMulligans", [1])]
    items: list[dict[str, Any]] = []
    for game_format in formats:
        eligible_decks = (
            [name for name in deck_names if _commander_deck_is_legal(decks[name])]
            if game_format in {"commander", "training2"}
            else [name for name in deck_names if _legacy_deck_is_candidate(decks[name])]
            if game_format == "legacy"
            else deck_names
        )
        if not eligible_decks:
            raise ValueError(
                f"{game_format} evaluation requires a 100-card deck with exactly one commander"
            )
        for player_count in player_counts:
            if game_format == "legacy" and player_count != 2:
                continue
            if player_count < 2 or player_count > 4:
                raise ValueError("evaluation scenario matrix player counts must be 2 to 4")
            for candidate_index, candidate_deck in enumerate(eligible_decks):
                opponents = [
                    eligible_decks[(candidate_index + offset) % len(eligible_decks)]
                    for offset in range(1, player_count)
                ]
                candidate_seat = candidate_index % player_count
                lineup = opponents.copy()
                lineup.insert(candidate_seat, candidate_deck)
                free_mulligan_count = (
                    1
                    if game_format in {"commander", "training2"}
                    else 0
                    if game_format == "legacy"
                    else int(matrix.get("trainingFreeMulligans", 3))
                    if game_format == "training"
                    else free_mulligans[(candidate_index + player_count) % len(free_mulligans)]
                )
                scenario_prefix = {
                    "free": "eval-matrix",
                    "legacy": "eval-matrix-legacy",
                    "commander": "eval-matrix-commander",
                    "training": "eval-matrix-training",
                    "training2": "eval-matrix-training2",
                }[game_format]
                items.append(
                    {
                        "id": f"{scenario_prefix}-{player_count}p-{_slug(candidate_deck)}",
                        "decks": lineup,
                        "candidatePlayerId": f"player-{candidate_seat + 1}",
                        "candidateDeck": candidate_deck,
                        "startingPlayer": (candidate_index + 1) % player_count,
                        "freeMulligans": free_mulligan_count,
                        "maxMulligans": (
                            int(matrix.get("trainingMaxMulligans", 3))
                            if game_format == "training"
                            else 3
                            if game_format == "training2"
                            else None
                        ),
                        "maxTurns": int(matrix.get("maxTurns", 80)),
                        "openingHandSize": (
                            5
                            if game_format == "training"
                            else 6
                            if game_format == "training2"
                            else 7
                        ),
                        "startingLife": (
                            40
                            if game_format == "commander"
                            else 5
                            if game_format == "training"
                            else 10
                            if game_format == "training2"
                            else int(matrix.get("startingLife", 20))
                        ),
                        "gameMode": game_format,
                    }
                )
    return items


def _evaluation_seed_map(
    scenarios: dict[str, EvaluationScenario],
    seed: int,
    games_per_scenario: int,
) -> dict[str, list[int]]:
    stream = UniqueSeedStream(seed)
    return {scenario_id: stream.take(games_per_scenario) for scenario_id in sorted(scenarios)}


def _write_learning_curve(evaluations: list[dict[str, Any]], output: Path) -> None:
    evaluations = [
        evaluation
        for evaluation in evaluations
        if "meanRoundsToCandidateWin" in evaluation.get("summary", {})
    ]
    if not evaluations:
        return
    csv_path = output / "learning-curve.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "period",
                "training_step",
                "champion_version",
                "candidate_win_rate",
                "mean_rounds_to_win",
                "perfect_streak",
                "promotion_count",
            ],
        )
        writer.writeheader()
        for evaluation in evaluations:
            writer.writerow(
                {
                    "period": evaluation["period"],
                    "training_step": evaluation["candidateTrainingStep"],
                    "champion_version": evaluation["opponentVersion"],
                    "candidate_win_rate": evaluation["summary"]["candidateWinRate"],
                    "mean_rounds_to_win": evaluation["summary"]["meanRoundsToCandidateWin"],
                    "perfect_streak": evaluation["perfectStreakAfter"],
                    "promotion_count": evaluation["promotionCountAfter"],
                }
            )

    width = 1000
    height = 620
    left = 80
    right = 40
    top = 55
    panel_height = 210
    panel_gap = 90
    plot_width = width - left - right
    count = len(evaluations)

    def x(index: int) -> float:
        return left + (plot_width * index / max(1, count - 1))

    win_points = [
        f"{x(index):.1f},{top + panel_height * (1.0 - row['summary']['candidateWinRate']):.1f}"
        for index, row in enumerate(evaluations)
    ]
    round_values = [
        row["summary"]["meanRoundsToCandidateWin"]
        for row in evaluations
        if row["summary"]["meanRoundsToCandidateWin"] is not None
    ]
    max_rounds = max(round_values, default=1.0)
    round_top = top + panel_height + panel_gap
    round_points = [
        (
            f"{x(index):.1f},"
            f"{round_top + panel_height * (row['summary']['meanRoundsToCandidateWin'] / max_rounds):.1f}"
        )
        for index, row in enumerate(evaluations)
        if row["summary"]["meanRoundsToCandidateWin"] is not None
    ]
    labels = []
    for index, row in enumerate(evaluations):
        labels.append(
            f'<text x="{x(index):.1f}" y="{height - 22}" text-anchor="middle" '
            f'font-size="11">{row["period"]}</text>'
        )
        labels.append(
            f'<text x="{x(index):.1f}" y="{top - 12}" text-anchor="middle" '
            f'font-size="10">{html.escape(row["opponentVersion"])}</text>'
        )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">
<rect width="100%" height="100%" fill="#111827"/>
<text x="{left}" y="28" fill="#f9fafb" font-size="20">Oracle AI learning evaluation</text>
<text x="{left}" y="{top - 30}" fill="#9ca3af" font-size="12">Opponent version is shown above each period</text>
<g stroke="#374151" fill="none">
  <rect x="{left}" y="{top}" width="{plot_width}" height="{panel_height}"/>
  <rect x="{left}" y="{round_top}" width="{plot_width}" height="{panel_height}"/>
</g>
<text x="20" y="{top + 16}" fill="#60a5fa" font-size="13">100%</text>
<text x="28" y="{top + panel_height}" fill="#60a5fa" font-size="13">0%</text>
<text x="18" y="{round_top + 16}" fill="#f59e0b" font-size="13">0 rounds</text>
<text x="8" y="{round_top + panel_height}" fill="#f59e0b" font-size="13">{max_rounds:.1f} rounds</text>
<polyline points="{" ".join(win_points)}" fill="none" stroke="#60a5fa" stroke-width="3"/>
<polyline points="{" ".join(round_points)}" fill="none" stroke="#f59e0b" stroke-width="3"/>
<g fill="#60a5fa">{"".join(f'<circle cx="{point.split(",")[0]}" cy="{point.split(",")[1]}" r="4"/>' for point in win_points)}</g>
<text x="{left}" y="{top + panel_height + 28}" fill="#60a5fa" font-size="14">Candidate win rate</text>
<text x="{left}" y="{round_top + panel_height + 28}" fill="#f59e0b" font-size="14">Mean rounds to candidate win</text>
<g fill="#d1d5db">{"".join(labels)}</g>
<text x="{width / 2}" y="{height - 4}" fill="#9ca3af" text-anchor="middle" font-size="12">Evaluation period (fixed seeds)</text>
</svg>"""
    (output / "learning-curve.svg").write_text(svg, encoding="utf-8")


def _apply_promotion_result(
    state: LeagueState,
    perfect: bool,
    required_streak: int,
    candidate_checkpoint: Path,
    output: Path,
    training_step: int,
) -> tuple[str, dict[str, Any] | None]:
    opponent_version = state.champion_name
    if perfect:
        state.perfect_streak += 1
        state.perfect_evaluation_periods += 1
    else:
        state.perfect_streak = 0
    if state.perfect_streak < required_streak:
        return opponent_version, None

    next_version = state.champion_version + 1
    champion_checkpoint = output / "champions" / f"ia-gt-{next_version}-step-{training_step}"
    shutil.copytree(candidate_checkpoint, champion_checkpoint)
    state.champion_version = next_version
    state.champion_checkpoint = str(champion_checkpoint.resolve())
    state.champion_training_step = training_step
    state.promotion_count += 1
    state.perfect_streak = 0
    return opponent_version, {
        "from": opponent_version,
        "to": state.champion_name,
        "checkpoint": state.champion_checkpoint,
        "trainingStep": training_step,
    }


def _training_episode_limit(config: dict[str, Any]) -> int | None:
    if bool(config.get("continuous", False)):
        return None
    episodes = int(config.get("episodes", 1000))
    if episodes <= 0:
        raise ValueError("episodes must be positive unless continuous is enabled")
    return episodes


def _set_additional_episode_limit(
    config: dict[str, Any], completed_episodes: int, additional_episodes: int
) -> None:
    if additional_episodes <= 0:
        raise ValueError("additional episodes must be positive")
    config["continuous"] = False
    config["episodes"] = completed_episodes + additional_episodes


def _rollout_batch_size(
    completed_episodes: int,
    configured_size: int,
    *,
    evaluation_cadences: tuple[int, ...],
    episode_limit: int | None,
) -> int:
    batch_size = configured_size
    for cadence in evaluation_cadences:
        if cadence <= 0:
            raise ValueError("evaluation cadences must be positive")
        batch_size = min(
            batch_size,
            cadence - (completed_episodes % cadence),
        )
    if episode_limit is not None:
        batch_size = min(batch_size, episode_limit - completed_episodes)
    if batch_size <= 0:
        raise ValueError("rollout batch has no remaining games")
    return batch_size


def _training_control_state(path: Path) -> str:
    if not path.is_file():
        return "running"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "running"
    desired_state = str(payload.get("desiredState", "running"))
    return desired_state if desired_state in {"paused", "running"} else "running"


class LeagueTrainer:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.output = Path(config.get("outputDir", "runs/oracle-ai-league"))
        self.output.mkdir(parents=True, exist_ok=True)
        self.saved_game_limit = max(0, min(500, int(config.get("savedGameLimit", 20))))
        self.model_evaluation_enabled = bool(config.get("modelEvaluationEnabled", True))
        self.ground_truth_evaluation_enabled = bool(
            config.get("groundTruthEvaluationEnabled", True)
        )
        opponent_mix = config.get(
            "trainingOpponentMix",
            {"self": 1.0},
        )
        if not isinstance(opponent_mix, dict):
            raise ValueError("trainingOpponentMix must be a mapping")
        self.training_opponent_mix = {
            str(mode): float(weight) for mode, weight in opponent_mix.items() if float(weight) > 0.0
        }
        if not self.training_opponent_mix or any(
            mode not in {"self", "v10", "anchor"} for mode in self.training_opponent_mix
        ):
            raise ValueError("trainingOpponentMix supports self, v10, and anchor")
        self.v10_opponent_client = (
            PolicyHttpClient(
                str(config.get("v10OpponentUrl", "http://127.0.0.1:8795")),
                float(config.get("engineTimeoutSeconds", 120)),
                str(config.get("v10OpponentControllerId", "ia-v10-in-training")),
            )
            if "v10" in self.training_opponent_mix
            else None
        )
        self.anchor_deadline_rounds = tuple(
            int(value) for value in config.get("anchorDeadlineRounds", range(1, 26))
        )
        if not self.anchor_deadline_rounds or any(
            value <= 0 for value in self.anchor_deadline_rounds
        ):
            raise ValueError("anchorDeadlineRounds must contain positive rounds")
        self.anchor_opening_hand_pool_sizes = tuple(
            int(value) for value in config.get("anchorOpeningHandPoolSizes", [20, 40, 60, 80, 100])
        )
        if not self.anchor_opening_hand_pool_sizes or any(
            value < 7 for value in self.anchor_opening_hand_pool_sizes
        ):
            raise ValueError("anchorOpeningHandPoolSizes must contain values of at least seven")
        anchor_player_counts = config.get("anchorPlayerCounts")
        if anchor_player_counts is None:
            randomizer_config = config.get("trainingScenarioRandomizer") or {}
            anchor_player_counts = randomizer_config.get("playerCounts", [2, 3, 4])
        self.anchor_player_counts = tuple(int(value) for value in anchor_player_counts)
        if not self.anchor_player_counts or any(
            value < 2 or value > 4 for value in self.anchor_player_counts
        ):
            raise ValueError("anchorPlayerCounts must contain values from two to four")
        configured_anchor_challenges = anchor_challenges(
            self.anchor_deadline_rounds,
            self.anchor_opening_hand_pool_sizes,
            self.anchor_player_counts,
        )
        default_leaderboard_labels = (
            {
                "v10": "V10 · AlphaZero + VQ-VAE",
                "v11": "V11 · AlphaStar",
            }
            if bool(config.get("includeDefaultTrainingLeaderboardParticipants", True))
            else {}
        )
        leaderboard_labels = {
            **default_leaderboard_labels,
            **{
                challenge.participant_id: challenge.label
                for challenge in configured_anchor_challenges
            },
            **{
                str(key): str(value)
                for key, value in config.get("trainingLeaderboardLabels", {}).items()
            },
        }
        self.training_leaderboard = TrainingLeaderboard(
            self.output / "training-leaderboard.json",
            leaderboard_labels,
        )
        self.training_participant_id = str(config.get("trainingParticipantId", "v11")).strip()
        if not self.training_participant_id:
            raise ValueError("trainingParticipantId cannot be empty")
        self.control_path = self.output / "training-control.json"
        previous_training_records = _read_jsonl(self.output / "training.jsonl")
        previous_error_records = _read_jsonl(self.output / "training-errors.jsonl")
        self.training_elapsed_seconds = sum(
            max(0.0, float(record.get("episodeSeconds", 0.0)))
            for record in previous_training_records
        ) + sum(
            max(0.0, float(record.get("secondsBeforeError", 0.0)))
            for record in previous_error_records
        )
        # Keep simulation and optimizer time separate. Older runs are restored
        # through their original field names so resumed counters remain valid.
        self.game_simulation_seconds = sum(
            max(
                0.0,
                float(
                    record.get(
                        "gameDurationSeconds",
                        record.get("collectionSeconds", 0.0),
                    )
                ),
            )
            for record in previous_training_records
        )
        self.model_training_seconds = sum(
            max(
                0.0,
                float(
                    record.get(
                        "trainingDurationSeconds",
                        record.get("ppoSeconds", 0.0),
                    )
                ),
            )
            for record in previous_training_records
        )
        self.hourly_gameplay_path = self.output / "training-gameplay-hourly.json"
        try:
            self.hourly_gameplay = json.loads(self.hourly_gameplay_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self.hourly_gameplay = {}
        decks = _deck_catalog(config)
        self.training_deck_names = tuple(decks)
        self.deck_catalog_revision = _deck_catalog_revision(config)
        randomizer_config = config.get("trainingScenarioRandomizer")
        if randomizer_config:
            randomizer_config = deepcopy(randomizer_config)
            matchmaking_config = randomizer_config.setdefault("matchmaking", {})
            matchmaking_config.setdefault("plackettLuceBeta", self.training_leaderboard.beta)
            matchmaking_config.setdefault(
                "plackettLuceLearningRate",
                self.training_leaderboard.learning_rate,
            )
        self.training_randomizer_config = deepcopy(randomizer_config)
        self.training_matchup_sampler = (
            RandomTrainingMatchupSampler(randomizer_config, decks) if randomizer_config else None
        )
        if self.training_matchup_sampler is not None:
            self.training_matchup_sampler.validate_commander_decks(
                str(config.get("engineUrl", "http://127.0.0.1:8787")),
                float(config.get("engineTimeoutSeconds", 120)),
            )
        if self.training_matchup_sampler is not None:
            template = self.training_matchup_sampler.template()
            self.training_matchups = {template.id: template}
        else:
            self.training_matchups = _training_matchups(config, decks)
        evaluation_config = config.get("evaluation", {})
        self.evaluation_scenarios = (
            _evaluation_scenarios(config, decks) if self.model_evaluation_enabled else {}
        )
        self.evaluation_seed_map = (
            _evaluation_seed_map(
                self.evaluation_scenarios,
                int(evaluation_config.get("seed", 0xE1A1)),
                int(evaluation_config.get("gamesPerScenario", 1)),
            )
            if self.model_evaluation_enabled
            else {}
        )
        self.evaluation_benchmark_opponents = (
            _evaluation_benchmark_opponents(config) if self.model_evaluation_enabled else []
        )
        evaluation_seeds = {seed for seeds in self.evaluation_seed_map.values() for seed in seeds}
        self.training_seed_stream = UniqueSeedStream(
            int(config.get("trainingSeed", config.get("seed", 20260729))),
            excluded=evaluation_seeds,
        )
        training_seed_skip = _resume_counter(
            config,
            self.output,
            "trainingSeedSkip",
        )
        self.training_seed_stream.take(training_seed_skip)
        self.training_seed_skip = training_seed_skip
        self.matchup_randomizer = random.Random(int(config.get("seed", 20260729)))
        matchup_randomizer_skip = _resume_counter(
            config,
            self.output,
            "matchupRandomizerSkip",
            training_seed_skip,
        )
        for _ in range(matchup_randomizer_skip):
            if self.training_matchup_sampler is not None:
                self.training_matchup_sampler.sample(self.matchup_randomizer)
            else:
                self.matchup_randomizer.choice(list(self.training_matchups))
        self.matchup_randomizer_skip = matchup_randomizer_skip
        self.evaluations: list[dict[str, Any]] = []
        self._matchmaking_warning: str | None = None

        torch_seed = int(config.get("seed", 20260729))
        torch.manual_seed(torch_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(torch_seed)
        self.device = torch.device(
            config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        )
        self.gpu_memory_limit_mb = int(config.get("gpuMemoryLimitMb", 0))
        self.resource_plan_path = (
            Path(str(config["learnerResourcePlan"])) if config.get("learnerResourcePlan") else None
        )
        self._apply_gpu_memory_limit(self.gpu_memory_limit_mb)
        ppo_config = PPOConfig(**config.get("ppo", {}))
        initial_model = build_model(config.get("model", {}))
        initial_ground_truth = self.output / "champions" / "ia-gt-0"
        if not initial_ground_truth.exists():
            _initialize_ground_truth_checkpoint(
                initial_ground_truth,
                initial_model,
                ppo_config,
                list(self.training_matchups),
                config,
            )
        self.state = LeagueState(
            champion_checkpoint=str(initial_ground_truth.resolve()),
        )
        if (
            bool(config.get("resumeLeagueState", False))
            and (self.output / "league-state.json").is_file()
        ):
            self.state, self.evaluations = _restore_league_state(
                self.output,
                initial_ground_truth,
            )
        resume_checkpoint = config.get("resumeCheckpoint")
        if resume_checkpoint and bool(config.get("resumeCheckpointOptional", False)):
            checkpoint_path = Path(resume_checkpoint)
            if not (checkpoint_path / "manifest.json").is_file():
                resume_checkpoint = None
        resume_optimizer = bool(config.get("resumeOptimizer", True))
        if resume_checkpoint:
            model, payload = load_checkpoint(Path(resume_checkpoint), self.device)
            if (
                model.model_family != initial_model.model_family
                or model.export_config() != initial_model.export_config()
            ):
                if not bool(config.get("upgradeResumeArchitecture", False)):
                    raise ValueError(
                        "resume checkpoint architecture or dimensions differ from "
                        "the requested model; set "
                        "upgradeResumeArchitecture: true for a supported migration"
                    )
                model = upgrade_model(model, initial_model)
                resume_optimizer = False
        else:
            model = initial_model
            payload = None
        encoder = encoder_for_model(
            model,
            max_state_tokens=int(config.get("maxStateTokens", 512)),
        )
        self.learner = PPOLearner(
            model,
            encoder,
            ppo_config,
            self.device,
        )
        if payload is not None:
            if resume_optimizer and "optimizer" in payload:
                self.learner.optimizer.load_state_dict(payload["optimizer"])
            self.learner.training_step = int(payload["training_step"])
        self.checkpoint_recovery: dict[str, Any] | None = None
        recorded_steps = [
            int(record.get("trainingStep", 0))
            for record in _read_jsonl(self.output / "training.jsonl")
        ]
        highest_recorded_step = max(recorded_steps, default=0)
        if highest_recorded_step > self.learner.training_step:
            self.checkpoint_recovery = {
                "checkpointTrainingStep": self.learner.training_step,
                "highestRecordedTrainingStep": highest_recorded_step,
                "lostStepDelta": highest_recorded_step - self.learner.training_step,
                "detectedAtUnixMs": int(time.time() * 1000),
            }
            recoveries_path = self.output / "checkpoint-recoveries.jsonl"
            previous_recoveries = _read_jsonl(recoveries_path)
            recovery_signature = {
                key: self.checkpoint_recovery[key]
                for key in (
                    "checkpointTrainingStep",
                    "highestRecordedTrainingStep",
                    "lostStepDelta",
                )
            }
            previous_signature = (
                {key: previous_recoveries[-1].get(key) for key in recovery_signature}
                if previous_recoveries
                else None
            )
            if previous_signature != recovery_signature:
                _append_jsonl(recoveries_path, self.checkpoint_recovery)
        self.parallel_game_workers = int(config.get("parallelGameWorkers", 1))
        self.rollout_batch_games = int(config.get("rolloutBatchGames", self.parallel_game_workers))
        if self.parallel_game_workers <= 0:
            raise ValueError("parallelGameWorkers must be positive")
        if self.rollout_batch_games <= 0:
            raise ValueError("rolloutBatchGames must be positive")
        if self.rollout_batch_games > self.parallel_game_workers:
            raise ValueError(
                "rolloutBatchGames cannot exceed parallelGameWorkers until workers "
                "can safely reuse environments within one rollout batch"
            )
        self.environments = [
            self._new_training_environment() for _ in range(self.parallel_game_workers)
        ]
        # Kept as a compatibility alias for diagnostics and existing callers.
        self.environment = self.environments[0]
        self.candidate_model_name = str(config.get("candidateModelName", "ia-in-training"))
        if not self.candidate_model_name:
            raise ValueError("candidateModelName cannot be empty")
        self.live_checkpoint = self.output / "live" / self.candidate_model_name
        self.model_registry = self.output / "model-registry.json"
        self.ground_truth_service: PolicyService | None = None
        self.training_service: PolicyService | None = None
        self.ground_truth_health: dict[str, Any] | None = None
        self.training_health: dict[str, Any] | None = None
        ground_truth_config = config.get("groundTruthEvaluation", {})
        self.ground_truth_scenarios = (
            load_ground_truth_scenarios(
                ground_truth_config.get("path"),
                minimum_confidence=int(ground_truth_config.get("minimumConfidence", 1)),
                deterministic_paths=ground_truth_config.get("deterministicScenarioPaths"),
                deterministic_scenario_ids=ground_truth_config.get("deterministicScenarioIds"),
                deterministic_decision_type=str(
                    ground_truth_config.get(
                        "deterministicDecisionType",
                        "fastDeterministicWin",
                    )
                ),
            )
            if self.ground_truth_evaluation_enabled
            else []
        )
        self._state_lock = threading.RLock()
        self.active_attempts: dict[int, dict[str, Any]] = {}
        self._active_attempt_trackers: dict[int, dict[str, Any]] = {}
        self.training_phase = "initializing"
        self.last_attempt: dict[str, Any] | None = None
        for worker_index, environment in enumerate(self.environments):
            environment.progress_callback = partial(
                self._observe_training_view,
                worker_index,
            )
        self._write_model_registry()
        resolved_config = dict(config)
        if self.training_matchup_sampler is not None:
            resolved_config["resolvedTrainingDecks"] = list(self.training_deck_names)
            resolved_config["resolvedCommanderDecks"] = list(
                self.training_matchup_sampler.commander_deck_names
            )
            resolved_config["resolvedLegacyDecks"] = list(
                self.training_matchup_sampler.legacy_deck_names
            )
            resolved_config["rejectedLegacyDecks"] = dict(
                self.training_matchup_sampler.rejected_legacy_decks
            )
            resolved_config["rejectedCommanderDecks"] = (
                self.training_matchup_sampler.rejected_commander_decks
            )
            resolved_config["resolvedTraining2Decks"] = list(
                self.training_matchup_sampler.training2_deck_names
            )
            resolved_config["rejectedTraining2Decks"] = (
                self.training_matchup_sampler.rejected_training2_decks
            )
        _write_json(self.output / "resolved-config.json", resolved_config)
        _write_json(
            self.output / "evaluation-seeds.json",
            {
                "fixedAcrossPeriods": True,
                "trainingSeedsExcluded": True,
                "trainingSeedSkip": training_seed_skip,
                "matchupRandomizerSkip": matchup_randomizer_skip,
                "seedsByScenario": self.evaluation_seed_map,
            },
        )

    def close(self) -> None:
        if self.v10_opponent_client is not None:
            self.v10_opponent_client.close()
            self.v10_opponent_client = None
        if self.training_service is not None:
            self.training_service.stop()
            self.training_service = None
        if self.ground_truth_service is not None:
            self.ground_truth_service.stop()
            self.ground_truth_service = None
        for environment in self.environments:
            environment.close()

    def _write_model_registry(self) -> None:
        models = []
        champions = self.output / "champions"
        for checkpoint in champions.iterdir():
            match = re.fullmatch(r"ia-gt-(\d+)(?:-step-\d+)?", checkpoint.name)
            manifest_path = checkpoint / "manifest.json"
            if match is None or not manifest_path.is_file():
                continue
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            models.append(
                {
                    "id": f"ia-gt-{int(match.group(1))}",
                    "checkpoint": str(checkpoint.resolve()),
                    "trainingStep": int(manifest.get("training_step", 0)),
                }
            )
        models.sort(key=lambda model: int(model["id"].removeprefix("ia-gt-")))
        _write_json(
            self.model_registry,
            {
                "schemaVersion": "oracle-ai-model-registry/v1",
                "currentGroundTruth": self.state.champion_name,
                "models": models,
            },
        )

    def _policy_service(
        self,
        *,
        port: int,
        model_name: str,
        checkpoint: Path | None = None,
        registry: Path | None = None,
        device: str | None = None,
    ) -> PolicyService:
        evaluation_config = self.config.get("evaluation", {})
        return PolicyService(
            port=port,
            model_name=model_name,
            log_dir=self.output / "service-logs",
            device=device or str(evaluation_config.get("device", "cpu")),
            checkpoint=checkpoint,
            registry=registry,
        )

    def _restart_ground_truth_service(self) -> None:
        if self.ground_truth_service is not None:
            self.ground_truth_service.stop()
        evaluation_config = self.config.get("evaluation", {})
        assert self.state.champion_checkpoint is not None
        self.ground_truth_service = self._policy_service(
            port=int(evaluation_config.get("championPort", 8790)),
            model_name=self.state.champion_name,
            registry=self.model_registry,
            device=str(
                evaluation_config.get("championDevice", evaluation_config.get("device", "cpu"))
            ),
        )
        self.ground_truth_health = self.ground_truth_service.start()

    def _save_live_checkpoint(self) -> None:
        save_checkpoint(
            self.live_checkpoint,
            self.learner.model,
            self.learner.optimizer,
            self.learner.training_step,
            list(self.training_matchups),
            include_optimizer=False,
        )

    def _refresh_training_service(self, *, checkpoint_is_current: bool = False) -> None:
        if not checkpoint_is_current:
            self._save_live_checkpoint()
        evaluation_config = self.config.get("evaluation", {})
        if self.training_service is None:
            self.training_service = self._policy_service(
                port=int(evaluation_config.get("candidatePort", 8791)),
                model_name=self.candidate_model_name,
                checkpoint=self.live_checkpoint,
                device=str(
                    evaluation_config.get("candidateDevice", evaluation_config.get("device", "cpu"))
                ),
            )
            self.training_health = self.training_service.start()
        else:
            self.training_health = self.training_service.reload()

    def _start_policy_services(self) -> None:
        if self.ground_truth_evaluation_enabled and bool(
            self.config.get("persistentGroundTruthService", True)
        ):
            self._restart_ground_truth_service()
        self._refresh_training_service()

    def _save_candidate(self) -> Path:
        checkpoint_root = self.output / "checkpoints"
        checkpoint = checkpoint_root / f"step-{self.learner.training_step}"
        save_checkpoint(
            checkpoint,
            self.learner.model,
            self.learner.optimizer,
            self.learner.training_step,
            list(self.training_matchups),
        )
        _prune_checkpoints(
            checkpoint_root,
            int(self.config.get("maxCheckpoints", 3)),
        )
        return checkpoint

    def _run_evaluation(self, period: int, checkpoint: Path) -> dict[str, Any]:
        evaluation_config = self.config.get("evaluation", {})
        candidate_name = self.candidate_model_name
        temporary_ground_truth_service = self.ground_truth_service is None
        if temporary_ground_truth_service:
            self._restart_ground_truth_service()
        if self.training_service is None:
            raise RuntimeError("policy services must be running before evaluation")
        runner = EvaluationRunner(
            self.config.get("engineUrl", "http://127.0.0.1:8787"),
            self.evaluation_scenarios,
            float(self.config.get("engineTimeoutSeconds", 120)),
            str(self.config.get("analyticsPilotId", "ia-in-training")),
            (
                str(self.config["initialGroundTruthAnalyticsPilotId"])
                if self.config.get("initialGroundTruthAnalyticsPilotId")
                else None
            ),
        )
        games: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []
        benchmarks: list[dict[str, Any]] = []
        started = time.perf_counter()
        games_per_opponent = sum(len(seeds) for seeds in self.evaluation_seed_map.values())
        evaluated_opponent_count = 1 + sum(
            period % benchmark.every_periods == 0
            for benchmark in self.evaluation_benchmark_opponents
        )
        evaluation_game_count = games_per_opponent * evaluated_opponent_count
        evaluation_game_index = 0
        champion_client = PolicyHttpClient(
            self.ground_truth_service.url,
            controller_id=self.state.champion_name,
        )
        candidate_client = PolicyHttpClient(
            self.training_service.url,
            controller_id=self.candidate_model_name,
        )

        def run_against(
            opponent_client: PolicyHttpClient,
            opponent_version: str,
        ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
            nonlocal evaluation_game_index
            opponent_games: list[dict[str, Any]] = []
            opponent_errors: list[dict[str, Any]] = []
            for scenario_id, seeds in self.evaluation_seed_map.items():
                for seed in seeds:
                    evaluation_game_index += 1
                    scenario = self.evaluation_scenarios[scenario_id]
                    player_ids = [
                        str(player["id"]) for player in scenario.matchup.setup.get("players", [])
                    ]
                    participants_by_player = {
                        player_id: (
                            self.training_participant_id
                            if player_id == scenario.candidate_player_id
                            else opponent_version
                        )
                        for player_id in player_ids
                    }
                    with self._state_lock:
                        self.active_attempts = {
                            0: {
                                "attempt": (self.state.attempted_episodes + evaluation_game_index),
                                "worker": evaluation_game_index,
                                "batchSize": evaluation_game_count,
                                "status": "evaluatingModel",
                                "evaluation": True,
                                "evaluationPeriod": period,
                                "scenarioId": scenario_id,
                                "seed": seed,
                                "gameMode": scenario.matchup.game_mode,
                                "players": len(player_ids),
                                "decks": list(scenario.matchup.deck_names),
                                "participantsByPlayer": participants_by_player,
                                "candidateDeck": scenario.candidate_deck,
                                "opponentVersion": opponent_version,
                                "decisions": 0,
                                "loopEventCount": 0,
                                "recentLoopEvents": [],
                                "startedAtUnixMs": int(time.time() * 1000),
                            }
                        }
                        self._active_attempt_trackers = {}
                    runner.environment.progress_callback = partial(
                        self._observe_training_view,
                        0,
                    )
                    self._write_state()
                    try:
                        opponent_games.append(
                            runner.run_game(
                                scenario_id,
                                seed,
                                candidate_client,
                                opponent_client,
                                opponent_version,
                            )
                        )
                    except Exception as error:
                        opponent_errors.append(
                            {
                                "scenarioId": scenario_id,
                                "seed": seed,
                                "opponentVersion": opponent_version,
                                "error": f"{type(error).__name__}: {error}",
                            }
                        )
            return opponent_games, opponent_errors

        try:
            champion_health = champion_client.health()
            candidate_health = candidate_client.health()
            champion_trace_start = len(runner.candidate_traces)
            games, errors = run_against(champion_client, self.state.champion_name)
            summary = summarize_evaluation(games, errors)
            summary["candidateBehavior"] = summarize_decision_traces(
                runner.candidate_traces[champion_trace_start:]
            )

            for benchmark in self.evaluation_benchmark_opponents:
                if period % benchmark.every_periods != 0:
                    continue
                service = self._policy_service(
                    port=benchmark.port,
                    model_name=benchmark.id,
                    checkpoint=benchmark.checkpoint,
                    device=benchmark.device,
                )
                benchmark_client: PolicyHttpClient | None = None
                benchmark_games: list[dict[str, Any]] = []
                benchmark_errors: list[dict[str, Any]] = []
                benchmark_health: dict[str, Any] | None = None
                benchmark_trace_start = len(runner.candidate_traces)
                try:
                    benchmark_health = service.start()
                    benchmark_client = PolicyHttpClient(
                        service.url,
                        controller_id=benchmark.id,
                    )
                    benchmark_games, benchmark_errors = run_against(
                        benchmark_client,
                        benchmark.id,
                    )
                except Exception as error:
                    benchmark_errors.append(
                        {
                            "opponentVersion": benchmark.id,
                            "error": f"{type(error).__name__}: {error}",
                        }
                    )
                finally:
                    if benchmark_client is not None:
                        benchmark_client.close()
                    service.stop()
                benchmark_summary = summarize_evaluation(
                    benchmark_games,
                    benchmark_errors,
                )
                benchmark_summary["candidateBehavior"] = summarize_decision_traces(
                    runner.candidate_traces[benchmark_trace_start:]
                )
                benchmarks.append(
                    {
                        "opponentVersion": benchmark.id,
                        "checkpoint": str(benchmark.checkpoint),
                        "everyPeriods": benchmark.every_periods,
                        "health": benchmark_health,
                        "summary": benchmark_summary,
                        "games": benchmark_games,
                        "errors": benchmark_errors,
                    }
                )
        finally:
            runner.environment.progress_callback = None
            with self._state_lock:
                self.active_attempts = {}
                self._active_attempt_trackers = {}
            candidate_client.close()
            champion_client.close()
            runner.close()

        streak_before = self.state.perfect_streak
        required_streak = int(evaluation_config.get("perfectPeriodsForPromotion", 3))
        opponent_version, promotion = _apply_promotion_result(
            self.state,
            summary["perfect"],
            required_streak,
            checkpoint,
            self.output,
            self.learner.training_step,
        )
        if promotion is not None:
            _append_jsonl(self.output / "promotions.jsonl", promotion)
            self._write_model_registry()

        evaluation = {
            "period": period,
            "candidate": candidate_name,
            "candidateTrainingStep": self.learner.training_step,
            "opponentVersion": opponent_version,
            "fixedSeeds": self.evaluation_seed_map,
            "championHealth": champion_health,
            "candidateHealth": candidate_health,
            "summary": summary,
            "games": games,
            "errors": errors,
            "benchmarks": benchmarks,
            "perfectStreakBefore": streak_before,
            "perfectStreakAfter": self.state.perfect_streak,
            "perfectEvaluationPeriodsAfter": self.state.perfect_evaluation_periods,
            "promotionCountAfter": self.state.promotion_count,
            "promotion": promotion,
            "evaluationSeconds": time.perf_counter() - started,
        }
        self.evaluations.append(evaluation)
        _append_jsonl(self.output / "evaluations.jsonl", evaluation)
        _write_learning_curve(self.evaluations, self.output)
        self._write_state()
        if (
            temporary_ground_truth_service
            and not bool(self.config.get("persistentGroundTruthService", True))
            and self.ground_truth_service is not None
        ):
            self.ground_truth_service.stop()
            self.ground_truth_service = None
            self.ground_truth_health = None
        return evaluation

    def _new_training_environment(self) -> RustSelfPlayEnvironment:
        environment = RustSelfPlayEnvironment(
            self.config.get("engineUrl", "http://127.0.0.1:8787"),
            self.training_matchups,
            float(self.config.get("engineTimeoutSeconds", 120)),
            str(self.config.get("multiplayerRewardMode", "winnerLoser")),
            str(self.config.get("analyticsPilotId", "ia-in-training")),
            float(self.config.get("noWinnerReward", 0.0)),
            float(self.config.get("legacyGameWinReward", 0.25)),
            float(self.config.get("legacyMatchWinReward", 1.0)),
            bool(self.config.get("scaleRewardsByPlackettLuce", False)),
        )
        environment.capture_replay = self.saved_game_limit > 0
        return environment

    def _save_training_replay(
        self,
        environment: Any,
        training_record: dict[str, Any],
    ) -> None:
        if self.saved_game_limit <= 0:
            return
        frames = getattr(environment, "replay_frames", None)
        if not isinstance(frames, list) or not frames:
            return
        replay_dir = self.output / "replays"
        replay_dir.mkdir(parents=True, exist_ok=True)
        replay_id = f"episode-{int(training_record['episode']):08d}"
        target = replay_dir / f"{replay_id}.json"
        saved_target = self.output / "saved-replays" / f"{replay_id}.json"
        if target.exists() or saved_target.exists():
            replay_id = (
                f"{replay_id}-{int(training_record.get('completedAtUnixMs') or time.time() * 1000)}"
            )
        payload = {
            "schemaVersion": "deepdeck-replay/v1",
            "id": replay_id,
            "createdAtUnixMs": training_record.get("completedAtUnixMs"),
            "metadata": training_record,
            "frames": frames,
        }
        target = replay_dir / f"{replay_id}.json"
        pending = replay_dir / f"{replay_id}.json.pending"
        pending.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        pending.replace(target)
        replay_files = sorted(
            replay_dir.glob("episode-*.json"),
            key=lambda path: path.name,
        )
        excess = max(0, len(replay_files) - self.saved_game_limit)
        for expired in replay_files:
            if excess <= 0:
                break
            lease_dir = replay_dir / ".leases" / expired.stem
            protected = False
            for lease in lease_dir.glob("*.lease") if lease_dir.is_dir() else ():
                try:
                    protected = int(lease.read_text(encoding="utf-8").strip()) > int(
                        time.time()
                    )
                except (OSError, ValueError):
                    continue
                if protected:
                    break
            if protected:
                continue
            expired.unlink(missing_ok=True)
            excess -= 1

    def _apply_gpu_memory_limit(self, limit_mb: int) -> None:
        if self.device.type != "cuda":
            return
        device_index = (
            self.device.index if self.device.index is not None else torch.cuda.current_device()
        )
        total = torch.cuda.get_device_properties(device_index).total_memory
        fraction = 1.0 if limit_mb <= 0 else min(1.0, (limit_mb * 1024 * 1024) / total)
        torch.cuda.set_per_process_memory_fraction(fraction, device_index)

    def _refresh_learner_resources(self) -> None:
        if self.resource_plan_path is None or not self.resource_plan_path.is_file():
            return
        try:
            plan = json.loads(self.resource_plan_path.read_text(encoding="utf-8"))
            workers = max(1, min(32, int(plan.get("trainingMatches", 1))))
            gpu_limit = max(0, int(plan.get("gpuMemoryMb", 0)))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return
        if workers != self.parallel_game_workers:
            if workers > len(self.environments):
                self.environments.extend(
                    self._new_training_environment()
                    for _ in range(workers - len(self.environments))
                )
            else:
                self.environments = self.environments[:workers]
            self.parallel_game_workers = workers
            self.rollout_batch_games = workers
            self.environment = self.environments[0]
            for worker_index, environment in enumerate(self.environments):
                environment.progress_callback = partial(
                    self._observe_training_view,
                    worker_index,
                )
        if gpu_limit != self.gpu_memory_limit_mb:
            self._apply_gpu_memory_limit(gpu_limit)
            self.gpu_memory_limit_mb = gpu_limit

    def _run_ground_truth_evaluation(self, period: int) -> dict[str, Any] | None:
        if not self.ground_truth_scenarios:
            return None
        if self.training_service is None:
            raise RuntimeError(
                "training policy service must be running before ground truth evaluation"
            )
        report = evaluate_ground_truth_service(
            self.ground_truth_scenarios,
            service_url=self.training_service.url,
            controller_id=self.candidate_model_name,
            timeout_seconds=float(self.config.get("engineTimeoutSeconds", 120)),
        )
        report["period"] = period
        report["completedEpisodes"] = self.state.completed_episodes
        report["trainingStep"] = self.learner.training_step
        _append_jsonl(self.output / "ground-truth-evaluations.jsonl", report)
        return report

    def _write_state(self) -> None:
        with self._state_lock:
            desired_state = _training_control_state(self.control_path)
            active_attempts = [
                self.active_attempts[index] for index in sorted(self.active_attempts)
            ]
            _write_json(
                self.output / "league-state.json",
                {
                    **asdict(self.state),
                    "championName": self.state.champion_name,
                    "trainingStep": self.learner.training_step,
                    "continuous": _training_episode_limit(self.config) is None,
                    "evaluationSeedsFixed": True,
                    "trainingSeedSkip": self.training_seed_skip,
                    "matchupRandomizerSkip": self.matchup_randomizer_skip,
                    "groundTruthHealth": self.ground_truth_health,
                    "trainingHealth": self.training_health,
                    "checkpointRecovery": self.checkpoint_recovery,
                    "matchmakingWarning": self._matchmaking_warning,
                    "trainingElapsedSeconds": self.training_elapsed_seconds,
                    "gameSimulationSeconds": self.game_simulation_seconds,
                    "modelTrainingSeconds": self.model_training_seconds,
                    "parallelGameWorkers": self.parallel_game_workers,
                    "rolloutBatchGames": self.rollout_batch_games,
                    "gpuMemoryLimitMb": self.gpu_memory_limit_mb,
                    "trainingPhase": self.training_phase,
                    "trainingDeckCount": len(self.training_deck_names),
                    "trainingDecks": list(self.training_deck_names),
                    "deckCatalogRevision": self.deck_catalog_revision,
                    "activeAttempt": active_attempts[0] if active_attempts else None,
                    "activeAttempts": active_attempts,
                    "lastAttempt": self.last_attempt,
                    "trainingLeaderboard": self.training_leaderboard.payload(),
                    "desiredState": desired_state,
                    "processId": os.getpid(),
                    "updatedAtUnixMs": int(time.time() * 1000),
                },
            )

    def _observe_training_view(
        self,
        worker_index: int,
        view: dict[str, Any],
    ) -> None:
        with self._state_lock:
            attempt = self.active_attempts.get(worker_index)
            if attempt is None:
                return
            tracker = self._active_attempt_trackers.setdefault(
                worker_index,
                {
                    "lastRevision": None,
                    "lastWrite": 0.0,
                    "lastTurn": None,
                    "lastEventCount": 0,
                },
            )
            revision = int(view.get("revision", -1))
            if revision == tracker["lastRevision"]:
                return
            tracker["lastRevision"] = revision
            state = view.get("state") or {}
            decision = view.get("decision")
            turn_number = int(state.get("turnNumber", 0))
            round_value = round_number(
                {"players": state.get("players", [])},
                turn_number,
            )
            events = state.get("events") if isinstance(state.get("events"), list) else []
            new_events = events[tracker["lastEventCount"] :]
            tracker["lastEventCount"] = len(events)
            loop_event_kinds = {
                "decisionLoopAvoided",
                "decisionLoopTurnEndRequested",
                "decisionLoopTurnEnded",
                "loopIterationCountChosen",
                "priorityLoopTerminated",
                "stackObjectExiledForDecisionLoop",
            }
            loop_events = [
                {
                    "sequence": event.get("sequence"),
                    "turnNumber": event.get("turnNumber"),
                    "roundNumber": round_number(
                        {"players": state.get("players", [])},
                        int(event.get("turnNumber") or 0),
                    ),
                    "kind": event.get("kind"),
                    "playerId": event.get("playerId"),
                    "cardInstanceId": event.get("cardInstanceId"),
                    "detail": event.get("detail"),
                }
                for event in new_events
                if isinstance(event, dict) and event.get("kind") in loop_event_kinds
            ]
            if loop_events:
                recent_loop_events = [
                    *attempt.get("recentLoopEvents", []),
                    *loop_events,
                ][-10:]
                attempt["recentLoopEvents"] = recent_loop_events
                attempt["loopEventCount"] = int(attempt.get("loopEventCount", 0)) + len(loop_events)
            recent_events = [
                {
                    "sequence": event.get("sequence"),
                    "turnNumber": event.get("turnNumber"),
                    "roundNumber": round_number(
                        {"players": state.get("players", [])},
                        int(event.get("turnNumber") or 0),
                    ),
                    "kind": event.get("kind"),
                    "playerId": event.get("playerId"),
                    "cardInstanceId": event.get("cardInstanceId"),
                    "detail": {
                        key: value
                        for key, value in (event.get("detail") or {}).items()
                        if isinstance(value, (str, int, float, bool)) or value is None
                    },
                }
                for event in events[-15:]
                if isinstance(event, dict)
            ]
            if decision is not None:
                attempt["decisions"] = int(attempt.get("decisions", 0)) + 1
            attempt.update(
                {
                    "sessionId": view.get("sessionId"),
                    "turnNumber": turn_number,
                    "roundNumber": round_value,
                    "gameStatus": state.get("status"),
                    "setScore": view.get("matchState"),
                    "playersState": [
                        {
                            "id": player.get("id"),
                            "name": player.get("name"),
                            "life": player.get("life"),
                            "hasLost": bool(player.get("hasLost")),
                            "handCount": len(player.get("hand") or []),
                            "battlefieldCount": len(player.get("battlefield") or []),
                        }
                        for player in state.get("players", [])
                    ],
                    "decision": (
                        {
                            "id": decision.get("id"),
                            "kind": decision.get("kind"),
                            "playerId": decision.get("playerId"),
                            "optionCount": len(decision.get("options") or []),
                            "sourceCardInstanceId": decision.get("sourceCardInstanceId"),
                            "sourceCardName": (decision.get("sourceCard") or {}).get("name"),
                        }
                        if decision is not None
                        else None
                    ),
                    "updatedAtUnixMs": int(time.time() * 1000),
                    "recentEvents": recent_events,
                }
            )
            now = time.monotonic()
            if (
                turn_number != tracker["lastTurn"]
                or now - tracker["lastWrite"] >= 5.0
                or decision is None
            ):
                tracker["lastTurn"] = turn_number
                tracker["lastWrite"] = now
                self._write_state()

    def _wait_until_running(self) -> None:
        while _training_control_state(self.control_path) == "paused":
            if not self.state.paused:
                self.state.paused = True
                self._write_state()
            time.sleep(0.5)
        if self.state.paused:
            self.state.paused = False
            self._write_state()

    def _leaderboard_matchmaking_stats(
        self,
        participant_id: str,
        deck_names: list[str] | tuple[str, ...],
    ) -> dict[str, dict[str, float | int | None]]:
        return self.training_leaderboard.deck_matchmaking_stats(
            participant_id,
            deck_names,
        )

    def _sample_anchor_challenge(
        self,
        learner_deck_name: str,
        player_count: int,
    ) -> tuple[int, int, dict[str, Any]]:
        sampler = self.training_matchup_sampler
        random_floor = sampler.matchmaking_random_floor if sampler else 0.20
        rating_scale = sampler.matchmaking_rating_scale if sampler else 10.0
        underplayed_strength = sampler.matchmaking_underplayed_strength if sampler else 0.35
        game_prior = sampler.matchmaking_match_prior if sampler else 10.0
        training_participant_id = getattr(self, "training_participant_id", "v11")
        learner_stat = self.training_leaderboard.deck_matchmaking_stats(
            training_participant_id,
            [learner_deck_name],
        )[learner_deck_name]
        candidates: list[tuple[int, int, str, dict[str, float | int | None]]] = []
        for deadline_round in self.anchor_deadline_rounds:
            for pool_size in self.anchor_opening_hand_pool_sizes:
                participant_id = anchor_participant_id(
                    deadline_round,
                    pool_size,
                    player_count,
                )
                stat = self.training_leaderboard.deck_matchmaking_stats(
                    participant_id,
                    ["Anchor"],
                )["Anchor"]
                candidates.append((deadline_round, pool_size, participant_id, stat))
        minimum_games = min(
            (max(0, int(candidate[3].get("games", 0) or 0)) for candidate in candidates),
            default=0,
        )
        target_ordinal = float(learner_stat.get("ordinal", 0.0) or 0.0)
        weights = [
            _plackett_luce_matchmaking_weight(
                candidate[3],
                target_ordinal=target_ordinal,
                minimum_games=minimum_games,
                random_floor=random_floor,
                rating_scale=rating_scale,
                underplayed_strength=underplayed_strength,
                game_prior=game_prior,
            )
            for candidate in candidates
        ]
        selected_index = self.matchup_randomizer.choices(
            range(len(candidates)),
            weights=weights,
            k=1,
        )[0]
        deadline_round, pool_size, participant_id, stat = candidates[selected_index]
        return (
            deadline_round,
            pool_size,
            {
                "participantId": participant_id,
                "deck": "Anchor",
                "playerCount": player_count,
                "anchorOpponentCount": player_count - 1,
                "mu": stat["mu"],
                "sigma": stat["sigma"],
                "ordinal": stat["ordinal"],
                "rank": stat["rank"],
                "games": stat["games"],
                "weight": weights[selected_index],
                "target": {
                    "participantId": training_participant_id,
                    "deck": learner_deck_name,
                    "ordinal": target_ordinal,
                },
            },
        )

    def _refresh_training_decks_if_changed(self) -> None:
        revision = _deck_catalog_revision(self.config)
        if revision is None or revision == self.deck_catalog_revision:
            return

        decks = _deck_catalog(self.config)
        randomizer_config = self.training_randomizer_config
        if randomizer_config:
            sampler = RandomTrainingMatchupSampler(randomizer_config, decks)
            sampler.validate_commander_decks(
                str(self.config.get("engineUrl", "http://127.0.0.1:8787")),
                float(self.config.get("engineTimeoutSeconds", 120)),
            )
            if self.training_matchup_sampler is not None:
                sampler.format_sample_index = self.training_matchup_sampler.format_sample_index
            self.training_matchup_sampler = sampler
            template = sampler.template()
            refreshed_matchups = {template.id: template}
        else:
            refreshed_matchups = _training_matchups(self.config, decks)
        # RustSelfPlayEnvironment keeps this mapping by reference. Mutate it in
        # place so live workers immediately see newly generated matchup ids.
        self.training_matchups.clear()
        self.training_matchups.update(refreshed_matchups)
        self.training_deck_names = tuple(decks)
        self.deck_catalog_revision = revision

    def _sample_training_opponent_mode(self) -> str:
        modes = list(self.training_opponent_mix)
        weights = [self.training_opponent_mix[mode] for mode in modes]
        return self.matchup_randomizer.choices(modes, weights=weights, k=1)[0]

    @staticmethod
    def _anchor_action(step: Any) -> int:
        priorities = (
            "keephand",
            "playland",
            "castspell",
            "activateability",
            "declareattacker",
            "finishattackers",
            "finishblockers",
            "passpriority",
        )
        by_kind: dict[str, list[int]] = {}
        for index, action in enumerate(step.actions):
            kind = str(action.get("kind", "")).replace("-", "").casefold()
            by_kind.setdefault(kind, []).append(index)
        for kind in priorities:
            candidates = by_kind.get(kind)
            if candidates:
                # Expanded numeric actions are ordered from minimum to maximum;
                # the simple anchor always spends the largest available amount.
                return candidates[-1]
        return 0

    def _external_action(self, environment: Any, step: Any) -> int:
        participant_by_player = getattr(
            environment,
            "participant_by_player_id",
            {},
        )
        participant = participant_by_player.get(step.player_id)
        if str(participant).startswith("anchor-m"):
            return self._anchor_action(step)
        if participant != "v10" or self.v10_opponent_client is None:
            raise RuntimeError(f"external training participant {participant!r} has no policy")
        current_view = environment.current_view
        if not isinstance(current_view, dict) or not isinstance(
            current_view.get("decision"),
            dict,
        ):
            raise RuntimeError("external V10 policy has no current Rust decision")
        decision = current_view["decision"]
        _, _, response = self.v10_opponent_client.choose_detailed(
            step.state,
            decision,
            f"{environment.session_id}:{step.player_id}:v10",
        )
        action_id = str(response.get("actionId", ""))
        number_value = response.get("numberValue")
        for index, action in enumerate(step.actions):
            if str(action.get("_engineActionId", action.get("id", ""))) != action_id:
                continue
            if number_value is not None and action.get("_numberValue") != number_value:
                continue
            return index
        raise RuntimeError(f"V10 returned action {action_id!r} outside the expanded legal choices")

    def train(self) -> None:
        episode_limit = _training_episode_limit(self.config)
        evaluation_every = int(
            self.config.get("modelEvaluationEvery", self.config.get("evaluationEvery", 100))
        )
        ground_truth_every = int(self.config.get("groundTruthEvaluationEvery", 10))
        checkpoint_every = int(self.config.get("checkpointEvery", evaluation_every))
        service_refresh_every = int(self.config.get("serviceRefreshEvery", 1))
        if service_refresh_every <= 0:
            raise ValueError("serviceRefreshEvery must be positive")
        if self.model_evaluation_enabled and evaluation_every <= 0:
            raise ValueError("model evaluation cadence must be positive")
        if self.ground_truth_evaluation_enabled and ground_truth_every <= 0:
            raise ValueError("ground-truth evaluation cadence must be positive")
        max_consecutive_errors = int(self.config.get("maxConsecutiveErrors", 20))
        consecutive_errors = 0
        self._start_policy_services()
        self._write_state()
        while episode_limit is None or self.state.completed_episodes < episode_limit:
            self._wait_until_running()
            self._refresh_learner_resources()
            self._refresh_training_decks_if_changed()
            batch_size = _rollout_batch_size(
                self.state.completed_episodes,
                self.rollout_batch_games,
                evaluation_cadences=tuple(
                    cadence
                    for enabled, cadence in (
                        (self.ground_truth_evaluation_enabled, ground_truth_every),
                        (self.model_evaluation_enabled, evaluation_every),
                    )
                    if enabled
                ),
                episode_limit=episode_limit,
            )

            contexts: list[dict[str, Any]] = []
            jobs: list[SelfPlayJob] = []
            temporary_matchup_ids: list[str] = []
            batch_profile: TrainingBatchProfile | None = None
            compatible_matchup_ids: list[str] | None = None
            batch_started = time.perf_counter()
            opponent_mode = self._sample_training_opponent_mode()
            with self._state_lock:
                self.training_phase = "collecting"
                self.active_attempts = {}
                self._active_attempt_trackers = {}
                if self.training_matchup_sampler is not None:
                    batch_profile = self.training_matchup_sampler.sample_batch_profile(
                        self.matchup_randomizer
                    )
                else:
                    anchor_matchup_id = self.matchup_randomizer.choice(list(self.training_matchups))
                    anchor_matchup = self.training_matchups[anchor_matchup_id]
                    anchor_player_count = len(anchor_matchup.setup["players"])
                    compatible_matchup_ids = [
                        matchup_id
                        for matchup_id, candidate in self.training_matchups.items()
                        if candidate.game_mode == anchor_matchup.game_mode
                        and len(candidate.setup["players"]) == anchor_player_count
                    ]
                for worker_index in range(batch_size):
                    self.state.attempted_episodes += 1
                    attempt_number = self.state.attempted_episodes
                    seed = self.training_seed_stream.next()
                    self.training_seed_skip += 1
                    if self.training_matchup_sampler is not None:
                        assert batch_profile is not None
                        opponent_participant = {
                            "self": self.training_participant_id,
                            "v10": "v10",
                            # Anchor source decks are replaced below. Keeping the learner
                            # here still balances which learner deck is exposed.
                            "anchor": self.training_participant_id,
                        }[opponent_mode]
                        sampling_participants = [self.training_participant_id] + [
                            opponent_participant for _ in range(batch_profile.player_count - 1)
                        ]
                        matchup = self.training_matchup_sampler.sample(
                            self.matchup_randomizer,
                            self._leaderboard_matchmaking_stats,
                            profile=batch_profile,
                            participant_ids=sampling_participants,
                        )
                        matchmaking = deepcopy(self.training_matchup_sampler.last_selection)
                        display_matchup_id = matchup.id
                        job_matchup_id = f"{matchup.id}:parallel-attempt-{attempt_number}"
                        matchup = replace(matchup, id=job_matchup_id)
                        self.training_matchups[job_matchup_id] = matchup
                        temporary_matchup_ids.append(job_matchup_id)
                    else:
                        job_matchup_id = self.matchup_randomizer.choice(compatible_matchup_ids)
                        matchup = self.training_matchups[job_matchup_id]
                        matchmaking = None
                        display_matchup_id = job_matchup_id
                    if opponent_mode == "anchor":
                        deadline_round, opening_hand_pool_size, anchor_selection = (
                            self._sample_anchor_challenge(
                                matchup.deck_names[0],
                                len(matchup.setup["players"]),
                            )
                        )
                        matchmaking = matchmaking or {}
                        matchmaking["anchor"] = anchor_selection
                        sampled_seats = list(matchmaking.get("seats", []))
                        learner_seat = sampled_seats[:1]
                        matchmaking["seats"] = learner_seat + [
                            {
                                **anchor_selection,
                                "seat": seat,
                            }
                            for seat in range(2, len(matchup.setup["players"]) + 1)
                        ]
                        matchmaking["ratingsAvailable"] = bool(
                            matchmaking.get("ratingsAvailable")
                            or int(anchor_selection.get("games", 0) or 0) > 0
                        )
                        anchor_matchup_id = (
                            f"{job_matchup_id}:anchor-r{deadline_round}:"
                            f"n{opening_hand_pool_size}:"
                            f"attempt-{attempt_number}"
                        )
                        matchup = replace(
                            _anchor_matchup(
                                matchup,
                                deadline_round,
                                opening_hand_pool_size,
                            ),
                            id=anchor_matchup_id,
                        )
                        self.training_matchups[anchor_matchup_id] = matchup
                        temporary_matchup_ids.append(anchor_matchup_id)
                        job_matchup_id = anchor_matchup_id
                    player_ids = [str(player["id"]) for player in matchup.setup.get("players", [])]
                    anchor_participant = (
                        anchor_participant_id(
                            matchup.anchor_deadline_round,
                            matchup.anchor_opening_hand_pool_size,
                            len(matchup.setup["players"]),
                        )
                        if (
                            opponent_mode == "anchor"
                            and matchup.anchor_deadline_round
                            and matchup.anchor_opening_hand_pool_size
                        )
                        else opponent_mode
                    )
                    participant_by_player = {
                        player_id: (
                            self.training_participant_id
                            if opponent_mode == "self" or index == 0
                            else anchor_participant
                        )
                        for index, player_id in enumerate(player_ids)
                    }
                    deck_by_player = {
                        str(player["id"]): (
                            matchup.deck_names[index]
                            if index < len(matchup.deck_names)
                            else str(player.get("name", f"Deck {index + 1}"))
                        )
                        for index, player in enumerate(matchup.setup["players"])
                    }
                    learner_player_ids = (
                        None if opponent_mode == "self" else frozenset((player_ids[0],))
                    )
                    environment = self.environments[worker_index]
                    environment.participant_by_player_id = participant_by_player
                    environment.plackett_luce_participant_by_player_id = (
                        self.training_leaderboard.deck_participants(
                            participant_by_player,
                            deck_by_player,
                        )
                    )
                    environment.plackett_luce_ratings_by_player_id = (
                        self.training_leaderboard.player_deck_ratings(
                            participant_by_player,
                            deck_by_player,
                        )
                    )
                    environment.analytics_pilot_override = {
                        player_id: (
                            str(
                                self.config.get(
                                    "analyticsPilotId",
                                    "ia-v11-in-training",
                                )
                            )
                            if participant == self.training_participant_id
                            else (
                                "ia-v10-in-training"
                                if participant == "v10"
                                else "ai-training-anchor"
                            )
                        )
                        for player_id, participant in participant_by_player.items()
                    }
                    self.matchup_randomizer_skip += 1
                    attempt = {
                        "attempt": attempt_number,
                        "worker": worker_index + 1,
                        "batchSize": batch_size,
                        "status": "collecting",
                        "seed": seed,
                        "matchupId": display_matchup_id,
                        "gameMode": matchup.game_mode,
                        "players": len(matchup.setup["players"]),
                        "decks": list(matchup.deck_names),
                        "startingLife": matchup.setup["players"][0]["startingLife"],
                        "openingHandSize": matchup.setup["openingHandSize"],
                        "startingPlayer": matchup.setup["startingPlayer"],
                        "freeMulligans": matchup.free_mulligans,
                        "maxMulligans": matchup.max_mulligans,
                        "retainedDeckSize": _retained_deck_size(matchup.game_mode),
                        "matchmaking": matchmaking,
                        "opponentMode": opponent_mode,
                        "participantsByPlayer": participant_by_player,
                        "anchorDeadlineRound": matchup.anchor_deadline_round,
                        "anchorOpeningHandPoolSize": matchup.anchor_opening_hand_pool_size,
                        "anchorPlayerCount": (
                            len(matchup.setup["players"]) if matchup.anchor_deadline_round else None
                        ),
                        "anchorOpponentCount": (
                            len(matchup.setup["players"]) - 1
                            if matchup.anchor_deadline_round
                            else None
                        ),
                        "decisions": 0,
                        "loopEventCount": 0,
                        "recentLoopEvents": [],
                        "startedAtUnixMs": int(time.time() * 1000),
                    }
                    self.active_attempts[worker_index] = attempt
                    contexts.append(
                        {
                            "workerIndex": worker_index,
                            "attempt": attempt,
                            "matchup": matchup,
                            "displayMatchupId": display_matchup_id,
                            "seed": seed,
                            "participantByPlayer": participant_by_player,
                            "deckByPlayer": deck_by_player,
                        }
                    )
                    jobs.append(
                        SelfPlayJob(
                            environment,
                            job_matchup_id,
                            seed,
                            learner_player_ids=learner_player_ids,
                            external_action_selector=(
                                None if opponent_mode == "self" else self._external_action
                            ),
                        )
                    )
            self._write_state()

            collections = self.learner.collect_self_play_batch(
                jobs,
                max_workers=self.parallel_game_workers,
            )
            collection_wall_seconds = time.perf_counter() - batch_started
            successful: list[tuple[dict[str, Any], Any]] = []
            failed: list[tuple[dict[str, Any], Any]] = []
            for context, collection in zip(contexts, collections):
                if collection.error is None and collection.terminal is not None:
                    successful.append((context, collection))
                else:
                    failed.append((context, collection))

            for context, collection in failed:
                attempt = context["attempt"]
                error = collection.error or RuntimeError(
                    "parallel collection ended without a terminal state"
                )
                error_record = {
                    **{
                        key: attempt.get(key)
                        for key in (
                            "attempt",
                            "seed",
                            "matchupId",
                            "gameMode",
                            "players",
                            "decks",
                            "startingLife",
                            "openingHandSize",
                            "startingPlayer",
                            "freeMulligans",
                            "maxMulligans",
                            "retainedDeckSize",
                            "matchmaking",
                        )
                    },
                    "completedEpisodes": self.state.completed_episodes,
                    "worker": attempt.get("worker"),
                    "batchSize": batch_size,
                    "secondsBeforeError": collection.collection_seconds,
                    "error": f"{type(error).__name__}: {error}",
                    "turnNumber": attempt.get("turnNumber"),
                    "roundNumber": attempt.get("roundNumber"),
                    "decision": attempt.get("decision"),
                    "recentEvents": attempt.get("recentEvents", []),
                    "recentLoopEvents": attempt.get("recentLoopEvents", []),
                }
                _append_jsonl(self.output / "training-errors.jsonl", error_record)
                print(
                    json.dumps(
                        {
                            "trainingError": {
                                "attempt": error_record["attempt"],
                                "worker": error_record["worker"],
                                "error": error_record["error"],
                            }
                        }
                    ),
                    flush=True,
                )
                self.last_attempt = {
                    **attempt,
                    "status": "failed",
                    "error": error_record["error"],
                    "failedAtUnixMs": int(time.time() * 1000),
                }

            previous_completed = self.state.completed_episodes
            if not successful:
                self.training_elapsed_seconds += collection_wall_seconds
                consecutive_errors += max(1, len(failed))
                with self._state_lock:
                    self.active_attempts = {}
                    self._active_attempt_trackers = {}
                for matchup_id in temporary_matchup_ids:
                    self.training_matchups.pop(matchup_id, None)
                self._write_state()
                if consecutive_errors >= max_consecutive_errors:
                    raise RuntimeError(
                        f"training stopped after {consecutive_errors} consecutive collection errors"
                    )
                continue

            with self._state_lock:
                self.training_phase = "optimizingWeights"
                for context, collection in successful:
                    context["attempt"].update(
                        {
                            "status": "optimizingWeights",
                            "decisions": len(collection.trajectory),
                            "collectionSeconds": collection.collection_seconds,
                            "updatedAtUnixMs": int(time.time() * 1000),
                        }
                    )
            self._write_state()
            batch_trajectory = [
                transition for _, collection in successful for transition in collection.trajectory
            ]
            update_started = time.perf_counter()
            try:
                metrics = self.learner.update(batch_trajectory)
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
            except Exception as error:
                is_cuda_oom = self.device.type == "cuda" and "out of memory" in str(error).lower()
                if is_cuda_oom:
                    self.learner.optimizer.zero_grad(set_to_none=True)
                    gc.collect()
                    torch.cuda.empty_cache()
                consecutive_errors += 1
                batch_error = {
                    "attempt": contexts[0]["attempt"]["attempt"],
                    "completedEpisodes": self.state.completed_episodes,
                    "batchSize": batch_size,
                    "successfulCollections": len(successful),
                    "decisions": len(batch_trajectory),
                    "secondsBeforeError": time.perf_counter() - batch_started,
                    "error": f"{type(error).__name__}: {error}",
                }
                _append_jsonl(self.output / "training-errors.jsonl", batch_error)
                print(
                    json.dumps(
                        {
                            "trainingBatchError": {
                                "attempt": batch_error["attempt"],
                                "batchSize": batch_error["batchSize"],
                                "error": batch_error["error"],
                            }
                        }
                    ),
                    flush=True,
                )
                with self._state_lock:
                    self.active_attempts = {}
                    self._active_attempt_trackers = {}
                for matchup_id in temporary_matchup_ids:
                    self.training_matchups.pop(matchup_id, None)
                self._write_state()
                if is_cuda_oom:
                    time.sleep(float(self.config.get("cudaOutOfMemoryBackoffSeconds", 30)))
                if consecutive_errors >= max_consecutive_errors:
                    raise RuntimeError(
                        f"training stopped after {consecutive_errors} consecutive errors"
                    ) from error
                continue

            update_seconds = time.perf_counter() - update_started
            batch_seconds = time.perf_counter() - batch_started
            self.training_elapsed_seconds += batch_seconds
            self.game_simulation_seconds += sum(
                collection.collection_seconds for _, collection in successful
            )
            self.model_training_seconds += update_seconds
            self.state.completed_episodes += len(successful)
            consecutive_errors = 0
            training_hour = max(
                1,
                math.ceil(self.training_elapsed_seconds / 3600.0),
            )
            update_share = update_seconds / len(successful)
            elapsed_share = batch_seconds / len(successful)
            for success_index, (context, collection) in enumerate(successful):
                attempt = context["attempt"]
                matchup = context["matchup"]
                terminal = collection.terminal
                assert terminal is not None
                state = terminal.state
                outcome = state.get("outcome")
                decisions_by_player = Counter(
                    transition.player_id for transition in collection.trajectory
                )
                gameplay = summarize_gameplay_metrics(state, matchup.setup)
                leaderboard_order = None
                if (
                    matchup.anchor_deadline_round
                    and state.get("status") == "turnLimitReached"
                    and not (state.get("outcome") or {}).get("winner")
                ):
                    anchors = list(matchup.training_anchor_player_ids)
                    leaderboard_order = anchors + [
                        player_id
                        for player_id in context["participantByPlayer"]
                        if player_id not in anchors
                    ]
                self.training_leaderboard.update(
                    context["participantByPlayer"],
                    state,
                    ordered_players=leaderboard_order,
                    deck_by_player=context["deckByPlayer"],
                )
                self.hourly_gameplay = update_hourly_gameplay_metrics(
                    self.hourly_gameplay,
                    gameplay,
                    training_hour,
                )
                training_record = {
                    "episode": previous_completed + success_index + 1,
                    "attempt": attempt["attempt"],
                    "trainingStep": self.learner.training_step,
                    "seed": context["seed"],
                    "matchupId": context["displayMatchupId"],
                    "gameMode": matchup.game_mode,
                    "players": len(matchup.setup["players"]),
                    "decks": list(matchup.deck_names),
                    "startingLife": matchup.setup["players"][0]["startingLife"],
                    "openingHandSize": matchup.setup["openingHandSize"],
                    "startingPlayer": matchup.setup["startingPlayer"],
                    "freeMulligans": matchup.free_mulligans,
                    "maxMulligans": matchup.max_mulligans,
                    "retainedDeckSize": _retained_deck_size(matchup.game_mode),
                    "matchmaking": attempt.get("matchmaking"),
                    "opponentMode": attempt.get("opponentMode"),
                    "participantsByPlayer": context["participantByPlayer"],
                    "anchorDeadlineRound": matchup.anchor_deadline_round,
                    "anchorOpeningHandPoolSize": matchup.anchor_opening_hand_pool_size,
                    "anchorPlayerCount": (
                        len(matchup.setup["players"]) if matchup.anchor_deadline_round else None
                    ),
                    "anchorOpponentCount": (
                        len(matchup.setup["players"]) - 1 if matchup.anchor_deadline_round else None
                    ),
                    "decisions": len(collection.trajectory),
                    "decisionsByPlayer": dict(decisions_by_player),
                    "rewardsByPlayer": terminal.rewards_by_player,
                    "turnNumber": state.get("turnNumber"),
                    "roundNumber": round_number(
                        matchup.setup,
                        int(state.get("turnNumber") or 0),
                    ),
                    "gameStatus": state.get("status"),
                    "setScore": state.get("_matchState"),
                    "outcome": outcome,
                    "gameDurationSeconds": collection.collection_seconds,
                    "trainingDurationSeconds": update_share,
                    "collectionSeconds": collection.collection_seconds,
                    "ppoSeconds": update_share,
                    "episodeSeconds": elapsed_share,
                    "rolloutBatch": {
                        "requestedGames": batch_size,
                        "completedGames": len(successful),
                        "failedGames": len(failed),
                        "parallelWorkers": self.parallel_game_workers,
                        "gameMode": matchup.game_mode,
                        "players": len(matchup.setup["players"]),
                        "totalDecisions": len(batch_trajectory),
                        "gameDurationsSeconds": [item.collection_seconds for _, item in successful],
                        "gameSimulationSeconds": sum(
                            item.collection_seconds for _, item in successful
                        ),
                        "collectionWallSeconds": collection_wall_seconds,
                        "trainingWallSeconds": update_seconds,
                        "totalWallSeconds": batch_seconds,
                        "ppoWallSeconds": update_seconds,
                    },
                    "completedAtUnixMs": int(time.time() * 1000),
                    "trainingHour": training_hour,
                    "gameplay": gameplay,
                    "ppo": metrics,
                    "behavior": summarize_decision_traces(
                        [
                            transition.decision_trace
                            for transition in collection.trajectory
                            if transition.decision_trace is not None
                        ],
                        sample_limit=5,
                    ),
                }
                _append_jsonl(self.output / "training.jsonl", training_record)
                self._save_training_replay(collection.job.environment, training_record)
                # The full record already lives in training.jsonl. Emitting it
                # again made learner-process.log grow by several kilobytes per
                # game, so stdout carries only the live progress summary.
                print(
                    json.dumps(
                        {
                            "training": {
                                "episode": training_record["episode"],
                                "trainingStep": training_record["trainingStep"],
                                "gameStatus": training_record["gameStatus"],
                                "gameDurationSeconds": training_record["gameDurationSeconds"],
                            }
                        }
                    ),
                    flush=True,
                )
                self.last_attempt = {
                    **attempt,
                    "status": "completed",
                    "episode": training_record["episode"],
                    "trainingStep": self.learner.training_step,
                    "gameDurationSeconds": collection.collection_seconds,
                    "trainingDurationSeconds": update_share,
                    "ppoSeconds": update_share,
                    "outcome": outcome,
                    "completedAtUnixMs": int(time.time() * 1000),
                }
            _write_json(self.hourly_gameplay_path, self.hourly_gameplay)
            self._save_live_checkpoint()
            with self._state_lock:
                self.training_phase = "betweenBatches"
                self.active_attempts = {}
                self._active_attempt_trackers = {}
            for matchup_id in temporary_matchup_ids:
                self.training_matchups.pop(matchup_id, None)
            del batch_trajectory
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

            def crossed(cadence: int) -> bool:
                return previous_completed // cadence < self.state.completed_episodes // cadence

            checkpoint: Path | None = None
            evaluation_due = self.model_evaluation_enabled and crossed(evaluation_every)
            ground_truth_due = self.ground_truth_evaluation_enabled and crossed(ground_truth_every)
            final_episode = (
                episode_limit is not None and self.state.completed_episodes >= episode_limit
            )
            if (
                crossed(service_refresh_every)
                or evaluation_due
                or ground_truth_due
                or final_episode
            ):
                self._refresh_training_service(checkpoint_is_current=True)
            if crossed(checkpoint_every) or evaluation_due or final_episode:
                checkpoint = self._save_candidate()
            if evaluation_due:
                self.training_phase = "modelEvaluation"
                self._write_state()
                assert checkpoint is not None
                evaluation = self._run_evaluation(
                    self.state.completed_episodes // evaluation_every,
                    checkpoint,
                )
                print(
                    json.dumps(
                        {
                            "evaluation": {
                                "period": evaluation["period"],
                                "candidateTrainingStep": evaluation["candidateTrainingStep"],
                                "opponentVersion": evaluation["opponentVersion"],
                                "summary": evaluation["summary"],
                                "perfectStreakAfter": evaluation["perfectStreakAfter"],
                                "promotion": evaluation["promotion"],
                            }
                        }
                    ),
                    flush=True,
                )
            if ground_truth_due:
                self.training_phase = "groundTruthEvaluation"
                self._write_state()
                ground_truth_report = self._run_ground_truth_evaluation(
                    self.state.completed_episodes // ground_truth_every,
                )
                if ground_truth_report is not None:
                    print(
                        json.dumps(
                            {
                                "groundTruthEvaluation": {
                                    "period": ground_truth_report["period"],
                                    "trainingStep": ground_truth_report["trainingStep"],
                                    "metrics": ground_truth_report["metrics"],
                                }
                            }
                        ),
                        flush=True,
                    )
            self.training_phase = "betweenBatches"
            self._write_state()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Oracle AI in shared-policy self-play with champion evaluation"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--additional-episodes",
        type=int,
        help="Run exactly this many episodes beyond the resumed league state.",
    )
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    trainer = LeagueTrainer(config)
    try:
        if args.additional_episodes is not None:
            _set_additional_episode_limit(
                config,
                trainer.state.completed_episodes,
                args.additional_episodes,
            )
        trainer.train()
    finally:
        trainer.close()


if __name__ == "__main__":
    main()
