from __future__ import annotations

import json
import random
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import httpx

from oracle_ai.decision_choices import expand_policy_actions
from oracle_ai.training.core import DecisionStep
from oracle_ai.training.plackett_luce import (
    PlackettLuceRating,
    ordered_finishers,
    rank_gradient_rewards,
)


@dataclass(frozen=True)
class Matchup:
    id: str
    setup: dict[str, Any]
    learner_player_id: str
    opponent_player_id: str
    max_turns: int = 200
    mulligan_enabled: bool = False
    free_mulligans: int = 0
    max_mulligans: int | None = None
    game_mode: str = "free"
    deck_names: tuple[str, ...] = ()
    deck_session_ids: tuple[str, ...] = ()
    punching_bag_player_ids: tuple[str, ...] = ()
    training_anchor_player_ids: tuple[str, ...] = ()
    anchor_deadline_round: int | None = None
    anchor_opening_hand_pool_size: int | None = None
    scenario_id: str | None = None
    scenario_family: str | None = None
    scenario_version: int = 1
    objective: str | None = None
    difficulty: int | None = None
    target_round: int | None = None
    fixed_opening_hand_definition_ids: tuple[str, ...] = ()
    opening_hand_target_roles: tuple[tuple[str, ...], ...] = ()
    success_card_names: tuple[str, ...] = ()
    success_zones: tuple[str, ...] = ("battlefield",)
    success_action_sequence: tuple[str, ...] = ()
    start_with_sideboarding: bool = False
    sideboard_target_card_names: tuple[str, ...] = ()
    sideboard_cut_card_names: tuple[str, ...] = ()


class RustSessionEnvironment:
    """Gym-like adapter over authoritative Rust `/game/sessions` endpoints.

    The setup is supplied by a versioned matchup manifest. Rust advances all
    non-learner players. The learner receives only the session projection and
    legal options published for its current decision.
    """

    replay_zones = ("library", "hand", "battlefield", "graveyard", "exile", "commandZone")

    def __init__(
        self,
        base_url: str,
        matchups: dict[str, Matchup],
        timeout_seconds: float = 30.0,
        learner_pilot_id: str = "ia-in-training",
    ) -> None:
        self.client = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout_seconds)
        self.wait_timeout_ms = max(
            1_000,
            min(600_000, int(timeout_seconds * 1000) - 1_000),
        )
        self.matchups = matchups
        self.session_id: str | None = None
        self.current_view: dict[str, Any] | None = None
        self.learner_player_id: str | None = None
        self.current_matchup: Matchup | None = None
        self.known_decks_by_player_id: dict[str, list[dict[str, Any]]] = {}
        self.pregame_commanders: list[dict[str, Any]] = []
        self.previous_observations_by_player_id: dict[str, dict[str, Any]] = {}
        self.learner_pilot_id = learner_pilot_id
        self.analytics_context_id = f"training:{learner_pilot_id}"
        self.analytics_pilot_override: dict[str, str] | None = None
        self.progress_callback: Callable[[dict[str, Any]], None] | None = None
        self.capture_replay = False
        self.replay_frames: list[dict[str, Any]] = []
        self.objective_completed = False
        self.objective_action_labels: list[str] = []
        self.objective_milestones_completed = 0
        self.opening_hand_selected_card_names: list[str] = []
        self.opening_hand_roles_available = 0
        self.objective_damage_progress = 0.0
        self.sideboard_cards_selected = 0
        self.sideboard_target_cards_selected = 0
        self.sideboard_target_cards_available = 0
        self.sideboard_cut_cards_selected = 0
        self.sideboard_cut_cards_expected = 0

    def _objective_sequence_progress(self) -> int:
        matchup = self.current_matchup
        if matchup is None or not matchup.success_action_sequence:
            return 0
        progress = 0
        for label in self.objective_action_labels:
            if matchup.success_action_sequence[progress].casefold() in label.casefold():
                progress += 1
                if progress == len(matchup.success_action_sequence):
                    break
        return progress

    @staticmethod
    def _opening_hand_card_name(action: dict[str, Any]) -> str:
        candidate = (action.get("decisions") or {}).get("openingHandCandidate")
        if not isinstance(candidate, dict):
            return ""
        definition = candidate.get("definition")
        return str(definition.get("name", "")) if isinstance(definition, dict) else ""

    def _initialize_opening_hand_roles(self, step: DecisionStep) -> None:
        matchup = self.current_matchup
        if matchup is None or not matchup.opening_hand_target_roles:
            self.opening_hand_roles_available = 0
            return
        candidate_names = {
            self._opening_hand_card_name(action) for action in step.actions
        }
        self.opening_hand_roles_available = sum(
            bool(candidate_names.intersection(role))
            for role in matchup.opening_hand_target_roles
        )

    def _opening_hand_role_progress(self) -> int:
        matchup = self.current_matchup
        if matchup is None or not matchup.opening_hand_target_roles:
            return 0
        selected = set(self.opening_hand_selected_card_names)
        return sum(bool(selected.intersection(role)) for role in matchup.opening_hand_target_roles)

    def _damage_progress(self, view: dict[str, Any]) -> float:
        matchup = self.current_matchup
        if (
            matchup is None
            or matchup.objective not in {"fast-win", "combo-win"}
            or not matchup.training_anchor_player_ids
        ):
            return 0.0
        opponent = next(
            (
                player
                for player in (view.get("state", {}) or {}).get("players", [])
                if isinstance(player, dict)
                and str(player.get("id", "")) == matchup.opponent_player_id
            ),
            None,
        )
        if opponent is None:
            return 0.0
        starting_life = next(
            (
                float(player.get("startingLife", 20) or 20)
                for player in matchup.setup.get("players", [])
                if str(player.get("id", "")) == matchup.opponent_player_id
            ),
            20.0,
        )
        if starting_life <= 0:
            return 0.0
        current_life = float(opponent.get("life", starting_life) or 0)
        return max(0.0, min(1.0, (starting_life - current_life) / starting_life))

    @staticmethod
    def _sideboard_cards_brought_in(selected_action: dict[str, Any]) -> int:
        decisions = selected_action.get("decisions") or {}
        final_main_deck_ids = {
            str(card_id)
            for key, values in decisions.items()
            if "configure:cards" in str(key)
            for card_id in (values if isinstance(values, list) else [])
        }
        initial_main_deck_ids = {
            str(card_id)
            for card_id in (
                decisions.get("initialMainDeckIds", [])
                if isinstance(decisions.get("initialMainDeckIds"), list)
                else []
            )
        }
        return len(final_main_deck_ids - initial_main_deck_ids)

    def _policy_actions(self, decision: dict[str, Any]) -> list[dict[str, Any]]:
        card_names = {
            str(card.get("id")): str(card.get("name", card.get("id", "")))
            for player in (
                self.current_matchup.setup.get("players", [])
                if self.current_matchup
                else []
            )
            for card in player.get("cards", [])
            if isinstance(card, dict) and card.get("id")
        }
        return expand_policy_actions(
            decision,
            card_names,
            self.current_matchup.sideboard_target_card_names
            if self.current_matchup
            else (),
            self.current_matchup.sideboard_cut_card_names
            if self.current_matchup
            else (),
        )

    def _sideboard_target_cards_brought_in(
        self,
        selected_action: dict[str, Any],
    ) -> int:
        matchup = self.current_matchup
        if matchup is None or not matchup.sideboard_target_card_names:
            return 0
        decisions = selected_action.get("decisions") or {}
        initial = {
            str(value) for value in decisions.get("initialMainDeckIds", [])
        }
        final = {
            str(card_id)
            for key, values in decisions.items()
            if "configure:cards" in str(key)
            for card_id in (values if isinstance(values, list) else [])
        }
        definitions = {
            str(card.get("id")): str(card.get("name", ""))
            for player in matchup.setup.get("players", [])
            if str(player.get("id", "")) == matchup.learner_player_id
            for card in player.get("cards", [])
            if isinstance(card, dict) and card.get("id")
        }
        targets = set(matchup.sideboard_target_card_names)
        selected = 0
        for instance_id in final - initial:
            parts = instance_id.split(":")
            definition_id = ":".join(parts[1:-1]) if len(parts) >= 3 else instance_id
            selected += int(definitions.get(definition_id) in targets)
        return selected

    def _sideboard_cut_cards_taken_out(self, selected_action: dict[str, Any]) -> int:
        matchup = self.current_matchup
        if matchup is None or not matchup.sideboard_cut_card_names:
            return 0
        decisions = selected_action.get("decisions") or {}
        initial = {str(value) for value in decisions.get("initialMainDeckIds", [])}
        final = {
            str(card_id)
            for key, values in decisions.items()
            if "configure:cards" in str(key)
            for card_id in (values if isinstance(values, list) else [])
        }
        definitions = {
            str(card.get("id")): str(card.get("name", ""))
            for player in matchup.setup.get("players", [])
            if str(player.get("id", "")) == matchup.learner_player_id
            for card in player.get("cards", [])
            if isinstance(card, dict) and card.get("id")
        }
        targets = set(matchup.sideboard_cut_card_names)
        selected = 0
        for instance_id in initial - final:
            parts = instance_id.split(":")
            definition_id = ":".join(parts[1:-1]) if len(parts) >= 3 else instance_id
            selected += int(definitions.get(definition_id) in targets)
        return min(selected, self.sideboard_cut_cards_expected)

    @staticmethod
    def _compact_replay_card(card: Any) -> dict[str, Any] | None:
        if not isinstance(card, dict):
            return None
        definition = card.get("definition")
        definition = definition if isinstance(definition, dict) else {}
        return {
            key: value
            for key, value in {
                "instanceId": card.get("instanceId"),
                "name": definition.get("name", card.get("name")),
                "cardId": definition.get("id", card.get("cardId")),
                "controllerId": card.get("controllerId", card.get("controller")),
                "ownerId": card.get("ownerId", card.get("owner")),
                "tapped": card.get("tapped"),
                "manaCost": definition.get("manaCost", card.get("manaCost")),
                "typeLine": definition.get("typeLine", card.get("typeLine")),
                "power": card.get("power", definition.get("power")),
                "toughness": card.get("toughness", definition.get("toughness")),
                "counters": card.get("counters"),
            }.items()
            if value not in (None, [], {})
        }

    @classmethod
    def _compact_replay_view(cls, view: dict[str, Any]) -> dict[str, Any]:
        state = view.get("state", {})
        compact_state = {
            key: deepcopy(value)
            for key, value in state.items()
            if key
            in {
                "activePlayerId",
                "priorityPlayerId",
                "turnNumber",
                "step",
                "status",
                "outcome",
                "winnerIds",
            }
        }
        compact_players = []
        for player in state.get("players", []):
            if not isinstance(player, dict):
                continue
            compact_player = {
                key: deepcopy(player.get(key))
                for key in ("id", "name", "life", "poisonCounters", "hasLost")
                if player.get(key) is not None
            }
            for zone in cls.replay_zones:
                cards = player.get(zone, [])
                if isinstance(cards, list):
                    compact_player[zone] = [
                        compact
                        for card in cards
                        if (compact := cls._compact_replay_card(card)) is not None
                    ]
            compact_players.append(compact_player)
        compact_state["players"] = compact_players
        stack = state.get("stack", [])
        if isinstance(stack, list):
            compact_state["stack"] = []
            for stack_object in stack:
                if not isinstance(stack_object, dict):
                    continue
                compact_card = cls._compact_replay_card(stack_object.get("card"))
                if compact_card is None:
                    continue
                compact_state["stack"].append(
                    {
                        key: deepcopy(value)
                        for key, value in {
                            "id": stack_object.get("id"),
                            "controller": stack_object.get("controller"),
                            "abilityKind": stack_object.get("abilityKind"),
                            "card": compact_card,
                        }.items()
                        if value is not None
                    }
                )
        decision = view.get("decision")
        compact_decision = None
        if isinstance(decision, dict):
            compact_decision = {
                key: deepcopy(decision.get(key))
                for key in ("id", "kind", "playerId", "choice", "sourceCardInstanceId")
                if decision.get(key) is not None
            }
        return {
            "schemaVersion": view.get("schemaVersion", "mtg-game-session/v1"),
            "revision": view.get("revision"),
            "state": compact_state,
            "decision": compact_decision,
            "matchState": deepcopy(view.get("matchState")),
        }

    @staticmethod
    def _compact_replay_action(action: dict[str, Any]) -> dict[str, Any]:
        return {
            key: deepcopy(action.get(key))
            for key in (
                "id",
                "kind",
                "label",
                "cardInstanceId",
                "decisions",
                "targets",
                "_numberValue",
                "_cardInstanceIds",
            )
            if action.get(key) is not None
        }

    def _start_replay(self, view: dict[str, Any]) -> None:
        self.replay_frames = []
        if self.capture_replay:
            self.replay_frames.append(self._compact_replay_view(view))

    def _record_replay_action(self, selected_action: dict[str, Any]) -> None:
        if self.capture_replay and self.replay_frames:
            self.replay_frames[-1]["selectedAction"] = self._compact_replay_action(
                selected_action
            )

    def _record_replay_view(self, view: dict[str, Any]) -> None:
        if self.capture_replay:
            self.replay_frames.append(self._compact_replay_view(view))

    def _report_progress(self, view: dict[str, Any]) -> None:
        if self.progress_callback is not None:
            self.progress_callback(view)

    def _remove_session(self) -> None:
        if self.session_id is None:
            return
        session_id = self.session_id
        self.session_id = None
        response = self.client.delete(f"/game/sessions/{session_id}")
        if response.status_code != 404:
            response.raise_for_status()

    def _discard_failed_session(self) -> None:
        try:
            self._remove_session()
        except httpx.HTTPError:
            self.session_id = None

    def close(self) -> None:
        self._remove_session()
        self.client.close()

    def _known_deck(self, player_id: str | None) -> list[dict[str, Any]]:
        if self.current_matchup is None or player_id is None:
            return []
        if player_id in self.known_decks_by_player_id:
            return list(self.known_decks_by_player_id[player_id])
        return next(
            (
                list(player.get("cards", []))
                for player in self.current_matchup.setup.get("players", [])
                if player.get("id") == player_id
                and isinstance(player.get("cards"), list)
            ),
            [],
        )

    def _capture_known_decks(self, view: dict[str, Any]) -> None:
        known_decks: dict[str, list[dict[str, Any]]] = {}
        commanders: list[dict[str, Any]] = []
        for player in view.get("state", {}).get("players", []):
            player_id = player.get("id")
            if not player_id:
                continue
            definitions: list[dict[str, Any]] = []
            seen_instances: set[str] = set()
            for zone_name in (
                "library",
                "hand",
                "battlefield",
                "graveyard",
                "exile",
                "commandZone",
            ):
                for card in player.get(zone_name, []):
                    instance_id = str(card.get("instanceId", ""))
                    definition = card.get("definition")
                    if (
                        instance_id
                        and instance_id not in seen_instances
                        and isinstance(definition, dict)
                    ):
                        seen_instances.add(instance_id)
                        definitions.append(definition)
            known_decks[str(player_id)] = definitions
            for definition in definitions:
                if bool(definition.get("isCommander")):
                    commanders.append(
                        {
                            "playerId": str(player_id),
                            "card": definition,
                        }
                    )
        self.known_decks_by_player_id = known_decks
        self.pregame_commanders = commanders

    @staticmethod
    def _observation_snapshot(state: dict[str, Any]) -> dict[str, Any]:
        return deepcopy(
            {
                key: value
                for key, value in state.items()
                if not str(key).startswith("_")
            }
        )

    def _decorate_observation(
        self,
        state: dict[str, Any],
        player_id: str | None,
    ) -> dict[str, Any]:
        if player_id is None:
            return state
        state["_knownDeck"] = self._known_deck(player_id)
        state["_pregameDeck"] = self._known_deck(player_id)
        state["_pregameCommanders"] = list(self.pregame_commanders)
        previous = self.previous_observations_by_player_id.get(player_id)
        if previous is not None:
            state["_previousObservation"] = previous
        self.previous_observations_by_player_id[player_id] = self._observation_snapshot(state)
        return state

    def _decision_context(self, decision: dict[str, Any]) -> dict[str, Any]:
        matchup = self.current_matchup
        free_mulligans = matchup.free_mulligans if matchup else 0
        maximum_mulligans = matchup.max_mulligans if matchup else None
        mulligans_taken: int | None = None
        if str(decision.get("kind", "")).casefold() in {
            "mulligan",
            "mulliganbottom",
        }:
            try:
                mulligans_taken = int(str(decision.get("id", "")).rsplit(":", 1)[-1])
            except ValueError:
                mulligans_taken = 0
        return {
            "id": decision.get("id"),
            "playerId": decision.get("playerId"),
            "kind": decision.get("kind"),
            "gameMode": matchup.game_mode if matchup else None,
            "mulliganEnabled": matchup.mulligan_enabled if matchup else False,
            "openingHandSize": (
                int(matchup.setup.get("openingHandSize", 7)) if matchup else 7
            ),
            "freeMulligans": free_mulligans,
            "maxMulligans": maximum_mulligans,
            "mulligansTaken": mulligans_taken,
            "freeMulligansRemaining": (
                max(0, free_mulligans - mulligans_taken)
                if mulligans_taken is not None
                else None
            ),
            "paidMulligansTaken": (
                max(0, mulligans_taken - free_mulligans)
                if mulligans_taken is not None
                else None
            ),
            "mulligansRemaining": (
                max(0, maximum_mulligans - mulligans_taken)
                if maximum_mulligans is not None and mulligans_taken is not None
                else None
            ),
        }

    def _to_step(self, view: dict[str, Any], reward: float = 0.0) -> DecisionStep:
        decision = view.get("decision")
        error = view.get("error")
        if error:
            self._remove_session()
            raise RuntimeError(f"Rust session failed: {error}")
        self._report_progress(view)
        state = view.get("state", {})
        matchup = self.current_matchup
        if (
            not self.objective_completed
            and matchup is not None
            and matchup.objective == "combo-trigger"
            and matchup.success_card_names
        ):
            observed_cards = [
                card
                for player in state.get("players", [])
                if isinstance(player, dict)
                for zone in (
                    "library",
                    "hand",
                    "battlefield",
                    "graveyard",
                    "exile",
                    "commandZone",
                )
                for card in player.get(zone, [])
                if isinstance(card, dict)
            ]
            stack_cards = [
                item.get("card")
                for item in state.get("stack", [])
                if isinstance(item, dict) and isinstance(item.get("card"), dict)
            ]
            observed_cards.extend(stack_cards)
            success_names = set(matchup.success_card_names)
            success_instance_ids = {
                str(card.get("instanceId", ""))
                for card in observed_cards
                if str((card.get("definition") or {}).get("name", card.get("name", "")))
                in success_names
            }
            zoned_success_names = {
                str((card.get("definition") or {}).get("name", card.get("name", "")))
                for player in state.get("players", [])
                if isinstance(player, dict)
                for zone in matchup.success_zones
                for card in player.get(zone, [])
                if isinstance(card, dict)
            }
            success_event = any(
                isinstance(event, dict)
                and str(event.get("cardInstanceId", "")) in success_instance_ids
                and str(event.get("kind", ""))
                in {"spellCast", "permanentEnteredBattlefield", "tokenCreated"}
                for event in state.get("events", [])
            )
            if zoned_success_names.intersection(success_names) or success_event:
                sequence_complete = (
                    not matchup.success_action_sequence
                    or self.objective_milestones_completed
                    == len(matchup.success_action_sequence)
                )
                if sequence_complete:
                    self.objective_completed = True
                    step = DecisionStep(dict(state), [], 1.0, True, self.learner_player_id)
                    self._remove_session()
                    return step
        terminal = decision is None
        if terminal:
            if view.get("matchState") is not None:
                state = dict(state)
                state["_matchState"] = view.get("matchState")
            outcome = state.get("outcome") or {}
            winners = set(state.get("winnerIds", []))
            if outcome.get("winner"):
                winners.add(outcome["winner"])
            losers = set(outcome.get("losers", []))
            reward = (
                1.0
                if self.learner_player_id in winners
                else (-1.0 if self.learner_player_id in losers else 0.0)
            )
            if (
                reward == 0.0
                and state.get("status") == "turnLimitReached"
                and matchup
                and matchup.anchor_deadline_round
            ):
                reward = -1.0
            if reward > 0 and matchup and matchup.objective in {"fast-win", "combo-win"}:
                turn_number = int(state.get("turnNumber", 0) or 0)
                round_number = max(1, (turn_number + 1) // 2)
                target_round = matchup.target_round or matchup.anchor_deadline_round
                if target_round:
                    reward += max(0.0, (target_round - round_number) / target_round)
            step = DecisionStep(view.get("state", {}), [], reward, True)
            self._remove_session()
            return step
        if decision.get("playerId") != self.learner_player_id:
            raise RuntimeError(
                "Rust returned a decision for a player not controlled by this learner; "
                "configure every other player as ai-random or a remote opponent"
            )
        state = dict(view.get("state", {}))
        state["_decisionContext"] = self._decision_context(decision)
        state = self._decorate_observation(state, decision.get("playerId"))
        return DecisionStep(
            state,
            self._policy_actions(decision),
            reward,
            False,
            decision.get("playerId"),
        )

    def reset(self, matchup_id: str, seed: int, seat_swap: bool) -> DecisionStep:
        self._remove_session()
        self.known_decks_by_player_id = {}
        self.pregame_commanders = []
        self.previous_observations_by_player_id = {}
        self.objective_completed = False
        self.objective_action_labels = []
        self.objective_milestones_completed = 0
        self.opening_hand_selected_card_names = []
        self.opening_hand_roles_available = 0
        self.objective_damage_progress = 0.0
        self.sideboard_cards_selected = 0
        self.sideboard_target_cards_selected = 0
        self.sideboard_cut_cards_selected = 0
        matchup = self.matchups[matchup_id]
        self.sideboard_target_cards_available = sum(
            1
            for player in matchup.setup.get("players", [])
            if str(player.get("id", "")) == matchup.learner_player_id
            for card in player.get("cards", [])
            if isinstance(card, dict)
            and bool(card.get("isSideboard"))
            and str(card.get("name", "")) in set(matchup.sideboard_target_card_names)
        )
        self.sideboard_cut_cards_expected = min(
            self.sideboard_target_cards_available,
            sum(
                1
                for player in matchup.setup.get("players", [])
                if str(player.get("id", "")) == matchup.learner_player_id
                for card in player.get("cards", [])
                if isinstance(card, dict)
                and not bool(card.get("isSideboard"))
                and str(card.get("name", "")) in set(matchup.sideboard_cut_card_names)
            ),
        )
        self.current_matchup = matchup
        setup = dict(matchup.setup)
        if seat_swap:
            players = list(setup.get("players", []))
            setup["players"] = list(reversed(players))
        deck_session_ids = list(matchup.deck_session_ids)
        if seat_swap:
            deck_session_ids.reverse()
        self.learner_player_id = matchup.learner_player_id
        player_ids = [player["id"] for player in setup.get("players", [])]
        analytics_deck_sessions = {
            player_id: deck_session_ids[index]
            for index, player_id in enumerate(player_ids)
            if index < len(deck_session_ids) and deck_session_ids[index]
        }
        analytics_pilots = self.analytics_pilot_override or {
            player_id: (
                self.learner_pilot_id
                if player_id == self.learner_player_id
                else "ai-random"
            )
            for player_id in player_ids
        }
        response = self.client.post(
            "/game/sessions",
            json={
                "setup": setup,
                "seed": seed,
                "gameMode": matchup.game_mode,
                "maxTurns": matchup.max_turns,
                "mulliganEnabled": matchup.mulligan_enabled,
                "freeMulligans": matchup.free_mulligans,
                "maxMulligans": matchup.max_mulligans,
                "waitTimeoutMs": self.wait_timeout_ms,
                "humanPlayerIds": [self.learner_player_id],
                "combatDeclarationRevisionPlayerIds": [],
                "holdPriorityPlayerIds": [],
                "analyticsContextId": self.analytics_context_id,
                "analyticsPilotByPlayerId": analytics_pilots,
                "analyticsDeckSessionByPlayerId": analytics_deck_sessions,
                "punchingBagPlayerIds": list(matchup.punching_bag_player_ids),
                "openingHandSelectionPoolSizeByPlayerId": (
                    {
                        self.learner_player_id: matchup.anchor_opening_hand_pool_size,
                    }
                    if matchup.anchor_opening_hand_pool_size is not None
                    else {}
                ),
                "fixedOpeningHandDefinitionIdsByPlayerId": (
                    {
                        self.learner_player_id: list(
                            matchup.fixed_opening_hand_definition_ids
                        )
                    }
                    if matchup.fixed_opening_hand_definition_ids
                    else {}
                ),
                "trainingAnchorDeadlineRoundByPlayerId": {
                    player_id: matchup.anchor_deadline_round
                    for player_id in matchup.training_anchor_player_ids
                    if matchup.anchor_deadline_round is not None
                },
                "startWithSideboarding": matchup.start_with_sideboarding,
            },
        )
        response.raise_for_status()
        self.current_view = response.json()
        self._start_replay(self.current_view)
        self._capture_known_decks(self.current_view)
        self.session_id = self.current_view["sessionId"]
        step = self._to_step(self.current_view)
        self._initialize_opening_hand_roles(step)
        return step

    def step(self, action_index: int) -> DecisionStep:
        if self.current_view is None or self.session_id is None:
            raise RuntimeError("environment must be reset before step")
        decision = self.current_view["decision"]
        options = self._policy_actions(decision)
        if action_index < 0 or action_index >= len(options):
            raise IndexError("selected action index is outside the current legal action list")
        try:
            selected_action = options[action_index]
            self._record_replay_action(selected_action)
            submission = {
                "revision": self.current_view["revision"],
                "decisionId": decision["id"],
                "actionId": selected_action.get(
                    "_engineActionId",
                    selected_action["id"],
                ),
            }
            if "_numberValue" in selected_action:
                submission["numberValue"] = selected_action["_numberValue"]
            if "_cardInstanceIds" in selected_action:
                submission["cardInstanceIds"] = selected_action["_cardInstanceIds"]
            choice = decision.get("choice")
            if isinstance(choice, dict) and choice.get("kind") == "cardNameSelection":
                decision_id = str(choice.get("decisionId", ""))
                selected_value = (selected_action.get("decisions") or {}).get(decision_id)
                if isinstance(selected_value, list) and selected_value:
                    selected_value = selected_value[0]
                if isinstance(selected_value, str) and selected_value.strip():
                    submission["cardName"] = selected_value.strip()
            response = self.client.post(
                f"/game/sessions/{self.session_id}/actions",
                json=submission,
            )
        except httpx.HTTPError:
            self._discard_failed_session()
            raise
        if response.is_error:
            selected_action = options[action_index]
            error_text = response.text
            self._discard_failed_session()
            raise RuntimeError(
                "Rust rejected a published legal action "
                f"(status={response.status_code}, revision={self.current_view['revision']}, "
                f"decision={decision.get('id')}, decisionKind={decision.get('kind')}, "
                f"action={selected_action.get('id')}, actionKind={selected_action.get('kind')}): "
                f"{error_text}"
            )
        previous_milestones = self.objective_milestones_completed
        self.objective_action_labels.append(str(selected_action.get("label", "")))
        if str(decision.get("kind", "")) == "openingHandSelection":
            selected_card_name = self._opening_hand_card_name(selected_action)
            if selected_card_name:
                self.opening_hand_selected_card_names.append(selected_card_name)
        self.objective_milestones_completed = (
            self._opening_hand_role_progress()
            if self.current_matchup and self.current_matchup.opening_hand_target_roles
            else self._objective_sequence_progress()
        )
        milestone_reward = 0.1 * max(
            0,
            self.objective_milestones_completed - previous_milestones,
        )
        sideboard_reward = 0.0
        if str(decision.get("kind", "")) == "sideboarding":
            previous_sideboard_count = self.sideboard_cards_selected
            self.sideboard_cards_selected = max(
                self.sideboard_cards_selected,
                self._sideboard_cards_brought_in(selected_action),
            )
            if self.sideboard_cards_selected > previous_sideboard_count:
                if self.current_matchup and self.current_matchup.sideboard_target_card_names:
                    self.sideboard_target_cards_selected = max(
                        self.sideboard_target_cards_selected,
                        self._sideboard_target_cards_brought_in(selected_action),
                    )
                    self.sideboard_cut_cards_selected = max(
                        self.sideboard_cut_cards_selected,
                        self._sideboard_cut_cards_taken_out(selected_action),
                    )
                    coverage = self.sideboard_target_cards_selected / max(
                        1, self.sideboard_target_cards_available
                    )
                    if self.sideboard_cut_cards_expected:
                        coverage = 0.5 * (
                            coverage
                            + self.sideboard_cut_cards_selected
                            / self.sideboard_cut_cards_expected
                        )
                    sideboard_reward = 0.3 * coverage
                else:
                    sideboard_reward = min(0.2, self.sideboard_cards_selected * 0.04)
        next_view = response.json()
        session_error = next_view.get("error")
        if session_error:
            action_context = {
                "decision": decision.get("id"),
                "decisionKind": decision.get("kind"),
                "action": selected_action.get("id"),
                "engineAction": selected_action.get("_engineActionId"),
                "actionKind": selected_action.get("kind"),
                "actionLabel": selected_action.get("label"),
                "cardInstanceId": selected_action.get("cardInstanceId"),
            }
            self._remove_session()
            raise RuntimeError(
                "Rust session failed after published action "
                f"{json.dumps(action_context, sort_keys=True)}: {session_error}"
            )
        self.current_view = next_view
        previous_damage_progress = self.objective_damage_progress
        self.objective_damage_progress = max(
            self.objective_damage_progress,
            self._damage_progress(self.current_view),
        )
        damage_reward = 0.2 * max(
            0.0,
            self.objective_damage_progress - previous_damage_progress,
        )
        self._record_replay_view(self.current_view)
        return self._to_step(
            self.current_view,
            reward=milestone_reward + sideboard_reward + damage_reward,
        )


class RustSelfPlayEnvironment(RustSessionEnvironment):
    """Rust session adapter where the shared learner controls every player."""

    def __init__(
        self,
        base_url: str,
        matchups: dict[str, Matchup],
        timeout_seconds: float = 30.0,
        multiplayer_reward_mode: str = "winnerLoser",
        learner_pilot_id: str = "ia-in-training",
        no_winner_reward: float = 0.0,
        legacy_game_win_reward: float = 0.25,
        legacy_match_win_reward: float = 1.0,
        scale_rewards_by_plackett_luce: bool = False,
    ) -> None:
        super().__init__(base_url, matchups, timeout_seconds, learner_pilot_id)
        if multiplayer_reward_mode not in {
            "winnerLoser",
            "centeredWinner",
            "plackettLuce",
            "alphaStarTwoPlayer",
        }:
            raise ValueError("unsupported multiplayer reward mode")
        self.multiplayer_reward_mode = multiplayer_reward_mode
        self.no_winner_reward = float(no_winner_reward)
        self.plackett_luce_ratings_by_player_id: dict[str, PlackettLuceRating] = {}
        self.participant_by_player_id: dict[str, str] = {}
        self.plackett_luce_participant_by_player_id: dict[str, str] = {}
        self.legacy_game_win_reward = float(legacy_game_win_reward)
        self.legacy_match_win_reward = float(legacy_match_win_reward)
        self.scale_rewards_by_plackett_luce = bool(scale_rewards_by_plackett_luce)
        self._match_wins_by_player_id: dict[str, int] = {}
        self._match_reward_emitted = False

    def _two_player_result_rewards(
        self,
        player_ids: list[str],
        winner_id: str,
        reward: float,
    ) -> dict[str, float]:
        loser_id = next(player_id for player_id in player_ids if player_id != winner_id)
        if not self.scale_rewards_by_plackett_luce:
            return {winner_id: reward, loser_id: -reward}

        # Ratings are intentionally keyed by seats here. A mirror can map both
        # seats to the same leaderboard entry, whose aggregate rating update is
        # zero, while its two trajectories still need opposite learning signals.
        seat_ratings = {
            player_id: self.plackett_luce_ratings_by_player_id.get(
                player_id,
                PlackettLuceRating(),
            )
            for player_id in player_ids
        }
        gradients = rank_gradient_rewards(
            [winner_id, loser_id],
            seat_ratings,
        )
        return {player_id: reward * gradients[player_id] for player_id in player_ids}

    def _legacy_boundary_rewards(
        self,
        view: dict[str, Any],
        player_ids: list[str],
    ) -> dict[str, float]:
        if self.multiplayer_reward_mode != "alphaStarTwoPlayer":
            return {}
        if len(player_ids) != 2:
            raise RuntimeError("alphaStarTwoPlayer requires exactly two players")
        match_state = view.get("matchState") or {}
        wins = match_state.get("winsByPlayerId") or {}
        rewards = {player_id: 0.0 for player_id in player_ids}
        for winner_id in player_ids:
            won_games = max(
                0,
                int(wins.get(winner_id, 0))
                - int(self._match_wins_by_player_id.get(winner_id, 0)),
            )
            if not won_games:
                continue
            delta = self.legacy_game_win_reward * won_games
            for player_id, scaled_reward in self._two_player_result_rewards(
                player_ids,
                winner_id,
                delta,
            ).items():
                rewards[player_id] += scaled_reward
        self._match_wins_by_player_id = {
            player_id: int(wins.get(player_id, 0)) for player_id in player_ids
        }
        match_winner = match_state.get("winnerPlayerId")
        if (
            match_state.get("phase") == "complete"
            and match_winner in player_ids
            and not self._match_reward_emitted
        ):
            for player_id, scaled_reward in self._two_player_result_rewards(
                player_ids,
                match_winner,
                self.legacy_match_win_reward,
            ).items():
                rewards[player_id] += scaled_reward
            self._match_reward_emitted = True
        return {player_id: reward for player_id, reward in rewards.items() if reward}

    def _to_step(self, view: dict[str, Any], reward: float = 0.0) -> DecisionStep:
        decision = view.get("decision")
        error = view.get("error")
        self._report_progress(view)
        if error:
            self._remove_session()
            raise RuntimeError(f"Rust session failed: {error}")
        player_ids = [
            player.get("id")
            for player in (self.current_matchup.setup.get("players", []) if self.current_matchup else [])
            if player.get("id")
        ]
        boundary_rewards = self._legacy_boundary_rewards(view, player_ids)
        if decision is not None:
            state = dict(view.get("state", {}))
            state["_decisionContext"] = self._decision_context(decision)
            state = self._decorate_observation(state, decision.get("playerId"))
            return DecisionStep(
                state,
                expand_policy_actions(decision),
                0.0,
                False,
                decision.get("playerId"),
                rewards_by_player=boundary_rewards or None,
            )

        state = view.get("state", {})
        if view.get("matchState") is not None:
            state = dict(state)
            state["_matchState"] = view.get("matchState")
        outcome = state.get("outcome") or {}
        winner = outcome.get("winner")
        losers = set(outcome.get("losers", []))
        loser_reward = -1.0
        if self.multiplayer_reward_mode == "centeredWinner" and losers:
            loser_reward = -1.0 / len(losers)
        if self.multiplayer_reward_mode == "alphaStarTwoPlayer":
            rewards_by_player = boundary_rewards
        elif self.multiplayer_reward_mode == "plackettLuce":
            reward_participant_by_player = (
                self.plackett_luce_participant_by_player_id
                or self.participant_by_player_id
            )
            order = ordered_finishers(state, player_ids)
            # A deterministic anchor deadline is a training loss when the real
            # deck has not won before the configured round.
            if (
                winner is None
                and state.get("status") == "turnLimitReached"
                and self.current_matchup
                and self.current_matchup.anchor_deadline_round
            ):
                anchors = list(self.current_matchup.training_anchor_player_ids)
                order = anchors + [player_id for player_id in order if player_id not in anchors]
            ordered_participants = list(
                dict.fromkeys(
                    reward_participant_by_player.get(player_id, player_id)
                    for player_id in order
                )
            )
            if len(ordered_participants) > 1:
                ratings_by_participant = {
                    participant: next(
                        (
                            self.plackett_luce_ratings_by_player_id.get(
                                player_id,
                                PlackettLuceRating(),
                            )
                            for player_id, candidate in reward_participant_by_player.items()
                            if candidate == participant
                        ),
                        PlackettLuceRating(),
                    )
                    for participant in ordered_participants
                }
                participant_rewards = rank_gradient_rewards(
                    ordered_participants,
                    ratings_by_participant,
                )
                rewards_by_player = {
                    player_id: participant_rewards.get(
                        reward_participant_by_player.get(player_id, player_id),
                        0.0,
                    )
                    for player_id in player_ids
                }
            else:
                ratings = {
                    player_id: self.plackett_luce_ratings_by_player_id.get(
                        player_id,
                        PlackettLuceRating(),
                    )
                    for player_id in player_ids
                }
                rewards_by_player = rank_gradient_rewards(order, ratings)
        else:
            rewards_by_player = {
                player_id: (
                    1.0
                    if player_id == winner
                    else (
                        loser_reward
                        if player_id in losers
                        else (self.no_winner_reward if winner is None else 0.0)
                    )
                )
                for player_id in player_ids
            }
        step = DecisionStep(
            state,
            [],
            0.0,
            True,
            rewards_by_player=rewards_by_player,
        )
        self._remove_session()
        return step

    def reset(self, matchup_id: str, seed: int, seat_swap: bool) -> DecisionStep:
        self._remove_session()
        self._match_wins_by_player_id = {}
        self._match_reward_emitted = False
        self.known_decks_by_player_id = {}
        self.pregame_commanders = []
        self.previous_observations_by_player_id = {}
        self.objective_completed = False
        self.objective_action_labels = []
        self.objective_milestones_completed = 0
        self.opening_hand_selected_card_names = []
        self.opening_hand_roles_available = 0
        self.objective_damage_progress = 0.0
        self.sideboard_cards_selected = 0
        matchup = self.matchups[matchup_id]
        self.sideboard_target_cards_selected = 0
        self.sideboard_cut_cards_selected = 0
        self.sideboard_target_cards_available = sum(
            1
            for player in matchup.setup.get("players", [])
            if str(player.get("id", "")) == matchup.learner_player_id
            for card in player.get("cards", [])
            if isinstance(card, dict)
            and bool(card.get("isSideboard"))
            and str(card.get("name", "")) in set(matchup.sideboard_target_card_names)
        )
        self.sideboard_cut_cards_expected = min(
            self.sideboard_target_cards_available,
            sum(
                1
                for player in matchup.setup.get("players", [])
                if str(player.get("id", "")) == matchup.learner_player_id
                for card in player.get("cards", [])
                if isinstance(card, dict)
                and not bool(card.get("isSideboard"))
                and str(card.get("name", "")) in set(matchup.sideboard_cut_card_names)
            ),
        )
        self.current_matchup = matchup
        setup = dict(matchup.setup)
        players = list(setup.get("players", []))
        if seat_swap:
            players.reverse()
            setup["players"] = players
        deck_session_ids = list(matchup.deck_session_ids)
        if seat_swap:
            deck_session_ids.reverse()
        player_ids = [player["id"] for player in players]
        analytics_deck_sessions = {
            player_id: deck_session_ids[index]
            for index, player_id in enumerate(player_ids)
            if index < len(deck_session_ids) and deck_session_ids[index]
        }
        analytics_pilots = self.analytics_pilot_override or {
            player_id: self.learner_pilot_id for player_id in player_ids
        }
        response = self.client.post(
            "/game/sessions",
            json={
                "setup": setup,
                "seed": seed,
                "gameMode": matchup.game_mode,
                "maxTurns": matchup.max_turns,
                "mulliganEnabled": matchup.mulligan_enabled,
                "freeMulligans": matchup.free_mulligans,
                "maxMulligans": matchup.max_mulligans,
                "waitTimeoutMs": self.wait_timeout_ms,
                "humanPlayerIds": player_ids,
                "combatDeclarationRevisionPlayerIds": [],
                "holdPriorityPlayerIds": [],
                "analyticsContextId": self.analytics_context_id,
                "analyticsPilotByPlayerId": analytics_pilots,
                "analyticsDeckSessionByPlayerId": analytics_deck_sessions,
                "punchingBagPlayerIds": list(matchup.punching_bag_player_ids),
                "openingHandSelectionPoolSizeByPlayerId": (
                    {
                        matchup.learner_player_id: matchup.anchor_opening_hand_pool_size,
                    }
                    if matchup.anchor_opening_hand_pool_size is not None
                    else {}
                ),
                "fixedOpeningHandDefinitionIdsByPlayerId": (
                    {
                        matchup.learner_player_id: list(
                            matchup.fixed_opening_hand_definition_ids
                        )
                    }
                    if matchup.fixed_opening_hand_definition_ids
                    else {}
                ),
                "trainingAnchorDeadlineRoundByPlayerId": {
                    player_id: matchup.anchor_deadline_round
                    for player_id in matchup.training_anchor_player_ids
                    if matchup.anchor_deadline_round is not None
                },
                "startWithSideboarding": matchup.start_with_sideboarding,
            },
        )
        response.raise_for_status()
        self.current_view = response.json()
        self._start_replay(self.current_view)
        self._capture_known_decks(self.current_view)
        self.session_id = self.current_view["sessionId"]
        step = self._to_step(self.current_view)
        self._initialize_opening_hand_roles(step)
        return step


class TinySelfPlayEnvironment:
    """Fast deterministic environment used to prove that PPO updates end-to-end."""

    def __init__(self, horizon: int = 8) -> None:
        self.horizon = horizon
        self.turn = 0
        self.target = 0
        self.score = 0

    def reset(self, matchup_id: str, seed: int, seat_swap: bool) -> DecisionStep:
        randomizer = random.Random(f"{matchup_id}:{seed}:{seat_swap}")
        self.turn = 0
        self.target = randomizer.randrange(2)
        self.score = 0
        return self._decision()

    def _decision(self) -> DecisionStep:
        done = self.turn >= self.horizon
        reward = float(self.score) / self.horizon if done else 0.0
        actions = [] if done else [{"id": "left", "kind": 0}, {"id": "right", "kind": 1}]
        state = {"turn": self.turn, "target": self.target, "score": self.score}
        return DecisionStep(state, actions, reward, done)

    def step(self, action_index: int) -> DecisionStep:
        self.score += 1 if action_index == self.target else -1
        self.turn += 1
        self.target = 1 - self.target
        return self._decision()
