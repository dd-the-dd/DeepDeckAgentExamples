from __future__ import annotations

import json
import math
import re
import secrets
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from .resources import find_model_run

REPLAY_ID = re.compile(r"^[A-Za-z0-9._-]{1,120}$")
REPLAY_LEASE_SECONDS = 90


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    pending.replace(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    records: list[dict[str, Any]] = []
    for line in lines:
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict):
            records.append(value)
    return records


def _bounded_int(
    payload: dict[str, Any], key: str, default: int, minimum: int, maximum: int
) -> int:
    try:
        value = int(payload.get(key, default))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{key} must be an integer") from error
    if value < minimum or value > maximum:
        raise ValueError(f"{key} must be between {minimum} and {maximum}")
    return value


def _yaml_config(run: Path) -> dict[str, Any]:
    try:
        import yaml

        value = yaml.safe_load((run / "training-config.yaml").read_text(encoding="utf-8"))
    except (ImportError, OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _nested_config_value(config: dict[str, Any], dotted_key: str, default: Any) -> Any:
    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def _set_nested_config_value(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    current = config
    for part in parts[:-1]:
        child = current.get(part)
        if not isinstance(child, dict):
            child = {}
            current[part] = child
        current = child
    current[parts[-1]] = value


def _write_yaml_config(run: Path, config: dict[str, Any]) -> None:
    import yaml

    path = run / "training-config.yaml"
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    pending.replace(path)


def load_training_settings(root: Path, model_id: str) -> dict[str, Any]:
    run, metadata = find_model_run(root, model_id)
    saved = _read_json(run / "training-platform.json", {})
    config = _yaml_config(run)
    architecture = str(metadata.get("architecture", "")).casefold()
    training = config.get("training", {}) if isinstance(config.get("training"), dict) else {}
    rl = config.get("rl", {}) if isinstance(config.get("rl"), dict) else {}
    evaluation = (
        config.get("evaluation", {}) if isinstance(config.get("evaluation"), dict) else {}
    )
    v13_engine_supported = bool(config.get("deckCatalog")) and rl.get("environment") == "engine"
    stages = saved.get("stages", {}) if isinstance(saved.get("stages"), dict) else {}
    targets = saved.get("targets", {}) if isinstance(saved.get("targets"), dict) else {}
    raw_curriculum = (
        config.get("trainingCurriculum", {})
        if isinstance(config.get("trainingCurriculum"), dict)
        else {}
    )
    saved_curriculum = (
        saved.get("curriculum", {}) if isinstance(saved.get("curriculum"), dict) else {}
    )
    return {
        "schemaVersion": "training-platform/v1",
        "modelId": model_id,
        "architecture": architecture,
        "savedGameLimit": int(saved.get("savedGameLimit", config.get("savedGameLimit", 20))),
        "checkpointEvery": int(
            saved.get(
                "checkpointEvery",
                rl.get(
                    "checkpoint_every",
                    training.get("checkpoint_every", config.get("checkpointEvery", 20)),
                ),
            )
        ),
        "evaluationEvery": int(
            saved.get(
                "evaluationEvery",
                config.get("modelEvaluationEvery", config.get("evaluationEvery", 100)),
            )
        ),
        "evaluationGamesPerScenario": int(
            saved.get("evaluationGamesPerScenario", evaluation.get("gamesPerScenario", 1))
        ),
        "stages": {
            "worldModel": bool(stages.get("worldModel", architecture in {"v13", "v15"})),
            "reinforcementLearning": bool(
                stages.get("reinforcementLearning", architecture != "v15")
            ),
            "engineEvaluation": bool(
                stages.get(
                    "engineEvaluation",
                    (
                        int(rl.get("evaluation_every_updates", 0) or 0) > 0
                        if architecture == "v13"
                        else config.get("modelEvaluationEnabled", False)
                    ),
                )
            ),
        },
        "targets": {
            "worldModelSteps": int(targets.get("worldModelSteps", training.get("steps", 0))),
            "reinforcementLearningEpisodes": int(
                targets.get("reinforcementLearningEpisodes", rl.get("episodes", 0))
            ),
        },
        "curriculum": {
            "enabled": bool(saved_curriculum.get("enabled", raw_curriculum.get("enabled", False))),
            "adaptive": bool(
                saved_curriculum.get("adaptive", raw_curriculum.get("adaptive", True))
            ),
            "scenarioIds": list(
                saved_curriculum.get("scenarioIds", raw_curriculum.get("scenarioIds", []))
            ),
        },
        "engineEvaluationSupported": architecture in {"v11", "v12"} or v13_engine_supported,
        "requiresRestart": True,
    }


def save_training_settings(root: Path, model_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    run, metadata = find_model_run(root, model_id)
    architecture = str(metadata.get("architecture", "")).casefold()
    current = load_training_settings(root, model_id)
    raw_stages = payload.get("stages", current["stages"])
    raw_targets = payload.get("targets", current["targets"])
    raw_curriculum = payload.get("curriculum", current["curriculum"])
    if (
        not isinstance(raw_stages, dict)
        or not isinstance(raw_targets, dict)
        or not isinstance(raw_curriculum, dict)
    ):
        raise ValueError("stages, targets, and curriculum must be objects")
    from oracle_ai.training.curriculum import BUILT_IN_CURRICULUM

    known_scenarios = {scenario.id for scenario in BUILT_IN_CURRICULUM}
    scenario_ids = [str(value) for value in raw_curriculum.get("scenarioIds", [])]
    unknown_scenarios = sorted(set(scenario_ids) - known_scenarios)
    if unknown_scenarios:
        raise ValueError("Unknown curriculum scenarios: " + ", ".join(unknown_scenarios))
    curriculum = {
        "enabled": bool(raw_curriculum.get("enabled", False)),
        "adaptive": bool(raw_curriculum.get("adaptive", True)),
        "scenarioIds": list(dict.fromkeys(scenario_ids)),
    }
    if curriculum["enabled"] and architecture != "v13":
        raise ValueError("Focused curriculum training is currently available for V13 agents.")
    stages = {
        "worldModel": bool(raw_stages.get("worldModel", current["stages"]["worldModel"])),
        "reinforcementLearning": bool(
            raw_stages.get("reinforcementLearning", current["stages"]["reinforcementLearning"])
        ),
        "engineEvaluation": bool(
            raw_stages.get("engineEvaluation", current["stages"]["engineEvaluation"])
        ),
    }
    engine_supported = current["engineEvaluationSupported"]
    if stages["engineEvaluation"] and not engine_supported:
        raise ValueError(
            "Configure an authoritative Engine deck catalog before enabling Engine evaluation."
        )
    targets = {
        "worldModelSteps": _bounded_int(
            raw_targets, "worldModelSteps", current["targets"]["worldModelSteps"], 0, 10_000_000
        ),
        "reinforcementLearningEpisodes": _bounded_int(
            raw_targets,
            "reinforcementLearningEpisodes",
            current["targets"]["reinforcementLearningEpisodes"],
            0,
            100_000_000,
        ),
    }
    if architecture in {"v13", "v15"} and stages["worldModel"] and targets["worldModelSteps"] < 1:
        raise ValueError("worldModelSteps must be positive while the world-model stage is enabled")
    if architecture == "v15" and stages["reinforcementLearning"]:
        raise ValueError("V15 plan-level reinforcement learning is not implemented yet.")
    if (
        architecture == "v13"
        and stages["reinforcementLearning"]
        and targets["reinforcementLearningEpisodes"] < 1
    ):
        raise ValueError(
            "reinforcementLearningEpisodes must be positive while reinforcement learning is enabled"
        )
    world_checkpoints = list((run / "checkpoints").glob("step-*/world-model.pt"))
    if (
        architecture == "v13"
        and stages["reinforcementLearning"]
        and not stages["worldModel"]
        and not world_checkpoints
    ):
        raise ValueError(
            "A V13 world-model checkpoint is required before reinforcement learning can run alone."
        )
    value = {
        "schemaVersion": "training-platform/v1",
        "modelId": model_id,
        "savedGameLimit": _bounded_int(
            payload, "savedGameLimit", current["savedGameLimit"], 0, 500
        ),
        "checkpointEvery": _bounded_int(
            payload, "checkpointEvery", current["checkpointEvery"], 1, 1_000_000
        ),
        "evaluationEvery": _bounded_int(
            payload, "evaluationEvery", current["evaluationEvery"], 1, 1_000_000
        ),
        "evaluationGamesPerScenario": _bounded_int(
            payload,
            "evaluationGamesPerScenario",
            current["evaluationGamesPerScenario"],
            1,
            100,
        ),
        "stages": stages,
        "targets": targets,
        "curriculum": curriculum,
    }
    pending = run / "training-platform.json.pending"
    pending.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    pending.replace(run / "training-platform.json")

    config = _yaml_config(run)
    if config:
        if architecture == "v13":
            training = config.setdefault("training", {})
            rl = config.setdefault("rl", {})
            training["steps"] = targets["worldModelSteps"] if stages["worldModel"] else 0
            training["checkpoint_every"] = value["checkpointEvery"]
            rl["episodes"] = (
                targets["reinforcementLearningEpisodes"]
                if stages["reinforcementLearning"]
                else 0
            )
            rl["checkpoint_every"] = value["checkpointEvery"]
            rl["evaluation_every_updates"] = (
                value["evaluationEvery"] if stages["engineEvaluation"] else 0
            )
            rl["evaluation_games"] = value["evaluationGamesPerScenario"]
            config["trainingCurriculum"] = curriculum
        else:
            config["checkpointEvery"] = value["checkpointEvery"]
            config["modelEvaluationEvery"] = value["evaluationEvery"]
            config["modelEvaluationEnabled"] = stages["engineEvaluation"]
            evaluation = config.setdefault("evaluation", {})
            evaluation["gamesPerScenario"] = value["evaluationGamesPerScenario"]
        config["savedGameLimit"] = value["savedGameLimit"]
        try:
            import yaml

            config_pending = run / "training-config.yaml.pending"
            config_pending.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            config_pending.replace(run / "training-config.yaml")
        except ImportError as error:
            raise ValueError("PyYAML is required to update training settings") from error
    return load_training_settings(root, model_id)


def _wilson_interval(wins: int, games: int) -> tuple[float | None, float | None]:
    if games <= 0:
        return None, None
    z = 1.959963984540054
    proportion = wins / games
    denominator = 1 + z * z / games
    centre = proportion + z * z / (2 * games)
    margin = z * math.sqrt(
        proportion * (1 - proportion) / games + z * z / (4 * games * games)
    )
    return (centre - margin) / denominator, (centre + margin) / denominator


def training_evidence(root: Path, model_id: str) -> dict[str, Any]:
    run, metadata = find_model_run(root, model_id)
    architecture = str(metadata.get("architecture", "")).casefold()
    evaluations = _read_jsonl(
        run
        / (
            "v13-engine-evaluations.jsonl"
            if architecture == "v13"
            else "evaluations.jsonl"
        )
    )
    promotions = _read_jsonl(run / "promotions.jsonl")
    from oracle_ai.training.curriculum import BUILT_IN_CURRICULUM

    current_scenario_versions = {
        scenario.id: scenario.revision for scenario in BUILT_IN_CURRICULUM
    }
    curriculum_evaluations = _read_jsonl(run / "v13-curriculum-evaluations.jsonl")
    curriculum_evaluation_results = [
        scenario
        for evaluation in curriculum_evaluations
        if isinstance(evaluation, dict)
        for scenario in evaluation.get("scenarios", [])
        if isinstance(scenario, dict)
        and int(scenario.get("scenarioVersion", 1) or 1)
        == current_scenario_versions.get(str(scenario.get("scenarioId")), 1)
    ]
    curriculum_games = [
        record
        for record in _read_jsonl(run / "v13-engine-games.jsonl")
        if isinstance(record, dict) and record.get("scenarioId")
        and int(record.get("scenarioVersion", 1) or 1)
        == current_scenario_versions.get(str(record.get("scenarioId")), 1)
    ]
    games = [
        game
        for evaluation in evaluations
        for game in evaluation.get("games", [])
        if isinstance(game, dict)
    ]
    wins = sum(game.get("result") == "candidateWin" for game in games)
    losses = sum(
        game.get("result") in {"championWin", "baselineWin"} for game in games
    )
    draws = sum(game.get("result") == "draw" for game in games)
    lower, upper = _wilson_interval(wins, len(games))
    trend: list[dict[str, Any]] = []
    for evaluation in evaluations:
        summary = evaluation.get("summary", {})
        completed = int(summary.get("completedGames", 0) or 0)
        period_wins = int(summary.get("candidateWins", 0) or 0)
        period_lower, period_upper = _wilson_interval(period_wins, completed)
        trend.append(
            {
                "period": int(evaluation.get("period", len(trend) + 1) or len(trend) + 1),
                "trainingStep": int(evaluation.get("candidateTrainingStep", 0) or 0),
                "opponent": evaluation.get("opponentVersion"),
                "games": completed,
                "winRate": period_wins / completed if completed else None,
                "lower95": period_lower,
                "upper95": period_upper,
                "promoted": evaluation.get("promotion") is not None,
            }
        )
    deck_totals: dict[str, dict[str, int]] = defaultdict(
        lambda: {"games": 0, "wins": 0, "losses": 0, "draws": 0}
    )
    for game in games:
        deck = str(game.get("candidateDeck") or "Unknown deck")
        deck_totals[deck]["games"] += 1
        result = str(game.get("result", ""))
        if result == "candidateWin":
            deck_totals[deck]["wins"] += 1
        elif result in {"championWin", "baselineWin"}:
            deck_totals[deck]["losses"] += 1
        else:
            deck_totals[deck]["draws"] += 1
    by_deck = []
    for deck, deck_total in sorted(deck_totals.items()):
        deck_lower, deck_upper = _wilson_interval(deck_total["wins"], deck_total["games"])
        by_deck.append(
            {
                "deck": deck,
                **deck_total,
                "winRate": (
                    deck_total["wins"] / deck_total["games"]
                    if deck_total["games"]
                    else None
                ),
                "lower95": deck_lower,
                "upper95": deck_upper,
            }
        )
    curriculum_totals: dict[str, dict[str, Any]] = {}
    for record in curriculum_games:
        scenario_id = str(record["scenarioId"])
        curriculum_total = curriculum_totals.setdefault(
            scenario_id,
            {
                "scenarioId": scenario_id,
                "family": str(record.get("scenarioFamily") or "other"),
                "scenarioVersion": int(record.get("scenarioVersion", 1) or 1),
                "objective": str(record.get("objective") or ""),
                "difficulty": record.get("difficulty"),
                "games": 0,
                "successes": 0,
                "roundTotal": 0.0,
                "roundSamples": 0,
                "decisionTotal": 0,
                "milestoneProgressTotal": 0.0,
                "milestoneSamples": 0,
                "damageProgressTotal": 0.0,
                "damageProgressSamples": 0,
                "sideboardCardsTotal": 0,
                "sideboardSamples": 0,
                "sideboardTargetSelectedTotal": 0,
                "sideboardTargetAvailableTotal": 0,
                "sideboardCutSelectedTotal": 0,
                "sideboardCutExpectedTotal": 0,
                "recent": [],
                "adaptiveWeight": None,
            },
        )
        learner_id = str(record.get("learnerPlayerId") or "player-1")
        rewards = record.get("rewardsByPlayer", {})
        success = bool(record.get("objectiveCompleted")) or (
            isinstance(rewards, dict) and float(rewards.get(learner_id, 0) or 0) > 0
        )
        curriculum_total["games"] += 1
        curriculum_total["successes"] += int(success)
        if isinstance(record.get("turnNumber"), (int, float)):
            curriculum_total["roundTotal"] += float(record["turnNumber"])
            curriculum_total["roundSamples"] += 1
        curriculum_total["decisionTotal"] += int(record.get("decisions", 0) or 0)
        milestone_total = int(record.get("objectiveMilestonesTotal", 0) or 0)
        if milestone_total > 0:
            curriculum_total["milestoneProgressTotal"] += min(
                1.0,
                int(record.get("objectiveMilestonesCompleted", 0) or 0)
                / milestone_total,
            )
            curriculum_total["milestoneSamples"] += 1
        if str(record.get("objective", "")) in {"fast-win", "combo-win"} and isinstance(
            record.get("objectiveDamageProgress"), (int, float)
        ):
            curriculum_total["damageProgressTotal"] += min(
                1.0, max(0.0, float(record["objectiveDamageProgress"]))
            )
            curriculum_total["damageProgressSamples"] += 1
        if str(record.get("scenarioFamily", "")) == "sideboard":
            curriculum_total["sideboardCardsTotal"] += int(
                record.get("sideboardCardsSelected", 0) or 0
            )
            curriculum_total["sideboardSamples"] += 1
            curriculum_total["sideboardTargetSelectedTotal"] += int(
                record.get("sideboardTargetCardsSelected", 0) or 0
            )
            curriculum_total["sideboardTargetAvailableTotal"] += int(
                record.get("sideboardTargetCardsAvailable", 0) or 0
            )
            curriculum_total["sideboardCutSelectedTotal"] += int(
                record.get("sideboardCutCardsSelected", 0) or 0
            )
            curriculum_total["sideboardCutExpectedTotal"] += int(
                record.get("sideboardCutCardsExpected", 0) or 0
            )
        curriculum_total["recent"].append(success)
        curriculum_total["recent"] = curriculum_total["recent"][-20:]
        curriculum = record.get("curriculum", {})
        scenario_stats = curriculum.get(scenario_id, {}) if isinstance(curriculum, dict) else {}
        if isinstance(scenario_stats, dict) and isinstance(
            scenario_stats.get("weight"), (int, float)
        ):
            curriculum_total["adaptiveWeight"] = float(scenario_stats["weight"])
    curriculum_by_scenario = []
    for total in curriculum_totals.values():
        games_played = int(total.pop("games"))
        successes = int(total.pop("successes"))
        round_total = float(total.pop("roundTotal"))
        round_samples = int(total.pop("roundSamples"))
        decision_total = int(total.pop("decisionTotal"))
        milestone_progress_total = float(total.pop("milestoneProgressTotal"))
        milestone_samples = int(total.pop("milestoneSamples"))
        damage_progress_total = float(total.pop("damageProgressTotal"))
        damage_progress_samples = int(total.pop("damageProgressSamples"))
        sideboard_cards_total = int(total.pop("sideboardCardsTotal"))
        sideboard_samples = int(total.pop("sideboardSamples"))
        sideboard_target_selected = int(total.pop("sideboardTargetSelectedTotal"))
        sideboard_target_available = int(total.pop("sideboardTargetAvailableTotal"))
        sideboard_cut_selected = int(total.pop("sideboardCutSelectedTotal"))
        sideboard_cut_expected = int(total.pop("sideboardCutExpectedTotal"))
        curriculum_by_scenario.append(
            {
                **total,
                "games": games_played,
                "successes": successes,
                "successRate": successes / games_played if games_played else None,
                "averageRound": round_total / round_samples if round_samples else None,
                "averageDecisions": decision_total / games_played if games_played else None,
                "averageMilestoneProgress": (
                    milestone_progress_total / milestone_samples
                    if milestone_samples
                    else None
                ),
                "averageDamageProgress": (
                    damage_progress_total / damage_progress_samples
                    if damage_progress_samples
                    else None
                ),
                "averageSideboardCards": (
                    sideboard_cards_total / sideboard_samples
                    if sideboard_samples
                    else None
                ),
                "sideboardTargetCoverage": (
                    sideboard_target_selected / sideboard_target_available
                    if sideboard_target_available
                    else None
                ),
                "sideboardCutCoverage": (
                    sideboard_cut_selected / sideboard_cut_expected
                    if sideboard_cut_expected
                    else None
                ),
            }
        )
    curriculum_by_scenario.sort(
        key=lambda item: (
            float(item["successRate"] or 0),
            float(item["averageMilestoneProgress"] or 0),
            -int(item.get("games") or 0),
        )
    )
    evaluation_totals: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"games": 0, "successes": 0, "masteryTotal": 0.0}
    )
    for evaluation_result in curriculum_evaluation_results:
        scenario_id = str(evaluation_result.get("scenarioId") or "")
        if not scenario_id:
            continue
        evaluation_totals[scenario_id]["games"] += 1
        evaluation_totals[scenario_id]["successes"] += int(
            bool(evaluation_result.get("success"))
        )
        evaluation_totals[scenario_id]["masteryTotal"] += float(
            evaluation_result.get("mastery", 0) or 0
        )
    curriculum_evaluation_by_scenario = [
        {
            "scenarioId": scenario_id,
            "games": int(total["games"]),
            "successes": int(total["successes"]),
            "successRate": total["successes"] / total["games"],
            "meanMastery": total["masteryTotal"] / total["games"],
        }
        for scenario_id, total in sorted(evaluation_totals.items())
        if total["games"]
    ]
    config = _yaml_config(run)
    rl = config.get("rl", {}) if isinstance(config.get("rl"), dict) else {}
    supported = architecture in {"v11", "v12"} or (
        architecture == "v13"
        and bool(config.get("deckCatalog"))
        and rl.get("environment") == "engine"
    )
    fixed_seeds = bool(evaluations) and all(bool(item.get("fixedSeeds")) for item in evaluations)
    if not supported:
        status = "not-connected"
        verdict = "This run is not yet configured for authoritative Magic evaluation."
    elif not games:
        status = "not-evaluated"
        verdict = "No fixed-seed Engine evaluation has completed yet."
    elif len(games) < 30:
        status = "insufficient"
        verdict = "Engine games exist, but the sample is too small for a strong claim."
    elif lower is not None and lower > 0.5:
        status = "verified"
        verdict = (
            "The 95% confidence interval is above 50% against the frozen pre-Engine baseline."
            if architecture == "v13"
            else "The 95% confidence interval is above 50% against the saved champion."
        )
    else:
        status = "measured"
        verdict = "Magic performance is measured, but improvement is not yet statistically clear."
    return {
        "schemaVersion": "training-evidence/v1",
        "modelId": model_id,
        "status": status,
        "verdict": verdict,
        "engineEvaluationSupported": supported,
        "magicEvidence": bool(games),
        "fixedSeeds": fixed_seeds,
        "evaluationPeriods": len(evaluations),
        "games": len(games),
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "winRate": wins / len(games) if games else None,
        "lower95": lower,
        "upper95": upper,
        "promotions": len(promotions),
        "latestPromotion": promotions[-1] if promotions else None,
        "trend": trend,
        "byDeck": by_deck,
        "curriculum": {
            "games": len(curriculum_games),
            "byScenario": curriculum_by_scenario,
            "evaluationPeriods": len(curriculum_evaluations),
            "evaluationByScenario": curriculum_evaluation_by_scenario,
            "latestEvaluation": (
                curriculum_evaluations[-1].get("summary")
                if curriculum_evaluations
                else None
            ),
        },
    }


def _replay_path(run: Path, replay_id: str) -> tuple[Path, bool]:
    saved = run / "saved-replays" / f"{replay_id}.json"
    if saved.is_file():
        return saved, True
    return run / "replays" / f"{replay_id}.json", False


def _active_replay_leases(run: Path, replay_id: str) -> list[Path]:
    now = int(time.time())
    active: list[Path] = []
    lease_dir = run / "replays" / ".leases" / replay_id
    for path in lease_dir.glob("*.lease") if lease_dir.is_dir() else ():
        try:
            expires_at = int(path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            continue
        if expires_at > now:
            active.append(path)
    return active


def list_saved_replays(root: Path, model_id: str) -> list[dict[str, Any]]:
    run, _ = find_model_run(root, model_id)
    rows: list[dict[str, Any]] = []
    paths = list((run / "replays").glob("*.json"))
    paths.extend((run / "saved-replays").glob("*.json"))
    for path in paths:
        value = _read_json(path, {})
        if not isinstance(value, dict):
            continue
        frames = value.get("frames", [])
        metadata = value.get("metadata", {})
        permanently_saved = path.parent.name == "saved-replays"
        rows.append(
            {
                "id": path.stem,
                "createdAtUnixMs": value.get("createdAtUnixMs"),
                "episode": metadata.get("episode"),
                "seed": metadata.get("seed"),
                "matchupId": metadata.get("matchupId"),
                "decks": metadata.get("decks", []),
                "gameMode": metadata.get("gameMode"),
                "gameStatus": metadata.get("gameStatus"),
                "roundNumber": metadata.get("roundNumber"),
                "winner": (metadata.get("outcome") or {}).get("winner"),
                "durationSeconds": metadata.get("gameDurationSeconds"),
                "frameCount": len(frames) if isinstance(frames, list) else 0,
                "bytes": path.stat().st_size,
                "saved": permanently_saved,
                "viewing": permanently_saved
                or bool(_active_replay_leases(run, path.stem)),
            }
        )
    return sorted(rows, key=lambda row: int(row.get("createdAtUnixMs") or 0), reverse=True)


def load_saved_replay(root: Path, model_id: str, replay_id: str) -> dict[str, Any]:
    if not REPLAY_ID.fullmatch(replay_id):
        raise ValueError("Invalid replay id")
    run, _ = find_model_run(root, model_id)
    path, _ = _replay_path(run, replay_id)
    value = _read_json(path, None)
    if not isinstance(value, dict):
        raise FileNotFoundError("Saved game not found")
    return value


def acquire_replay_lease(root: Path, model_id: str, replay_id: str) -> dict[str, Any]:
    if not REPLAY_ID.fullmatch(replay_id):
        raise ValueError("Invalid replay id")
    run, _ = find_model_run(root, model_id)
    replay = load_saved_replay(root, model_id, replay_id)
    _, permanently_saved = _replay_path(run, replay_id)
    lease_id = secrets.token_urlsafe(18)
    expires_at = int(time.time()) + REPLAY_LEASE_SECONDS
    if not permanently_saved:
        lease_dir = run / "replays" / ".leases" / replay_id
        lease_dir.mkdir(parents=True, exist_ok=True)
        (lease_dir / f"{lease_id}.lease").write_text(str(expires_at), encoding="utf-8")
    return {
        "replay": replay,
        "leaseId": lease_id,
        "expiresAtUnixMs": expires_at * 1000,
        "saved": permanently_saved,
    }


def renew_replay_lease(
    root: Path, model_id: str, replay_id: str, lease_id: str
) -> dict[str, int]:
    if not REPLAY_ID.fullmatch(replay_id) or not REPLAY_ID.fullmatch(lease_id):
        raise ValueError("Invalid replay lease")
    run, _ = find_model_run(root, model_id)
    path, permanently_saved = _replay_path(run, replay_id)
    if not path.is_file():
        raise FileNotFoundError("Saved game not found")
    expires_at = int(time.time()) + REPLAY_LEASE_SECONDS
    if not permanently_saved:
        lease_path = run / "replays" / ".leases" / replay_id / f"{lease_id}.lease"
        if not lease_path.is_file():
            raise FileNotFoundError("Replay viewing lease not found")
        lease_path.write_text(str(expires_at), encoding="utf-8")
    return {"expiresAtUnixMs": expires_at * 1000}


def release_replay_lease(
    root: Path, model_id: str, replay_id: str, lease_id: str
) -> dict[str, bool]:
    if not REPLAY_ID.fullmatch(replay_id) or not REPLAY_ID.fullmatch(lease_id):
        raise ValueError("Invalid replay lease")
    run, _ = find_model_run(root, model_id)
    lease_path = run / "replays" / ".leases" / replay_id / f"{lease_id}.lease"
    lease_path.unlink(missing_ok=True)
    return {"released": True}


def save_replay_forever(root: Path, model_id: str, replay_id: str) -> dict[str, bool]:
    if not REPLAY_ID.fullmatch(replay_id):
        raise ValueError("Invalid replay id")
    run, _ = find_model_run(root, model_id)
    source = run / "replays" / f"{replay_id}.json"
    destination = run / "saved-replays" / f"{replay_id}.json"
    if destination.is_file():
        return {"saved": True}
    if not source.is_file():
        raise FileNotFoundError("Saved game not found")
    destination.parent.mkdir(parents=True, exist_ok=True)
    source.replace(destination)
    return {"saved": True}


def load_agent_training_contract(root: Path, model_id: str) -> dict[str, Any]:
    run, _ = find_model_run(root, model_id)
    contract = _read_json(run / "agent-training-contract.json", None)
    if not isinstance(contract, dict):
        raise FileNotFoundError("This agent has not published a training contract")
    published_metrics = _read_jsonl(run / "agent-training-metrics.jsonl")
    metrics: list[dict[str, Any]] = []
    if not metrics:
        for record in _read_jsonl(run / "v13-metrics.jsonl"):
            values = {
                key: value
                for key, value in record.items()
                if (key == "loss" or key.endswith("_loss"))
                and isinstance(value, (int, float))
            }
            metrics.append(
                {
                    "schemaVersion": "deepdeck-agent-training-metrics/v1",
                    "agentId": contract.get("agentId"),
                    "phase": "world-model",
                    "step": int(record.get("step", 0) or 0),
                    "metrics": values,
                    "recordedAtUnixMs": None,
                    "context": {"compatibilitySource": "v13-metrics.jsonl"},
                }
            )
        for record in _read_jsonl(run / "v13-rl-metrics.jsonl"):
            values = {
                key: float(record[key])
                for key in (
                    "loss",
                    "policy_loss",
                    "value_loss",
                    "entropy",
                    "approx_kl",
                    "episode_reward",
                )
                if isinstance(record.get(key), (int, float))
            }
            metrics.append(
                {
                    "schemaVersion": "deepdeck-agent-training-metrics/v1",
                    "agentId": contract.get("agentId"),
                    "phase": "reinforcement-learning",
                    "step": int(record.get("trainingStep", 0) or 0),
                    "metrics": values,
                    "recordedAtUnixMs": None,
                    "context": {
                        "compatibilitySource": "v13-rl-metrics.jsonl",
                        "rolloutBatch": record.get("rolloutBatch"),
                    },
                }
            )
        for record in _read_jsonl(run / "v13-engine-evaluations.jsonl"):
            summary = record.get("summary", {})
            lower_95, _ = _wilson_interval(
                int(summary.get("candidateWins", 0) or 0),
                int(summary.get("completedGames", 0) or 0),
            )
            metrics.append(
                {
                    "schemaVersion": "deepdeck-agent-training-metrics/v1",
                    "agentId": contract.get("agentId"),
                    "phase": "engine-evaluation",
                    "step": int(
                        record.get("candidateTrainingStep", record.get("trainingStep", 0))
                        or 0
                    ),
                    "metrics": {
                        "win_rate": float(summary.get("candidateWinRate", 0.0) or 0.0),
                        "lower_95": float(lower_95 or 0.0),
                    },
                    "recordedAtUnixMs": None,
                    "context": {"compatibilitySource": "v13-engine-evaluations.jsonl"},
                }
            )
    metric_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for record in [*metrics, *published_metrics]:
        key = (str(record.get("phase", "")), int(record.get("step", 0) or 0))
        metric_by_key[key] = record
    metrics_by_phase: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in metric_by_key.values():
        metrics_by_phase[str(record.get("phase", ""))].append(record)
    metrics = []
    phase_ids = [
        str(phase.get("id"))
        for phase in contract.get("phases", [])
        if isinstance(phase, dict) and phase.get("id")
    ]
    for phase_id in phase_ids:
        phase_metrics = sorted(
            metrics_by_phase.pop(phase_id, []),
            key=lambda record: int(record.get("step", 0) or 0),
        )
        metrics.extend(phase_metrics[-650:])
    for phase_metrics in metrics_by_phase.values():
        metrics.extend(phase_metrics[-50:])
    state = _read_json(run / "agent-training-state.json", None)
    if not isinstance(state, dict):
        state = _read_json(run / "v13-training-state.json", None)
        if isinstance(state, dict) and "phase" not in state:
            raw_phase = str(state.get("trainingPhase", ""))
            state = {
                **state,
                "phase": (
                    "reinforcement-learning"
                    if raw_phase in {"collecting-engine-games", "recovering-engine-game"}
                    else raw_phase
                ),
                "step": state.get("trainingStep", state.get("step", 0)),
            }
    config = _yaml_config(run)
    parameter_values = {
        str(parameter["key"]): _nested_config_value(
            config, str(parameter["key"]), parameter.get("default")
        )
        for phase in contract.get("phases", [])
        if isinstance(phase, dict)
        for parameter in phase.get("parameters", [])
        if isinstance(parameter, dict) and parameter.get("key")
    }
    return {
        "contract": contract,
        "state": state,
        "control": _read_json(run / "agent-training-control.json", None),
        "metrics": metrics[-2_000:],
        "parameterValues": parameter_values,
    }


def _validate_agent_parameter(definition: dict[str, Any], value: Any) -> None:
    kind = definition.get("kind")
    if kind == "boolean" and not isinstance(value, bool):
        raise ValueError(f"{definition.get('key')} must be a boolean")
    if kind == "integer" and (not isinstance(value, int) or isinstance(value, bool)):
        raise ValueError(f"{definition.get('key')} must be an integer")
    if kind == "number" and (
        not isinstance(value, (int, float)) or isinstance(value, bool)
    ):
        raise ValueError(f"{definition.get('key')} must be a number")
    if kind == "string" and not isinstance(value, str):
        raise ValueError(f"{definition.get('key')} must be a string")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        minimum = definition.get("min")
        maximum = definition.get("max")
        if isinstance(minimum, (int, float)) and value < minimum:
            raise ValueError(f"{definition.get('key')} must be at least {minimum}")
        if isinstance(maximum, (int, float)) and value > maximum:
            raise ValueError(f"{definition.get('key')} must be at most {maximum}")
    choices = definition.get("choices")
    if kind == "choice" and isinstance(choices, list) and value not in choices:
        raise ValueError(f"{definition.get('key')} is not an allowed choice")


def save_agent_training_control(
    root: Path, model_id: str, payload: dict[str, Any]
) -> dict[str, Any]:
    run, _ = find_model_run(root, model_id)
    published = load_agent_training_contract(root, model_id)
    contract = published["contract"]
    phases = {
        str(phase.get("id")): phase
        for phase in contract.get("phases", [])
        if isinstance(phase, dict) and phase.get("id")
    }
    phase_id = str(payload.get("phase", ""))
    if phase_id not in phases:
        raise ValueError("phase must identify a published training phase")
    action = str(payload.get("action", ""))
    controls = phases[phase_id].get("controls", [])
    if action not in controls:
        raise ValueError(f"{action or 'action'} is not supported by phase {phase_id}")
    definitions = {
        str(parameter.get("key")): parameter
        for phase in phases.values()
        for parameter in phase.get("parameters", [])
        if isinstance(parameter, dict) and parameter.get("key")
    }
    parameters = payload.get("parameters", {})
    if not isinstance(parameters, dict):
        raise ValueError("parameters must be an object")
    for key, value in parameters.items():
        definition = definitions.get(str(key))
        if definition is None:
            raise ValueError(f"unknown training parameter: {key}")
        _validate_agent_parameter(definition, value)
    if parameters:
        config = _yaml_config(run)
        for key, value in parameters.items():
            _set_nested_config_value(config, str(key), value)
        _write_yaml_config(run, config)
    previous = published.get("control")
    revision = int(previous.get("revision", 0) if isinstance(previous, dict) else 0) + 1
    control = {
        "schemaVersion": "deepdeck-agent-training-control/v1",
        "revision": revision,
        "phase": phase_id,
        "action": action,
        "parameters": parameters,
        "requestedAtUnixMs": int(time.time() * 1000),
    }
    _write_json_atomic(run / "agent-training-control.json", control)
    return control
