from __future__ import annotations

import contextlib
import json
import os
import random
import re
import shutil
import sqlite3
import subprocess
import sys
import threading
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
import psutil

from oracle_ai.training.matchmaking import plackett_luce_matchmaking_weight

from .card_models import (
    CardModelError,
    card_identifier,
    compile_oracle_rules,
    enrich_card_characteristics,
    refresh_playtest_deck,
)
from .dependencies import current_revision, engine_binary, engine_build_current, pinned_revision
from .resources import active_games, find_model_run, load_resource_plan, resource_snapshot

MAX_LOG_LINES = 500
PUBLIC_LOG_LINES = 20
MAX_PUBLIC_LOG_LINE_CHARS = 2_000
MAX_RANDOM_SEED = 2**31 - 1
TRAINING_KINDS = {
    "training.smoke",
    "training.dataset",
    "training.pool",
    "training.v13-world-model",
}
PLAYTEST_KIND = "playtest.agent"
MATCHMAKING_KIND = "matchmaking.agent"
DEPENDENCY_KINDS = {
    "dependency.engine.start",
    "dependency.pixi.prepare",
    "dependency.sync",
    "dependency.stack.prepare",
}
SUPPORTED_KINDS = TRAINING_KINDS | {PLAYTEST_KIND, MATCHMAKING_KIND} | DEPENDENCY_KINDS


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def public_logs(values: Any) -> list[str]:
    if not isinstance(values, (list, tuple, deque)):
        return []
    lines = [str(value) for value in values][-PUBLIC_LOG_LINES:]
    return [
        line
        if len(line) <= MAX_PUBLIC_LOG_LINE_CHARS
        else f"{line[:MAX_PUBLIC_LOG_LINE_CHARS]}... [truncated]"
        for line in lines
    ]


def is_loopback_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and parsed.hostname in {
        "127.0.0.1",
        "localhost",
        "::1",
    }


@dataclass
class Job:
    id: str
    kind: str
    label: str
    argv: list[str]
    status: str = "queued"
    created_at: str = field(default_factory=utc_now)
    started_at: str | None = None
    finished_at: str | None = None
    exit_code: int | None = None
    artifact_path: str | None = None
    logs: deque[str] = field(default_factory=lambda: deque(maxlen=MAX_LOG_LINES))
    process: subprocess.Popen[str] | None = field(default=None, repr=False)
    payload: dict[str, Any] | None = field(default=None, repr=False)
    model_id: str | None = None
    worker_slots: int = 1
    details: dict[str, Any] | None = None
    stop_requested: bool = field(default=False, repr=False)

    def public(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "label": self.label,
            "argv": list(self.argv),
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "exit_code": self.exit_code,
            "artifact_path": self.artifact_path,
            "logs": public_logs(self.logs),
            "model_id": self.model_id,
            "worker_slots": self.worker_slots,
            "details": self.details,
        }


class JobValidationError(ValueError):
    pass


class JobManager:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.checkpoint_root = self.root / ".deepdeck" / "checkpoints"
        self.trajectory_path = self.root / ".deepdeck" / "trajectories" / "decisions.jsonl"
        self.trajectory_path.parent.mkdir(parents=True, exist_ok=True)
        self.trajectory_path.touch(exist_ok=True)
        self.database_path = self.root / ".deepdeck" / "learner.db"
        self._initialize_database()
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def list_jobs(self) -> list[dict[str, Any]]:
        with self._lock:
            current = {job.id: job.public() for job in self._jobs.values()}
        with sqlite3.connect(self.database_path) as connection:
            rows = connection.execute(
                "SELECT payload FROM jobs ORDER BY created_at DESC LIMIT 100"
            ).fetchall()
        stored = [json.loads(row[0]) for row in rows]
        live_commands = (
            self._live_process_commands()
            if any(item.get("status") == "running" for item in stored)
            else []
        )
        persisted = [
            {
                **reconciled,
                "logs": public_logs(reconciled.get("logs", [])),
            }
            for item in stored
            for reconciled in [self._reconcile_persisted_job(item, live_commands)]
        ]
        merged = {item["id"]: item for item in persisted}
        merged.update(current)
        return sorted(merged.values(), key=lambda item: item["created_at"], reverse=True)[:100]

    @staticmethod
    def _live_process_commands() -> list[str]:
        commands: list[str] = []
        for process in psutil.process_iter(["cmdline"]):
            try:
                command = " ".join(process.info.get("cmdline") or []).casefold()
            except (psutil.Error, OSError):
                continue
            if command:
                commands.append(command)
        return commands

    @staticmethod
    def _reconcile_persisted_job(
        job: dict[str, Any], live_commands: list[str] | None = None
    ) -> dict[str, Any]:
        if job.get("status") != "running":
            return job
        commands = live_commands
        if commands is None:
            commands = JobManager._live_process_commands()
        kind = job.get("kind")
        if kind in {PLAYTEST_KIND, MATCHMAKING_KIND}:
            argv = [str(value).casefold() for value in job.get("argv", [])]
            signature = next(
                (
                    argv[index + 1]
                    for flag in ("--local-game-setup", "--checkpoint")
                    for index, value in enumerate(argv[:-1])
                    if value == flag
                ),
                "",
            )
            for command in commands:
                if "deepdeck_examples.run" in command and signature and signature in command:
                    return job
            return {
                **job,
                "status": "stopped",
                "finished_at": job.get("finished_at") or utc_now(),
                "logs": [
                    *list(job.get("logs", [])),
                    "The workbench restarted; this local process is no longer attached.",
                ][-MAX_LOG_LINES:],
            }
        if kind != "training.pool":
            return {
                **job,
                "status": "stopped",
                "finished_at": job.get("finished_at") or utc_now(),
            }
        artifact = str(job.get("artifact_path", "")).strip()
        if not artifact:
            return job
        expected = str(Path(artifact).resolve()).casefold()
        for command in commands:
            if "oracle_ai.training.league" in command and expected in command:
                return job
        return {**job, "status": "stopped"}

    def _initialize_database(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.database_path) as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS jobs ("
                "id TEXT PRIMARY KEY, created_at TEXT NOT NULL, "
                "kind TEXT NOT NULL, payload TEXT NOT NULL)"
            )

    def _persist(self, job: Job) -> None:
        payload = job.public()
        with sqlite3.connect(self.database_path) as connection:
            connection.execute(
                "INSERT OR REPLACE INTO jobs (id, created_at, kind, payload) VALUES (?, ?, ?, ?)",
                (job.id, job.created_at, job.kind, json.dumps(payload)),
            )

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return job.public() if job else None

    def resources(self) -> dict[str, Any]:
        with self._lock:
            jobs = list(self._jobs.values())
        snapshot = resource_snapshot(self.root, jobs)
        game_count = len(self.games())
        snapshot["engine"]["activeLocalGames"] = game_count
        snapshot["engine"]["ramPerGameEstimate"] = (
            snapshot["engine"]["ramBytes"] // game_count if game_count else 0
        )
        return snapshot

    def games(self) -> list[dict[str, Any]]:
        return active_games(self.root, self.list_jobs())

    def model_has_active_workers(self, model_id: str) -> bool:
        return any(
            worker.get("modelId") == model_id for worker in self.resources().get("workers", [])
        )

    def prepare_model(self, raw: dict[str, Any]) -> str:
        model = str(raw.get("model", "v12"))
        if model not in {"v11", "v12", "v13", "v15"}:
            raise JobValidationError("Model must be v11, v12, v13, or v15.")
        self._local_model_name(raw)
        self._bounded_int(raw, "parallel_matches", default=1, minimum=1, maximum=32)
        self._bounded_int(raw, "gpu_memory_mb", default=0, minimum=0, maximum=24 * 1024)
        if model == "v13":
            return self._prepare_v13_model(raw)
        if model == "v15":
            return self._prepare_v15_model(raw)
        _, _, run = self._pool_training_command(model, raw)
        control = run / "training-control.json"
        control.write_text(
            json.dumps({"desiredState": "paused"}, indent=2) + "\n",
            encoding="utf-8",
        )
        metadata = json.loads((run / "local-model.json").read_text(encoding="utf-8"))
        return str(metadata["id"])

    def _prepare_v13_model(self, raw: dict[str, Any]) -> str:
        try:
            import yaml
        except ModuleNotFoundError as error:
            raise JobValidationError(
                "Install DeepDeckLearner's deep-learning dependencies."
            ) from error
        model_name = self._local_model_name(raw)
        now = f"{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"
        slug = re.sub(r"[^a-z0-9]+", "-", model_name.casefold()).strip("-") or "v13-agent"
        model_id = f"{slug[:40]}-{now.lower()}"
        run = self.root / ".deepdeck" / "runs" / f"{now}-{slug[:40]}-v13"
        run.mkdir(parents=True, exist_ok=False)
        template = self.root / "configs" / "oracle-ai" / "v13-world-model-smoke.yaml"
        if not template.is_file():
            raise JobValidationError("The V13 world-model configuration is unavailable.")
        config = yaml.safe_load(template.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise JobValidationError("The V13 world-model configuration is invalid.")
        config["name"] = model_name
        config["outputDir"] = str(run)
        training = config.setdefault("training", {})
        training["seed"] = self._random_seed()
        training["steps"] = self._bounded_int(
            raw, "training_steps", default=1000, minimum=1, maximum=1_000_000
        )
        config_path = run / "training-config.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        resource_plan = {
            "trainingMatches": 1,
            "leagueMatches": 0,
            "localMatches": 1,
            "gpuMemoryMb": self._bounded_int(
                raw, "gpu_memory_mb", default=0, minimum=0, maximum=24 * 1024
            ),
        }
        (run / "learner-resources.json").write_text(
            json.dumps(resource_plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        metadata = {
            "schemaVersion": "local-model/v1",
            "id": model_id,
            "name": model_name,
            "architecture": "v13",
            "format": "legacy",
            "description": "V13 structured Transformer world-model training agent.",
            "createdAt": utc_now(),
            "checkpointPath": str(run / "checkpoints" / f"step-{training['steps']}"),
            "reservePlaytest": True,
            "selfPlayAllSeats": False,
            "source": "user-trained",
            "decks": [],
            "resourcePlan": resource_plan,
            "agentSdkRuntime": {
                "module": "deepdeck_examples.run",
                "arguments": ["v13"],
                "checkpointArgument": "--checkpoint",
                "requiredFiles": ["rl-model.pt"],
            },
        }
        (run / "local-model.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (run / "training-control.json").write_text(
            json.dumps({"desiredState": "paused"}, indent=2) + "\n", encoding="utf-8"
        )
        return model_id

    def _prepare_v15_model(self, raw: dict[str, Any]) -> str:
        try:
            import yaml
        except ModuleNotFoundError as error:
            raise JobValidationError(
                "Install DeepDeckLearner's deep-learning dependencies."
            ) from error
        model_name = self._local_model_name(raw)
        now = f"{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"
        slug = re.sub(r"[^a-z0-9]+", "-", model_name.casefold()).strip("-") or "v15-agent"
        model_id = f"{slug[:40]}-{now.lower()}"
        run = self.root / ".deepdeck" / "runs" / f"{now}-{slug[:40]}-v15"
        run.mkdir(parents=True, exist_ok=False)
        template = self.root / "configs" / "oracle-ai" / "v15-typed-latent-engine.yaml"
        if not template.is_file():
            raise JobValidationError("The V15 typed-latent configuration is unavailable.")
        config = yaml.safe_load(template.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise JobValidationError("The V15 typed-latent configuration is invalid.")
        config["name"] = model_name
        config["outputDir"] = str(run)
        training = config.setdefault("training", {})
        training["seed"] = self._random_seed()
        training["steps"] = self._bounded_int(
            raw, "training_steps", default=1000, minimum=1, maximum=1_000_000
        )
        config_path = run / "training-config.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        resource_plan = {
            "trainingMatches": 1,
            "leagueMatches": 0,
            "localMatches": 1,
            "gpuMemoryMb": self._bounded_int(
                raw, "gpu_memory_mb", default=0, minimum=0, maximum=24 * 1024
            ),
        }
        (run / "learner-resources.json").write_text(
            json.dumps(resource_plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        checkpoint = run / "checkpoints" / f"step-{training['steps']}"
        metadata = {
            "schemaVersion": "local-model/v1",
            "id": model_id,
            "name": model_name,
            "architecture": "v15",
            "format": "legacy",
            "description": "V15 typed-latent conditional planning agent.",
            "createdAt": utc_now(),
            "checkpointPath": str(checkpoint),
            "reservePlaytest": True,
            "selfPlayAllSeats": False,
            "source": "user-trained",
            "decks": [],
            "resourcePlan": resource_plan,
            "agentSdkRuntime": {
                "module": "deepdeck_examples.run",
                "arguments": ["v15"],
                "checkpointArgument": "--checkpoint",
                "requiredFiles": ["v15-model.pt"],
            },
        }
        (run / "local-model.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (run / "training-control.json").write_text(
            json.dumps({"desiredState": "paused"}, indent=2) + "\n", encoding="utf-8"
        )
        return model_id

    def update_model(self, model_id: str, raw: dict[str, Any]) -> str:
        if self.model_has_active_workers(model_id):
            raise JobValidationError("Stop this agent's active jobs before editing it.")
        try:
            import yaml
        except ModuleNotFoundError as error:
            raise JobValidationError(
                "Install DeepDeckLearner's deep-learning dependencies."
            ) from error
        try:
            run, metadata = find_model_run(self.root, model_id)
        except ValueError as error:
            raise JobValidationError(str(error)) from error
        if metadata.get("source") == "local-frozen-checkpoint":
            raise JobValidationError("Built-in reference agents cannot be edited.")

        model_name = self._local_model_name({"model_name": raw.get("name", metadata.get("name"))})
        required_format = str(metadata.get("format", "")).casefold()
        decks = raw.get("decks")
        if not isinstance(decks, list) or not 1 <= len(decks) <= 100:
            raise JobValidationError("Select between 1 and 100 decks for this agent.")
        compatible = [
            deck
            for deck in decks
            if isinstance(deck, dict)
            and isinstance(deck.get("id"), str)
            and str(deck.get("format", "")).casefold() == required_format
        ]
        if len(compatible) != len(decks):
            raise JobValidationError(f"Every selected deck must be a valid {required_format} deck.")

        config_path = run / "training-config.yaml"
        try:
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise JobValidationError(
                "This agent's training configuration is unavailable."
            ) from error
        if not isinstance(config, dict):
            raise JobValidationError("This agent's training configuration is invalid.")
        engine_url = str(config.get("engineUrl", "http://127.0.0.1:8787"))
        catalog: dict[str, list[dict[str, Any]]] = {}
        for deck in compatible:
            version_id = str(deck["id"])
            snapshot_path = self.root / ".deepdeck" / "decks" / f"{version_id}.json"
            try:
                snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as error:
                raise JobValidationError(
                    f"Download {deck.get('name', version_id)} again before saving."
                ) from error
            entries = [
                entry
                for entry in snapshot.get("cards", [])
                if isinstance(entry, dict) and entry.get("section") != "considering"
            ]
            entries = self._enrich_card_characteristics(entries)
            compiled_rules = self._compile_oracle_rules(engine_url, entries)
            cards: list[dict[str, Any]] = []
            for entry in entries:
                quantity = max(0, int(entry.get("quantity", 0)))
                card_id = self._card_identifier(entry)
                if not card_id:
                    raise JobValidationError(
                        f"{deck.get('name', version_id)} contains a card without an id."
                    )
                raw_rules = entry.get("rules")
                current_rules = compiled_rules.get(card_id)
                rules = (
                    current_rules
                    if isinstance(current_rules, list)
                    else raw_rules
                    if isinstance(raw_rules, list)
                    else []
                )
                section = str(entry.get("section", "main"))
                is_token, is_game_piece = self._auxiliary_game_piece_flags(entry)
                normalized = {
                    "id": card_id,
                    "name": str(entry.get("name", "")),
                    "typeLine": str(entry.get("typeLine", "")),
                    "manaCost": str(entry.get("manaCost") or ""),
                    "oracleText": entry.get("oracleText"),
                    "layout": entry.get("layout"),
                    "faces": entry.get("faces", []),
                    "power": entry.get("power"),
                    "toughness": entry.get("toughness"),
                    "rules": rules,
                    "isToken": is_token,
                    "isGamePiece": is_game_piece,
                    "isSideboard": section in {"sideboard", "companion"},
                    "isCommander": section == "commander",
                    "sourceSessionId": version_id,
                }
                cards.extend(dict(normalized) for _ in range(quantity))
            label = f"{deck.get('name', 'Deck')} · v{deck.get('version', 1)} · {version_id[:8]}"
            catalog[label] = cards

        reserve_playtest = bool(raw.get("reservePlaytest", metadata.get("reservePlaytest", True)))
        shared_self_play = bool(raw.get("selfPlayAllSeats", metadata.get("selfPlayAllSeats", True)))
        settings = config.setdefault("learnerSettings", {})
        settings.update(
            {
                "modelId": model_id,
                "modelName": model_name,
                "reservePlaytest": reserve_playtest,
                "selfPlayAllSeats": shared_self_play,
                "selectedDeckVersionIds": [str(deck["id"]) for deck in compatible],
            }
        )
        config["trainingOpponentMix"] = (
            {"self": 1.0} if shared_self_play else {"self": 0.5, "anchor": 0.5}
        )
        config["serviceRefreshEvery"] = 1 if reserve_playtest else 1_000_000
        if str(metadata.get("architecture", "")).casefold() == "v13":
            config["deckCatalog"] = str(run / "training-decks.json")
            config.setdefault("engineUrl", "http://127.0.0.1:8787")
            config.setdefault("engineTimeoutSeconds", 300)
            config["savedGameLimit"] = int(config.get("savedGameLimit", 20))
            config["trainingScenarioRandomizer"] = {
                "formats": ["legacy"],
                "formatSampling": "roundRobin",
                "playerCounts": [2],
                "maxTurns": 80,
                "matchmaking": {"enabled": False},
            }
            rl = config.setdefault("rl", {})
            rl["environment"] = "engine"
            rl.setdefault("evaluation_every_updates", 10)
            rl.setdefault("evaluation_games", 4)
            rl.setdefault("max_consecutive_errors", 10)
        metadata.update(
            {
                "name": model_name,
                "reservePlaytest": reserve_playtest,
                "selfPlayAllSeats": shared_self_play,
                "decks": compatible,
            }
        )

        pending_catalog = run / "training-decks.json.pending"
        pending_config = run / "training-config.yaml.pending"
        pending_metadata = run / "local-model.json.pending"
        pending_catalog.write_text(
            json.dumps(catalog, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        pending_config.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        pending_metadata.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        pending_catalog.replace(run / "training-decks.json")
        pending_config.replace(config_path)
        pending_metadata.replace(run / "local-model.json")
        return model_id

    def create(self, raw: dict[str, Any]) -> dict[str, Any]:
        kind = str(raw.get("kind", ""))
        if kind not in SUPPORTED_KINDS:
            raise JobValidationError(f"Unsupported job kind: {kind or '(missing)'}")
        argv: list[str]
        label: str
        artifact: Path | None
        deferred_payload: dict[str, Any] | None
        model_id: str | None
        worker_slots: int
        if kind == "training.pool" and str(raw.get("model_id", "")).strip():
            model_id = str(raw["model_id"]).strip()
            requested_phase = str(raw.get("phase_id", "")).strip() or None
            argv, label, artifact = self._existing_pool_training_command(model_id, requested_phase)
            deferred_payload = None
            plan = load_resource_plan(artifact)
            if plan["trainingMatches"] <= 0:
                raise JobValidationError(
                    "Allocate at least one simultaneous training game before starting."
                )
            worker_slots = plan["trainingMatches"]
        elif kind == "training.pool":
            model = str(raw.get("model", "v12"))
            if model not in {"v11", "v12"}:
                raise JobValidationError("Model must be v11 or v12.")
            model_name = self._local_model_name(raw)
            self._bounded_int(raw, "parallel_matches", default=1, minimum=1, maximum=32)
            self._bounded_int(raw, "gpu_memory_mb", default=0, minimum=0, maximum=24 * 1024)
            argv = []
            label = f"{model_name} · {model.upper()} · preparing"
            artifact = None
            deferred_payload = dict(raw)
            model_id = None
            worker_slots = self._bounded_int(
                raw, "parallel_matches", default=1, minimum=1, maximum=32
            )
        else:
            argv, label, artifact = self._command(kind, raw)
            deferred_payload = None
            model_id = str(raw.get("model_id", "")).strip() or None
            worker_slots = (
                self._bounded_int(raw, "connections", default=1, minimum=1, maximum=32)
                if kind == MATCHMAKING_KIND
                else 1
            )
        job = Job(
            id=str(uuid.uuid4()),
            kind=kind,
            label=label,
            argv=argv,
            artifact_path=str(artifact) if artifact else None,
            payload=deferred_payload,
            model_id=model_id,
            worker_slots=worker_slots,
            details=raw.get("_job_details"),
        )
        with self._lock:
            if kind in TRAINING_KINDS and any(
                candidate.kind in TRAINING_KINDS
                and candidate.status in {"queued", "running"}
                and (model_id is None or candidate.model_id in {None, model_id})
                for candidate in self._jobs.values()
            ):
                raise JobValidationError("This agent already has an active training job.")
            if model_id and kind in {PLAYTEST_KIND, MATCHMAKING_KIND}:
                run, _ = find_model_run(self.root, model_id)
                plan = load_resource_plan(run)
                limit_key = "localMatches" if kind == PLAYTEST_KIND else "leagueMatches"
                running = sum(
                    int(worker.get("workerSlots", 1))
                    for worker in resource_snapshot(self.root, list(self._jobs.values()))["workers"]
                    if worker["kind"] == kind and worker["modelId"] == model_id
                )
                queued = sum(
                    candidate.worker_slots
                    for candidate in self._jobs.values()
                    if candidate.kind == kind
                    and candidate.model_id == model_id
                    and candidate.status == "queued"
                )
                active = running + queued
                if active + worker_slots > plan[limit_key]:
                    raise JobValidationError(
                        f"{limit_key} allocation is full for this local model."
                    )
            dependency = self._dependency_for(kind, raw)
            active_dependency_jobs = [
                candidate
                for candidate in self._jobs.values()
                if candidate.kind in DEPENDENCY_KINDS and candidate.status in {"queued", "running"}
            ]
            if dependency == "stack" and active_dependency_jobs:
                raise JobValidationError("A local runtime task is already running.")
            if dependency and any(
                candidate.kind == "dependency.stack.prepare" for candidate in active_dependency_jobs
            ):
                raise JobValidationError("The local runtime setup is already running.")
            if dependency and any(
                dependency in candidate.label.lower() for candidate in active_dependency_jobs
            ):
                raise JobValidationError(f"A {dependency} dependency task is already running.")
            self._jobs[job.id] = job
            self._persist(job)
        threading.Thread(target=self._run, args=(job,), daemon=True).start()
        return job.public()

    def stop(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            process = job.process if job else None
        if not job:
            workers = self.resources().get("workers", [])
            worker = next(
                (item for item in workers if item.get("jobId") == job_id),
                None,
            )
            # After a controller restart, persisted jobs keep their original id while
            # resource discovery identifies the still-running process as recovered-PID.
            # Reconnect the two when there is exactly one safe model/kind match.
            if not worker:
                persisted = next(
                    (
                        item
                        for item in self.list_jobs()
                        if item.get("id") == job_id and item.get("status") == "running"
                    ),
                    None,
                )
                candidates = [
                    item
                    for item in workers
                    if str(item.get("jobId", "")).startswith("recovered-")
                    and persisted
                    and item.get("kind") == persisted.get("kind")
                    and item.get("modelId") == persisted.get("model_id")
                ]
                if len(candidates) == 1:
                    worker = candidates[0]
            if not worker:
                return None
            pids = [int(pid) for pid in worker.get("pids", []) if int(pid) > 0]
            for pid in reversed(pids):
                try:
                    candidate = psutil.Process(pid)
                    candidate.terminate()
                except (psutil.Error, OSError):
                    continue
            remaining: list[psutil.Process] = []
            for pid in pids:
                with contextlib.suppress(psutil.Error, OSError):
                    remaining.append(psutil.Process(pid))
            _, alive = psutil.wait_procs(remaining, timeout=3)
            for candidate in alive:
                with contextlib.suppress(psutil.Error, OSError):
                    candidate.kill()
            return {
                "id": job_id,
                "kind": worker.get("kind", "recovered"),
                "label": worker.get("label", "Recovered local process"),
                "argv": [],
                "status": "stopped",
                "created_at": utc_now(),
                "started_at": None,
                "finished_at": utc_now(),
                "exit_code": None,
                "artifact_path": None,
                "logs": ["Recovered local process stopped by user."],
                "model_id": worker.get("modelId"),
                "worker_slots": worker.get("workerSlots", 1),
                "details": None,
            }
        job.stop_requested = True
        if job.kind == "training.pool" and job.artifact_path:
            (Path(job.artifact_path) / "training-control.json").write_text(
                json.dumps({"desiredState": "paused"}, indent=2) + "\n",
                encoding="utf-8",
            )
        if process and process.poll() is None:
            job.logs.append("Stop requested by user.")
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    capture_output=True,
                    check=False,
                    shell=False,
                    text=True,
                )
            else:
                process.terminate()
        job.status = "stopped"
        job.finished_at = utc_now()
        self._persist(job)
        return job.public()

    def cancel_game(self, game_id: str) -> dict[str, Any] | None:
        game = next((item for item in self.games() if item.get("id") == game_id), None)
        if not game or not game.get("sessionId"):
            return None
        if game.get("source") == "local" and game.get("jobId"):
            self.stop(str(game["jobId"]))
        session_id = str(game["sessionId"])
        if not re.fullmatch(r"[A-Za-z0-9:._-]+", session_id):
            raise JobValidationError("Engine returned an invalid game-session identifier.")
        try:
            response = httpx.delete(
                f"http://127.0.0.1:8787/game/sessions/{session_id}", timeout=10.0
            )
            if response.status_code not in {200, 404}:
                response.raise_for_status()
        except httpx.HTTPError as error:
            raise JobValidationError(f"Unable to cancel the Engine game: {error}") from error
        return {**game, "status": "cancelled", "canCancel": False}

    def _command(self, kind: str, raw: dict[str, Any]) -> tuple[list[str], str, Path | None]:
        if kind in TRAINING_KINDS:
            return self._training_command(kind, raw)
        if kind in DEPENDENCY_KINDS:
            return self._dependency_command(kind, raw)
        if kind == MATCHMAKING_KIND:
            return self._matchmaking_command(raw)
        return self._playtest_command(raw)

    @staticmethod
    def _dependency_for(kind: str, raw: dict[str, Any]) -> str | None:
        if kind == "dependency.engine.start":
            return "engine"
        if kind == "dependency.pixi.prepare":
            return "pixi"
        if kind == "dependency.sync":
            dependency = str(raw.get("dependency", ""))
            return dependency if dependency in {"engine", "pixi"} else None
        if kind == "dependency.stack.prepare":
            return "stack"
        return None

    def _dependency_command(self, kind: str, raw: dict[str, Any]) -> tuple[list[str], str, None]:
        if kind == "dependency.stack.prepare":
            return (
                [
                    sys.executable,
                    "-m",
                    "deepdeck_learner.dependencies",
                    "bootstrap",
                    "--root",
                    str(self.root),
                ],
                "Local Engine + Pixi setup",
                None,
            )
        if kind == "dependency.engine.start":
            manifest = self.root / "external" / "deepdeck-engine" / "Cargo.toml"
            if not manifest.is_file():
                raise JobValidationError("DeepDeckEngine is missing. Synchronize it first.")
            if current_revision(self.root, "engine") != pinned_revision(self.root, "engine"):
                raise JobValidationError(
                    "DeepDeckEngine is not at the compatible revision. Sync it first."
                )
            cargo = shutil.which("cargo")
            if not cargo and not engine_build_current(self.root):
                raise JobValidationError("Install Rust and Cargo before starting DeepDeckEngine.")
            if engine_build_current(self.root):
                return (
                    [str(engine_binary(self.root))],
                    "DeepDeckEngine local server",
                    None,
                )
            assert cargo is not None
            return (
                [
                    cargo,
                    "run",
                    "--release",
                    "--manifest-path",
                    str(manifest),
                    "--locked",
                    "--bin",
                    "mtg-engine-server",
                ],
                "DeepDeckEngine local server",
                None,
            )
        if kind == "dependency.pixi.prepare":
            return (
                [
                    sys.executable,
                    "-m",
                    "deepdeck_learner.dependencies",
                    "prepare-pixi",
                    "--root",
                    str(self.root),
                ],
                "Prepare DeepDeckPixi",
                None,
            )
        dependency = str(raw.get("dependency", ""))
        if dependency not in {"engine", "pixi"}:
            raise JobValidationError("Dependency must be engine or pixi.")
        return (
            [
                sys.executable,
                "-m",
                "deepdeck_learner.dependencies",
                "sync",
                "--root",
                str(self.root),
                "--dependency",
                dependency,
            ],
            f"Sync DeepDeck{dependency.title()}",
            None,
        )

    def _training_command(self, kind: str, raw: dict[str, Any]) -> tuple[list[str], str, Path]:
        if kind == "training.v13-world-model":
            config = self.root / "configs" / "oracle-ai" / "v13-world-model-smoke.yaml"
            if not config.is_file():
                raise JobValidationError("The V13 world-model configuration is unavailable.")
            target = self.root / ".deepdeck" / "runs" / f"v13-world-model-{uuid.uuid4().hex[:8]}"
            return (
                [
                    sys.executable,
                    "-m",
                    "oracle_ai.training.world_model",
                    "--config",
                    str(config),
                    "--output",
                    str(target),
                ],
                "V13 structured world-model smoke",
                target,
            )
        model = str(raw.get("model", "v12"))
        if model not in {"v11", "v12"}:
            raise JobValidationError("Model must be v11 or v12.")
        if kind == "training.pool":
            return self._pool_training_command(model, raw)
        epochs = self._bounded_int(raw, "epochs", default=3, minimum=1, maximum=1000)
        seed = self._bounded_int(
            raw,
            "seed",
            default=self._random_seed(),
            minimum=0,
            maximum=MAX_RANDOM_SEED,
        )
        learning_rate = self._bounded_float(
            raw, "learning_rate", default=0.0003, minimum=1e-8, maximum=1.0
        )
        device = str(raw.get("device", "cuda"))
        if device != "cpu" and not device.startswith("cuda"):
            raise JobValidationError("Device must be cpu or a cuda device.")
        argv = [
            sys.executable,
            "-m",
            "deepdeck_examples.deep_learning.training",
            model,
        ]
        if kind == "training.smoke":
            argv.append("--smoke")
        else:
            dataset = Path(str(raw.get("dataset", ""))).expanduser().resolve()
            if not dataset.is_file() or dataset.suffix.lower() != ".jsonl":
                raise JobValidationError("Dataset must be an existing .jsonl file.")
            argv.extend(["--dataset", str(dataset)])
        target = self._new_checkpoint_target(model)
        argv.extend(
            [
                "--output",
                str(target),
                "--epochs",
                str(epochs),
                "--learning-rate",
                str(learning_rate),
                "--device",
                device,
                "--seed",
                str(seed),
            ]
        )
        resume = raw.get("resume")
        if resume:
            resume_path = Path(str(resume)).expanduser().resolve()
            if not (resume_path / "config.json").is_file():
                raise JobValidationError("Resume must be a checkpoint directory.")
            argv.extend(["--resume", str(resume_path)])
        return argv, f"{model.upper()} {'smoke' if kind.endswith('smoke') else 'dataset'}", target

    def _pool_training_command(
        self, model: str, raw: dict[str, Any]
    ) -> tuple[list[str], str, Path]:
        try:
            import yaml
        except ModuleNotFoundError as error:
            raise JobValidationError(
                "Install DeepDeckLearner's deep-learning dependencies."
            ) from error
        workers = self._bounded_int(raw, "parallel_matches", default=1, minimum=1, maximum=32)
        model_name = self._local_model_name(raw)
        reserve_playtest = bool(raw.get("reserve_playtest", True))
        pool_path = self.root / ".deepdeck" / "training-deck-pool.json"
        try:
            pool = json.loads(pool_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise JobValidationError("Create a training deck pool before starting.") from error
        selected = pool.get("decks", []) if isinstance(pool, dict) else []
        required_format = "legacy" if model == "v12" else "commander"
        compatible = [
            deck
            for deck in selected
            if isinstance(deck, dict) and deck.get("format") == required_format
        ]
        if not compatible:
            raise JobValidationError(
                f"{model.upper()} requires at least one {required_format} deck in the pool."
            )
        now = f"{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"
        slug = re.sub(r"[^a-z0-9]+", "-", model_name.casefold()).strip("-") or "local-model"
        model_id = f"{slug[:40]}-{now.lower()}"
        run = self.root / ".deepdeck" / "runs" / f"{now}-{slug[:40]}-{model}"
        run.mkdir(parents=True, exist_ok=False)
        template = (
            self.root
            / "configs"
            / "oracle-ai"
            / ("league-v12-legacy.yaml" if model == "v12" else "league-v11-alphastar.yaml")
        )
        config = yaml.safe_load(template.read_text(encoding="utf-8"))
        config["seed"] = self._random_seed()
        config["trainingSeed"] = self._random_seed()
        evaluation = config.get("evaluation")
        if isinstance(evaluation, dict):
            evaluation["seed"] = self._random_seed()
        engine_url = str(config.get("engineUrl", "http://127.0.0.1:8787"))
        catalog: dict[str, list[dict[str, Any]]] = {}
        for deck in compatible:
            version_id = str(deck["id"])
            snapshot_path = self.root / ".deepdeck" / "decks" / f"{version_id}.json"
            try:
                snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as error:
                raise JobValidationError(
                    f"Download {deck.get('name', version_id)} again before training."
                ) from error
            entries = [
                entry
                for entry in snapshot.get("cards", [])
                if isinstance(entry, dict) and entry.get("section") != "considering"
            ]
            entries = self._enrich_card_characteristics(entries)
            compiled_rules = self._compile_oracle_rules(engine_url, entries)
            cards: list[dict[str, Any]] = []
            for entry in entries:
                quantity = max(0, int(entry.get("quantity", 0)))
                card_id = self._card_identifier(entry)
                if not card_id:
                    raise JobValidationError(
                        f"{deck.get('name', version_id)} contains a card without an id."
                    )
                raw_rules = entry.get("rules")
                rules: list[Any]
                current_rules = compiled_rules.get(card_id)
                if isinstance(current_rules, list):
                    rules = current_rules
                elif isinstance(raw_rules, list):
                    rules = raw_rules
                else:
                    rules = []
                section = str(entry.get("section", "main"))
                is_token, is_game_piece = self._auxiliary_game_piece_flags(entry)
                normalized = {
                    "id": card_id,
                    "name": str(entry.get("name", "")),
                    "typeLine": str(entry.get("typeLine", "")),
                    "manaCost": str(entry.get("manaCost") or ""),
                    "oracleText": entry.get("oracleText"),
                    "layout": entry.get("layout"),
                    "faces": entry.get("faces", []),
                    "power": entry.get("power"),
                    "toughness": entry.get("toughness"),
                    "rules": rules,
                    "isToken": is_token,
                    "isGamePiece": is_game_piece,
                    "isSideboard": section in {"sideboard", "companion"},
                    "isCommander": section == "commander",
                    "sourceSessionId": version_id,
                }
                cards.extend(dict(normalized) for _ in range(quantity))
            name = f"{deck.get('name', 'Deck')} · v{deck.get('version', 1)} · {version_id[:8]}"
            catalog[name] = cards
        catalog_path = run / "training-decks.json"
        catalog_path.write_text(
            json.dumps(catalog, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        config.pop("deckSource", None)
        config.pop("metaLegacyDeckSelection", None)
        config["deckCatalog"] = str(catalog_path)
        config["outputDir"] = str(run)
        config["candidateModelName"] = model_id
        config["analyticsPilotId"] = model_id
        config["trainingParticipantId"] = model_id
        shared_self_play = bool(raw.get("self_play_all_seats", True))
        config["trainingOpponentMix"] = (
            {"self": 1.0} if shared_self_play else {"self": 0.5, "anchor": 0.5}
        )
        config["continuous"] = True
        config["parallelGameWorkers"] = workers
        config["rolloutBatchGames"] = workers
        # A V12 checkpoint includes both weights and Adam optimizer state and
        # can occupy hundreds of megabytes. Keep two resumable historical
        # points in addition to the live checkpoint by default.
        config["maxCheckpoints"] = self._bounded_int(
            raw, "max_checkpoints", default=2, minimum=1, maximum=3
        )
        resource_plan = {
            "trainingMatches": workers,
            "leagueMatches": 1,
            "localMatches": 1,
            "gpuMemoryMb": self._bounded_int(
                raw, "gpu_memory_mb", default=0, minimum=0, maximum=24 * 1024
            ),
        }
        resource_path = run / "learner-resources.json"
        resource_path.write_text(
            json.dumps(resource_plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        config["learnerResourcePlan"] = str(resource_path)
        config["gpuMemoryLimitMb"] = resource_plan["gpuMemoryMb"]
        config["modelEvaluationEnabled"] = False
        config["groundTruthEvaluationEnabled"] = False
        config["resumeLeagueState"] = False
        base_model = self._reference_checkpoint(model)
        if base_model is not None:
            config["resumeCheckpoint"] = str(base_model)
            config["resumeCheckpointOptional"] = False
            config["resumeOptimizer"] = False
        else:
            config.pop("resumeCheckpoint", None)
            config.pop("resumeCheckpointOptional", None)
            config.pop("resumeOptimizer", None)
        config["serviceRefreshEvery"] = 1 if reserve_playtest else 1_000_000
        config["learnerSettings"] = {
            "modelId": model_id,
            "modelName": model_name,
            "reservePlaytest": reserve_playtest,
            "selfPlayAllSeats": shared_self_play,
            "selectedDeckVersionIds": [str(deck["id"]) for deck in compatible],
        }
        config_path = run / "training-config.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        checkpoint_path = run / "live" / model_id
        metadata = {
            "schemaVersion": "local-model/v1",
            "id": model_id,
            "name": model_name,
            "architecture": model,
            "format": required_format,
            "description": (
                "Structured two-player Legacy policy with two relative value slots."
                if model == "v12"
                else "Structured Commander policy with four multiplayer value slots."
            ),
            "createdAt": utc_now(),
            "checkpointPath": str(checkpoint_path),
            "reservePlaytest": reserve_playtest,
            "selfPlayAllSeats": shared_self_play,
            "source": "user-trained",
            "baseCheckpointPath": str(base_model) if base_model is not None else None,
            "decks": compatible,
            "resourcePlan": resource_plan,
            "agentSdkRuntime": {
                "module": "deepdeck_examples.run",
                "arguments": [model],
                "checkpointArgument": "--checkpoint",
                "requiredFiles": ["manifest.json", "checkpoint.pt"],
            },
        }
        (run / "local-model.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        argv = [sys.executable, "-m", "oracle_ai.training.league", "--config", str(config_path)]
        deck_label = "deck" if len(compatible) == 1 else "decks"
        return argv, f"{model_name} · {model.upper()} · {len(compatible)} {deck_label}", run

    def _existing_pool_training_command(
        self, model_id: str, requested_phase: str | None = None
    ) -> tuple[list[str], str, Path]:
        try:
            import yaml
        except ModuleNotFoundError as error:
            raise JobValidationError(
                "Install DeepDeckLearner's deep-learning dependencies."
            ) from error
        try:
            run, metadata = find_model_run(self.root, model_id)
        except ValueError as error:
            raise JobValidationError(str(error)) from error
        if metadata.get("source") == "local-frozen-checkpoint":
            raise JobValidationError(
                "Create your own agent from V11 or V12 before starting training."
            )
        config_path = run / "training-config.yaml"
        try:
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise JobValidationError(
                "This agent's training configuration is unavailable."
            ) from error
        if not isinstance(config, dict):
            raise JobValidationError("This agent's training configuration is invalid.")
        if str(metadata.get("architecture", "")).casefold() == "v15":
            if requested_phase not in {None, "world-model", "typed-latent-world-model"}:
                raise JobValidationError(
                    f"V15 does not publish a training phase named {requested_phase}."
                )
            (run / "training-control.json").write_text(
                json.dumps({"desiredState": "running"}, indent=2) + "\n",
                encoding="utf-8",
            )
            return (
                [
                    sys.executable,
                    "-m",
                    "oracle_ai.training.world_model_v15",
                    "--config",
                    str(config_path),
                    "--output",
                    str(run),
                ],
                f"{metadata.get('name', model_id)} · V15 · typed latent world model",
                run,
            )
        if str(metadata.get("architecture", "")).casefold() == "v13":
            supported_phases = {
                "world-model",
                "reinforcement-learning",
                "engine-evaluation",
            }
            if requested_phase is not None and requested_phase not in supported_phases:
                raise JobValidationError(
                    f"V13 does not publish a training phase named {requested_phase}."
                )
            (run / "training-control.json").write_text(
                json.dumps({"desiredState": "running"}, indent=2) + "\n",
                encoding="utf-8",
            )
            training_config = config.get("training", {})
            world_steps = int(training_config.get("steps", 0) or 0)
            rl_config = config.get("rl", {})
            rl_episodes = int(rl_config.get("episodes", 10_000) or 0)
            curriculum_config = config.get("trainingCurriculum", {})
            curriculum_enabled = bool(
                isinstance(curriculum_config, dict) and curriculum_config.get("enabled", False)
            )
            rl_label = "adaptive scenario curriculum" if curriculum_enabled else "PPO self-play"
            world_checkpoint = run / "checkpoints" / f"step-{world_steps}" / "world-model.pt"
            available_world_checkpoint = world_checkpoint.is_file() or any(
                (run / "checkpoints").glob("step-*/world-model.pt")
            )
            extra_args: list[str] = []
            if requested_phase == "world-model":
                module = "oracle_ai.training.world_model"
                phase_label = "world model"
            elif requested_phase == "reinforcement-learning":
                if not available_world_checkpoint:
                    raise JobValidationError(
                        "Train a V13 world-model checkpoint before starting reinforcement learning."
                    )
                module = "oracle_ai.training.rl_v13"
                phase_label = rl_label
            elif requested_phase == "engine-evaluation":
                if not available_world_checkpoint:
                    raise JobValidationError(
                        "Train a V13 world-model checkpoint before starting Engine evaluation."
                    )
                if not (run / "v13-engine-baseline.pt").is_file():
                    raise JobValidationError(
                        "Run V13 Engine self-play once before starting a standalone evaluation."
                    )
                module = "oracle_ai.training.rl_v13"
                extra_args = ["--evaluate-only"]
                phase_label = "Engine evaluation"
            elif world_steps > 0 and not world_checkpoint.is_file():
                module = "oracle_ai.training.world_model"
                phase_label = "world model"
            elif rl_episodes > 0 and available_world_checkpoint:
                module = "oracle_ai.training.rl_v13"
                phase_label = rl_label
            elif rl_episodes > 0:
                raise JobValidationError(
                    "Train a V13 world-model checkpoint before starting reinforcement learning."
                )
            else:
                raise JobValidationError("Enable at least one unfinished V13 training stage.")
            return (
                [
                    sys.executable,
                    "-m",
                    module,
                    "--config",
                    str(config_path),
                    "--output",
                    str(run),
                    *extra_args,
                ],
                f"{metadata.get('name', model_id)} · V13 · {phase_label}",
                run,
            )
        checkpoint = Path(str(metadata.get("checkpointPath", "")))
        resumable = (checkpoint / "manifest.json").is_file() and (
            checkpoint / "checkpoint.pt"
        ).is_file()
        config["resumeLeagueState"] = (run / "league-state.json").is_file()
        if resumable:
            config["resumeCheckpoint"] = str(checkpoint)
            config["resumeCheckpointOptional"] = True
            config["resumeOptimizer"] = True
        elif metadata.get("baseCheckpointPath"):
            config["resumeCheckpoint"] = str(metadata["baseCheckpointPath"])
            config["resumeCheckpointOptional"] = False
            config["resumeOptimizer"] = False
        self._sanitize_training_catalog(run)
        plan = load_resource_plan(run)
        config["parallelGameWorkers"] = max(1, plan["trainingMatches"])
        config["rolloutBatchGames"] = max(1, plan["trainingMatches"])
        config["gpuMemoryLimitMb"] = plan["gpuMemoryMb"]
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        (run / "training-control.json").write_text(
            json.dumps({"desiredState": "running"}, indent=2) + "\n",
            encoding="utf-8",
        )
        decks = metadata.get("decks", [])
        deck_count = len(decks) if isinstance(decks, list) else 0
        architecture = str(metadata.get("architecture", "")).upper()
        return (
            [
                sys.executable,
                "-m",
                "oracle_ai.training.league",
                "--config",
                str(config_path),
            ],
            f"{metadata.get('name', model_id)} · {architecture} · {deck_count} decks",
            run,
        )

    @staticmethod
    def _local_model_name(raw: dict[str, Any]) -> str:
        name = str(raw.get("model_name", "")).strip()
        if not 2 <= len(name) <= 64 or any(ord(character) < 32 for character in name):
            raise JobValidationError("Model name must contain between 2 and 64 visible characters.")
        return name

    @staticmethod
    def _random_seed() -> int:
        return random.SystemRandom().randrange(MAX_RANDOM_SEED + 1)

    @staticmethod
    def _card_identifier(card: dict[str, Any]) -> str:
        return card_identifier(card)

    @staticmethod
    def _auxiliary_game_piece_flags(card: dict[str, Any]) -> tuple[bool, bool]:
        type_line = str(card.get("typeLine", "")).strip().casefold()
        section = str(card.get("section", "")).strip().casefold()
        inferred_piece = type_line.startswith(("token ", "emblem", "dungeon")) or section in {
            "token",
            "tokens",
            "emblem",
            "dungeon",
        }
        is_token = bool(card.get("isToken")) or type_line.startswith("token ")
        return is_token, bool(card.get("isGamePiece")) or inferred_piece or is_token

    def _sanitize_training_catalog(self, run: Path) -> dict[str, list[dict[str, Any]]] | None:
        catalog_path = run / "training-decks.json"
        try:
            catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(catalog, dict):
            return None
        changed = False
        for cards in catalog.values():
            if not isinstance(cards, list):
                continue
            for card in cards:
                if not isinstance(card, dict):
                    continue
                is_token, is_game_piece = self._auxiliary_game_piece_flags(card)
                if card.get("isToken") is not is_token:
                    card["isToken"] = is_token
                    changed = True
                if card.get("isGamePiece") is not is_game_piece:
                    card["isGamePiece"] = is_game_piece
                    changed = True
        if changed:
            catalog_path.write_text(
                json.dumps(catalog, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        return catalog

    def _enrich_card_characteristics(self, cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
        try:
            return enrich_card_characteristics(self.root, cards)
        except CardModelError as error:
            raise JobValidationError(str(error)) from error

    def _compile_oracle_rules(
        self, engine_url: str, cards: list[dict[str, Any]]
    ) -> dict[str, list[Any]]:
        try:
            return compile_oracle_rules(self.root, engine_url, cards)
        except CardModelError as error:
            raise JobValidationError(str(error)) from error

    def _refresh_playtest_deck_rules(
        self,
        engine_url: str,
        version_id: str,
        cards: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        try:
            return refresh_playtest_deck(self.root, engine_url, version_id, cards)
        except CardModelError as error:
            raise JobValidationError(str(error)) from error

    def _new_checkpoint_target(self, model: str) -> Path:
        self.checkpoint_root.mkdir(parents=True, exist_ok=True)
        base = self.checkpoint_root / f"{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{model}"
        target = base
        suffix = 2
        while target.exists():
            target = Path(f"{base}-{suffix}")
            suffix += 1
        return target

    def _playtest_command(self, raw: dict[str, Any]) -> tuple[list[str], str, None]:
        runtime_command, checkpoint_path, model_name, checkpoint_argument = self._owned_model(raw)
        engine_url = str(raw.get("engine_url", "http://127.0.0.1:8787"))
        if not is_loopback_url(engine_url):
            raise JobValidationError("Local playtesting requires a loopback Engine URL.")
        own_deck = str(raw.get("deck_version_id", "")).strip()
        opponent_deck = str(raw.get("opponent_deck_version_id", "")).strip()
        if not own_deck or not opponent_deck:
            raise JobValidationError("Choose a deck or use the random option for both seats.")
        game_format = str(raw.get("format", "legacy"))
        if game_format not in {"legacy", "commander"}:
            raise JobValidationError("Format must be legacy or commander.")
        run = checkpoint_path.parent.parent
        catalog = self._sanitize_training_catalog(run)
        if catalog is None:
            try:
                catalog = self._materialize_playtest_catalog(run, game_format)
            except JobValidationError:
                raise
            except (OSError, ValueError) as fallback_error:
                raise JobValidationError(
                    "This model's local training deck catalog is unavailable."
                ) from fallback_error
        configured_version_ids = self._playtest_configured_deck_ids(run)
        eligible_catalog = {
            str(deck_name): cards
            for deck_name, cards in catalog.items()
            if isinstance(cards, list)
            and (
                not configured_version_ids
                or any(
                    isinstance(card, dict)
                    and str(card.get("sourceSessionId", "")) in configured_version_ids
                    for card in cards
                )
            )
        }

        def selected_deck(version_id: str) -> tuple[str, list[dict[str, Any]]]:
            for deck_name, cards in eligible_catalog.items():
                if isinstance(cards, list) and any(
                    isinstance(card, dict) and card.get("sourceSessionId") == version_id
                    for card in cards
                ):
                    return str(deck_name), cards
            raise JobValidationError("Choose decks that belong to this model's training pool.")

        available = [
            (str(deck_name), cards)
            for deck_name, cards in eligible_catalog.items()
            if isinstance(cards, list)
            and any(isinstance(card, dict) and card.get("sourceSessionId") for card in cards)
        ]
        if not available:
            raise JobValidationError("This model has no playable deck in its training pool.")

        def version_id(cards: list[dict[str, Any]]) -> str:
            return str(
                next(card["sourceSessionId"] for card in cards if card.get("sourceSessionId"))
            )

        randomizer = random.SystemRandom()
        if own_deck == "random":
            own_name, own_cards = randomizer.choice(available)
            own_deck = version_id(own_cards)
        else:
            own_name, own_cards = selected_deck(own_deck)

        if opponent_deck == "random":
            candidates = [
                candidate for candidate in available if version_id(candidate[1]) != own_deck
            ] or available
            stats_by_id = {
                str(item.get("deckVersionId")): item
                for item in self._deck_statistics_for_model(str(raw.get("model_id", "")))
            }
            target = stats_by_id.get(own_deck, {"ordinal": 0.0, "games": 0})
            config = self._playtest_matchmaking_config(run)
            weights = [
                plackett_luce_matchmaking_weight(
                    stats_by_id.get(version_id(cards), {"ordinal": 0.0, "games": 0}),
                    target_ordinal=float(target.get("ordinal", 0.0) or 0.0),
                    minimum_games=int(target.get("games", 0) or 0),
                    random_floor=float(config.get("randomFloor", 0.2)),
                    rating_scale=float(config.get("ratingScale", 10.0)),
                    underplayed_strength=float(config.get("underplayedStrength", 0.35)),
                    game_prior=float(config.get("matchPrior", 10.0)),
                )
                for _, cards in candidates
            ]
            opponent_name, opponent_cards = randomizer.choices(candidates, weights=weights, k=1)[0]
            opponent_deck = version_id(opponent_cards)
        else:
            opponent_name, opponent_cards = selected_deck(opponent_deck)
        own_cards = self._refresh_playtest_deck_rules(engine_url, own_deck, own_cards)
        opponent_cards = self._refresh_playtest_deck_rules(
            engine_url, opponent_deck, opponent_cards
        )
        setup_path = self.root / ".deepdeck" / "playtests" / f"{uuid.uuid4()}.json"
        setup_path.parent.mkdir(parents=True, exist_ok=True)
        starting_life = 40 if game_format == "commander" else 20
        setup_path.write_text(
            json.dumps(
                {
                    "setup": {
                        "openingHandSize": 7,
                        "startingPlayer": 0,
                        "players": [
                            {
                                "id": "local-human",
                                "name": own_name,
                                "startingLife": starting_life,
                                "cards": own_cards,
                            },
                            {
                                "id": "local-agent",
                                "name": opponent_name,
                                "startingLife": starting_life,
                                "cards": opponent_cards,
                            },
                        ],
                    },
                    "seed": self._random_seed(),
                    "gameMode": game_format,
                    "maxTurns": 200,
                    "humanPlayerIds": ["local-human"],
                    "analyticsDeckSessionByPlayerId": {
                        "local-human": own_deck,
                        "local-agent": opponent_deck,
                    },
                    "mulliganEnabled": True,
                    "freeMulligans": 1 if game_format == "commander" else 0,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        argv = [
            *runtime_command,
            "--target",
            "local",
            "--engine-url",
            engine_url,
            "--start-local-game",
            "--local-game-setup",
            str(setup_path),
            "--local-format",
            game_format,
        ]
        argv.extend([checkpoint_argument, str(checkpoint_path)])
        raw["_job_details"] = {
            "engineUrl": engine_url,
            "playerDeck": {"id": own_deck, "name": own_name},
            "opponentDeck": {"id": opponent_deck, "name": opponent_name},
            "selectionOrder": "player-then-rating-proximity",
        }
        return argv, f"{model_name} · local {game_format}", None

    def _materialize_playtest_catalog(
        self, run: Path, game_format: str
    ) -> dict[str, list[dict[str, Any]]]:
        try:
            metadata = json.loads((run / "local-model.json").read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise JobValidationError("The selected local model metadata is unavailable.") from error
        configured = metadata.get("decks", []) if isinstance(metadata, dict) else []
        version_ids = [
            str(deck.get("id", ""))
            for deck in configured
            if isinstance(deck, dict) and deck.get("id")
        ]
        if not version_ids:
            try:
                resolved = json.loads((run / "resolved-config.json").read_text(encoding="utf-8"))
                raw_ids = resolved.get("learnerSettings", {}).get("selectedDeckVersionIds", [])
                version_ids = [str(value) for value in raw_ids]
            except (AttributeError, OSError, ValueError):
                version_ids = []
        catalog: dict[str, list[dict[str, Any]]] = {}
        for version_id in version_ids:
            try:
                snapshot = json.loads(
                    (self.root / ".deepdeck" / "decks" / f"{version_id}.json").read_text(
                        encoding="utf-8"
                    )
                )
            except (OSError, ValueError):
                continue
            if not isinstance(snapshot, dict) or str(snapshot.get("format", "")) != game_format:
                continue
            cards: list[dict[str, Any]] = []
            for entry in snapshot.get("cards", []):
                if not isinstance(entry, dict) or entry.get("section") == "considering":
                    continue
                card_id = self._card_identifier(entry)
                if not card_id:
                    continue
                is_token, is_game_piece = self._auxiliary_game_piece_flags(entry)
                normalized = {
                    **entry,
                    "id": card_id,
                    "isToken": is_token,
                    "isGamePiece": is_game_piece,
                    "isSideboard": entry.get("section") in {"sideboard", "companion"},
                    "isCommander": entry.get("section") == "commander",
                    "sourceSessionId": version_id,
                    "rules": entry.get("rules", []),
                }
                cards.extend(dict(normalized) for _ in range(max(0, int(entry.get("quantity", 0)))))
            if cards:
                name = (
                    f"{snapshot.get('name', 'Deck')} · v{snapshot.get('version', 1)} · "
                    f"{version_id[:8]}"
                )
                catalog[name] = cards
        if not catalog:
            raise JobValidationError("This model has no downloaded deck in its training pool.")
        (run / "training-decks.json").write_text(
            json.dumps(catalog, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return catalog

    def _reference_checkpoint(self, architecture: str) -> Path | None:
        runs = self.root / ".deepdeck" / "runs"
        for metadata_path in runs.glob("*/local-model.json"):
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            checkpoint = Path(str(metadata.get("checkpointPath", "")))
            if (
                metadata.get("source") == "local-frozen-checkpoint"
                and metadata.get("architecture") == architecture
                and (checkpoint / "manifest.json").is_file()
                and (checkpoint / "checkpoint.pt").is_file()
            ):
                return checkpoint
        return None

    def _deck_statistics_for_model(self, model_id: str) -> list[dict[str, Any]]:
        from .models import deck_statistics

        return [item for item in deck_statistics(self.root) if item.get("modelId") == model_id]

    @staticmethod
    def _playtest_matchmaking_config(run: Path) -> dict[str, Any]:
        try:
            resolved = json.loads((run / "resolved-config.json").read_text(encoding="utf-8"))
            config = resolved.get("trainingScenarioRandomizer", {}).get("matchmaking", {})
            return config if isinstance(config, dict) else {}
        except (AttributeError, OSError, ValueError):
            return {}

    @staticmethod
    def _playtest_configured_deck_ids(run: Path) -> set[str]:
        """Return the agent's current pool, not a stale trainer resolution by deck name."""
        try:
            metadata = json.loads((run / "local-model.json").read_text(encoding="utf-8"))
            decks = metadata.get("decks", [])
            ids = {
                str(deck.get("id")) for deck in decks if isinstance(deck, dict) and deck.get("id")
            }
            if ids:
                return ids
        except (OSError, ValueError):
            pass
        try:
            import yaml

            config = yaml.safe_load((run / "training-config.yaml").read_text(encoding="utf-8"))
            raw_ids = config.get("learnerSettings", {}).get("selectedDeckVersionIds", [])
            return {str(value) for value in raw_ids} if isinstance(raw_ids, list) else set()
        except (AttributeError, OSError, ValueError):
            return set()

    def _matchmaking_command(self, raw: dict[str, Any]) -> tuple[list[str], str, None]:
        if not os.getenv("DEEPDECK_API_KEY"):
            raise JobValidationError(
                "Add DEEPDECK_API_KEY to the project .env and restart DeepDeckLearner."
            )
        runtime_command, checkpoint_path, model_name, checkpoint_argument = self._owned_model(raw)
        competition = str(raw.get("competition_version_id", "")).strip()
        deck_ids = raw.get("deck_version_ids")
        decks = (
            [str(deck).strip() for deck in deck_ids if str(deck).strip()]
            if isinstance(deck_ids, list)
            else [str(raw.get("deck_version_id", "")).strip()]
        )
        if not competition or not decks or not decks[0]:
            raise JobValidationError("Choose an active competition and a deck.")
        connections = self._bounded_int(raw, "connections", default=1, minimum=1, maximum=32)
        speed = str(raw.get("speed", "1s"))
        if speed not in {"100ms", "1s", "10s"}:
            raise JobValidationError("Speed must be 100ms, 1s, or 10s.")
        if str(raw.get("agent", "")) in {"v11", "v12"} and speed == "100ms":
            speed = "1s"
        argv = [
            *runtime_command,
            "--target",
            "ddl",
            "--speed",
            speed,
            "--competition-version-id",
            competition,
            "--deck-version-id",
            decks[0],
            "--matchmaking-concurrency",
            str(connections),
        ]
        for deck in decks[1:]:
            argv.extend(["--additional-deck-version-id", deck])
        if not bool(raw.get("continuous", False)):
            argv.append("--once")
        argv.extend([checkpoint_argument, str(checkpoint_path)])
        return argv, f"{model_name} · Deep Deck League", None

    def _owned_model(self, raw: dict[str, Any]) -> tuple[list[str], Path, str, str]:
        model_id = str(raw.get("model_id", "")).strip()
        try:
            _run, metadata = find_model_run(self.root, model_id)
        except ValueError as error:
            raise JobValidationError("The selected local model metadata is unavailable.") from error
        architecture = str(metadata.get("architecture", ""))
        checkpoint = Path(str(metadata.get("checkpointPath", ""))).expanduser().resolve()
        runs_root = (self.root / ".deepdeck" / "runs").resolve()
        try:
            checkpoint.relative_to(runs_root)
        except ValueError as error:
            raise JobValidationError("Choose a model created by this Learner workspace.") from error
        if metadata.get("source") == "local-frozen-checkpoint":
            raise JobValidationError(
                "Choose an agent trained in this workspace, not a reference implementation."
            )
        runtime = metadata.get("agentSdkRuntime")
        if isinstance(runtime, dict):
            module = runtime.get("module")
            arguments = runtime.get("arguments", [])
            checkpoint_argument = runtime.get("checkpointArgument", "--checkpoint")
            required_files = runtime.get("requiredFiles", [])
            if (
                not isinstance(module, str)
                or not re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", module)
                or not isinstance(arguments, list)
                or not all(isinstance(value, str) for value in arguments)
                or not isinstance(checkpoint_argument, str)
                or not checkpoint_argument.startswith("--")
                or not isinstance(required_files, list)
                or not required_files
                or not all(isinstance(value, str) for value in required_files)
            ):
                raise JobValidationError("This model's Agent SDK runtime descriptor is invalid.")
        else:
            # Models created before the SDK runtime contract keep working.
            module = "deepdeck_examples.run"
            arguments = [architecture]
            checkpoint_argument = "--checkpoint"
            required_files = (
                ["rl-model.pt"] if architecture == "v13" else ["manifest.json", "checkpoint.pt"]
            )
        checkpoint_ready = all(
            not Path(value).is_absolute()
            and ".." not in Path(value).parts
            and (checkpoint / value).is_file()
            for value in required_files
        )
        if not checkpoint_ready:
            raise JobValidationError("This model is still preparing its first playable weights.")
        command = [sys.executable, "-m", module, *arguments]
        return (
            command,
            checkpoint,
            str(metadata.get("name", architecture.upper())),
            checkpoint_argument,
        )

    @staticmethod
    def _bounded_int(
        raw: dict[str, Any], key: str, *, default: int, minimum: int, maximum: int
    ) -> int:
        try:
            value = int(raw.get(key, default))
        except (TypeError, ValueError) as error:
            raise JobValidationError(f"{key} must be an integer.") from error
        if not minimum <= value <= maximum:
            raise JobValidationError(f"{key} must be between {minimum} and {maximum}.")
        return value

    @staticmethod
    def _bounded_float(
        raw: dict[str, Any], key: str, *, default: float, minimum: float, maximum: float
    ) -> float:
        try:
            value = float(raw.get(key, default))
        except (TypeError, ValueError) as error:
            raise JobValidationError(f"{key} must be a number.") from error
        if not minimum <= value <= maximum:
            raise JobValidationError(f"{key} must be between {minimum} and {maximum}.")
        return value

    def _run(self, job: Job) -> None:
        job.status = "running"
        job.started_at = utc_now()
        self._persist(job)
        try:
            if job.kind == "training.pool" and job.payload is not None:
                job.logs.append("Preparing local cards and Oracle rules for training.")
                self._persist(job)
                try:
                    argv, label, artifact = self._training_command(job.kind, job.payload)
                except (JobValidationError, httpx.HTTPError, OSError, ValueError) as error:
                    job.logs.append(f"Unable to prepare training: {error}")
                    job.status = "failed"
                    job.exit_code = -1
                    return
                job.argv = argv
                job.label = label
                job.artifact_path = str(artifact)
                metadata = json.loads((artifact / "local-model.json").read_text(encoding="utf-8"))
                job.model_id = str(metadata["id"])
                job.payload = None
                self._persist(job)
            environment = self._child_environment()
            if job.kind in {"dependency.engine.start", "dependency.stack.prepare"}:
                engine_environment = self._engine_environment()
                environment.update(engine_environment)
                if engine_environment:
                    job.logs.append("Using Rust's bundled linker for the local Windows build.")
            if job.kind == "training.pool" and job.artifact_path:
                environment["PYTHONUNBUFFERED"] = "1"
                job.exit_code = self._run_durable_training_process(job, environment)
            else:
                process = subprocess.Popen(
                    job.argv,
                    cwd=self.root,
                    env=environment,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    shell=False,
                )
                job.process = process
                assert process.stdout is not None
                for line in process.stdout:
                    clean = line.rstrip()
                    job.logs.append(clean)
                    self._consume_process_marker(job, clean)
                job.exit_code = process.wait()
            job.status = (
                "stopped" if job.stop_requested else "completed" if job.exit_code == 0 else "failed"
            )
        except OSError as error:
            job.logs.append(f"Unable to start process: {error}")
            job.status = "failed"
            job.exit_code = -1
        finally:
            job.process = None
            job.finished_at = utc_now()
            self._persist(job)

    def _consume_process_marker(self, job: Job, clean: str) -> None:
        playtest_marker = "DEEPDECK_PLAYTEST_SESSION "
        if clean.startswith(playtest_marker):
            try:
                session = json.loads(clean[len(playtest_marker) :])
            except ValueError:
                session = None
            if isinstance(session, dict):
                job.details = {**(job.details or {}), **session}
                self._persist(job)
            return

        league_marker = "DEEPDECK_LEAGUE_MATCH "
        if not clean.startswith(league_marker):
            return
        try:
            match = json.loads(clean[len(league_marker) :])
        except ValueError:
            match = None
        if not isinstance(match, dict) or not match.get("matchId"):
            return
        details = dict(job.details or {})
        matches = {
            str(item.get("matchId")): item
            for item in details.get("leagueMatches", [])
            if isinstance(item, dict) and item.get("matchId")
        }
        match_id = str(match["matchId"])
        if match.get("status") in {"complete", "failed", "cancelled"}:
            matches.pop(match_id, None)
        else:
            matches[match_id] = match
        details["leagueMatches"] = list(matches.values())
        job.details = details
        self._persist(job)

    def _run_durable_training_process(self, job: Job, environment: dict[str, str]) -> int:
        log_path = Path(str(job.artifact_path)) / "learner-process.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        offset = log_path.stat().st_size if log_path.is_file() else 0
        with log_path.open("a", encoding="utf-8") as output:
            process = subprocess.Popen(
                job.argv,
                cwd=self.root,
                env=environment,
                stdout=output,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                shell=False,
            )
            job.process = process
            while True:
                try:
                    exit_code = process.wait(timeout=0.5)
                except subprocess.TimeoutExpired:
                    exit_code = None
                offset = self._append_process_log(job, log_path, offset)
                if exit_code is not None:
                    return exit_code

    @staticmethod
    def _append_process_log(job: Job, path: Path, offset: int) -> int:
        with path.open("r", encoding="utf-8", errors="replace") as source:
            source.seek(offset)
            for line in source:
                job.logs.append(line.rstrip())
            return source.tell()

    def _child_environment(self) -> dict[str, str]:
        allowed = {
            "PATH",
            "Path",
            "PYTHONPATH",
            "SYSTEMROOT",
            "SystemRoot",
            "TEMP",
            "TMP",
            "CUDA_VISIBLE_DEVICES",
            "HOME",
            "LOCALAPPDATA",
            "USERPROFILE",
            "ProgramData",
            "ProgramFiles",
            "ProgramFiles(x86)",
            "ProgramW6432",
            "DEEPDECK_API_KEY",
            "DEEPDECK_PLATFORM_URL",
            "DDL_PLATFORM_API_URL",
        }
        environment = {key: value for key, value in os.environ.items() if key in allowed}
        sdk_source = str((self.root / "external" / "deepdeck-agent" / "src").resolve())
        existing = environment.get("PYTHONPATH", "")
        environment["PYTHONPATH"] = (
            sdk_source if not existing else os.pathsep.join((sdk_source, existing))
        )
        return environment

    @staticmethod
    def _engine_environment() -> dict[str, str]:
        if os.name != "nt" or shutil.which("link"):
            return {}
        rustc = shutil.which("rustc")
        if not rustc:
            return {}
        result = subprocess.run(
            [rustc, "--print", "sysroot"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
            shell=False,
        )
        sysroot = Path(result.stdout.strip()) if result.returncode == 0 else None
        linker = (
            sysroot / "lib" / "rustlib" / "x86_64-pc-windows-msvc" / "bin" / "rust-lld.exe"
            if sysroot
            else None
        )
        if not linker or not linker.is_file():
            return {}
        return {"RUSTFLAGS": f"-C linker={linker}"}
