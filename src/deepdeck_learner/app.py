from __future__ import annotations

import json
import os
import secrets
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from oracle_ai.training.curriculum import curriculum_catalog

from .catalogs import (
    CatalogAuthenticationError,
    CatalogError,
    CatalogNotFoundError,
    account_api_key,
    active_competitions,
    cached_deck_download,
    cached_decks,
    download_platform_deck,
    local_deck_presentation,
    local_legal_decks,
    platform_decks,
    scryfall_card_names,
    scryfall_image,
)
from .jobs import JobManager, JobValidationError
from .models import deck_statistics, local_models, training_statistics
from .resources import (
    delete_model_run,
    find_model_run,
    load_resource_plan,
    save_resource_plan,
)
from .settings import load_api_key, save_api_key
from .status import capability_status, project_root
from .training_platform import (
    acquire_replay_lease,
    list_saved_replays,
    load_agent_training_contract,
    load_saved_replay,
    load_training_settings,
    release_replay_lease,
    renew_replay_lease,
    save_agent_training_control,
    save_replay_forever,
    save_training_settings,
    training_evidence,
)


def create_app(root: Path | None = None) -> FastAPI:
    resolved_root = (root or project_root()).resolve()
    load_api_key(resolved_root)
    manager = JobManager(resolved_root)
    session_token = secrets.token_urlsafe(32)
    app = FastAPI(title="DeepDeckLearner local controller", version="1.0")
    app.state.manager = manager
    app.state.session_token = session_token
    account_cache: dict[str, Any] = {"key": None, "checked": 0.0, "value": None}

    def authorize(value: str | None) -> None:
        if not value or not secrets.compare_digest(value, session_token):
            raise HTTPException(status_code=403, detail="Invalid local session token.")

    @app.get("/api/v1/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/v1/session")
    def session() -> dict[str, str]:
        return {"token": session_token}

    @app.get("/api/v1/status")
    def status(engine_url: str = "http://127.0.0.1:8787") -> dict[str, Any]:
        return capability_status(resolved_root, engine_url)

    @app.get("/api/v1/account/status")
    def account_status() -> dict[str, Any]:
        key = os.getenv("DEEPDECK_API_KEY", "").strip()
        if not key:
            return {"configured": False, "valid": False, "reason": "Add an agent API key."}
        now = time.monotonic()
        cached = account_cache.get("value")
        if account_cache.get("key") == key and isinstance(cached, dict) and now - float(
            account_cache.get("checked", 0.0)
        ) < 60:
            return cached
        try:
            active_competitions()
            value = {"configured": True, "valid": True, "reason": "League account connected."}
        except CatalogAuthenticationError as error:
            value = {"configured": True, "valid": False, "reason": str(error)}
        except CatalogError as error:
            value = {"configured": True, "valid": None, "reason": str(error)}
        account_cache.update({"key": key, "checked": now, "value": value})
        return value

    @app.post("/api/v1/settings/api-key")
    def configure_api_key(
        payload: dict[str, Any], x_deepdeck_token: str | None = Header(default=None)
    ) -> dict[str, bool]:
        authorize(x_deepdeck_token)
        try:
            save_api_key(resolved_root, str(payload.get("api_key", "")))
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        account_cache.update({"key": None, "checked": 0.0, "value": None})
        return {"configured": True}

    @app.get("/api/v1/jobs")
    def jobs() -> list[dict[str, Any]]:
        return manager.list_jobs()

    @app.get("/api/v1/models")
    def models() -> dict[str, Any]:
        return {"items": local_models(resolved_root, manager.list_jobs())}

    @app.post("/api/v1/models", status_code=201)
    def create_model(
        payload: dict[str, Any], x_deepdeck_token: str | None = Header(default=None)
    ) -> dict[str, Any]:
        authorize(x_deepdeck_token)
        try:
            model_id = manager.prepare_model(payload)
        except (JobValidationError, OSError, ValueError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return next(
            model
            for model in local_models(resolved_root, manager.list_jobs())
            if model.get("id") == model_id
        )

    @app.delete("/api/v1/models/{model_id}")
    def delete_model(
        model_id: str, x_deepdeck_token: str | None = Header(default=None)
    ) -> dict[str, Any]:
        authorize(x_deepdeck_token)
        if manager.model_has_active_workers(model_id):
            raise HTTPException(
                status_code=409,
                detail="Stop this agent's active jobs before deleting its files.",
            )
        try:
            return delete_model_run(resolved_root, model_id)
        except ValueError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @app.put("/api/v1/models/{model_id}")
    def update_model(
        model_id: str,
        payload: dict[str, Any],
        x_deepdeck_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        authorize(x_deepdeck_token)
        try:
            updated_id = manager.update_model(model_id, payload)
        except (JobValidationError, OSError, ValueError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return next(
            model
            for model in local_models(resolved_root, manager.list_jobs())
            if model.get("id") == updated_id
        )

    @app.get("/api/v1/models/{model_id}/resources")
    def model_resources(model_id: str) -> dict[str, int]:
        try:
            run, _ = find_model_run(resolved_root, model_id)
        except ValueError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        return load_resource_plan(run)

    @app.put("/api/v1/models/{model_id}/resources")
    def update_model_resources(
        model_id: str,
        payload: dict[str, Any],
        x_deepdeck_token: str | None = Header(default=None),
    ) -> dict[str, int]:
        authorize(x_deepdeck_token)
        try:
            return save_resource_plan(resolved_root, model_id, payload)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.get("/api/v1/resources")
    def resources() -> dict[str, Any]:
        return manager.resources()

    @app.get("/api/v1/statistics/decks")
    def training_deck_statistics() -> dict[str, Any]:
        return {"items": deck_statistics(resolved_root)}

    @app.get("/api/v1/statistics/training")
    def local_training_statistics(window: str = "200") -> dict[str, Any]:
        try:
            return {"items": training_statistics(resolved_root, window)}
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.get("/api/v1/training/curriculum")
    def training_curriculum() -> dict[str, Any]:
        return curriculum_catalog()

    @app.get("/api/v1/models/{model_id}/training-settings")
    def model_training_settings(model_id: str) -> dict[str, Any]:
        try:
            return load_training_settings(resolved_root, model_id)
        except ValueError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @app.put("/api/v1/models/{model_id}/training-settings")
    def update_model_training_settings(
        model_id: str,
        payload: dict[str, Any],
        x_deepdeck_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        authorize(x_deepdeck_token)
        if manager.model_has_active_workers(model_id):
            raise HTTPException(
                status_code=409,
                detail="Stop this agent before changing its training stages or cadence.",
            )
        try:
            return save_training_settings(resolved_root, model_id, payload)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.get("/api/v1/models/{model_id}/evidence")
    def model_training_evidence(model_id: str) -> dict[str, Any]:
        try:
            return training_evidence(resolved_root, model_id)
        except ValueError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @app.get("/api/v1/models/{model_id}/training-contract")
    def model_training_contract(model_id: str) -> dict[str, Any]:
        try:
            return load_agent_training_contract(resolved_root, model_id)
        except (FileNotFoundError, ValueError) as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @app.put("/api/v1/models/{model_id}/training-control")
    def update_model_training_control(
        model_id: str,
        payload: dict[str, Any],
        x_deepdeck_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        authorize(x_deepdeck_token)
        try:
            return save_agent_training_control(resolved_root, model_id, payload)
        except FileNotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.get("/api/v1/models/{model_id}/replays")
    def model_replays(model_id: str) -> dict[str, Any]:
        try:
            return {"items": list_saved_replays(resolved_root, model_id)}
        except ValueError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @app.get("/api/v1/models/{model_id}/replays/{replay_id}")
    def model_replay(model_id: str, replay_id: str) -> dict[str, Any]:
        try:
            return load_saved_replay(resolved_root, model_id, replay_id)
        except FileNotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.post("/api/v1/models/{model_id}/replays/{replay_id}/leases")
    def acquire_model_replay_lease(
        model_id: str,
        replay_id: str,
        x_deepdeck_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        authorize(x_deepdeck_token)
        try:
            return acquire_replay_lease(resolved_root, model_id, replay_id)
        except FileNotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.put("/api/v1/models/{model_id}/replays/{replay_id}/leases/{lease_id}")
    def renew_model_replay_lease(
        model_id: str,
        replay_id: str,
        lease_id: str,
        x_deepdeck_token: str | None = Header(default=None),
    ) -> dict[str, int]:
        authorize(x_deepdeck_token)
        try:
            return renew_replay_lease(resolved_root, model_id, replay_id, lease_id)
        except FileNotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.delete("/api/v1/models/{model_id}/replays/{replay_id}/leases/{lease_id}")
    def release_model_replay_lease(
        model_id: str,
        replay_id: str,
        lease_id: str,
        x_deepdeck_token: str | None = Header(default=None),
    ) -> dict[str, bool]:
        authorize(x_deepdeck_token)
        try:
            return release_replay_lease(resolved_root, model_id, replay_id, lease_id)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.post("/api/v1/models/{model_id}/replays/{replay_id}/save")
    def permanently_save_model_replay(
        model_id: str,
        replay_id: str,
        x_deepdeck_token: str | None = Header(default=None),
    ) -> dict[str, bool]:
        authorize(x_deepdeck_token)
        try:
            return save_replay_forever(resolved_root, model_id, replay_id)
        except FileNotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.get("/api/v1/games")
    def active_local_games() -> dict[str, Any]:
        return {"items": manager.games()}

    @app.post("/api/v1/games/{game_id}/stop")
    def stop_game(
        game_id: str, x_deepdeck_token: str | None = Header(default=None)
    ) -> dict[str, Any]:
        authorize(x_deepdeck_token)
        try:
            found = manager.cancel_game(game_id)
        except JobValidationError as error:
            raise HTTPException(status_code=502, detail=str(error)) from error
        if not found:
            raise HTTPException(status_code=404, detail="Active game not found.")
        return found

    @app.get("/api/v1/catalog/decks")
    def deck_catalog(search: str = "", format: str = "legacy", page: int = 1) -> dict[str, Any]:
        try:
            account_api_key()
            return platform_decks(search, format, max(1, page))
        except CatalogAuthenticationError as error:
            local = cached_decks(resolved_root, search, format, max(1, page))
            if local["items"]:
                return local
            raise HTTPException(status_code=401, detail=str(error)) from error
        except CatalogError as error:
            raise HTTPException(status_code=502, detail=str(error)) from error

    @app.post("/api/v1/catalog/decks/{version_id}/download")
    def download_deck(
        version_id: str, x_deepdeck_token: str | None = Header(default=None)
    ) -> dict[str, Any]:
        authorize(x_deepdeck_token)
        local = cached_deck_download(resolved_root, version_id)
        if local is not None:
            return local
        try:
            return download_platform_deck(resolved_root, version_id)
        except CatalogAuthenticationError as error:
            raise HTTPException(status_code=401, detail=str(error)) from error
        except CatalogError as error:
            raise HTTPException(status_code=502, detail=str(error)) from error

    @app.get("/api/v1/catalog/decks/{version_id}/presentation")
    def deck_presentation(version_id: str) -> dict[str, Any]:
        try:
            return local_deck_presentation(resolved_root, version_id)
        except CatalogNotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except CatalogError as error:
            raise HTTPException(status_code=500, detail=str(error)) from error

    @app.get("/api/scryfall-images/{image_path:path}", include_in_schema=False)
    def card_image(image_path: str) -> Response:
        try:
            content, content_type = scryfall_image(image_path)
        except CatalogError as error:
            raise HTTPException(status_code=502, detail=str(error)) from error
        return Response(
            content=content,
            media_type=content_type,
            headers={"Cache-Control": "public, max-age=604800, immutable"},
        )

    @app.get("/api/v1/catalog/cards/autocomplete")
    def card_name_autocomplete(query: str = "") -> dict[str, list[str]]:
        try:
            return {"data": scryfall_card_names(query)}
        except CatalogError as error:
            raise HTTPException(status_code=502, detail=str(error)) from error

    @app.get("/api/v1/training/deck-pool")
    def training_deck_pool() -> dict[str, Any]:
        path = resolved_root / ".deepdeck" / "training-deck-pool.json"
        if not path.is_file():
            return {"decks": []}
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {"decks": []}
        return value if isinstance(value, dict) else {"decks": []}

    @app.put("/api/v1/training/deck-pool")
    def save_training_deck_pool(
        payload: dict[str, Any], x_deepdeck_token: str | None = Header(default=None)
    ) -> dict[str, Any]:
        authorize(x_deepdeck_token)
        decks = payload.get("decks", [])
        if (
            not isinstance(decks, list)
            or len(decks) > 100
            or not all(isinstance(deck, dict) and isinstance(deck.get("id"), str) for deck in decks)
        ):
            raise HTTPException(
                status_code=422, detail="A training pool may contain up to 100 valid decks."
            )
        path = resolved_root / ".deepdeck" / "training-deck-pool.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        pending = path.with_suffix(".pending")
        value = {"decks": decks}
        pending.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        pending.replace(path)
        return value

    @app.get("/api/v1/catalog/competitions")
    def competition_catalog() -> dict[str, Any]:
        try:
            account_api_key()
            return active_competitions()
        except CatalogAuthenticationError as error:
            raise HTTPException(status_code=401, detail=str(error)) from error
        except CatalogError as error:
            raise HTTPException(status_code=502, detail=str(error)) from error

    @app.get("/api/v1/catalog/local-decks")
    def local_deck_catalog(
        format: str = "legacy", engine_url: str = "http://127.0.0.1:8787"
    ) -> list[dict[str, str]]:
        try:
            return local_legal_decks(engine_url, format)
        except CatalogError as error:
            raise HTTPException(status_code=502, detail=str(error)) from error

    @app.get("/api/v1/jobs/{job_id}")
    def job(job_id: str) -> dict[str, Any]:
        found = manager.get(job_id)
        if not found:
            raise HTTPException(status_code=404, detail="Job not found.")
        return found

    @app.post("/api/v1/jobs", status_code=202)
    def start_job(
        payload: dict[str, Any], x_deepdeck_token: str | None = Header(default=None)
    ) -> dict[str, Any]:
        authorize(x_deepdeck_token)
        try:
            return manager.create(payload)
        except JobValidationError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.post("/api/v1/jobs/{job_id}/stop")
    def stop_job(
        job_id: str, x_deepdeck_token: str | None = Header(default=None)
    ) -> dict[str, Any]:
        authorize(x_deepdeck_token)
        found = manager.stop(job_id)
        if not found:
            raise HTTPException(status_code=404, detail="Job not found.")
        return found

    source_frontend = resolved_root / "apps" / "learner-web" / "dist"
    packaged_frontend = Path(__file__).resolve().parent / "web"
    frontend = source_frontend if source_frontend.is_dir() else packaged_frontend
    assets = frontend / "assets"
    if assets.is_dir():
        app.mount("/assets", StaticFiles(directory=assets), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def frontend_route(path: str) -> FileResponse:
        index = frontend / "index.html"
        if not index.is_file():
            raise HTTPException(
                status_code=503,
                detail="Frontend is not built. Run npm run build in apps/learner-web.",
            )
        return FileResponse(
            index,
            headers={"Cache-Control": "no-store, max-age=0"},
        )

    return app


app = create_app()
