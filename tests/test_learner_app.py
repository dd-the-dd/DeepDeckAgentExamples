from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from deepdeck_learner import app as learner_app
from deepdeck_learner.app import create_app


def test_health_status_and_protected_job_start(tmp_path: Path) -> None:
    client = TestClient(create_app(tmp_path))
    assert client.get("/api/v1/health").json() == {"status": "ok"}
    status = client.get("/api/v1/status").json()
    assert status["controller"]["ready"] is True
    assert status["hosted"]["trajectory_training"] is False
    forbidden = client.post("/api/v1/jobs", json={"kind": "training.smoke"})
    assert forbidden.status_code == 403
    token = client.get("/api/v1/session").json()["token"]
    invalid = client.post(
        "/api/v1/jobs",
        headers={"X-DeepDeck-Token": token},
        json={"kind": "unknown"},
    )
    assert invalid.status_code == 422


def test_training_curriculum_catalog_is_available(tmp_path: Path) -> None:
    response = TestClient(create_app(tmp_path)).get("/api/v1/training/curriculum")

    assert response.status_code == 200
    payload = response.json()
    assert payload["schemaVersion"] == "deepdeck-training-curriculum/v1"
    assert {item["family"] for item in payload["scenarios"]} >= {
        "opening-hand",
        "free-cast",
        "predictable-opponent",
        "combo",
        "known-combo",
        "sideboard",
    }
    assert len(payload["scenarios"]) == 21


def test_catalog_routes_keep_deck_identifiers_behind_named_results(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("DEEPDECK_API_KEY", "ddl_agent_test")
    monkeypatch.setattr(
        learner_app,
        "platform_decks",
        lambda search, game_format, page: {
            "items": [{"id": "deck-version", "name": "Reanimator", "format": game_format}],
            "pagination": {"page": page},
        },
    )
    client = TestClient(create_app(tmp_path))
    response = client.get("/api/v1/catalog/decks?search=reanimator&format=legacy")
    assert response.status_code == 200
    assert response.json()["items"][0]["name"] == "Reanimator"


def test_card_name_autocomplete_uses_scryfall_without_game_state(
    tmp_path: Path, monkeypatch
) -> None:
    searches: list[str] = []
    monkeypatch.setattr(
        learner_app,
        "scryfall_card_names",
        lambda query: searches.append(query) or ["Force of Will", "Force of Negation"],
    )

    response = TestClient(create_app(tmp_path)).get(
        "/api/v1/catalog/cards/autocomplete?query=Force"
    )

    assert response.status_code == 200
    assert response.json() == {"data": ["Force of Will", "Force of Negation"]}
    assert searches == ["Force"]


def test_local_deck_presentation_keeps_normal_token_and_double_faced_art(
    tmp_path: Path,
) -> None:
    deck_dir = tmp_path / ".deepdeck" / "decks"
    deck_dir.mkdir(parents=True)
    (deck_dir / "deck-version.json").write_text(
        json.dumps(
            {
                "name": "Lands",
                "cards": [
                    {
                        "cardId": "dark-depths-id",
                        "imageUri": "https://cards.scryfall.io/front/dark-depths.jpg",
                        "name": "Dark Depths",
                        "quantity": 4,
                        "typeLine": "Legendary Snow Land",
                    },
                    {
                        "cardId": "marit-lage-id",
                        "imageUri": "https://cards.scryfall.io/front/marit-lage.jpg",
                        "name": "Marit Lage",
                        "quantity": 1,
                        "typeLine": "Token Legendary Creature — Avatar",
                    },
                    {
                        "cardId": "dfc-id",
                        "imageBackUri": "https://cards.scryfall.io/back/land.jpg",
                        "imageUri": "https://cards.scryfall.io/front/spell.jpg",
                        "name": "Witch Enchanter // Witch-Blessed Meadow",
                        "quantity": 2,
                        "typeLine": "Creature — Human Warlock // Land",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    response = TestClient(create_app(tmp_path)).get(
        "/api/v1/catalog/decks/deck-version/presentation"
    )

    assert response.status_code == 200
    cards = response.json()["cards"]
    assert cards[0]["imageUrl"].endswith("dark-depths.jpg")
    assert cards[1]["isToken"] is True
    assert cards[1]["name"] == "Marit Lage"
    assert cards[2]["urlBack"].endswith("land.jpg")
    assert [face["name"] for face in cards[2]["faces"]] == [
        "Witch Enchanter",
        "Witch-Blessed Meadow",
    ]


def test_scryfall_image_proxy_returns_cacheable_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        learner_app,
        "scryfall_image",
        lambda image_path: (f"image:{image_path}".encode(), "image/jpeg"),
    )

    response = TestClient(create_app(tmp_path)).get(
        "/api/scryfall-images/border_crop/front/a/b/card.jpg?revision"
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/jpeg")
    assert response.headers["cache-control"] == "public, max-age=604800, immutable"
    assert response.content == b"image:border_crop/front/a/b/card.jpg"


def test_platform_catalog_requires_account_api_key(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("DEEPDECK_API_KEY", raising=False)
    client = TestClient(create_app(tmp_path))

    assert client.get("/api/v1/catalog/decks?format=legacy").status_code == 401
    assert client.get("/api/v1/catalog/competitions").status_code == 401


def test_downloaded_decks_remain_selectable_without_an_api_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("DEEPDECK_API_KEY", raising=False)
    deck_dir = tmp_path / ".deepdeck" / "decks"
    deck_dir.mkdir(parents=True)
    (deck_dir / "cached-legacy.json").write_text(
        json.dumps(
            {
                "id": "cached-legacy",
                "name": "Cached Reanimator",
                "format": "legacy",
                "version": 3,
                "cards": [
                    {
                        "name": "Swamp",
                        "quantity": 60,
                        "section": "main",
                        "typeLine": "Basic Land — Swamp",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    client = TestClient(create_app(tmp_path))

    response = client.get("/api/v1/catalog/decks?format=legacy&search=reanimator")

    assert response.status_code == 200
    assert response.json()["source"] == "local-cache"
    assert response.json()["items"][0]["id"] == "cached-legacy"
    assert response.json()["items"][0]["playableCardCount"] == 60


def test_api_key_can_be_saved_locally_without_being_returned(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("DEEPDECK_API_KEY", raising=False)
    client = TestClient(create_app(tmp_path))
    token = client.get("/api/v1/session").json()["token"]
    key = "ddl_agent_complete_local_test_key"
    try:
        response = client.post(
            "/api/v1/settings/api-key",
            headers={"X-DeepDeck-Token": token},
            json={"api_key": key},
        )
        assert response.status_code == 200
        assert response.json() == {"configured": True}
        assert key not in response.text
        assert client.get("/api/v1/status").json()["hosted"]["api_key_configured"]
        stored = (tmp_path / ".deepdeck" / "secrets.json").read_text("utf-8")
        assert key in stored
    finally:
        os.environ.pop("DEEPDECK_API_KEY", None)


def test_models_route_only_exposes_user_owned_local_model_metadata(tmp_path: Path) -> None:
    run = tmp_path / ".deepdeck" / "runs" / "my-model"
    checkpoint = run / "live" / "my-model-id"
    checkpoint.mkdir(parents=True)
    (checkpoint / "manifest.json").write_text("{}", encoding="utf-8")
    (checkpoint / "checkpoint.pt").touch()
    (run / "local-model.json").write_text(
        json.dumps(
            {
                "schemaVersion": "local-model/v1",
                "id": "my-model-id",
                "name": "My Model",
                "architecture": "v12",
                "format": "legacy",
                "description": "User-owned weights",
                "createdAt": "2026-08-30T00:00:00+00:00",
                "checkpointPath": str(checkpoint),
                "reservePlaytest": True,
            }
        ),
        encoding="utf-8",
    )

    response = TestClient(create_app(tmp_path)).get("/api/v1/models")

    assert response.status_code == 200
    item = response.json()["items"][0]
    assert item.pop("diskBytes") > 0
    assert item.pop("weightsBytes") == 2
    assert item.pop("trainingState") == {}
    assert item == {
        "schemaVersion": "local-model/v1",
        "id": "my-model-id",
        "name": "My Model",
        "architecture": "v12",
        "format": "legacy",
        "description": "User-owned weights",
        "createdAt": "2026-08-30T00:00:00+00:00",
        "checkpointPath": str(checkpoint),
        "reservePlaytest": True,
        "runPath": str(run.resolve()),
        "status": "stopped",
        "ready": True,
        "decks": [],
    }


def test_model_resources_and_deck_statistics_are_local_and_persistent(tmp_path: Path) -> None:
    run = tmp_path / ".deepdeck" / "runs" / "rated-model"
    run.mkdir(parents=True)
    (run / "local-model.json").write_text(
        json.dumps(
            {
                "schemaVersion": "local-model/v1",
                "id": "rated-model-id",
                "name": "Rated Model",
                "architecture": "v12",
                "format": "legacy",
                "checkpointPath": str(run / "live" / "rated-model-id"),
                "decks": [
                    {
                        "id": "deck-version-12345678",
                        "name": "Reanimator",
                        "format": "legacy",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (run / "training-leaderboard.json").write_text(
        json.dumps(
            {
                "ratingSystem": "plackett-luce",
                "deckParticipants": [
                    {
                        "participantId": "rated-model-id",
                        "deckName": "Reanimator Â· v1 Â· deck-ver",
                        "mu": 28.0,
                        "sigma": 5.0,
                        "ordinal": 13.0,
                        "rank": 2,
                        "games": 12,
                        "gameWins": 8,
                        "gameLosses": 4,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    client = TestClient(create_app(tmp_path))
    token = client.get("/api/v1/session").json()["token"]

    saved = client.put(
        "/api/v1/models/rated-model-id/resources",
        headers={"X-DeepDeck-Token": token},
        json={
            "trainingMatches": 3,
            "leagueMatches": 2,
            "localMatches": 1,
            "gpuMemoryMb": 4096,
        },
    )

    assert saved.status_code == 200
    assert client.get("/api/v1/models/rated-model-id/resources").json() == saved.json()
    assert json.loads((run / "training-control.json").read_text("utf-8")) == {
        "desiredState": "running"
    }
    statistic = client.get("/api/v1/statistics/decks").json()["items"][0]
    assert statistic["ratingSystem"] == "plackett-luce"
    assert statistic["ordinal"] == 13.0
    assert statistic["winRate"] == pytest.approx(2 / 3)
    snapshot = client.get("/api/v1/resources").json()
    assert snapshot["system"]["ramTotalBytes"] > 0
    assert snapshot["engine"]["activeLocalGames"] == 0


def test_training_platform_settings_evidence_and_replays(tmp_path: Path) -> None:
    run = tmp_path / ".deepdeck" / "runs" / "platform-model"
    replay_dir = run / "replays"
    replay_dir.mkdir(parents=True)
    (run / "local-model.json").write_text(
        json.dumps(
            {
                "schemaVersion": "local-model/v1",
                "id": "platform-model-id",
                "name": "Platform Model",
                "architecture": "v12",
                "format": "legacy",
            }
        ),
        encoding="utf-8",
    )
    (run / "training-config.yaml").write_text(
        "checkpointEvery: 20\nmodelEvaluationEvery: 100\n"
        "modelEvaluationEnabled: false\nevaluation:\n  gamesPerScenario: 1\n",
        encoding="utf-8",
    )
    evaluation_games = [
        {
            "candidateDeck": "Reanimator",
            "result": "candidateWin" if index < 24 else "championWin",
        }
        for index in range(30)
    ]
    (run / "evaluations.jsonl").write_text(
        json.dumps(
            {
                "period": 1,
                "candidateTrainingStep": 50,
                "opponentVersion": "champion-0",
                "fixedSeeds": {"legacy": list(range(30))},
                "summary": {"completedGames": 30, "candidateWins": 24},
                "games": evaluation_games,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (replay_dir / "episode-00000030.json").write_text(
        json.dumps(
            {
                "schemaVersion": "deepdeck-replay/v1",
                "id": "episode-00000030",
                "createdAtUnixMs": 123,
                "metadata": {
                    "episode": 30,
                    "seed": 9,
                    "decks": ["Reanimator", "Delver"],
                    "outcome": {"winner": "candidate"},
                },
                "frames": [{"state": {"turnNumber": 1}}, {"state": {"turnNumber": 2}}],
            }
        ),
        encoding="utf-8",
    )
    (run / "agent-training-contract.json").write_text(
        json.dumps(
            {
                "schemaVersion": "deepdeck-agent-training/v1",
                "agentId": "platform-model-id",
                "phases": [
                    {
                        "id": "self-play",
                        "label": "Self-play",
                        "controls": ["pause", "resume"],
                        "metrics": [],
                        "parameters": [
                            {
                                "key": "rollout_games",
                                "label": "Rollout games",
                                "kind": "integer",
                                "min": 1,
                                "max": 32,
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (run / "agent-training-metrics.jsonl").write_text(
        json.dumps(
            {
                "schemaVersion": "deepdeck-agent-training-metrics/v1",
                "agentId": "platform-model-id",
                "phase": "self-play",
                "step": 7,
                "metrics": {"reward": 0.25},
                "recordedAtUnixMs": 123,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    client = TestClient(create_app(tmp_path))
    token = client.get("/api/v1/session").json()["token"]

    saved = client.put(
        "/api/v1/models/platform-model-id/training-settings",
        headers={"X-DeepDeck-Token": token},
        json={
            "savedGameLimit": 12,
            "checkpointEvery": 10,
            "evaluationEvery": 25,
            "evaluationGamesPerScenario": 5,
            "stages": {
                "worldModel": False,
                "reinforcementLearning": True,
                "engineEvaluation": True,
            },
            "targets": {"worldModelSteps": 0, "reinforcementLearningEpisodes": 0},
        },
    )

    assert saved.status_code == 200
    assert saved.json()["savedGameLimit"] == 12
    assert saved.json()["stages"]["engineEvaluation"] is True
    evidence = client.get("/api/v1/models/platform-model-id/evidence").json()
    assert evidence["status"] == "verified"
    assert evidence["winRate"] == pytest.approx(0.8)
    assert evidence["byDeck"][0]["wins"] == 24
    replay = client.get("/api/v1/models/platform-model-id/replays").json()["items"][0]
    assert replay["frameCount"] == 2
    assert replay["saved"] is False
    assert replay["viewing"] is False
    assert (
        client.get("/api/v1/models/platform-model-id/replays/episode-00000030").json()["frames"][1][
            "state"
        ]["turnNumber"]
        == 2
    )
    leased = client.post(
        "/api/v1/models/platform-model-id/replays/episode-00000030/leases",
        headers={"X-DeepDeck-Token": token},
    )
    assert leased.status_code == 200
    assert leased.json()["replay"]["id"] == "episode-00000030"
    assert (
        client.get("/api/v1/models/platform-model-id/replays").json()["items"][0]["viewing"] is True
    )
    kept = client.post(
        "/api/v1/models/platform-model-id/replays/episode-00000030/save",
        headers={"X-DeepDeck-Token": token},
    )
    assert kept.json() == {"saved": True}
    assert not (replay_dir / "episode-00000030.json").exists()
    assert (run / "saved-replays" / "episode-00000030.json").is_file()
    permanent = client.get("/api/v1/models/platform-model-id/replays").json()["items"][0]
    assert permanent["saved"] is True
    assert permanent["viewing"] is True
    publication = client.get("/api/v1/models/platform-model-id/training-contract").json()
    assert publication["contract"]["phases"][0]["id"] == "self-play"
    assert publication["metrics"][0]["step"] == 7
    control = client.put(
        "/api/v1/models/platform-model-id/training-control",
        headers={"X-DeepDeck-Token": token},
        json={
            "phase": "self-play",
            "action": "pause",
            "parameters": {"rollout_games": 4},
        },
    )
    assert control.status_code == 200
    assert control.json()["revision"] == 1
    assert "rollout_games: 4" in (run / "training-config.yaml").read_text("utf-8")
    invalid_control = client.put(
        "/api/v1/models/platform-model-id/training-control",
        headers={"X-DeepDeck-Token": token},
        json={
            "phase": "self-play",
            "action": "pause",
            "parameters": {"rollout_games": 100},
        },
    )
    assert invalid_control.status_code == 422


def test_v13_cannot_claim_engine_evaluation_before_integration(tmp_path: Path) -> None:
    run = tmp_path / ".deepdeck" / "runs" / "v13-model"
    run.mkdir(parents=True)
    (run / "local-model.json").write_text(
        json.dumps(
            {
                "schemaVersion": "local-model/v1",
                "id": "v13-model-id",
                "name": "V13 Model",
                "architecture": "v13",
                "format": "legacy",
            }
        ),
        encoding="utf-8",
    )
    (run / "training-config.yaml").write_text(
        "training:\n  steps: 10\n  checkpoint_every: 5\n"
        "rl:\n  episodes: 20\n  checkpoint_every: 5\n",
        encoding="utf-8",
    )
    client = TestClient(create_app(tmp_path))
    token = client.get("/api/v1/session").json()["token"]
    settings = client.get("/api/v1/models/v13-model-id/training-settings").json()
    settings["stages"]["engineEvaluation"] = True

    response = client.put(
        "/api/v1/models/v13-model-id/training-settings",
        headers={"X-DeepDeck-Token": token},
        json=settings,
    )

    assert response.status_code == 422
    evidence = client.get("/api/v1/models/v13-model-id/evidence").json()
    assert evidence["status"] == "not-connected"
    assert evidence["magicEvidence"] is False


def test_agent_files_can_be_deleted_with_an_explicit_authorized_request(
    tmp_path: Path,
) -> None:
    run = tmp_path / ".deepdeck" / "runs" / "retired-model"
    checkpoint = run / "live" / "retired-model-id"
    checkpoint.mkdir(parents=True)
    (checkpoint / "checkpoint.pt").write_bytes(b"weights")
    (run / "local-model.json").write_text(
        json.dumps(
            {
                "schemaVersion": "local-model/v1",
                "id": "retired-model-id",
                "name": "Retired Model",
                "architecture": "v12",
                "format": "legacy",
                "checkpointPath": str(checkpoint),
            }
        ),
        encoding="utf-8",
    )
    client = TestClient(create_app(tmp_path))
    token = client.get("/api/v1/session").json()["token"]

    response = client.delete(
        "/api/v1/models/retired-model-id",
        headers={"X-DeepDeck-Token": token},
    )

    assert response.status_code == 200
    assert response.json()["deleted"] is True
    assert response.json()["reclaimedBytes"] >= len(b"weights")
    assert not run.exists()


def test_live_training_games_are_visible_and_cancelled_with_the_raw_session_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = tmp_path / ".deepdeck" / "runs" / "live-model"
    run.mkdir(parents=True)
    (run / "local-model.json").write_text(
        json.dumps(
            {
                "schemaVersion": "local-model/v1",
                "id": "live-model-id",
                "name": "Live Model",
                "architecture": "v12",
                "format": "legacy",
            }
        ),
        encoding="utf-8",
    )
    (run / "league-state.json").write_text(
        json.dumps(
            {
                "processId": os.getpid(),
                "completed_episodes": 12,
                "trainingStep": 4,
                "parallelGameWorkers": 2,
                "trainingPhase": "collecting",
                "desiredState": "running",
                "activeAttempts": [
                    {
                        "sessionId": "game-session:9",
                        "worker": 1,
                        "status": "collecting",
                        "opponentMode": "self",
                        "decks": ["Reanimator", "Reanimator"],
                        "players": 2,
                        "turnNumber": 3,
                        "decisions": 8,
                    },
                    {
                        "sessionId": "game-session:finished",
                        "worker": 2,
                        "status": "optimizingWeights",
                        "opponentMode": "self",
                        "decks": ["Delver", "Delver"],
                        "players": 2,
                        "turnNumber": 18,
                        "decisions": 240,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    (run / "training.jsonl").write_text(
        json.dumps(
            {
                "episode": 12,
                "trainingStep": 4,
                "gameDurationSeconds": 18.5,
                "decisions": 12,
                "rolloutBatch": {
                    "requestedGames": 2,
                    "completedGames": 1,
                    "failedGames": 1,
                    "totalDecisions": 12,
                    "collectionWallSeconds": 20.0,
                    "trainingWallSeconds": 2.5,
                },
                "ppo": {"loss": 0.25, "entropy": 0.7},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    cancelled_urls: list[str] = []

    class EngineResponse:
        status_code = 200

        @staticmethod
        def raise_for_status() -> None:
            return None

    def cancel_engine_game(url: str, **_: object) -> EngineResponse:
        cancelled_urls.append(url)
        return EngineResponse()

    monkeypatch.setattr("deepdeck_learner.jobs.httpx.delete", cancel_engine_game)
    client = TestClient(create_app(tmp_path))
    token = client.get("/api/v1/session").json()["token"]

    game = client.get("/api/v1/games").json()["items"][0]
    assert len(client.get("/api/v1/games").json()["items"]) == 1
    statistic = client.get("/api/v1/statistics/training").json()["items"][0]
    stopped = client.post(
        "/api/v1/games/game-session%3A9/stop",
        headers={"X-DeepDeck-Token": token},
    )

    assert game["sessionId"] == "game-session:9"
    assert game["mode"] == "Self-play"
    assert statistic["completedGames"] == 12
    assert statistic["averageGameSeconds"] == 18.5
    assert statistic["windowSummary"]["generatedSamples"] == 1
    assert statistic["windowSummary"]["usedSamples"] == 1
    assert statistic["windowSummary"]["failedSamples"] == 1
    assert statistic["windowSummary"]["generatedDecisions"] == 12
    assert statistic["windowSummary"]["p95GameSeconds"] == 18.5
    assert stopped.status_code == 200
    assert stopped.json()["status"] == "cancelled"
    assert cancelled_urls == ["http://127.0.0.1:8787/game/sessions/game-session:9"]


def test_v13_world_model_losses_are_exposed_to_the_dashboard(tmp_path: Path) -> None:
    run = tmp_path / ".deepdeck" / "runs" / "v13-smoke"
    run.mkdir(parents=True)
    (run / "local-model.json").write_text(
        json.dumps(
            {
                "id": "v13-smoke",
                "name": "V13 smoke",
                "architecture": "v13",
                "format": "legacy",
            }
        ),
        encoding="utf-8",
    )
    (run / "v13-training-state.json").write_text(
        json.dumps({"status": "running", "step": 7}), encoding="utf-8"
    )
    losses = {
        "loss": 2.5,
        "reconstruction_loss": 0.4,
        "dynamics_loss": 0.3,
        "kl_loss": 0.02,
        "belief_loss": 0.5,
        "opponent_loss": 0.6,
        "search_loss": 0.7,
        "value_loss": 0.8,
    }
    (run / "v13-metrics.jsonl").write_text(
        json.dumps({"step": 7, **losses}) + "\n", encoding="utf-8"
    )

    statistic = (
        TestClient(create_app(tmp_path)).get("/api/v1/statistics/training").json()["items"][0]
    )

    assert statistic["architecture"] == "v13"
    assert statistic["trainingStep"] == 7
    assert statistic["latestMetrics"][0]["losses"] == losses


def test_models_route_does_not_offer_pre_v13_2_checkpoint_for_playtest(
    tmp_path: Path,
) -> None:
    run = tmp_path / ".deepdeck" / "runs" / "old-v13"
    checkpoint = run / "rl-checkpoints" / "update-10"
    checkpoint.mkdir(parents=True)
    (checkpoint / "rl-model.pt").touch()
    (run / "resolved-v13-config.json").write_text(
        json.dumps({"model": {"latent_dim": 128}}), encoding="utf-8"
    )
    (run / "local-model.json").write_text(
        json.dumps(
            {
                "schemaVersion": "local-model/v1",
                "id": "old-v13-id",
                "name": "Old V13",
                "architecture": "v13",
                "format": "legacy",
                "createdAt": "2026-09-21T00:00:00+00:00",
                "checkpointPath": str(checkpoint),
                "reservePlaytest": True,
            }
        ),
        encoding="utf-8",
    )

    item = TestClient(create_app(tmp_path)).get("/api/v1/models").json()["items"][0]

    assert item["ready"] is False


def test_models_route_offers_any_agent_sdk_runtime_with_ready_artifacts(
    tmp_path: Path,
) -> None:
    run = tmp_path / ".deepdeck" / "runs" / "future-agent"
    checkpoint = run / "artifacts" / "release"
    checkpoint.mkdir(parents=True)
    (checkpoint / "agent.bundle").touch()
    (run / "local-model.json").write_text(
        json.dumps(
            {
                "schemaVersion": "local-model/v1",
                "id": "future-agent-id",
                "name": "Future Agent",
                "architecture": "v-next",
                "format": "legacy",
                "createdAt": "2026-10-03T00:00:00+00:00",
                "checkpointPath": str(checkpoint),
                "reservePlaytest": True,
                "agentSdkRuntime": {
                    "module": "future_agent.sdk_runner",
                    "arguments": ["serve"],
                    "checkpointArgument": "--artifact",
                    "requiredFiles": ["agent.bundle"],
                },
            }
        ),
        encoding="utf-8",
    )

    item = TestClient(create_app(tmp_path)).get("/api/v1/models").json()["items"][0]

    assert item["architecture"] == "v-next"
    assert item["ready"] is True


def test_v13_rl_metrics_replace_pretraining_view_without_claiming_engine_games(
    tmp_path: Path,
) -> None:
    run = tmp_path / ".deepdeck" / "runs" / "v13-rl"
    run.mkdir(parents=True)
    (run / "local-model.json").write_text(
        json.dumps(
            {
                "id": "v13-rl",
                "name": "V13 RL",
                "architecture": "v13",
                "format": "legacy",
            }
        ),
        encoding="utf-8",
    )
    (run / "v13-training-state.json").write_text(
        json.dumps(
            {
                "status": "running",
                "trainingPhase": "reinforcement-learning",
                "trainingStep": 2,
                "completed_episodes": 8,
            }
        ),
        encoding="utf-8",
    )
    (run / "v13-metrics.jsonl").write_text(
        json.dumps({"step": 99, "loss": 9.9}) + "\n", encoding="utf-8"
    )
    record = {
        "episode": 8,
        "trainingStep": 2,
        "loss": 0.2,
        "policy_loss": -0.01,
        "value_loss": 0.4,
        "entropy": 0.6,
        "approx_kl": 0.001,
        "clip_fraction": 0.02,
        "episode_reward": 0.75,
        "data_source": "tiny-self-play-v1",
        "gameDurationsSeconds": [0.1, 0.2, 0.3, 0.4],
        "decisions": 32,
        "ppo": {"loss": 0.2},
        "rolloutBatch": {
            "requestedGames": 4,
            "completedGames": 4,
            "failedGames": 0,
            "totalDecisions": 32,
            "collectionWallSeconds": 1.0,
            "trainingWallSeconds": 0.5,
        },
    }
    (run / "v13-rl-metrics.jsonl").write_text(
        json.dumps({**record, "episode_reward": 0.1}) + "\n" + json.dumps(record) + "\n",
        encoding="utf-8",
    )

    statistic = (
        TestClient(create_app(tmp_path))
        .get("/api/v1/statistics/training?window=50")
        .json()["items"][0]
    )

    assert statistic["metricRecordCount"] == 1
    assert statistic["latestMetrics"][0]["trainingStep"] == 2
    assert statistic["latestMetrics"][0]["episodeReward"] == 0.75
    assert statistic["windowSummary"]["sampleType"] == "rl-episodes"
    assert statistic["windowSummary"]["engineGameMetrics"] is False
    assert statistic["windowSummary"]["generatedSamples"] == 4
    assert statistic["windowSummary"]["averageGameSeconds"] == pytest.approx(0.25)


def test_v13_production_summary_uses_current_data_source(tmp_path: Path) -> None:
    run = tmp_path / ".deepdeck" / "runs" / "v13-engine"
    run.mkdir(parents=True)
    (run / "local-model.json").write_text(
        json.dumps(
            {
                "id": "v13-engine",
                "name": "V13 Engine",
                "architecture": "v13",
                "format": "legacy",
            }
        ),
        encoding="utf-8",
    )
    (run / "v13-training-state.json").write_text(
        json.dumps({"status": "running", "trainingStep": 2, "completed_episodes": 18}),
        encoding="utf-8",
    )
    tiny = {
        "episode": 16,
        "trainingStep": 1,
        "data_source": "tiny-self-play-v1",
        "gameDurationsSeconds": [0.1] * 16,
        "decisions": 64,
        "ppo": {"loss": 0.1},
        "rolloutBatch": {
            "requestedGames": 16,
            "completedGames": 16,
            "failedGames": 0,
            "totalDecisions": 64,
            "collectionWallSeconds": 2.0,
            "trainingWallSeconds": 0.5,
        },
    }
    engine = {
        "episode": 18,
        "trainingStep": 2,
        "data_source": "engine-self-play-legacy-v1",
        "gameDurationsSeconds": [10.0, 20.0],
        "decisions": 200,
        "ppo": {"loss": 0.2},
        "rolloutBatch": {
            "requestedGames": 2,
            "completedGames": 2,
            "failedGames": 0,
            "totalDecisions": 200,
            "collectionWallSeconds": 30.0,
            "trainingWallSeconds": 1.0,
        },
    }
    (run / "v13-rl-metrics.jsonl").write_text(
        json.dumps(tiny) + "\n" + json.dumps(engine) + "\n",
        encoding="utf-8",
    )

    statistic = (
        TestClient(create_app(tmp_path))
        .get("/api/v1/statistics/training?window=all")
        .json()["items"][0]
    )

    assert statistic["metricRecordCount"] == 2
    assert len(statistic["latestMetrics"]) == 2
    assert statistic["averageGameSeconds"] == pytest.approx(15.0)
    assert statistic["windowSummary"]["sampleType"] == "games"
    assert statistic["windowSummary"]["generatedSamples"] == 2
    assert statistic["windowSummary"]["generatedDecisions"] == 200


def test_training_metric_windows_support_all_time_with_downsampling(tmp_path: Path) -> None:
    run = tmp_path / ".deepdeck" / "runs" / "v13-long"
    run.mkdir(parents=True)
    (run / "local-model.json").write_text(
        json.dumps(
            {
                "id": "v13-long",
                "name": "V13 long",
                "architecture": "v13",
                "format": "legacy",
            }
        ),
        encoding="utf-8",
    )
    (run / "v13-training-state.json").write_text(
        json.dumps({"status": "completed", "step": 600}), encoding="utf-8"
    )
    (run / "v13-metrics.jsonl").write_text(
        "\n".join(
            json.dumps(
                {
                    "step": step,
                    "loss": 1 / step,
                    "value_loss": 1 / (step + 1),
                    "elapsed_seconds": float(step),
                }
            )
            for step in range(1, 601)
        )
        + "\n",
        encoding="utf-8",
    )
    client = TestClient(create_app(tmp_path))

    recent = client.get("/api/v1/statistics/training?window=50").json()["items"][0]
    lifetime = client.get("/api/v1/statistics/training?window=all").json()["items"][0]

    assert recent["metricRecordCount"] == 50
    assert recent["latestMetrics"][0]["trainingStep"] == 551
    assert lifetime["metricRecordCount"] == 600
    assert lifetime["metricPointsReturned"] == 500
    assert lifetime["latestMetrics"][0]["trainingStep"] == 1
    assert lifetime["latestMetrics"][-1]["trainingStep"] == 600
    assert lifetime["windowSummary"]["coveredTrainingSteps"] == 600
    assert client.get("/api/v1/statistics/training?window=bad").status_code == 422


def test_frontend_entrypoint_is_never_cached_across_asset_rebuilds(tmp_path: Path) -> None:
    frontend = tmp_path / "apps" / "learner-web" / "dist"
    frontend.mkdir(parents=True)
    (frontend / "index.html").write_text("<html>workbench</html>", encoding="utf-8")

    response = TestClient(create_app(tmp_path)).get("/")

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store, max-age=0"
