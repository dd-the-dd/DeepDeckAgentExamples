from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

from oracle_ai.training.environments import RustSessionEnvironment
from oracle_ai.training.league import LeagueTrainer


def test_environment_replay_frames_are_deep_copied_and_annotated() -> None:
    environment = RustSessionEnvironment.__new__(RustSessionEnvironment)
    environment.capture_replay = True
    environment.replay_frames = []
    view = {"revision": 1, "state": {"turnNumber": 1}}

    environment._start_replay(view)
    environment._record_replay_action({"id": "play-card", "label": "Play card"})
    view["state"]["turnNumber"] = 2
    environment._record_replay_view(view)

    assert environment.replay_frames[0]["state"]["turnNumber"] == 1
    assert environment.replay_frames[0]["selectedAction"]["id"] == "play-card"
    assert environment.replay_frames[1]["state"]["turnNumber"] == 2


def test_replay_retention_keeps_only_the_configured_recent_games(tmp_path: Path) -> None:
    trainer = LeagueTrainer.__new__(LeagueTrainer)
    trainer.output = tmp_path
    trainer.saved_game_limit = 2
    environment = SimpleNamespace(replay_frames=[{"state": {"turnNumber": 1}}])

    for episode in range(1, 4):
        trainer._save_training_replay(
            environment,
            {
                "episode": episode,
                "completedAtUnixMs": episode,
                "seed": episode,
                "outcome": {"winner": "player-1"},
            },
        )

    paths = sorted((tmp_path / "replays").glob("*.json"))
    assert [path.stem for path in paths] == ["episode-00000002", "episode-00000003"]
    payload = json.loads(paths[-1].read_text(encoding="utf-8"))
    assert payload["schemaVersion"] == "deepdeck-replay/v1"
    assert payload["frames"][0]["state"]["turnNumber"] == 1


def test_replay_retention_never_removes_a_viewed_game(tmp_path: Path) -> None:
    trainer = LeagueTrainer.__new__(LeagueTrainer)
    trainer.output = tmp_path
    trainer.saved_game_limit = 2
    environment = SimpleNamespace(replay_frames=[{"state": {"turnNumber": 1}}])
    trainer._save_training_replay(
        environment,
        {"episode": 1, "completedAtUnixMs": 1, "seed": 1, "outcome": {}},
    )
    lease_dir = tmp_path / "replays" / ".leases" / "episode-00000001"
    lease_dir.mkdir(parents=True)
    (lease_dir / "viewer.lease").write_text(
        str(int(time.time()) + 90), encoding="utf-8"
    )

    for episode in (2, 3):
        trainer._save_training_replay(
            environment,
            {
                "episode": episode,
                "completedAtUnixMs": episode,
                "seed": episode,
                "outcome": {},
            },
        )

    paths = sorted((tmp_path / "replays").glob("*.json"))
    assert [path.stem for path in paths] == ["episode-00000001", "episode-00000003"]
