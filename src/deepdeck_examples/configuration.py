from __future__ import annotations

import os

from deepdeck_agent import AgentConfig, DeckPolicy, PlaySpeed


def _repository_commit() -> str:
    return os.getenv("DEEPDECK_AGENT_COMMIT", "local-worktree").strip() or "local-worktree"


def random_config() -> AgentConfig:
    return AgentConfig(
        agent_id=os.getenv(
            "DEEPDECK_AGENT_ID", "com.deepdeckleague.example.random"
        ).strip(),
        name="Public random baseline",
        version=os.getenv("DEEPDECK_AGENT_VERSION", "0.1.0").strip(),
        author="Deep Deck League",
        description="Samples one legal action with a reproducible pseudo-random generator.",
        formats=("legacy", "commander"),
        decks=DeckPolicy.all(),
        speeds=(PlaySpeed.MS_100, PlaySpeed.SECOND_1, PlaySpeed.SECONDS_10),
        repository_url="https://github.com/dd-the-dd/DeepDeckLearner",
        repository_commit=_repository_commit(),
    )


def forge_reference_config(profile: str = "balanced") -> AgentConfig:
    normalized = profile.strip().casefold()
    if normalized not in {"cautious", "balanced", "aggressive"}:
        raise ValueError("Forge reference profile must be cautious, balanced, or aggressive")
    return AgentConfig(
        agent_id=os.getenv(
            "DEEPDECK_AGENT_ID", f"com.deepdeckleague.example.forge-reference-{normalized}"
        ).strip(),
        name=f"Forge-style reference ({normalized})",
        version=os.getenv("DEEPDECK_AGENT_VERSION", "0.1.0").strip(),
        author="Deep Deck League",
        description=(
            "Independent MIT heuristic reference inspired by Forge's programmatic AI "
            "architecture; contains no Forge GPL source code."
        ),
        formats=("legacy", "commander"),
        agent_type="rules_engine",
        decks=DeckPolicy.all(),
        speeds=(PlaySpeed.MS_100, PlaySpeed.SECOND_1, PlaySpeed.SECONDS_10),
        repository_url="https://github.com/dd-the-dd/DeepDeckLearner",
        repository_commit=_repository_commit(),
    )


def alexios_config() -> AgentConfig:
    deck_id = os.getenv("ALEXIOS_DECK_ID", "alexios").strip() or "alexios"
    return AgentConfig(
        agent_id=os.getenv(
            "DEEPDECK_AGENT_ID", "com.deepdeckleague.example.alexios"
        ).strip(),
        name="Alexios fast equipment",
        version=os.getenv("DEEPDECK_AGENT_VERSION", "0.1.0").strip(),
        author="Deep Deck League",
        description="Transparent rule-based Alexios equipment and protection policy.",
        formats=("commander",),
        decks=DeckPolicy.only(deck_id),
        speeds=(PlaySpeed.MS_100, PlaySpeed.SECOND_1, PlaySpeed.SECONDS_10),
        repository_url="https://github.com/dd-the-dd/DeepDeckLearner",
        repository_commit=_repository_commit(),
    )


def deep_learning_config(version: str) -> AgentConfig:
    if version not in {"v11", "v12", "v13"}:
        raise ValueError("deep learning version must be v11, v12 or v13")
    default_id = f"com.deepdeckleague.example.{version}"
    return AgentConfig(
        agent_id=os.getenv("DEEPDECK_AGENT_ID", default_id).strip(),
        name=f"Deep learning {version.upper()} example",
        version=os.getenv("DEEPDECK_AGENT_VERSION", "0.1.0").strip(),
        author="Deep Deck League",
        description={
            "v11": "Trainable V11 recurrent multiplayer policy example.",
            "v12": "Trainable V12 recurrent two-player zero-sum policy example.",
            "v13": "Trainable V13 structured world-model two-player policy.",
        }[version],
        formats=("legacy", "commander") if version == "v11" else ("legacy",),
        decks=DeckPolicy.all(),
        speeds=(PlaySpeed.SECOND_1, PlaySpeed.SECONDS_10),
        repository_url="https://github.com/dd-the-dd/DeepDeckLearner",
        repository_commit=_repository_commit(),
    )
