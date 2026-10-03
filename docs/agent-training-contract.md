# Agent SDK training contract

DeepDeckLearner orchestrates training without importing a model architecture. The agent
remains the AI scientist: it defines the curriculum, phases, metrics, promotion criteria,
and which parameter changes are safe while training.

## Workspace protocol

An SDK `TrainingBridge` publishes four versioned files in an agent run directory:

- `agent-training-contract.json`: immutable phase, metric, parameter, replay, and control
  capabilities;
- `agent-training-state.json`: current phase, status, step, acknowledged control revision,
  and agent-specific details;
- `agent-training-metrics.jsonl`: append-only, phase-qualified metric observations;
- `agent-training-control.json`: the latest desired action and validated parameter values
  requested by DeepDeckLearner.

Writes that replace state or control are atomic. Metrics stay append-only so dashboards,
offline analysis, and future experiment comparisons share the same source of truth.

## Responsibility boundary

The SDK contract owns semantics. Every metric declares its kind, optimization direction,
unit, group, and explanation. Every parameter declares its type, bounds, choices, and
whether it applies live, at the next phase, or on the next run.

DeepDeckLearner owns presentation and orchestration. It discovers the contract, renders
generic controls, validates requests against the published schema, and writes a monotonic
control revision. It never assumes that all agents use PPO, PyTorch, or V13 phases.

The agent acknowledges a control revision only at a safe boundary. V13 currently polls at
update boundaries, supports pause/resume/stop/checkpoint, and can change replay retention
live. Optimizer and rollout parameters marked for a later boundary are intentionally not
mutated in the middle of a rollout.

## Replay lifecycle

Training replays have three states:

1. **temporary** — part of the rolling configured limit;
2. **viewing** — protected by a renewable 90-second lease while the replay is open;
3. **permanent** — moved to `saved-replays/` and never considered by retention cleanup.

When a new temporary replay exceeds the limit, cleanup removes the oldest eligible replay.
Active leases are skipped. A crashed browser naturally releases protection when its lease
expires; a permanently saved replay has no expiry.

## Next protocol increments

The current filesystem transport is local, inspectable, and dependency-free. A later SDK
version can expose the same schemas over authenticated local HTTP or WebSocket transport
without changing the contract consumed by DeepDeckLearner. Experiment proposals and
promotion decisions should become separate append-only records rather than opaque UI
flags, allowing reproducible comparisons across seeds, decks, and frozen baselines.
