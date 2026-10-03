# V15 typed latent planner

V15 is an Agent SDK planning architecture. It keeps the Engine authoritative for
observable state and legal actions while learning compact models for imagined future
states.

## Model boundary

The state pyramid has detailed, tactical, and strategic latents. Actions use separate
kind, cost, target, timing, and effect encoders. Five transition operators predict
different deltas:

1. cost payment;
2. announcement or stack insertion;
3. response;
4. resolution;
5. state-based-action and trigger closure.

The model composes those deltas for latent rollouts. Every real observation is encoded
again, so prediction drift cannot replace the Engine's current state. Training compares
the composed latent with a detached future-state encoding, checks cheap invariants on
every sample, and reconstructs detailed tokens only on a configured sample of nodes.

## SDK planning extension

`deepdeck_agent.planning` adds an optional `PlanningModel` contract and `PlannedAgent`
adapter. Existing reactive agents continue to implement `Agent.make_decision` unchanged.
The planning adapter maintains a `PlanGraph`, accumulates ordered events, requests
`KEEP`, `ADVANCE`, `PATCH`, or `REPLAN`, and binds semantic intentions only to exact
actions offered by the Engine.

The Engine owns the optional chess-style decision clock. It charges wall-clock time,
applies the increment once per turn (never once per priority exchange), publishes the
remaining bank, reserve, delay, and hard deadline to the SDK, and records a loss with
reason `time` when the bank reaches zero. V15 only uses this read-only budget to select
reflex, tactical, or strategic inference.

## Training and current limit

Run the smoke trainer with:

```powershell
python -m oracle_ai.training.world_model_v15 `
  --config configs/oracle-ai/v15-typed-latent-smoke.yaml
```

Run training on authoritative Engine replays with:

```powershell
python -m oracle_ai.training.world_model_v15 `
  --config configs/oracle-ai/v15-typed-latent-engine.yaml
```

It writes `v15-metrics.jsonl`, `v15-training-state.json`, and resumable-style
`checkpoints/step-N/v15-model.pt` artifacts. The synthetic generator is enabled only by
`synthetic_smoke: true` and validates the architecture, not Magic competence. The
production configuration `v15-typed-latent-engine.yaml` consumes actions and consecutive
authoritative states from Engine replay frames. It aligns nodes by stable identity and
derives typed cost, announcement, response, resolution, and rules-closure targets from
the observed action/stack boundary. Plan-level reinforcement learning remains separate
so it cannot overwrite the action-effect or legality world model.
