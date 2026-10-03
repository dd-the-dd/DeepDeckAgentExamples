# V13.2 structured world model

V13.2 uses an ordered structured-token Transformer with a separate on-policy PPO stage.
It does not replace V12 as the default playing agent.

## Implemented first stage

- An observer-safe ordered sequence with player, zone, visible-card, per-ability
  Oracle and ordered-stack tokens. Hidden zones expose counts, not identities.
- Numeric, token-type, owner, controller, zone and within-zone position channels,
  plus a variable-length Oracle-text Transformer for every ability.
- A state Transformer followed by a persistent recurrent strategic memory.
- Independent trigger/timing, non-mana cost and effect encoders for legal actions.
- Action-conditioned afterstates and separate stochastic prior/posterior heads.
- Reconstruction, dynamics, KL, belief, opponent-policy, search-policy and
  win/draw/loss value objectives.
- JSONL metrics and periodic resumable PyTorch checkpoints.
- A DeepDeck Learner statistics dashboard that plots every objective separately.
- A variable-size legal-action policy head and scalar critic on the shared state
  representation.
- On-policy trajectory collection, generalized advantage estimation, clipped PPO,
  entropy regularization, gradient clipping and resumable RL checkpoints.
- Separate world-model and RL metric streams. Once RL starts, the dashboard shows
  policy/value losses, entropy, approximate KL, clipping fraction, reward,
  episode production, utilization, decisions, latency and throughput.

The world-model configuration uses controlled synthetic sequences. The current
RL bridge uses `TinySelfPlayEnvironment`, a deterministic in-process benchmark
that proves the policy learns from trajectories without an API key. The
dashboard labels these samples as RL episodes and never presents them as Engine
Magic games. Connecting the same policy contract to Engine games requires a
running Engine and local decks. Engine hypothesis simulation, a constrained
particle filter, recursive belief search, reanalysis and league promotion remain
later V13 stages. During Engine PPO, a detached worker schedules one random
staging exhibition from the current checkpoint every 100 updates. Staging
failure is recorded and does not stop local learning.

## Run

```powershell
$env:PYTHONPATH = "src"
python -m oracle_ai.training.world_model --config configs/oracle-ai/v13-world-model-smoke.yaml
python -m oracle_ai.training.rl_v13 --config configs/oracle-ai/v13-world-model-smoke.yaml
```

Starting an existing V13 agent through the controller runs world-model training
until its configured final checkpoint exists, then automatically advances to
PPO. Pretraining metrics are saved as `v13-metrics.jsonl`; RL metrics are saved
as `v13-rl-metrics.jsonl`. RL checkpoints live under
`rl-checkpoints/update-N/rl-model.pt`.

The dashboard endpoint is `GET /api/v1/statistics/training?window=200`. Supported
windows are 50, 200, 1,000, 5,000 and all time. Open the Statistics page and
select the V13 run to inspect each loss curve and its trajectory-production
metrics.
