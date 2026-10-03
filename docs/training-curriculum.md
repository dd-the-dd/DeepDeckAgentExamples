# Focused training curriculum

DeepDeckLearner can train a V13 policy on short, controlled Engine games in
addition to ordinary self-play. Open **Statistics → Training cockpit**, enable
**Focused game curriculum**, choose one or more scenarios, save the plan, and
start the reinforcement-learning phase.

The built-in catalog currently includes:

- opening-hand sprints at easy, medium, and hard levels, scaling the candidate
  pool from twenty to sixty cards while tightening the round deadline;
- a controlled Reanimator opening-hand clinic whose twenty-card pool gives dense
  rewards for selecting mana, discard, a reanimation target, and a reanimation spell;
- three zero-cost sequencing levels, progressing from a small forgiving action
  space to a fifty-card pool with a round-three deadline;
- guided, standard, and expert combo rehearsals that vary both search space and
  execution deadline;
- deterministic known-combo puzzles with exact opening hands for Reanimator,
  Show and Tell, Aluren, Dark Depths, and Isochron Reversal;
- three seeded, non-neural Miracle opponents with increasing pressure; their
  zero-cost Miracle spells span four printed mana costs and exercise life gain,
  direct damage, card draw, and forced discard;
- three reproducible best-of-three Legacy matchups for sideboard decisions:
  Dimir vs Reanimator, Lands vs Sneak and Show, and Initiative vs Delver. Each
  matchup defines both the cards to bring in and the cards to take out.

Each scenario card can be launched by itself with **Train only this**. The
statistics view ranks scenario skills weakest-first and reports success rate,
average completion turn, recent attempts, and the adaptive sampling weight.
Sideboard evidence separately reports correct cards brought in and correct
cards removed; mastery combines both halves of the plan.

When adaptive difficulty is enabled, the sampler restores prior scenario
results from `v13-engine-games.jsonl` and gives more weight to scenarios with a
low mastery rate. Combo mastery includes partial ordered-line progress, so a
three-of-four attempt is distinguished from a complete miss. Opening-hand role
coverage provides the same partial-credit signal before the game is won. Every game record includes `scenarioId`, `scenarioFamily`,
`objective`, `difficulty`, `objectiveCompleted`, the learner reward, and the
current curriculum statistics.

Fast-win and combo-win games also shape reward from irreversible damage
progress against the training anchor. The evidence panel reports this progress,
so zero-win scenarios still reveal whether the policy is getting closer.

The progressive variants deliberately keep the same reward semantics. This lets
the policy transfer a skill learned in the easy exercise to a larger search
space instead of learning an unrelated objective at every level.

Fixed-seed curriculum evaluation retries an individual scenario up to three
times after a transient Engine failure. Recovered scenarios and Engine error
counts are preserved in the evaluation artifact and shown in the dashboard.

The YAML representation is:

```yaml
trainingCurriculum:
  enabled: true
  adaptive: true
  scenarioIds: [] # empty selects every built-in scenario
```

V13 retains only the three newest complete RL checkpoints by default. Override
this with `rl.max_checkpoints` when a run needs a different retention policy.
