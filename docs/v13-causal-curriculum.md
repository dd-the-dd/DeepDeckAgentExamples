# V13 causal curriculum proposal

## Goal

V13 should learn Magic strategy only as fast as it learns to predict the causal
impact of Magic actions.  DeepDeckEngine remains authoritative for rules, legal
actions and state transitions; the learner uses Engine as a teacher to acquire a
compact causal representation that can later support plan discovery, clocks,
opponent beliefs and search.

The core principle is:

```text
understand what an action does
        -> compress the causal impact
        -> understand plans
        -> optimize winning
```

A terminal win must not strongly reinforce a decision whose consequence model was
locally wrong.  We should not teach an agent to win from a state transition it does
not yet understand.

## Why this is not an Engine replacement

Engine already computes an exact transition:

```text
(state, legal action) -> next authoritative state
```

The neural causal model has a different job:

1. predict the structured impact of an action;
2. learn which primitives and relations caused that impact;
3. compress functionally similar impacts into reusable latent capabilities;
4. expose uncertainty so training can focus on misunderstood interactions;
5. support multi-step projection without enumerating every exact Engine branch.

The prediction loss is therefore the measurable teacher signal, while the latent
representation is the product consumed by higher strategic layers.

## V13 learning objective

For a player-relative graph `G_t` and one legal action `a_t`, predict a structured
change graph and several future horizons:

```text
(G_t, a_t) -> delta_G[t+1], delta_G[t+2], ..., latent impact, uncertainty
```

The first implementation may continue to reuse the existing future-feature heads,
but the target schema should evolve from aggregate counters toward structured
changes: zone movement, stack changes, mana, life, counters, tap state, object
creation/removal, triggers, copies, replacements, continuous modifications and
information revealed.

The causal training loss is always fully active:

```text
L_causal = L_next_state
         + lambda_event * L_event_chain
         + lambda_relation * L_relation
         + lambda_multistep * L_multistep
         + lambda_counterfactual * L_counterfactual
```

The exact weighting belongs in a later implementation PR after the structured
transition contract exists.

## RL gate: understanding before strategy

Let `E_t` be normalized prediction error for the transition actually taken.
Convert it into local causal confidence:

```text
confidence_t = exp(-E_t / temperature)
```

Let `M` be global causal maturity derived from a frozen validation set, not from
training loss.  `M = 0` means the held-out world model is still poor and `M = 1`
means the target validation threshold has been reached.

Strategic losses are multiplied by:

```text
strategy_gate = floor + (1 - floor) * M^2 * confidence_t
```

while causal loss remains at weight `1.0`:

```text
L = L_causal
  + strategy_gate * (L_policy + L_value + L_Q + L_plan + ...)
```

Consequences:

- a lucky win after a badly predicted interaction produces little policy/value
  reinforcement;
- a well-understood transition in a mature world model receives full strategic
  reinforcement;
- causal learning never turns off after the agent becomes strategically strong.

The initial pure implementation lives in `oracle_ai.causal.curriculum`; integration
into `PPOLearner.update` should be a separate PR so existing V11/V12 checkpoints
remain behaviorally unchanged.

## Exploration curriculum

Early training should explore new *useful causal scenarios*, not random novelty.
For each candidate action, estimate:

```text
exploration_bonus = uncertainty
                  * predicted_impact
                  * causal_novelty
                  * curriculum_exploration_weight
```

This is a behavior-policy bonus only.  It must never be written into terminal
rewards because novelty is not a strategic objective.

As validation maturity increases, exploration pressure falls automatically and
policy/value/plan learning takes over.

### Prioritized causal replay

After Engine resolves the chosen action, prioritize the transition by:

```text
priority = prediction_error * actual_impact * novelty
```

This directly concentrates compute on interactions that are both misunderstood and
material.  A small impact floor keeps rare representation failures debuggable
without letting zero-impact loops dominate the replay buffer.

## Net impact and ghost loops

Do not equate "battlefield unchanged" with "action had no effect".

A Counterspell may leave the battlefield unchanged while cards and stack objects
move zones.  It is material.  Conversely, a creature that repeatedly taps and
untaps without any triggered consequence may return the material state to exactly
the same place and should not receive exploration credit.

The initial `causal_impact_score` deliberately tracks net material state rather
than strategic value.  It is intended for ghost-loop filtering and exploration
priority only.

Examples:

```text
tap -> untap, no other change
net impact = 0

cast Counterspell
hands/stack/graveyards change
net impact > 0

tap equipped creature -> equipment deals 1
life changes
net impact > 0
```

Passing priority is the important exception: zero material impact can have large
strategic/information value.  It must not be penalized by the ghost detector.
Higher strategic layers remain authoritative for intentional waiting.

## Resource ledger

A later V13.1 PR should add an explicit resource ledger to causal trajectories.
Each produced temporary resource should carry:

```text
(type, amount, source, created_at, expires_at, voluntary)
```

and each consumption should point back to the produced resource when possible.

The auxiliary waste target is:

```text
waste = produced - consumed_before_expiry
```

Only voluntarily created resources should be penalized.  An obligatory trigger
that creates mana which cannot be spent is not a policy error.

Example:

```text
upkeep: choose to produce 1 temporary mana
main phase: mana is gone, no dependent action occurred
=> one unit of voluntary resource waste
```

This is an auxiliary efficiency target, never a replacement for terminal reward.

## Causal graph representation

The Rule IR from DeepDeckEngine should become the static causal skeleton.  Do not
flatten `triggeredAbility`, `event`, `condition`, `cost`, `target`, `effect` and
`replacementEffect` into one text token.

A card rule graph should expose causal primitives and typed relations, for example:

```text
CAST EVENT
   -> satisfies NONCREATURE filter
   -> triggers ABILITY
   -> creates TRIGGER object
   -> resolves MILL primitive
   -> moves N cards library -> graveyard
```

Card instances in a game reference static card/rule nodes while dynamic state
(controller, owner, zone, tapped state, counters, modifications) remains on object
nodes and relation nodes.

The causal network should eventually expose three levels of latent representation:

1. contextualized primitive/object embeddings;
2. reusable causal motif embeddings;
3. quantized or slot-based capability embeddings summarizing what the current
   deck/state can causally produce.

The third level is the interface to future Plan Discovery.

## VQ impact abstraction

V10 already proves that DeepDeckLearner can quantize context-dependent action/event
representations and reconstruct future features.  V13 should reuse that lesson but
move the abstraction toward causal primitives and impact rather than treating it as
an isolated V10 policy family.

The codebook should prefer functional equivalence:

```text
pay 2 mana -> draw 2
pay 2 life -> draw 2
sacrifice artifact -> draw 2
```

share a strong `card access` component while retaining different cost relations.
Magnitude and target remain parameters, not separate memorized concepts.

## Multi-step learning

One-step correctness is necessary but not sufficient.  Magic contains delayed
causal chains such as:

```text
creature enters with summoning sickness
-> next turn it can attack
-> attack triggers create a token
-> future combat contains the original creature plus produced tokens
```

Train local transitions and multi-horizon rollouts together.  The world model must
learn deterministic public transitions exactly where possible and calibrated
distributions for hidden/stochastic transitions such as unknown draws.

For hidden draws, the target is not "guess the exact top card".  The representation
should support distributions over relevant capabilities, conditioned on known deck
composition and revealed information.

## Validation and scientific benchmark

Causal understanding is intentionally a classic measurable ML task.  Maintain
frozen train/validation/test splits with no leakage.

The held-out suites should include:

- known cards in unseen states;
- unseen cards composed from known Rule IR primitives;
- unseen combinations of known primitives;
- boards with many irrelevant distractor permanents;
- replacement effects and continuous effects;
- stack/copy/trigger interactions;
- combat and delayed attack triggers;
- multi-step rollouts longer than training horizons;
- hidden draws with known deck lists;
- counterfactual variants where one possible cause is removed.

Report at minimum:

```text
causal train loss
causal validation loss
frozen test loss
next-transition error
event-chain accuracy
relation precision/recall
multi-step rollout error
uncertainty calibration
OOD/compositional error
impactful-transition error
```

The strategic learner must use validation maturity, never training loss, for its
curriculum gate.

## Delivery sequence

### V13.0 - causal curriculum foundation

This proposal PR:

- pure curriculum/gating functions;
- coarse net-impact scoring;
- causal replay/exploration priority helpers;
- unit tests;
- architecture contract.

No existing policy behavior changes.

### V13.1 - structured causal targets

- Engine/Learner transition contract;
- structured delta graph;
- event-chain supervision;
- resource ledger;
- frozen train/validation/test benchmark.

### V13.2 - causal world model

- graph encoder over Rule IR + dynamic game graph;
- action-conditioned transition decoder;
- uncertainty;
- multi-step rollout;
- latent capability/VQ abstraction;
- counterfactual training.

### V13.3 - gated RL integration

- attach local causal prediction error to `Transition`;
- gate PPO/value/Q/plan losses by causal confidence and validation maturity;
- keep causal losses at full weight;
- publish causal and strategic curves separately.

### V13.4 - active causal exploration

- behavior-policy bonus `uncertainty * impact * novelty`;
- prioritized causal replay;
- targeted scenario generation around high-error concepts;
- automatic shift from exploration to plan/win optimization as validation improves.

### Later - Plan Discovery

Plan Discovery should consume causal capability embeddings rather than raw Oracle
text.  Its job is to discover multiple routes from resources/capabilities to a win,
assign cards/primitives to plan roles, estimate plan clocks, blockers and
protectors, and expose plan tokens to the recurrent AlphaStar-style policy.
