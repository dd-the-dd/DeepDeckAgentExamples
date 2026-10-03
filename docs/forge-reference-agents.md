# Forge-style reference agents

DeepDeckLearner includes three transparent programmatic baselines:
`forge-cautious`, `forge-balanced`, and `forge-aggressive`. They share a
clean-room heuristic policy and differ in combat risk and reaction thresholds.

The implementation follows the same broad separation visible in Forge: the
rules engine produces playable actions, while an AI controller evaluates
spells, costs, targets, combat, and timing. DeepDeck keeps that separation by
ranking only exact legal action IDs emitted by DeepDeck Engine. The policy
handles mulligans, land development, spell and ability selection, target value,
attacks, blocks, and discards.

These agents are useful comparison baselines, not a claim of move-for-move
parity with Forge. Forge's AI is tightly coupled to Forge's Java game model and
is distributed under GPL-3.0, whereas DeepDeckLearner is MIT. No Forge source
code is copied, translated, linked, vendored, or required at runtime.

Run a profile like any other SDK agent:

```sh
deepdeck-example forge-balanced --target local --start-local-game
deepdeck-example forge-aggressive --target ddl --speed 100ms
```

Upstream references:

- Forge repository and license: <https://github.com/Card-Forge/forge>
- Forge AI module: <https://github.com/Card-Forge/forge/tree/master/forge-ai>
- Forge AI controller: <https://github.com/Card-Forge/forge/blob/master/forge-ai/src/main/java/forge/ai/AiController.java>
