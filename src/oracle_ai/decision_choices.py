from __future__ import annotations

from collections import defaultdict
from typing import Any


def _definition_id(instance_id: str) -> str:
    parts = instance_id.split(":")
    return ":".join(parts[1:-1]) if len(parts) >= 3 else instance_id


def _expand_sideboard_actions(
    decision: dict[str, Any],
    actions: list[dict[str, Any]],
    card_names_by_definition_id: dict[str, str],
    sideboard_target_card_names: tuple[str, ...],
    sideboard_cut_card_names: tuple[str, ...],
    maximum_actions: int = 256,
) -> list[dict[str, Any]]:
    choice = decision.get("choice") or {}
    if len(actions) != 1:
        return actions
    engine_action = actions[0]
    engine_decisions = dict(engine_action.get("decisions") or {})
    initial = [str(value) for value in engine_decisions.get("initialMainDeckIds", [])]
    candidates = [str(value) for value in choice.get("candidateCardInstanceIds", [])]
    decision_id = str(choice.get("decisionId", ""))
    if not initial or not candidates or not decision_id:
        return actions

    initial_set = set(initial)
    sideboard = [value for value in candidates if value not in initial_set]
    if not sideboard:
        return actions

    grouped_main: dict[str, list[str]] = defaultdict(list)
    grouped_sideboard: dict[str, list[str]] = defaultdict(list)
    for instance_id in initial:
        grouped_main[_definition_id(instance_id)].append(instance_id)
    for instance_id in sideboard:
        grouped_sideboard[_definition_id(instance_id)].append(instance_id)

    def label(definition_id: str) -> str:
        return card_names_by_definition_id.get(definition_id, definition_id)

    def policy_action(
        action_id: str,
        action_label: str,
        final_main: list[str],
    ) -> dict[str, Any]:
        return {
            **engine_action,
            "id": action_id,
            "label": action_label,
            "decisions": {
                **engine_decisions,
                decision_id: final_main,
            },
            "_engineActionId": engine_action["id"],
            "_cardInstanceIds": final_main,
        }

    expanded = [
        policy_action(
            f"{engine_action['id']}:keep",
            "Keep the current main deck",
            initial,
        )
    ]
    target_names = set(sideboard_target_card_names)
    target_instances = [
        instance_id
        for definition_id, instances in sorted(grouped_sideboard.items())
        if label(definition_id) in target_names
        for instance_id in instances
    ]
    main_groups = list(sorted(grouped_main.items()))
    if target_instances and len(initial) >= len(target_instances):
        preferred_cut_instances = [
            instance_id
            for cut_name in sideboard_cut_card_names
            for definition_id, instances in main_groups
            if label(definition_id) == cut_name
            for instance_id in instances
        ][: len(target_instances)]
        seen_known_plans: set[tuple[str, ...]] = set()
        for offset in range(min(16, len(main_groups))):
            rotated = main_groups[offset:] + main_groups[:offset]
            removed = list(preferred_cut_instances)
            for _, instances in rotated:
                needed = len(target_instances) - len(removed)
                if needed <= 0:
                    break
                chosen = [value for value in instances if value not in removed][:needed]
                if chosen:
                    removed.extend(chosen)
            if len(removed) != len(target_instances):
                continue
            cut_names = list(
                dict.fromkeys(label(_definition_id(instance_id)) for instance_id in removed)
            )
            removed_set = set(removed)
            final_main = [value for value in initial if value not in removed_set]
            final_main.extend(target_instances)
            plan_key = tuple(sorted(final_main))
            if plan_key in seen_known_plans:
                continue
            seen_known_plans.add(plan_key)
            expanded.append(
                policy_action(
                    f"{engine_action['id']}:known-plan:{offset}",
                    (
                        f"Known matchup plan: bring in {len(target_instances)} cards; "
                        f"cut {', '.join(cut_names)}"
                    ),
                    final_main,
                )
            )
    for side_definition_id, side_instances in sorted(grouped_sideboard.items()):
        for main_definition_id, main_instances in sorted(grouped_main.items()):
            swap_count = min(len(side_instances), len(main_instances))
            removed = set(main_instances[:swap_count])
            final_main = [value for value in initial if value not in removed]
            final_main.extend(side_instances[:swap_count])
            expanded.append(
                policy_action(
                    f"{engine_action['id']}:swap:{len(expanded)}",
                    (
                        f"Sideboard {swap_count} {label(side_definition_id)} "
                        f"for {label(main_definition_id)}"
                    ),
                    final_main,
                )
            )
            if len(expanded) >= maximum_actions:
                return expanded
    return expanded


def expand_policy_actions(
    decision: dict[str, Any],
    card_names_by_definition_id: dict[str, str] | None = None,
    sideboard_target_card_names: tuple[str, ...] = (),
    sideboard_cut_card_names: tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    actions = [dict(action) for action in decision.get("options", [])]
    choice = decision.get("choice")
    if (
        isinstance(choice, dict)
        and choice.get("kind") == "cardSelection"
        and str(decision.get("kind", "")).casefold() in {"sideboard", "sideboarding"}
    ):
        return _expand_sideboard_actions(
            decision,
            actions,
            card_names_by_definition_id or {},
            sideboard_target_card_names,
            sideboard_cut_card_names,
        )
    if not isinstance(choice, dict) or choice.get("kind") != "numberSelection":
        return actions
    if len(actions) != 1:
        raise ValueError("number selection must expose exactly one engine action")
    minimum = int(choice.get("minimum", 0))
    maximum = int(choice.get("maximum", minimum))
    if maximum < minimum or maximum - minimum > 1_000:
        raise ValueError("number selection range is invalid or too large")
    decision_id = str(choice.get("decisionId", "numberValue"))
    engine_action = actions[0]
    return [
        {
            **engine_action,
            "id": f"{engine_action['id']}:value:{number_value}",
            "label": f"{engine_action.get('label', 'Choose a number')}: {number_value}",
            "decisions": {
                **dict(engine_action.get("decisions") or {}),
                decision_id: number_value,
            },
            "_engineActionId": engine_action["id"],
            "_numberValue": number_value,
        }
        for number_value in range(minimum, maximum + 1)
    ]
