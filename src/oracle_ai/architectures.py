from __future__ import annotations

from typing import Any

import torch

from oracle_ai.encoding import HashingObservationEncoder
from oracle_ai.encoding_v2 import StructuredObservationEncoder
from oracle_ai.encoding_v3 import CyclicStructuredObservationEncoder
from oracle_ai.encoding_v4 import OracleStructuredObservationEncoder
from oracle_ai.encoding_v6 import PlanningObservationEncoder
from oracle_ai.encoding_v7 import StrategicPlanningObservationEncoder
from oracle_ai.encoding_v11 import AlphaStarObservationEncoder
from oracle_ai.encoding_v13 import GraphObservationEncoderV13
from oracle_ai.model import MagicTransformerActorCritic, ModelConfig
from oracle_ai.model_v2 import MagicTransformerActorCriticV2, ModelConfigV2
from oracle_ai.model_v3 import MagicTransformerActorCriticV3, ModelConfigV3
from oracle_ai.model_v4 import MagicTransformerActorCriticV4, ModelConfigV4
from oracle_ai.model_v5 import MagicTransformerActorCriticV5, ModelConfigV5
from oracle_ai.model_v6 import MagicTransformerActorCriticV6, ModelConfigV6
from oracle_ai.model_v7 import MagicTransformerActorCriticV7, ModelConfigV7
from oracle_ai.model_v9 import MagicTransformerActorCriticV9, ModelConfigV9
from oracle_ai.model_v10 import MagicTransformerActorCriticV10, ModelConfigV10
from oracle_ai.model_v11 import MagicTransformerActorCriticV11, ModelConfigV11
from oracle_ai.model_v12 import MagicTransformerActorCriticV12, ModelConfigV12
from oracle_ai.model_v13 import GraphBeliefWorldModelV13, ModelConfigV13
from oracle_ai.training.future_features import FUTURE_FEATURE_NAMES

_LEGACY_FUTURE_FEATURE_NAMES = (
    "life",
    "hand_count",
    "library_count",
    "graveyard_count",
    "exile_count",
    "land_count",
    "mana_available",
    "creature_count",
    "permanent_count",
    "enchantment_count",
    "artifact_count",
    "total_power",
    "total_toughness",
    "battlefield_mana_value",
    "poison_counters",
    "commander_damage_received",
    "cards_drawn",
    "spells_cast",
    "creatures_died",
    "damage_dealt",
    "mana_produced",
)

PolicyModel = (
    MagicTransformerActorCritic
    | MagicTransformerActorCriticV2
    | MagicTransformerActorCriticV3
    | MagicTransformerActorCriticV4
    | MagicTransformerActorCriticV5
    | MagicTransformerActorCriticV6
    | MagicTransformerActorCriticV7
    | MagicTransformerActorCriticV9
    | MagicTransformerActorCriticV10
    | MagicTransformerActorCriticV11
    | MagicTransformerActorCriticV12
    | GraphBeliefWorldModelV13
)
PolicyEncoder = (
    HashingObservationEncoder
    | StructuredObservationEncoder
    | CyclicStructuredObservationEncoder
    | OracleStructuredObservationEncoder
    | PlanningObservationEncoder
    | StrategicPlanningObservationEncoder
    | AlphaStarObservationEncoder
    | GraphObservationEncoderV13
)

_MODEL_TYPES = {
    "hashing-v1": (ModelConfig, MagicTransformerActorCritic),
    "structured-v2": (ModelConfigV2, MagicTransformerActorCriticV2),
    "structured-v3": (ModelConfigV3, MagicTransformerActorCriticV3),
    "structured-v4": (ModelConfigV4, MagicTransformerActorCriticV4),
    "structured-v5": (ModelConfigV5, MagicTransformerActorCriticV5),
    "structured-v6": (ModelConfigV6, MagicTransformerActorCriticV6),
    "structured-v7": (ModelConfigV7, MagicTransformerActorCriticV7),
    "structured-v9": (ModelConfigV9, MagicTransformerActorCriticV9),
    "structured-v10": (ModelConfigV10, MagicTransformerActorCriticV10),
    "structured-v11": (ModelConfigV11, MagicTransformerActorCriticV11),
    "structured-v12": (ModelConfigV12, MagicTransformerActorCriticV12),
    "graph-belief-v13": (ModelConfigV13, GraphBeliefWorldModelV13),
    "structured-world-v13": (ModelConfigV13, GraphBeliefWorldModelV13),
}


def validate_model_config(config: dict[str, Any]) -> None:
    """Validate a serialized architecture without allocating model weights."""
    values = dict(config)
    architecture = str(values.pop("architecture", "hashing-v1"))
    model_types = _MODEL_TYPES.get(architecture)
    if model_types is None:
        raise ValueError(f"unsupported Oracle AI architecture: {architecture}")
    config_type, _ = model_types
    config_type(**values)


def build_model(config: dict[str, Any]) -> PolicyModel:
    values = dict(config)
    architecture = str(values.pop("architecture", "hashing-v1"))
    model_types = _MODEL_TYPES.get(architecture)
    if model_types is None:
        raise ValueError(f"unsupported Oracle AI architecture: {architecture}")
    config_type, model_type = model_types
    return model_type(config_type(**values))


def encoder_for_model(
    model: PolicyModel,
    *,
    max_state_tokens: int = 512,
) -> PolicyEncoder:
    if isinstance(model, GraphBeliefWorldModelV13):
        return GraphObservationEncoderV13()
    if isinstance(model, MagicTransformerActorCriticV11):
        return AlphaStarObservationEncoder(
            word_vocab_size=model.config.word_vocab_size,
            max_words=model.config.max_words,
            max_relative_players=model.config.max_relative_players,
            max_state_tokens=max_state_tokens,
            player_angle_steps=model.config.player_angle_steps,
            max_events=model.config.max_events,
            max_deck_cards=model.config.max_deck_cards,
            max_delta_tokens=model.config.max_delta_tokens,
            max_commander_cards=model.config.max_commander_cards,
        )
    if isinstance(
        model,
        (
            MagicTransformerActorCriticV7,
            MagicTransformerActorCriticV9,
            MagicTransformerActorCriticV10,
        ),
    ):
        return StrategicPlanningObservationEncoder(
            word_vocab_size=model.config.word_vocab_size,
            max_words=model.config.max_words,
            max_relative_players=model.config.max_relative_players,
            max_state_tokens=max_state_tokens,
            player_angle_steps=model.config.player_angle_steps,
            max_events=model.config.max_events,
            max_deck_cards=model.config.max_deck_cards,
        )
    if isinstance(model, MagicTransformerActorCriticV6):
        return PlanningObservationEncoder(
            word_vocab_size=model.config.word_vocab_size,
            max_words=model.config.max_words,
            max_relative_players=model.config.max_relative_players,
            max_state_tokens=max_state_tokens,
            player_angle_steps=model.config.player_angle_steps,
            max_events=model.config.max_events,
        )
    if isinstance(
        model, (MagicTransformerActorCriticV4, MagicTransformerActorCriticV5)
    ):
        return OracleStructuredObservationEncoder(
            word_vocab_size=model.config.word_vocab_size,
            max_words=model.config.max_words,
            max_relative_players=model.config.max_relative_players,
            max_state_tokens=max_state_tokens,
            player_angle_steps=model.config.player_angle_steps,
        )
    if isinstance(model, MagicTransformerActorCriticV3):
        return CyclicStructuredObservationEncoder(
            word_vocab_size=model.config.word_vocab_size,
            max_words=model.config.max_words,
            max_relative_players=model.config.max_relative_players,
            max_state_tokens=max_state_tokens,
            player_angle_steps=model.config.player_angle_steps,
        )
    if isinstance(model, MagicTransformerActorCriticV2):
        return StructuredObservationEncoder(
            word_vocab_size=model.config.word_vocab_size,
            max_words=model.config.max_words,
            max_relative_players=model.config.max_relative_players,
            max_state_tokens=max_state_tokens,
        )
    return HashingObservationEncoder(
        feature_dim=model.config.feature_dim,
        max_state_tokens=max_state_tokens,
    )


def _future_feature_index_pairs(
    outer_count: int,
    source_names: tuple[str, ...],
    target_names: tuple[str, ...],
) -> list[tuple[int, int]]:
    target_by_name = {name: index for index, name in enumerate(target_names)}
    pairs: list[tuple[int, int]] = []
    for outer_index in range(outer_count):
        for source_index, source_name in enumerate(source_names):
            target_name = (
                "total_hand_count" if source_name == "hand_count" else source_name
            )
            pairs.append(
                (
                    outer_index * len(source_names) + source_index,
                    outer_index * len(target_names) + target_by_name[target_name],
                )
            )
    return pairs


def _copy_linear_output_features(
    source: torch.nn.Linear,
    target: torch.nn.Linear,
    pairs: list[tuple[int, int]],
) -> None:
    with torch.no_grad():
        for source_index, target_index in pairs:
            target.weight[target_index].copy_(source.weight[source_index])
            if source.bias is not None and target.bias is not None:
                target.bias[target_index].copy_(source.bias[source_index])


def _copy_expanded_future_feature_heads(
    source: MagicTransformerActorCriticV9,
    target: MagicTransformerActorCriticV9,
) -> None:
    source_names = _LEGACY_FUTURE_FEATURE_NAMES
    target_names = tuple(FUTURE_FEATURE_NAMES)
    horizons = source.config.future_horizons
    players = source.config.future_player_slots
    future_groups = horizons * players

    # The consequence output interleaves mean and log-variance for every feature.
    consequence_pairs = []
    for source_index, target_index in _future_feature_index_pairs(
        future_groups,
        source_names,
        target_names,
    ):
        consequence_pairs.extend(
            (
                (source_index * 2, target_index * 2),
                (source_index * 2 + 1, target_index * 2 + 1),
            )
        )
    _copy_linear_output_features(
        source.consequence_head[-1],
        target.consequence_head[-1],
        consequence_pairs,
    )

    future_pairs = _future_feature_index_pairs(
        future_groups,
        source_names,
        target_names,
    )
    source_future_values = future_groups * len(source_names)
    target_future_values = future_groups * len(target_names)
    future_encoder_pairs = [
        *future_pairs,
        *[
            (
                source_index + source_future_values,
                target_index + target_future_values,
            )
            for source_index, target_index in future_pairs
        ],
    ]
    with torch.no_grad():
        for source_index, target_index in future_encoder_pairs:
            target.future_encoder[0].weight[target_index].copy_(
                source.future_encoder[0].weight[source_index]
            )
            target.future_encoder[0].bias[target_index].copy_(
                source.future_encoder[0].bias[source_index]
            )
            target.future_encoder[1].weight[:, target_index].copy_(
                source.future_encoder[1].weight[:, source_index]
            )

    belief_pairs = _future_feature_index_pairs(
        players,
        source_names,
        target_names,
    )
    _copy_linear_output_features(
        source.belief_prediction_head[-1],
        target.belief_prediction_head[-1],
        belief_pairs,
    )

    if isinstance(source, MagicTransformerActorCriticV10) and isinstance(
        target,
        MagicTransformerActorCriticV10,
    ):
        _copy_linear_output_features(
            source.event_decoder[-1],
            target.event_decoder[-1],
            future_pairs,
        )


def upgrade_model(
    source: PolicyModel,
    target: PolicyModel,
) -> PolicyModel:
    if type(source) is type(target) and isinstance(
        source,
        (MagicTransformerActorCriticV9, MagicTransformerActorCriticV10),
    ):
        source_config = source.export_config()
        target_config = target.export_config()
        differing_fields = {
            name
            for name in source_config.keys() | target_config.keys()
            if source_config.get(name) != target_config.get(name)
        }
        if differing_fields != {"future_feature_dim"}:
            raise ValueError(
                "same-family future-schema upgrade only supports "
                f"future_feature_dim, not {sorted(differing_fields)}"
            )
        if source.config.future_feature_dim != len(
            _LEGACY_FUTURE_FEATURE_NAMES
        ) or target.config.future_feature_dim != len(FUTURE_FEATURE_NAMES):
            raise ValueError(
                "unsupported future-feature schema expansion: "
                f"{source.config.future_feature_dim} -> "
                f"{target.config.future_feature_dim}"
            )
        device = next(source.parameters()).device
        target = target.to(device)
        target_state = target.state_dict()
        compatible = {
            name: value
            for name, value in source.state_dict().items()
            if name in target_state and target_state[name].shape == value.shape
        }
        target.load_state_dict(compatible, strict=False)
        _copy_expanded_future_feature_heads(source, target)
        return target

    if (
        type(source) is MagicTransformerActorCriticV9
        and type(target) is MagicTransformerActorCriticV10
    ):
        source_config = source.export_config()
        target_config = target.export_config()
        structural_fields = {
            "numeric_dim",
            "d_model",
            "layers",
            "heads",
            "feedforward_dim",
            "word_vocab_size",
            "max_words",
            "max_relative_players",
            "player_angle_steps",
            "action_layers",
            "plan_layers",
            "future_horizons",
            "future_player_slots",
            "future_feature_dim",
        }
        if any(
            source_config.get(name) != target_config.get(name)
            for name in structural_fields
        ):
            raise ValueError("V9 to V10 upgrade requires identical shared dimensions")
        device = next(source.parameters()).device
        target = target.to(device)
        target_state = target.state_dict()
        compatible = {
            name: value
            for name, value in source.state_dict().items()
            if name in target_state and target_state[name].shape == value.shape
        }
        target.load_state_dict(compatible, strict=False)
        return target

    if (
        type(source) is MagicTransformerActorCriticV5
        and type(target) is MagicTransformerActorCriticV6
    ):
        source_config = source.export_config()
        target_config = target.export_config()
        structural_fields = {
            "numeric_dim",
            "d_model",
            "layers",
            "heads",
            "feedforward_dim",
            "max_relative_players",
            "player_angle_steps",
            "action_layers",
        }
        if any(
            source_config.get(name) != target_config.get(name)
            for name in structural_fields
        ):
            raise ValueError(
                "V5 to V6 upgrade requires identical state-model dimensions"
            )
        device = next(source.parameters()).device
        target = target.to(device)
        target_state = target.state_dict()
        compatible = {
            name: value
            for name, value in source.state_dict().items()
            if not name.startswith("word_encoder.")
            and name != "token_type_embedding.weight"
            and name in target_state
            and target_state[name].shape == value.shape
        }
        target.load_state_dict(compatible, strict=False)
        with torch.no_grad():
            source_types = source.token_type_embedding.weight
            target.token_type_embedding.weight[: source_types.shape[0]].copy_(
                source_types
            )
        return target

    if (
        type(source) is MagicTransformerActorCriticV4
        and type(target) is MagicTransformerActorCriticV5
    ):
        source_config = source.export_config()
        target_config = target.export_config()
        structural_fields = {
            "numeric_dim",
            "d_model",
            "layers",
            "heads",
            "feedforward_dim",
            "word_vocab_size",
            "max_words",
            "max_relative_players",
            "token_type_count",
            "player_angle_steps",
        }
        if any(
            source_config.get(name) != target_config.get(name)
            for name in structural_fields
        ):
            raise ValueError(
                "V4 to V5 upgrade requires identical state-model dimensions"
            )
        device = next(source.parameters()).device
        target = target.to(device)
        target_state = target.state_dict()
        compatible = {
            name: value
            for name, value in source.state_dict().items()
            if name in target_state and target_state[name].shape == value.shape
        }
        result = target.load_state_dict(compatible, strict=False)
        expected_missing = {
            name for name in target_state if name.startswith("action_conditioning.")
        } | {
            "action_output_norm.weight",
            "action_output_norm.bias",
            "policy_score.weight",
            "context_gate",
        }
        if set(result.missing_keys) != expected_missing or result.unexpected_keys:
            raise ValueError(
                "V4 to V5 upgrade found incompatible model parameters: "
                f"missing={result.missing_keys}, unexpected={result.unexpected_keys}"
            )
        return target

    if (
        type(source) is MagicTransformerActorCriticV3
        and type(target) is MagicTransformerActorCriticV4
    ):
        if source.export_config() != target.export_config():
            raise ValueError("V3 to V4 upgrade requires identical model dimensions")
        device = next(source.parameters()).device
        target = target.to(device)
        target.load_state_dict(source.state_dict(), strict=True)
        return target

    if not (
        type(source) is MagicTransformerActorCriticV2
        and type(target) is MagicTransformerActorCriticV3
    ):
        raise ValueError(
            f"unsupported architecture upgrade: "
            f"{source.model_family} -> {target.model_family}"
        )
    if source.config.max_relative_players != target.config.max_relative_players:
        raise ValueError("V2 to V3 upgrade requires the same maximum player count")

    device = next(source.parameters()).device
    target = target.to(device)
    target_state = target.state_dict()
    compatible = {
        name: value
        for name, value in source.state_dict().items()
        if name in target_state and target_state[name].shape == value.shape
    }
    result = target.load_state_dict(compatible, strict=False)
    expected_missing = {
        "relative_player_projection.weight",
        "no_player_embedding",
    }
    if set(result.missing_keys) != expected_missing or result.unexpected_keys:
        raise ValueError(
            "V2 to V3 upgrade found incompatible model parameters: "
            f"missing={result.missing_keys}, unexpected={result.unexpected_keys}"
        )

    with torch.no_grad():
        old_positions = source.relative_player_embedding.weight
        player_positions = old_positions[:-1]
        position_count = player_positions.shape[0]
        angles = torch.arange(
            position_count,
            device=device,
            dtype=player_positions.dtype,
        ) * (2.0 * torch.pi / position_count)
        coordinates = torch.stack((torch.sin(angles), torch.cos(angles)), dim=-1)
        projection = torch.linalg.lstsq(coordinates, player_positions).solution
        target.relative_player_projection.weight.copy_(projection.transpose(0, 1))
        target.no_player_embedding.copy_(old_positions[-1:])
    return target
