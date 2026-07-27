"""Checkpoint input-layer migration performed before strict state loading."""

import torch


INPUT_WEIGHT_KEYS = ("actor.0.weight", "critic.0.weight")


def adapt_observation_input_state(
    loaded_state,
    current_state,
    column_migration=None,
):
    adapted_state = dict(loaded_state)
    adaptations = []
    for key in INPUT_WEIGHT_KEYS:
        if key not in adapted_state or key not in current_state:
            raise ValueError(f"checkpoint observation adapter missing key: {key}")
        old_weight = adapted_state[key]
        new_weight = current_state[key]
        if old_weight.shape == new_weight.shape:
            continue
        if (
            old_weight.ndim != 2
            or new_weight.ndim != 2
            or old_weight.shape[0] != new_weight.shape[0]
            or old_weight.shape[1] > new_weight.shape[1]
        ):
            raise ValueError(
                "Observation adapter only supports widening actor/critic "
                f"first input layers with unchanged output width, got {key}: "
                f"{tuple(old_weight.shape)} -> {tuple(new_weight.shape)}"
            )
        widened = torch.zeros_like(new_weight)
        if column_migration is None:
            widened[:, : old_weight.shape[1]].copy_(old_weight)
        else:
            source_width = int(column_migration["source_width"])
            destination_width = int(
                column_migration["destination_width"]
            )
            if (
                old_weight.shape[1] != source_width
                or new_weight.shape[1] != destination_width
            ):
                raise ValueError(
                    "Observation column migration width mismatch for "
                    f"{key}: configured {source_width}->{destination_width}, "
                    f"actual {old_weight.shape[1]}->{new_weight.shape[1]}"
                )
            copy_prefix = int(column_migration.get("copy_prefix", 0))
            widened[:, :copy_prefix].copy_(
                old_weight[:, :copy_prefix]
            )
            for mapping in column_migration.get("column_mappings", []):
                source = int(mapping["source"])
                destination = int(mapping["destination"])
                scale = float(mapping.get("scale", 1.0))
                widened[:, destination].copy_(
                    old_weight[:, source] * scale
                )
        adapted_state[key] = widened
        adaptations.append(
            f"{key} {old_weight.shape[1]}->{new_weight.shape[1]}"
        )
    return adapted_state, adaptations
