"""Checkpoint input-layer migration performed before strict state loading."""

import torch


INPUT_WEIGHT_KEYS = ("actor.0.weight", "critic.0.weight")


def adapt_observation_input_state(loaded_state, current_state):
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
        widened[:, : old_weight.shape[1]].copy_(old_weight)
        adapted_state[key] = widened
        adaptations.append(
            f"{key} {old_weight.shape[1]}->{new_weight.shape[1]}"
        )
    return adapted_state, adaptations
