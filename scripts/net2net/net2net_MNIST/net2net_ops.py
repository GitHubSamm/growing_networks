import torch
import torch.nn as nn
import numpy as np


def _find_layer(model, layer_name):

    for name, layer in model.named_modules():
        if name == layer_name:
            return layer


def net2wider_linear(
    layer: nn.Module, next_layer: nn.Module, new_width=None, noise_std=0.01
):
    """
    Return a wider version of a Linear layer with function-preserving weight transfer.

    Args:
        layer (nn.Module): The Linear layer to be widened.
        next_layer (nn.Module): The next Linear layer to adjust accordingly.
        new_width (int, optional): The desired output size of the new layer. Must be > current width.
        noise_std (float, optional): Standard deviation of noise added during duplication. Default is 0.01.

    Returns:
        tuple: The widened Linear layer and the adjusted next Linear layer.

    Example:
        >>> layer = nn.Linear(10, 20)
        >>> next_layer = nn.Linear(20, 30)
        >>> wider_layer, adjusted_next = net2wider_linear(layer, next_layer, new_width=25)

    @author Sam Collin
    """

    # Add safety checks
    if new_width is None:
        new_width = layer.out_features + 1
    elif new_width <= layer.out_features:
        raise ValueError(
            "The new layer size must be greater than the current layer size"
        )

    n_add_weights = new_width - layer.out_features

    # Define our 2 new layers
    new_layer = nn.Linear(layer.in_features, new_width)
    new_next_layer = nn.Linear(new_width, next_layer.out_features)

    # First, we will copy the already trained weights
    new_layer.weight.data[: layer.out_features] = layer.weight.data
    new_layer.bias.data[: layer.out_features] = layer.bias.data
    new_next_layer.weight.data[:, : layer.out_features] = next_layer.weight.data

    for i in range(n_add_weights):

        idx_split = np.random.randint(0, layer.out_features)

        # Retrieve the node that will be duplicated
        weights_splitted_node = layer.weight.data[idx_split]
        new_layer.weight.data[layer.out_features + i] = weights_splitted_node

        # Duplicate also the bias in the new layer
        new_layer.bias.data[layer.out_features + i] = layer.bias.data[idx_split]

        # Add noise handling here
        if noise_std:
            pass

        # To handle the next layer and perform function preservig, we divide
        # the weights associated with the splitted node and copy this divided
        # one
        new_next_layer.weight.data[:, idx_split] /= 2
        new_next_layer.weight.data[:, layer.out_features + i] = (
            new_next_layer.weight.data[:, idx_split]
        )

    return new_layer, new_next_layer


def net2deeper_linear(layer: nn.Module):

    old_bias = layer.bias.data

    duplicated_layer = nn.Linear(layer.out_features, layer.out_features)
    duplicated_layer.weight.data = torch.eye(old_bias.size(0), dtype=old_bias.dtype)
    duplicated_layer.bias.data = torch.zeros_like(old_bias)

    return nn.Sequential(layer, duplicated_layer)
