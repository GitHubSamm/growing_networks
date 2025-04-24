"""
net2net_ops.py

This file implements Net2Net operations for dynamically growing neural networks
while preserving the function learned so far.

It includes:
- Net2WiderNet for widening linear layers with optional BatchNorm handling
- Net2DeeperNet for inserting identity-initialized layers
- Utilities for locating layers by name within PyTorch models

These methods are used to grow networks during training without reinitializing,
as part of a strategy to progressively expand capacity.

Reference:
Chen, T., Goodfellow, I., & Shlens, J. (2016). Net2Net: Accelerating Learning via Knowledge Transfer.
https://arxiv.org/abs/1511.05641

@author Sam Collin
"""

import torch
import torch.nn as nn
import numpy as np
import speechbrain as sb


def _find_layer(model, layer_name):

    for name, layer in model.named_modules():
        if name == layer_name:
            return layer


def net2wider_linear(
    layer: nn.Module,
    next_layer: nn.Module,
    new_width=None,
    norm_layer=None,
    noise_std=0.01,
    last_block=False,
):
    """Return a wider version of a Linear layer with function-preserving
    weight transfer.

    This function implements a Net2WiderNet operation by replicating
    existing neurons and adjusting the subsequent layer to preserve the
    function learned so far.
    Optionally, it also adapts an associated BatchNorm1d layer accordingly.

    Arguments
    ---------
    layer : nn.Module
        The Linear layer to be widened.
    next_layer : nn.Module
        The next Linear layer whose input size must be adapted.
    new_width : int, optional
        The desired output size of the widened layer. Must be greater than the
        current width.
    norm_layer : sb.nnet.normalization.BatchNorm1d, optional
        Associated normalization layer to widen and copy if provided.
    noise_std : float, optional
        Standard deviation of noise added to duplicated weights for symmetry
        breaking. Default is 0.01.
    last_block : bool, optional
        Whether the current layer is the final layer (avoids adding
        noise to output).

    Returns
    -------
    new_layer : nn.Linear
        The widened Linear layer.
    new_next_layer : nn.Linear
        The adjusted next Linear layer.
    new_norm_layer : sb.nnet.normalization.BatchNorm1d (optional)
        The widened batch norm layer, only if norm_layer is provided.

    Example
    -------
    >>> layer = nn.Linear(10, 20)
    >>> next_layer = nn.Linear(20, 30)
    >>> wider, next_wider = net2wider_linear(layer, next_layer, new_width=25)

    @author Sam Collin
    """

    # Add safety checks
    # Add one neuron if no target size
    if new_width is None:
        new_width = layer.out_features + 1
    # Make sure the goal is to grow
    elif new_width <= layer.out_features:
        raise ValueError(
            "The new layer size must be greater than the current layer size"
        )

    # Number of neurons to add
    n_add_weights = new_width - layer.out_features

    # Define our 2 new layers
    new_layer = nn.Linear(layer.in_features, new_width)
    new_next_layer = nn.Linear(new_width, next_layer.out_features)

    # First, we will copy the already trained weights
    # Current Layer
    new_layer.weight.data[: layer.out_features] = layer.weight.data
    new_layer.bias.data[: layer.out_features] = layer.bias.data
    # Next Layer
    new_next_layer.weight.data[:, : layer.out_features] = next_layer.weight.data

    # BatchNorm handling (Copy the layer but with new size)
    if norm_layer is not None:

        # Create new sb batchnorm with the updated size
        new_norm_layer = sb.nnet.normalization.BatchNorm1d(input_size=new_width)

        # Copy the learned scale parameters (gamma) for the existing units
        new_norm_layer.norm.weight.data[: layer.out_features] = (
            norm_layer.norm.weight.data
        )

        # Copy the learned shift parameters (beta) for the existing units
        new_norm_layer.norm.bias.data[: layer.out_features] = norm_layer.norm.bias.data

        # Copy the running mean statistics for the existing units
        new_norm_layer.norm.running_mean.data[: layer.out_features] = (
            norm_layer.norm.running_mean.data
        )

        # Copy the running var statistics for the existing units
        new_norm_layer.norm.running_var.data[: layer.out_features] = (
            norm_layer.norm.running_var.data
        )

        # Copy the number of batches seen so far
        new_norm_layer.norm.num_batches_tracked.data = (
            norm_layer.norm.num_batches_tracked.data
        )

    for i in range(n_add_weights):

        # Choose a random idx that will be the neuron to replicate
        idx_split = np.random.randint(0, layer.out_features)

        # Clone the node that will be duplicated
        weights_splitted_node = layer.weight.data[idx_split].clone()
        bias_splitted_node = layer.bias.data[idx_split].clone()

        # Add noise to the replicated weight and bias
        if noise_std:
            weights_splitted_node += torch.randn_like(weights_splitted_node) * noise_std
            bias_splitted_node += (
                torch.randn(1).to(layer.weight.device).item() * noise_std
            )

        # Insert the new node
        new_layer.weight.data[layer.out_features + i] = weights_splitted_node
        new_layer.bias.data[layer.out_features + i] = bias_splitted_node

        # Duplicate also the norm params
        if norm_layer is not None:

            # Copy the scale parameter (gamma) from the duplicated unit to the new unit
            new_norm_layer.norm.weight.data[layer.out_features + i] = (
                new_norm_layer.norm.weight.data[idx_split : idx_split + 1]
            )
            # Copy the shift parameter (beta) from the duplicated unit to the new unit
            new_norm_layer.norm.bias.data[layer.out_features + i] = (
                new_norm_layer.norm.bias.data[idx_split : idx_split + 1]
            )
            # Copy the running mean of the duplicated unit
            new_norm_layer.norm.running_mean.data[layer.out_features + i] = (
                new_norm_layer.norm.running_mean.data[idx_split : idx_split + 1]
            )
            # Copy the running variance of the duplicated unit
            new_norm_layer.norm.running_var.data[layer.out_features + i] = (
                new_norm_layer.norm.running_var.data[idx_split : idx_split + 1]
            )

        # To handle the next layer and perform function preservig, we divide
        # the outgoing weight associated with the splitted node and copy this divided
        # one.
        new_next_layer.weight.data[:, idx_split] /= 2
        new_next_layer.weight.data[:, layer.out_features + i] = (
            new_next_layer.weight.data[:, idx_split]
        )

        # Add output noise (same idea) but make sure it is not applied to last
        # layer to preserve the logits
        if noise_std and not last_block:
            new_next_layer.weight.data[:, layer.out_features + i] += (
                torch.randn_like(new_next_layer.weight.data[:, idx_split]) * noise_std
            )

    if norm_layer is not None:
        return new_layer, new_next_layer, new_norm_layer

    return new_layer, new_next_layer


def net2deeper_linear(layer: nn.Module):
    """Insert a new Linear layer initialized to preserve the identity function.

    This function implements the Net2DeeperNet operation by duplicating a Linear layer
    and inserting a new one initialized as an identity mapping, so that the transformation
    applied by the network remains unchanged.

    Arguments
    ---------
    layer : nn.Module
        The Linear layer after which an identity-initialized Linear layer will be inserted.

    Returns
    -------
    nn.Sequential
        A sequential module containing the original layer followed by the new identity layer.

    Example
    -------
    >>> layer = nn.Linear(128, 128)
    >>> deeper = net2deeper_linear(layer)
    >>> print(deeper)
    Sequential(
        (0): Linear(in_features=128, out_features=128, bias=True)
        (1): Linear(in_features=128, out_features=128, bias=True)
    )

    @author Sam Collin
    """
    old_bias = layer.bias.data

    # Create a new layer with the same number of I/O units
    duplicated_layer = nn.Linear(layer.out_features, layer.out_features)
    # Initialize the weight matrix to identity
    duplicated_layer.weight.data = torch.eye(old_bias.size(0), dtype=old_bias.dtype)
    # Bias set to 0
    duplicated_layer.bias.data = torch.zeros_like(old_bias)

    return nn.Sequential(layer, duplicated_layer)
