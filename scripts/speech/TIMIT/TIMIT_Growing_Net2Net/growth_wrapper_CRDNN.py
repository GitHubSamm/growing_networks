"""
GrowingCRDNN wrapper
====================

This module defines a GrowingCRDNN class that wraps a pretrained CRDNN ASR model
and provides a method to apply Net2Net “wider” growth to its DNN blocks in a
function‐preserving way.

Usage:
    >>> from growing_crdnn import GrowingCRDNN
    >>> model = GrowingCRDNN(original_model, model_output)
    >>> # grow each hidden layer by 1.5× with small noise
    >>> model.grow(width_factor=1.5, noise_std=0.01)

Dependencies:
    - torch
    - scripts.net2net.net2net_MNIST.net2net_ops.net2wider_linear

Author: Sam Collin
"""

import torch.nn as nn
from scripts.net2net.net2net_MNIST.net2net_ops import net2wider_linear


class GrowingCRDNN(nn.Module):
    """
    Wrapper module to dynamically grow a CRDNN-based ASR model.

    This class takes an existing CRDNN (`original_model`) and its final
    output linear layer (`model_output`), and exposes a `grow` method
    which applies Net2WiderNet to each DNN block in turn.

    Attributes
    ----------
    model : nn.Module
        The core CRDNN model (without the final linear projection).
    output : nn.Module
        The final linear projection layer to adjust after the last DNN block.
    """

    def __init__(self, original_model, model_output):
        """
        Initialize the GrowingCRDNN wrapper.

        Parameters
        ----------
        original_model : nn.Module
            Pretrained CRDNN backbone (all convolutional/RNN layers plus DNN blocks).
        model_output : nn.Module
            The final linear layer (often named `.w`) whose input size must match
            the last DNN block’s output.
        """
        super().__init__()
        self.model = original_model
        self.output = model_output

    def forward(self, x):
        """
        Perform a forward pass through the underlying CRDNN.

        Parameters
        ----------
        x : torch.Tensor
            Input feature tensor (e.g., log-Mel spectrogram) of shape
            (batch_size, time_steps, n_mels).

        Returns
        -------
        torch.Tensor
            The raw output logits from the CRDNN (before softmax or CTC log-softmax).
        """
        return self.model(x)

    def grow(self, width_factor, noise_std):
        """
        Apply Net2WiderNet growth to each DNN block, in place.

        Iterates over the DNN blocks stored in `self.model.DNN`, widens each
        block’s linear layer by `width_factor`, transfers its BatchNorm1d
        parameters if present, and adjusts the subsequent layer’s weights
        to exactly preserve the function (up to optional noise).

        Parameters
        ----------
        width_factor : float
            Factor by which to multiply each hidden layer’s width.
        noise_std : float
            Standard deviation of Gaussian noise to add to duplicated neurons
            to break symmetry.

        Notes
        -----
        - The final DNN block is linked to `self.output`; all others are
            linked block→block.
        - Growth is done in‐place: after calling `.grow()`, both the DNN blocks
            and the final linear layer have been replaced by their widened versions.
        """
        n_blocks = len(self.model.DNN)
        last_block = False
        device = next(self.parameters()).device
        for idx in range(n_blocks):

            if idx + 1 == n_blocks:
                current_block, out_linear = (
                    self.model.DNN[f"block_{idx}"],
                    self.output.w,
                )
                last_block = True
            else:
                current_block, next_block = (
                    self.model.DNN[f"block_{idx}"],
                    self.model.DNN[f"block_{idx+1}"],
                )
            # Grow the layer
            new_width = int(current_block.linear.w.out_features * width_factor)
            new_linear, new_next_layer, new_norm = net2wider_linear(
                layer=current_block.linear.w,
                next_layer=next_block.linear.w if not last_block else out_linear,
                norm_layer=current_block.norm,
                new_width=new_width,
                noise_std=noise_std,
                last_block=last_block,
            )
            # Assign the new layers
            current_block.linear.w = new_linear.to(device)
            current_block.norm = new_norm.to(device)
            if last_block:
                self.output.w = new_next_layer.to(device)
            else:
                next_block.linear.w = new_next_layer.to(device)
