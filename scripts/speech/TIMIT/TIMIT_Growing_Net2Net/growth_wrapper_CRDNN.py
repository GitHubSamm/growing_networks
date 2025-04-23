import torch.nn as nn
from scripts.net2net.net2net_MNIST.net2net_ops import net2wider_linear


class GrowingCRDNN(nn.Module):
    def __init__(self, original_model, model_output):
        super().__init__()
        self.model = original_model
        self.output = model_output

    def forward(self, x):
        return self.model(x)

    def grow(self, width_factor, noise_std):

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
            )
            # Assign the new layers
            current_block.linear.w = new_linear.to(device)
            current_block.norm = new_norm.to(device)
            if last_block:
                self.output.w = new_next_layer.to(device)
            else:
                next_block.linear.w = new_next_layer.to(device)
