import torch.nn as nn
import torch.nn.functional as F


class BasicMLP(nn.Module):
    """
    A basic Multi-Layer Perceptron (MLP) model for classification tasks.

    This model processes flattened inputs through a series of linear layers
    with non-linear activation functions to produce class scores.

    Args:
        None

    Attributes:
        lin1 (nn.Linear): First linear transformation.
        lin2 (nn.Linear): Second linear transformation.
        lin3 (nn.Linear): Final linear transformation projecting to class scores.

    Example:
        >>> model = BasicMLP()
        >>> x = torch.randn(32, 1, 28, 28)  # Batch of 32 MNIST images
        >>> output = model(x)
        >>> print(output.shape)  # torch.Size([32, 10])
    """

    def __init__(self):
        """
        Initialize the BasicMLP layers.
        """

        super(BasicMLP, self).__init__()

        self.lin1 = nn.Linear(28 * 28, 512)
        self.lin2 = nn.Linear(512, 128)
        self.lin3 = nn.Linear(128, 10)

    def forward(self, x):
        """
        Perform a forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 28, 28).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes).
        """

        x = x.view(x.size(0), -1)
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        out = self.lin3(x)
        return out
