import torch.nn as nn
import torch.nn.functional as F
import speechbrain as sb


class SuperBasicMLP_strat1(nn.Module):
    """
    A minimal Multi-Layer Perceptron (MLP) for MNIST-like classification tasks.

    This model flattens 28x28 input images and applies two linear layers with
    ReLU activation in between. It is designed as a simple baseline architecture
    for experiments involving model growth.

    Architecture:
        - Linear(784 → 16) + ReLU
        - Linear(16 → 10)

    Example:
        >>> model = SuperBasicMLP_strat1()
        >>> x = torch.randn(32, 1, 28, 28)
        >>> output = model(x)
        >>> output.shape
        torch.Size([32, 10])
    """

    def __init__(self):
        """
        Initializes the MLP with two fully connected layers.
        """

        super(SuperBasicMLP_strat1, self).__init__()

        self.lin1 = nn.Linear(28 * 28, 16)
        self.lin2 = nn.Linear(16, 10)

    def forward(self, x):
        """
        Forward pass through the MLP.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 28, 28).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 10).
        """

        x = x.view(x.size(0), -1)
        x = F.relu(self.lin1(x))
        out = self.lin2(x)
        return out


class SuperBasicMLP_big_strat1(nn.Module):
    """
    A larger variant of the SuperBasicMLP model for MNIST classification.

    This model increases the capacity of the hidden layer to 128 units,
    allowing more complex representations compared to the smaller version.

    Architecture:
        - Linear(784 → 64) + ReLU
        - Linear(64 → 10)

    Example:
        >>> model = SuperBasicMLP_big_strat1()
        >>> x = torch.randn(32, 1, 28, 28)
        >>> output = model(x)
        >>> output.shape
        torch.Size([32, 10])
    """

    def __init__(self):
        """
        Initializes the larger MLP with a wider hidden layer.
        """
        super(SuperBasicMLP_big_strat1, self).__init__()

        self.lin1 = nn.Linear(28 * 28, 64)
        self.lin2 = nn.Linear(64, 10)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 28, 28).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 10).
        """

        x = x.view(x.size(0), -1)
        x = F.relu(self.lin1(x))
        out = self.lin2(x)
        return out


class SuperBasicMLP_strat2(nn.Module):
    """
    A super basic Multi-Layer Perceptron (MLP) model for classification tasks.

    This model processes flattened inputs through a series of linear layers
    with non-linear activation functions to produce class scores.

    Args:
        None

    Attributes:
        lin1 (nn.Linear): First linear transformation.
        lin2 (nn.Linear): Final linear transformation projecting to class
        scores.

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

        super(SuperBasicMLP_strat2, self).__init__()

        self.lin1 = nn.Linear(28 * 28, 32)
        self.lin2 = nn.Linear(32, 10)

    def forward(self, x):
        """
        Perform a forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 28, 28).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes).
        """

        x = x.view(x.size(0), -1)
        x = self.lin1(x)
        x = F.relu(x)
        out = self.lin2(x)
        return out


class SuperBasicMLP_big_strat2(nn.Module):

    def __init__(self):

        super(SuperBasicMLP_big_strat2, self).__init__()

        self.lin1 = nn.Linear(28 * 28, 32)
        self.lin2 = nn.Linear(32, 64)
        self.lin3 = nn.Linear(64, 10)

    def forward(self, x):
        """
        Perform a forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 28, 28).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes).
        """

        x = x.view(x.size(0), -1)
        x = self.lin1(x)
        x = self.lin2(x)
        x = F.relu(x)
        out = self.lin3(x)
        return out


class SuperBasicMLP_BN1D(nn.Module):

    def __init__(self):

        super(SuperBasicMLP_BN1D, self).__init__()

        self.lin1 = nn.Linear(28 * 28, 8)
        self.lin2 = nn.Linear(8, 10)
        self.norm = sb.nnet.normalization.BatchNorm1d(input_size=8)

    def forward(self, x):

        x = x.view(x.size(0), -1)
        x = self.lin1(x)
        x = self.norm(x)
        x = F.relu(x)
        out = self.lin2(x)
        return out


class SuperBasicMLP_big_BN1D(nn.Module):

    def __init__(self):

        super(SuperBasicMLP_big_BN1D, self).__init__()

        self.lin1 = nn.Linear(28 * 28, 128)
        self.lin2 = nn.Linear(128, 10)
        self.norm = sb.nnet.normalization.BatchNorm1d(input_size=128)

    def forward(self, x):

        x = x.view(x.size(0), -1)
        x = self.lin1(x)
        x = self.norm(x)
        x = F.relu(x)
        out = self.lin2(x)
        return out
