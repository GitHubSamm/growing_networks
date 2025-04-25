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
    A minimal MLP model used as the 'young' baseline in strategy 2.

    This model consists of a single hidden layer with a ReLU activation,
    followed by a final linear projection to class scores. It is designed
    to be gradually deepened and widened during training.

    Arguments
    ---------
    None

    Attributes
    ----------
    lin1 : nn.Linear
        First linear transformation from flattened input to hidden space.
    lin2 : nn.Linear
        Final projection to class scores.

    Example
    -------
    >>> model = SuperBasicMLP_strat2()
    >>> x = torch.randn(32, 1, 28, 28)
    >>> output = model(x)
    >>> output.shape
    torch.Size([32, 10])
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
    """
    The 'adult' version of the MLP used in strategy 2, with increased depth and width.

    This model adds an intermediate hidden layer and doubles the capacity of
    the original network. It is used as a reference to evaluate the effectiveness
    of Net2DeeperNet and Net2WiderNet growth applied to the smaller version.

    Arguments
    ---------
    None

    Attributes
    ----------
    lin1 : nn.Linear
        First linear layer from input to hidden space.
    lin2 : nn.Linear
        Second hidden layer expanding the representation.
    lin3 : nn.Linear
        Final projection to class scores.

    Example
    -------
    >>> model = SuperBasicMLP_big_strat2()
    >>> x = torch.randn(32, 1, 28, 28)
    >>> output = model(x)
    >>> output.shape
    torch.Size([32, 10])
    """

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
    """
    A minimal MLP model with BatchNorm1d for MNIST-like classification tasks.

    This model flattens 28×28 input images, applies a linear transform,
    normalizes via BatchNorm1d, passes through a ReLU activation, and
    finally projects to 10 class scores.

    Architecture:
        - Linear(784 → 8)
        - BatchNorm1d(8)
        - ReLU
        - Linear(8 → 10)

    Example
    -------
    >>> model = SuperBasicMLP_BN1D()
    >>> x = torch.randn(4, 1, 28, 28)
    >>> out = model(x)
    >>> out.shape
    torch.Size([4, 10])
    """

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
    """
    A larger MLP model with BatchNorm1d for enhanced capacity on MNIST-like data.

    This model flattens 28×28 input images, applies a wider linear transform,
    normalizes via BatchNorm1d, uses a ReLU activation, then projects to
    10 class scores.

    Architecture:
        - Linear(784 → 32)
        - BatchNorm1d(32)
        - ReLU
        - Linear(32 → 10)

    Example
    -------
    >>> model = SuperBasicMLP_big_BN1D()
    >>> x = torch.randn(4, 1, 28, 28)
    >>> out = model(x)
    >>> out.shape
    torch.Size([4, 10])
    """

    def __init__(self):

        super(SuperBasicMLP_big_BN1D, self).__init__()

        self.lin1 = nn.Linear(28 * 28, 32)
        self.lin2 = nn.Linear(32, 10)
        self.norm = sb.nnet.normalization.BatchNorm1d(input_size=32)

    def forward(self, x):

        x = x.view(x.size(0), -1)
        x = self.lin1(x)
        x = self.norm(x)
        x = F.relu(x)
        out = self.lin2(x)
        return out
