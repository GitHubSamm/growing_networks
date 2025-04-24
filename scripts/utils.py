import torch
import os
import numpy as np
import random


def get_device():
    """
    Detect the available hardware device (CUDA, MPS, or CPU) and return it.

    Returns:
        torch.device: The available device to be used for computation.
            - 'cuda' if an NVIDIA GPU with CUDA is available.
            - 'mps' if an Apple Silicon GPU (M1/M2/M3/M4) is available.
            - 'cpu' otherwise.
    """

    if torch.backends.mps.is_available():
        print("Device used is MPS - Apple")
        return torch.device("mps")

    elif torch.cuda.is_available():
        print("Device used is CUDA - Nvidia")
        return torch.device("cuda")

    else:
        print("Device used is CPU")
        return torch.device("cpu")


def get_project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_data_folder():
    return os.path.join(get_project_root(), "data", "raw")


def seed_everything(
    seed: int = 42, verbose: bool = True, deterministic: bool = True
) -> int:
    """
    Strongly inspired by the homonym function in speechbrain
    Set the seed for random number generators in Python, NumPy, and PyTorch.

    Args:
        seed (int): The random seed to use.
        verbose (bool): Whether to print the seed being set.
        deterministic (bool): Whether to enforce deterministic
        behavior in PyTorch.

    Returns:
        int: The seed that was set.

    Example:
        >>> seed_everything(42)

    @author Sam Collin
    """
    if verbose:
        print(f"Setting global seed to {seed}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        # Ensure it works on colab
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)

    return seed


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
