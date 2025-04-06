import torch
import os


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
