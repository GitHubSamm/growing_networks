import torch
import matplotlib.pyplot as plt
import json
import os
import datetime


def compute_n_correct(output, target):
    """
    Compute the number of correct predictions in a batch.

    Args:
        output (torch.Tensor): Model outputs (logits) of shape (batch_size, num_classes).
        target (torch.Tensor): True labels of shape (batch_size,).

    Returns:
        int: Number of correct predictions in the batch.

    Example:
        >>> output = model(data)
        >>> correct = compute_n_correct(output, target)
    """

    predictions = output.argmax(dim=1)
    n_correct_predictions = torch.sum(predictions == target).item()
    return n_correct_predictions


def save_training_curves(losses, accuracies, save_dir):
    """
    Save training loss and accuracy curves as PNG images.

    Args:
        losses (list of float): List of training loss values for each epoch.
        accuracies (list of float): List of training accuracy values for each epoch.
        save_dir (str): Directory where the figures will be saved.

    Returns:
        None

    Example:
        >>> save_training_curves(train_losses, train_accuracies, "./results/run1/")
    """
    plt.figure()
    plt.plot(losses, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(save_dir, "train_loss.png"))
    plt.close()

    plt.figure()
    plt.plot(accuracies, label="Training Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.savefig(os.path.join(save_dir, "train_acc.png"))
    plt.close()


def save_metrics(metrics, save_dir):
    """
    Save final metrics (e.g., loss, accuracy) into a JSON file.

    Args:
        metrics (dict): Dictionary containing metrics to save (e.g., {"final_loss": 0.1, "final_acc": 97.5}).
        save_dir (str): Directory where the metrics.json file will be saved.

    Returns:
        None

    Example:
        >>> save_metrics({"final_loss": 0.1, "final_acc": 97.5}, "./results/run1/")
    """

    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)


def create_result_dir(task_name="experiment"):
    """
    Create a timestamped directory under 'results/' to store experiment outputs.

    Args:
        task_name (str, optional): Name of the task or experiment. Default is "experiment".

    Returns:
        str: Path to the created results directory.

    Example:
        >>> results_dir = create_result_dir(task_name="basic_MNIST")
        >>> print(results_dir)  # Outputs something like './results/2025-04-06_14-30-00_basic_MNIST'
    """
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    results_base_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "results",
    )

    # Create results/ if it doesn't exist
    os.makedirs(results_base_dir, exist_ok=True)

    # Create subdirectory for this specific run
    run_dir = os.path.join(results_base_dir, f"{timestamp}_{task_name}")
    os.makedirs(run_dir, exist_ok=True)

    print(f"Results will be saved in: {run_dir}")

    return run_dir
