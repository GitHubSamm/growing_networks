"""
train.py

Script to train and evaluate a basic Multi-Layer Perceptron (MLP) on the MNIST dataset.

This script handles:
- Loading and preprocessing MNIST data
- Training a simple MLP model
- Evaluating the model on the test set
- Saving training curves, metrics, and model weights
- Logging training outputs

Usage:
    make run-basic-MNIST
    To be run from the root folder ("/growing-networks-project/")
"""

import os
import sys
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
import torchvision.transforms as tf
import torch.nn as nn

import scripts.net2net.logger as logger
import scripts.utils as utils
import scripts.net2net.utils_net2net as utils_net2net
from scripts.net2net.basic_MNIST.model import BasicMLP


# == Hparams == #
BATCH_SIZE = 64
LEARNING_RATE = 0.01
N_EPOCHS = 10


# == Pre-Processing == #
preprocess = tf.Compose([tf.ToTensor(), tf.Normalize((0.1307,), (0.3081,))])


# == Functions == #
def train(train_loader, model, epochs, criterion, optimizer, device="cpu"):
    """
    Train a model on a given dataset.

    Args:
        train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        model (torch.nn.Module): The model to train.
        epochs (int): Number of training epochs.
        criterion (torch.nn.Module): Loss function to optimize.
        optimizer (torch.optim.Optimizer): Optimizer to update model weights.
        device (str, optional): Device to use ("cpu", "cuda", "mps"). Default is "cpu".

    Returns:
        tuple:
            - train_losses (list of float): List of epoch-wise training losses.
            - train_accs (list of float): List of epoch-wise training accuracies.
            - final_epoch_loss (float): Final epoch loss.
            - final_epoch_acc (float): Final epoch accuracy.

    Example:
        >>> train_losses, train_accs, final_loss, final_acc = train(
        >>>     train_loader, model, epochs=10, criterion=loss_fn, optimizer=opt, device="cpu"
        >>> )
    """

    model.train()

    train_losses = []
    train_accs = []

    for epoch in range(epochs):

        epoch_correct = 0
        epoch_loss = 0
        n_samples = 0

        for data, target in train_loader:

            n_correct_predictions = 0

            # move to GPU if available
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()

            # Forwad pass
            output = model(data)

            loss = criterion(output, target)

            loss.backward()
            optimizer.step()

            # Compute the # of correct predictions
            n_correct_predictions = utils_net2net.compute_n_correct(output, target)

            epoch_loss += loss.item() * data.size(0)
            epoch_correct += n_correct_predictions
            n_samples += data.size(0)

        epoch_loss /= n_samples
        epoch_acc = (epoch_correct / n_samples) * 100

        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)

        print(
            f"Epoch {epoch+1}/{epochs} "
            + f"| Loss : {epoch_loss:.2f}"
            + f"| Train Acc : {epoch_acc:.2f}%"
        )

    return train_losses, train_accs, epoch_loss, epoch_acc


def evaluate(test_loader, model, criterion, device="cpu"):
    """
    Evaluate a model on a given test dataset.

    Args:
        test_loader (torch.utils.data.DataLoader): DataLoader for the test dataset.
        model (torch.nn.Module): The model to evaluate.
        criterion (torch.nn.Module): Loss function for evaluation.
        device (str, optional): Device to use ("cpu", "cuda", "mps"). Default is "cpu".

    Returns:
        tuple:
            - test_loss (float): Average loss over the test set.
            - test_acc (float): Accuracy (%) over the test set.

    Example:
        >>> test_loss, test_acc = evaluate(
        >>>     test_loader, model=model, criterion=loss_fn, device="cpu"
        >>> )
    """
    model.eval()

    test_loss = 0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for data, target in test_loader:
            # Copy data and targets to GPU
            data = data.to(device)
            target = target.to(device)

            # Do a forward pass
            output = model(data)

            # Calculate the loss
            loss = criterion(output, target)
            test_loss += loss.item() * data.size(0)

            # Count number of correct digits
            total_correct += utils_net2net.compute_n_correct(output, target)
            total_samples += data.size(0)

    test_loss = test_loss / total_samples
    test_acc = (total_correct / total_samples) * 100

    print(f"Evaluation -> Loss: {test_loss:.3f}, Accuracy: {test_acc:.2f}%")

    return test_loss, test_acc


if __name__ == "__main__":

    device = utils.get_device()
    project_folder = utils.get_project_root()
    data_folder = utils.get_data_folder()
    results_dir = utils_net2net.create_result_dir(task_name="basic_MNIST")
    sys.stdout = logger.Logger(os.path.join(results_dir, "train_log.txt"))

    # Import MNIST data
    print(f"Data will be saved (if not already) in:\n{data_folder}")
    train_dataset = datasets.MNIST(
        root=data_folder, train=True, download=True, transform=preprocess
    )

    test_dataset = datasets.MNIST(
        root=data_folder, train=False, download=True, transform=preprocess
    )

    train_loader = DataLoader(
        dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )

    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Toy model
    model = BasicMLP().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("Start training ...")
    train_losses, train_accs, final_loss, final_acc = train(
        train_loader,
        model=model,
        epochs=N_EPOCHS,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
    )

    print("Start evaluation ...")
    test_loss, test_acc = evaluate(
        test_loader, model=model, criterion=criterion, device=device
    )

    # Save final model
    torch.save(model.state_dict(), os.path.join(results_dir, "model_final.pt"))

    # Save metrics
    utils_net2net.save_training_curves(train_losses, train_accs, results_dir)
    utils_net2net.save_metrics(
        {
            "train_final_loss": final_loss,
            "train_final_acc": final_acc,
            "test_final_loss": test_loss,
            "test_final_acc": test_acc,
        },
        results_dir,
    )
