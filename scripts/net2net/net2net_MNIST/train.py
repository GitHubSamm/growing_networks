"""
Net2Net Growth Recipe for MNIST and FashionMNIST
=================================================

Author: Sam Collin
Framework: PyTorch
Inspired by: SpeechBrain Recipe Structure

Description:
------------
This script implements a pipeline for training Multi-Layer Perceptrons (MLPs)
on MNIST or FashionMNIST using growth strategies inspired by the Net2Net framework.
Three different strategies are supported:

1. **Strategy 1 (Wider)**:
    - Simple Net2Wider growth of the first linear layer (`lin1`) and its output layer (`lin2`).

2. **Strategy 2 (Deeper + Wider)**:
    - Step 1: Deepen `lin1` using Net2Deeper by inserting an identity layer.
    - Step 2: Apply Net2Wider on the added middle layer and `lin2`.

3. **Strategy 3 (Wider + BatchNorm)**:
    - Similar to strategy 1, but includes propagation of BatchNorm parameters.

Key Features:
-------------
- Growth can be applied multiple times (via `N_GROWTH`).
- All experiments are reproducible thanks to fixed random seeds.
- Logs and metrics are saved per growth step.
- Function-preserving transformations are validated via pre-training evaluation.

Usage:
------
You can run the script from the command line:

```bash
python scripts/net2net/net2net_MNIST/train.py \
    --model young \
    --strat 1 \
    --dataset MNIST
```
"""

import os
import sys
import torch
import argparse
from torchvision import datasets
from torch.utils.data import DataLoader
import torchvision.transforms as tf
import torch.nn as nn
import time

import scripts.net2net.net2net_MNIST.net2net_ops as net2net_ops
import scripts.net2net.logger as logger
import scripts.utils as utils
import scripts.net2net.utils_net2net as utils_net2net
import scripts.net2net.net2net_MNIST.model as models

# Argument Parser inspired by the one in speechbrain
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="young", choices=["young", "adult"])
parser.add_argument("--strat", type=int, default=1, choices=[1, 2, 3])
parser.add_argument("--no_growth_for_baseline", action="store_true")
parser.add_argument("--dataset", default="MNIST", choices=["MNIST", "FASHION_MNIST"])
command_line_args = parser.parse_args()

# == Hparams == #
STRAT_NUMBER = command_line_args.strat
BATCH_SIZE = 64
LEARNING_RATE = 0.01
N_EPOCHS_BEFORE_GROW = 8
N_GROWTH = 2

# == Pre-Processing == #
preprocess_MNIST = tf.Compose([tf.ToTensor(), tf.Normalize((0.1307,), (0.3081,))])
preprocess_FASHION = tf.Compose([tf.ToTensor(), tf.Normalize((0.2860,), (0.3530,))])

# == Models == #
model_dict = {
    "model_young_strat1": models.SuperBasicMLP_strat1,
    "model_adult_strat1": models.SuperBasicMLP_big_strat1,
    "model_young_strat2": models.SuperBasicMLP_strat2,
    "model_adult_strat2": models.SuperBasicMLP_big_strat2,
    "model_young_strat3": models.SuperBasicMLP_BN1D,
    "model_adult_strat3": models.SuperBasicMLP_big_BN1D,
}


# == Functions == #
def growth_strategy(model, growth_number, strat_number=STRAT_NUMBER):
    """
    Applies a growth transformation to a given model according to the specified strategy.

    This function supports multiple growth strategies for widening or deepening
    the architecture of a neural network during training (e.g., Net2Net).

    Args:
        model (nn.Module): The model to be grown.
        growth_number (int): The growth stage index. Used to decide which growth operation to apply
                            (e.g., for multi-step strategies).
        strat_number (int, optional): The strategy index to use for growing the network.
                                    Defaults to the global STRAT_NUMBER.

    Supported Strategies:
        - Strategy 1: Net2Wider applied on a two-layer MLP (lin1 → lin2).
        - Strategy 2:
            - Step 1: Net2Deeper inserted between input and hidden (lin1 becomes Sequential).
            - Step 2: Net2Wider applied on the inner hidden layer and lin2.
        - Strategy 3: Net2Wider with BatchNorm support.

    Returns:
        nn.Module: The updated model with increased capacity according to the selected strategy.

    Raises:
        ValueError: If an unknown strategy number is provided.

    Example:
        >>> model = SuperBasicMLP_strat1()
        >>> model = growth_strategy(model, growth_number=1, strat_number=1)
    """
    # In any case, the first set of epoch will be done without any growth
    if growth_number == 0:
        print("No growth here (Original step)")
        return model

    # Growth rules for strategy 1 (Wider)
    if strat_number == 1:

        # Retrieve the current layers
        print("Model growing ...")
        layer, next_layer = model.lin1, model.lin2

        # Make the hidden layer four times wider using net2wider at each growth
        new_width = int(layer.out_features * 2)
        new_layer, new_next_layer = net2net_ops.net2wider_linear(
            layer, next_layer, new_width, noise_std=0.01, last_block=True
        )
        # Assign the new layers
        model.lin1 = new_layer
        model.lin2 = new_next_layer

        return model

    # Growth rules for strategy 2 (Deeper + Wider)
    if strat_number == 2:

        # For the first growth, deepen the model by adding a layer
        if growth_number == 1:

            # Add a new layer using net2deeper
            layer = model.lin1
            duplicated_layer = net2net_ops.net2deeper_linear(layer)
            model.lin1 = duplicated_layer
            return model

        # For the second one, make the newly created layer 2 times wider
        if growth_number == 2:
            layer = model.lin1[1]
            next_layer = model.lin2
            new_width = int(layer.out_features * 2)
            model.lin1[1], model.lin2 = net2net_ops.net2wider_linear(
                layer, next_layer, new_width
            )
            return model

    # Growth rules for strategy 3 (Wider + BatchNorm)
    if strat_number == 3:

        # Retrieve the layers + the batchnorm layer
        print("Model growing ...")
        layer, next_layer, norm_layer = model.lin1, model.lin2, model.norm

        # Make the hidden layer four times wider using net2wider at each growth
        new_width = int(layer.out_features * 4)
        new_layer, new_next_layer, new_norm_layer = net2net_ops.net2wider_linear(
            layer, next_layer, new_width, norm_layer
        )

        # Assign the new layers
        model.lin1 = new_layer
        model.lin2 = new_next_layer
        model.norm = new_norm_layer

        return model

    else:
        raise ValueError(f"No strategy implemented for number {strat_number}.")


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

            # Forward pass
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


def evaluate(test_loader, model, criterion, device="cpu", train=False):
    """
    Evaluate a model on a given test dataset.

    Args:
        test_loader (torch.utils.data.DataLoader): DataLoader for the test dataset.
        model (torch.nn.Module): The model to evaluate.
        criterion (torch.nn.Module): Loss function used during evaluation.
        device (str, optional): Device to use ("cpu", "cuda", or "mps"). Default is "cpu".
        train (bool, optional): If True, suppress evaluation printing (e.g., during training). Default is False.

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

    # Just make sure that we display it when really evaluating on the test set
    if not train:
        print(f"Evaluation -> Loss: {test_loss:.3f}, Accuracy: {test_acc:.2f}%")

    return test_loss, test_acc


if __name__ == "__main__":
    """
    Main entry point for the training and evaluation pipeline.

    This function initializes data loaders, model, optimizer,
    and loss function, then trains and evaluates the model.
    It optionally applies a growth strategy at predefined points to
    increase model capacity during training.

    Typical steps:
    1. Load and preprocess the dataset.
    2. Initialize model, loss, and optimizer.
    3. Train the model (possibly applying a growth strategy mid-training).
    4. Evaluate the model on test data.
    5. Save or log the results.

    Returns:
        None
    """

    # Set the seed, the result folder and create the logger for the results
    utils.seed_everything(691)
    device = utils.get_device()
    project_folder = utils.get_project_root()
    data_folder = utils.get_data_folder()
    results_dir = utils_net2net.create_result_dir(
        task_name=f"net2net_{command_line_args.dataset}"
    )
    sys.stdout = logger.Logger(os.path.join(results_dir, "train_log.txt"))

    NO_GROWTH_BASELINE = command_line_args.no_growth_for_baseline

    # Diplay run info
    print(
        "\n##########"
        + "\nStarting net2net_MNIST ..."
        + f"\nModel used: {command_line_args.model} "
        + f"| Dataset used: {command_line_args.dataset} "
        + f"| Growth enabled: {not NO_GROWTH_BASELINE} "
        + f"| Strat number: {STRAT_NUMBER}"
        + "\n##########"
    )

    # Import data
    print(f"\nData will be saved (if not already) in:\n{data_folder}")

    # MNIST
    if command_line_args.dataset == "MNIST":
        train_dataset = datasets.MNIST(
            root=data_folder, train=True, download=True, transform=preprocess_MNIST
        )
        test_dataset = datasets.MNIST(
            root=data_folder, train=False, download=True, transform=preprocess_MNIST
        )
    # Fashion MNIST
    elif command_line_args.dataset == "FASHION_MNIST":
        train_dataset = datasets.FashionMNIST(
            root=data_folder, train=True, download=True, transform=preprocess_FASHION
        )
        test_dataset = datasets.FashionMNIST(
            root=data_folder, train=False, download=True, transform=preprocess_FASHION
        )

    train_loader = DataLoader(
        dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )

    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Choose model (young is the small version, adult is the final)
    model = model_dict[f"model_{command_line_args.model}_strat{STRAT_NUMBER}"]().to(
        device
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0005
    )

    global_losses_over_growth = []
    global_accs_over_growth = []
    global_train_duration = 0

    print("\nStart training for multiple growth ...")
    for growth_number in range(N_GROWTH + 1):

        growth_results_dir = os.path.join(results_dir, f"growth{growth_number}")
        os.makedirs(growth_results_dir, exist_ok=True)

        # Check wether a growth strat is required
        print(f"\nBegin step {growth_number}")
        if not NO_GROWTH_BASELINE:
            model = growth_strategy(model, growth_number, STRAT_NUMBER)
            model = model.to(device)
            optimizer = torch.optim.SGD(
                model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0005
            )
        else:
            print("No growth here")

        # Compare the number of parameters
        n_parameters = utils.count_parameters(model)
        print(f"The number of parameters at this step is: {n_parameters}")
        print("The model is :\n", model)
        print(f"\nBegin training for growth step {growth_number}...")

        # Evaluate model raw after growth on the train set (Check for function preserving)
        train_loss, train_acc = evaluate(
            train_loader, model=model, criterion=criterion, device=device, train=True
        )
        # Display it in the logs
        print(
            "\n###### Pre-Training Evaluation ######\n"
            + f"Epoch 0/{N_EPOCHS_BEFORE_GROW} "
            + f"| Loss : {train_loss:.2f}"
            + f"| Train Acc : {train_acc:.2f}%\n"
            + "#####################################"
        )
        # Train the new model
        begin = time.time()
        train_losses, train_accs, final_loss, final_acc = train(
            train_loader,
            model=model,
            epochs=N_EPOCHS_BEFORE_GROW,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        end = time.time()
        step_train_duration = end - begin
        global_train_duration += step_train_duration
        print(f"Training finished in {step_train_duration:.2f}s")

        print("\nStart evaluation ...")
        test_loss, test_acc = evaluate(
            test_loader, model=model, criterion=criterion, device=device
        )

        # Save final model
        torch.save(
            model.state_dict(),
            os.path.join(growth_results_dir, f"model_final_growth{growth_number}.pt"),
        )

        # Save metrics
        utils_net2net.save_training_curves(train_losses, train_accs, growth_results_dir)
        utils_net2net.save_metrics(
            {
                f"n_parameters_step{growth_number}": n_parameters,
                f"train_final_loss_step{growth_number}": final_loss,
                f"train_final_acc_step{growth_number}": final_acc,
                f"test_final_loss_step{growth_number}": test_loss,
                f"test_final_acc_step{growth_number}": test_acc,
                f"train_duration_step{growth_number}": step_train_duration,
            },
            growth_results_dir,
        )

        global_losses_over_growth.extend(train_losses)
        global_accs_over_growth.extend(train_accs)

    # Create learning curves that will be saved with the results
    utils_net2net.save_training_curves(
        global_losses_over_growth, global_accs_over_growth, results_dir
    )
    utils_net2net.save_metrics(
        {
            "m_parameters_final": n_parameters,
            "train_final_loss": final_loss,
            "train_final_acc": final_acc,
            "test_final_loss": test_loss,
            "test_final_acc": test_acc,
            "global_train_duration": global_train_duration,
        },
        results_dir,
    )

    print(
        "\nGlobal Experiment finished with a total"
        + f" training time of {global_train_duration:.2f}s"
    )
    print(f"Detailed results available in {results_dir}\n")
