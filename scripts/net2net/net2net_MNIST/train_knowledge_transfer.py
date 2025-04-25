"""
Net2Net Growth & Knowledge-Transfer Recipe for MNIST / Fashion-MNIST
===================================================================

Author   : Sam Collin  
Framework: PyTorch + wandb (optional)  
Inspired : SpeechBrain Recipe Structure

Description
-----------
This recipe trains Multi-Layer Perceptrons (MLPs) under two regimes and
makes it easy to compare them:

1. **Scratch baseline**  
   Train the *adult* (large) network for *N* epochs from random
    initialisation.

2. **Net2Net knowledge-transfer path**  
   a. Train a *young* (small) network for *N₁* « pre-train » epochs.  
   b. Expand it with a Net2Net transformation *(wider / deeper / BN)*.  
   c. Continue training the *student* for *N₂* epochs.

Three growth strategies are implemented:

* **Strategy 1 (Wider)**     Net2Wider on `lin1` → `lin2`  
* **Strategy 2 (Deeper + Wider)**  
    – Net2Deeper identity insertion, then Net2Wider  
* **Strategy 3 (Wider + BatchNorm)**  
    – Net2Wider with BatchNorm parameter copying

Usage
----------
# Scratch baseline (adult network), wandb disabled
python scripts/net2net/net2net_MNIST/train_knowledge_transfer.py \
    --model adult           \
    --strat 1               \
    --no_pretrain_for_baseline \
    --dataset MNIST
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
import wandb

import scripts.net2net.net2net_MNIST.net2net_ops as net2net_ops
import scripts.net2net.logger as logger
import scripts.utils as utils
import scripts.net2net.utils_net2net as utils_net2net
import scripts.net2net.net2net_MNIST.model as models

# Argument Parser inspired by the one in speechbrain
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="young", choices=["young", "adult"])
parser.add_argument("--strat", type=int, default=1, choices=[1, 2, 3])
parser.add_argument("--no_pretrain_for_baseline", action="store_true")
parser.add_argument("--dataset", default="MNIST", choices=["MNIST", "FASHION_MNIST"])
parser.add_argument("--wandb", default="disabled", choices=["disabled", "online"])
parser.add_argument("--run_name", type=str, default="xyz")
command_line_args = parser.parse_args()

# == Hparams == #
NO_PRETRAIN_FOR_BASELINE = command_line_args.no_pretrain_for_baseline
STRAT_NUMBER = command_line_args.strat
BATCH_SIZE = 64
LEARNING_RATE = 0.01
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005

if NO_PRETRAIN_FOR_BASELINE:
    N_EPOCHS_BEFORE_GROW = [24]  # [8, 24]
    N_GROWTH = 0  # 1
    PRETRAIN_OFFSET = 0
else:
    N_EPOCHS_BEFORE_GROW = [8, 24]
    N_GROWTH = 1
    PRETRAIN_OFFSET = N_EPOCHS_BEFORE_GROW[0]

# If required, instantiate a new wandb run
wandb.init(
    mode=command_line_args.wandb,
    project="toy-Net2Net",
    name=command_line_args.run_name,
    config=dict(
        scratch_baseline=NO_PRETRAIN_FOR_BASELINE,
        model=command_line_args.model,
        strategy=STRAT_NUMBER,
        dataset=command_line_args.dataset,
        batch_size=BATCH_SIZE,
        lr=LEARNING_RATE,
        pretrain_offset=PRETRAIN_OFFSET,
        n_growth=N_GROWTH,
    ),
)

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
        new_width = int(layer.out_features * 4)
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

            # For the second one, make the newly created layer 2 times wider
            layer = model.lin1[1]
            next_layer = model.lin2
            new_width = int(layer.out_features * 2)
            model.lin1[1], model.lin2 = net2net_ops.net2wider_linear(
                layer, next_layer, new_width, noise_std=0.01, last_block=True
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


def train(
    train_loader,
    test_loader,
    model,
    epochs,
    criterion,
    optimizer,
    growth_number,
    device="cpu",
):
    """
    Train a model on a given dataset. It records the metrics in wandb if necessary, except for pretraining.

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

        # If we are currently pretraining the model before growing, we ignore wandb logging
        if not NO_PRETRAIN_FOR_BASELINE and growth_number == 0:
            pass
        else:
            test_loss, test_acc = evaluate(
                test_loader,
                model=model,
                criterion=criterion,
                device=device,
                train=True,
            )
            wandb.log(
                {
                    "train_loss": epoch_loss,
                    "train_acc": epoch_acc,
                    "epoch": epoch,
                    "test_loss": test_loss,
                    "test_acc": test_acc,
                }
            )

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
    Main of the Net2Net / scratch experiments
    -------------------------------------------------------

    Workflow
    --------
    1.   **Reproducibility & logging**
         * `seed_everything` sets all RNG seeds.
         * A results folder is created; `logger.Logger` mirrors stdout to
           *train_log.txt*.
         * `wandb` is initialised *online* or *disabled* according to the
        `--wandb` CLI flag.

    2.   **Data preparation**
        The requested dataset (MNIST / Fashion-MNIST) is downloaded into
         *data/raw* and wrapped in `DataLoader`s with basic normalisation.

    3.   **Model definition**
         *young* (small) or *adult* (large) MLP is instantiated according
         to `--model` and growth *strategy* (`--strat`).

    4.   **Optimiser & criterion**
        SGD (lr = 0.01, momentum = 0.9, weight-decay = 5 × 10⁻⁴) and
        Cross-Entropy loss.

    5.   **Training loop over growth stages**
        For `growth_number = 0 … N_GROWTH`
        ────────────────────────────────────────
         * Optionally apply **Net2Net** (`growth_strategy`) unless the run
           is the *scratch baseline* (`--no_pretrain_for_baseline`).
         * Re-initialise the optimiser (LR is kept identical here)
         * **Pre-training evaluation** on the *training* set validates
            function-preservation.
         * Train for `N_EPOCHS_BEFORE_GROW[growth_number]` epochs:
                – Pre-train block (if any) is *not* logged to wandb
                to keep curves aligned.
                – Subsequent stages log per-epoch loss / accuracy.
         * Evaluate on the *test* set and log the final metrics.
         * Persist model checkpoint, JSON metrics and PNG learning-curves.

    6.   **Global summary**
            After the last growth stage, aggregate curves are saved and the
            total wall-clock training time is printed.

    Returns
    -------
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

    # Diplay run info
    print(
        "\n##########"
        + "\nStarting net2net_MNIST ..."
        + f"\nModel used: {command_line_args.model} "
        + f"| Dataset used: {command_line_args.dataset} "
        + f"| Growth enabled: {not NO_PRETRAIN_FOR_BASELINE} "
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
        model.parameters(),
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
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
        if not NO_PRETRAIN_FOR_BASELINE:
            model = growth_strategy(model, growth_number, STRAT_NUMBER)
            model = model.to(device)
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=LEARNING_RATE,
                momentum=MOMENTUM,
                weight_decay=WEIGHT_DECAY,
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
            + f"Epoch 0/{N_EPOCHS_BEFORE_GROW[growth_number]} "
            + f"| Loss : {train_loss:.2f}"
            + f"| Train Acc : {train_acc:.2f}%\n"
            + "#####################################"
        )
        # Train the new model
        begin = time.time()
        train_losses, train_accs, final_loss, final_acc = train(
            train_loader,
            test_loader,
            model=model,
            epochs=N_EPOCHS_BEFORE_GROW[growth_number],
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            growth_number=growth_number,
        )
        end = time.time()
        step_train_duration = end - begin
        global_train_duration += step_train_duration
        print(f"Training finished in {step_train_duration:.2f}s")

        print("\nStart evaluation ...")
        test_loss, test_acc = evaluate(
            test_loader, model=model, criterion=criterion, device=device
        )

        # If we are pretraining, we don't monitor the pretrain part
        if not NO_PRETRAIN_FOR_BASELINE and growth_number == 0:
            pass
        else:
            wandb.log({"final_test_loss": test_loss, "final_test_accuracy": test_acc})

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
