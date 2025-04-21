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

    if growth_number == 0:
        print("No growth here (Original step)")
        return model

    if strat_number == 1:

        print("Model growing ...")
        layer, next_layer = model.lin1, model.lin2

        new_width = int(layer.out_features * 4)
        new_layer, new_next_layer = net2net_ops.net2wider_linear(
            layer, next_layer, new_width
        )

        model.lin1 = new_layer
        model.lin2 = new_next_layer

        return model

    if strat_number == 2:
        if growth_number == 1:
            layer = model.lin1
            duplicated_layer = net2net_ops.net2deeper_linear(layer)
            model.lin1 = duplicated_layer
            return model
        if growth_number == 2:
            layer = model.lin1[1]
            next_layer = model.lin2
            new_width = int(layer.out_features * 2)
            model.lin1[1], model.lin2 = net2net_ops.net2wider_linear(
                layer, next_layer, new_width
            )
            return model
    if strat_number == 3:
        print("Model growing ...")
        layer, next_layer, norm_layer = model.lin1, model.lin2, model.norm

        new_width = int(layer.out_features * 4)
        new_layer, new_next_layer, new_norm_layer = net2net_ops.net2wider_linear(
            layer, next_layer, new_width, norm_layer
        )

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


def evaluate(test_loader, model, criterion, device="cpu", train=False):
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

    # Just make sure that we display it when really evaluating on the test set
    if not train:
        print(f"Evaluation -> Loss: {test_loss:.3f}, Accuracy: {test_acc:.2f}%")

    return test_loss, test_acc


if __name__ == "__main__":

    utils.seed_everything(691)
    device = utils.get_device()
    project_folder = utils.get_project_root()
    data_folder = utils.get_data_folder()
    results_dir = utils_net2net.create_result_dir(
        task_name=f"net2net_{command_line_args.dataset}"
    )
    sys.stdout = logger.Logger(os.path.join(results_dir, "train_log.txt"))

    NO_GROWTH_BASELINE = command_line_args.no_growth_for_baseline

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
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    global_losses_over_growth = []
    global_accs_over_growth = []
    global_train_duration = 0

    print("\nStart training for multiple growth ...")
    for growth_number in range(N_GROWTH + 1):

        growth_results_dir = os.path.join(results_dir, f"growth{growth_number}")
        os.makedirs(growth_results_dir, exist_ok=True)

        print(f"\nBegin step {growth_number}")
        if not NO_GROWTH_BASELINE:
            model = growth_strategy(model, growth_number, STRAT_NUMBER)
            model = model.to(device)
            optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
        else:
            print("No growth here")

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
