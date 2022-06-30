#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
Optuna example that optimizes multi-layer perceptrons using PyTorch.

In this example, we optimize the validation accuracy of fashion product recognition using
PyTorch and FashionMNIST. We optimize the neural network architecture as well as the optimizer
configuration. As it is too time consuming to use the whole FashionMNIST dataset,
we here use a small subset of it.

"""

import os
import argparse

import optuna
from optuna.trial import TrialState
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torchvision import datasets
from torchvision import transforms

from network import Net
from dataset import MyDataset, Flip

DEVICE = torch.device("cpu")

try:
    if torch.backends.mps.is_available():
        DEVICE = torch.device("mps")      # Apple Silicon Metal
except:
    DEVICE = torch.device("cpu")

BATCHSIZE = 100
DIR = os.path.join(os.environ["HOME"], ".pytorch")
EPOCHS = 15
N_TRAIN_EXAMPLES = BATCHSIZE * 100
N_VALID_EXAMPLES = BATCHSIZE * 30


def define_model(trial):

    # network dim
    dim1 = trial.suggest_int('dim1', 10, 50)
    dim2 = trial.suggest_int('dim2', 10, 50)
    dim3 = trial.suggest_int('dim3', 10, 50)
    dim4 = trial.suggest_int('dim4', 10, 100)
    dim5 = trial.suggest_int('dim5', 10, 100)

    # ドロップアウト
    dropout = trial.suggest_float("dropout", 0.2, 0.5)

    return Net(dim1, dim2, dim3, dim4, dim5, dropout)


def get_data_loader():
    # Load FashionMNIST dataset.
    train_loader = torch.utils.data.DataLoader(
        MyDataset(csvfile="train.csv", flip=Flip.both, transform=True, repeat=4),
        batch_size=BATCHSIZE,
        shuffle=True,
    )
    valid_loader = torch.utils.data.DataLoader(
        MyDataset(csvfile="test.csv",  flip=Flip.both, transform=False, repeat=1),
        batch_size=BATCHSIZE,
        shuffle=True,
    )

    return train_loader, valid_loader

global train_loader, valid_loader
train_loader, valid_loader = get_data_loader()

def objective(trial):

    # Generate the model.
    model = define_model(trial).to(DEVICE)

    # Generate the optimizers.
    # lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    lr = trial.suggest_float("lr", 0.0005, 0.005, log=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Get the FashionMNIST dataset.
    # train_loader, valid_loader = get_data_loader()
    global train_loader, valid_loader

    # Training of the model.
    for epoch in range(EPOCHS):
        model.train()
        for batch_idx, (data, labels) in enumerate(train_loader):
            # Limiting training data for faster epochs.
            if batch_idx * BATCHSIZE >= N_TRAIN_EXAMPLES:
                break

            data, labels = data.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            loss = model.loss_function(data, labels)
            loss.backward()
            optimizer.step()

        # Validation of the model.
        model.eval()
        correct = 0
        with torch.no_grad():
            for batch_idx, (data, labels) in enumerate(valid_loader):
                # Limiting validation data.
                if batch_idx * BATCHSIZE >= N_VALID_EXAMPLES:
                    break

                data, labels = data.to(DEVICE), labels.to(DEVICE)
                
                output = model(data)
                # Get the index of the max log-probability.
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(labels.view_as(pred)).sum().item()

        accuracy = correct / min(len(valid_loader.dataset), N_VALID_EXAMPLES)

        trial.report(accuracy, epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return accuracy


def main():
    parser = argparse.ArgumentParser(description="Optuna, MNIST")
    parser.add_argument('--trial', type=int, default=10, metavar='N',
                        help='number of trial (default: 10)')
    args = parser.parse_args()
    
    n_trials = args.trial
    study_name ="hand-sign"

    study = optuna.create_study(pruner=optuna.pruners.PercentilePruner(50),
                                storage="sqlite:///result/optuna.db",
                                study_name=study_name,
                                load_if_exists=True)
    
    study.optimize(objective, n_trials=n_trials, gc_after_trial=True)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("# Study statistics: ")
    print("#   Number of finished trials: ", len(study.trials))
    print("#   Number of pruned trials: ", len(pruned_trials))
    print("#   Number of complete trials: ", len(complete_trials))

    print("# Best trial:")
    trial = study.best_trial

    print("#   Value: ", trial.value)

    print("#   Params: ")
    for key, value in trial.params.items():
        print("{} = {}".format(key, value))


if __name__ == '__main__':
    main()


# import pdb; pdb.set_trace()

### Local Variables: ###
### truncate-lines:t ###
### End: ###
