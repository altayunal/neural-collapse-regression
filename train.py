import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from torch.utils.data import DataLoader, Dataset
import os
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import json
from collections import OrderedDict

import config as cfg


def load_data(df):
    # Select all columns that are NOT run columns
    feature_cols = [c for c in df.columns if "(ms)" not in c]
    # Select all run columns as targets
    target_cols = [c for c in df.columns if "(ms)" in c]

    X = df[feature_cols]
    y = df[target_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train.to_numpy(), X_test.to_numpy(), y_train.to_numpy(), y_test.to_numpy()


class PrepareData(Dataset):
    def __init__(self, X, y, device, scale_X=True):
        if not torch.is_tensor(X):
            if scale_X:
                X = StandardScaler().fit_transform(X)
                self.X = torch.Tensor(X)
            else:
                self.X = torch.Tensor(X)
        if not torch.is_tensor(y):
            self.y = torch.Tensor(y)
        self.X, self.y = self.X.to(device), self.y.to(device)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class MLP(nn.Module):
    '''
    Multilayer Perceptron for regression.
    '''
    def __init__(self, dim, model_dim, last_dim, num_layers=5):
        super().__init__()
        self.last_dim = last_dim
        self.model_dim = model_dim
        layers = []
        layers.append(nn.Linear(dim, self.model_dim, bias=False))
        layers.append(nn.ReLU())
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(self.model_dim, self.model_dim, bias=False))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(self.model_dim, self.last_dim, bias=False))
        self.layers = nn.Sequential(*layers)
        self.layers[-1].lastLayer = True

    def forward(self, x):
        return self.layers(x)


class MLPBN(nn.Module):
    """MLP with BatchNorm."""
    def __init__(self, dim, model_dim, last_dim, num_hidden_layers=6, use_batchnorm=True):
        super().__init__()
        layers = []
        self.last_dim = last_dim
        self.model_dim = model_dim
        layers.append(nn.Linear(dim, self.model_dim, bias=False))
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(self.model_dim))
        layers.append(nn.ReLU())
        for _ in range(num_hidden_layers - 2):
            layers.append(nn.Linear(self.model_dim, self.model_dim, bias=False))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(self.model_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(self.model_dim, self.last_dim, bias=False))
        self.layers = nn.Sequential(*layers)
        self.layers[-1].lastLayer = True

    def forward(self, x):
        return self.layers(x)


class MLPLN(nn.Module):
    """MLP with LayerNorm."""
    def __init__(self, dim, model_dim, last_dim, num_hidden_layers=6, use_layernorm=True):
        super().__init__()
        layers = []
        self.last_dim = last_dim
        self.model_dim = model_dim
        layers.append(nn.Linear(dim, self.model_dim, bias=False))
        if use_layernorm:
            layers.append(nn.LayerNorm(self.model_dim))
        layers.append(nn.ReLU())
        for _ in range(num_hidden_layers - 2):
            layers.append(nn.Linear(self.model_dim, self.model_dim, bias=False))
            if use_layernorm:
                layers.append(nn.LayerNorm(self.model_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(self.model_dim, self.last_dim, bias=False))
        self.layers = nn.Sequential(*layers)
        self.layers[-1].lastLayer = True

    def forward(self, x):
        return self.layers(x)


if __name__ == '__main__':
    # ── Read everything from config.py ──────────────────────────
    dataset_name    = cfg.DATASET
    batch_size      = cfg.BATCH_SIZE
    learning_rate   = cfg.LEARNING_RATE
    weight_decay    = cfg.WEIGHT_DECAY
    momentum        = cfg.MOMENTUM
    epochs          = cfg.EPOCHS
    num_layers      = cfg.NUM_LAYERS
    hidden_dim      = cfg.HIDDEN_DIM
    use_layernorm   = cfg.USE_LAYERNORM
    use_batchnorm   = cfg.USE_BATCHNORM
    criterion_name  = cfg.CRITERION
    device          = cfg.DEVICE
    lr_decay        = 0.1       # kept as original default
    lr_decay_steps  = 3         # kept as original default

    save_epoch_list = [1,   2,   3,   4,   5,   6,   7,   8,   9,   10,   11,
                       12,  13,  14,  16,  17,  19,  20,  22,  24,  27,   29,
                       32,  35,  38,  42,  45,  50,  54,  59,  65,  71,   77,
                       85,  92,  100, 101, 110, 121, 132, 144, 158, 172, 188, 199, 206,
                       225, 245, 268, 293, 320, 350, 380, 420, 450, 500, 530, 560,
                       600, 630, 650, 680, 720, 750, 790, 820, 850, 880, 900, 930, 970, 1000,
                       2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000,
                       20000, 30000, 40000, 50000, 60000, 75000, 90000]
    save_epoch_list = [i - 1 for i in save_epoch_list]

    # If config supplies an explicit epoch list, use that instead
    if cfg.EPOCH_LIST is not None:
        save_epoch_list = [i - 1 for i in cfg.EPOCH_LIST]

    # ── Load dataset ────────────────────────────────────────────
    print("Reading data")
    if dataset_name == "sgemm":
        last_dim = 4
        save_dir = f'./{dataset_name}/MLP'
        df = pd.read_csv(cfg.SGEMM_CSV)
        X_train, X_test, y_train, y_test = load_data(df)
    elif dataset_name == "swimmer":
        last_dim = 2
        save_dir = f'./{dataset_name}/MLP'
        df = pd.read_csv(cfg.SWIMMER_CSV, header=None)
        X = df.iloc[:, :8]
        y = df.iloc[:, 8:]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train, X_test, y_train, y_test = X_train.to_numpy(), X_test.to_numpy(), y_train.to_numpy(), y_test.to_numpy()
    elif dataset_name == "hopper":
        last_dim = 3
        save_dir = f'./{dataset_name}/MLP'
        df = pd.read_csv(cfg.HOPPER_CSV, header=None)
        X = df.iloc[:, :11]
        y = df.iloc[:, 11:]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train, X_test, y_train, y_test = X_train.to_numpy(), X_test.to_numpy(), y_train.to_numpy(), y_test.to_numpy()
    elif dataset_name == "reacher":
        last_dim = 2
        save_dir = f'./{dataset_name}/MLP'
        df = pd.read_csv(cfg.REACHER_CSV, header=None)
        X = df.iloc[:, :11]
        y = df.iloc[:, 11:]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train, X_test, y_train, y_test = X_train.to_numpy(), X_test.to_numpy(), y_train.to_numpy(), y_test.to_numpy()
    else:
        raise ValueError(f"Unknown MLP dataset: {dataset_name}")

    # ── Scale features & targets ────────────────────────────────
    scaler, scaler_y = StandardScaler(), StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    y_train = scaler_y.fit_transform(y_train)
    y_test = scaler_y.transform(y_test)

    dataset_train = PrepareData(X_train, y_train, device, scale_X=False)
    dataset_test  = PrepareData(X_test,  y_test,  device, scale_X=False)

    train_loader = DataLoader(dataset=dataset_train, batch_size=batch_size,
                              shuffle=True, drop_last=False)
    test_loader  = DataLoader(dataset=dataset_test,  batch_size=100,
                              shuffle=False)

    # ── Build model ─────────────────────────────────────────────
    print("Creating model")
    dimension = X_train.shape[1]
    model_dim = hidden_dim

    if use_layernorm:
        model = MLPLN(dimension, model_dim, last_dim, num_layers, use_layernorm=True)
        save_dir += f"LN_{num_layers}_layers"
    elif use_batchnorm:
        model = MLPBN(dimension, model_dim, last_dim, num_layers, use_batchnorm=True)
        save_dir += f"BN_{num_layers}_layers"
    else:
        model = MLP(dimension, model_dim, last_dim, num_layers=num_layers)
        save_dir += f"_{model_dim}_{num_layers}_layers"

    model = model.to(device)
    print(model)

    save_dir += '_batch%d_lr%.5f_wd_%.5f_mom_%.2f_epoch_%d' % (
        batch_size, learning_rate, weight_decay, momentum, epochs)

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,
                                momentum=momentum, weight_decay=weight_decay)
    epochs_lr_decay = [i * epochs // lr_decay_steps for i in range(1, lr_decay_steps)]
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=epochs_lr_decay,
                                                        gamma=lr_decay)
    criterion = nn.MSELoss() if criterion_name == 'mse' else nn.CrossEntropyLoss()

    # ── Training loop ───────────────────────────────────────────
    print("Starting training")
    training_losses, test_losses = [], []
    training_dict, test_dict = {}, {}

    for i in range(epochs):
        print("Epoch %d" % i)
        model.train()
        train_loss_plot = 0.0
        test_loss_plot = 0.0

        for ix, (X_batch, y_batch) in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            prediction = model(X_batch)
            train_loss = criterion(prediction, y_batch)
            train_loss.backward()
            optimizer.step()
            train_loss_plot += train_loss.item()

        train_loss_plot /= len(train_loader)
        print("Training Loss: %.5f" % train_loss_plot)
        training_losses.append(train_loss_plot)
        training_dict[i] = train_loss_plot
        lr_scheduler.step()

        if i in save_epoch_list:
            print('Saving..')
            state = {
                'model': model.state_dict(),
                'epoch': i,
            }
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir, exist_ok=True)
            torch.save(state, os.path.join(save_dir, str(i + 1) + '.pth'))

        model.eval()
        with torch.no_grad():
            for ix, (X_batch, y_batch) in enumerate(test_loader):
                prediction = model(X_batch)
                test_loss = criterion(prediction, y_batch)
                test_loss_plot += test_loss.item()
        test_loss_plot /= len(test_loader)
        print("Test Loss: %.5f" % test_loss_plot)
        test_losses.append(test_loss_plot)
        test_dict[i] = test_loss_plot

    # ── Plotting ────────────────────────────────────────────────
    print("Starting plotting")
    plot_dir = os.path.join(save_dir, "plots")
    if not os.path.isdir(plot_dir):
        os.makedirs(plot_dir, exist_ok=True)

    titles = ["Training Loss", "Test Loss"]
    overall_losses = [training_losses, test_losses]

    with open(os.path.join(plot_dir, f"train_{dataset_name}.json"), "w") as outfile:
        json.dump(training_dict, outfile)

    with open(os.path.join(plot_dir, f"test_{dataset_name}.json"), "w") as outfile:
        json.dump(test_dict, outfile)

    for i in range(len(titles)):
        plt.yscale("log")
        plt.plot(overall_losses[i])
        plt.title(titles[i])
        plt.savefig(os.path.join(plot_dir, '%s.pdf' % titles[i]))
        plt.close()
