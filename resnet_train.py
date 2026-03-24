'''Train ResNet with PyTorch'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import time
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset
from tqdm import tqdm
from datasets import load_dataset
import os
import json
import matplotlib.pyplot as plt

import config as cfg


class UTKFaceDataset(Dataset):
    def __init__(self, dataset, max_age):
        self.dataset = dataset
        self.max_age = max_age
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.transform(self.dataset[idx]["image"])
        label = torch.tensor(self.dataset[idx]["age"], dtype=torch.float32)
        return image, label


if __name__ == "__main__":
    # ── Read everything from config.py ──────────────────────────
    dataset_name    = cfg.DATASET
    batch_size      = cfg.BATCH_SIZE
    learning_rate   = cfg.LEARNING_RATE
    weight_decay    = cfg.WEIGHT_DECAY
    momentum        = cfg.MOMENTUM
    epochs          = cfg.EPOCHS
    criterion_name  = cfg.CRITERION
    target_dim      = cfg.TARGET_DIM
    use_default_head = cfg.USE_DEFAULT_HEAD
    device          = cfg.DEVICE
    lr_decay        = 0.1       # kept as original default
    lr_decay_steps  = 3         # kept as original default

    epoch_list = [1,   2,   3,   4,   5,   6,   7,   8,   9,   10,   11,
                  12,  13,  14,  16,  17,  19,  20,  22,  24,  27,   29,
                  32,  35,  38,  42,  45,  50,  54,  59,  65,  71,   77,
                  85,  92,  100, 101, 110, 121, 132, 144, 158, 172, 188, 206,
                  225, 245, 268, 293, 320, 350]
    epoch_list = [i - 1 for i in epoch_list]

    # If config supplies an explicit epoch list, use that instead
    if cfg.EPOCH_LIST is not None:
        epoch_list = [i - 1 for i in cfg.EPOCH_LIST]

    start_epoch = 0
    train_losses, test_losses = [], []
    training_dict, test_dict = {}, {}

    # ── Prepare data ────────────────────────────────────────────
    print('==> Preparing data..')

    if dataset_name == "carla":
        carla_dir = cfg.CARLA_DATA_DIR
        filenames = os.listdir(carla_dir)
        carla_data = []
        target_data = []
        carla_name = os.path.join(carla_dir, filenames[0])
        target_name = os.path.join(carla_dir, filenames[6])
        with open(carla_name, "rb") as f:
            a = np.load(f)
        carla_data.append(a)
        with open(target_name, "rb") as f:
            a = np.load(f)
        target_data.append(a)

        if target_dim == 2:
            carla_data_numpy = carla_data[0][:50000]
            target_data_numpy = target_data[0][:50000, [0, 10]]
            target_data_numpy[:, 0] = (target_data_numpy[:, 0] - min(target_data_numpy[:, 0])) / (max(target_data_numpy[:, 0]) - min(target_data_numpy[:, 0]))
            target_data_numpy[:, 1] = (target_data_numpy[:, 1] - min(target_data_numpy[:, 1])) / (max(target_data_numpy[:, 1]) - min(target_data_numpy[:, 1]))
            carla_train, carla_test, target_train, target_test = train_test_split(
                carla_data_numpy, target_data_numpy, test_size=0.2, random_state=42)
            last_dim = target_dim
            save_dir = 'regression/carla2d/resnet18'
            net = torchvision.models.resnet18()
        elif target_dim == 1:
            carla_data_numpy = carla_data[0][:50000]
            target_data_numpy = target_data[0][:50000, [10]]
            target_data_numpy[:, 0] = (target_data_numpy[:, 0] - min(target_data_numpy[:, 0])) / (max(target_data_numpy[:, 0]) - min(target_data_numpy[:, 0]))
            carla_train, carla_test, target_train, target_test = train_test_split(
                carla_data_numpy, target_data_numpy, test_size=0.2, random_state=42)
            last_dim = target_dim
            save_dir = 'regression/carla1d/resnet18'
            net = torchvision.models.resnet18()

        carla_train = torch.Tensor(carla_train)
        target_train = torch.Tensor(target_train)
        carla_train_dataset = TensorDataset(carla_train, target_train)
        train_data = DataLoader(carla_train_dataset, batch_size=batch_size,
                                shuffle=True, drop_last=True)

        carla_test = torch.Tensor(carla_test)
        target_test = torch.Tensor(target_test)
        carla_test_dataset = TensorDataset(carla_test, target_test)
        val_data = DataLoader(carla_test_dataset, batch_size=100,
                              shuffle=False, drop_last=True)

    elif dataset_name == "utkface":
        last_dim = target_dim
        save_dir = 'regression/utkface/resnet34'
        net = torchvision.models.resnet34()
        dataset = load_dataset("nu-delta/utkface", split="train")
        dataset = dataset.shuffle(seed=42)
        utkface_train_test_split = dataset.train_test_split(test_size=0.2, seed=42)
        train_dataset = utkface_train_test_split["train"]
        val_dataset = utkface_train_test_split["test"]
        max_age = max([item["age"] for item in train_dataset])
        train_utkface_dataset = UTKFaceDataset(train_dataset, max_age)
        val_utkface_dataset = UTKFaceDataset(val_dataset, max_age)
        train_data = DataLoader(train_utkface_dataset, batch_size=batch_size,
                                shuffle=True, drop_last=True)
        val_data = DataLoader(val_utkface_dataset, batch_size=100,
                              shuffle=False, drop_last=True)
    else:
        raise ValueError(f"Unknown ResNet dataset: {dataset_name}")

    # ── Build model ─────────────────────────────────────────────
    print('==> Building model..')
    if use_default_head:
        net.fc = nn.Linear(net.fc.in_features, last_dim)
        save_dir += '_batch%d_lr%.5f_wd_%.5f_mom_%.2f_epoch_%d' % (
            batch_size, learning_rate, weight_decay, momentum, epochs)
    else:
        net.fc = nn.Sequential(
            nn.Linear(net.fc.in_features, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, last_dim)
        )
        save_dir += '_MLPLN_batch%d_lr%.5f_wd_%.5f_mom_%.2f_epoch_%d' % (
            batch_size, learning_rate, weight_decay, momentum, epochs)

    net = net.to(device)
    print(net)

    if criterion_name == "mse":
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(net.parameters(), lr=learning_rate,
                          momentum=momentum, weight_decay=weight_decay)
    epochs_lr_decay = [i * epochs // lr_decay_steps for i in range(1, lr_decay_steps)]
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=epochs_lr_decay,
                                                        gamma=lr_decay)

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    # ── Training & test functions ───────────────────────────────
    def train(epoch):
        global train_losses, training_dict
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        for batch_idx, (inputs, targets) in enumerate(tqdm(train_data)):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            if dataset_name == "carla":
                inputs = inputs.permute(0, 3, 1, 2)
            outputs = net(inputs)
            loss = criterion(outputs.squeeze(dim=1), targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_data)
        train_losses.append(train_loss)
        print("Training Loss: %.5f" % train_loss)
        training_dict[epoch] = train_loss
        if epoch in epoch_list:
            print('Saving..')
            state = {
                'net': net.state_dict(),
                'epoch': epoch,
            }
            torch.save(state, os.path.join(save_dir, str(epoch + 1) + '.pth'))

    def test(epoch):
        global test_losses, test_dict
        net.eval()
        test_loss = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(val_data):
                inputs, targets = inputs.to(device), targets.to(device)
                if dataset_name == "carla":
                    inputs = inputs.permute(0, 3, 1, 2)
                outputs = net(inputs)
                loss = criterion(outputs.squeeze(dim=1), targets)
                test_loss += loss.item()
        test_loss /= len(val_data)
        test_losses.append(test_loss)
        test_dict[epoch] = test_loss
        print("Test Loss: %.5f" % test_loss)

    # ── Main loop ───────────────────────────────────────────────
    for epoch in range(start_epoch, epochs):
        start_time = time.time()
        train(epoch)
        test(epoch)
        lr_scheduler.step()

    # ── Save logs & plots ───────────────────────────────────────
    titles = ["Training Loss", "Test Loss"]
    overall_losses = [train_losses, test_losses]

    with open(os.path.join(save_dir, f"train_{dataset_name}.json"), "w") as outfile:
        json.dump(training_dict, outfile)

    with open(os.path.join(save_dir, f"test_{dataset_name}.json"), "w") as outfile:
        json.dump(test_dict, outfile)

    for i in range(len(titles)):
        plt.yscale("log")
        plt.plot(overall_losses[i])
        plt.title(titles[i])
        plt.savefig(os.path.join(save_dir, '%s.pdf' % titles[i]))
        plt.close()
