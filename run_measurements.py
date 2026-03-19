"""
run_measurements.py — Unified NRC (1, 3, 4) and CKA measurement script.

Supports both MLP and ResNet architectures across multiple regression datasets.
All parameters are read from config.py.

Usage:
    python run_measurements.py
"""

import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable
from scipy import linalg

import config as cfg


# ============================================================
# DATASET LOADING HELPERS
# ============================================================

def load_tabular_csv(csv_path, dataset_name):
    """Load a tabular CSV and return (X_train, X_test, y_train, y_test) as numpy arrays."""
    import pandas as pd
    from sklearn.model_selection import train_test_split

    if dataset_name == "sgemm":
        df = pd.read_csv(csv_path)
        feature_cols = [c for c in df.columns if "(ms)" not in c]
        target_cols = [c for c in df.columns if "(ms)" in c]
        X, y = df[feature_cols], df[target_cols]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train.to_numpy(), X_test.to_numpy(), y_train.to_numpy(), y_test.to_numpy()
    else:
        df = pd.read_csv(csv_path, header=None)
        split_map = {"swimmer": (8, 2), "hopper": (11, 3), "reacher": (11, 2)}
        feat_end, _ = split_map[dataset_name]
        X = df.iloc[:, :feat_end]
        y = df.iloc[:, feat_end:]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train.to_numpy(), X_test.to_numpy(), y_train.to_numpy(), y_test.to_numpy()


def get_csv_path(dataset_name):
    """Return the CSV path for a given tabular dataset name."""
    return {
        "sgemm": cfg.SGEMM_CSV,
        "swimmer": cfg.SWIMMER_CSV,
        "hopper": cfg.HOPPER_CSV,
        "reacher": cfg.REACHER_CSV,
    }[dataset_name]


def get_last_dim(dataset_name):
    """Return the target dimension for a given dataset."""
    return {"sgemm": 4, "swimmer": 2, "hopper": 3, "reacher": 2}[dataset_name]


def prepare_mlp_data():
    """Prepare DataLoaders and scalers for MLP tabular datasets."""
    from sklearn.preprocessing import StandardScaler
    from train import PrepareData

    csv_path = get_csv_path(cfg.DATASET)
    X_train, X_test, y_train, y_test = load_tabular_csv(csv_path, cfg.DATASET)

    scaler_X, scaler_y = StandardScaler(), StandardScaler()
    X_train = scaler_X.fit_transform(X_train)
    X_test = scaler_X.transform(X_test)
    y_train = scaler_y.fit_transform(y_train)
    y_test = scaler_y.transform(y_test)

    device = cfg.DEVICE
    train_ds = PrepareData(X_train, y_train, device, scale_X=False)
    test_ds = PrepareData(X_test, y_test, device, scale_X=False)

    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=100, shuffle=False)

    input_dim = X_train.shape[1]
    return train_loader, test_loader, input_dim


def prepare_resnet_data():
    """Prepare DataLoaders for ResNet image datasets (CARLA / UTKFace)."""
    from sklearn.model_selection import train_test_split as tts

    if cfg.DATASET == "carla":
        filenames = os.listdir(cfg.CARLA_DATA_DIR)
        with open(os.path.join(cfg.CARLA_DATA_DIR, filenames[0]), "rb") as f:
            carla_data = np.load(f)
        with open(os.path.join(cfg.CARLA_DATA_DIR, filenames[6]), "rb") as f:
            target_data = np.load(f)

        carla_data = carla_data[:50000]
        if cfg.TARGET_DIM == 2:
            targets = target_data[:50000, [0, 10]]
            targets[:, 0] = (targets[:, 0] - targets[:, 0].min()) / (targets[:, 0].max() - targets[:, 0].min())
            targets[:, 1] = (targets[:, 1] - targets[:, 1].min()) / (targets[:, 1].max() - targets[:, 1].min())
            test_size = 0.2
        else:
            targets = target_data[:50000, [10]]
            targets[:, 0] = (targets[:, 0] - targets[:, 0].min()) / (targets[:, 0].max() - targets[:, 0].min())
            test_size = 0.6

        X_tr, X_te, y_tr, y_te = tts(carla_data, targets, test_size=test_size, random_state=42)
        train_loader = DataLoader(
            TensorDataset(torch.Tensor(X_tr), torch.Tensor(y_tr)),
            batch_size=cfg.BATCH_SIZE, shuffle=True, drop_last=True,
        )
        val_loader = DataLoader(
            TensorDataset(torch.Tensor(X_te), torch.Tensor(y_te)),
            batch_size=100, shuffle=False, drop_last=True,
        )
        return train_loader, val_loader

    elif cfg.DATASET == "utkface":
        from datasets import load_dataset
        from resnet_train import UTKFaceDataset

        dataset = load_dataset("nu-delta/utkface", split="train").shuffle(seed=42)
        splits = dataset.train_test_split(test_size=0.2, seed=42)
        max_age = max(item["age"] for item in splits["train"])
        train_loader = DataLoader(
            UTKFaceDataset(splits["train"], max_age),
            batch_size=cfg.BATCH_SIZE, shuffle=True, drop_last=True,
        )
        val_loader = DataLoader(
            UTKFaceDataset(splits["test"], max_age),
            batch_size=100, shuffle=False, drop_last=True,
        )
        return train_loader, val_loader


# ============================================================
# MODEL BUILDERS
# ============================================================

def build_mlp_model(input_dim, last_dim):
    """Build and return an MLP model + save_dir string."""
    from train import MLP, MLPLN, MLPBN

    if cfg.USE_LAYERNORM:
        model = MLPLN(input_dim, cfg.HIDDEN_DIM, last_dim, cfg.NUM_LAYERS, use_layernorm=True)
        save_dir = f"./{cfg.DATASET}/MLPLN_{cfg.NUM_LAYERS}_layers"
    elif cfg.USE_BATCHNORM:
        model = MLPBN(input_dim, cfg.HIDDEN_DIM, last_dim, cfg.NUM_LAYERS, use_batchnorm=True)
        save_dir = f"./{cfg.DATASET}/MLPBN_{cfg.NUM_LAYERS}_layers"
    else:
        model = MLP(input_dim, cfg.HIDDEN_DIM, last_dim, num_layers=cfg.NUM_LAYERS)
        save_dir = f"./{cfg.DATASET}/MLP_{cfg.HIDDEN_DIM}_{cfg.NUM_LAYERS}_layers"

    save_dir += "_batch%d_lr%.5f_wd_%.5f_mom_%.2f_epoch_%d" % (
        cfg.BATCH_SIZE, cfg.LEARNING_RATE, cfg.WEIGHT_DECAY, cfg.MOMENTUM, cfg.EPOCHS,
    )
    model = model.to(cfg.DEVICE)
    return model, save_dir


def build_resnet_model():
    """Build and return a ResNet model + save_dir string."""
    import torchvision

    last_dim = cfg.TARGET_DIM
    if cfg.DATASET == "carla":
        model = torchvision.models.resnet18()
        base_dir = f"regression/carla{last_dim}d/resnet18"
    elif cfg.DATASET == "utkface":
        model = torchvision.models.resnet34()
        base_dir = "regression/utkface/resnet34"

    if cfg.USE_DEFAULT_HEAD:
        model.fc = nn.Linear(model.fc.in_features, last_dim)
        save_dir = base_dir + "_batch%d_lr%.5f_wd_%.5f_mom_%.2f_epoch_%d" % (
            cfg.BATCH_SIZE, cfg.LEARNING_RATE, cfg.WEIGHT_DECAY, cfg.MOMENTUM, cfg.EPOCHS,
        )
    else:
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, last_dim),
        )
        save_dir = base_dir + "_MLPLN_batch%d_lr%.5f_wd_%.5f_mom_%.2f_epoch_%d" % (
            cfg.BATCH_SIZE, cfg.LEARNING_RATE, cfg.WEIGHT_DECAY, cfg.MOMENTUM, cfg.EPOCHS,
        )

    model = model.to(cfg.DEVICE)
    return model, save_dir


# ============================================================
# MEASUREMENTS CONTAINER
# ============================================================

class Measurements:
    """Stores NRC metric values across epochs and layers."""

    def __init__(self, num_layers):
        # NRC1 — Signal vs Noise
        self.noise_component = [[] for _ in range(num_layers)]
        self.stable_rank_H = [[] for _ in range(num_layers)]

        # NRC2 — Signal Target Alignment
        self.cka = [[] for _ in range(num_layers)]

        # NRC3 — Feature-Weight Alignment
        self.WH_alignment = [[] for _ in range(num_layers)]
        self.stable_rank_W = [[] for _ in range(num_layers)]

        # NRC4 — Prediction Error
        self.prediction_error = [[] for _ in range(num_layers)]

        # Misc
        self.target_stable_rank = 0.0
        self.loss = []


# ============================================================
# HOOK UTILITY
# ============================================================

def make_hook(storage_dict, name, target_device=None):
    """Return a forward hook that stores the input to a layer."""
    def hook(module, inp, output):
        tensor = inp[0].detach()
        if target_device:
            tensor = tensor.to(target_device, non_blocking=True)
        storage_dict[name] = tensor
    return hook


# ============================================================
# NRC METRICS  (works for both MLP and ResNet)
# ============================================================

def compute_nrc_mlp(measurements, model, criterion, dataloader, layer_idx, target_dim):
    """Compute NRC for a single layer of an MLP model."""
    model.eval()
    div = 3 if (cfg.USE_LAYERNORM or cfg.USE_BATCHNORM) else 2
    lin_layer_index = div * layer_idx
    device = cfg.DEVICE

    # --- Collect features via hook ---
    layers = {}
    handle = model.layers[lin_layer_index].register_forward_hook(
        make_hook(layers, "h")
    )

    all_H, all_Y = [], []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            model(inputs)
            all_H.append(layers["h"].clone())
            all_Y.append(labels)
            layers.clear()
    handle.remove()

    H = torch.cat(all_H, dim=0)
    Y = torch.cat(all_Y, dim=0)
    n = H.shape[0]

    H_center = H - H.mean(dim=0)
    Y_center = Y - Y.mean(dim=0)

    # Covariance of H
    Sh = (H_center.t() @ H_center) / n
    U_H, S_H, V_H = torch.linalg.svd(Sh, full_matrices=False)

    i = layer_idx  # shorthand

    # --- NRC1: Signal vs Noise ---
    if "nrc1" in cfg.METRICS:
        measurements.noise_component[i].append(
            torch.abs(
                1.0 - torch.trace(V_H[:target_dim] @ (Sh @ V_H[:target_dim].t())) / torch.trace(Sh).item()
            ).cpu().numpy()
        )
        measurements.stable_rank_H[i].append(
            (S_H.norm() ** 2 / S_H[0] ** 2).cpu().numpy()
        )

    # --- NRC2: CKA: Layer features vs Labels ---
    if "nrc2" in cfg.METRICS:
        target_dev = cfg.TARGET_DEVICE
        H1_cka = H.clone()
        H2_cka = Y.clone()
        if H1_cka.dim() > 2:
            H1_cka = H1_cka.reshape(H1_cka.shape[0], -1)
        if H2_cka.dim() > 2:
            H2_cka = H2_cka.reshape(H2_cka.shape[0], -1)
        cka_val = compute_cka_core(H1_cka, H2_cka, target_dev)
        measurements.cka[i].append(cka_val)
        print(f"    CKA(layer {i}) = {cka_val:.4f}")

    # --- NRC3: Feature-Weight Alignment ---
    if "nrc3" in cfg.METRICS:
        W = model.layers[lin_layer_index].weight.data.clone()
        U_W, S_W, V_W = torch.linalg.svd(W, full_matrices=False)
        measurements.WH_alignment[i].append(
            torch.linalg.svdvals(V_H[:target_dim] @ V_W[:target_dim].t()).mean().item()
        )
        measurements.stable_rank_W[i].append(
            (S_W.norm() ** 2 / S_W[0] ** 2).cpu().numpy()
        )

    # --- NRC4: Prediction Error ---
    if "nrc4" in cfg.METRICS:
        P = torch.linalg.pinv(H, rtol=1e-3) @ Y_center
        measurements.prediction_error[i].append(criterion(H @ P, Y_center).item())

    # Target stable rank (only needs computing once; last call wins — fine)
    SY = (Y_center.t() @ Y_center) / n
    _, S_Y, _ = torch.linalg.svd(SY, full_matrices=False)
    measurements.target_stable_rank = (S_Y.norm() ** 2 / S_Y[0] ** 2).item()


def get_resnet_layer_names(model):
    """Return all Conv2d layer names inside layer1-4 (excluding downsample) plus fc."""
    layer_names = [
        name for name, module in model.named_modules()
        if name.startswith(("layer1", "layer2", "layer3", "layer4"))
        and isinstance(module, nn.Conv2d)
        and "downsample" not in name
    ]
    layer_names.append("fc")
    return layer_names


def compute_nrc_resnet(measurements, model, criterion, dataloader, layer_name, target_dim, layer_idx=0):
    """Compute NRC1, NRC2, NRC3, NRC4, for a named layer of a ResNet model.
    
    Args:
        layer_idx: index into the measurements lists (0 for single-layer mode,
                   or the layer's position when iterating all layers).
    """
    model.eval()
    device = cfg.DEVICE
    target_dev = cfg.TARGET_DEVICE
    eps = 1e-12

    layers = {}
    all_H, all_Y = [], []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            handle = model.get_submodule(layer_name).register_forward_hook(
                make_hook(layers, layer_name, target_device=target_dev)
            )
            if cfg.DATASET == "carla":
                inputs = inputs.permute(0, 3, 1, 2)
            model(inputs)
            all_H.append(layers[layer_name])
            all_Y.append(labels)
            handle.remove()
            layers.clear()

    H_raw = torch.cat(all_H, dim=0)
    Y = torch.cat(all_Y, dim=0).to(target_dev)
    n = H_raw.shape[0]

    # Global average pooling for conv layers
    is_conv = model.get_submodule(layer_name).weight.dim() > 2
    H = H_raw.mean(dim=(2, 3)) if is_conv else H_raw

    H_center = H - H.mean(dim=0)
    Y_center = (Y - Y.mean(dim=0)).reshape(-1, target_dim)

    # Covariance
    Sh = (H_center.t() @ H_center) / n
    Sh_np = Sh.cpu().numpy()
    U_H, S_H, V_H = np.linalg.svd(Sh_np, full_matrices=False)
    U_H = torch.from_numpy(U_H).to(target_dev)
    S_H = torch.from_numpy(S_H).to(target_dev)
    V_H = torch.from_numpy(V_H).to(target_dev)
    Sh = torch.from_numpy(Sh_np).to(target_dev)

    i = layer_idx

    # --- NRC1 ---
    if "nrc1" in cfg.METRICS:
        measurements.noise_component[i].append(
            torch.abs(
                1.0 - torch.trace(V_H[:target_dim] @ (Sh @ V_H[:target_dim].t())) / torch.trace(Sh).item()
            ).cpu().numpy()
        )
        sr = (S_H.norm() ** 2 / S_H[0] ** 2).cpu().numpy()
        if np.isnan(sr):
            sr = (S_H.norm() ** 2 / (S_H[0] ** 2 + eps)).cpu().numpy()
        measurements.stable_rank_H[i].append(sr)
        measurements.nrc1[i].append(
            (S_H[target_dim:].norm() ** 2 / S_H.norm() ** 2).cpu().numpy()
        )
    
    # --- NRC2: CKA: Layer features vs Labels ---
    if "nrc2" in cfg.METRICS:
        H1_cka = H.clone()
        H2_cka = Y.clone().to(target_dev)
        if H1_cka.dim() > 2:
            H1_cka = H1_cka.mean(dim=(2, 3))
        if H2_cka.dim() > 2:
            H2_cka = H2_cka.mean(dim=(2, 3))
        if H2_cka.dim() < 2:
            H2_cka = H2_cka.unsqueeze(1)
        cka_val = compute_cka_core(H1_cka, H2_cka, target_dev)
        measurements.cka[i].append(cka_val)
        print(f"    CKA({layer_name}) = {cka_val:.4f}")

    # --- NRC3 ---
    if "nrc3" in cfg.METRICS:
        W = model.get_submodule(layer_name).weight.data.clone().to(target_dev)
        if is_conv:
            W = W.transpose(1, 0).flatten(start_dim=1)
        U_W, S_W, V_W = torch.linalg.svd(W, full_matrices=False)
        if is_conv:
            measurements.WH_alignment[i].append(
                torch.linalg.svdvals(V_H[:target_dim] @ U_W[:, :target_dim]).mean().item()
            )
        else:
            measurements.WH_alignment[i].append(
                torch.linalg.svdvals(V_H[:target_dim] @ V_W[:target_dim].t()).mean().item()
            )
        measurements.stable_rank_W[i].append(
            (S_W.norm() ** 2 / S_W[0] ** 2).cpu().numpy()
        )

    # --- NRC4 ---
    if "nrc4" in cfg.METRICS:
        P = torch.linalg.pinv(H, rtol=1e-3) @ Y_center
        measurements.prediction_error[i].append(criterion(H @ P, Y_center).item())

    # Target stable rank
    SY = (Y_center.t() @ Y_center) / n
    if SY.dim() < 2:
        SY = SY.reshape(1, 1)
    _, S_Y, _ = torch.linalg.svd(SY, full_matrices=False)
    measurements.target_stable_rank = (S_Y.norm() ** 2 / S_Y[0] ** 2).item()


# ============================================================
# CKA COMPUTATION  (shared core, architecture-agnostic)
# ============================================================

def compute_cka_core(H1, H2, target_device):
    """
    Compute CKA between two feature matrices H1 (N×D1) and H2 (N×D2).
    Returns a scalar float.
    """
    N = H1.shape[0]
    eps = 1e-12

    # Truncate H1 columns to match H2 if needed
    H1 = H1[:, :H2.shape[1]]

    centering = torch.eye(N, device=target_device) - torch.ones(N, N, device=target_device) / N

    K = (H1 @ H1.t()).to(target_device)
    L = (H2 @ H2.t()).to(target_device)

    Kc = centering @ K @ centering
    Lc = centering @ L @ centering

    HSIC_KL = torch.trace(Kc @ Lc)
    HSIC_KK = torch.trace(Kc @ Kc)
    HSIC_LL = torch.trace(Lc @ Lc)

    denom = torch.sqrt(HSIC_KK * HSIC_LL + eps)
    if denom <= eps:
        return 0.0
    return (HSIC_KL / denom).item()



# ============================================================
# DEFAULT EPOCH LISTS
# ============================================================

DEFAULT_MLP_EPOCHS = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 19, 20,
    22, 24, 27, 29, 32, 35, 38, 42, 45, 50, 54, 59, 65, 71, 77, 85,
    92, 100, 101, 110, 121, 132, 144, 158, 172, 188, 206, 225, 245,
    268, 293, 320, 350, 380, 420, 450, 500, 530, 560, 600, 630, 650,
    680, 720, 750, 790, 820, 850, 880, 900, 930, 970, 1000,
]

DEFAULT_RESNET_EPOCHS = [1, 10, 50, 100, 206, 245]


# ============================================================
# PLOTTING
# ============================================================

def plot_nrc_metrics(measurements, epoch_list, save_dir, num_layers, layer_labels=None):
    """Generate and save plots for NRC/CKA metrics.
    
    Args:
        layer_labels: optional list of human-readable layer names for plot titles.
                      Falls back to "layer 0", "layer 1", etc.
    """
    plot_dir = os.path.join(save_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    attrs = []
    if "nrc1" in cfg.METRICS:
        attrs += ["stable_rank_H", "noise_component"]
    if "nrc3" in cfg.METRICS:
        attrs += ["WH_alignment", "stable_rank_W"]
    if "nrc4" in cfg.METRICS:
        attrs += ["prediction_error"]
    if "nrc2" in cfg.METRICS:
        attrs += ["cka"]

    for attr_name in attrs:
        data = getattr(measurements, attr_name)
        for layer_i in range(num_layers):
            if not data[layer_i]:
                continue
            label = layer_labels[layer_i] if layer_labels else f"layer {layer_i}"
            safe_label = label.replace(".", "_")
            plt.figure()
            plt.plot(epoch_list[: len(data[layer_i])], data[layer_i], "bx-")
            plt.title(f"{attr_name} — {label}")
            plt.xlabel("Epoch")
            plt.grid(True)
            plt.savefig(os.path.join(plot_dir, f"{attr_name}_{safe_label}.pdf"))
            plt.close()



# ============================================================
# MAIN
# ============================================================

def main():
    has_metrics = any(m in cfg.METRICS for m in ["nrc1", "nrc3", "nrc4", "cka"])

    # --------------------------------------------------------
    # MLP ARCHITECTURE
    # --------------------------------------------------------
    if cfg.ARCHITECTURE == "mlp":
        print("==> Preparing MLP data...")
        train_loader, test_loader, input_dim = prepare_mlp_data()
        last_dim = get_last_dim(cfg.DATASET)
        model, save_dir = build_mlp_model(input_dim, last_dim)
        print(model)

        div = 3 if (cfg.USE_LAYERNORM or cfg.USE_BATCHNORM) else 2
        num_layers = cfg.NUM_LAYERS # (len(model.layers) // div) + 1
        criterion = nn.MSELoss() if cfg.CRITERION == "mse" else nn.CrossEntropyLoss(reduction="sum")

        if has_metrics:
            epoch_list = cfg.EPOCH_LIST or DEFAULT_MLP_EPOCHS
            epoch_list = [e for e in epoch_list if e <= cfg.EPOCHS]
            measurements = Measurements(num_layers)

            for e in epoch_list:
                print(f"[Metrics] Loading checkpoint epoch {e}...")
                ckpt = torch.load(os.path.join(save_dir, f"{e}.pth"), map_location="cpu")
                model.load_state_dict(ckpt["model"])
                for layer_i in range(num_layers):
                    compute_nrc_mlp(measurements, model, criterion, train_loader, layer_i, last_dim)

            # Save results
            os.makedirs(os.path.join(save_dir, "plots"), exist_ok=True)
            with open(os.path.join(save_dir, f"{cfg.CRITERION}_nrc.pkl"), "wb") as f:
                pickle.dump(measurements, f)
            plot_nrc_metrics(measurements, epoch_list, save_dir, num_layers)

            # Save CKA arrays separately for convenience
            if "nrc2" in cfg.METRICS:
                for layer_i in range(num_layers):
                    cka_arr = np.array(measurements.cka[layer_i])
                    np.save(os.path.join(save_dir, f"CKA_layer{layer_i}.npy"), cka_arr)

            print(f"[Results] Saved to {save_dir}")

    # --------------------------------------------------------
    # RESNET ARCHITECTURE
    # --------------------------------------------------------
    elif cfg.ARCHITECTURE == "resnet":
        print("==> Preparing ResNet data...")
        train_loader, val_loader = prepare_resnet_data()
        model, save_dir = build_resnet_model()
        print(model)

        criterion = nn.MSELoss() if cfg.CRITERION == "mse" else nn.CrossEntropyLoss(reduction="sum")
        target_dim = cfg.TARGET_DIM

        if has_metrics:
            epoch_list = cfg.EPOCH_LIST or DEFAULT_RESNET_EPOCHS
            epoch_list = [e for e in epoch_list if e <= cfg.EPOCHS]

            # Determine which layers to measure
            if cfg.RESNET_LAYER_NAME:
                # Single-layer mode
                layer_names = [cfg.RESNET_LAYER_NAME]
            else:
                # All conv layers in blocks + fc
                layer_names = get_resnet_layer_names(model)

            num_layers = len(layer_names)
            measurements = Measurements(num_layers)
            print(f"[Metrics] Measuring {num_layers} layer(s): {layer_names}")

            for e in epoch_list:
                print(f"[Metrics] Loading checkpoint epoch {e}...")
                ckpt = torch.load(os.path.join(save_dir, f"{e}.pth"), map_location="cpu")
                model.load_state_dict(ckpt["net"])
                for layer_i, layer_name in enumerate(layer_names):
                    print(f"  Processing layer: {layer_name}")
                    compute_nrc_resnet(
                        measurements, model, criterion, train_loader,
                        layer_name, target_dim, layer_idx=layer_i,
                    )

            # Save — use layer subdirectory for single-layer, main save_dir for all-layers
            if len(layer_names) == 1:
                results_dir = os.path.join(save_dir, layer_names[0])
            else:
                results_dir = os.path.join(save_dir, "all_layers")
            os.makedirs(os.path.join(results_dir, "plots"), exist_ok=True)

            with open(os.path.join(results_dir, f"{cfg.CRITERION}_nrc.pkl"), "wb") as f:
                pickle.dump(measurements, f)

            # For plotting, use layer names as labels when multiple layers
            plot_nrc_metrics(measurements, epoch_list, results_dir, num_layers,
                             layer_labels=layer_names)

            # Save CKA arrays
            if "nrc2" in cfg.METRICS:
                for layer_i, lname in enumerate(layer_names):
                    cka_arr = np.array(measurements.cka[layer_i])
                    safe_name = lname.replace(".", "_")
                    np.save(os.path.join(results_dir, f"CKA_{safe_name}.npy"), cka_arr)

            print(f"[Results] Saved to {results_dir}")

    else:
        raise ValueError(f"Unknown architecture: {cfg.ARCHITECTURE}. Use 'mlp' or 'resnet'.")

    print("Done!")


if __name__ == "__main__":
    main()
