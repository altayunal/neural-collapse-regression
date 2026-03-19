"""
config.py — Central configuration for NRC & CKA measurements.

Edit the variables below, then run:
    python run_measurements.py
"""

# ============================================================
# 1. ARCHITECTURE
# ============================================================
# "mlp" or "resnet"
ARCHITECTURE = "mlp"

# ============================================================
# 2. MEASUREMENT MODE
# ============================================================
# Which metrics to compute. Any combination of: "nrc1", "nrc3", "nrc4", "cka"
METRICS = ["nrc1", "nrc2", "nrc3", "nrc4"]

# ============================================================
# 3. DATASET
# ============================================================
# For MLP:    "sgemm", "swimmer", "hopper", "reacher"
# For ResNet: "carla", "utkface"
DATASET = "sgemm"

# ============================================================
# 4. MODEL HYPERPARAMETERS (used to locate saved checkpoints)
# ============================================================
BATCH_SIZE = 512
LEARNING_RATE = 0.1
WEIGHT_DECAY = 5e-4
MOMENTUM = 0.9
EPOCHS = 1000

# ============================================================
# 5. MLP-SPECIFIC SETTINGS
# ============================================================
NUM_LAYERS = 8
HIDDEN_DIM = 512

# Normalization: exactly one should be True, or both False for plain MLP
USE_LAYERNORM = False
USE_BATCHNORM = False

# ============================================================
# 6. RESNET-SPECIFIC SETTINGS
# ============================================================
# Target output dimension (1 or 2 for CARLA, 1 for UTKFace age)
TARGET_DIM = 2

# Use default fc head (single Linear) vs MLP+LayerNorm head
USE_DEFAULT_HEAD = False

# Layer to probe for NRC measurements (ResNet only).
# Set to a specific layer name to measure a single layer, e.g. "layer4.1.conv2", "fc", "fc.3"
# Set to None or "" to automatically measure ALL conv layers in layer1-4 + fc
RESNET_LAYER_NAME = None

# ============================================================
# 7. LOSS FUNCTION
# ============================================================
# "mse" or "ce" (cross-entropy)
CRITERION = "mse"

# ============================================================
# 8. EPOCH CHECKPOINTS
# ============================================================
# Which saved checkpoints to evaluate.
# Set to None to use built-in defaults per architecture.
EPOCH_LIST = None  # e.g. [1, 10, 50, 100, 206, 245] or None

# ============================================================
# 9. DEVICE
# ============================================================
DEVICE = "cuda:0"
TARGET_DEVICE = "cuda:1"  # used for heavy matrix ops (CKA, NRC on ResNet)

# ============================================================
# 10. PATHS  (adjust if your directory layout differs)
# ============================================================
# Root directory for CARLA .npy files
CARLA_DATA_DIR = "./carla/"

# CSV paths for tabular datasets
SGEMM_CSV = "./sgemm_product.csv"
SWIMMER_CSV = "./swimmer.csv"
HOPPER_CSV = "./hopper.csv"
REACHER_CSV = "./reacher.csv"
