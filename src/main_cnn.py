
import os
import random
import numpy as np
import tensorflow as tf

from cnn_model import (
    build_cnn,
    train_cnn,
    evaluate_cnn
)

SEED = 42

os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

os.makedirs("results", exist_ok=True)

# ==========================
# Load fixed dataset split
# ==========================
X_train = np.load(
    "data/processed/X_train.npy"
)

X_valid = np.load(
    "data/processed/X_valid.npy"
)

y_train = np.load(
    "data/processed/y_train_binary.npy"
)

y_valid = np.load(
    "data/processed/y_valid_binary.npy"
)

# ==========================
# Build CNN
# ==========================
cnn = build_cnn()

# ==========================
# Train CNN
# ==========================
cnn, history = train_cnn(
    cnn,
    X_train,
    y_train,
    X_valid,
    y_valid
)

# ==========================
# Evaluate
# ==========================
accuracy = evaluate_cnn(
    cnn,
    X_valid,
    y_valid
)

# ==========================
# Save model
# ==========================
cnn.save(
    "results/fashion_cnn.keras"
)

print("\nCNN model saved successfully.")
