
import os
import random
import pickle
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Model

from cnn_model import build_cnn, train_cnn
from lightgbm_model import build_lightgbm

SEED = 42

os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

os.makedirs("results", exist_ok=True)

# ==========================
# Load fixed dataset split
# ==========================
X_train = np.load("data/processed/X_train.npy")
X_valid = np.load("data/processed/X_valid.npy")

y_train = np.load("data/processed/y_train_binary.npy")
y_valid = np.load("data/processed/y_valid_binary.npy")

# ==========================
# Train CNN
# ==========================
cnn = build_cnn()

cnn, history = train_cnn(
    cnn,
    X_train,
    y_train,
    X_valid,
    y_valid
)

# ==========================
# Feature extractor
# ==========================
feature_extractor = Model(
    inputs=cnn.input,
    outputs=cnn.get_layer("feature_layer").output
)

# ==========================
# Extract features
# ==========================
X_train_features = feature_extractor.predict(
    X_train,
    verbose=0
)

X_valid_features = feature_extractor.predict(
    X_valid,
    verbose=0
)

# ==========================
# Train LightGBM
# ==========================
lgbm = build_lightgbm()

lgbm.fit(
    X_train_features,
    y_train
)

# ==========================
# Save feature extractor
# ==========================
feature_extractor.save(
    "results/feature_extractor.keras"
)

# ==========================
# Save LightGBM
# ==========================
with open(
    "results/lightgbm.pkl",
    "wb"
) as f:
    pickle.dump(lgbm, f)

print("=" * 60)
print("Hybrid model training complete.")
print("=" * 60)
print("Saved:")
print("results/feature_extractor.keras")
print("results/lightgbm.pkl")
