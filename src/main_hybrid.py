
import os
import random
import pickle
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import load_model, Model

from lightgbm_model import build_lightgbm

SEED = 42

os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

os.makedirs("results", exist_ok=True)

# =====================================================
# Load fixed dataset split
# =====================================================
X_train = np.load("data/processed/X_train.npy")
X_valid = np.load("data/processed/X_valid.npy")

y_train = np.load("data/processed/y_train_binary.npy")
y_valid = np.load("data/processed/y_valid_binary.npy")

# =====================================================
# Load trained CNN (DO NOT RETRAIN)
# =====================================================
cnn = load_model("results/fashion_cnn.keras")

# Build model once
_ = cnn.predict(X_train[:1], verbose=0)

# =====================================================
# Feature extractor
# =====================================================
feature_extractor = Model(
    inputs=cnn.inputs,
    outputs=cnn.get_layer("feature_layer").output
)

# =====================================================
# Extract CNN Features
# =====================================================
print("Extracting CNN features...")

X_train_features = feature_extractor.predict(
    X_train,
    verbose=1
)

X_valid_features = feature_extractor.predict(
    X_valid,
    verbose=1
)

# =====================================================
# Train LightGBM
# =====================================================
print("Training LightGBM...")

lgbm = build_lightgbm()

lgbm.fit(
    X_train_features,
    y_train
)

# =====================================================
# Save Feature Extractor
# =====================================================
feature_extractor.save(
    "results/feature_extractor.keras"
)

# =====================================================
# Save LightGBM
# =====================================================
with open(
    "results/lightgbm.pkl",
    "wb"
) as f:
    pickle.dump(lgbm, f)

# =====================================================
# Save Extracted Features
# =====================================================
np.save(
    "data/processed/X_train_features.npy",
    X_train_features
)

np.save(
    "data/processed/X_valid_features.npy",
    X_valid_features
)

print("=" * 60)
print("HYBRID MODEL TRAINING COMPLETE")
print("=" * 60)
print("Saved:")
print("results/feature_extractor.keras")
print("results/lightgbm.pkl")
print("data/processed/X_train_features.npy")
print("data/processed/X_valid_features.npy")
