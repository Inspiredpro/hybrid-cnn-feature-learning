import random
import warnings

import numpy as np
import tensorflow as tf

from preprocess import load_data
from cnn_model import (
    build_cnn,
    train_cnn,
    evaluate_cnn
)

warnings.filterwarnings("ignore")

# ============================================================
# Reproducibility
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ============================================================
# Load Dataset
# ============================================================

(
    train,
    test,
    X,
    y,
    X_train,
    X_valid,
    X_test,
    y_train,
    y_valid,
    y_train_binary,
    y_valid_binary
) = load_data()

# ============================================================
# Train CNN
# ============================================================

cnn = build_cnn()

cnn, history = train_cnn(

    cnn,

    X_train,

    y_train_binary,

    X_valid,

    y_valid_binary

)

# ============================================================
# CNN Evaluation
# ============================================================

accuracy = evaluate_cnn(

    cnn,

    X_valid,

    y_valid_binary

)

print("\n========================================")
print("CNN BASELINE RESULTS")
print("========================================")
print(f"Validation Accuracy : {accuracy:.4f}")
print("========================================")

print("\nCNN Baseline completed successfully!")
