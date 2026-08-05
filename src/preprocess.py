
import os
import random
import numpy as np
import tensorflow as tf

from tensorflow.keras.datasets import fashion_mnist
from sklearn.model_selection import StratifiedShuffleSplit


SEED = 42

os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


def load_data(
    test_size=0.2,
    random_state=SEED
):
    """
    Load Fashion-MNIST and create a fixed train/validation split.
    """

    (X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

    # Normalize
    X_train_full = X_train_full.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0

    # Add channel dimension
    X_train_full = X_train_full[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    # Binary classification:
    # T-Shirt/Top = 0
    # Others = 1
    y_binary = (y_train_full == 0).astype(int)

    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state
    )

    train_idx, valid_idx = next(
        splitter.split(X_train_full, y_binary)
    )

    X_train = X_train_full[train_idx]
    X_valid = X_train_full[valid_idx]

    y_train = y_train_full[train_idx]
    y_valid = y_train_full[valid_idx]

    y_train_binary = (y_train == 0).astype(int)
    y_valid_binary = (y_valid == 0).astype(int)

    return (
        X_train,
        X_valid,
        X_test,
        y_train,
        y_valid,
        y_test,
        y_train_binary,
        y_valid_binary
    )
