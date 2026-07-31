import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedShuffleSplit


def process_pixels(series):
    """
    Convert pixel strings into 28x28 NumPy arrays.
    """

    images = []

    for row in series:
        pixels = np.fromstring(row, sep=" ")
        pixels = pixels.reshape(28, 28)
        images.append(pixels)

    return np.array(images)


def load_data(
    train_path="train.csv",
    test_path="test.csv",
    test_size=0.2,
    random_state=42
):
    """
    Load, preprocess and split the dataset.
    """

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    # Process images
    X = process_pixels(train["values"])
    y = train["label"]

    X_test = process_pixels(test["values"])

    # Normalize
    X = X.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0

    # Add channel dimension
    X = X[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    # Stratified split
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state
    )

    train_idx, valid_idx = next(
        splitter.split(X, y)
    )

    X_train = X[train_idx]
    X_valid = X[valid_idx]

    y_train = y.iloc[train_idx]
    y_valid = y.iloc[valid_idx]

    # Convert labels {-1,1} -> {0,1}
    y_train_binary = (y_train == 1).astype(int)
    y_valid_binary = (y_valid == 1).astype(int)

    return (
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
    )
