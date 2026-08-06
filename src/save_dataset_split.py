
import os
import random
import numpy as np
import tensorflow as tf

from preprocess import load_data

SEED = 42

os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

os.makedirs("data/processed", exist_ok=True)

(
    X_train,
    X_valid,
    X_test,
    y_train,
    y_valid,
    y_test,
    y_train_binary,
    y_valid_binary
) = load_data()

np.save("data/processed/X_train.npy", X_train)
np.save("data/processed/X_valid.npy", X_valid)
np.save("data/processed/X_test.npy", X_test)

np.save("data/processed/y_train.npy", y_train)
np.save("data/processed/y_valid.npy", y_valid)
np.save("data/processed/y_test.npy", y_test)

np.save("data/processed/y_train_binary.npy", y_train_binary)
np.save("data/processed/y_valid_binary.npy", y_valid_binary)

print("=" * 50)
print("Dataset split saved successfully.")
print("=" * 50)
print(f"X_train: {X_train.shape}")
print(f"X_valid: {X_valid.shape}")
print(f"X_test : {X_test.shape}")
