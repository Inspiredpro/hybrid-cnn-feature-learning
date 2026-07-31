
import numpy as np

from sklearn.metrics import accuracy_score

from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.callbacks import EarlyStopping


def build_cnn():
    """
    Build and compile the CNN model.
    """

    cnn = models.Sequential([

        layers.Input(shape=(28, 28, 1)),

        layers.Conv2D(
            32,
            (3, 3),
            activation="relu",
            padding="same"
        ),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Conv2D(
            64,
            (3, 3),
            activation="relu",
            padding="same"
        ),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Conv2D(
            128,
            (3, 3),
            activation="relu",
            padding="same"
        ),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),

        layers.Dense(
            256,
            activation="relu"
        ),

        layers.Dropout(0.5),

        layers.Dense(
            128,
            activation="relu",
            name="feature_layer"
        ),

        layers.Dense(
            1,
            activation="sigmoid"
        )

    ])

    cnn.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return cnn


def train_cnn(
    cnn,
    X_train,
    y_train,
    X_valid,
    y_valid,
    epochs=25,
    batch_size=64
):
    """
    Train the CNN model.
    """

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    history = cnn.fit(
        X_train,
        y_train,
        validation_data=(X_valid, y_valid),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=1
    )

    return cnn, history


def evaluate_cnn(
    cnn,
    X_valid,
    y_valid
):
    """
    Evaluate the CNN model.
    """

    probabilities = cnn.predict(
        X_valid,
        verbose=0
    )

    predictions = (
        probabilities >= 0.5
    ).astype(int).flatten()

    accuracy = accuracy_score(
        y_valid,
        predictions
    )

    print("=" * 50)
    print("CNN BASELINE")
    print("=" * 50)
    print(f"Validation Accuracy : {accuracy:.4f}")
    print("=" * 50)

    return accuracy
