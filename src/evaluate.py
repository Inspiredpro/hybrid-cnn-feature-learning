import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score


def evaluate_model(
    lgb_model,
    X_valid_features,
    y_valid_binary
):
    """
    Evaluate the trained LightGBM model on the validation set.
    """

    valid_pred = lgb_model.predict(

        X_valid_features

    )

    score = accuracy_score(

        y_valid_binary,

        valid_pred

    )

    print("=" * 40)
    print(f"Validation Accuracy : {score:.4f}")
    print("=" * 40)

    return score


def convert_predictions(predictions):
    """
    Convert predictions from {0,1} back to {-1,1}.
    """

    predictions = np.where(

        predictions == 1,

        1,

        -1

    )

    return predictions


def create_submission(
    test,
    predictions,
    filename="submission.csv"
):
    """
    Create the Kaggle/competition submission file.
    """

    submission = pd.DataFrame({

        "row_id": test["row_id"],

        "label": predictions

    })

    submission.to_csv(

        filename,

        index=False

    )

    print(f"{filename} generated successfully!")

    print(submission.head())

    return submission
