from lightgbm import LGBMClassifier


def build_lightgbm():
    """
    Create a LightGBM classifier.
    """

    lgb_model = LGBMClassifier(

        n_estimators=500,

        learning_rate=0.03,

        max_depth=8,

        num_leaves=31,

        subsample=0.8,

        colsample_bytree=0.8,

        random_state=42

    )

    return lgb_model


def train_lightgbm(
    lgb_model,
    X_train_features,
    y_train_binary
):
    """
    Train the LightGBM model.
    """

    print("Training LightGBM...")

    lgb_model.fit(

        X_train_features,

        y_train_binary

    )

    return lgb_model


def retrain_lightgbm(
    lgb_model,
    X_all_features,
    y_all
):
    """
    Retrain LightGBM on the complete training dataset.
    """

    print("Retraining on Full Training Set...")

    lgb_model.fit(

        X_all_features,

        y_all

    )

    return lgb_model


def predict(
    lgb_model,
    features
):
    """
    Generate predictions.
    """

    predictions = lgb_model.predict(

        features

    )

    return predictions
