import tensorflow as tf


def create_feature_extractor(cnn):
    """
    Create a feature extractor from the trained CNN.
    """

    feature_extractor = tf.keras.Model(
        inputs=cnn.input,
        outputs=cnn.get_layer("feature_layer").output
    )

    return feature_extractor


def extract_features(
    feature_extractor,
    X_train,
    X_valid,
    X_test
):
    """
    Extract deep features from the CNN.
    """

    print("Extracting CNN Features...")

    X_train_features = feature_extractor.predict(
        X_train,
        verbose=1
    )

    X_valid_features = feature_extractor.predict(
        X_valid,
        verbose=1
    )

    X_test_features = feature_extractor.predict(
        X_test,
        verbose=1
    )

    print("Training Features :", X_train_features.shape)
    print("Validation Features:", X_valid_features.shape)
    print("Test Features      :", X_test_features.shape)

    return (
        X_train_features,
        X_valid_features,
        X_test_features
    )


def extract_all_features(
    feature_extractor,
    X
):
    """
    Extract features from the complete training dataset.
    """

    print("Extracting Features From Full Training Set...")

    X_all_features = feature_extractor.predict(
        X,
        verbose=1
    )

    return X_all_features
