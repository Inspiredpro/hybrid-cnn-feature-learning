import random
import warnings

import numpy as np
import tensorflow as tf

from preprocess import load_data
from cnn_model import build_cnn, train_cnn
from feature_extractor import (
    create_feature_extractor,
    extract_features,
    extract_all_features
)
from lightgbm_model import (
    build_lightgbm,
    train_lightgbm,
    retrain_lightgbm,
    predict
)
from evaluate import (
    evaluate_model,
    convert_predictions,
    create_submission
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
# Feature Extraction
# ============================================================

feature_extractor = create_feature_extractor(cnn)

(
    X_train_features,
    X_valid_features,
    X_test_features
) = extract_features(

    feature_extractor,

    X_train,

    X_valid,

    X_test

)

# ============================================================
# Train LightGBM
# ============================================================

lgb_model = build_lightgbm()

lgb_model = train_lightgbm(

    lgb_model,

    X_train_features,

    y_train_binary

)

# ============================================================
# Evaluate
# ============================================================

evaluate_model(

    lgb_model,

    X_valid_features,

    y_valid_binary

)

# ============================================================
# Retrain on Full Dataset
# ============================================================

X_all_features = extract_all_features(

    feature_extractor,

    X

)

y_all = (y == 1).astype(int)

lgb_model = retrain_lightgbm(

    lgb_model,

    X_all_features,

    y_all

)

# ============================================================
# Predict Test Set
# ============================================================

X_test_features = feature_extractor.predict(

    X_test,

    verbose=1

)

test_prediction = predict(

    lgb_model,

    X_test_features

)

test_prediction = convert_predictions(

    test_prediction

)

# ============================================================
# Submission
# ============================================================

create_submission(

    test,

    test_prediction

)

print("\nPipeline completed successfully!")
