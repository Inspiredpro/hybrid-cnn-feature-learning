
import os
import random
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.models import load_model

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

SEED = 42

os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

os.makedirs("results", exist_ok=True)

# ======================================================
# Load validation data
# ======================================================
X_valid = np.load("data/processed/X_valid.npy")
y_valid = np.load("data/processed/y_valid_binary.npy")

# ======================================================
# Load models
# ======================================================
feature_extractor = load_model(
    "results/feature_extractor.keras"
)

with open(
    "results/lightgbm.pkl",
    "rb"
) as f:
    lgbm = pickle.load(f)

# ======================================================
# Extract CNN Features
# ======================================================
X_valid_features = feature_extractor.predict(
    X_valid,
    verbose=0
)

# ======================================================
# Predictions
# ======================================================
y_prob = lgbm.predict_proba(
    X_valid_features
)[:,1]

y_pred = (y_prob >= 0.5).astype(int)

# ======================================================
# Metrics
# ======================================================
accuracy = accuracy_score(
    y_valid,
    y_pred
)

precision = precision_score(
    y_valid,
    y_pred
)

recall = recall_score(
    y_valid,
    y_pred
)

f1 = f1_score(
    y_valid,
    y_pred
)

auc = roc_auc_score(
    y_valid,
    y_prob
)

metrics = pd.DataFrame({

    "Metric":[
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score",
        "AUC"
    ],

    "Value":[
        accuracy,
        precision,
        recall,
        f1,
        auc
    ]

})

metrics.to_csv(
    "results/hybrid_metrics.csv",
    index=False
)

# ======================================================
# Confusion Matrix
# ======================================================
cm = confusion_matrix(
    y_valid,
    y_pred
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "Others",
        "T-Shirt/Top"
    ]
)

disp.plot(cmap="Blues")

plt.title("Hybrid CNN + LightGBM Confusion Matrix")

plt.tight_layout()

plt.savefig(
    "results/hybrid_confusion_matrix.png",
    dpi=300
)

plt.close()

# ======================================================
# Classification Report
# ======================================================
report = classification_report(
    y_valid,
    y_pred,
    target_names=[
        "Others",
        "T-Shirt/Top"
    ]
)

with open(
    "results/hybrid_classification_report.txt",
    "w"
) as f:
    f.write(report)

# ======================================================
# ROC Curve
# ======================================================
fpr, tpr, _ = roc_curve(
    y_valid,
    y_prob
)

plt.figure(figsize=(6,6))

plt.plot(
    fpr,
    tpr,
    label=f"AUC = {auc:.4f}"
)

plt.plot(
    [0,1],
    [0,1],
    "--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Hybrid CNN + LightGBM ROC Curve")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "results/hybrid_roc_curve.png",
    dpi=300
)

plt.close()

# ======================================================
# Summary
# ======================================================
with open(
    "results/hybrid_summary.txt",
    "w"
) as f:

    f.write(f"Accuracy : {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall   : {recall:.4f}\n")
    f.write(f"F1-score : {f1:.4f}\n")
    f.write(f"AUC      : {auc:.4f}\n")

print("="*60)
print("HYBRID EVALUATION COMPLETE")
print("="*60)
print(metrics)
