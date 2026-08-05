
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.models import load_model

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    roc_curve,
    roc_auc_score
)

SEED = 42

os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

os.makedirs("results", exist_ok=True)

# ==========================
# Load fixed validation split
# ==========================
X_valid = np.load("data/processed/X_valid.npy")
y_valid = np.load("data/processed/y_valid_binary.npy")

# ==========================
# Load trained CNN
# ==========================
cnn = load_model("results/fashion_cnn.keras")

# ==========================
# Predictions
# ==========================
y_prob = cnn.predict(
    X_valid,
    verbose=0
).flatten()

y_pred = (y_prob >= 0.5).astype(int)

# ==========================
# Accuracy
# ==========================
accuracy = accuracy_score(
    y_valid,
    y_pred
)

# ==========================
# Confusion Matrix
# ==========================
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

plt.title("CNN Confusion Matrix")

plt.tight_layout()

plt.savefig(
    "results/cnn_confusion_matrix.png",
    dpi=300
)

plt.close()

# ==========================
# Classification Report
# ==========================
report_text = classification_report(
    y_valid,
    y_pred,
    target_names=[
        "Others",
        "T-Shirt/Top"
    ]
)

with open(
    "results/cnn_classification_report.txt",
    "w"
) as f:
    f.write(report_text)

report = classification_report(
    y_valid,
    y_pred,
    target_names=[
        "Others",
        "T-Shirt/Top"
    ],
    output_dict=True
)

# ==========================
# ROC Curve
# ==========================
fpr, tpr, _ = roc_curve(
    y_valid,
    y_prob
)

auc = roc_auc_score(
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
plt.title("CNN ROC Curve")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "results/cnn_roc_curve.png",
    dpi=300
)

plt.close()

# ==========================
# Metrics CSV
# ==========================
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
        report["T-Shirt/Top"]["precision"],
        report["T-Shirt/Top"]["recall"],
        report["T-Shirt/Top"]["f1-score"],
        auc
    ]

})

metrics.to_csv(
    "results/cnn_metrics.csv",
    index=False
)

# ==========================
# Summary
# ==========================
with open(
    "results/cnn_summary.txt",
    "w"
) as f:

    f.write(f"Accuracy : {accuracy:.4f}\n")
    f.write(f"Precision: {report['T-Shirt/Top']['precision']:.4f}\n")
    f.write(f"Recall   : {report['T-Shirt/Top']['recall']:.4f}\n")
    f.write(f"F1-score : {report['T-Shirt/Top']['f1-score']:.4f}\n")
    f.write(f"AUC      : {auc:.4f}\n")

print("=" * 60)
print("CNN EVALUATION COMPLETE")
print("=" * 60)
print(metrics)
