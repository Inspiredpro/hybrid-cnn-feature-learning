
import pandas as pd
import os

os.makedirs("results", exist_ok=True)

# =====================================================
# Load Metrics
# =====================================================
cnn = pd.read_csv("results/cnn_metrics.csv")
hybrid = pd.read_csv("results/hybrid_metrics.csv")

cnn = cnn.set_index("Metric")
hybrid = hybrid.set_index("Metric")

# =====================================================
# Create Comparison Table
# =====================================================
comparison = pd.DataFrame({
    "Metric": cnn.index,
    "Value_CNN": cnn["Value"].values,
    "Value_Hybrid": hybrid["Value"].values
})

comparison["Difference"] = (
    comparison["Value_Hybrid"] -
    comparison["Value_CNN"]
)

comparison.to_csv(
    "results/model_comparison.csv",
    index=False
)

print("=" * 60)
print("MODEL COMPARISON")
print("=" * 60)
print(comparison.round(6))

print("\nSaved to:")
print("results/model_comparison.csv")
