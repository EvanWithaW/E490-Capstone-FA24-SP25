import sys
import pandas as pd
import matplotlib.pyplot as plt
from lp_metrics.data_loader import merge_data
from lp_metrics.metrics_calculator import calculate_metrics

# Load data
label_df = pd.read_csv(sys.argv[1])
pred_df = pd.read_csv(sys.argv[2])

# Process data
ufm_dict, sorted_results = merge_data(label_df, pred_df)

# Generate metrics for different thresholds
thresholds = range(800, 1001, 10)  # From 800 to 1000 in 10-point steps
precisions = []
f1_scores = []

for threshold in thresholds:
    metrics = calculate_metrics(ufm_dict, sorted_results, conf_threshold=threshold)
    precisions.append(metrics['precision'])
    f1_scores.append(metrics['f1'])

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(thresholds, precisions, label='Precision', marker='o', linestyle='-')
plt.plot(thresholds, f1_scores, label='F1 Score', marker='s', linestyle='--')

# Formatting
plt.xlabel('Confidence Threshold', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.title('Precision and F1 Score vs Confidence Threshold', fontsize=14)
plt.ylim(0, 1.05)  # Add some padding above 1
plt.xlim(min(thresholds), max(thresholds))
plt.grid(True, alpha=0.3)
plt.legend()

# Save and show
plt.tight_layout()
plt.savefig('confidence_metrics_plot.png', dpi=300)
plt.show()

# Print metrics for selected thresholds
print("\nKey Threshold Metrics:")
for threshold in [800, 850, 900, 910, 920, 950, 1000]:
    if threshold in thresholds:
        idx = thresholds.index(threshold)
        print(f"{threshold}: Precision={precisions[idx]:.3f}, F1={f1_scores[idx]:.3f}")