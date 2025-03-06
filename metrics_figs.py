import sys
import pandas as pd
import matplotlib.pyplot as plt
from lp_metrics.data_loader import merge_data
from lp_metrics.metrics_calculator import calculate_metrics

label_df = pd.read_csv(sys.argv[1])
pred_df = pd.read_csv(sys.argv[2])

ufm_dict, sorted_results = merge_data(label_df, pred_df)

# Generate metrics for different thresholds
thresholds = range(200, 971, 10)  # From 800 to 950 in 10-point steps
precisions = []
automation_rates = []
recalls = []
f_scores = []
accuracies = []

for threshold in thresholds:
    metrics = calculate_metrics(ufm_dict, sorted_results, conf_threshold=threshold)
    precisions.append(metrics['precision'])
    recalls.append(metrics['recall'])
    f_scores.append(metrics['f1'])
    accuracies.append(metrics['accuracy'])
    automation_rates.append(metrics['automation_rate'])

plt.figure(figsize=(10, 6))
plt.plot(thresholds, precisions, label='Precision', marker='o', linestyle='-', markersize=3)
plt.plot(thresholds, recalls, label='Recall', marker='^', linestyle='-.', markersize=3)
plt.plot(thresholds, f_scores, label='F1 Score', marker='D', linestyle=':', markersize=3)
plt.plot(thresholds, accuracies, label='Accuracy', marker='v', linestyle='-', markersize=3)
plt.plot(thresholds, automation_rates, label='Automation Rate', marker='s', linestyle='--', markersize=3)

# Formatting
plt.xlabel('Confidence Threshold', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.title('Classification Metrics vs Confidence Threshold', fontsize=14)
plt.ylim(0.93, 1.005)  # Add some padding above 1
plt.xlim(min(thresholds), max(thresholds))
plt.grid(True, alpha=0.3)
plt.legend()

# Save and show
plt.tight_layout()
plt.savefig('figs/confidence_metrics_plot.png', dpi=300)
plt.show()

# Print metrics for selected thresholds
print("\nKey Threshold Metrics:")
for threshold in [800, 850, 900, 910, 920, 950]:
    if threshold in thresholds:
        idx = thresholds.index(threshold)
        print(f"{threshold}: Precision={precisions[idx]:.3f}, Automation Rate={automation_rates[idx]:.3f}, F-score={f_scores[idx]:.3f}")