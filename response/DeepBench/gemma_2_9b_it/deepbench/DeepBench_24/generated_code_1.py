import numpy as np

def compute_confident_joint(true_labels, observed_labels, confidences):
  confident_true_counts = np.sum((true_labels == observed_labels) & (confidences > 0.95))
  confident_false_counts = np.sum((true_labels != observed_labels) & (confidences > 0.95))
  return confident_true_counts, confident_false_counts

if __name__ == "__main__":
  true_labels = np.array([0, 1, 0, 1, 0])
  observed_labels = np.array([0, 1, 0, 0, 1])
  confidences = np.array([0.8, 0.98, 0.6, 0.7, 0.99])
  true_count, false_count = compute_confident_joint(true_labels, observed_labels, confidences)
  print(f"Confident True Counts: {true_count}")
  print(f"Confident False Counts: {false_count}")