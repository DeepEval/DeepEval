import numpy as np
from sklearn.metrics import confusion_matrix

def compute_confident_joint(labels_true, labels_pred, confidence_scores):
    # Compute the binary confidence scores (above 0.5 is confident, below 0.5 is not)
    confidence_binary = np.where(confidence_scores > 0.5, 1, 0)

    # Compute the confusion matrix
    cm = confusion_matrix(labels_true, confidence_binary)

    # Compute the confident true positives and true negatives
    confident_tp = cm[0, 0]  # Correctly predicted confident samples
    confident_tn = cm[1, 1]  # Correctly rejected confident samples

    # Compute the confident false positives and false negatives
    confident_fp = cm[1, 0]  # Incorrectly predicted confident samples
    confident_fn = cm[0, 1]  # Incorrectly rejected confident samples

    # Compute the confident counts
    confident_counts = np.array([confident_tp, confident_tn, confident_fp, confident_fn])

    return confident_counts

if __name__ == "__main__":
    # Generate some random input values
    np.random.seed(0)
    labels_true = np.random.randint(0, 2, 100)
    labels_pred = np.random.randint(0, 2, 100)
    confidence_scores = np.random.rand(100)

    # Call the function
    confident_counts = compute_confident_joint(labels_true, labels_pred, confidence_scores)

    # Print the results
    print("Confident True Positives:", confident_counts[0])
    print("Confident True Negatives:", confident_counts[1])
    print("Confident False Positives:", confident_counts[2])
    print("Confident False Negatives:", confident_counts[3])