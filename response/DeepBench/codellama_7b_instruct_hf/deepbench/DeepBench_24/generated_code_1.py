import numpy as np

def compute_confident_joint(true_labels, noisy_labels):
    # Compute the number of true positives
    tp = np.sum(true_labels * noisy_labels)

    # Compute the number of false positives
    fp = np.sum(true_labels * (1 - noisy_labels))

    # Compute the number of true negatives
    tn = np.sum((1 - true_labels) * (1 - noisy_labels))

    # Compute the number of false negatives
    fn = np.sum((1 - true_labels) * noisy_labels)

    # Compute the confident counts
    confident_counts = np.array([[tp, tn], [fp, fn]])

    return confident_counts

if __name__ == "__main__":
    # Example usage
    true_labels = np.array([0, 1, 1, 0])
    noisy_labels = np.array([0, 1, 0, 1])
    confident_counts = compute_confident_joint(true_labels, noisy_labels)
    print(confident_counts)