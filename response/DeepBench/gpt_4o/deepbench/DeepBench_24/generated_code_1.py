import numpy as np
from sklearn.utils import check_array
from collections import Counter

def compute_confident_joint(noisy_labels, true_labels, num_classes=None):
    noisy_labels = check_array(noisy_labels, ensure_2d=False, dtype=int)
    true_labels = check_array(true_labels, ensure_2d=False, dtype=int)

    if num_classes is None:
        num_classes = max(max(noisy_labels), max(true_labels)) + 1
    
    confident_joint = np.zeros((num_classes, num_classes), dtype=int)

    label_pairs = zip(true_labels, noisy_labels)
    for true, noisy in label_pairs:
        confident_joint[true][noisy] += 1

    return confident_joint

if __name__ == "__main__":
    true_labels = np.array([0, 1, 2, 1, 0, 2, 1])
    noisy_labels = np.array([0, 1, 1, 1, 0, 2, 2])
    
    joint_counts = compute_confident_joint(noisy_labels, true_labels)

    print("Confident Joint Matrix:")
    print(joint_counts)