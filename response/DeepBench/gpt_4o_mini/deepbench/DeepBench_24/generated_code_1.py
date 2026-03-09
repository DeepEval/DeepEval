import numpy as np

def compute_confident_joint(true_labels, noisy_labels, confidence_threshold=0.7):
    confident_counts = {}
    
    for true_label in np.unique(true_labels):
        mask = true_labels == true_label
        observed_noisy_labels = noisy_labels[mask]
        
        if len(observed_noisy_labels) > 0:
            confident_count = np.sum(observed_noisy_labels >= confidence_threshold)
            confident_counts[true_label] = confident_count
            
    return confident_counts

if __name__ == "__main__":
    true_labels = np.array([0, 1, 0, 1, 2, 2, 1, 0])
    noisy_labels = np.array([0.9, 0.8, 0.6, 0.95, 0.5, 0.85, 0.7, 0.4])
    
    result = compute_confident_joint(true_labels, noisy_labels, confidence_threshold=0.7)
    print(result)