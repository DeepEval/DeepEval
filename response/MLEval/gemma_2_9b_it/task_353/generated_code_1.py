import numpy as np

def method(predictions, thresholds):
    # Assume 'predictions' is an array of probabilities (0-1)
    # 'thresholds' is a list of confidence thresholds to try

    for threshold in thresholds:
        correct_count = np.sum((predictions > threshold) == (target_labels == 1)) # Assuming binary classification
        accuracy = correct_count / len(target_labels)
        print(f"Threshold: {threshold}, Accuracy: {accuracy}")

    return  # You can return the best threshold and its accuracy if needed

# Example usage
predictions = np.random.rand(100)  # Replace with your actual predictions
target_labels = np.random.randint(0, 2, size=100)  # Replace with your actual target labels
method(predictions, thresholds=[0.5, 0.6, 0.7, 0.8])