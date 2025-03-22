import numpy as np
from sklearn.metrics import accuracy_score

def method(predictions, true_labels, confidence_scores):
    """
    Adjust the confidence threshold to maintain 95% correctness.
    
    Parameters:
    - predictions: numpy array of predicted labels
    - true_labels: numpy array of true labels
    - confidence_scores: numpy array of confidence scores associated with the predictions
    
    Returns:
    - lowest_confidence_correct: lowest confidence score for correct predictions
    - highest_confidence_wrong: highest confidence score for incorrect predictions
    - adjusted_predictions: predictions after adjusting the threshold
    - adjusted_accuracy: accuracy after adjusting the threshold
    """
    
    # Calculate initial accuracy
    initial_accuracy = accuracy_score(true_labels, predictions)
    print(f"Initial Accuracy: {initial_accuracy * 100:.2f}%")
    
    # Get indices of correct and incorrect predictions
    correct_indices = np.where(predictions == true_labels)[0]
    incorrect_indices = np.where(predictions != true_labels)[0]
    
    # Sort correct predictions by their confidence scores (ascending)
    correct_confidences_sorted = np.sort(confidence_scores[correct_indices])
    
    # Sort incorrect predictions by their confidence scores (descending)
    incorrect_confidences_sorted = np.sort(confidence_scores[incorrect_indices])[::-1]
    
    # Calculate the number of correct predictions we need to maintain (95%)
    num_correct_to_maintain = int(len(correct_indices) * 0.95)
    
    # Determine the confidence threshold for correct predictions
    lowest_confidence_correct = correct_confidences_sorted[-num_correct_to_maintain]
    
    # Determine the highest confidence for wrong predictions
    highest_confidence_wrong = incorrect_confidences_sorted[0]
    
    # Adjust predictions based on the new threshold
    adjusted_predictions = np.where(confidence_scores >= lowest_confidence_correct, predictions, -1) # Assuming -1 as a default for uncertain predictions
    
    # Calculate adjusted accuracy
    adjusted_correct_indices = np.where(adjusted_predictions == true_labels)[0]
    adjusted_accuracy = len(adjusted_correct_indices) / len(true_labels)
    
    return lowest_confidence_correct, highest_confidence_wrong, adjusted_predictions, adjusted_accuracy

# Example usage
predictions = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0])
true_labels = np.array([1, 0, 0, 1, 0, 1, 1, 0, 1, 0])
confidence_scores = np.array([0.9, 0.8, 0.4, 0.95, 0.7, 0.6, 0.5, 0.9, 0.85, 0.6])

lowest_confidence_correct, highest_confidence_wrong, adjusted_predictions, adjusted_accuracy = method(predictions, true_labels, confidence_scores)

print(f"Lowest Confidence for Correct Output: {lowest_confidence_correct}")
print(f"Highest Confidence for Wrong Output: {highest_confidence_wrong}")
print(f"Adjusted Predictions: {adjusted_predictions}")
print(f"Adjusted Accuracy: {adjusted_accuracy * 100:.2f}%")