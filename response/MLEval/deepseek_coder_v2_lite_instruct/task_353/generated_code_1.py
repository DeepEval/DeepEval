import numpy as np

def method():
    # Example predictions and labels (replace these with actual model outputs)
    predictions = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    confidences = np.array([0.9, 0.8, 0.7, 0.6, 0.55, 0.54, 0.53, 0.52, 0.51, 0.50, 0.49, 0.48, 0.47, 0.46, 0.45, 0.44, 0.43, 0.42, 0.41, 0.40])
    labels = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

    # Ensure the number of correct predictions is at least 95%
    correct_indices = predictions == labels
    incorrect_indices = predictions != labels

    # Sort by confidence for correct predictions
    correct_confidences = confidences[correct_indices]
    sorted_correct_confidences = correct_confidences[np.argsort(correct_confidences)]

    # Sort by confidence for incorrect predictions
    incorrect_confidences = confidences[incorrect_indices]
    sorted_incorrect_confidences = incorrect_confidences[np.argsort(incorrect_confidences)]

    # Ensure the number of correct predictions is at least 95%
    num_samples = len(predictions)
    num_correct_needed = int(num_samples * 0.95)
    num_correct_now = np.sum(correct_indices)

    if num_correct_now < num_correct_needed:
        # Adjust predictions for incorrect samples to maintain 95% accuracy
        additional_correct_needed = num_correct_needed - num_correct_now
        adjusted_confidences = np.concatenate([sorted_correct_confidences, sorted_incorrect_confidences[:additional_correct_needed]])
        adjusted_predictions = np.concatenate([np.ones(num_correct_now), np.zeros(additional_correct_needed), np.ones(num_samples - num_correct_now - additional_correct_needed)])
    else:
        adjusted_confidences = confidences
        adjusted_predictions = predictions

    # Find the lowest confidence for a correct output and the highest confidence for a wrong output
    lowest_correct_confidence = np.min(sorted_correct_confidences)
    highest_wrong_confidence = np.max(sorted_incorrect_confidences)

    # Output the results
    output = {
        "lowest_correct_confidence": lowest_correct_confidence,
        "highest_wrong_confidence": highest_wrong_confidence
    }

    return output

# Call the method for validation
print(method())