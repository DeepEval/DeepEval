import numpy as np

class MetricsCalculator:
    """
    The class calculates precision, recall, F1 score, and accuracy based on predicted and true labels.
    """

    def __init__(self):
        """
        Initialize the number of all four samples to 0
        """
        self.true_positives = 0
        self.false_positives = 0
        self.false_negatives = 0
        self.true_negatives = 0

    def update(self, predicted_labels, true_labels):
        """
        Update the number of all four samples(true_positives, false_positives, false_negatives, true_negatives)
        :param predicted_labels: list, predicted results
        :param true_labels: list, true labels
        :return: None, change the number of corresponding samples
        """
        for pred, true in zip(predicted_labels, true_labels):
            if pred == 1 and true == 1:
                self.true_positives += 1
            elif pred == 0 and true == 0:
                self.true_negatives += 1
            elif pred == 0 and true == 1:
                self.false_positives += 1
            elif pred == 1 and true == 0:
                self.false_negatives += 1

    def precision(self, predicted_labels, true_labels):
        """
        Calculate precision
        :param predicted_labels: list, predicted results
        :param true_labels: list, true labels
        :return: float
        """
        TP = self.true_positives
        FP = self.false_positives
        return TP / (TP + FP) if (TP + FP) > 0 else 0

    def recall(self, predicted_labels, true_labels):
        """
        Calculate recall
        :param predicted_labels: list, predicted results
        :param true_labels: list, true labels
        :return: float
        """
        TP = self.true_positives
        FN = self.false_negatives
        return TP / (TP + FN) if (TP + FN) > 0 else 0

    def f1_score(self, predicted_labels, true_labels):
        """
        Calculate f1 score, which is the harmonic mean of precision and recall
        :param predicted_labels: list, predicted results
        :param true_labels: list, true labels
        :return: float
        """
        P = self.precision(predicted_labels, true_labels)
        R = self.recall(predicted_labels, true_labels)
        return 2 * (P * R) / (P + R) if (P + R) > 0 else 0

    def accuracy(self, predicted_labels, true_labels):
        """
        Calculate accuracy
        :param predicted_labels: list, predicted results
        :param true_labels: list, true labels
        :return: float
        """
        TP = self.true_positives
        TN = self.true_negatives
        FP = self.false_positives
        FN = self.false_negatives
        return (TP + TN) / (TP + FP + FN + TN) if (TP + FP + FN + TN) > 0 else 0

# Test cases
if __name__ == "__main__":
    mc = MetricsCalculator()
    predicted_labels = [1, 0, 1, 1]
    true_labels = [1, 0, 1, 1]
    mc.update(predicted_labels, true_labels)
    print("Precision:", mc.precision(predicted_labels, true_labels))
    print("Recall:", mc.recall(predicted_labels, true_labels))
    print("F1 Score:", mc.f1_score(predicted_labels, true_labels))
    print("Accuracy:", mc.accuracy(predicted_labels, true_labels))