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
            elif pred == 1 and true == 0:
                self.false_positives += 1
            elif pred == 0 and true == 1:
                self.false_negatives += 1
            elif pred == 0 and true == 0:
                self.true_negatives += 1

    def precision(self):
        """
        Calculate precision
        :return: float
        """
        if self.true_positives + self.false_positives == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_positives)

    def recall(self):
        """
        Calculate recall
        :return: float
        """
        if self.true_positives + self.false_negatives == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_negatives)

    def f1_score(self):
        """
        Calculate f1 score, which is the harmonic mean of precision and recall
        :return: float
        """
        p = self.precision()
        r = self.recall()
        if p + r == 0:
            return 0.0
        return 2 * (p * r) / (p + r)

    def accuracy(self):
        """
        Calculate accuracy
        :return: float
        """
        total = self.true_positives + self.false_positives + self.false_negatives + self.true_negatives
        if total == 0:
            return 0.0
        return (self.true_positives + self.true_negatives) / total


if __name__ == "__main__":
    mc = MetricsCalculator()

    # Test case for update method
    mc.update([1, 1, 0, 0], [1, 0, 0, 1])
    print(f"TP: {mc.true_positives}, FP: {mc.false_positives}, FN: {mc.false_negatives}, TN: {mc.true_negatives}")
    # Expected output: TP: 1, FP: 1, FN: 1, TN: 1

    # Test case for precision
    print(f"Precision: {mc.precision()}")  # Expected output: 0.5

    # Test case for recall
    print(f"Recall: {mc.recall()}")  # Expected output: 0.5

    # Test case for f1_score
    print(f"F1 Score: {mc.f1_score()}")  # Expected output: 0.5

    # Test case for accuracy
    print(f"Accuracy: {mc.accuracy()}")  # Expected output: 0.5