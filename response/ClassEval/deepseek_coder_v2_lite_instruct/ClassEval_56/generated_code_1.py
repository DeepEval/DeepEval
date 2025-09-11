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

    def precision(self, predicted_labels, true_labels):
        """
        Calculate precision
        :param predicted_labels: list, predicted results
        :param true_labels: list, true labels
        :return: float
        """
        self.update(predicted_labels, true_labels)
        if (self.true_positives + self.false_positives) == 0:
            return 0
        return self.true_positives / (self.true_positives + self.false_positives)

    def recall(self, predicted_labels, true_labels):
        """
        Calculate recall
        :param predicted_labels: list, predicted results
        :param true_labels: list, true labels
        :return: float
        """
        self.update(predicted_labels, true_labels)
        if (self.true_positives + self.false_negatives) == 0:
            return 0
        return self.true_positives / (self.true_positives + self.false_negatives)

    def f1_score(self, predicted_labels, true_labels):
        """
        Calculate f1 score, which is the harmonic mean of precision and recall
        :param predicted_labels: list, predicted results
        :param true_labels: list, true labels
        :return: float
        """
        prec = self.precision(predicted_labels, true_labels)
        rec = self.recall(predicted_labels, true_labels)
        if (prec + rec) == 0:
            return 0
        return 2 * (prec * rec) / (prec + rec)

    def accuracy(self, predicted_labels, true_labels):
        """
        Calculate accuracy
        :param predicted_labels: list, predicted results
        :param true_labels: list, true labels
        :return: float
        """
        self.update(predicted_labels, true_labels)
        total = self.true_positives + self.false_positives + self.false_negatives + self.true_negatives
        if total == 0:
            return 0
        return (self.true_positives + self.true_negatives) / total


if __name__ == "__main__":
    mc = MetricsCalculator()
    mc.update([1, 1, 0, 0], [1, 0, 0, 1])
    print((mc.true_positives, mc.false_positives, mc.false_negatives, mc.true_negatives))
    print(mc.precision([1, 1, 0, 0], [1, 0, 0, 1]))
    print(mc.recall([1, 1, 0, 0], [1, 0, 0, 1]))
    print(mc.f1_score([1, 1, 0, 0], [1, 0, 0, 1]))
    print(mc.accuracy([1, 1, 0, 0], [1, 0, 0, 1]))