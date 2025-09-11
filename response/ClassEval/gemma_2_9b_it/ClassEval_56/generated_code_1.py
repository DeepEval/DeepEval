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
        >>> mc = MetricsCalculator()
        >>> mc.update([1, 1, 0, 0], [1, 0, 0, 1])
        (self.true_positives, self.false_positives, self.false_negatives, self.true_negatives) = (1, 1, 1, 1)
        """
        for i in range(len(predicted_labels)):
            if predicted_labels[i] == true_labels[i] == 1:
                self.true_positives += 1
            elif predicted_labels[i] == 1 and true_labels[i] == 0:
                self.false_positives += 1
            elif predicted_labels[i] == 0 and true_labels[i] == 1:
                self.false_negatives += 1
            else:
                self.true_negatives += 1


    def precision(self, predicted_labels, true_labels):
        """
        Calculate precision
        :param predicted_labels: list, predicted results
        :param true_labels: list, true labels
        :return: float
        >>> mc = MetricsCalculator()
        >>> mc.precision([1, 1, 0, 0], [1, 0, 0, 1])
        0.5
        """
        return self.true_positives / (self.true_positives + self.false_positives) if (self.true_positives + self.false_positives) > 0 else 0


    def recall(self, predicted_labels, true_labels):
        """
        Calculate recall
        :param predicted_labels: list, predicted results
        :param true_labels: list, true labels
        :return: float
        >>> mc = MetricsCalculator()
        >>> mc.recall([1, 1, 0, 0], [1, 0, 0, 1])
        0.5
        """
        return self.true_positives / (self.true_positives + self.false_negatives) if (self.true_positives + self.false_negatives) > 0 else 0


    def f1_score(self, predicted_labels, true_labels):
        """
        Calculate f1 score, which is the harmonic mean of precision and recall
        :param predicted_labels: list, predicted results
        :param true_labels: list, true labels
        :return: float
        >>> mc = MetricsCalculator()
        >>> mc.f1_score([1, 1, 0, 0], [1, 0, 0, 1])
        0.5
        """
        return 2 * (self.precision(predicted_labels, true_labels) * self.recall(predicted_labels, true_labels)) / (self.precision(predicted_labels, true_labels) + self.recall(predicted_labels, true_labels)) if (self.precision(predicted_labels, true_labels) + self.recall(predicted_labels, true_labels)) > 0 else 0


    def accuracy(self, predicted_labels, true_labels):
        """
        Calculate accuracy
        :param predicted_labels: list, predicted results
        :param true_labels: list, true labels
        :return: float
        >>> mc = MetricsCalculator()
        >>>mc.accuracy([1, 1, 0, 0], [1, 0, 0, 1])
        0.5
        """
        return (self.true_positives + self.true_negatives) / (len(predicted_labels)) 

if __name__ == "__main__":
    instance = MetricsCalculator()
    # Test case for update
    instance.update([1, 1, 0, 0], [1, 0, 0, 1])
    print(f"True Positives: {instance.true_positives}, False Positives: {instance.false_positives}, False Negatives: {instance.false_negatives}, True Negatives: {instance.true_negatives}")

    # Test case for precision
    output_precision = instance.precision([1, 1, 0, 0], [1, 0, 0, 1])
    print(f"Precision: {output_precision}")

    # Test case for recall
    output_recall = instance.recall([1, 1, 0, 0], [1, 0, 0, 1])
    print(f"Recall: {output_recall}")

    # Test case for f1_score
    output_f1 = instance.f1_score([1, 1, 0, 0], [1, 0, 0, 1])
    print(f"F1 Score: {output_f1}")

    # Test case for accuracy
    output_accuracy = instance.accuracy([1, 1, 0, 0], [1, 0, 0, 1])
    print(f"Accuracy: {output_accuracy}")