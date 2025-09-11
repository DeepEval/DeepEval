from typing import List

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


    def update(self, predicted_labels: List[int], true_labels: List[int]) -> None:
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
            else:
                self.true_negatives += 1


    def precision(self, predicted_labels: List[int], true_labels: List[int]) -> float:
        """
        Calculate precision
        :param predicted_labels: list, predicted results
        :param true_labels: list, true labels
        :return: float
        """
        self.update(predicted_labels, true_labels)
        if self.false_positives + self.true_positives == 0:
            return 0
        return self.true_positives / (self.false_positives + self.true_positives)


    def recall(self, predicted_labels: List[int], true_labels: List[int]) -> float:
        """
        Calculate recall
        :param predicted_labels: list, true labels
        :param true_labels: list, true labels
        :return: float
        """
        self.update(predicted_labels, true_labels)
        if self.false_negatives + self.true_positives == 0:
            return 0
        return self.true_positives / (self.false_negatives + self.true_positives)


    def f1_score(self, predicted_labels: List[int], true_labels: List[int]) -> float:
        """
        Calculate f1 score, which is the harmonic mean of precision and recall
        :param predicted_labels: list, predicted results
        :param true_labels: list, true labels
        :return: float
        """
        precision = self.precision(predicted_labels, true_labels)
        recall = self.recall(predicted_labels, true_labels)
        if precision + recall == 0:
            return 0
        return 2 * precision * recall / (precision + recall)


    def accuracy(self, predicted_labels: List[int], true_labels: List[int]) -> float:
        """
        Calculate accuracy
        :param predicted_labels: list, predicted results
        :param true_labels: list, true labels
        :return: float
        """
        self.update(predicted_labels, true_labels)
        if self.true_positives + self.false_positives + self.false_negatives + self.true_negatives == 0:
            return 0
        return (self.true_positives + self.true_negatives) / (self.true_positives + self.false_positives + self.false_negatives + self.true_negatives)


if __name__ == "__main__":
    instance = MetricsCalculator()
    
    # Test case for update method
    predicted_labels = [1, 1, 0, 0]
    true_labels = [1, 0, 0, 1]
    instance.update(predicted_labels, true_labels)
    print("Update method test case:")
    print(f"True positives: {instance.true_positives}")
    print(f"False positives: {instance.false_positives}")
    print(f"False negatives: {instance.false_negatives}")
    print(f"True negatives: {instance.true_negatives}")

    # Test case for precision method
    predicted_labels = [1, 1, 0, 0]
    true_labels = [1, 0, 0, 1]
    print("\nPrecision method test case:")
    print(f"Precision: {instance.precision(predicted_labels, true_labels)}")

    # Test case for recall method
    predicted_labels = [1, 1, 0, 0]
    true_labels = [1, 0, 0, 1]
    print("\nRecall method test case:")
    print(f"Recall: {instance.recall(predicted_labels, true_labels)}")

    # Test case for f1_score method
    predicted_labels = [1, 1, 0, 0]
    true_labels = [1, 0, 0, 1]
    print("\nf1_score method test case:")
    print(f"f1_score: {instance.f1_score(predicted_labels, true_labels)}")

    # Test case for accuracy method
    predicted_labels = [1, 1, 0, 0]
    true_labels = [1, 0, 0, 1]
    print("\nAccuracy method test case:")
    print(f"Accuracy: {instance.accuracy(predicted_labels, true_labels)}")