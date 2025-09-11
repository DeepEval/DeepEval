import numpy as np

class MetricsCalculator:
    def __init__(self):
        self.true_positives = 0
        self.false_positives = 0
        self.false_negatives = 0
        self.true_negatives = 0

    def update(self, predicted_labels, true_labels):
        self.true_positives += np.sum(np.logical_and(predicted_labels, true_labels))
        self.false_positives += np.sum(np.logical_and(predicted_labels, ~true_labels))
        self.false_negatives += np.sum(np.logical_and(~predicted_labels, true_labels))
        self.true_negatives += np.sum(np.logical_and(~predicted_labels, ~true_labels))

    def precision(self, predicted_labels, true_labels):
        precision = self.true_positives / (self.true_positives + self.false_positives)
        return precision

    def recall(self, predicted_labels, true_labels):
        recall = self.true_positives / (self.true_positives + self.false_negatives)
        return recall

    def f1_score(self, predicted_labels, true_labels):
        f1 = 2 * (self.precision(predicted_labels, true_labels) * self.recall(predicted_labels, true_labels)) / (self.precision(predicted_labels, true_labels) + self.recall(predicted_labels, true_labels))
        return f1

    def accuracy(self, predicted_labels, true_labels):
        accuracy = (self.true_positives + self.true_negatives) / (self.true_positives + self.true_negatives + self.false_positives + self.false_negatives)
        return accuracy

# Test case
if __name__ == "__main__":
    mc = MetricsCalculator()
    predicted_labels = np.array([1, 1, 0, 0])
    true_labels = np.array([1, 0, 0, 1])
    mc.update(predicted_labels, true_labels)
    precision = mc.precision(predicted_labels, true_labels)
    recall = mc.recall(predicted_labels, true_labels)
    f1 = mc.f1_score(predicted_labels, true_labels)
    accuracy = mc.accuracy(predicted_labels, true_labels)
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print(f"Accuracy: {accuracy:.2f}")