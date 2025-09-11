import numpy as np


class MetricsCalculator2:
    """
    The class provides to calculate Mean Reciprocal Rank (MRR) and Mean Average Precision (MAP) based on input data, where MRR measures the ranking quality and MAP measures the average precision.
    """

    def __init__(self):
        pass

    @staticmethod
    def mrr(data):
        """
        Compute the MRR of the input data.
        :param data: the data must be a tuple, list 0,1,eg.([1,0,...],5). In each tuple (actual result, ground truth num), ground truth num is the total ground num.
        or list of tuple eg. [([1,0,1,...],5),([1,0,...],6),([0,0,...],5)].
        1 stands for a correct answer, 0 stands for a wrong answer.
        :return: if input data is list, return the recall of this list. if the input data is list of list, return the
        average recall on all list. The second return value is a list of precision for each input.
        """
        
        if isinstance(data[0], list):  # List of tuples
            results = [MetricsCalculator2.mrr(item) for item in data]
            mean_mrr = np.mean([result[0] for result in results])
            return mean_mrr, [result[0] for result in results]
        else:  # Single tuple
            actual, ground_truth_num = data
            reciprocal_ranks = [1/(i+1) for i, value in enumerate(actual) if value == 1]
            return np.mean(reciprocal_ranks), [np.mean(reciprocal_ranks)]

    @staticmethod
    def map(data):
        """
        Compute the MAP of the input data.
        :param data: the data must be a tuple, list 0,1,eg.([1,0,...],5). In each tuple (actual result, ground truth num), ground truth num is the total ground num.
        or list of tuple eg. [([1,0,1,...],5),([1,0,...],6),([0,0,...],5)].
        1 stands for a correct answer, 0 stands for a wrong answer.
        :return: if input data is list, return the recall of this list. if the input data is list of list, return the
        average recall on all list. The second return value is a list of precision for each input.
        """
        def average_precision(actual, ground_truth_num):
            precisions = []
            correct_count = 0
            for i, value in enumerate(actual):
                if value == 1:
                    correct_count += 1
                    precisions.append(correct_count / (i + 1))
            return np.mean(precisions)

        if isinstance(data[0], list):  # List of tuples
            results = [MetricsCalculator2.map(item) for item in data]
            mean_map = np.mean([result[0] for result in results])
            return mean_map, [result[0] for result in results]
        else:  # Single tuple
            actual, ground_truth_num = data
            ap = average_precision(actual, ground_truth_num)
            return ap, [ap]


if __name__ == "__main__":
    # Test cases for MRR
    print(MetricsCalculator2.mrr(([1, 0, 1, 0], 4)))  # Expected: (1.0, [1.0])
    print(MetricsCalculator2.mrr([([1, 0, 1, 0], 4), ([0, 1, 0, 1], 4)]))  # Expected: (0.75, [1.0, 0.5])

    # Test cases for MAP
    print(MetricsCalculator2.map(([1, 0, 1, 0], 4)))  # Expected: (0.41666666666666663, [0.41666666666666663])
    print(MetricsCalculator2.map([([1, 0, 1, 0], 4), ([0, 1, 0, 1], 4)]))  # Expected: (0.3333333333333333, [0.41666666666666663, 0.25])