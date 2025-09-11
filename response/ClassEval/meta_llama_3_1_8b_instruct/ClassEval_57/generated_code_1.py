import numpy as np


class MetricsCalculator2:
    """
    The class provides to calculate Mean Reciprocal Rank (MRR) and Mean Average Precision (MAP) based on input data, 
    where MRR measures the ranking quality and MAP measures the average precision.
    """

    def __init__(self):
        pass

    @staticmethod
    def mrr(data):
        """
        compute the MRR of the input data. MRR is a widely used evaluation index. It is the mean of reciprocal rank.
        :param data: the data must be a tuple, list 0,1,eg.([1,0,...],5).  In each tuple (actual result,ground truth num), 
        ground truth num is the total ground num. ([1,0,...],5),
        or list of tuple eg. [([1,0,1,...],5),([1,0,...],6),([0,0,...],5)].
        1 stands for a correct answer, 0 stands for a wrong answer.
        :return: if input data is list, return the recall of this list. if the input data is list of list, return the 
        average recall on all list. The second return value is a list of precision for each input.
        >>> MetricsCalculator2.mrr(([1, 0, 1, 0], 4))
        >>> MetricsCalculator2.mrr([([1, 0, 1, 0], 4), ([0, 1, 0, 1], 4)])
        1.0, [1.0]
        0.75, [1.0, 0.5]
        """
        if isinstance(data, list):
            # If data is a list, return the recall of this list
            recalls = []
            precisions = []
            for actual_result, ground_truth_num in data:
                correct_results = actual_result[:ground_truth_num]
                total_results = actual_result
                reciprocal_rank = 1 / correct_results.index(1) if 1 in correct_results else 0
                recall = ground_truth_num / len(actual_result)
                precision = correct_results.count(1) / len(correct_results)
                recalls.append(reciprocal_rank)
                precisions.append(precision)
            return np.mean(recalls), precisions
        elif isinstance(data, tuple):
            # If data is a tuple, return the MRR and precision
            actual_result, ground_truth_num = data
            correct_results = actual_result[:ground_truth_num]
            total_results = actual_result
            reciprocal_rank = 1 / correct_results.index(1) if 1 in correct_results else 0
            recall = ground_truth_num / len(actual_result)
            precision = correct_results.count(1) / len(correct_results)
            return reciprocal_rank, precision
        else:
            raise ValueError("Input data must be a tuple, list or a list of tuples")

    @staticmethod
    def map(data):
        """
        compute the MAP of the input data. MAP is a widely used evaluation index. It is the mean of AP (average precision).
        :param data: the data must be a tuple, list 0,1,eg.([1,0,...],5).  In each tuple (actual result,ground truth num), 
        ground truth num is the total ground num. ([1,0,...],5),
        or list of tuple eg. [([1,0,1,...],5),([1,0,...],6),([0,0,...],5)].
        1 stands for a correct answer, 0 stands for a wrong answer.
        :return: if input data is list, return the recall of this list. if the input data is list of list, return the 
        average recall on all list. The second return value is a list of precision for each input.
        >>> MetricsCalculator2.map(([1, 0, 1, 0], 4))
        >>> MetricsCalculator2.map([([1, 0, 1, 0], 4), ([0, 1, 0, 1], 4)])
        0.41666666666666663, [0.41666666666666663]
        0.3333333333333333, [0.41666666666666663, 0.25]
        """
        if isinstance(data, list):
            # If data is a list, return the recall of this list
            recalls = []
            precisions = []
            for actual_result, ground_truth_num in data:
                correct_results = actual_result[:ground_truth_num]
                total_results = actual_result
                precision_values = []
                for i in range(1, len(total_results) + 1):
                    precision = correct_results[:i].count(1) / i
                    precision_values.append(precision)
                recall = ground_truth_num / len(actual_result)
                reciprocal_rank = 1 / correct_results.index(1) if 1 in correct_results else 0
                precision = np.mean(precision_values)
                recalls.append(reciprocal_rank)
                precisions.append(precision)
            return np.mean(recalls), precisions
        elif isinstance(data, tuple):
            # If data is a tuple, return the MRR and precision
            actual_result, ground_truth_num = data
            correct_results = actual_result[:ground_truth_num]
            total_results = actual_result
            precision_values = []
            for i in range(1, len(total_results) + 1):
                precision = correct_results[:i].count(1) / i
                precision_values.append(precision)
            recall = ground_truth_num / len(actual_result)
            reciprocal_rank = 1 / correct_results.index(1) if 1 in correct_results else 0
            precision = np.mean(precision_values)
            return recall, precision
        else:
            raise ValueError("Input data must be a tuple, list or a list of tuples")

if __name__ == "__main__":
    # Test case
    calculator = MetricsCalculator2()
    print(calculator.mrr(([1, 0, 1, 0], 4)))
    print(calculator.mrr([([1, 0, 1, 0], 4), ([0, 1, 0, 1], 4)]))
    print(calculator.map(([1, 0, 1, 0], 4)))
    print(calculator.map([([1, 0, 1, 0], 4), ([0, 1, 0, 1], 4)]))