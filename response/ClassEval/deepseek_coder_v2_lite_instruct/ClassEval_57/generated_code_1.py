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
        compute the MRR of the input data. MRR is a widely used evaluation index. It is the mean of reciprocal rank.
        :param data: the data must be a tuple, list 0,1,eg.([1,0,...],5).  In each tuple (actual result,ground truth num),ground truth num is the total ground num.
         ([1,0,...],5),
        or list of tuple eg. [([1,0,1,...],5),([1,0,...],6),([0,0,...],5)].
        1 stands for a correct answer, 0 stands for a wrong answer.
        :return: if input data is list, return the recall of this list. if the input data is list of list, return the
        average recall on all list. The second return value is a list of precision for each input.
        >>> MetricsCalculator2.mrr(([1, 0, 1, 0], 4))
        >>> MetricsCalculator2.mrr([([1, 0, 1, 0], 4), ([0, 1, 0, 1], 4)])
        1.0, [1.0]
        0.75, [1.0, 0.5]
        """
        mrr_values = []
        precision_values = []

        for item in data:
            result, total_ground_truth = item
            rank = 0
            for i, val in enumerate(result):
                if val == 1:
                    rank = i + 1
                    break
            reciprocal_rank = 1 / rank if rank > 0 else 0
            mrr_values.append(reciprocal_rank)
        
        if len(mrr_values) == 1:
            return mrr_values[0], precision_values
        else:
            return np.mean(mrr_values), mrr_values

    @staticmethod
    def map(data):
        """
        compute the MAP of the input data. MAP is a widely used evaluation index. It is the mean of AP (average precision).
        :param data: the data must be a tuple, list 0,1,eg.([1,0,...],5).  In each tuple (actual result,ground truth num),ground truth num is the total ground num.
         ([1,0,...],5),
        or list of tuple eg. [([1,0,1,...],5),([1,0,...],6),([0,0,...],5)].
        1 stands for a correct answer, 0 stands for a wrong answer.
        :return: if input data is list, return the recall of this list. if the input data is list of list, return the
        average recall on all list. The second return value is a list of precision for each input.
        >>> MetricsCalculator2.map(([1, 0, 1, 0], 4))
        >>> MetricsCalculator2.map([([1, 0, 1, 0], 4), ([0, 1, 0, 1], 4)])
        0.41666666666666663, [0.41666666666666663]
        0.3333333333333333, [0.41666666666666663, 0.25]
        """
        map_values = []
        precision_per_query = []

        for item in data:
            result, total_ground_truth = item
            precision_at_k = []
            cum_precision = 0
            num_relevant = 0
            for i, val in enumerate(result):
                if val == 1:
                    num_relevant += 1
                    precision = num_relevant / (i + 1)
                    precision_at_k.append(precision)
                    cum_precision += precision
            if num_relevant > 0:
                average_precision = cum_precision / num_relevant
            else:
                average_precision = 0
            map_values.append(average_precision)
            precision_per_query.append(precision_at_k)
        
        if len(map_values) == 1:
            return map_values[0], precision_per_query[0]
        else:
            return np.mean(map_values), precision_per_query

if __name__ == "__main__":
    # Test cases
    print(MetricsCalculator2.mrr(([1, 0, 1, 0], 4)))  # Expected output: 1.0, [1.0]
    print(MetricsCalculator2.mrr([([1, 0, 1, 0], 4), ([0, 1, 0, 1], 4)]))  # Expected output: 0.75, [1.0, 0.5]
    print(MetricsCalculator2.map(([1, 0, 1, 0], 4)))  # Expected output: 0.41666666666666663, [0.41666666666666663]
    print(MetricsCalculator2.map([([1, 0, 1, 0], 4), ([0, 1, 0, 1], 4)]))  # Expected output: 0.3333333333333333, [0.41666666666666663, 0.25]