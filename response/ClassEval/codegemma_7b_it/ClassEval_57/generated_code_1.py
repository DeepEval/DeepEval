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
        if isinstance(data, list):
            if isinstance(data[0], list):
                return np.mean([self.mrr(d) for d in data]), [self.mrr(d)[1] for d in data]
            else:
                return self.mrr(data)
        else:
            sum_reciprocal_rank = 0
            for i, (actual_result, ground_truth_num) in enumerate(data):
                if actual_result == 1:
                    sum_reciprocal_rank += 1 / (i + 1)
            return sum_reciprocal_rank / ground_truth_num, [1 / (i + 1) for i, (actual_result, ground_truth_num) in enumerate(data) if actual_result == 1]

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
        if isinstance(data, list):
            if isinstance(data[0], list):
                return np.mean([self.map(d) for d in data]), [self.map(d)[1] for d in data]
            else:
                return self.map(data)
        else:
            sum_ap = 0
            for i, (actual_result, ground_truth_num) in enumerate(data):
                if actual_result == 1:
                    sum_ap += (i + 1) / (ground_truth_num * (data[:i + 1].count(1) + 1))
            return sum_ap / ground_truth_num, [(i + 1) / (ground_truth_num * (data[:i + 1].count(1) + 1)) for i, (actual_result, ground_truth_num) in enumerate(data) if actual_result == 1]

if __name__ == "__main__":
    instance = MetricsCalculator2()
    # Test case
    data = ([1, 0, 1, 0], 4)
    output = instance.mrr(data)
    print(output)
    
    data = [([1, 0, 1, 0], 4), ([0, 1, 0, 1], 4)]
    output = instance.mrr(data)
    print(output)
    
    data = ([1, 0, 1, 0], 4)
    output = instance.map(data)
    print(output)
    
    data = [([1, 0, 1, 0], 4), ([0, 1, 0, 1], 4)]
    output = instance.map(data)
    print(output)