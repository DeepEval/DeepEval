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
        if isinstance(data, tuple):
            data = [data]
        mrr_sum = 0
        precision_list = []
        for actual_result, ground_truth_num in data:
            # Calculate precision at each rank
            precision_at_k = []
            correct_count = 0
            for i, result in enumerate(actual_result):
                if result == 1:
                    correct_count += 1
                    precision_at_k.append(correct_count / (i + 1))
            
            # Calculate MRR for this example
            if correct_count > 0:
                mrr_sum += 1 / (np.argmax(precision_at_k) + 1)
            precision_list.append(precision_at_k[0] if precision_at_k else 0)
        
        if len(data) > 1:
            return mrr_sum / len(data), precision_list
        else:
            return mrr_sum, precision_list

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
        if isinstance(data, tuple):
            data = [data]
        map_sum = 0
        for actual_result, ground_truth_num in data:
            # Calculate average precision
            ap = 0
            correct_count = 0
            for i, result in enumerate(actual_result):
                if result == 1:
                    correct_count += 1
                    ap += correct_count / (i + 1)
            ap /= ground_truth_num
            map_sum += ap
        
        if len(data) > 1:
            return map_sum / len(data)
        else:
            return map_sum

if __name__ == "__main__":
    instance = MetricsCalculator2()
    # Test case for mrr
    output_mrr = instance.mrr(([1, 0, 1, 0], 4))
    print(f"MRR: {output_mrr[0]}, Precision: {output_mrr[1]}")

    # Test case for map
    output_map = instance.map(([1, 0, 1, 0], 4))
    print(f"MAP: {output_map}")