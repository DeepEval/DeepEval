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
        Compute the MRR of the input data. MRR is a widely used evaluation index. It is the mean of reciprocal rank.
        :param data: the data must be a tuple, list 0,1,eg. ([1,0,...],5). 
        In each tuple (actual result,ground truth num), ground truth num is the total ground num.
        ([1,0,...],5), or list of tuple eg. [([1,0,1,...],5),([1,0,...],6),([0,0,...],5)].
        1 stands for a correct answer, 0 stands for a wrong answer.
        :return: if input data is list, return the MRR of this list. if the input data is list of lists,
        return the average MRR on all lists. The second return value is a list of MRR for each input.
        """
        if isinstance(data, tuple):
            data = [data]
        
        mrr_values = []
        
        for results, total_ground in data:
            ranks = [i + 1 for i, r in enumerate(results) if r == 1]
            if ranks:
                reciprocal_rank = 1 / ranks[0]
            else:
                reciprocal_rank = 0
            mrr_values.append(reciprocal_rank)

        average_mrr = np.mean(mrr_values) if mrr_values else 0
        return average_mrr, mrr_values

    @staticmethod
    def map(data):
        """
        Compute the MAP of the input data. MAP is a widely used evaluation index. It is the mean of AP (average precision).
        :param data: the data must be a tuple, list 0,1,eg. ([1,0,...],5). 
        In each tuple (actual result,ground truth num), ground truth num is the total ground num.
        ([1,0,...],5), or list of tuple eg. [([1,0,1,...],5),([1,0,...],6),([0,0,...],5)].
        1 stands for a correct answer, 0 stands for a wrong answer.
        :return: if input data is list, return the MAP of this list. if the input data is list of lists,
        return the average MAP on all lists. The second return value is a list of MAP for each input.
        """
        if isinstance(data, tuple):
            data = [data]
        
        map_values = []
        
        for results, total_ground in data:
            tp = 0
            ap = 0
            for i, r in enumerate(results):
                if r == 1:
                    tp += 1
                    ap += tp / (i + 1)
            ap /= min(total_ground, len(results)) if tp > 0 else 1
            map_values.append(ap)

        average_map = np.mean(map_values) if map_values else 0
        return average_map, map_values


if __name__ == "__main__":
    # Test case for MRR
    mrr_instance = MetricsCalculator2()
    mrr_output = mrr_instance.mrr(([1, 0, 1, 0], 4))
    print("MRR Output:", mrr_output)  # Should return (1.0, [1.0])

    mrr_output_multi = mrr_instance.mrr([([1, 0, 1, 0], 4), ([0, 1, 0, 1], 4)])
    print("MRR Output (Multi):", mrr_output_multi)  # Should return (0.75, [1.0, 0.5])

    # Test case for MAP
    map_instance = MetricsCalculator2()
    map_output = map_instance.map(([1, 0, 1, 0], 4))
    print("MAP Output:", map_output)  # Should return (0.41666666666666663, [0.41666666666666663])

    map_output_multi = map_instance.map([([1, 0, 1, 0], 4), ([0, 1, 0, 1], 4)])
    print("MAP Output (Multi):", map_output_multi)  # Should return (0.3333333333333333, [0.41666666666666663, 0.25])