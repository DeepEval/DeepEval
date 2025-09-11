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
        Compute the MRR of the input data. MRR is the mean of reciprocal rank.
        :param data: a list of tuples where each tuple contains (actual result, ground truth number)
        :return: the MRR of the data
        """
        # Calculate reciprocal ranks for each list in data
        reciprocal_ranks = []
        for list_result in data:
            rank = 1
            for actual, ground_truth in list_result:
                if actual == 1:
                    reciprocal_ranks.append(rank)
                rank += 1
        
        # Compute MRR
        if reciprocal_ranks:
            return sum(1.0 / rank for rank in reciprocal_ranks) / len(reciprocal_ranks)
        else:
            return 0.0

    @staticmethod
    def map(data):
        """
        Compute the MAP of the input data. MAP is the mean of average precisions.
        :param data: a list of tuples where each tuple contains (actual result, ground truth number)
        :return: the MAP of the data
        """
        # Calculate average precisions for each list in data
        average_precisions = []
        for list_result in data:
            num_relevant = list_result.count(1)
            num_items = len(list_result)
            average_precision = num_relevant / (num_items + 1e-10)
            average_precisions.append(average_precision)
        
        # Compute MAP
        if average_precisions:
            return sum(average_precisions) / len(average_precisions)
        else:
            return 0.0

if __name__ == "__main__":
    # Test cases
    data1 = [(1, 4), (0, 4), (1, 4), (0, 4)]
    data2 = [(0, 4), (1, 4), (0, 4), (1, 4)]
    data3 = [(1, 5), (0, 5), (1, 5), (0, 5)]
    data4 = [(0, 4), (0, 4), (0, 4), (1, 4)]

    print("MRR:", MetricsCalculator2.mrr([data1, data2, data3, data4]))
    print("MAP:", MetricsCalculator2.map([data1, data2, data3, data4]))