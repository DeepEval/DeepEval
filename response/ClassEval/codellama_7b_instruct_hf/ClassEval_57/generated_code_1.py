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
        Compute the MRR of the input data. MRR is a widely used evaluation index.
        It is the mean of reciprocal rank.
        :param data: The data must be a tuple, list of 0,1, e.g. ([1,0,...], 5).
                     In each tuple (actual result, ground truth num), ground truth num is the total ground num.
                     Or list of tuples, e.g. [([1,0,1,...],5), ([1,0,...],6), ([0,0,...],5)].
                     1 stands for a correct answer, 0 stands for a wrong answer.
        :return: If input data is a single tuple, return its reciprocal rank.
                 If input data is a list of tuples, return the average MRR and a list of individual reciprocal ranks.
        """
        if isinstance(data, tuple):
            preds, k = data
            for idx, pred in enumerate(preds):
                if pred == 1:
                    return 1.0 / (idx + 1), [1.0 / (idx + 1)]
            return 0.0, [0.0]

        elif isinstance(data, list):
            reciprocal_ranks = []
            for preds, k in data:
                rr = 0.0
                for idx, pred in enumerate(preds):
                    if pred == 1:
                        rr = 1.0 / (idx + 1)
                        break
                reciprocal_ranks.append(rr)
            mrr_score = np.mean(reciprocal_ranks)
            return mrr_score, reciprocal_ranks

        else:
            raise ValueError("Input data must be a tuple or a list of tuples.")

    @staticmethod
    def map(data):
        """
        Compute the MAP of the input data. MAP is a widely used evaluation index.
        It is the mean of AP (average precision).
        :param data: The data must be a tuple, list of 0,1, e.g. ([1,0,...], 5).
                     In each tuple (actual result, ground truth num), ground truth num is the total ground num.
                     Or list of tuples, e.g. [([1,0,1,...],5), ([1,0,...],6), ([0,0,...],5)].
                     1 stands for a correct answer, 0 stands for a wrong answer.
        :return: If input data is a single tuple, return its AP and [AP].
                 If input data is a list of tuples, return the average AP (MAP) and a list of individual APs.
        """
        def compute_ap(preds, k):
            ap = 0.0
            num_correct = 0
            for idx, pred in enumerate(preds):
                if pred == 1:
                    num_correct += 1
                    precision_at_k = num_correct / (idx + 1)
                    ap += precision_at_k
            if k == 0:
                return 0.0
            return ap / k

        if isinstance(data, tuple):
            preds, k = data
            ap = compute_ap(preds, k)
            return ap, [ap]

        elif isinstance(data, list):
            ap_scores = []
            for preds, k in data:
                ap = compute_ap(preds, k)
                ap_scores.append(ap)
            map_score = np.mean(ap_scores)
            return map_score, ap_scores

        else:
            raise ValueError("Input data must be a tuple or a list of tuples.")

if __name__ == "__main__":
    # Test MRR
    print("Testing MRR:")
    # Single tuple
    result1 = MetricsCalculator2.mrr(([1, 0, 1, 0], 4))
    print("MRR single:", result1)  # Expected: (1.0, [1.0])

    # List of tuples
    result2 = MetricsCalculator2.mrr([([1, 0, 1, 0], 4), ([0, 1, 0, 1], 4)])
    print("MRR list:", result2)  # Expected: (0.75, [1.0, 0.5])

    print("\nTesting MAP:")
    # Single tuple
    result3 = MetricsCalculator2.map(([1, 0, 1, 0], 4))
    print("MAP single:", result3)  # Expected: (0.41666666666666663, [0.41666666666666663])

    # List of tuples
    result4 = MetricsCalculator2.map([([1, 0, 1, 0], 4), ([0, 1, 0, 1], 4)])
    print("MAP list:", result4)  # Expected: (0.3333333333333333, [0.41666666666666663, 0.25])