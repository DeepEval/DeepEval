import numpy as np
from scipy.stats import cohens_kappa
from scipy.stats import fleiss_kappa


class KappaCalculator:
    """
    This is a class as KappaCalculator, supporting to calculate Cohen's and Fleiss' kappa coefficient.
    """

    @staticmethod
    def kappa(testData, k):
        """
        Calculate the cohens kappa value of a k-dimensional matrix
        :param testData: The k-dimensional matrix that needs to calculate the cohens kappa value
        :param k: int, Matrix dimension
        :return:float, the cohens kappa value of the matrix
        >>> KappaCalculator.kappa([[2, 1, 1], [1, 2, 1], [1, 1, 2]], 3)
        0.25
        """
        # Reshape the data to be 1-D array
        ratings = np.array(testData).flatten()
        observed_agreement = np.bincount(np.minimum(ratings, ratings[:, None]))
        expected_agreement = np.sum(ratings) * np.sum(ratings) / len(ratings)
        # Calculate Cohen's kappa
        kappa_value = cohens_kappa(observed_agreement, expected_agreement)
        return kappa_value

    @staticmethod
    def fleiss_kappa(testData, N, k, n):
        """
        Calculate the fliss kappa value of an N * k matrix
        :param testData: Input data matrix, N * k
        :param N: int, Number of samples
        :param k: int, Number of categories
        :param n: int, Number of raters
        :return: float, fleiss kappa value
        >>> KappaCalculator.fleiss_kappa([[0, 0, 0, 0, 14],
        >>>                              [0, 2, 6, 4, 2],
        >>>                              [0, 0, 3, 5, 6],
        >>>                              [0, 3, 9, 2, 0],
        >>>                              [2, 2, 8, 1, 1],
        >>>                              [7, 7, 0, 0, 0],
        >>>                              [3, 2, 6, 3, 0],
        >>>                              [2, 5, 3, 2, 2],
        >>>                              [6, 5, 2, 1, 0],
        >>>                              [0, 2, 2, 3, 7]], 10, 5, 14)
        0.20993070442195522
        """
        # Calculate Fleiss' kappa
        kappa_value = fleiss_kappa(testData, k)
        return kappa_value


if __name__ == "__main__":
    # Test case for kappa method
    kappa_testData = [[2, 1, 1], [1, 2, 1], [1, 1, 2]]
    kappa_result = KappaCalculator.kappa(kappa_testData, 3)
    print("Cohen's Kappa Value:", kappa_result)

    # Test case for fleiss_kappa method
    fleiss_testData = [[0, 0, 0, 0, 14],
                       [0, 2, 6, 4, 2],
                       [0, 0, 3, 5, 6],
                       [0, 3, 9, 2, 0],
                       [2, 2, 8, 1, 1],
                       [7, 7, 0, 0, 0],
                       [3, 2, 6, 3, 0],
                       [2, 5, 3, 2, 2],
                       [6, 5, 2, 1, 0],
                       [0, 2, 2, 3, 7]]
    fleiss_N = 10
    fleiss_k = 5
    fleiss_n = 14
    fleiss_result = KappaCalculator.fleiss_kappa(fleiss_testData, fleiss_N, fleiss_k, fleiss_n)
    print("Fleiss' Kappa Value:", fleiss_result)