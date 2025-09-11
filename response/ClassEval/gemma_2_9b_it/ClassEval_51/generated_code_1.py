import numpy as np

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
        # Calculate the observed agreement
        observed_agreement = np.sum(testData * np.eye(k)) / np.sum(testData)
        
        # Calculate the expected agreement
        expected_agreement = np.sum(np.sum(testData, axis=0) * np.sum(testData, axis=1)[:, None]) / np.sum(testData)**2
        
        # Calculate Cohen's kappa
        return (observed_agreement - expected_agreement) / (1 - expected_agreement)

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
        # Calculate the sum of squared differences between each pair of raters
        sum_squared_differences = 0
        for i in range(n):
            for j in range(i + 1, n):
                sum_squared_differences += np.sum((testData[:, i] - testData[:, j])**2)

        # Calculate the Fleiss' kappa
        return 1 - (sum_squared_differences / (N * (k - 1) * (k - 2)))

if __name__ == "__main__":
    # Test case for kappa method
    testData = np.array([[2, 1, 1], [1, 2, 1], [1, 1, 2]])
    k = 3
    kappa_output = KappaCalculator.kappa(testData, k)
    print(f"Cohen's Kappa: {kappa_output}")

    # Test case for fleiss_kappa method
    testData = np.array([[0, 0, 0, 0, 14],
                         [0, 2, 6, 4, 2],
                         [0, 0, 3, 5, 6],
                         [0, 3, 9, 2, 0],
                         [2, 2, 8, 1, 1],
                         [7, 7, 0, 0, 0],
                         [3, 2, 6, 3, 0],
                         [2, 5, 3, 2, 2],
                         [6, 5, 2, 1, 0],
                         [0, 2, 2, 3, 7]])
    N = 10
    k = 5
    n = 14
    fleiss_kappa_output = KappaCalculator.fleiss_kappa(testData, N, k, n)
    print(f"Fleiss' Kappa: {fleiss_kappa_output}")