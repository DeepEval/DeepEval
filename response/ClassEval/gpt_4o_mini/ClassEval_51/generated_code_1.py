import numpy as np

class KappaCalculator:
    """
    This is a class as KappaCalculator, supporting to calculate Cohen's and Fleiss' kappa coefficient.
    """

    @staticmethod
    def kappa(testData, k):
        """
        Calculate the Cohen's kappa value of a k-dimensional matrix
        :param testData: The k-dimensional matrix that needs to calculate the Cohen's kappa value
        :param k: int, Matrix dimension
        :return: float, the Cohen's kappa value of the matrix
        >>> KappaCalculator.kappa([[2, 1, 1], [1, 2, 1], [1, 1, 2]], 3)
        0.25
        """
        # Convert testData to a numpy array
        matrix = np.array(testData)
        
        # Total number of ratings
        total_ratings = np.sum(matrix)
        
        # Calculate observed agreement
        observed_agreement = np.sum(np.diag(matrix)) / total_ratings
        
        # Calculate expected agreement
        row_sums = np.sum(matrix, axis=1)
        col_sums = np.sum(matrix, axis=0)
        expected_agreement = np.sum((row_sums * col_sums) / (total_ratings ** 2))
        
        # Calculate Cohen's kappa
        kappa_value = (observed_agreement - expected_agreement) / (1 - expected_agreement)
        
        return kappa_value

    @staticmethod
    def fleiss_kappa(testData, N, k, n):
        """
        Calculate the Fleiss' kappa value of an N * k matrix
        :param testData: Input data matrix, N * k
        :param N: int, Number of samples
        :param k: int, Number of categories
        :param n: int, Number of raters
        :return: float, Fleiss kappa value
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
        # Convert testData to a numpy array
        matrix = np.array(testData)
        
        # Calculate proportions
        p = np.sum(matrix, axis=0) / (N * n)
        
        # Calculate the overall agreement
        P_bar = np.sum((np.sum(matrix**2, axis=1) - n) / (n * (n - 1))) / N
        
        # Calculate the expected agreement
        P_e = np.sum(p**2)

        # Calculate Fleiss' kappa
        fleiss_kappa_value = (P_bar - P_e) / (1 - P_e)
        
        return fleiss_kappa_value

if __name__ == "__main__":
    # Test case for Cohen's Kappa
    cohen_output = KappaCalculator.kappa([[2, 1, 1], [1, 2, 1], [1, 1, 2]], 3)
    print("Cohen's Kappa Output:", cohen_output)

    # Test case for Fleiss' Kappa
    fleiss_output = KappaCalculator.fleiss_kappa([[0, 0, 0, 0, 14],
                                                  [0, 2, 6, 4, 2],
                                                  [0, 0, 3, 5, 6],
                                                  [0, 3, 9, 2, 0],
                                                  [2, 2, 8, 1, 1],
                                                  [7, 7, 0, 0, 0],
                                                  [3, 2, 6, 3, 0],
                                                  [2, 5, 3, 2, 2],
                                                  [6, 5, 2, 1, 0],
                                                  [0, 2, 2, 3, 7]], 10, 5, 14)
    print("Fleiss' Kappa Output:", fleiss_output)