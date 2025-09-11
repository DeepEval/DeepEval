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
        :return: float, the cohens kappa value of the matrix
        """
        # Convert to numpy array for easier manipulation
        data = np.array(testData)
        # Calculate the observed agreement
        observed_agreement = np.sum(np.sum(data == np.arange(k)[:, None, None])) / (data.size)
        
        # Calculate the expected agreement
        row_sums = np.sum(data, axis=1)
        col_sums = np.sum(data, axis=0)
        total_sum = np.sum(row_sums)
        expected_agreement = np.sum(col_sums * row_sums) / (total_sum ** 2)
        
        # Calculate kappa
        kappa = (observed_agreement - expected_agreement) / (1 - expected_agreement)
        return kappa

    @staticmethod
    def fleiss_kappa(testData, N, k, n):
        """
        Calculate the fliss kappa value of an N * k matrix
        :param testData: Input data matrix, N * k
        :param N: int, Number of samples
        :param k: int, Number of categories
        :param n: int, Number of raters
        :return: float, fleiss kappa value
        """
        # Convert to numpy array for easier manipulation
        data = np.array(testData)
        # Calculate the expected probability of each category
        p = np.sum(data, axis=0) / (N * n)
        # Calculate the observed probability of agreement
        P = np.sum(data * (data - 1), axis=1) / (n * (n - 1))
        P_sum = np.sum(P) / N
        # Calculate the expected probability of agreement
        P_e = np.sum(p * p)
        # Calculate fleiss kappa
        fleiss_kappa = (P_sum - P_e) / (1 - P_e)
        return fleiss_kappa

# Test cases
if __name__ == "__main__":
    # Test for kappa method
    print(KappaCalculator.kappa([[2, 1, 1], [1, 2, 1], [1, 1, 2]], 3))  # Expected output: 0.25

    # Test for fleiss_kappa method
    print(KappaCalculator.fleiss_kappa([[0, 0, 0, 0, 14],
                                       [0, 2, 6, 4, 2],
                                       [0, 0, 3, 5, 6],
                                       [0, 3, 9, 2, 0],
                                       [2, 2, 8, 1, 1],
                                       [7, 7, 0, 0, 0],
                                       [3, 2, 6, 3, 0],
                                       [2, 5, 3, 2, 2],
                                       [6, 5, 2, 1, 0],
                                       [0, 2, 2, 3, 7]], 10, 5, 14))  # Expected output: 0.20993070442195522