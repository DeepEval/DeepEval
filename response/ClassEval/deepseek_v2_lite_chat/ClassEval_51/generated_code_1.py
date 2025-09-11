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
        # Calculate the observed and expected frequencies
        n = len(testData)
        observed = np.sum(testData, axis=0)
        total = np.sum(testData, axis=1)[:, np.newaxis] - observed
        expected = total * (n / np.sum(total))

        # Calculate the kappa value
        num = np.sum((observed - expected)**2)
        denom = np.sum((observed - n * 1/k)**2) + np.sum((expected - n * 1/k)**2)
        cohens_kappa = (num / denom) if denom != 0 else 0

        return cohens_kappa

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
        # Flatten the matrix and count occurrences of each category
        flat_data = testData.flatten()
        categories = np.unique(flat_data)
        category_counts = np.bincount(flat_data, minlength=k)

        # Calculate the observed agreement
        observed_agreement = np.sum(category_counts >= 1)

        # Calculate the maximum possible agreement
        max_possible_agreement = k * N

        # Calculate the fleiss kappa value
        fleiss_kappa = (observed_agreement - max_possible_agreement) / (max_possible_agreement - (N - n) / (N - 1))
        return fleiss_kappa

# Test cases
if __name__ == "__main__":
    test_data = np.array([[0, 0, 0, 0, 14],
                          [0, 2, 6, 4, 2],
                          [0, 0, 3, 5, 6],
                          [0, 3, 9, 2, 0],
                          [2, 2, 8, 1, 1],
                          [7, 7, 0, 0, 0],
                          [3, 2, 6, 3, 0],
                          [2, 5, 3, 2, 2],
                          [6, 5, 2, 1, 0],
                          [0, 2, 2, 3, 7]])

    print(KappaCalculator.kappa(test_data, 3))  # Should output 0.25
    print(KappaCalculator.fleiss_kappa(test_data, 10, 5, 14))  # Should output 0.20993070442195522