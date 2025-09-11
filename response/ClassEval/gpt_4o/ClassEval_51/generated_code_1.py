import numpy as np

class KappaCalculator:
    """
    This is a class for KappaCalculator, supporting the calculation of Cohen's and Fleiss' kappa coefficients.
    """

    @staticmethod
    def kappa(testData, k):
        """
        Calculate the Cohen's kappa value of a k-dimensional matrix.
        :param testData: The k-dimensional matrix that needs to calculate the Cohen's kappa value
        :param k: int, Matrix dimension
        :return: float, the Cohen's kappa value of the matrix
        """
        # Convert test data into a numpy array
        matrix = np.array(testData)
        
        # Calculate the observed agreement
        total = np.sum(matrix)
        p0 = np.trace(matrix) / total
        
        # Calculate the expected agreement
        row_marginals = np.sum(matrix, axis=1)
        col_marginals = np.sum(matrix, axis=0)
        pe = np.sum((row_marginals * col_marginals) / total**2)
        
        # Calculate Cohen's kappa
        kappa = (p0 - pe) / (1 - pe)
        
        return kappa

    @staticmethod
    def fleiss_kappa(testData, N, k, n):
        """
        Calculate the Fleiss' kappa value of an N * k matrix.
        :param testData: Input data matrix, N * k
        :param N: int, Number of samples
        :param k: int, Number of categories
        :param n: int, Number of raters
        :return: float, Fleiss' kappa value
        """
        # Convert test data into a numpy array
        matrix = np.array(testData)

        # Calculate the proportion of raters who assigned each category
        p = np.sum(matrix, axis=0) / (N * n)
        
        # Calculate the extent of agreement for each subject
        P = (np.sum(matrix * matrix, axis=1) - n) / (n * (n - 1))
        
        # Calculate the overall mean of P
        Pbar = np.mean(P)
        
        # Calculate the mean of the square of p
        PbarE = np.sum(p * p)
        
        # Calculate Fleiss' kappa
        kappa = (Pbar - PbarE) / (1 - PbarE)
        
        return kappa

# Test case for Cohen's kappa
if __name__ == "__main__":
    cohen_output = KappaCalculator.kappa([[2, 1, 1], [1, 2, 1], [1, 1, 2]], 3)
    print(f"Cohen's Kappa: {cohen_output}")

    # Test case for Fleiss' kappa
    fleiss_output = KappaCalculator.fleiss_kappa([
        [0, 0, 0, 0, 14],
        [0, 2, 6, 4, 2],
        [0, 0, 3, 5, 6],
        [0, 3, 9, 2, 0],
        [2, 2, 8, 1, 1],
        [7, 7, 0, 0, 0],
        [3, 2, 6, 3, 0],
        [2, 5, 3, 2, 2],
        [6, 5, 2, 1, 0],
        [0, 2, 2, 3, 7]
    ], 10, 5, 14)
    print(f"Fleiss' Kappa: {fleiss_output}")