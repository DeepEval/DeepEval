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
        observed_agreement = 0
        for row in testData:
            for i in range(k):
                for j in range(i + 1, k):
                    if row[i] == row[j]:
                        observed_agreement += 1
        expected_agreement = (sum(row) / (k * (k - 1)) for row in testData)
        expected_agreement = sum(expected_agreement)
        cohens_kappa = (observed_agreement - expected_agreement) / (
            (k * (k - 1)) - expected_agreement
        )
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
        sum_of_each_category = np.sum(testData, axis=0)
        sum_of_each_category_square = np.sum(sum_of_each_category ** 2)
        sum_of_each_row = np.sum(testData, axis=1)
        sum_of_each_row_square = np.sum(sum_of_each_row ** 2)
        sum_of_all_data = np.sum(testData)
        sum_of_all_data_square = np.sum(sum_of_all_data ** 2)
        pe = (
            sum_of_each_row_square
            - (sum_of_all_data_square / N)
            * sum_of_each_row.shape[0]
        ) / (N - 1)
        pe /= (
            sum_of_each_category_square
            - (sum_of_all_data_square / N)
            * sum_of_each_category.shape[0]
        ) / (N - 1)
        pe /= (
            sum_of_all_data_square
            - (sum_of_all_data_square / N) ** 2
        ) / (N - 1)
        kappa = (pe - (n / (n - 1))) / (1 - (n / (n - 1)))
        return kappa


if __name__ == "__main__":
    # Test case for kappa method
    testData = np.array([[2, 1, 1], [1, 2, 1], [1, 1, 2]])
    k = 3
    output = KappaCalculator.kappa(testData, k)
    print(output)

    # Test case for fleiss_kappa method
    testData = np.array(
        [
            [0, 0, 0, 0, 14],
            [0, 2, 6, 4, 2],
            [0, 0, 3, 5, 6],
            [0, 3, 9, 2, 0],
            [2, 2, 8, 1, 1],
            [7, 7, 0, 0, 0],
            [3, 2, 6, 3, 0],
            [2, 5, 3, 2, 2],
            [6, 5, 2, 1, 0],
            [0, 2, 2, 3, 7],
        ]
    )
    N = 10
    k = 5
    n = 14
    output = KappaCalculator.fleiss_kappa(testData, N, k, n)
    print(output)