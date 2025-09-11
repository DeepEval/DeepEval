import math
import statistics
import numpy as np

class Statistics3:
    """
    This is a class that implements methods for calculating indicators such as median, mode, correlation matrix, and Z-score in statistics.
    """

    @staticmethod
    def median(data):
        """
        calculates the median of the given list.
        :param data: the given list, list.
        :return: the median of the given list, float.
        """
        return statistics.median(data)

    @staticmethod
    def mode(data):
        """
        calculates the mode of the given list.
        :param data: the given list, list.
        :return: the mode of the given list, list.
        """
        return statistics.multimode(data)

    @staticmethod
    def correlation(x, y):
        """
        calculates the correlation of the given list.
        :param x: the given list, list.
        :param y: the given list, list.
        :return: the correlation of the given list, float.
        """
        if len(x) != len(y):
            raise ValueError("Lists x and y must have the same length")
        return np.corrcoef(x, y)[0, 1]

    @staticmethod
    def mean(data):
        """
        calculates the mean of the given list.
        :param data: the given list, list.
        :return: the mean of the given list, float.
        """
        return statistics.mean(data)

    @staticmethod
    def correlation_matrix(data):
        """
        calculates the correlation matrix of the given list.
        :param data: the given list, list.
        :return: the correlation matrix of the given list, list.
        """
        return np.corrcoef(data)

    @staticmethod
    def standard_deviation(data):
        """
        calculates the standard deviation of the given list.
        :param data: the given list, list.
        :return: the standard deviation of the given list, float.
        """
        return statistics.stdev(data)

    @staticmethod
    def z_score(data):
        """
        calculates the z-score of the given list.
        :param data: the given list, list.
        :return: the z-score of the given list, list.
        """
        mean = Statistics3.mean(data)
        std_dev = Statistics3.standard_deviation(data)
        return [(x - mean) / std_dev for x in data]

if __name__ == "__main__":
    statistics3 = Statistics3()
    
    # Test case for median
    output = statistics3.median([1, 2, 3, 4])
    print(f"Median: {output}")  # Output: 2.5

    # Test case for mode
    output = statistics3.mode([1, 2, 3, 3])
    print(f"Mode: {output}")  # Output: [3]

    # Test case for correlation
    output = statistics3.correlation([1, 2, 3], [4, 5, 6])
    print(f"Correlation: {output}")  # Output: 1.0

    # Test case for mean
    output = statistics3.mean([1, 2, 3])
    print(f"Mean: {output}")  # Output: 2.0

    # Test case for correlation matrix
    output = statistics3.correlation_matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print(f"Correlation Matrix: {output}")  # Output: [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]

    # Test case for standard deviation
    output = statistics3.standard_deviation([1, 2, 3])
    print(f"Standard Deviation: {output}")  # Output: 1.0

    # Test case for z-score
    output = statistics3.z_score([1, 2, 3, 4])
    print(f"Z-score: {output}")  # Output: [-1.161895003862225, -0.3872983346207417, 0.3872983346207417, 1.161895003862225]