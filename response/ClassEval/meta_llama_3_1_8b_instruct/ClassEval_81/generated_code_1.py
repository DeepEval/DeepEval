import math
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
        >>> statistics3 = Statistics3()
        >>> statistics3.median([1, 2, 3, 4])
        2.5

        """
        sorted_data = sorted(data)
        n = len(sorted_data)
        if n % 2 == 0:
            median = (sorted_data[n // 2 - 1] + sorted_data[n // 2]) / 2
        else:
            median = sorted_data[n // 2]
        return median

    @staticmethod
    def mode(data):
        """
        calculates the mode of the given list.
        :param data: the given list, list.
        :return: the mode of the given list, list.
        >>> statistics3 = Statistics3()
        >>> statistics3.mode([1, 2, 3, 3])
        [3]

        """
        count_dict = {}
        for num in data:
            if num in count_dict:
                count_dict[num] += 1
            else:
                count_dict[num] = 1
        max_count = max(count_dict.values())
        mode = [num for num, count in count_dict.items() if count == max_count]
        return mode

    @staticmethod
    def correlation(x, y):
        """
        calculates the correlation of the given list.
        :param x: the given list, list.
        :param y: the given list, list.
        :return: the correlation of the given list, float.
        >>> statistics3 = Statistics3()
        >>> statistics3.correlation([1, 2, 3], [4, 5, 6])
        1.0

        """
        n = len(x)
        numerator = sum((x[i] - np.mean(x)) * (y[i] - np.mean(y)) for i in range(n))
        denominator = math.sqrt(sum((x[i] - np.mean(x)) ** 2 for i in range(n)) * sum((y[i] - np.mean(y)) ** 2 for i in range(n)))
        if denominator == 0:
            return 0
        else:
            return numerator / denominator

    @staticmethod
    def mean(data):
        """
        calculates the mean of the given list.
        :param data: the given list, list.
        :return: the mean of the given list, float.
        >>> statistics3 = Statistics3()
        >>> statistics3.mean([1, 2, 3])
        2.0

        """
        return sum(data) / len(data)

    @staticmethod
    def correlation_matrix(data):
        """
        calculates the correlation matrix of the given list.
        :param data: the given list, list.
        :return: the correlation matrix of the given list, list.
        >>> statistics3 = Statistics3()
        >>> statistics3.correlation_matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]

        """
        n = len(data)
        correlation_matrix = [[Statistics3.correlation(data[i], data[j]) for j in range(n)] for i in range(n)]
        return correlation_matrix

    @staticmethod
    def standard_deviation(data):
        """
        calculates the standard deviation of the given list.
        :param data: the given list, list.
        :return: the standard deviation of the given list, float.
        >>> statistics3 = Statistics3()
        >>> statistics3.standard_deviation([1, 2, 3])
        1.0

        """
        mean = Statistics3.mean(data)
        variance = sum((x - mean) ** 2 for x in data) / (len(data) - 1)
        return math.sqrt(variance)

    @staticmethod
    def z_score(data):
        """
        calculates the z-score of the given list.
        :param data: the given list, list.
        :return: the z-score of the given list, list.
        >>> statistics3 = Statistics3()
        >>> statistics3.z_score([1, 2, 3, 4])
        [-1.161895003862225, -0.3872983346207417, 0.3872983346207417, 1.161895003862225]

        """
        mean = Statistics3.mean(data)
        std_dev = Statistics3.standard_deviation(data)
        return [(x - mean) / std_dev for x in data]

if __name__ == "__main__":
    instance = Statistics3()

    # Test case for median
    output = instance.median([1, 2, 3, 4])
    print("Median:", output)

    # Test case for mode
    output = instance.mode([1, 2, 3, 3])
    print("Mode:", output)

    # Test case for correlation
    output = instance.correlation([1, 2, 3], [4, 5, 6])
    print("Correlation:", output)

    # Test case for mean
    output = instance.mean([1, 2, 3])
    print("Mean:", output)

    # Test case for correlation matrix
    data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    output = instance.correlation_matrix(data)
    print("Correlation Matrix:", output)

    # Test case for standard deviation
    output = instance.standard_deviation([1, 2, 3])
    print("Standard Deviation:", output)

    # Test case for z-score
    output = instance.z_score([1, 2, 3, 4])
    print("Z-Score:", output)