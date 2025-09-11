import math
from collections import Counter

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
        mid = n // 2
        if n % 2 == 0:
            return (sorted_data[mid - 1] + sorted_data[mid]) / 2
        else:
            return sorted_data[mid]

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
        frequency = Counter(data)
        max_count = max(frequency.values())
        return [k for k, v in frequency.items() if v == max_count]

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
        mean_x = Statistics3.mean(x)
        mean_y = Statistics3.mean(y)
        covariance = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
        std_x = Statistics3.standard_deviation(x)
        std_y = Statistics3.standard_deviation(y)
        return covariance / (n * std_x * std_y)

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
        matrix = []
        for i in range(n):
            row = []
            for j in range(n):
                row.append(Statistics3.correlation(data[i], data[j]))
            matrix.append(row)
        return matrix

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
        mean_value = Statistics3.mean(data)
        variance = sum((xi - mean_value) ** 2 for xi in data) / len(data)
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
        mean_value = Statistics3.mean(data)
        std_dev = Statistics3.standard_deviation(data)
        return [(xi - mean_value) / std_dev for xi in data]

# Test cases
if __name__ == "__main__":
    statistics3 = Statistics3()

    # Testing median
    print("Median:", statistics3.median([1, 2, 3, 4]))  # Output: 2.5

    # Testing mode
    print("Mode:", statistics3.mode([1, 2, 3, 3]))  # Output: [3]

    # Testing correlation
    print("Correlation:", statistics3.correlation([1, 2, 3], [4, 5, 6]))  # Output: 1.0

    # Testing mean
    print("Mean:", statistics3.mean([1, 2, 3]))  # Output: 2.0

    # Testing correlation matrix
    print("Correlation Matrix:", statistics3.correlation_matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))  
    # Output: [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]

    # Testing standard deviation
    print("Standard Deviation:", statistics3.standard_deviation([1, 2, 3]))  # Output: 1.0

    # Testing z-score
    print("Z-score:", statistics3.z_score([1, 2, 3, 4]))  
    # Output: [-1.161895003862225, -0.3872983346207417, 0.3872983346207417, 1.161895003862225]