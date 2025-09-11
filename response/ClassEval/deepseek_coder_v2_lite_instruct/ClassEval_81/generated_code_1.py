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
        """
        sorted_data = sorted(data)
        n = len(sorted_data)
        if n % 2 == 1:
            return float(sorted_data[n // 2])
        else:
            return (sorted_data[n // 2 - 1] + sorted_data[n // 2]) / 2.0

    @staticmethod
    def mode(data):
        """
        calculates the mode of the given list.
        :param data: the given list, list.
        :return: the mode of the given list, list.
        """
        count = Counter(data)
        max_count = max(count.values())
        modes = [num for num, cnt in count.items() if cnt == max_count]
        return modes

    @staticmethod
    def correlation(x, y):
        """
        calculates the correlation of the given list.
        :param x: the given list, list.
        :param y: the given list, list.
        :return: the correlation of the given list, float.
        """
        n = len(x)
        if n != len(y):
            raise ValueError("Data lists must be of the same length")
        mean_x = sum(x) / n
        mean_y = sum(y) / n
        covariance = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n)) / n
        std_dev_x = math.sqrt(sum((x[i] - mean_x) ** 2 for i in range(n)) / n)
        std_dev_y = math.sqrt(sum((y[i] - mean_y) ** 2 for i in range(n)) / n)
        correlation = covariance / (std_dev_x * std_dev_y)
        return correlation

    @staticmethod
    def mean(data):
        """
        calculates the mean of the given list.
        :param data: the given list, list.
        :return: the mean of the given list, float.
        """
        return sum(data) / len(data)

    @staticmethod
    def correlation_matrix(data):
        """
        calculates the correlation matrix of the given list.
        :param data: the given list, list.
        :return: the correlation matrix of the given list, list.
        """
        n = len(data)
        matrix = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                matrix[i][j] = Statistics3.correlation(data[i], data[j])
        return matrix

    @staticmethod
    def standard_deviation(data):
        """
        calculates the standard deviation of the given list.
        :param data: the given list, list.
        :return: the standard deviation of the given list, float.
        """
        mean_val = Statistics3.mean(data)
        variance = sum((x - mean_val) ** 2 for x in data) / len(data)
        return math.sqrt(variance)

    @staticmethod
    def z_score(data):
        """
        calculates the z-score of the given list.
        :param data: the given list, list.
        :return: the z-score of the given list, list.
        """
        mean_val = Statistics3.mean(data)
        std_dev = Statistics3.standard_deviation(data)
        return [(x - mean_val) / std_dev for x in data]

# Test cases
if __name__ == "__main__":
    statistics3 = Statistics3()
    print("Median of [1, 2, 3, 4]:", statistics3.median([1, 2, 3, 4]))  # Expected output: 2.5
    print("Mode of [1, 2, 3, 3]:", statistics3.mode([1, 2, 3, 3]))  # Expected output: [3]
    print("Correlation of [1, 2, 3] and [4, 5, 6]:", statistics3.correlation([1, 2, 3], [4, 5, 6]))  # Expected output: 1.0
    print("Mean of [1, 2, 3]:", statistics3.mean([1, 2, 3]))  # Expected output: 2.0
    print("Correlation matrix of [[1, 2, 3], [4, 5, 6], [7, 8, 9]]:", statistics3.correlation_matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))  # Expected output: [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
    print("Standard deviation of [1, 2, 3]:", statistics3.standard_deviation([1, 2, 3]))  # Expected output: 1.0
    print("Z-score of [1, 2, 3, 4]:", statistics3.z_score([1, 2, 3, 4]))  # Expected output: [-1.161895003862225, -0.3872983346207417, 0.3872983346207417, 1.161895003862225]