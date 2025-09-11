import math

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
        data_length = len(data)
        if data_length % 2 == 0:
            return (sorted_data[data_length // 2 - 1] + sorted_data[data_length // 2]) / 2
        else:
            return sorted_data[data_length // 2]

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
        counts = {}
        for item in data:
            if item in counts:
                counts[item] += 1
            else:
                counts[item] = 1
        return [item for item, count in counts.items() if count == max(counts.values())]

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
        if n != len(y):
            raise ValueError("Lists must have the same length")
        
        x_mean = sum(x) / n
        y_mean = sum(y) / n
        
        numerator = sum((x_i - x_mean) * (y_i - y_mean) for x_i, y_i in zip(x, y))
        denominator = math.sqrt(sum((x_i - x_mean)**2 for x_i in x) * sum((y_i - y_mean)**2 for y_i in y))
        
        if denominator == 0:
            return 0  # Avoid division by zero
        
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
        correlation_matrix = [[1.0 for _ in range(n)] for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                correlation_matrix[i][j] = Statistics3.correlation(data[i], data[j])
                correlation_matrix[j][i] = correlation_matrix[i][j]  # Correlation is symmetric
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
        sum_of_squared_differences = sum((x - mean)**2 for x in data)
        return math.sqrt(sum_of_squared_differences / len(data))

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
    print(f"Median: {instance.median([1, 2, 3, 4])}")

    # Test case for mode
    print(f"Mode: {instance.mode([1, 2, 3, 3])}")

    # Test case for correlation
    print(f"Correlation: {instance.correlation([1, 2, 3], [4, 5, 6])}")

    # Test case for mean
    print(f"Mean: {instance.mean([1, 2, 3])}")

    # Test case for correlation matrix
    print(f"Correlation Matrix: {instance.correlation_matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])}")

    # Test case for standard deviation
    print(f"Standard Deviation: {instance.standard_deviation([1, 2, 3])}")

    # Test case for z-score
    print(f"Z-score: {instance.z_score([1, 2, 3, 4])}")