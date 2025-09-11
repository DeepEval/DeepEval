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
        data.sort()
        if len(data) % 2 == 0:
            return (data[int(len(data) / 2)] + data[int(len(data) / 2 - 1)]) / 2
        else:
            return data[int(len(data) / 2)]

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
        from collections import Counter
        mode = Counter(data).most_common(1)
        return [item for item, count in mode]

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
        if len(x) != len(y):
            raise ValueError("The two lists must be of the same length")
        mean_x = sum(x) / len(x)
        mean_y = sum(y) / len(y)
        covariance = sum((x_i - mean_x) * (y_i - mean_y) for x_i, y_i in zip(x, y))
        variance_x = sum((x_i - mean_x) ** 2 for x_i in x)
        variance_y = sum((y_i - mean_y) ** 2 for y_i in y)
        correlation = covariance / (math.sqrt(variance_x) * math.sqrt(variance_y))
        return correlation

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
        if not data:
            raise ValueError("The list must not be empty")
        correlation_matrix = []
        for i in range(len(data)):
            row = []
            for j in range(len(data)):
                row.append(Statistics3.correlation(data[i], data[j]))
            correlation_matrix.append(row)
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
        if not data:
            raise ValueError("The list must not be empty")
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
        if not data:
            raise ValueError("The list must not be empty")
        mean = Statistics3.mean(data)
        standard_deviation = Statistics3.standard_deviation(data)
        z_score = [(x - mean) / standard_deviation for x in data]
        return z_score


if __name__ == "__main__":
    statistics3 = Statistics3()
    # Test case for median
    output = statistics3.median([1, 2, 3, 4])
    print(output)
    # Test case for mode
    output = statistics3.mode([1, 2, 3, 3])
    print(output)
    # Test case for correlation
    output = statistics3.correlation([1, 2, 3], [4, 5, 6])
    print(output)
    # Test case for mean
    output = statistics3.mean([1, 2, 3])
    print(output)
    # Test case for correlation matrix
    output = statistics3.correlation_matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print(output)
    # Test case for standard deviation
    output = statistics3.standard_deviation([1, 2, 3])
    print(output)
    # Test case for z-score
    output = statistics3.z_score([1, 2, 3, 4])
    print(output)