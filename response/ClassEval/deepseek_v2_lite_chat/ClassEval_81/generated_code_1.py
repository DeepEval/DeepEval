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
        """
        sorted_data = sorted(data)
        n = len(sorted_data)
        if n % 2 == 0:
            median1 = sorted_data[n//2]
            median2 = sorted_data[n//2 - 1]
            median = (median1 + median2) / 2
        else:
            median = sorted_data[n//2]
        return median

    @staticmethod
    def mode(data):
        """
        calculates the mode of the given list.
        :param data: the given list, list.
        :return: the mode of the given list, list.
        """
        from collections import Counter
        counts = Counter(data)
        max_count = max(counts.values())
        mode_val = [k for k, c in counts.items() if c == max_count]
        return mode_val

    @staticmethod
    def correlation(x, y):
        """
        calculates the correlation of the given list.
        :param x: the given list, list.
        :param y: the given list, list.
        :return: the correlation of the given list, float.
        """
        n = len(x)
        if n == 0:
            return 0
        mean_x = sum(x) / n
        mean_y = sum(y) / n
        covariance = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y)) / n
        std_dev_x = math.sqrt(sum((xi - mean_x) ** 2 for xi in x) / n)
        std_dev_y = math.sqrt(sum((yi - mean_y) ** 2 for yi in y) / n)
        return covariance / (std_dev_x * std_dev_y)

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
        corr_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                corr_matrix[i][j] = corr_matrix[j][i] = Statistics3.correlation(data[i], data[j])
        return corr_matrix

    @staticmethod
    def standard_deviation(data):
        """
        calculates the standard deviation of the given list.
        :param data: the given list, list.
        :return: the standard deviation of the given list, float.
        """
        mean_data = Statistics3.mean(data)
        variance = sum((x - mean_data) ** 2 for x in data) / len(data)
        return math.sqrt(variance)

    @staticmethod
    def z_score(data):
        """
        calculates the z-score of the given list.
        :param data: the given list, list.
        :return: the z-score of the given list, list.
        """
        mean_data = Statistics3.mean(data)
        std_dev_data = Statistics3.standard_deviation(data)
        z_scores = [(x - mean_data) / std_dev_data for x in data]
        return z_scores

# Example usage:
if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)