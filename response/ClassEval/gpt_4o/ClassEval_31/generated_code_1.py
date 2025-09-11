import math

class DataStatistics4:
    """
    This is a class that performs advanced mathematical calculations and statistics, including correlation coefficient, skewness, kurtosis, and probability density function (PDF) for a normal distribution.
    """

    @staticmethod
    def correlation_coefficient(data1, data2):
        """
        Calculate the correlation coefficient of two sets of data.
        :param data1: The first set of data, list.
        :param data2: The second set of data, list.
        :return: The correlation coefficient, float.
        """
        if len(data1) != len(data2):
            raise ValueError("Data sets must have the same number of elements")

        n = len(data1)
        mean1 = sum(data1) / n
        mean2 = sum(data2) / n

        covariance = sum((x - mean1) * (y - mean2) for x, y in zip(data1, data2))
        standard_deviation1 = math.sqrt(sum((x - mean1) ** 2 for x in data1))
        standard_deviation2 = math.sqrt(sum((y - mean2) ** 2 for y in data2))

        return covariance / (standard_deviation1 * standard_deviation2)

    @staticmethod
    def skewness(data):
        """
        Calculate the skewness of a set of data.
        :param data: The input data list, list.
        :return: The skewness, float.
        """
        n = len(data)
        mean = sum(data) / n
        m3 = sum((x - mean) ** 3 for x in data) / n
        m2 = sum((x - mean) ** 2 for x in data) / n

        return m3 / (m2 ** 1.5)

    @staticmethod
    def kurtosis(data):
        """
        Calculate the kurtosis of a set of data.
        :param data: The input data list, list.
        :return: The kurtosis, float.
        """
        n = len(data)
        mean = sum(data) / n
        m4 = sum((x - mean) ** 4 for x in data) / n
        m2 = sum((x - mean) ** 2 for x in data) / n

        return m4 / (m2 ** 2) - 3

    @staticmethod
    def pdf(data, mu, sigma):
        """
        Calculate the probability density function (PDF) of a set of data under a normal distribution.
        :param data: The input data list, list.
        :param mu: The mean of the normal distribution, float.
        :param sigma: The standard deviation of the normal distribution, float.
        :return: The probability density function (PDF), list.
        """
        return [(1 / (sigma * math.sqrt(2 * math.pi))) * math.exp(-0.5 * ((x - mu) / sigma) ** 2) for x in data]

if __name__ == "__main__":
    # Test cases
    output = DataStatistics4.correlation_coefficient([1, 2, 3], [4, 5, 6])
    print("Correlation Coefficient:", output)  # Expected: 0.9999999999999998

    output = DataStatistics4.skewness([1, 2, 5])
    print("Skewness:", output)  # Expected: 2.3760224064818463

    output = DataStatistics4.kurtosis([1, 20, 100])
    print("Kurtosis:", output)  # Expected: -1.5000000000000007

    output = DataStatistics4.pdf([1, 2, 3], 1, 1)
    print("PDF:", output)  # Expected: [0.3989422804014327, 0.24197072451914337, 0.05399096651318806]