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
        n = len(data1)
        sum1 = sum(data1)
        sum2 = sum(data2)
        sum1_squared = sum(map(lambda x: x**2, data1))
        sum2_squared = sum(map(lambda x: x**2, data2))
        cross_product = sum(map(lambda i, j: data1[i] * data2[j], range(n)))

        numerator = n * sum(map(lambda i, j: data1[i] * data2[j], range(n))) - sum1 * sum2
        denominator = math.sqrt((n * sum1_squared - pow(sum1, 2)) * (n * sum2_squared - pow(sum2, 2)))

        return numerator / denominator if denominator != 0 else 0

    @staticmethod
    def skewness(data):
        """
        Calculate the skewness of a set of data.
        :param data: The input data list, list.
        :return: The skewness, float.
        """
        mean = sum(data) / len(data)
        variance = sum((x - mean) ** 2 for x in data) / len(data)
        skewness = (sum((x - mean) ** 3 for x in data) * len(data) / ((len(data) - 1) * (len(data) - 2) * (variance ** 1.5))) if len(data) > 2 else float('inf')
        return skewness

    @staticmethod
    def kurtosis(data):
        """
        Calculate the kurtosis of a set of data.
        :param data: The input data list, list.
        :return: The kurtosis, float.
        """
        mean = sum(data) / len(data)
        variance = sum((x - mean) ** 2 for x in data) / len(data)
        kurtosis = (sum((x - mean) ** 4 for x in data) * len(data) / ((len(data) - 1) * (len(data) - 2) * (len(data) - 3) * (variance ** 2))) if len(data) > 4 else float('inf')
        return kurtosis

    @staticmethod
    def pdf(data, mu, sigma):
        """
        Calculate the probability density function (PDF) of a set of data under a normal distribution.
        :param data: The input data list, list.
        :param mu: The mean of the normal distribution, float.
        :param sigma: The standard deviation of the normal distribution, float.
        :return: The probability density function (PDF), list.
        """
        return [1 / (sigma * math.sqrt(2 * math.pi)) * math.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) for x in data]


if __name__ == "__main__":
    # Test cases
    test_data1 = [1, 2, 3]
    test_data2 = [4, 5, 6]
    test_correlation = DataStatistics4.correlation_coefficient(test_data1, test_data2)
    print("Correlation Coefficient:", test_correlation)

    test_data = [1, 2, 5]
    test_skewness = DataStatistics4.skewness(test_data)
    print("Skewness:", test_skewness)

    test_data = [1, 20, 100]
    test_kurtosis = DataStatistics4.kurtosis(test_data)
    print("Kurtosis:", test_kurtosis)

    test_data = [1, 2, 3]
    test_pdf = DataStatistics4.pdf(test_data, 1, 1)
    print("PDF:", test_pdf)