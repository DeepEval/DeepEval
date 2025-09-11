import math

class DataStatistics4:
    """
    This is a class that performs advanced mathematical calculations and statistics,
    including correlation coefficient, skewness, kurtosis, and probability density function
    (PDF) for a normal distribution.
    """

    @staticmethod
    def correlation_coefficient(data1, data2):
        """
        Calculate the correlation coefficient of two sets of data.
        :param data1: The first set of data,list.
        :param data2: The second set of data,list.
        :return: The correlation coefficient, float.
        >>> DataStatistics4.correlation_coefficient([1, 2, 3], [4, 5, 6])
        0.9999999999999998

        """
        # Calculate the mean and standard deviation of each set of data
        mean1 = sum(data1) / len(data1)
        mean2 = sum(data2) / len(data2)
        std1 = math.sqrt(sum(x**2 for x in data1) / len(data1) - mean1**2)
        std2 = math.sqrt(sum(x**2 for x in data2) / len(data2) - mean2**2)

        # Calculate the covariance between the two sets of data
        cov = sum((x - mean1) * (y - mean2) for x, y in zip(data1, data2)) / len(data1)

        # Calculate the correlation coefficient
        correlation_coefficient = cov / (std1 * std2)

        return correlation_coefficient

    @staticmethod
    def skewness(data):
        """
        Calculate the skewness of a set of data.
        :param data: The input data list, list.
        :return: The skewness, float.
        >>> DataStatistics4.skewness([1, 2, 5])
        2.3760224064818463

        """
        # Calculate the mean and standard deviation of the data
        mean = sum(data) / len(data)
        std = math.sqrt(sum(x**2 for x in data) / len(data) - mean**2)

        # Calculate the skewness
        skewness = sum(x**3 for x in data) / len(data) - 3 * mean**3
        skewness /= std**3

        return skewness

    @staticmethod
    def kurtosis(data):
        """
        Calculate the kurtosis of a set of data.
        :param data: The input data list, list.
        :return: The kurtosis, float.
        >>> DataStatistics4.kurtosis([1, 20,100])
        -1.5000000000000007

        """
        # Calculate the mean and standard deviation of the data
        mean = sum(data) / len(data)
        std = math.sqrt(sum(x**2 for x in data) / len(data) - mean**2)

        # Calculate the kurtosis
        kurtosis = sum(x**4 for x in data) / len(data) - 4 * mean**4
        kurtosis /= std**4

        return kurtosis

    @staticmethod
    def pdf(data, mu, sigma):
        """
        Calculate the probability density function (PDF) of a set of data under
        a normal distribution.
        :param data: The input data list, list.
        :param mu: The mean of the normal distribution, float.
        :param sigma: The standard deviation of the normal distribution, float.
        :return: The probability density function (PDF), list.
        >>> DataStatistics4.pdf([1, 2, 3], 1, 1)
        [0.3989422804014327, 0.24197072451914337, 0.05399096651318806]

        """
        # Calculate the probability density function (PDF)
        pdf = [math.exp(-((x - mu) / sigma)**2 / 2) / (sigma * math.sqrt(2 * math.pi)) for x in data]

        return pdf

if __name__ == "__main__":
    # Test case 1: correlation coefficient
    data1 = [1, 2, 3]
    data2 = [4, 5, 6]
    output = DataStatistics4.correlation_coefficient(data1, data2)
    print(output)

    # Test case 2: skewness
    data = [1, 2, 5]
    output = DataStatistics4.skewness(data)
    print(output)

    # Test case 3: kurtosis
    data = [1, 20, 100]
    output = DataStatistics4.kurtosis(data)
    print(output)

    # Test case 4: probability density function (PDF)
    data = [1, 2, 3]
    mu = 1
    sigma = 1
    output = DataStatistics4.pdf(data, mu, sigma)
    print(output)