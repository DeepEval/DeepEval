import numpy as np

class DataStatistics2:
    """
    This is a class for performing data statistics, supporting to get the sum, minimum, maximum, variance, standard deviation, and correlation of a given dataset.
    """

    def __init__(self, data):
        """
        Initialize Data List
        :param data: list
        """
        self.data = np.array(data)

    def get_sum(self):
        """
        Calculate the sum of data
        :return: float
        """
        return np.sum(self.data)

    def get_min(self):
        """
        Calculate the minimum value in the data
        :return: float
        """
        return np.min(self.data)

    def get_max(self):
        """
        Calculate the maximum value in the data
        :return: float
        """
        return np.max(self.data)

    def get_variance(self):
        """
        Calculate variance, accurate to two digits after the Decimal separator
        :return: float
        """
        return round(np.var(self.data), 2)

    def get_std_deviation(self):
        """
        Calculate standard deviation, accurate to two digits after the Decimal separator
        :return: float
        """
        return round(np.std(self.data), 2)

    def get_correlation(self):
        """
        Calculate correlation
        :return: float
        """
        return np.corrcoef(self.data)[0, 1]

# Test cases
if __name__ == "__main__":
    ds2 = DataStatistics2([1, 2, 3, 4])
    print("Sum:", ds2.get_sum())  # Expected output: 10
    print("Min:", ds2.get_min())  # Expected output: 1
    print("Max:", ds2.get_max())  # Expected output: 4
    print("Variance:", ds2.get_variance())  # Expected output: 1.25
    print("Standard Deviation:", ds2.get_std_deviation())  # Expected output: 1.12
    print("Correlation:", ds2.get_correlation())  # Expected output: 1.0